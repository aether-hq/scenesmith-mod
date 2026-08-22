"""Provider-neutral policy and capability boundary for SceneSmith LLM calls.

The rest of SceneSmith should reason in terms of required capabilities instead
of vendor names.  Native API providers are routed through LiteLLM, OpenAI-style
endpoints use the OpenAI client directly, and authenticated local CLI providers
use SceneSmith's loopback bridge.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
import uuid

from dataclasses import dataclass, replace
from typing import Any, Protocol

from openai import OpenAI

from scenesmith.agent_utils.llm.contracts.cli_request_policy import (
    _cancel_subscription_turn,
    _cli_proxy_http_timeout_seconds,
    _with_cli_request_timeout,
)
from scenesmith.agent_utils.llm.contracts.errors import (
    LLMCapabilityError,
    LLMCircuitOpenError,
    LLMHarnessError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)
from scenesmith.agent_utils.llm.contracts.response_normalization import (
    _agent_request_contract,
    _has_images,
    _request_json_in_prompt,
    _to_responses_messages,
    _with_vision_detail,
    extract_json_object,
    normalize_agent_model_response,
)
from scenesmith.agent_utils.llm.contracts.retry_policy import (
    _is_timeout_error,
    is_transient_provider_error,
)

CLI_PROVIDERS = frozenset({"codex-cli", "claude-cli"})
OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openai", "openai-compatible"})
PROVIDER_ALIASES = {
    "claude": "claude-cli",
    "codex": "codex-cli",
    "chatgpt": "codex-cli",
    "compatible": "openai-compatible",
}

console_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCapabilities:
    vision: bool
    tools: bool
    structured_output: bool
    native_reasoning_controls: bool = False

    def missing(
        self,
        *,
        vision: bool = False,
        tools: bool = False,
        structured_output: bool = False,
    ) -> list[str]:
        required = {
            "vision": vision,
            "tools": tools,
            "structured_output": structured_output,
        }
        return [
            name
            for name, needed in required.items()
            if needed and not getattr(self, name)
        ]


@dataclass(frozen=True)
class LLMHarnessConfig:
    provider: str
    model: str
    timeout_seconds: float
    max_attempts: int
    fallback_models: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, default_model: str = "gpt-5") -> "LLMHarnessConfig":
        requested = os.environ.get("SCENESMITH_LLM_PROVIDER", "openai").strip().lower()
        provider = PROVIDER_ALIASES.get(requested, requested)
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", provider):
            raise ValueError(f"Invalid SCENESMITH_LLM_PROVIDER: {requested!r}")
        timeout = float(os.environ.get("SCENESMITH_LLM_TIMEOUT_SECONDS", "30"))
        attempts = int(os.environ.get("SCENESMITH_LLM_MAX_ATTEMPTS", "2"))
        if timeout <= 0:
            raise ValueError("SCENESMITH_LLM_TIMEOUT_SECONDS must be positive")
        if attempts not in {1, 2, 3}:
            raise ValueError("SCENESMITH_LLM_MAX_ATTEMPTS must be from 1 to 3")
        model = os.environ.get("SCENESMITH_LLM_MODEL", default_model).strip()
        if not model:
            raise ValueError("SCENESMITH_LLM_MODEL must not be empty")
        raw_fallbacks = os.environ.get("SCENESMITH_LLM_FALLBACK_MODELS", "")
        fallback_models: list[str] = []
        for fallback in raw_fallbacks.split(","):
            fallback = fallback.strip()
            if not fallback or fallback == model or fallback in fallback_models:
                continue
            if not re.fullmatch(r"[^\s,]+", fallback):
                raise ValueError(
                    "SCENESMITH_LLM_FALLBACK_MODELS must be a comma-separated "
                    "list of model identifiers"
                )
            fallback_models.append(fallback)
        return cls(
            provider=provider,
            model=model,
            timeout_seconds=timeout,
            max_attempts=attempts,
            fallback_models=tuple(fallback_models),
        )

    @property
    def uses_cli_bridge(self) -> bool:
        return self.provider in CLI_PROVIDERS

    @property
    def uses_openai_client(self) -> bool:
        return self.provider in OPENAI_COMPATIBLE_PROVIDERS or self.uses_cli_bridge

    @property
    def uses_litellm(self) -> bool:
        return not self.uses_openai_client

    @property
    def routed_model(self) -> str:
        return self.routed_model_for(self.model)

    def routed_model_for(self, model: str) -> str:
        if not self.uses_litellm or "/" in model:
            return model
        return f"{self.provider}/{model}"

    @property
    def model_chain(self) -> tuple[str, ...]:
        return (self.model, *self.fallback_models)

    def agents_model_chain(self, requested_model: str | None) -> tuple[str, ...]:
        primary = requested_model or self.routed_model
        fallbacks = tuple(
            self.routed_model_for(model) for model in self.fallback_models
        )
        return tuple(dict.fromkeys((primary, *fallbacks)))


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: list[dict[str, Any]]
    reasoning_effort: str = "low"
    verbosity: str = "low"
    response_format: dict[str, str] | None = None
    vision_detail: str = "auto"


class LLMAdapter(Protocol):
    """Minimal transport contract. Vendor SDK objects never escape this boundary."""

    def complete(self, request: LLMRequest) -> str: ...


class _LLMCircuitBreaker:
    """Stop a slow provider from consuming one deadline per scene object.

    Exhausting every configured model route opens the breaker for a short
    cooldown. A timeout on one attempt does not poison its retry or the fallback
    models. The breaker is process-local, so a resumed build starts cleanly.
    """

    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._open_until = 0.0
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            remaining = self._open_until - time.monotonic()
        if remaining > 0:
            raise LLMCircuitOpenError(
                f"LLM circuit is open for another {remaining:.1f}s after a "
                "fully exhausted model turn; resume the stage later"
            )

    def record_timeout(self) -> None:
        with self._lock:
            self._open_until = max(
                self._open_until, time.monotonic() + self.cooldown_seconds
            )


_CIRCUIT = _LLMCircuitBreaker(
    cooldown_seconds=float(
        os.environ.get("SCENESMITH_LLM_CIRCUIT_COOLDOWN_SECONDS", "0")
    )
)


def _env_capability(name: str) -> bool | None:
    value = os.environ.get(f"SCENESMITH_LLM_CAP_{name.upper()}")
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"SCENESMITH_LLM_CAP_{name.upper()} must be true or false")


def detect_capabilities(config: LLMHarnessConfig) -> LLMCapabilities:
    """Resolve capabilities conservatively, with explicit env overrides.

    CLI bridges implement and validate all three normalized surfaces.  OpenAI is
    known natively.  A generic OpenAI-compatible endpoint must either advertise
    overrides or satisfy the conformance probe.  LiteLLM's model metadata is used
    for native third-party providers.
    """
    if config.uses_cli_bridge:
        detected = LLMCapabilities(True, True, True, False)
    elif config.provider == "openai":
        detected = LLMCapabilities(True, True, True, True)
    elif config.provider == "openai-compatible":
        detected = LLMCapabilities(False, False, False, False)
    else:
        try:
            import litellm

            model = config.routed_model
            detected = LLMCapabilities(
                vision=bool(litellm.supports_vision(model=model)),
                tools=bool(litellm.supports_function_calling(model=model)),
                structured_output=bool(litellm.supports_response_schema(model=model)),
                native_reasoning_controls=False,
            )
        except Exception:
            # Unknown is intentionally not treated as supported.  Users can opt
            # in via explicit capability flags after running the conformance test.
            detected = LLMCapabilities(False, False, False, False)

    values = {}
    for name in ("vision", "tools", "structured_output"):
        override = _env_capability(name)
        values[name] = getattr(detected, name) if override is None else override
    return LLMCapabilities(
        **values,
        native_reasoning_controls=detected.native_reasoning_controls,
    )


def require_capabilities(
    config: LLMHarnessConfig,
    *,
    vision: bool = True,
    tools: bool = True,
    structured_output: bool = True,
) -> LLMCapabilities:
    capabilities = detect_capabilities(config)
    missing = capabilities.missing(
        vision=vision,
        tools=tools,
        structured_output=structured_output,
    )
    if missing:
        raise LLMCapabilityError(
            f"{config.provider}/{config.model} has not demonstrated required "
            f"capabilities: {', '.join(missing)}. Run the SceneSmith LLM "
            "conformance check or set the matching SCENESMITH_LLM_CAP_* flags "
            "only when the endpoint supports them."
        )
    return capabilities


class HardenedModel:
    """Agents SDK model wrapper enforcing the same policy as direct calls."""

    def __init__(
        self,
        provider: Any,
        requested_model: str | None,
        config: LLMHarnessConfig,
    ) -> None:
        self._provider = provider
        self._requested_model = requested_model
        self._config = config

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        attempts = 0
        try:
            _CIRCUIT.before_call()
            failures: list[tuple[str, Exception]] = []
            model_chain = self._config.agents_model_chain(self._requested_model)
            for model_name in model_chain:
                model = self._provider.get_model(model_name)
                for attempt in range(1, self._config.max_attempts + 1):
                    attempts += 1
                    try:
                        if self._config.uses_cli_bridge:
                            call_args, call_kwargs = _with_cli_request_timeout(
                                args, kwargs
                            )
                            response = await model.get_response(
                                *call_args, **call_kwargs
                            )
                        else:
                            async with asyncio.timeout(self._config.timeout_seconds):
                                response = await model.get_response(*args, **kwargs)
                        tools, output_schema = _agent_request_contract(args, kwargs)
                        return normalize_agent_model_response(
                            response, tools=tools, output_schema=output_schema
                        )
                    except Exception as exc:
                        failures.append((model_name, exc))
                        retryable = (
                            isinstance(exc, LLMStructuredOutputError)
                            or _is_timeout_error(exc)
                            or is_transient_provider_error(exc)
                        )
                        console_logger.warning(
                            "LLM route failed request=%s model=%s attempt=%d/%d: %s",
                            request_id,
                            model_name,
                            attempt,
                            self._config.max_attempts,
                            exc,
                        )
                        if self._config.uses_cli_bridge and _is_timeout_error(exc):
                            await asyncio.to_thread(_cancel_subscription_turn)
                        if retryable and attempt < self._config.max_attempts:
                            await asyncio.sleep(0.25)
                            continue
                        break
            _raise_exhausted_routes(self._config, failures)
            raise AssertionError("unreachable LLM route loop")
        finally:
            console_logger.info(
                "LLM agent request=%s provider=%s model=%s attempts=%d elapsed=%.3fs",
                request_id,
                self._config.provider,
                ",".join(self._config.model_chain),
                attempts,
                time.monotonic() - started,
            )

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        return self._stream_response(*args, **kwargs)

    async def _stream_response(self, *args: Any, **kwargs: Any) -> Any:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        _CIRCUIT.before_call()
        failures: list[tuple[str, Exception]] = []
        yielded = False
        try:
            for model_name in self._config.agents_model_chain(self._requested_model):
                model = self._provider.get_model(model_name)
                try:
                    if self._config.uses_cli_bridge:
                        call_args, call_kwargs = _with_cli_request_timeout(args, kwargs)
                        async for event in model.stream_response(
                            *call_args, **call_kwargs
                        ):
                            yielded = True
                            yield event
                    else:
                        async with asyncio.timeout(self._config.timeout_seconds):
                            async for event in model.stream_response(*args, **kwargs):
                                yielded = True
                                yield event
                    return
                except Exception as exc:
                    if yielded:
                        raise
                    if self._config.uses_cli_bridge and _is_timeout_error(exc):
                        await asyncio.to_thread(_cancel_subscription_turn)
                    failures.append((model_name, exc))
                    console_logger.warning(
                        "LLM stream route failed request=%s model=%s: %s",
                        request_id,
                        model_name,
                        exc,
                    )
            _raise_exhausted_routes(self._config, failures)
        finally:
            console_logger.info(
                "LLM agent stream request=%s provider=%s model=%s elapsed=%.3fs",
                request_id,
                self._config.provider,
                ",".join(self._config.model_chain),
                time.monotonic() - started,
            )


class HardenedModelProvider:
    """Single Agents SDK provider boundary for every configured transport."""

    def __init__(self, provider: Any, config: LLMHarnessConfig) -> None:
        self._provider = provider
        self._config = config

    def get_model(self, model_name: str | None) -> HardenedModel:
        return HardenedModel(self._provider, model_name, self._config)


_agents_provider: HardenedModelProvider | None = None
_agents_provider_key: tuple[str, str, tuple[str, ...], float, int] | None = None


def _provider_cache_key(
    config: LLMHarnessConfig,
) -> tuple[str, str, tuple[str, ...], float, int]:
    return (
        config.provider,
        config.model,
        config.fallback_models,
        config.timeout_seconds,
        config.max_attempts,
    )


def agents_model_provider(config: LLMHarnessConfig) -> HardenedModelProvider:
    """Return the process-wide hardened provider for all agentic model turns."""
    global _agents_provider, _agents_provider_key
    key = _provider_cache_key(config)
    if _agents_provider is None or _agents_provider_key != key:
        # Normal startup installs this once from main.py. This fallback keeps
        # isolated tests and library consumers on the exact same path.
        install_agents_runtime(config)
    assert _agents_provider is not None
    return _agents_provider


def install_agents_runtime(
    config: LLMHarnessConfig | None = None,
) -> tuple[str, LLMCapabilities]:
    """Install the selected adapter into the OpenAI Agents SDK.

    This is the only place where SceneSmith configures vendor transports for
    agentic tool loops. Feature code receives only the normalized model name.
    """
    harness = config or LLMHarnessConfig.from_env()
    capabilities = require_capabilities(harness)

    global _agents_provider, _agents_provider_key

    from agents import (
        set_default_openai_api,
        set_default_openai_client,
        set_tracing_disabled,
    )
    from agents.models.openai_provider import OpenAIProvider
    from openai import AsyncOpenAI

    if harness.provider == "openai":
        client = AsyncOpenAI(
            max_retries=0,
            timeout=harness.timeout_seconds,
        )
        set_default_openai_client(client, use_for_tracing=False)
        raw_provider = OpenAIProvider(openai_client=client, use_responses=True)
        _agents_provider = HardenedModelProvider(raw_provider, harness)
        _agents_provider_key = _provider_cache_key(harness)
        return harness.model, capabilities

    if harness.provider == "openai-compatible":
        base_url = os.environ.get("SCENESMITH_LLM_BASE_URL")
        api_key = os.environ.get("SCENESMITH_LLM_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not base_url or not api_key:
            raise LLMHarnessError(
                "openai-compatible requires SCENESMITH_LLM_BASE_URL and "
                "SCENESMITH_LLM_API_KEY"
            )
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=harness.timeout_seconds,
        )
        set_default_openai_client(client, use_for_tracing=False)
        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)
        raw_provider = OpenAIProvider(openai_client=client, use_responses=False)
        _agents_provider = HardenedModelProvider(raw_provider, harness)
        _agents_provider_key = _provider_cache_key(harness)
        return harness.model, capabilities

    if harness.provider == "codex-cli":
        from scenesmith.agent_utils.llm.cli_llm_proxy import start_codex_cli_proxy

        base_url = start_codex_cli_proxy()
        api_key = "chatgpt-subscription"
    elif harness.provider == "claude-cli":
        from scenesmith.agent_utils.llm.claude_cli_proxy import start_claude_cli_proxy

        base_url = start_claude_cli_proxy()
        api_key = "claude-subscription"
    else:
        try:
            from agents.extensions.models.litellm_provider import LitellmProvider
        except ImportError as exc:
            raise LLMHarnessError(
                "This provider requires LiteLLM. Install the SceneSmith locked "
                "dependencies before starting."
            ) from exc
        _agents_provider = HardenedModelProvider(LitellmProvider(), harness)
        _agents_provider_key = _provider_cache_key(harness)
        set_tracing_disabled(True)
        return harness.routed_model, capabilities

    os.environ["SCENESMITH_CLI_PROXY_URL"] = base_url
    proxy_timeout = _cli_proxy_http_timeout_seconds()
    console_logger.info(
        "Subscription LLM policy provider=%s response_start=%ss inactivity=%ss "
        "absolute=%ss loopback=%ss circuit=%ss",
        harness.provider,
        os.environ.get("SCENESMITH_LLM_RESPONSE_START_TIMEOUT_SECONDS", "15"),
        os.environ.get("SCENESMITH_LLM_CLI_INACTIVITY_TIMEOUT_SECONDS", "45"),
        os.environ.get("SCENESMITH_LLM_HARD_TIMEOUT_SECONDS", "300"),
        f"{proxy_timeout:g}",
        os.environ.get("SCENESMITH_LLM_CIRCUIT_COOLDOWN_SECONDS", "0"),
    )
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=proxy_timeout,
    )
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    raw_provider = OpenAIProvider(openai_client=client, use_responses=False)
    _agents_provider = HardenedModelProvider(raw_provider, harness)
    _agents_provider_key = _provider_cache_key(harness)
    return harness.model, capabilities


class OpenAIProtocolAdapter:
    """OpenAI/compatible/CLI transport normalized to the harness contract."""

    _reasoning_models = ("gpt-5", "gpt-5.2", "o3", "o4")

    def __init__(self, config: LLMHarnessConfig, service_tier: str | None = None):
        self.config = config
        self.service_tier = service_tier
        if config.uses_cli_bridge:
            base_url = os.environ.get("SCENESMITH_CLI_PROXY_URL")
            if not base_url:
                raise LLMHarnessError("Subscription CLI proxy has not been started")
            self.client = OpenAI(
                api_key="local-cli-subscription",
                base_url=base_url,
                max_retries=0,
                timeout=_cli_proxy_http_timeout_seconds(),
            )
        elif config.provider == "openai-compatible":
            base_url = os.environ.get("SCENESMITH_LLM_BASE_URL")
            api_key = os.environ.get("SCENESMITH_LLM_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
            if not base_url or not api_key:
                raise LLMHarnessError(
                    "openai-compatible requires SCENESMITH_LLM_BASE_URL and "
                    "SCENESMITH_LLM_API_KEY"
                )
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=config.timeout_seconds,
            )
        else:
            self.client = OpenAI(max_retries=0, timeout=config.timeout_seconds)

    def complete(self, request: LLMRequest) -> str:
        model = self.config.model
        use_responses = self.config.provider == "openai" and any(
            model == name or model.startswith(f"{name}-")
            for name in self._reasoning_models
        )
        if use_responses:
            messages = _to_responses_messages(request.messages, request.vision_detail)
            if request.response_format:
                messages = _request_json_in_prompt(messages, responses_format=True)
            kwargs: dict[str, Any] = {
                "model": model,
                "input": messages,
                "reasoning": {"effort": request.reasoning_effort},
                "text": {"verbosity": request.verbosity},
            }
            if self.service_tier:
                kwargs["service_tier"] = self.service_tier
            response = self.client.responses.create(**kwargs)
            content = response.output_text
        else:
            messages = _with_vision_detail(request.messages, request.vision_detail)
            kwargs = {"model": model, "messages": messages}
            if request.response_format:
                kwargs["response_format"] = request.response_format
            if self.service_tier and self.config.provider == "openai":
                kwargs["service_tier"] = self.service_tier
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
        if not content or not str(content).strip():
            raise LLMHarnessError(
                f"{self.config.provider}/{model} returned empty output"
            )
        return str(content)


class LiteLLMAdapter:
    """Native provider transport via LiteLLM's normalized completion contract."""

    def __init__(self, config: LLMHarnessConfig):
        self.config = config

    def complete(self, request: LLMRequest) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise LLMHarnessError(
                "Native third-party LLM providers require the locked LiteLLM dependency"
            ) from exc
        kwargs: dict[str, Any] = {
            "model": self.config.routed_model,
            "messages": _with_vision_detail(request.messages, request.vision_detail),
            "timeout": self.config.timeout_seconds,
            "num_retries": 0,
            "max_tokens": int(os.environ.get("SCENESMITH_LLM_MAX_TOKENS", "8192")),
        }
        if request.response_format:
            kwargs["response_format"] = request.response_format
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
        if not content or not str(content).strip():
            raise LLMHarnessError(
                f"{self.config.provider}/{self.config.model} returned empty output"
            )
        return str(content)


class LLMHarness:
    """The sole application-facing path for direct text and multimodal turns."""

    def __init__(
        self,
        config: LLMHarnessConfig | None = None,
        *,
        service_tier: str | None = None,
        adapter: LLMAdapter | None = None,
    ) -> None:
        self.config = config or LLMHarnessConfig.from_env()
        self.capabilities = detect_capabilities(self.config)
        self._service_tier = service_tier
        self._injected_adapter = adapter
        self.adapter = adapter or self._adapter_for(self.config)

    def _adapter_for(self, config: LLMHarnessConfig) -> LLMAdapter:
        return (
            OpenAIProtocolAdapter(config, service_tier=self._service_tier)
            if config.uses_openai_client
            else LiteLLMAdapter(config)
        )

    def complete(self, request: LLMRequest) -> str:
        missing = self.capabilities.missing(
            vision=_has_images(request.messages),
            structured_output=request.response_format is not None,
        )
        if missing:
            raise LLMCapabilityError(
                f"{self.config.provider}/{self.config.model} cannot satisfy this "
                f"request: missing {', '.join(missing)}"
            )
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        attempts = 0

        try:
            _CIRCUIT.before_call()
            failures: list[tuple[str, Exception]] = []
            content: str | None = None
            for model in self.config.model_chain:
                route_config = replace(
                    self.config,
                    model=model,
                    fallback_models=(),
                )
                adapter = self._injected_adapter or self._adapter_for(route_config)
                for attempt in range(1, self.config.max_attempts + 1):
                    attempts += 1
                    try:
                        content = adapter.complete(request)
                        if request.response_format:
                            content = extract_json_object(content)
                        break
                    except Exception as exc:
                        content = None
                        failures.append((model, exc))
                        retryable = (
                            isinstance(exc, LLMStructuredOutputError)
                            or _is_timeout_error(exc)
                            or is_transient_provider_error(exc)
                        )
                        console_logger.warning(
                            "LLM route failed request=%s model=%s attempt=%d/%d: %s",
                            request_id,
                            model,
                            attempt,
                            self.config.max_attempts,
                            exc,
                        )
                        if retryable and attempt < self.config.max_attempts:
                            time.sleep(0.25)
                            continue
                        break
                if content is not None:
                    break
            if content is None:
                _raise_exhausted_routes(self.config, failures)
            return content
        finally:
            console_logger.info(
                "LLM harness request=%s provider=%s model=%s attempts=%d elapsed=%.3fs",
                request_id,
                self.config.provider,
                ",".join(self.config.model_chain),
                attempts,
                time.monotonic() - started,
            )


def _raise_exhausted_routes(
    config: LLMHarnessConfig,
    failures: list[tuple[str, Exception]],
) -> None:
    if not failures:
        raise LLMHarnessError("llm_routes_exhausted: no model route was attempted")
    route_names = ", ".join(dict.fromkeys(model for model, _ in failures))
    timeout_failures = sum(_is_timeout_error(exc) for _, exc in failures)
    if timeout_failures:
        _CIRCUIT.record_timeout()
        policy = (
            "subscription response policy"
            if config.uses_cli_bridge
            else f"{config.timeout_seconds:g}s direct-API deadline"
        )
        other_failures = len(failures) - timeout_failures
        raise LLMTimeoutError(
            "llm_turn_exhausted: all configured routes failed across "
            f"{route_names} ({timeout_failures} timeout failures under the "
            f"{policy}, {other_failures} other failures)"
        ) from failures[-1][1]
    raise LLMHarnessError(
        f"llm_routes_exhausted: all configured models failed ({route_names}): "
        f"{failures[-1][1]}"
    ) from failures[-1][1]
