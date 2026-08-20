"""Provider-neutral policy and capability boundary for SceneSmith LLM calls.

The rest of SceneSmith should reason in terms of required capabilities instead
of vendor names.  Native API providers are routed through LiteLLM, OpenAI-style
endpoints use the OpenAI client directly, and authenticated local CLI providers
use SceneSmith's loopback bridge.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
import threading
import time
import uuid

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, TypeVar
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from openai import OpenAI


CLI_PROVIDERS = frozenset({"codex-cli", "claude-cli"})
OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openai", "openai-compatible"})
PROVIDER_ALIASES = {
    "claude": "claude-cli",
    "codex": "codex-cli",
    "chatgpt": "codex-cli",
    "compatible": "openai-compatible",
}

console_logger = logging.getLogger(__name__)


def _cli_proxy_http_timeout_seconds() -> float:
    timeout = float(os.environ.get("SCENESMITH_LLM_PROXY_HTTP_TIMEOUT_SECONDS", "150"))
    if timeout <= 0:
        raise ValueError("SCENESMITH_LLM_PROXY_HTTP_TIMEOUT_SECONDS must be positive")
    return timeout


def _cancel_subscription_turn() -> bool:
    """Cancel an orphaned CLI worker and wait for its serialized lock to release."""
    base_url = os.environ.get("SCENESMITH_CLI_PROXY_URL")
    if not base_url:
        return False
    request = Request(urljoin(base_url, "cancel"), data=b"", method="POST")
    try:
        with urlopen(request, timeout=5.0) as response:
            return response.status in {200, 202}
    except Exception as exc:
        console_logger.warning("Could not cancel subscription CLI turn: %s", exc)
        return False


def _with_cli_request_timeout(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Force the loopback timeout on each SDK request, not only its client."""
    timeout = _cli_proxy_http_timeout_seconds()
    updated_args = list(args)
    updated_kwargs = dict(kwargs)
    if len(updated_args) >= 3:
        settings = updated_args[2]
        extra_args = dict(getattr(settings, "extra_args", None) or {})
        extra_args["timeout"] = timeout
        updated_args[2] = replace(settings, extra_args=extra_args)
    elif "model_settings" in updated_kwargs:
        settings = updated_kwargs["model_settings"]
        extra_args = dict(getattr(settings, "extra_args", None) or {})
        extra_args["timeout"] = timeout
        updated_kwargs["model_settings"] = replace(settings, extra_args=extra_args)
    else:
        raise LLMHarnessError("Agents SDK request omitted model_settings")
    return tuple(updated_args), updated_kwargs


class LLMHarnessError(RuntimeError):
    """Base error surfaced at the provider boundary."""


class LLMCapabilityError(LLMHarnessError):
    """Raised before a run when the selected model cannot satisfy the contract."""


class LLMStructuredOutputError(LLMHarnessError):
    """Raised when deterministic structured-output recovery fails."""


class LLMTimeoutError(LLMHarnessError):
    """Raised when one model turn exceeds the common wall-clock deadline."""


class LLMCircuitOpenError(LLMHarnessError):
    """Raised immediately after a timeout opens the process-local circuit."""


def _json_fence_body(raw_output: str) -> str:
    """Remove a Markdown fence while retaining any surrounding explanation."""

    candidate = str(raw_output or "").strip()
    fenced = re.search(r"```(?:json|javascript|js|python)?\s*([\s\S]*?)```", candidate, re.I)
    return fenced.group(1).strip() if fenced else candidate


def _json_container_candidate(raw_output: str) -> str:
    """Extract the first object/array and close truncated delimiters safely."""

    candidate = _json_fence_body(raw_output)
    starts = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
    if not starts:
        raise LLMStructuredOutputError(
            f"Model did not return JSON: {candidate[:300]!r}"
        )
    start = min(starts)
    stack: list[str] = []
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                continue
            stack.pop()
            if not stack:
                return candidate[start : index + 1]
    return candidate[start:] + "".join(reversed(stack))


def _strip_json_comments(candidate: str) -> str:
    """Remove JavaScript comments without touching comment markers in strings."""

    output: list[str] = []
    index = 0
    in_string = False
    quote = ""
    escaped = False
    while index < len(candidate):
        char = candidate[index]
        following = candidate[index + 1] if index + 1 < len(candidate) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            index += 1
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            index += 2
            while index < len(candidate) and candidate[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and following == "*":
            end = candidate.find("*/", index + 2)
            index = len(candidate) if end < 0 else end + 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _pythonize_json_literals(candidate: str) -> str:
    """Translate JSON literals outside strings for ``ast.literal_eval``."""

    output: list[str] = []
    token: list[str] = []
    in_string = False
    quote = ""
    escaped = False

    def flush() -> None:
        if not token:
            return
        raw = "".join(token)
        output.append({"true": "True", "false": "False", "null": "None"}.get(raw, raw))
        token.clear()

    for char in candidate:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            flush()
            in_string = True
            quote = char
            output.append(char)
        elif char.isalpha():
            token.append(char)
        else:
            flush()
            output.append(char)
    flush()
    return "".join(output)


def parse_json_value(raw_output: Any) -> Any:
    """Parse common model JSON variants without another paid model turn.

    Recovery is deliberately syntactic: prose/fences, single quotes, comments,
    unquoted identifier keys, trailing commas, JS literals, and truncated closing
    delimiters are accepted. Semantic defaults remain the owning tool's job.
    """

    if isinstance(raw_output, (dict, list, int, float, bool)) or raw_output is None:
        return raw_output
    candidate = _json_container_candidate(str(raw_output))
    attempts = [candidate]
    repaired = _strip_json_comments(candidate).translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    )
    repaired = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*:)", r'\1"\2"\3', repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if repaired not in attempts:
        attempts.append(repaired)

    failures: list[Exception] = []
    for value in attempts:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            failures.append(exc)
        try:
            return ast.literal_eval(_pythonize_json_literals(value))
        except (SyntaxError, ValueError) as exc:
            failures.append(exc)
    last = failures[-1] if failures else None
    raise LLMStructuredOutputError(
        f"Model returned unrecoverable JSON: {candidate[:300]!r}"
    ) from last


def parse_json_object(raw_output: Any) -> dict[str, Any]:
    """Parse a JSON-ish value and require a mapping at the provider boundary."""

    value = parse_json_value(raw_output)
    if not isinstance(value, dict):
        raise LLMStructuredOutputError("Structured model response must be a JSON object")
    return value


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
        fallbacks = tuple(self.routed_model_for(model) for model in self.fallback_models)
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
    cooldown_seconds=float(os.environ.get("SCENESMITH_LLM_CIRCUIT_COOLDOWN_SECONDS", "0"))
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


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, asyncio.TimeoutError))
        or "timeout" in name
        or "timed out" in message
        or "subscription_cli_stalled" in message
        or "produced no progress" in message
        or status == 504
    )


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
                            response = await model.get_response(*call_args, **call_kwargs)
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
                        call_args, call_kwargs = _with_cli_request_timeout(
                            args, kwargs
                        )
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
        from scenesmith.agent_utils.cli_llm_proxy import start_codex_cli_proxy

        base_url = start_codex_cli_proxy()
        api_key = "chatgpt-subscription"
    elif harness.provider == "claude-cli":
        from scenesmith.agent_utils.claude_cli_proxy import start_claude_cli_proxy

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
        os.environ.get("SCENESMITH_LLM_HARD_TIMEOUT_SECONDS", "120"),
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


def extract_json_object(raw_output: str) -> str:
    """Deterministically extract and repair one JSON object."""

    return json.dumps(parse_json_object(raw_output), separators=(",", ":"))


def _normalized_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def normalize_arguments_to_schema(value: Any, schema: dict[str, Any] | None) -> Any:
    """Coerce harmless provider differences to the declared tool schema.

    This intentionally fixes representation, not meaning: camel/snake/case key
    differences and primitive string encodings are normalized, but required
    semantic values are never invented.
    """

    if not isinstance(schema, dict):
        return value
    expected = schema.get("type")
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), None)
    if expected == "object":
        if not isinstance(value, dict):
            properties = schema.get("properties") or {}
            if len(properties) == 1:
                value = {next(iter(properties)): value}
            else:
                return value
        properties = schema.get("properties") or {}
        aliases = {_normalized_identifier(key): key for key in properties}
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            canonical = aliases.get(_normalized_identifier(key), str(key))
            if canonical not in properties and schema.get("additionalProperties") is False:
                continue
            normalized[canonical] = normalize_arguments_to_schema(
                item, properties.get(canonical)
            )
        return normalized
    if expected == "array":
        items = value if isinstance(value, (list, tuple)) else [value]
        return [normalize_arguments_to_schema(item, schema.get("items")) for item in items]
    if expected in {"number", "integer"} and isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            number = float(match.group())
            return int(number) if expected == "integer" else number
    if expected == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    if expected == "string" and value is not None and not isinstance(value, str):
        return str(value)
    enum = schema.get("enum") or []
    if enum and value not in enum:
        matched = next(
            (item for item in enum if _normalized_identifier(item) == _normalized_identifier(value)),
            None,
        )
        if matched is not None:
            return matched
    return value


def _agent_request_contract(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], Any | None]:
    tools = kwargs.get("tools")
    output_schema = kwargs.get("output_schema")
    if tools is None and len(args) >= 4:
        tools = args[3]
    if output_schema is None and len(args) >= 5:
        output_schema = args[4]
    return list(tools or []), output_schema


def normalize_agent_model_response(
    response: Any,
    *,
    tools: list[Any],
    output_schema: Any | None,
) -> Any:
    """Repair native Agents SDK tool arguments and structured final output."""

    tool_by_name = {
        _normalized_identifier(getattr(tool, "name", "")): tool
        for tool in tools
        if getattr(tool, "name", None)
    }
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call":
            supplied_name = getattr(item, "name", "")
            tool = tool_by_name.get(_normalized_identifier(supplied_name))
            if tool is None and len(tool_by_name) == 1:
                tool = next(iter(tool_by_name.values()))
            if tool is None:
                raise LLMStructuredOutputError(
                    f"Model selected unknown tool {supplied_name!r}"
                )
            arguments = parse_json_object(getattr(item, "arguments", "{}"))
            arguments = normalize_arguments_to_schema(
                arguments, getattr(tool, "params_json_schema", None)
            )
            item.name = tool.name
            item.arguments = json.dumps(arguments, separators=(",", ":"))

    if output_schema is None or getattr(output_schema, "is_plain_text", lambda: False)():
        return response
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) != "output_text":
                continue
            repaired = extract_json_object(getattr(part, "text", ""))
            try:
                output_schema.validate_json(repaired)
            except Exception as exc:
                raise LLMStructuredOutputError(
                    f"Model output did not match {output_schema.name()!r} after "
                    "deterministic JSON repair"
                ) from exc
            part.text = repaired
    return response


def _with_vision_detail(
    messages: list[dict[str, Any]], vision_detail: str
) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": message["role"], "content": content})
            continue
        parts = []
        for item in content:
            if item.get("type") != "image_url":
                parts.append(item)
                continue
            image = item.get("image_url", {})
            url = image.get("url") if isinstance(image, dict) else image
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": vision_detail},
                }
            )
        normalized.append({"role": message["role"], "content": parts})
    return normalized


def _to_responses_messages(
    messages: list[dict[str, Any]], vision_detail: str
) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": content})
            continue
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image = item.get("image_url", {})
                url = image.get("url") if isinstance(image, dict) else image
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                        "detail": vision_detail,
                    }
                )
        converted.append({"role": message["role"], "content": parts})
    return converted


def _request_json_in_prompt(
    messages: list[dict[str, Any]], *, responses_format: bool = False
) -> list[dict[str, Any]]:
    updated = [
        {"role": message["role"], "content": message.get("content", "")}
        for message in messages
    ]
    if not updated:
        return updated
    part_type = "input_text" if responses_format else "text"
    instruction = "Return only one valid JSON object, with no Markdown fence."
    content = updated[-1]["content"]
    if isinstance(content, str):
        updated[-1]["content"] = f"{content}\n\n{instruction}"
    else:
        updated[-1]["content"] = [*content, {"type": part_type, "text": instruction}]
    return updated


def _has_images(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(message.get("content"), list)
        and any(
            item.get("type") in {"image_url", "input_image"}
            for item in message["content"]
        )
        for message in messages
    )


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


def is_transient_provider_error(exc: Exception) -> bool:
    """Classify retryable provider failures without retrying invalid model output."""
    if _is_timeout_error(exc):
        return False
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if "subscription_queue_busy" in message or "queue admission" in message:
        return False
    if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
        return True
    name = type(exc).__name__.lower()
    return any(
        marker in name or marker in message
        for marker in (
            "connectionerror",
            "connection error",
            "rate limit",
            "temporarily unavailable",
        )
    )


T = TypeVar("T")


def run_with_transient_retry(operation: Callable[[int], T], *, max_attempts: int) -> T:
    """Retry transient transport/provider failures for compatibility callers."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(attempt)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts or not is_transient_provider_error(exc):
                raise
    assert last_error is not None
    raise last_error
