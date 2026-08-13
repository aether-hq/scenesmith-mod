"""Provider-neutral OpenAI-compatible client construction.

SceneSmith normally uses standard Bearer authentication. Some compatible
providers use a different Authorization scheme; keeping that choice here avoids
embedding provider-specific clients throughout the agent stages.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI, OpenAI


def _client_options() -> dict[str, object]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for SceneSmith inference")

    options: dict[str, object] = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        options["base_url"] = base_url

    scheme = os.environ.get("SCENESMITH_INFERENCE_AUTH_SCHEME", "bearer").lower()
    if scheme == "key":
        options["api_key"] = "openai-compatible"
        options["default_headers"] = {"Authorization": f"Key {api_key}"}
    elif scheme != "bearer":
        raise RuntimeError(f"unsupported SceneSmith inference auth scheme: {scheme}")
    return options


def create_openai_client() -> OpenAI:
    """Create the synchronous client used by direct VLM calls."""
    return OpenAI(**_client_options())


def create_async_openai_client() -> AsyncOpenAI:
    """Create the asynchronous client shared by the Agents SDK and summaries."""
    return AsyncOpenAI(**_client_options())


def agents_sdk_model_name(model: str) -> str:
    """Force slash-qualified model IDs through the configured OpenAI seam.

    The Agents SDK otherwise interprets the text before the first slash as a
    native provider selector. OpenAI-compatible gateways need the complete
    model ID (for example ``google/gemini-...``) to reach their router.
    """
    if "/" in model and not model.startswith("openai/"):
        return f"openai/{model}"
    return model


def configure_agents_sdk() -> None:
    """Bind the Agents SDK to the same OpenAI-compatible inference seam."""
    from agents import (
        set_default_openai_api,
        set_default_openai_client,
        set_tracing_disabled,
    )

    set_default_openai_client(create_async_openai_client(), use_for_tracing=False)
    set_default_openai_api("chat_completions")
    if not os.environ.get("OPENAI_TRACING_KEY"):
        set_tracing_disabled(True)
