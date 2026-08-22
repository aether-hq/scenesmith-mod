"""Timeout and cancellation policy for local CLI bridge requests."""

from __future__ import annotations

import logging
import os

from dataclasses import replace
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from scenesmith.agent_utils.llm.contracts.errors import LLMHarnessError

console_logger = logging.getLogger(__name__)


def _cli_proxy_http_timeout_seconds() -> float:
    timeout = float(os.environ.get("SCENESMITH_LLM_PROXY_HTTP_TIMEOUT_SECONDS", "330"))
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
