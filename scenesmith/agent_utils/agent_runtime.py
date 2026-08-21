"""Bounded adapter for complete OpenAI Agents SDK workflows."""

import asyncio
import json
import logging
import os

from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from agents import Runner as OpenAIRunner

from scenesmith.agent_utils.llm_harness import LLMHarnessConfig


console_logger = logging.getLogger(__name__)
DEFAULT_AGENT_RUN_TIMEOUT_SECONDS = 120.0


class AgentWorkflowTimeout(TimeoutError):
    """A recoverable wall-clock limit for an agent workflow."""


def subscription_turn_active() -> bool:
    """Return whether the local subscription proxy is executing a model turn."""
    base_url = os.environ.get("SCENESMITH_CLI_PROXY_URL")
    if not base_url:
        return False
    request = Request(urljoin(base_url, "status"), method="GET")
    try:
        with urlopen(request, timeout=0.5) as response:
            payload = json.loads(response.read())
        return payload.get("status") == "active"
    except Exception as exc:
        console_logger.warning("Could not read subscription CLI status: %s", exc)
        return False


def cancel_subscription_turn() -> bool:
    """Stop the CLI process behind a cancelled loopback model request."""
    base_url = os.environ.get("SCENESMITH_CLI_PROXY_URL")
    if not base_url:
        return False
    request = Request(urljoin(base_url, "cancel"), data=b"", method="POST")
    try:
        with urlopen(request, timeout=2.0) as response:
            return response.status in {200, 202}
    except Exception as exc:
        console_logger.warning("Could not cancel subscription CLI turn: %s", exc)
        return False


def minimum_routed_workflow_timeout_seconds(max_turns: int = 1) -> float:
    """Return enough time for every bounded model route in a workflow.

    ``SCENESMITH_LLM_TIMEOUT_SECONDS`` is a per-provider-response deadline. An
    Agents SDK workflow can legitimately make several responses, and each one
    may use all configured retries and fallback models. Keep the outer workflow
    deadline outside that complete bounded envelope so it never cancels a valid
    fallback just because a primary route was slow.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    config = LLMHarnessConfig.from_env()
    route_count = len(config.model_chain)
    response_budget = (
        config.timeout_seconds * config.max_attempts * route_count * max_turns
    )
    retry_backoff_budget = (
        0.25 * max(0, config.max_attempts - 1) * route_count * max_turns
    )
    cleanup_grace = max(10.0, 2.0 * max_turns)
    return response_budget + retry_backoff_budget + cleanup_grace


def agent_run_timeout_seconds(
    role: str | None = None,
    *,
    max_turns: int = 1,
) -> float:
    """Return the configured total deadline for one agent workflow.

    Per-response route retries are bounded independently by the LLM harness.
    Expanding this deadline by ``routes × retries × turns`` previously turned a
    requested two-minute workflow ceiling into an hour-long effective timeout.
    The role setting is an actual wall-clock budget, not a lower bound.
    """
    role_key = f"SCENESMITH_AGENT_{role.upper()}_TIMEOUT_SECONDS" if role else None
    raw = (
        os.environ.get(role_key)
        if role_key is not None and os.environ.get(role_key) is not None
        else os.environ.get(
            "SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS",
            str(DEFAULT_AGENT_RUN_TIMEOUT_SECONDS),
        )
    )
    timeout = float(raw)
    if timeout <= 0:
        setting = role_key or "SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS"
        raise ValueError(f"{setting} must be positive")
    return timeout


class BoundedRunner:
    """Runner-compatible facade enforcing one wall-clock workflow deadline."""

    @staticmethod
    async def run(*args: Any, **kwargs: Any) -> Any:
        # This is a facade-only argument and must not reach the Agents SDK.
        timeout = float(kwargs.pop("timeout_seconds", agent_run_timeout_seconds()))
        configured_active_stream_hard_timeout = float(
            kwargs.pop(
                "active_stream_hard_timeout_seconds",
                os.environ.get(
                    "SCENESMITH_AGENT_ACTIVE_STREAM_HARD_TIMEOUT_SECONDS",
                    os.environ.get("SCENESMITH_LLM_HARD_TIMEOUT_SECONDS", "300"),
                ),
            )
        )
        active_stream_hard_timeout = max(timeout, configured_active_stream_hard_timeout)
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if configured_active_stream_hard_timeout <= 0:
            raise ValueError("active_stream_hard_timeout_seconds must be positive")
        started = asyncio.get_running_loop().time()
        task = asyncio.create_task(OpenAIRunner.run(*args, **kwargs))
        limit = timeout
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if not done and await asyncio.to_thread(subscription_turn_active):
                limit = active_stream_hard_timeout
                console_logger.info(
                    "Agent workflow reached %.1fs with an active subscription turn; "
                    "allowing it to continue to the %.1fs hard ceiling",
                    timeout,
                    active_stream_hard_timeout,
                )
                remaining = max(
                    0.0,
                    active_stream_hard_timeout
                    - (asyncio.get_running_loop().time() - started),
                )
                done, _ = await asyncio.wait({task}, timeout=remaining)
            if done:
                result = await task
            else:
                raise asyncio.TimeoutError
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        except TimeoutError as exc:
            elapsed = asyncio.get_running_loop().time() - started
            console_logger.error(
                "Agent workflow exceeded %.1fs after %.1fs; cancelling",
                limit,
                elapsed,
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Cancelling the async HTTP client does not stop the proxy's worker
            # thread. Explicitly terminate its active CLI process so the next
            # planner turn cannot queue behind an orphan.
            await asyncio.to_thread(cancel_subscription_turn)
            raise AgentWorkflowTimeout(f"Agent workflow exceeded {limit:g}s") from exc
        console_logger.info(
            "Agent workflow completed in %.3fs",
            asyncio.get_running_loop().time() - started,
        )
        return result
