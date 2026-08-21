"""Bounded adapter for complete OpenAI Agents SDK workflows."""

import asyncio
import json
import logging
import os
import threading
import time

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
        heartbeat_interval = float(
            kwargs.pop(
                "heartbeat_interval_seconds",
                os.environ.get("SCENESMITH_AGENT_HEARTBEAT_SECONDS", "10"),
            )
        )
        active_stream_hard_timeout = max(timeout, configured_active_stream_hard_timeout)
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if configured_active_stream_hard_timeout <= 0:
            raise ValueError("active_stream_hard_timeout_seconds must be positive")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        started = asyncio.get_running_loop().time()
        monitor_started = time.monotonic()
        monitor_stop = threading.Event()
        monitor_deadline_exceeded = threading.Event()
        monitor_cancel_started = threading.Event()

        def monitor_workflow() -> None:
            """Keep bounded liveness visible even if the event loop is blocked."""

            hard_deadline = monitor_started + active_stream_hard_timeout
            next_heartbeat = monitor_started + heartbeat_interval
            while True:
                wake_at = min(next_heartbeat, hard_deadline)
                if monitor_stop.wait(max(0.0, wake_at - time.monotonic())):
                    return
                now = time.monotonic()
                if now >= hard_deadline:
                    monitor_deadline_exceeded.set()
                    console_logger.error(
                        "Agent workflow independent watchdog reached the %.1fs hard "
                        "ceiling; cancelling subscription worker",
                        active_stream_hard_timeout,
                    )
                    monitor_cancel_started.set()
                    cancel_subscription_turn()
                    return
                console_logger.info(
                    "Agent workflow active for %.1fs (bounded hard ceiling %.1fs)",
                    now - monitor_started,
                    active_stream_hard_timeout,
                )
                next_heartbeat = now + heartbeat_interval

        monitor = threading.Thread(
            target=monitor_workflow,
            name="scenesmith-agent-workflow-monitor",
            daemon=True,
        )
        monitor.start()
        task = asyncio.create_task(OpenAIRunner.run(*args, **kwargs))
        limit = timeout
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            if monitor_deadline_exceeded.is_set():
                limit = active_stream_hard_timeout
                raise asyncio.TimeoutError
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
            if monitor_deadline_exceeded.is_set():
                limit = active_stream_hard_timeout
                raise asyncio.TimeoutError
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
            if not monitor_cancel_started.is_set():
                await asyncio.to_thread(cancel_subscription_turn)
            raise AgentWorkflowTimeout(f"Agent workflow exceeded {limit:g}s") from exc
        finally:
            monitor_stop.set()
            monitor.join(timeout=min(heartbeat_interval, 0.1))
        console_logger.info(
            "Agent workflow completed in %.3fs",
            asyncio.get_running_loop().time() - started,
        )
        return result
