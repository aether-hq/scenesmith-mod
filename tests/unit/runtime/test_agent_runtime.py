import asyncio
import logging
import threading
import time

from unittest.mock import patch

import pytest

from scenesmith.agent_utils.runtime.agent_runtime import (
    AgentWorkflowTimeout,
    BoundedRunner,
    agent_run_timeout_seconds,
    minimum_routed_workflow_timeout_seconds,
)


async def _stuck_run(*args, **kwargs):
    await asyncio.sleep(10)


async def _slow_but_healthy_run(*args, **kwargs):
    await asyncio.sleep(0.04)
    return "completed"


async def _synchronously_silent_run(*args, **kwargs):
    time.sleep(0.08)
    return "incorrectly completed"


async def _cancellation_resistant_run(release):
    while not release.is_set():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            continue


def test_bounded_runner_cancels_stuck_workflow(monkeypatch):
    with patch(
        "scenesmith.agent_utils.runtime.agent_runtime.OpenAIRunner.run",
        side_effect=_stuck_run,
    ):
        with pytest.raises(AgentWorkflowTimeout, match="exceeded 0.01s"):
            asyncio.run(BoundedRunner.run(timeout_seconds=0.01))


def test_bounded_runner_stops_orphaned_subscription_turn(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CLI_PROXY_URL", "http://127.0.0.1:1/v1/")
    with (
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.OpenAIRunner.run",
            side_effect=_stuck_run,
        ),
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.cancel_subscription_turn",
            return_value=True,
        ) as cancel,
    ):
        with pytest.raises(AgentWorkflowTimeout):
            asyncio.run(BoundedRunner.run(timeout_seconds=0.01))

    cancel.assert_called_once_with()


def test_bounded_runner_cancels_fallback_routes_started_during_unwind():
    release = threading.Event()
    cancel_calls = 0

    async def resistant_run(*args, **kwargs):
        return await _cancellation_resistant_run(release)

    def cancel_route():
        nonlocal cancel_calls
        cancel_calls += 1
        if cancel_calls == 3:
            release.set()
        return True

    with (
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.OpenAIRunner.run",
            side_effect=resistant_run,
        ),
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.cancel_subscription_turn",
            side_effect=cancel_route,
        ),
    ):
        with pytest.raises(AgentWorkflowTimeout):
            asyncio.run(BoundedRunner.run(timeout_seconds=0.01))

    assert cancel_calls == 3


def test_bounded_runner_accepts_per_call_timeout(monkeypatch):
    monkeypatch.setenv("SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS", "10")
    with patch(
        "scenesmith.agent_utils.runtime.agent_runtime.OpenAIRunner.run",
        side_effect=_stuck_run,
    ):
        with pytest.raises(AgentWorkflowTimeout, match="exceeded 0.01s"):
            asyncio.run(BoundedRunner.run(timeout_seconds=0.01))


def test_role_timeout_overrides_global(monkeypatch):
    monkeypatch.setenv("SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("SCENESMITH_AGENT_CRITIC_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("SCENESMITH_LLM_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("SCENESMITH_LLM_MAX_ATTEMPTS", "1")
    monkeypatch.delenv("SCENESMITH_LLM_FALLBACK_MODELS", raising=False)
    assert agent_run_timeout_seconds("critic") == 45
    assert agent_run_timeout_seconds("designer") == 180


def test_spatial_compiler_has_a_bounded_large_schema_deadline(monkeypatch):
    monkeypatch.delenv("SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv(
        "SCENESMITH_AGENT_SPATIAL_COMPILER_TIMEOUT_SECONDS", raising=False
    )

    assert agent_run_timeout_seconds("spatial_compiler") == 600


def test_workflow_timeout_does_not_expand_into_route_retry_product(monkeypatch):
    monkeypatch.setenv("SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("SCENESMITH_LLM_MODEL", "sonnet")
    monkeypatch.setenv("SCENESMITH_LLM_FALLBACK_MODELS", "haiku,opus")
    monkeypatch.setenv("SCENESMITH_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("SCENESMITH_LLM_MAX_ATTEMPTS", "2")

    minimum = minimum_routed_workflow_timeout_seconds(max_turns=2)

    assert minimum >= 370
    assert agent_run_timeout_seconds("designer", max_turns=2) == 5


def test_bounded_runner_allows_active_subscription_turn_to_finish():
    with (
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.OpenAIRunner.run",
            side_effect=_slow_but_healthy_run,
        ),
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.subscription_turn_active",
            return_value=True,
            create=True,
        ),
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.cancel_subscription_turn"
        ) as cancel,
    ):
        result = asyncio.run(
            BoundedRunner.run(
                timeout_seconds=0.01,
                active_stream_hard_timeout_seconds=0.08,
            )
        )

    assert result == "completed"
    cancel.assert_not_called()


def test_bounded_runner_still_stops_active_turn_at_hard_ceiling():
    with (
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.OpenAIRunner.run",
            side_effect=_stuck_run,
        ),
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.subscription_turn_active",
            return_value=True,
            create=True,
        ),
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.cancel_subscription_turn",
            return_value=True,
        ) as cancel,
    ):
        with pytest.raises(AgentWorkflowTimeout, match="exceeded 0.03s"):
            asyncio.run(
                BoundedRunner.run(
                    timeout_seconds=0.01,
                    active_stream_hard_timeout_seconds=0.03,
                )
            )

    # The hard-deadline watchdog cancels the active worker immediately. Cleanup
    # may repeat that request while the SDK task unwinds so a fallback route
    # cannot escape the same deadline.
    assert cancel.call_count >= 1
    assert all(call.args == () and call.kwargs == {} for call in cancel.call_args_list)


def test_bounded_runner_watchdog_stops_synchronously_silent_workflow(caplog):
    caplog.set_level(logging.INFO)
    with (
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.OpenAIRunner.run",
            side_effect=_synchronously_silent_run,
        ),
        patch(
            "scenesmith.agent_utils.runtime.agent_runtime.cancel_subscription_turn",
            return_value=True,
        ) as cancel,
    ):
        with pytest.raises(AgentWorkflowTimeout, match="exceeded 0.03s"):
            asyncio.run(
                BoundedRunner.run(
                    timeout_seconds=0.01,
                    active_stream_hard_timeout_seconds=0.03,
                    heartbeat_interval_seconds=0.005,
                )
            )

    assert "Agent workflow active for" in caplog.text
    cancel.assert_called_once_with()
