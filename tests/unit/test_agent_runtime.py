import asyncio

from unittest.mock import patch

import pytest

from scenesmith.agent_utils.agent_runtime import (
    AgentWorkflowTimeout,
    BoundedRunner,
    agent_run_timeout_seconds,
    minimum_routed_workflow_timeout_seconds,
)


async def _stuck_run(*args, **kwargs):
    await asyncio.sleep(10)


def test_bounded_runner_cancels_stuck_workflow(monkeypatch):
    with patch(
        "scenesmith.agent_utils.agent_runtime.OpenAIRunner.run",
        side_effect=_stuck_run,
    ):
        with pytest.raises(AgentWorkflowTimeout, match="exceeded 0.01s"):
            asyncio.run(BoundedRunner.run(timeout_seconds=0.01))


def test_bounded_runner_stops_orphaned_subscription_turn(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CLI_PROXY_URL", "http://127.0.0.1:1/v1/")
    with (
        patch(
            "scenesmith.agent_utils.agent_runtime.OpenAIRunner.run",
            side_effect=_stuck_run,
        ),
        patch(
            "scenesmith.agent_utils.agent_runtime.cancel_subscription_turn",
            return_value=True,
        ) as cancel,
    ):
        with pytest.raises(AgentWorkflowTimeout):
            asyncio.run(BoundedRunner.run(timeout_seconds=0.01))

    cancel.assert_called_once_with()


def test_bounded_runner_accepts_per_call_timeout(monkeypatch):
    monkeypatch.setenv("SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS", "10")
    with patch(
        "scenesmith.agent_utils.agent_runtime.OpenAIRunner.run",
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


def test_workflow_timeout_does_not_expand_into_route_retry_product(monkeypatch):
    monkeypatch.setenv("SCENESMITH_AGENT_RUN_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("SCENESMITH_LLM_MODEL", "sonnet")
    monkeypatch.setenv("SCENESMITH_LLM_FALLBACK_MODELS", "haiku,opus")
    monkeypatch.setenv("SCENESMITH_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("SCENESMITH_LLM_MAX_ATTEMPTS", "2")

    minimum = minimum_routed_workflow_timeout_seconds(max_turns=2)

    assert minimum >= 370
    assert agent_run_timeout_seconds("designer", max_turns=2) == 5
