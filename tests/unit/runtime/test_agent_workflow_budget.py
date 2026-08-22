"""Regression tests for programmatic planner/designer/critic budgets."""

import asyncio

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.runtime.agent_runtime import AgentWorkflowTimeout
from scenesmith.agent_utils.runtime.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType


class BudgetAgent(BaseStatefulAgent):
    def __init__(self, *, critique_rounds: int = 1, timeout_initial: bool = False):
        super().__init__(
            cfg=SimpleNamespace(max_critique_rounds=critique_rounds), logger=None
        )
        self.timeout_initial = timeout_initial
        self.initial_impl_calls = 0
        self.critique_impl_calls = 0
        self.change_impl_calls = 0

    @property
    def agent_type(self) -> AgentType:
        return AgentType.FURNITURE

    def _get_final_scores_directory(self) -> Path:
        return Path("unused")

    def _get_critique_prompt_enum(self) -> Any:
        return None

    def _get_design_change_prompt_enum(self) -> Any:
        return None

    def _get_initial_design_prompt_enum(self) -> Any:
        return None

    def _get_initial_design_prompt_kwargs(self) -> dict:
        return {}

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        pass

    async def _request_initial_design_impl(self) -> str:
        self.initial_impl_calls += 1
        if self.timeout_initial:
            raise AgentWorkflowTimeout("test timeout")
        return "initial complete"

    async def _request_critique_impl(self, update_checkpoint: bool = True) -> str:
        self.critique_impl_calls += 1
        return "critique complete"

    async def _request_design_change_impl(self, instruction: str) -> str:
        self.change_impl_calls += 1
        return "change complete"


def test_each_expensive_phase_is_limited_in_code():
    agent = BudgetAgent(critique_rounds=1)

    async def run_calls():
        assert await agent._request_initial_design_bounded() == "initial complete"
        assert "already ran" in await agent._request_initial_design_bounded()
        assert await agent._request_critique_bounded() == "critique complete"
        assert "budget exhausted" in await agent._request_critique_bounded()
        assert await agent._request_design_change_bounded("fix") == "change complete"
        assert "budget exhausted" in await agent._request_design_change_bounded("fix")

    asyncio.run(run_calls())
    assert agent.initial_impl_calls == 1
    assert agent.critique_impl_calls == 1
    assert agent.change_impl_calls == 1


def test_timed_out_partial_phase_forces_planner_to_stop():
    agent = BudgetAgent(timeout_initial=True)

    async def run_calls():
        message = await agent._request_initial_design_bounded()
        assert "Keep the current scene" in message
        critique_message = await agent._request_critique_bounded()
        assert "already exhausted" in critique_message

    asyncio.run(run_calls())
    assert agent._workflow_limit_reached is True
    assert agent.initial_impl_calls == 1
    assert agent.critique_impl_calls == 0
