"""Tests for durable floor-plan checkpoints around long critic turns."""

import json
import asyncio

import pytest

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.experiments.indoor_scene_generation import (
    _copy_checkpoint_for_stage,
)


def test_valid_layout_is_atomically_checkpointed_before_critique(tmp_path):
    agent = StatefulFloorPlanAgent.__new__(StatefulFloorPlanAgent)
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent.layout = MagicMock()
    agent.layout.room_specs = [SimpleNamespace(room_id="studio")]
    agent.layout.placement_valid = True
    agent.layout.connectivity_valid = True
    agent.layout.to_dict.return_value = {"rooms": [{"room_id": "studio"}]}
    agent._generate_all_room_geometries = MagicMock()

    saved = agent._write_resumable_layout_checkpoint()

    assert saved is True
    assert json.loads((tmp_path / "house_layout.json").read_text()) == {
        "rooms": [{"room_id": "studio"}]
    }
    assert not (tmp_path / "house_layout.json.pending").exists()
    agent._generate_all_room_geometries.assert_called_once_with(
        output_dir=tmp_path / "floor_plans"
    )


def test_invalid_layout_is_not_exposed_as_resumable(tmp_path):
    agent = StatefulFloorPlanAgent.__new__(StatefulFloorPlanAgent)
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent.layout = MagicMock()
    agent.layout.room_specs = [SimpleNamespace(room_id="studio")]
    agent.layout.placement_valid = True
    agent.layout.connectivity_valid = False
    agent._generate_all_room_geometries = MagicMock()

    saved = agent._write_resumable_layout_checkpoint()

    assert saved is False
    assert not (tmp_path / "house_layout.json").exists()
    agent._generate_all_room_geometries.assert_not_called()


def test_generation_refuses_to_export_an_invalid_partial_layout(tmp_path):
    agent = StatefulFloorPlanAgent.__new__(StatefulFloorPlanAgent)
    agent.mode = "room"
    agent.cfg = SimpleNamespace(
        max_critique_rounds=0,
        min_floor_plan_dim_m=1.5,
        max_floor_plan_dim_m=20.0,
        wall_height=SimpleNamespace(min=2.0, max=12.0),
    )
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent._reset_workflow_budget = MagicMock()
    agent._create_designer_tools = MagicMock(return_value=[])
    agent._create_designer_agent = MagicMock(return_value=MagicMock())
    agent._create_critic_tools = MagicMock(return_value=[])
    agent._create_critic_agent = MagicMock(return_value=MagicMock())
    agent._create_planner_tools = MagicMock(return_value=[])
    agent._create_planner_agent = MagicMock(return_value=MagicMock())
    agent.prompt_registry = MagicMock()
    agent._run_planner_with_partial_recovery = MagicMock(return_value=None)
    agent._request_initial_design_impl = AsyncMock(return_value=None)
    agent._write_resumable_layout_checkpoint = MagicMock(return_value=False)
    agent._generate_all_room_geometries = MagicMock()
    agent._export_floor_plan = MagicMock()

    async def run_planner(**_kwargs):
        return None

    agent._run_planner_with_partial_recovery = run_planner

    with pytest.raises(RuntimeError, match="structurally valid checkpoint"):
        asyncio.run(agent.generate_house_layout("empty room", tmp_path / "floor_plans"))

    agent._generate_all_room_geometries.assert_not_called()
    agent._export_floor_plan.assert_not_called()


def test_wall_resume_copies_canonical_checkpoint_without_legacy_room_geometry(
    tmp_path,
):
    source = tmp_path / "source" / "scene_000"
    target = tmp_path / "target" / "scene_000"
    (source / "floor_plans" / "room" / "structural").mkdir(parents=True)
    (source / "floor_plans" / "room" / "structural" / "platform.glb").write_text(
        "platform"
    )
    (source / "house_layout.json").write_text('{"rooms": []}')
    checkpoint = source / "room_room" / "scene_states" / "scene_after_furniture"
    checkpoint.mkdir(parents=True)
    (checkpoint / "scene_state.json").write_text("{}")

    _copy_checkpoint_for_stage(source, target, "wall_mounted")

    assert (
        target / "floor_plans" / "room" / "structural" / "platform.glb"
    ).read_text() == "platform"
    assert (target / "house_layout.json").is_file()
    assert (
        target
        / "room_room"
        / "scene_states"
        / "scene_after_furniture"
        / "scene_state.json"
    ).is_file()
    assert not (target / "room_geometry").exists()


def test_wall_resume_still_copies_legacy_room_geometry(tmp_path):
    source = tmp_path / "source" / "scene_000"
    target = tmp_path / "target" / "scene_000"
    (source / "floor_plans").mkdir(parents=True)
    (source / "room_geometry").mkdir(parents=True)
    (source / "room_geometry" / "room.sdf").write_text("legacy")
    (source / "house_layout.json").write_text('{"rooms": []}')

    _copy_checkpoint_for_stage(source, target, "wall_mounted")

    assert (target / "room_geometry" / "room.sdf").read_text() == "legacy"
