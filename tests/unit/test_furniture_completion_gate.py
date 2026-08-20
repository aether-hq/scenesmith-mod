"""Regression tests for semantic furniture-stage completion gates."""

import json

from types import SimpleNamespace

import pytest

from agents.exceptions import ModelBehaviorError

from scenesmith.agent_utils.room import ObjectType
from scenesmith.furniture_agents.stateful_furniture_agent import (
    StatefulFurnitureAgent,
    _validate_room_kit_completion,
)


def _library_kit():
    return SimpleNamespace(
        kit_id="library-reading-hall-v1",
        slots=(
            SimpleNamespace(
                role="bookshelf",
                aliases=("bookcase",),
                required=True,
                minimum_count=2,
                placement_class="wall",
                facing_target="room_center",
            ),
            SimpleNamespace(
                role="reading_table",
                aliases=("table",),
                required=True,
                minimum_count=1,
                placement_class="floor",
                facing_target="primary_aisle",
            ),
            SimpleNamespace(
                role="reading_chair",
                aliases=("chair",),
                required=True,
                minimum_count=4,
                placement_class="floor",
                facing_target="reading_table",
            ),
            SimpleNamespace(
                role="task_lamp",
                aliases=("lamp",),
                required=False,
                minimum_count=1,
                placement_class="surface",
                facing_target=None,
            ),
        ),
    )


def _scene_with_furniture(count: int):
    objects = {
        "wall": SimpleNamespace(object_type=ObjectType.WALL),
        **{
            f"furniture_{index}": SimpleNamespace(object_type=ObjectType.FURNITURE)
            for index in range(count)
        },
    }
    return SimpleNamespace(objects=objects)


def _scene_with_role_counts(**role_counts: int):
    objects = {"wall": SimpleNamespace(object_type=ObjectType.WALL)}
    for role, count in role_counts.items():
        objects.update(
            {
                f"{role}_{index}": SimpleNamespace(
                    object_type=ObjectType.FURNITURE,
                    name=role,
                    description=role.replace("_", " "),
                )
                for index in range(count)
            }
        )
    return SimpleNamespace(objects=objects)


def test_matched_library_kit_rejects_zero_furniture():
    with pytest.raises(
        ModelBehaviorError,
        match=r"library-reading-hall-v1.*0 furniture objects.*minimum is 7",
    ):
        _validate_room_kit_completion(_scene_with_furniture(0), _library_kit())


def test_matched_library_kit_accepts_required_minimum():
    scene = _scene_with_role_counts(
        bookshelf=2,
        reading_table=1,
        reading_chair=4,
    )

    assert _validate_room_kit_completion(scene, _library_kit()) == 7


def test_matched_library_kit_rejects_role_deficit_despite_large_total():
    room_kit = _library_kit()
    room_kit.slots[2].minimum_count = 12
    scene = _scene_with_role_counts(
        bookshelf=27,
        classical_statue=3,
        reading_table=7,
        reading_chair=8,
        task_lamp=1,
    )

    with pytest.raises(ModelBehaviorError, match=r"reading_chair.*8.*12"):
        _validate_room_kit_completion(scene, room_kit)


def test_unmatched_room_allows_intentionally_empty_scene():
    assert _validate_room_kit_completion(_scene_with_furniture(0), None) == 0


def test_library_kit_recovers_required_minimums_from_cached_assets():
    assets = [
        SimpleNamespace(
            object_id=f"{role}_0",
            object_type=ObjectType.FURNITURE,
            name=role,
            description=role.replace("_", " "),
            metadata={"asset_quality_score": 1.0},
        )
        for role in ("bookshelf", "reading_table", "reading_chair", "task_lamp")
    ]
    placements = []

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 4.0, 8.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            placements.append(kwargs)
            return json.dumps({"success": True})

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = SimpleNamespace(
        objects={"wall": SimpleNamespace(object_type=ObjectType.WALL)},
        room_geometry=SimpleNamespace(length=7.0, width=7.0),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: assets)
    agent.furniture_tools = FakeTools()

    placed = agent._place_room_kit_minimums_deterministically(_library_kit())

    assert placed == 7
    placed_roles = [call["asset_id"].removesuffix("_0") for call in placements]
    assert placed_roles.count("bookshelf") == 2
    assert placed_roles.count("reading_table") == 1
    assert placed_roles.count("reading_chair") == 4
    assert "task_lamp" not in placed_roles
    assert len({(call["x"], call["y"], call["z"]) for call in placements}) == 7
    assert {call["z"] for call in placements} == {0.0, 4.0, 8.0}
