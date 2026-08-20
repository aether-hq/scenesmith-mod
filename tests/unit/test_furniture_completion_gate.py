"""Regression tests for semantic furniture-stage completion gates."""

from types import SimpleNamespace

import pytest

from agents.exceptions import ModelBehaviorError

from scenesmith.agent_utils.room import ObjectType
from scenesmith.furniture_agents.stateful_furniture_agent import (
    _validate_room_kit_completion,
)


def _library_kit():
    return SimpleNamespace(
        kit_id="library-reading-hall-v1",
        slots=(
            SimpleNamespace(role="bookshelf", required=True, minimum_count=2),
            SimpleNamespace(role="reading_table", required=True, minimum_count=1),
            SimpleNamespace(role="reading_chair", required=True, minimum_count=4),
            SimpleNamespace(role="task_lamp", required=False, minimum_count=1),
        ),
    )


def _scene_with_furniture(count: int):
    objects = {
        "wall": SimpleNamespace(object_type=ObjectType.WALL),
        **{
            f"furniture_{index}": SimpleNamespace(
                object_type=ObjectType.FURNITURE
            )
            for index in range(count)
        },
    }
    return SimpleNamespace(objects=objects)


def test_matched_library_kit_rejects_zero_furniture():
    with pytest.raises(
        ModelBehaviorError,
        match=r"library-reading-hall-v1.*0 furniture objects.*minimum is 7",
    ):
        _validate_room_kit_completion(_scene_with_furniture(0), _library_kit())


def test_matched_library_kit_accepts_required_minimum():
    assert _validate_room_kit_completion(
        _scene_with_furniture(7), _library_kit()
    ) == 7


def test_unmatched_room_allows_intentionally_empty_scene():
    assert _validate_room_kit_completion(_scene_with_furniture(0), None) == 0
