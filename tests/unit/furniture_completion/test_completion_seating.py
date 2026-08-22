"""Regression tests for semantic furniture-stage completion gates."""

import json
import math

from types import SimpleNamespace

import pytest

from agents.exceptions import ModelBehaviorError
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType
from scenesmith.furniture_agents.room_kit.planning import _chair_cluster_poses
from scenesmith.furniture_agents.room_kit.validation import (
    _validate_room_kit_completion,
)
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent

_EXACT_MULTILEVEL_LIBRARY_PROMPT = (
    "a large, multi-level library with thousands of books and a bunch of "
    "tables and chairs for patrons"
)


def _full_height_bookshelf(
    index: int,
    elevation: float,
    *,
    x: float | None = None,
    y: float = 0.0,
    yaw: float = 0.0,
):
    return SimpleNamespace(
        object_id=f"renaissance_bookshelf_{index}",
        object_type=ObjectType.FURNITURE,
        name="renaissance_bookshelf",
        description="full-height Renaissance library bookcase",
        bbox_min=(-0.48, -0.18, 0.0),
        bbox_max=(0.48, 0.18, 2.0),
        metadata={
            "asset_quality_score": 0.76,
            "catalog_semantics": ("Reproduction Bookcase hssd/wordnet/bookcase.n.01"),
        },
        transform=RigidTransform(
            RollPitchYaw(0.0, 0.0, math.radians(yaw)),
            ((-2.0 + index % 5) * 1.05 if x is None else x, y, elevation),
        ),
    )


def _dense_multilevel_bookshelf_kit():
    return SimpleNamespace(
        kit_id="library-reading-hall-v1",
        slots=(
            SimpleNamespace(
                role="bookshelf",
                aliases=("bookcase",),
                query=(
                    "full-height Renaissance library bookcase densely filled "
                    "with visible books"
                ),
                nominal_dimensions_m=(1.0, 0.35, 2.0),
                required=True,
                minimum_count=15,
                placement_class="wall",
            ),
        ),
    )


def _dense_multilevel_library_kit():
    return SimpleNamespace(
        kit_id="library-reading-hall-v1",
        slots=(
            *_dense_multilevel_bookshelf_kit().slots,
            SimpleNamespace(
                role="reading_table",
                aliases=("table",),
                query="library reading table",
                required=True,
                minimum_count=5,
                placement_class="floor",
            ),
            SimpleNamespace(
                role="reading_chair",
                aliases=("chair",),
                query="stationary upholstered library reading chair",
                required=True,
                minimum_count=12,
                placement_class="floor",
            ),
        ),
    )


def _role_furniture(
    role: str,
    index: int,
    elevation: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
):
    return SimpleNamespace(
        object_id=f"{role}_{index}",
        object_type=ObjectType.FURNITURE,
        name=role,
        description=role.replace("_", " "),
        metadata={"asset_quality_score": 1.0},
        transform=RigidTransform(
            RollPitchYaw(0.0, 0.0, math.radians(yaw)),
            (x, y, elevation),
        ),
    )


def _library_with_ground_only_tables():
    shelves = [
        *(_full_height_bookshelf(index, 0.0) for index in range(9)),
        *(_full_height_bookshelf(index + 9, 4.0) for index in range(5)),
        *(_full_height_bookshelf(index + 14, 8.0) for index in range(5)),
    ]
    tables = [
        _role_furniture(
            "reading_table",
            index,
            0.0,
            x=1.25 + index,
            y=-1.5,
        )
        for index in range(5)
    ]
    chairs = []
    for level_index, elevation in enumerate((0.0, 4.0, 8.0)):
        chairs.extend(
            (
                _role_furniture(
                    "reading_chair",
                    level_index * 3,
                    elevation,
                    x=1.25,
                    y=-2.8,
                    yaw=0.0,
                ),
                _role_furniture(
                    "reading_chair",
                    level_index * 3 + 1,
                    elevation,
                    x=2.55,
                    y=-1.5,
                    yaw=90.0,
                ),
                _role_furniture(
                    "reading_chair",
                    level_index * 3 + 2,
                    elevation,
                    x=1.25,
                    y=-0.2,
                    yaw=180.0,
                ),
            )
        )
    chairs.extend(
        (
            _role_furniture("reading_chair", 9, 4.0, x=-5.0, y=-5.0),
            _role_furniture("reading_chair", 10, 8.0, x=-5.0, y=-5.0),
            _role_furniture("reading_chair", 11, 8.0, x=5.0, y=-5.0),
        )
    )
    return [*shelves, *tables, *chairs]


def test_large_multilevel_library_gate_rejects_isolated_upper_chairs():
    furniture = _library_with_ground_only_tables()
    furniture.extend(
        (
            _role_furniture("reading_table", 5, 4.0, x=-2.8, y=2.5),
            _role_furniture("reading_table", 6, 8.0, x=-2.8, y=2.5),
        )
    )
    for obj in furniture:
        if obj.name == "reading_chair" and obj.transform.translation()[2] > 0:
            elevation = float(obj.transform.translation()[2])
            index = int(obj.object_id.rsplit("_", 1)[1])
            obj.transform = RigidTransform(
                RollPitchYaw(0.0, 0.0, 0.0),
                (4.5 - index * 0.2, -4.5, elevation),
            )
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in furniture},
    )

    with pytest.raises(
        ModelBehaviorError,
        match=r"patron ensemble at 4\.000m.*required 3",
    ):
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_library_kit(),
            support_elevations=(0.0, 4.0, 8.0),
        )


def test_large_multilevel_library_recovery_places_chairs_around_upper_tables():
    furniture = _library_with_ground_only_tables()
    furniture.extend(
        (
            _role_furniture("reading_table", 5, 4.0, x=-2.8, y=2.5),
            _role_furniture("reading_table", 6, 8.0, x=-2.8, y=2.5),
        )
    )
    for obj in furniture:
        if obj.name == "reading_chair" and obj.transform.translation()[2] > 0:
            elevation = float(obj.transform.translation()[2])
            obj.transform = RigidTransform(
                RollPitchYaw(0.0, 0.0, 0.0),
                (5.0, -5.0, elevation),
            )
    chair_asset = next(obj for obj in furniture if obj.name == "reading_chair")
    placements = []

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 4.0, 8.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            placements.append(kwargs)
            return json.dumps(
                {"success": True, "object_id": f"recovered_{len(placements)}"}
            )

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in furniture},
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [chair_asset])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_library_kit()
        )
        == 9
    )
    assert [call["z"] for call in placements] == [0.0] + [4.0] * 4 + [8.0] * 4
    for call in placements[1:]:
        distance = math.dist((call["x"], call["y"]), (-2.8, 2.5))
        assert 0.75 <= distance <= 2.25


def test_chair_cluster_search_covers_outer_strict_annulus():
    table = SimpleNamespace(
        transform=RigidTransform(p=[-3.98, 2.01, 4.0]),
        bbox_min=(-0.375, -0.375, 0.0),
        bbox_max=(0.375, 0.375, 0.75),
    )
    chair = SimpleNamespace(
        bbox_min=(-0.3, -0.3, 0.0),
        bbox_max=(0.3, 0.3, 0.9),
    )

    poses = _chair_cluster_poses(table, chair)
    distances = [math.dist((x, y), (-3.98, 2.01)) for x, y, _yaw in poses]

    assert min(distances) >= 0.75
    assert max(distances) <= 2.25
    assert max(distances) >= 1.75


def test_library_recovery_extends_the_existing_table_ensemble():
    furniture = _library_with_ground_only_tables()
    furniture.extend(
        (
            _role_furniture("reading_table", 5, 4.0, x=1.25, y=-1.5),
            _role_furniture("reading_table", 6, 8.0, x=1.25, y=-1.5),
        )
    )
    ground_chairs = [
        obj
        for obj in furniture
        if obj.name == "reading_chair" and obj.transform.translation()[2] == 0.0
    ]
    ground_chairs[0].transform = RigidTransform(
        RollPitchYaw(0.0, 0.0, math.radians(90.0)),
        (6.9, -1.5, 0.0),
    )
    for index, chair in enumerate(ground_chairs[1:]):
        chair.transform = RigidTransform(
            RollPitchYaw(0.0, 0.0, 0.0),
            (-5.0 + index, 5.0, 0.0),
        )
    chair_asset = ground_chairs[0]
    placements = []

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 4.0, 8.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            placements.append(kwargs)
            return json.dumps(
                {"success": True, "object_id": f"recovered_{len(placements)}"}
            )

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in furniture},
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [chair_asset])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_library_kit()
        )
        == 5
    )
    assert [call["z"] for call in placements] == [0.0] * 3 + [4.0, 8.0]
    for call in placements:
        level_tables = [
            obj
            for obj in furniture
            if obj.name == "reading_table"
            and float(obj.transform.translation()[2]) == call["z"]
        ]
        distance = min(
            math.dist(
                (call["x"], call["y"]),
                tuple(table.transform.translation()[:2]),
            )
            for table in level_tables
        )
        assert 0.75 <= distance <= 2.25


def test_library_recovery_rolls_back_incomplete_new_chair_cluster():
    furniture = _library_with_ground_only_tables()
    furniture.extend(
        (
            _role_furniture("reading_table", 5, 4.0, x=-2.8, y=2.5),
            _role_furniture("reading_table", 6, 8.0, x=-2.8, y=2.5),
        )
    )
    for obj in furniture:
        if obj.name == "reading_chair" and obj.transform.translation()[2] > 0:
            elevation = float(obj.transform.translation()[2])
            obj.transform = RigidTransform(
                RollPitchYaw(0.0, 0.0, 0.0),
                (5.0, -5.0, elevation),
            )
    chair_asset = next(obj for obj in furniture if obj.name == "reading_chair")
    attempts = []
    removed = []

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 4.0, 8.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) <= 2:
                return json.dumps(
                    {"success": True, "object_id": f"recovered_{len(attempts)}"}
                )
            return json.dumps({"success": False, "object_id": ""})

        def _remove_furniture_impl(self, object_id):
            removed.append(object_id)
            return json.dumps({"success": True})

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in furniture},
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [chair_asset])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_library_kit()
        )
        == 1
    )
    assert removed == ["recovered_2"]
