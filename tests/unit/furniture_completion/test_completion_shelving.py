"""Regression tests for semantic furniture-stage completion gates."""

import json
import math

from types import SimpleNamespace

import pytest

from agents.exceptions import ModelBehaviorError
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType
from scenesmith.furniture_agents.room_kit.validation import (
    _validate_room_kit_completion,
)
from scenesmith.furniture_agents.stateful_furniture_agent import (
    StatefulFurnitureAgent,
    _normalize_dense_library_bookcases,
)

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


def test_large_multilevel_library_gate_rejects_ground_only_bookshelves():
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={
            shelf.object_id: shelf
            for shelf in (_full_height_bookshelf(index, 0.0) for index in range(16))
        },
    )

    with pytest.raises(ModelBehaviorError, match=r"bookshelf.*4\.000m.*0.*5"):
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 4.0, 8.0),
        )


def test_large_multilevel_library_gate_rejects_sparse_upper_bookshelves():
    shelves = [
        *(_full_height_bookshelf(index, 0.0) for index in range(15)),
        *(_full_height_bookshelf(index + 15, 4.0) for index in range(3)),
        *(_full_height_bookshelf(index + 18, 8.0) for index in range(3)),
    ]
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
    )

    with pytest.raises(ModelBehaviorError, match=r"bookshelf.*4\.000m.*3.*5"):
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 4.0, 8.0),
        )


def test_large_multilevel_library_gate_rejects_isolated_bookcase_pillars():
    shelves = []
    isolated_poses = (
        (-4.8, 5.5, 180.0),
        (-1.8, 5.5, 180.0),
        (1.8, 5.5, 180.0),
        (4.8, 5.5, 180.0),
        (5.5, -2.0, 90.0),
    )
    for level_index, elevation in enumerate((0.0, 4.0, 8.0)):
        shelves.extend(
            _full_height_bookshelf(
                level_index * len(isolated_poses) + pose_index,
                elevation,
                x=x,
                y=y,
                yaw=yaw,
            )
            for pose_index, (x, y, yaw) in enumerate(isolated_poses)
        )
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
    )

    with pytest.raises(
        ModelBehaviorError,
        match=r"bookshelf wall run at 0\.000m.*1.*3",
    ):
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 4.0, 8.0),
        )


def test_large_multilevel_library_gate_accepts_contiguous_bookcase_wall_runs():
    shelves = []
    for level_index, elevation in enumerate((0.0, 4.0, 8.0)):
        shelves.extend(
            _full_height_bookshelf(
                level_index * 5 + position_index,
                elevation,
                x=-1.05 + position_index * 1.05,
                y=5.5,
                yaw=180.0,
            )
            for position_index in range(5)
        )
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
    )

    assert (
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 4.0, 8.0),
            enforce_exact_level_counts=True,
        )
        == 15
    )
    for elevation in (0.0, 4.0, 8.0):
        assert (
            sum(
                shelf.metadata.get("dense_library_grouped_run") == elevation
                for shelf in shelves
            )
            >= 3
        )


def test_large_multilevel_library_exact_gate_rejects_unpruned_surplus():
    shelves = []
    for level_index, elevation in enumerate((0.0, 4.0, 8.0)):
        shelves.extend(
            _full_height_bookshelf(
                level_index * 6 + position_index,
                elevation,
                x=-2.625 + position_index * 1.05,
                y=5.5,
                yaw=180.0,
            )
            for position_index in range(6)
        )
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
    )

    with pytest.raises(ModelBehaviorError, match=r"noncanonical bookshelf density"):
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 4.0, 8.0),
            enforce_exact_level_counts=True,
        )


def test_large_multilevel_library_prunes_surplus_and_marks_all_retained_cases(
    monkeypatch,
):
    shelves = []
    for level_index, elevation in enumerate((0.0, 4.0, 8.0)):
        level_count = 18 if elevation == 0.0 else 5
        for position_index in range(level_count):
            shelf = _full_height_bookshelf(
                level_index * 20 + position_index,
                elevation,
                x=-5.5 + (position_index % 6) * 2.0,
                y=5.5 - (position_index // 6) * 5.5,
                yaw=180.0,
            )
            if position_index < 3:
                shelf.metadata["dense_library_grouped_run"] = elevation
            shelves.append(shelf)
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
        room_geometry=SimpleNamespace(length=13.8, width=13.8, openings=[]),
    )
    window_blockers = {
        shelf.object_id for shelf in shelves if shelf.transform.translation()[2] == 0.0
    }
    window_blockers = set(sorted(window_blockers)[3:9])
    monkeypatch.setattr(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "compute_window_clearance_violations",
        lambda _scene: [
            SimpleNamespace(furniture_id=object_id)
            for object_id in sorted(window_blockers)
        ],
    )
    monkeypatch.setattr(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "compute_door_clearance_violations",
        lambda _scene: [],
    )
    removed = []

    assert (
        _normalize_dense_library_bookcases(
            scene,
            _dense_multilevel_bookshelf_kit(),
            (0.0, 4.0, 8.0),
            remove_object=lambda object_id: (
                removed.append(object_id),
                scene.objects.pop(object_id),
            ),
        )
        == 13
    )
    retained = list(scene.objects.values())
    assert {
        elevation: sum(
            float(shelf.transform.translation()[2]) == elevation for shelf in retained
        )
        for elevation in (0.0, 4.0, 8.0)
    } == {0.0: 5, 4.0: 5, 8.0: 5}
    assert window_blockers <= set(removed)
    assert all(
        shelf.metadata.get("dense_library_populated_case")
        == float(shelf.transform.translation()[2])
        for shelf in retained
    )
    for elevation in (0.0, 4.0, 8.0):
        assert (
            sum(
                shelf.metadata.get("dense_library_grouped_run") == elevation
                for shelf in retained
            )
            == 3
        )


def test_large_multilevel_library_prunes_table_and_chair_surplus(monkeypatch):
    shelves = [
        _full_height_bookshelf(level * 5 + index, float(level * 4))
        for level in range(3)
        for index in range(5)
    ]
    tables = [
        _role_furniture("reading_table", level * 3 + index, float(level * 4))
        for level in range(3)
        for index in range(3)
    ]
    chairs = [
        _role_furniture("reading_chair", level * 6 + index, float(level * 4))
        for level in range(3)
        for index in range(6)
    ]
    objects = shelves + tables + chairs
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in objects},
        room_geometry=SimpleNamespace(length=13.8, width=13.8, openings=[]),
    )
    monkeypatch.setattr(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "compute_window_clearance_violations",
        lambda _scene: [],
    )
    monkeypatch.setattr(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "compute_door_clearance_violations",
        lambda _scene: [],
    )
    room_kit = _dense_multilevel_library_kit()
    room_kit.slots[-1].minimum_count = 13

    removed = _normalize_dense_library_bookcases(
        scene,
        room_kit,
        (0.0, 4.0, 8.0),
        remove_object=lambda object_id: scene.objects.pop(object_id),
    )

    assert removed == 9
    assert sum("reading_table" in key for key in scene.objects) == 5
    assert sum("reading_chair" in key for key in scene.objects) == 13
    assert {
        elevation: sum(
            obj.name == "reading_table"
            and float(obj.transform.translation()[2]) == elevation
            for obj in scene.objects.values()
        )
        for elevation in (0.0, 4.0, 8.0)
    } == {0.0: 2, 4.0: 2, 8.0: 1}
    assert {
        elevation: sum(
            obj.name == "reading_chair"
            and float(obj.transform.translation()[2]) == elevation
            for obj in scene.objects.values()
        )
        for elevation in (0.0, 4.0, 8.0)
    } == {0.0: 5, 4.0: 4, 8.0: 4}


def test_two_level_library_preserves_aggregate_bookshelf_target():
    shelves = [
        *(
            _full_height_bookshelf(
                index,
                0.0,
                x=-3.675 + (index % 8) * 1.05,
                y=5.5 - (index // 8) * 1.0,
                yaw=180.0,
            )
            for index in range(12)
        ),
        *(
            _full_height_bookshelf(
                20 + index,
                6.0,
                x=-3.675 + index * 1.05,
                y=5.5,
                yaw=180.0,
            )
            for index in range(8)
        ),
    ]
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
        room_geometry=SimpleNamespace(length=13.8, width=13.8, openings=[]),
    )
    removed = []

    assert (
        _normalize_dense_library_bookcases(
            scene,
            _dense_multilevel_bookshelf_kit(),
            (0.0, 6.0),
            remove_object=lambda object_id: (
                removed.append(object_id),
                scene.objects.pop(object_id),
            ),
        )
        == 5
    )
    retained_counts = {
        elevation: sum(
            float(shelf.transform.translation()[2]) == elevation
            for shelf in scene.objects.values()
        )
        for elevation in (0.0, 6.0)
    }
    assert retained_counts == {0.0: 8, 6.0: 7}
    assert len(removed) == 5
    assert (
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 6.0),
            enforce_exact_level_counts=True,
        )
        == 15
    )


def test_two_level_library_recovery_fills_exact_targets_before_pruning():
    isolated_ground_poses = (
        (-5.8, -4.5, 90.0),
        (-5.8, -1.5, 90.0),
        (-5.8, 1.5, 90.0),
        (-5.8, 4.5, 90.0),
        (5.8, -4.5, -90.0),
        (5.8, -1.5, -90.0),
        (5.8, 1.5, -90.0),
        (5.8, 4.5, -90.0),
        (-4.5, -5.8, 0.0),
        (-1.5, -5.8, 0.0),
        (1.5, -5.8, 0.0),
        (4.5, -5.8, 0.0),
        (-3.0, 5.8, 180.0),
        (0.0, 5.8, 180.0),
        (3.0, 5.8, 180.0),
    )
    shelves = [
        _full_height_bookshelf(index, 0.0, x=x, y=y, yaw=yaw)
        for index, (x, y, yaw) in enumerate(isolated_ground_poses)
    ]
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
        room_geometry=SimpleNamespace(length=13.8, width=13.8, openings=[]),
    )
    placements = []

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 6.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            placements.append(kwargs)
            object_id = f"recovered_two_level_case_{len(placements)}"
            recovered = _full_height_bookshelf(
                100 + len(placements),
                kwargs["z"],
                x=kwargs["x"],
                y=kwargs["y"],
                yaw=kwargs["yaw"],
            )
            recovered.object_id = object_id
            scene.objects[object_id] = recovered
            return json.dumps({"success": True, "object_id": object_id})

        def _remove_furniture_impl(self, object_id):
            scene.objects.pop(object_id, None)
            return json.dumps({"success": True})

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = scene
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [shelves[0]])
    agent.furniture_tools = FakeTools()

    prepruned, recovered = agent._preprune_and_recover_room_kit(
        _dense_multilevel_bookshelf_kit()
    )

    assert prepruned == 7
    assert recovered == 10
    assert [call["z"] for call in placements] == [0.0] * 3 + [6.0] * 7
    assert (
        _normalize_dense_library_bookcases(
            scene,
            _dense_multilevel_bookshelf_kit(),
            (0.0, 6.0),
            remove_object=agent.furniture_tools._remove_furniture_impl,
        )
        == 3
    )
    assert (
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 6.0),
            enforce_exact_level_counts=True,
        )
        == 15
    )


def test_library_recovery_preprunes_overcrowded_story_before_wall_run(monkeypatch):
    ground_poses = (
        (-5.8, -4.5, 90.0),
        (-5.8, -1.5, 90.0),
        (-5.8, 1.5, 90.0),
        (-5.8, 4.5, 90.0),
        (5.8, -4.5, -90.0),
        (5.8, -1.5, -90.0),
        (5.8, 1.5, -90.0),
        (5.8, 4.5, -90.0),
        (-4.5, -5.8, 0.0),
        (-1.5, -5.8, 0.0),
        (1.5, -5.8, 0.0),
        (4.5, -5.8, 0.0),
        (-3.0, 5.8, 180.0),
        (0.0, 5.8, 180.0),
        (3.0, 5.8, 180.0),
    )
    shelves = [
        *(
            _full_height_bookshelf(index, 0.0, x=x, y=y, yaw=yaw)
            for index, (x, y, yaw) in enumerate(ground_poses)
        ),
        *(
            _full_height_bookshelf(
                level_index * 20 + index,
                elevation,
                x=-4.8 + index * 2.4,
                y=5.5,
                yaw=180.0,
            )
            for level_index, elevation in ((1, 4.0), (2, 8.0))
            for index in range(5)
        ),
    ]
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
        room_geometry=SimpleNamespace(length=13.8, width=13.8, openings=[]),
    )
    window_blockers = {
        "renaissance_bookshelf_5",
        "renaissance_bookshelf_6",
        "renaissance_bookshelf_7",
        "renaissance_bookshelf_8",
        "renaissance_bookshelf_10",
        "renaissance_bookshelf_11",
    }
    monkeypatch.setattr(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "compute_window_clearance_violations",
        lambda _scene: [
            SimpleNamespace(furniture_id=object_id)
            for object_id in sorted(window_blockers)
        ],
    )
    monkeypatch.setattr(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "compute_door_clearance_violations",
        lambda _scene: [SimpleNamespace(furniture_id="renaissance_bookshelf_13")],
    )
    placements = []
    removed = []

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 4.0, 8.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            placements.append(kwargs)
            object_id = f"recovered_wall_case_{len(placements)}"
            recovered = _full_height_bookshelf(
                100 + len(placements),
                kwargs["z"],
                x=kwargs["x"],
                y=kwargs["y"],
                yaw=kwargs["yaw"],
            )
            recovered.object_id = object_id
            scene.objects[object_id] = recovered
            return json.dumps({"success": True, "object_id": object_id})

        def _remove_furniture_impl(self, object_id):
            removed.append(object_id)
            scene.objects.pop(object_id, None)
            return json.dumps({"success": True})

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = scene
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [shelves[0]])
    agent.furniture_tools = FakeTools()

    prepruned, recovered = agent._preprune_and_recover_room_kit(
        _dense_multilevel_bookshelf_kit()
    )

    assert prepruned == 10
    assert recovered == 9
    assert window_blockers <= set(removed)
    assert [call["z"] for call in placements] == [0.0] * 3 + [4.0] * 3 + [8.0] * 3
    assert (
        _normalize_dense_library_bookcases(
            scene,
            _dense_multilevel_bookshelf_kit(),
            (0.0, 4.0, 8.0),
            remove_object=agent.furniture_tools._remove_furniture_impl,
        )
        == 9
    )
    assert (
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 4.0, 8.0),
            enforce_exact_level_counts=True,
        )
        == 15
    )
