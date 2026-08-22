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


def test_large_multilevel_library_recovery_builds_atomic_bookcase_wall_runs():
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
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
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

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_bookshelf_kit()
        )
        == 9
    )
    assert removed == []
    assert [call["z"] for call in placements] == [0.0] * 3 + [4.0] * 3 + [8.0] * 3
    assert (
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_bookshelf_kit(),
            support_elevations=(0.0, 4.0, 8.0),
        )
        == 24
    )


def test_large_multilevel_library_recovery_rolls_back_incomplete_bookcase_run():
    shelves = [
        *(
            _full_height_bookshelf(
                index,
                0.0,
                x=-4.8 + index * 2.4,
                y=5.5,
                yaw=180.0,
            )
            for index in range(5)
        ),
        *(
            _full_height_bookshelf(
                level_index * 5 + index,
                elevation,
                x=-2.1 + index * 1.05,
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
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )
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
                object_id = f"partial_wall_case_{len(attempts)}"
                recovered = _full_height_bookshelf(
                    100 + len(attempts),
                    kwargs["z"],
                    x=kwargs["x"],
                    y=kwargs["y"],
                    yaw=kwargs["yaw"],
                )
                recovered.object_id = object_id
                scene.objects[object_id] = recovered
                return json.dumps({"success": True, "object_id": object_id})
            return json.dumps({"success": False})

        def _remove_furniture_impl(self, object_id):
            removed.append(object_id)
            scene.objects.pop(object_id, None)
            return json.dumps({"success": True})

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = scene
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [shelves[0]])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_bookshelf_kit()
        )
        == 0
    )
    assert removed == ["partial_wall_case_2", "partial_wall_case_1"]
    assert set(scene.objects) == {shelf.object_id for shelf in shelves}


def test_large_multilevel_library_recovery_fills_sparse_upper_bookshelves():
    shelves = [
        *(_full_height_bookshelf(index, 0.0) for index in range(16)),
        *(
            _full_height_bookshelf(index + 16, 4.0, x=-1.05 + index * 1.05)
            for index in range(3)
        ),
        *(
            _full_height_bookshelf(index + 19, 8.0, x=-1.05 + index * 1.05)
            for index in range(3)
        ),
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
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [shelves[0]])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_bookshelf_kit()
        )
        == 4
    )
    assert [call["z"] for call in placements] == [4.0, 4.0, 8.0, 8.0]


def test_large_multilevel_library_recovery_fills_bookshelves_on_every_story():
    shelves = [_full_height_bookshelf(index, 0.0) for index in range(16)]
    placements = []

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 4.0, 8.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            placements.append(kwargs)
            object_id = f"recovered_bookshelf_{len(placements)}"
            recovered = _full_height_bookshelf(
                100 + len(placements),
                kwargs["z"],
                x=kwargs["x"],
                y=kwargs["y"],
                yaw=kwargs["yaw"],
            )
            recovered.object_id = object_id
            agent.scene.objects[object_id] = recovered
            return json.dumps({"success": True, "object_id": object_id})

        def _remove_furniture_impl(self, object_id):
            agent.scene.objects.pop(object_id, None)
            return json.dumps({"success": True})

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={shelf.object_id: shelf for shelf in shelves},
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [shelves[0]])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_bookshelf_kit()
        )
        == 10
    )
    assert [call["z"] for call in placements].count(4.0) == 5
    assert [call["z"] for call in placements].count(8.0) == 5
    assert not any(call["z"] == 0.0 for call in placements)


def test_large_multilevel_library_gate_rejects_ground_only_reading_tables():
    furniture = _library_with_ground_only_tables()
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in furniture},
    )

    with pytest.raises(ModelBehaviorError, match=r"reading_table.*4\.000m.*0.*1"):
        _validate_room_kit_completion(
            scene,
            _dense_multilevel_library_kit(),
            support_elevations=(0.0, 4.0, 8.0),
        )


def test_large_multilevel_library_recovery_fills_patron_ensemble_on_each_story():
    furniture = _library_with_ground_only_tables()
    table_asset = next(obj for obj in furniture if obj.name == "reading_table")
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
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in furniture},
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [table_asset])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_library_kit()
        )
        == 3
    )
    assert [call["z"] for call in placements] == [4.0, 4.0, 8.0]
    assert all(call["asset_id"] == table_asset.object_id for call in placements)
    assert all(call["y"] == -1.5 for call in placements)
    assert [call["x"] for call in placements] == [1.25, 2.25, 1.25]


def test_level_recovery_retries_when_support_snaps_table_to_another_story():
    furniture = _library_with_ground_only_tables()
    table_asset = next(obj for obj in furniture if obj.name == "reading_table")
    attempts = []
    removed = []
    scene = SimpleNamespace(
        text_description=_EXACT_MULTILEVEL_LIBRARY_PROMPT,
        objects={obj.object_id: obj for obj in furniture},
        room_geometry=SimpleNamespace(length=13.8, width=13.8),
    )

    class FakeTools:
        def set_noise_profile(self, _mode):
            pass

        def _major_support_elevations(self):
            return (0.0, 4.0, 8.0)

        def _add_furniture_to_scene_impl(self, **kwargs):
            attempts.append(kwargs)
            object_id = f"recovered_table_{len(attempts)}"
            actual_z = 0.0 if len(attempts) == 1 else kwargs["z"]
            placed = _role_furniture(
                "reading_table",
                100 + len(attempts),
                actual_z,
                x=kwargs["x"],
                y=kwargs["y"],
                yaw=kwargs["yaw"],
            )
            placed.object_id = object_id
            scene.objects[object_id] = placed
            return json.dumps({"success": True, "object_id": object_id})

        def _remove_furniture_impl(self, object_id):
            removed.append(object_id)
            scene.objects.pop(object_id, None)
            return json.dumps({"success": True})

    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = scene
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [table_asset])
    agent.furniture_tools = FakeTools()

    assert (
        agent._place_room_kit_minimums_deterministically(
            _dense_multilevel_library_kit()
        )
        == 3
    )
    assert [call["z"] for call in attempts] == [4.0, 4.0, 4.0, 8.0]
    assert removed == ["recovered_table_1"]
    assert scene.objects["recovered_table_2"].transform.translation()[2] == 4.0
    assert scene.objects["recovered_table_3"].transform.translation()[2] == 4.0
    assert scene.objects["recovered_table_4"].transform.translation()[2] == 8.0


def test_library_recovery_prefers_stable_armchair_over_role_exact_rocker():
    assets = [
        SimpleNamespace(
            object_id="reading_chair_0",
            object_type=ObjectType.FURNITURE,
            name="reading_chair",
            description="burgundy leather rocking chair",
            metadata={
                "asset_quality_score": 0.76,
                "catalog_semantics": "Chesterfield rocking chair rocking_chair.n.01",
            },
        ),
        SimpleNamespace(
            object_id="library_reading_chair_0",
            object_type=ObjectType.FURNITURE,
            name="library_reading_chair",
            description="stationary upholstered armchair",
            metadata={
                "asset_quality_score": 1.0,
                "catalog_semantics": "Wooden armchair Furniture/Seating/Chairs",
            },
        ),
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

    room_kit = SimpleNamespace(
        slots=(
            SimpleNamespace(
                role="reading_chair",
                aliases=("chair", "seat"),
                query="stationary upholstered library reading chair",
                required=True,
                minimum_count=1,
                placement_class="floor",
            ),
        )
    )
    agent = object.__new__(StatefulFurnitureAgent)
    agent.scene = SimpleNamespace(
        objects={"wall": SimpleNamespace(object_type=ObjectType.WALL)},
        room_geometry=SimpleNamespace(length=7.0, width=7.0),
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: assets)
    agent.furniture_tools = FakeTools()

    assert agent._place_room_kit_minimums_deterministically(room_kit) == 1
    assert placements[0]["asset_id"] == "library_reading_chair_0"


def test_library_recovery_prefers_filled_bookcase_over_bare_shelf():
    slot = SimpleNamespace(
        role="bookshelf",
        aliases=("bookcase",),
        query=(
            "full-height Renaissance library bookcase densely filled with "
            "visible books"
        ),
    )
    bare_shelf = SimpleNamespace(
        object_id="library_bookshelf_0",
        name="library_bookshelf",
        description="short empty wooden shelf",
        metadata={
            "asset_quality_score": 1.0,
            "catalog_semantics": "wooden bookshelf furniture storage",
        },
    )
    filled_bookcase = SimpleNamespace(
        object_id="renaissance_bookcase_0",
        name="renaissance_bookcase",
        description=(
            "full-height Renaissance library bookcase densely filled with "
            "visible books"
        ),
        metadata={
            "asset_quality_score": 0.76,
            "catalog_semantics": (
                "ornate Renaissance bookcase with shelves full of visible books"
            ),
        },
    )

    ranked = max(
        (bare_shelf, filled_bookcase),
        key=lambda asset: StatefulFurnitureAgent._slot_relevance(asset, slot),
    )

    assert ranked.object_id == "renaissance_bookcase_0"


def test_library_recovery_prefers_rich_bookcase_over_exact_generic_role():
    slot = SimpleNamespace(
        role="bookshelf",
        aliases=("bookcase",),
        query=(
            "full-height Renaissance library bookcase densely filled with visible books"
        ),
        nominal_dimensions_m=(1.0, 0.35, 2.0),
    )
    exact_generic = SimpleNamespace(
        object_id="bookshelf_0",
        name="bookshelf",
        description="generic storage shelf",
        bbox_min=(-0.5, -0.175, 0.0),
        bbox_max=(0.5, 0.175, 1.8),
        metadata={
            "asset_quality_score": 1.0,
            "catalog_semantics": "generic wooden bookshelf furniture storage",
        },
    )
    rich_bookcase = SimpleNamespace(
        object_id="renaissance_bookcase_0",
        name="renaissance_bookcase",
        description=(
            "full-height Renaissance bookcase densely filled with visible books"
        ),
        bbox_min=(-0.5, -0.175, 0.0),
        bbox_max=(0.5, 0.175, 1.8),
        metadata={
            "asset_quality_score": 0.76,
            "catalog_semantics": (
                "ornate Renaissance bookcase populated with visible leather books"
            ),
        },
    )

    ranked = max(
        (exact_generic, rich_bookcase),
        key=lambda asset: StatefulFurnitureAgent._slot_relevance(asset, slot),
    )

    assert ranked.object_id == "renaissance_bookcase_0"


def test_exact_hssd_bookcase_is_intrinsically_fillable_without_support_zones():
    slot = SimpleNamespace(
        role="bookshelf",
        aliases=("bookcase",),
        query=(
            "full-height Renaissance library bookcase densely filled with visible books"
        ),
        nominal_dimensions_m=(1.0, 0.35, 2.0),
    )
    hssd_bookcase = SimpleNamespace(
        object_id="tall_renaissance_bookshelf_0",
        name="tall_renaissance_bookshelf",
        description=(
            "Tall ornate Renaissance library bookcase with full-height shelving "
            "densely packed with books"
        ),
        bbox_min=(-0.48, -0.18, 0.0),
        bbox_max=(0.48, 0.18, 1.8),
        metadata={
            "asset_quality_score": 0.76,
            "catalog_semantics": (
                "Revolve Lexington Tall Bookcase Walnut " "hssd/wordnet/bookcase.n.01"
            ),
        },
    )

    assert StatefulFurnitureAgent._slot_relevance(hssd_bookcase, slot)[0] > 0
