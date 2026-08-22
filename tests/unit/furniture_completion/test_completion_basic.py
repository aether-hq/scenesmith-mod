"""Regression tests for semantic furniture-stage completion gates."""

import json

from types import SimpleNamespace

import pytest

from agents.exceptions import ModelBehaviorError
from pydrake.all import RigidTransform

from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType
from scenesmith.furniture_agents.room_kit.validation import (
    _validate_room_kit_completion,
)
from scenesmith.furniture_agents.stateful_furniture_agent import (
    StatefulFurnitureAgent,
    _validate_furniture_collision_free,
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
        match=(
            r"library-reading-hall-v1.*bookshelf placed 0, required 2.*"
            r"reading_table placed 0, required 1.*reading_chair placed 0, required 4"
        ),
    ):
        _validate_room_kit_completion(_scene_with_furniture(0), _library_kit())


def test_furniture_workflow_batch_gate_rejects_deep_snap_collision(monkeypatch):
    collision = SimpleNamespace(
        object_a_id="reading_table_1",
        object_b_id="reading_chair_2",
        penetration_depth=0.1323,
        to_description=lambda: (
            "reading_table_1 collides with reading_chair_2 " "(13.2cm penetration)"
        ),
    )
    monkeypatch.setattr(
        "scenesmith.furniture_agents.stateful_furniture_agent."
        "compute_scene_collisions",
        lambda **_kwargs: [collision],
    )
    scene = SimpleNamespace(
        objects={
            "reading_table_1": SimpleNamespace(
                object_id="reading_table_1",
                object_type=ObjectType.FURNITURE,
            ),
            "reading_chair_2": SimpleNamespace(
                object_id="reading_chair_2",
                object_type=ObjectType.FURNITURE,
            ),
        }
    )
    cfg = SimpleNamespace(
        object_penetration_threshold_m=0.001,
        floor_penetration_tolerance_m=0.05,
        manipuland_furniture_tolerance_m=0.02,
    )

    with pytest.raises(ModelBehaviorError, match=r"13.2cm penetration"):
        _validate_furniture_collision_free(scene, cfg)


def test_matched_library_kit_accepts_required_minimum():
    scene = _scene_with_role_counts(
        bookshelf=2,
        reading_table=1,
        reading_chair=4,
    )

    assert _validate_room_kit_completion(scene, _library_kit()) == 7


def test_matched_library_kit_rejects_statue_deficit_against_expanded_count():
    room_kit = _library_kit()
    room_kit.slots = (
        *room_kit.slots,
        SimpleNamespace(
            role="classical_statue",
            aliases=("statue", "sculpture"),
            required=True,
            minimum_count=2,
            placement_class="floor",
            facing_target="room_center",
        ),
    )
    room_kit.slot_counts = {
        "bookshelf": 2,
        "reading_table": 1,
        "reading_chair": 4,
        "task_lamp": 1,
        "classical_statue": 3,
    }
    scene = _scene_with_role_counts(
        bookshelf=2,
        reading_table=1,
        reading_chair=4,
        classical_statue=2,
        task_lamp=1,
    )

    with pytest.raises(ModelBehaviorError, match=r"classical_statue.*2.*3"):
        _validate_room_kit_completion(scene, room_kit)


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


def test_library_gate_rejects_renamed_short_book_props_as_bookshelves():
    kit = _library_kit()
    kit.slots[0].query = (
        "full-height Renaissance library bookcase densely filled with visible books"
    )
    kit.slots[0].nominal_dimensions_m = (1.0, 0.35, 2.0)
    objects = {
        "wall": SimpleNamespace(object_type=ObjectType.WALL),
        **{
            f"book_prop_{index}": SimpleNamespace(
                object_type=ObjectType.FURNITURE,
                object_id=f"full_height_bookshelf_{index}",
                name="full_height_bookshelf",
                description=kit.slots[0].query,
                bbox_min=(-0.5, -0.148, 0.0),
                bbox_max=(0.5, 0.148, 0.431),
                metadata={
                    "asset_quality_score": 1.0,
                    "catalog_semantics": (
                        "Book Encyclopedia Set 01 books bookshelf "
                        "polyhaven/Office & Stationery/Books & Documents/Books"
                    ),
                },
            )
            for index in range(2)
        },
        "reading_table_0": SimpleNamespace(
            object_type=ObjectType.FURNITURE,
            name="reading_table",
            description="reading table",
        ),
        **{
            f"reading_chair_{index}": SimpleNamespace(
                object_type=ObjectType.FURNITURE,
                name="reading_chair",
                description="reading chair",
            )
            for index in range(4)
        },
    }

    with pytest.raises(ModelBehaviorError, match=r"bookshelf.*0.*2"):
        _validate_room_kit_completion(SimpleNamespace(objects=objects), kit)


def test_library_gate_rejects_bare_shelf_that_barely_passes_generic_height():
    kit = _library_kit()
    kit.slots[0].query = (
        "full-height Renaissance library bookcase densely filled with visible books"
    )
    kit.slots[0].nominal_dimensions_m = (1.0, 0.35, 2.0)
    scene = _scene_with_role_counts(
        reading_table=1,
        reading_chair=4,
    )
    for index in range(2):
        scene.objects[f"bare_shelf_{index}"] = SimpleNamespace(
            object_type=ObjectType.FURNITURE,
            object_id=f"library_bookshelf_{index}",
            name="library_bookshelf",
            description="Full-height library bookshelf in dark wood",
            bbox_min=(-0.414, -0.175, 0.0),
            bbox_max=(0.414, 0.175, 1.242),
            metadata={
                "asset_quality_score": 1.0,
                "catalog_semantics": (
                    "Wooden Bookshelf Worn antique bookcase rustic shelves storage "
                    "polyhaven/Furniture/Storage Furniture/Shelving & Bookcases"
                ),
            },
        )

    with pytest.raises(ModelBehaviorError, match=r"bookshelf.*0.*2"):
        _validate_room_kit_completion(scene, kit)


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


def test_library_recovery_fills_expanded_statue_count():
    statue_slot = SimpleNamespace(
        role="classical_statue",
        aliases=("statue", "sculpture"),
        query="classical Renaissance marble statue on pedestal",
        required=True,
        minimum_count=2,
        placement_class="floor",
        facing_target="room_center",
    )
    room_kit = SimpleNamespace(
        kit_id="library-reading-hall-v1",
        slot_counts={"classical_statue": 3},
        slots=(statue_slot,),
    )
    statue_asset = SimpleNamespace(
        object_id="classical_statue_asset",
        object_type=ObjectType.FURNITURE,
        name="classical_statue",
        description="classical Renaissance marble statue on pedestal",
        metadata={"asset_quality_score": 1.0},
    )
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
        objects={
            f"classical_statue_{index}": SimpleNamespace(
                object_id=f"classical_statue_{index}",
                object_type=ObjectType.FURNITURE,
                name="classical_statue",
                description="classical Renaissance marble statue on pedestal",
                metadata={"asset_quality_score": 1.0},
                transform=RigidTransform([0.0, 0.0, elevation]),
            )
            for index, elevation in enumerate((0.0, 4.0))
        },
        room_geometry=SimpleNamespace(length=13.0, width=13.0),
        text_description="large multi-level Renaissance library with statues",
    )
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: [statue_asset])
    agent.furniture_tools = FakeTools()

    placed = agent._place_room_kit_minimums_deterministically(room_kit)

    assert placed == 1
    assert len(placements) == 1
    assert placements[0]["asset_id"] == "classical_statue_asset"


def test_library_recovery_acquires_missing_required_statue_once():
    statue_slot = SimpleNamespace(
        role="classical_statue",
        aliases=("statue", "sculpture"),
        query="classical Renaissance marble statue on pedestal",
        nominal_dimensions_m=(0.7, 0.7, 2.0),
        required=True,
        minimum_count=2,
        placement_class="floor",
        facing_target="room_center",
    )
    room_kit = SimpleNamespace(
        kit_id="library-reading-hall-v1",
        slot_counts={"classical_statue": 3},
        slots=(statue_slot,),
    )
    acquired_assets = []
    acquisition_requests = []

    class FakeAssetManager:
        def list_available_assets(self):
            return list(acquired_assets)

        def generate_assets(self, request):
            acquisition_requests.append(request)
            asset = SimpleNamespace(
                object_id="classical_statue_asset",
                object_type=ObjectType.FURNITURE,
                name="classical_statue",
                description="classical Renaissance marble statue on pedestal",
                metadata={
                    "asset_quality_score": 1.0,
                    "catalog_semantics": (
                        "Gothic Statue polyhaven/Decor & Art/Sculptures & "
                        "Figurines/Busts & Human Figures"
                    ),
                },
            )
            acquired_assets.append(asset)
            return SimpleNamespace(successful_assets=[asset], failed_assets=[])

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
        objects={
            f"classical_statue_{index}": SimpleNamespace(
                object_id=f"classical_statue_{index}",
                object_type=ObjectType.FURNITURE,
                name="classical_statue",
                description="classical Renaissance marble statue on pedestal",
                metadata={"asset_quality_score": 1.0},
                transform=RigidTransform([0.0, 0.0, elevation]),
            )
            for index, elevation in enumerate((0.0, 4.0))
        },
        room_geometry=SimpleNamespace(length=13.0, width=13.0),
        text_description="large multi-level Renaissance library with statues",
        scene_dir=SimpleNamespace(name="room_room"),
    )
    agent.asset_manager = FakeAssetManager()
    agent.furniture_tools = FakeTools()

    placed = agent._place_room_kit_minimums_deterministically(room_kit)

    assert placed == 1
    assert len(acquisition_requests) == 1
    assert acquisition_requests[0].short_names == ["classical_statue"]
    assert len(placements) == 1
