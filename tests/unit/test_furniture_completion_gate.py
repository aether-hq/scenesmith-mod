"""Regression tests for semantic furniture-stage completion gates."""

import json
import math
from types import SimpleNamespace

import pytest
from agents.exceptions import ModelBehaviorError
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType
from scenesmith.furniture_agents.stateful_furniture_agent import (
    StatefulFurnitureAgent,
    _chair_cluster_poses,
    _validate_furniture_collision_free,
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


_EXACT_MULTILEVEL_LIBRARY_PROMPT = (
    "a large, multi-level library with thousands of books and a bunch of "
    "tables and chairs for patrons"
)


def _full_height_bookshelf(index: int, elevation: float):
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
        transform=SimpleNamespace(translation=lambda: (0.0, 0.0, elevation)),
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
        == 6
    )
    assert [call["z"] for call in placements] == [4.0] * 3 + [8.0] * 3
    for call in placements:
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
        == 2
    )
    assert [call["z"] for call in placements] == [0.0, 0.0]
    for call in placements:
        distance = math.dist((call["x"], call["y"]), (5.25, -1.5))
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
        == 0
    )
    assert removed == ["recovered_2", "recovered_1"]


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


def test_large_multilevel_library_recovery_fills_sparse_upper_bookshelves():
    shelves = [
        *(_full_height_bookshelf(index, 0.0) for index in range(16)),
        *(_full_height_bookshelf(index + 16, 4.0) for index in range(3)),
        *(_full_height_bookshelf(index + 19, 8.0) for index in range(3)),
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
        == 2
    )
    assert [call["z"] for call in placements] == [4.0, 8.0]
    assert all(call["asset_id"] == table_asset.object_id for call in placements)
    assert all((call["x"], call["y"]) == (1.25, -1.5) for call in placements)


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
        == 2
    )
    assert [call["z"] for call in attempts] == [4.0, 4.0, 8.0]
    assert removed == ["recovered_table_1"]
    assert scene.objects["recovered_table_2"].transform.translation()[2] == 4.0
    assert scene.objects["recovered_table_3"].transform.translation()[2] == 8.0


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
