import json
import math

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.exceptions import ModelBehaviorError
from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.agent_utils.physics_validation import CollisionPair
from scenesmith.experiments.indoor_scene_generation import (
    _validate_final_dense_library_book_rows,
)
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools


def _surface(surface_id: str, transform: RigidTransform) -> SupportSurface:
    return SupportSurface(
        surface_id=UniqueID(surface_id),
        bounding_box_min=np.array([-0.5, -0.8, 0.0]),
        bounding_box_max=np.array([0.5, 0.8, 0.0]),
        transform=transform,
    )


def test_identical_furniture_transfers_surface_relative_arrangement(tmp_path: Path):
    source_surface = _surface("source_surface", RigidTransform(p=[1.0, 2.0, 0.6]))
    target_surface = _surface(
        "target_surface",
        RigidTransform(RollPitchYaw(0.0, 0.0, 1.2), [4.0, -1.0, 0.6]),
    )
    geometry = tmp_path / "bed.glb"
    source = SceneObject(
        object_id=UniqueID("bed_0"),
        object_type=ObjectType.FURNITURE,
        name="bed",
        description="treatment bed",
        transform=RigidTransform(),
        geometry_path=geometry,
        support_surfaces=[source_surface],
    )
    target = SceneObject(
        object_id=UniqueID("bed_1"),
        object_type=ObjectType.FURNITURE,
        name="bed",
        description="treatment bed",
        transform=RigidTransform(),
        geometry_path=geometry,
        support_surfaces=[target_surface],
    )
    relative = RigidTransform(RollPitchYaw(0.0, 0.0, 0.3), [0.2, -0.1, 0.04])
    pillow = SceneObject(
        object_id=UniqueID("pillow_0"),
        object_type=ObjectType.MANIPULAND,
        name="pillow",
        description="medical pillow",
        transform=source_surface.transform @ relative,
        geometry_path=tmp_path / "pillow.glb",
        placement_info=PlacementInfo(
            parent_surface_id=source_surface.surface_id,
            position_2d=np.array([0.2, -0.1]),
            rotation_2d=0.3,
        ),
        bbox_min=np.array([-0.2, -0.1, 0.0]),
        bbox_max=np.array([0.2, 0.1, 0.08]),
    )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        objects={
            source.object_id: source,
            target.object_id: target,
            pillow.object_id: pillow,
        },
    )
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = scene

    count = agent._clone_manipulands_between_identical_furniture(
        source.object_id, target.object_id
    )

    assert count == 1
    clone = next(obj for obj in scene.objects.values() if obj.object_id == "pillow_1")
    expected = target_surface.transform @ relative
    assert np.allclose(clone.transform.GetAsMatrix4(), expected.GetAsMatrix4())
    assert clone.placement_info.parent_surface_id == target_surface.surface_id
    assert clone.placement_info.placement_method == "template_transfer"
    assert np.allclose(clone.placement_info.position_2d, [0.2, -0.1])


def test_furniture_template_key_requires_same_geometry_and_scale(tmp_path: Path):
    furniture = SceneObject(
        object_id=UniqueID("bed_0"),
        object_type=ObjectType.FURNITURE,
        name="bed",
        description="bed",
        transform=RigidTransform(),
        geometry_path=tmp_path / "bed.glb",
        scale_factor=1.25,
    )

    assert StatefulManipulandAgent._furniture_template_key(furniture) == (
        str((tmp_path / "bed.glb").resolve()),
        1.25,
    )


def test_initial_design_preloads_deterministic_reads():
    class FakeTools:
        def _get_current_scene_state_impl(self):
            return '{"surface":"bed"}'

        def _list_available_assets_impl(self):
            return '{"assets":["pillow_0"]}'

    agent = object.__new__(StatefulManipulandAgent)
    agent.manipuland_tools = FakeTools()
    agent.manipuland_context_image_path = None

    result = agent._build_initial_design_input("Populate the treatment bed.")

    assert isinstance(result, str)
    assert "PRELOADED_CURRENT_SCENE_STATE" in result
    assert '{"surface":"bed"}' in result
    assert "PRELOADED_AVAILABLE_ASSETS" in result
    assert '{"assets":["pillow_0"]}' in result
    assert "first response must call" in result


def test_deterministic_fallback_places_semantic_cached_assets(tmp_path: Path):
    surface = _surface("bed_surface", RigidTransform(p=[0.0, 0.0, 0.6]))
    assets = [
        SceneObject(
            object_id=UniqueID(object_id),
            object_type=ObjectType.MANIPULAND,
            name=name,
            description=description,
            transform=RigidTransform(),
            geometry_path=tmp_path / f"{name}.glb",
            metadata={"asset_quality_score": quality},
        )
        for object_id, name, description, quality in [
            ("pillow_0", "pillow", "white medical pillow", 0.8),
            ("blanket_0", "folded_blanket", "folded medical blanket", 1.0),
            ("monitor_0", "monitor_accessory", "compact patient monitor", 0.7),
            ("mug_0", "mug", "ceramic coffee mug", 1.0),
        ]
    ]
    placements = []

    class FakeTools:
        support_surfaces = {str(surface.surface_id): surface}

        def _place_manipuland_on_surface_impl(self, **kwargs):
            placements.append(kwargs)
            return '{"success":true}'

    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_selection = SimpleNamespace(
        suggested_items="pillow, folded blanket, compact patient monitor"
    )
    agent.current_furniture_id = UniqueID("bed_0")
    agent.manipuland_tools = FakeTools()
    agent.asset_manager = SimpleNamespace(list_available_assets=lambda: assets)

    placed = agent._place_cached_assets_deterministically()

    assert placed == 3
    assert {call["asset_id"] for call in placements} == {
        "pillow_0",
        "blanket_0",
        "monitor_0",
    }
    assert len({(call["position_x"], call["position_z"]) for call in placements}) == 3


def _bookcase_surface(surface_id: str, elevation: float) -> SupportSurface:
    return SupportSurface(
        surface_id=UniqueID(surface_id),
        bounding_box_min=np.array([-0.32, -0.15, 0.0]),
        bounding_box_max=np.array([0.32, 0.15, 0.35]),
        transform=RigidTransform(p=[0.0, 0.0, elevation]),
    )


def _dense_bookcase(object_id: str, elevation: float, tmp_path: Path):
    local_heights = (0.01, 0.30, 0.60, 0.90, 1.20, 1.50, 2.01)
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.FURNITURE,
        name="renaissance_bookcase",
        description="full-height Renaissance library bookcase",
        transform=RigidTransform(p=[0.0, 0.0, elevation]),
        geometry_path=tmp_path / "bookcase.gltf",
        bbox_min=np.array([-0.5, -0.18, 0.0]),
        bbox_max=np.array([0.5, 0.18, 2.0]),
        metadata={
            "asset_source": "hssd",
            "catalog_id": "hssd__e3631a629a1ac3b71a75dff721192b90d26246e0",
            "ontology_path": "hssd/wordnet/bookcase.n.01",
        },
        support_surfaces=[
            _bookcase_surface(f"{object_id}_surface_{index}", elevation + height)
            for index, height in enumerate(local_heights)
        ],
    )


def test_dense_library_places_intrinsic_catalog_book_rows_on_internal_tiers(
    tmp_path: Path,
):
    furniture = _dense_bookcase("bookcase_0", 4.0, tmp_path)
    row_asset = SceneObject(
        object_id=UniqueID("encyclopedia_book_row_0"),
        object_type=ObjectType.MANIPULAND,
        name="encyclopedia_book_row",
        description="upright encyclopedia book set",
        transform=RigidTransform(),
        geometry_path=tmp_path / "book_encyclopedia_set_01.gltf",
        bbox_min=np.array([-0.225, -0.06, 0.0]),
        bbox_max=np.array([0.225, 0.06, 0.22]),
        metadata={
            "asset_source": "polyhaven",
            "catalog_id": "polyhaven__book_encyclopedia_set_01",
            "ontology_path": ("polyhaven/Office & Stationery/Books & Documents/Books"),
            "catalog_semantics": "Book Encyclopedia Set 01",
        },
    )
    placements = []
    original_noise_profile = SimpleNamespace(
        position_xy_std_meters=0.01,
        rotation_yaw_std_degrees=3.0,
    )

    class FakeTools:
        support_surfaces = {
            str(surface.surface_id): surface for surface in furniture.support_surfaces
        }
        active_noise_profile = original_noise_profile

        def _place_manipuland_on_surface_impl(self, **kwargs):
            kwargs["active_noise"] = (
                self.active_noise_profile.position_xy_std_meters,
                self.active_noise_profile.rotation_yaw_std_degrees,
            )
            placements.append(kwargs)
            return '{"success":true,"object_id":"encyclopedia_book_row_1"}'

    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = furniture.object_id
    agent.current_furniture_selection = SimpleNamespace(
        suggested_items="dense rows of visible leather-bound books"
    )
    agent.manipuland_tools = FakeTools()
    agent.asset_manager = SimpleNamespace(
        list_available_assets=lambda: [row_asset],
        generate_assets=lambda _request: (_ for _ in ()).throw(
            AssertionError("intrinsic catalog row should be reused")
        ),
    )
    agent.scene = SimpleNamespace(get_object=lambda _object_id: None)

    placed = agent._place_dense_book_rows_deterministically(furniture)

    assert placed == 5
    assert all(call["asset_id"] == row_asset.object_id for call in placements)
    assert {call["surface_id"] for call in placements} == {
        f"bookcase_0_surface_{index}" for index in range(1, 6)
    }
    assert all(call["active_noise"] == (0.0, 0.0) for call in placements)
    assert agent.manipuland_tools.active_noise_profile is original_noise_profile


def test_dense_library_book_rows_retry_and_rollback_deep_owner_collisions(
    tmp_path: Path,
):
    furniture = _dense_bookcase("bookcase_0", 0.0, tmp_path)
    row_asset = SceneObject(
        object_id=UniqueID("encyclopedia_book_row_asset_0"),
        object_type=ObjectType.MANIPULAND,
        name="encyclopedia_book_row",
        description="upright encyclopedia book set",
        transform=RigidTransform(),
        geometry_path=tmp_path / "book_encyclopedia_set_01.gltf",
        bbox_min=np.array([-0.013, -0.066, 0.0]),
        bbox_max=np.array([0.350, 0.054, 0.156]),
        metadata={"catalog_id": "polyhaven__book_encyclopedia_set_01"},
    )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        objects={furniture.object_id: furniture},
    )
    placement_calls: dict[UniqueID, dict] = {}
    placement_attempts: list[dict] = []

    class FakeTools:
        support_surfaces = {
            str(surface.surface_id): surface for surface in furniture.support_surfaces
        }

        def _place_manipuland_on_surface_impl(self, **kwargs):
            object_id = scene.generate_unique_id("encyclopedia_book_row")
            placement_calls[object_id] = kwargs
            placement_attempts.append(kwargs)
            scene.add_object(
                SceneObject(
                    object_id=object_id,
                    object_type=ObjectType.MANIPULAND,
                    name="encyclopedia_book_row",
                    description="upright encyclopedia book set",
                    transform=RigidTransform(),
                    geometry_path=row_asset.geometry_path,
                    bbox_min=row_asset.bbox_min,
                    bbox_max=row_asset.bbox_max,
                    metadata=row_asset.metadata.copy(),
                    placement_info=PlacementInfo(
                        parent_surface_id=UniqueID(kwargs["surface_id"]),
                        position_2d=np.array(
                            [kwargs["position_x"], kwargs["position_z"]]
                        ),
                        rotation_2d=np.deg2rad(kwargs["rotation_degrees"]),
                    ),
                )
            )
            return json.dumps({"success": True, "object_id": str(object_id)})

    agent = object.__new__(StatefulManipulandAgent)
    agent.current_furniture_id = furniture.object_id
    agent.current_furniture_selection = SimpleNamespace(
        suggested_items="dense rows of visible leather-bound books"
    )
    agent.manipuland_tools = FakeTools()
    agent.asset_manager = SimpleNamespace(
        list_available_assets=lambda: [row_asset],
        generate_assets=lambda _request: (_ for _ in ()).throw(
            AssertionError("intrinsic catalog row should be reused")
        ),
    )
    agent.scene = scene

    def collision_free(_furniture_id, object_id):
        call = placement_calls[object_id]
        return call["rotation_degrees"] == 180.0 and call["position_x"] > 0.0

    agent._dense_book_row_pose_is_collision_free = collision_free

    placed = agent._place_dense_book_rows_deterministically(furniture)

    surviving = [
        obj
        for obj in scene.objects.values()
        if obj.object_type == ObjectType.MANIPULAND
    ]
    assert placed == 5
    assert len(surviving) == 5
    assert all(obj.metadata.get("dense_library_book_row") for obj in surviving)
    assert all(
        obj.metadata.get("dense_library_owner_bound") == str(furniture.object_id)
        for obj in surviving
    )
    assert all(
        placement_calls[obj.object_id]["rotation_degrees"] == 180.0
        and placement_calls[obj.object_id]["position_x"] > 0.0
        for obj in surviving
    )
    assert len(placement_attempts) > len(surviving)


def test_requested_book_row_description_is_not_intrinsic_catalog_evidence(
    tmp_path: Path,
):
    requested_only = SceneObject(
        object_id=UniqueID("encyclopedia_book_row_0"),
        object_type=ObjectType.MANIPULAND,
        name="encyclopedia_book_row",
        description="upright encyclopedia book set row with visible books",
        transform=RigidTransform(),
        geometry_path=tmp_path / "flat_book.gltf",
        metadata={"catalog_id": "objaverse__generic_flat_book"},
    )

    assert not StatefulManipulandAgent._is_intrinsic_catalog_book_row_asset(
        requested_only
    )


def test_exact_llm_book_rows_are_bound_or_discarded_before_dynamics(tmp_path: Path):
    bookcase = _dense_bookcase("bookcase_0", 0.0, tmp_path)
    surface = bookcase.support_surfaces[2]

    def exact_row(object_id: str, transform: RigidTransform) -> SceneObject:
        return SceneObject(
            object_id=UniqueID(object_id),
            object_type=ObjectType.MANIPULAND,
            name="encyclopedia_book_row",
            description="upright encyclopedia book set row",
            transform=transform,
            geometry_path=tmp_path / "book_encyclopedia_set_01.gltf",
            metadata={"catalog_id": "polyhaven__book_encyclopedia_set_01"},
            bbox_min=np.array([-0.1, -0.05, 0.0]),
            bbox_max=np.array([0.1, 0.05, 0.2]),
            placement_info=PlacementInfo(
                parent_surface_id=surface.surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
                placement_method="surface_placement",
            ),
        )

    contained = exact_row("encyclopedia_book_row_0", surface.transform)
    outside = exact_row(
        "encyclopedia_book_row_1",
        surface.transform @ RigidTransform(p=[0.8, 0.0, 0.0]),
    )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in (bookcase, contained, outside)},
    )
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = scene

    bound, discarded = agent._normalize_intrinsic_dense_book_rows()

    assert (bound, discarded) == (1, 1)
    assert contained.metadata["dense_library_book_row"] is True
    assert contained.metadata["dense_library_owner_bound"] == str(bookcase.object_id)
    assert contained.metadata["dense_library_owner_surface_local_transform"]
    assert outside.object_id not in scene.objects


def test_dense_library_book_row_gate_requires_each_authored_story(tmp_path: Path):
    bookcases = [
        _dense_bookcase(f"bookcase_{int(level)}", level, tmp_path)
        for level in (0.0, 4.0, 8.0)
    ]
    rows = [
        SimpleNamespace(
            object_id=UniqueID(f"book_row_{index}"),
            object_type=ObjectType.MANIPULAND,
            metadata={"dense_library_book_row": True},
            placement_info=PlacementInfo(
                parent_surface_id=bookcases[0].support_surfaces[index + 1].surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        for index in range(4)
    ]
    scene = SimpleNamespace(
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )

    with pytest.raises(ModelBehaviorError, match=r"4\.000m.*0.*12"):
        StatefulManipulandAgent._validate_dense_library_book_rows(scene)


def test_dense_library_book_row_gate_accepts_twelve_rows_per_story(tmp_path: Path):
    bookcases = [
        _dense_bookcase(f"bookcase_{level_index}_{case_index}", level, tmp_path)
        for level_index, level in enumerate((0.0, 4.0, 8.0))
        for case_index in range(3)
    ]
    rows = [
        SimpleNamespace(
            object_id=UniqueID(f"book_row_{level_index}_{row_index}"),
            object_type=ObjectType.MANIPULAND,
            metadata={"dense_library_book_row": True},
            placement_info=PlacementInfo(
                parent_surface_id=bookcases[level_index * 3 + row_index // 4]
                .support_surfaces[row_index % 4 + 1]
                .surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        for level_index in range(3)
        for row_index in range(12)
    ]
    scene = SimpleNamespace(
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )

    assert StatefulManipulandAgent._validate_dense_library_book_rows(scene) == 36


def test_dense_library_book_row_gate_rejects_empty_grouped_wall_run(
    tmp_path: Path,
):
    bookcases = [
        _dense_bookcase(f"bookcase_{level_index}_{case_index}", level, tmp_path)
        for level_index, level in enumerate((0.0, 4.0, 8.0))
        for case_index in range(4)
    ]
    for level_index, level in enumerate((0.0, 4.0, 8.0)):
        for bookcase in bookcases[level_index * 4 : level_index * 4 + 3]:
            bookcase.metadata["dense_library_grouped_run"] = level
    rows = [
        SimpleNamespace(
            object_id=UniqueID(f"book_row_{level_index}_{row_index}"),
            object_type=ObjectType.MANIPULAND,
            metadata={
                "dense_library_book_row": True,
                "dense_library_owner_bound": str(
                    bookcases[level_index * 4 + 3].object_id
                ),
            },
            placement_info=PlacementInfo(
                parent_surface_id=bookcases[level_index * 4 + 3]
                .support_surfaces[row_index % 6 + 1]
                .surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        for level_index in range(3)
        for row_index in range(12)
    ]
    scene = SimpleNamespace(
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )

    with pytest.raises(
        ModelBehaviorError,
        match=r"populated bookcase wall run at 0\.000m.*0.*required 3",
    ):
        StatefulManipulandAgent._validate_dense_library_book_rows(scene)


def test_dense_library_book_row_gate_rejects_eight_rows_per_story(tmp_path: Path):
    bookcases = [
        _dense_bookcase(f"bookcase_{level_index}_{case_index}", level, tmp_path)
        for level_index, level in enumerate((0.0, 4.0, 8.0))
        for case_index in range(2)
    ]
    rows = [
        SimpleNamespace(
            object_id=UniqueID(f"book_row_{level_index}_{row_index}"),
            object_type=ObjectType.MANIPULAND,
            metadata={"dense_library_book_row": True},
            placement_info=PlacementInfo(
                parent_surface_id=bookcases[level_index * 2 + row_index // 4]
                .support_surfaces[row_index % 4 + 1]
                .surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        for level_index in range(3)
        for row_index in range(8)
    ]
    scene = SimpleNamespace(
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )

    with pytest.raises(ModelBehaviorError, match=r"4\.000m placed 8, required 12"):
        StatefulManipulandAgent._validate_dense_library_book_rows(scene)


def test_dense_library_recovers_variable_capacity_templates_to_twelve_rows(
    tmp_path: Path,
    monkeypatch,
):
    bookcases = [
        _dense_bookcase(f"bookcase_{level_index}_{case_index}", level, tmp_path)
        for level_index, level in enumerate((0.0, 4.0, 8.0))
        for case_index in range(6 if level_index == 0 else 3)
    ]
    rows = []
    for level_index, level in enumerate((0.0, 4.0, 8.0)):
        cases_at_level = [
            bookcase
            for bookcase in bookcases
            if float(bookcase.transform.translation()[2]) == level
        ]
        rows_per_case = 2 if level_index == 0 else 4
        for case_index, bookcase in enumerate(cases_at_level[:3]):
            for row_index in range(rows_per_case):
                surface = bookcase.support_surfaces[row_index + 1]
                rows.append(
                    SceneObject(
                        object_id=UniqueID(
                            f"book_row_{level_index}_{case_index}_{row_index}"
                        ),
                        object_type=ObjectType.MANIPULAND,
                        name="encyclopedia_book_row",
                        description="upright encyclopedia book set",
                        transform=surface.transform,
                        geometry_path=tmp_path / "book_row.gltf",
                        metadata={
                            "dense_library_book_row": True,
                            "dense_library_owner_bound": str(bookcase.object_id),
                        },
                        bbox_min=np.array([-0.1, -0.05, 0.0]),
                        bbox_max=np.array([0.1, 0.05, 0.2]),
                        placement_info=PlacementInfo(
                            parent_surface_id=surface.surface_id,
                            position_2d=np.array([0.0, 0.0]),
                            rotation_2d=0.0,
                        ),
                    )
                )
    accented_source = next(
        bookcase for bookcase in bookcases if str(bookcase.object_id) == "bookcase_0_2"
    )
    accent_surface = accented_source.support_surfaces[1]
    accent = SceneObject(
        object_id=UniqueID("candlestick_0"),
        object_type=ObjectType.MANIPULAND,
        name="candlestick",
        description="antique brass candlestick",
        transform=accent_surface.transform,
        geometry_path=tmp_path / "candlestick.gltf",
        placement_info=PlacementInfo(
            parent_surface_id=accent_surface.surface_id,
            position_2d=np.array([0.1, 0.0]),
            rotation_2d=0.0,
        ),
    )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows, accent]},
    )
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.compute_scene_collisions",
        lambda **_kwargs: [],
    )
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = scene
    agent.cfg = SimpleNamespace(
        physics_validation=SimpleNamespace(
            object_penetration_threshold_m=0.001,
            floor_penetration_tolerance_m=0.05,
            manipuland_furniture_tolerance_m=0.02,
        )
    )

    recovered = agent._recover_dense_library_book_row_deficits()

    assert recovered == 6
    assert StatefulManipulandAgent._validate_dense_library_book_rows(scene) == 36
    recovered_rows = [
        obj
        for obj in scene.objects.values()
        if obj.placement_info is not None
        and obj.placement_info.placement_method == "template_transfer"
    ]
    assert len(recovered_rows) == 6
    assert all(row.metadata.get("dense_library_book_row") for row in recovered_rows)
    assert all(
        row.metadata.get("dense_library_owner_surface_local_transform") is not None
        for row in recovered_rows
    )
    assert all(row.name != "candlestick" for row in recovered_rows)


def test_dense_library_recovery_prioritizes_empty_grouped_wall_run(
    tmp_path: Path,
    monkeypatch,
):
    bookcases = [
        _dense_bookcase(f"bookcase_{level_index}_{case_index}", level, tmp_path)
        for level_index, level in enumerate((0.0, 4.0, 8.0))
        for case_index in range(6)
    ]
    rows = []
    for level_index, level in enumerate((0.0, 4.0, 8.0)):
        cases_at_level = bookcases[level_index * 6 : (level_index + 1) * 6]
        for bookcase in cases_at_level[:3]:
            bookcase.metadata["dense_library_grouped_run"] = level
        for case_index, bookcase in enumerate(cases_at_level[3:]):
            for row_index in range(4):
                surface = bookcase.support_surfaces[row_index + 1]
                rows.append(
                    SceneObject(
                        object_id=UniqueID(
                            f"source_row_{level_index}_{case_index}_{row_index}"
                        ),
                        object_type=ObjectType.MANIPULAND,
                        name="encyclopedia_book_row",
                        description="upright encyclopedia book set",
                        transform=surface.transform,
                        geometry_path=tmp_path / "book_row.gltf",
                        metadata={
                            "dense_library_book_row": True,
                            "dense_library_owner_bound": str(bookcase.object_id),
                        },
                        bbox_min=np.array([-0.1, -0.05, 0.0]),
                        bbox_max=np.array([0.1, 0.05, 0.2]),
                        placement_info=PlacementInfo(
                            parent_surface_id=surface.surface_id,
                            position_2d=np.array([0.0, 0.0]),
                            rotation_2d=0.0,
                        ),
                    )
                )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.compute_scene_collisions",
        lambda **_kwargs: [],
    )
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = scene
    agent.cfg = SimpleNamespace(
        physics_validation=SimpleNamespace(
            object_penetration_threshold_m=0.001,
            floor_penetration_tolerance_m=0.05,
            manipuland_furniture_tolerance_m=0.02,
        )
    )

    recovered = agent._recover_dense_library_book_row_deficits()

    assert recovered == 36
    assert StatefulManipulandAgent._validate_dense_library_book_rows(scene) == 72
    for bookcase in bookcases:
        if bookcase.metadata.get("dense_library_grouped_run") is not None:
            assert len(agent._dense_book_rows_on_furniture(scene, bookcase)) >= 3

    assert agent._normalize_dense_library_book_row_surplus() == 36
    assert StatefulManipulandAgent._validate_dense_library_book_rows(scene) == 36
    for bookcase in bookcases:
        if bookcase.metadata.get("dense_library_grouped_run") is not None:
            assert len(agent._dense_book_rows_on_furniture(scene, bookcase)) >= 3


def test_dense_library_recovery_clusters_millimeter_story_drift(
    tmp_path: Path,
    monkeypatch,
):
    """Post-simulation support drift must not split one authored story."""

    story_elevations = (
        (-0.0011208129834938, -0.0023808841993158, -0.0023808841648540),
        (3.9974164006940396,) * 3,
        (7.9974164006646555,) * 3,
    )
    bookcases = [
        _dense_bookcase(f"bookcase_{story_index}_{case_index}", elevation, tmp_path)
        for story_index, elevations in enumerate(story_elevations)
        for case_index, elevation in enumerate(elevations)
    ]
    rows = []
    for story_index, source_index in enumerate((0, 3, 6)):
        source = bookcases[source_index]
        for row_index in range(4 if story_index == 0 else 12):
            owner = (
                source if story_index == 0 else bookcases[source_index + row_index // 4]
            )
            surface = owner.support_surfaces[row_index % 4 + 1]
            rows.append(
                SceneObject(
                    object_id=UniqueID(f"book_row_{story_index}_{row_index}"),
                    object_type=ObjectType.MANIPULAND,
                    name="encyclopedia_book_row",
                    description="upright encyclopedia book set",
                    transform=surface.transform,
                    geometry_path=tmp_path / "book_row.gltf",
                    metadata={
                        "dense_library_book_row": True,
                        "dense_library_owner_bound": str(owner.object_id),
                    },
                    bbox_min=np.array([-0.1, -0.05, 0.0]),
                    bbox_max=np.array([0.1, 0.05, 0.2]),
                    placement_info=PlacementInfo(
                        parent_surface_id=surface.surface_id,
                        position_2d=np.array([0.0, 0.0]),
                        rotation_2d=0.0,
                    ),
                )
            )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.compute_scene_collisions",
        lambda **_kwargs: [],
    )
    agent = object.__new__(StatefulManipulandAgent)
    agent.scene = scene
    agent.cfg = SimpleNamespace(
        physics_validation=SimpleNamespace(
            object_penetration_threshold_m=0.001,
            floor_penetration_tolerance_m=0.05,
        )
    )

    recovered = agent._recover_dense_library_book_row_deficits()

    assert recovered == 8
    assert StatefulManipulandAgent._validate_dense_library_book_rows(scene) == 36


def test_dense_library_book_row_gate_rejects_physically_invalid_tagged_rows(
    tmp_path: Path,
):
    bookcases = [
        _dense_bookcase(f"bookcase_{int(level)}", level, tmp_path)
        for level in (0.0, 4.0, 8.0)
    ]
    rows = [
        SimpleNamespace(
            object_id=UniqueID(f"book_row_{level_index}_{row_index}"),
            object_type=ObjectType.MANIPULAND,
            metadata={"dense_library_book_row": True},
            placement_info=PlacementInfo(
                parent_surface_id=bookcases[level_index]
                .support_surfaces[row_index + 1]
                .surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        for level_index in range(3)
        for row_index in range(4)
    ]
    scene = SimpleNamespace(
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )

    with pytest.raises(ModelBehaviorError, match="physically invalid"):
        StatefulManipulandAgent._validate_dense_library_book_rows(
            scene,
            invalid_row_ids={rows[0].object_id},
        )


def test_dense_library_physics_allows_only_shallow_owner_support_contact(
    tmp_path: Path,
    monkeypatch,
):
    bookcase = _dense_bookcase("bookcase_0", 0.0, tmp_path)
    rows = [
        SceneObject(
            object_id=UniqueID(f"book_row_{index}"),
            object_type=ObjectType.MANIPULAND,
            name="encyclopedia_book_row",
            description="upright encyclopedia book set",
            transform=RigidTransform(),
            geometry_path=tmp_path / "book_row.gltf",
            metadata={"dense_library_book_row": True},
            bbox_min=np.array([-0.1, -0.05, 0.0]),
            bbox_max=np.array([0.1, 0.05, 0.2]),
            placement_info=PlacementInfo(
                parent_surface_id=bookcase.support_surfaces[index + 1].surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        for index in range(4)
    ]
    rows[3].transform = RigidTransform(p=[0.55, 0.0, 1.2])
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        objects={obj.object_id: obj for obj in [bookcase, *rows]},
    )
    collisions = [
        CollisionPair(
            bookcase.name,
            str(bookcase.object_id),
            rows[0].name,
            str(rows[0].object_id),
            0.019,
        ),
        CollisionPair(
            bookcase.name,
            str(bookcase.object_id),
            rows[1].name,
            str(rows[1].object_id),
            0.021,
        ),
        CollisionPair(
            "wall",
            "room_geometry",
            rows[2].name,
            str(rows[2].object_id),
            0.005,
        ),
    ]
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.compute_scene_collisions",
        lambda **_kwargs: collisions,
    )
    cfg = SimpleNamespace(
        physics_validation=SimpleNamespace(
            object_penetration_threshold_m=0.001,
            floor_penetration_tolerance_m=0.05,
            manipuland_furniture_tolerance_m=0.02,
        )
    )

    invalid = StatefulManipulandAgent._physically_invalid_dense_book_row_ids(
        scene,
        cfg,
    )

    assert rows[0].object_id not in invalid
    assert rows[1].object_id not in invalid
    assert invalid == {rows[2].object_id, rows[3].object_id}


def test_dense_library_containment_follows_owner_translation_and_rotation(
    tmp_path: Path,
    monkeypatch,
):
    bookcase = _dense_bookcase("bookcase_0", 0.0, tmp_path)
    surface = bookcase.support_surfaces[2]
    row = SceneObject(
        object_id=UniqueID("book_row_0"),
        object_type=ObjectType.MANIPULAND,
        name="encyclopedia_book_row",
        description="upright encyclopedia book set",
        transform=surface.transform,
        geometry_path=tmp_path / "book_row.gltf",
        metadata={
            "dense_library_book_row": True,
            "dense_library_owner_bound": str(bookcase.object_id),
            "dense_library_owner_surface_local_transform": (
                bookcase.transform.inverse() @ surface.transform
            )
            .GetAsMatrix4()
            .tolist(),
        },
        bbox_min=np.array([-0.1, -0.05, 0.0]),
        bbox_max=np.array([0.1, 0.05, 0.2]),
        placement_info=PlacementInfo(
            parent_surface_id=surface.surface_id,
            position_2d=np.array([0.0, 0.0]),
            rotation_2d=0.0,
        ),
    )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        objects={bookcase.object_id: bookcase, row.object_id: row},
    )
    collisions = []
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.compute_scene_collisions",
        lambda **_kwargs: collisions,
    )
    cfg = SimpleNamespace(
        physics_validation=SimpleNamespace(
            object_penetration_threshold_m=0.001,
            floor_penetration_tolerance_m=0.05,
        )
    )
    owner_delta = RigidTransform(
        RollPitchYaw(0.0, 0.0, math.radians(37.0)),
        [0.4, -0.3, 0.1],
    )
    bookcase.transform = owner_delta @ bookcase.transform
    row.transform = owner_delta @ row.transform

    invalid = StatefulManipulandAgent._physically_invalid_dense_book_row_ids(
        scene,
        cfg,
    )

    assert invalid == set()

    collisions.append(
        CollisionPair(
            "wall",
            "room_geometry",
            row.name,
            str(row.object_id),
            0.01,
        )
    )
    assert StatefulManipulandAgent._physically_invalid_dense_book_row_ids(
        scene,
        cfg,
    ) == {row.object_id}


def test_final_dense_library_gate_rejects_row_displaced_after_simulation(
    tmp_path: Path,
    monkeypatch,
):
    bookcases = [
        _dense_bookcase(f"bookcase_{int(level)}", level, tmp_path)
        for level in (0.0, 4.0, 8.0)
    ]
    rows = []
    for level_index, bookcase in enumerate(bookcases):
        for row_index in range(4):
            surface = bookcase.support_surfaces[row_index + 1]
            row = SceneObject(
                object_id=UniqueID(f"book_row_{level_index}_{row_index}"),
                object_type=ObjectType.MANIPULAND,
                name="encyclopedia_book_row",
                description="upright encyclopedia book set",
                transform=surface.transform,
                geometry_path=tmp_path / "book_row.gltf",
                metadata={
                    "dense_library_book_row": True,
                    "dense_library_owner_bound": str(bookcase.object_id),
                },
                bbox_min=np.array([-0.1, -0.05, 0.0]),
                bbox_max=np.array([0.1, 0.05, 0.2]),
                placement_info=PlacementInfo(
                    parent_surface_id=surface.surface_id,
                    position_2d=np.array([0.0, 0.0]),
                    rotation_2d=0.0,
                ),
            )
            rows.append(row)
    rows[0].transform = rows[0].transform @ RigidTransform(
        RollPitchYaw(0.0, 0.0, math.radians(130.0)),
        [0.4, 0.0, 0.0],
    )
    scene = RoomScene(
        room_geometry=object(),
        scene_dir=tmp_path,
        text_description="large multi-level library with thousands of books",
        objects={obj.object_id: obj for obj in [*bookcases, *rows]},
    )
    monkeypatch.setattr(
        "scenesmith.manipuland_agents.stateful_manipuland_agent.compute_scene_collisions",
        lambda **_kwargs: [],
    )
    agent = SimpleNamespace(
        cfg=SimpleNamespace(
            physics_validation=SimpleNamespace(
                object_penetration_threshold_m=0.001,
                floor_penetration_tolerance_m=0.05,
            )
        )
    )

    with pytest.raises(ModelBehaviorError, match="physically invalid"):
        _validate_final_dense_library_book_rows(scene, agent)


def test_patient_support_zone_rejects_equipment_and_mislabeled_tape(tmp_path: Path):
    bed = SceneObject(
        object_id=UniqueID("bed_0"),
        object_type=ObjectType.FURNITURE,
        name="medical_bed",
        description="science-fiction treatment bed",
        transform=RigidTransform(),
        geometry_path=tmp_path / "bed.glb",
    )
    monitor = SceneObject(
        object_id=UniqueID("monitor_0"),
        object_type=ObjectType.MANIPULAND,
        name="monitor_accessory",
        description="compact patient monitor",
        transform=RigidTransform(),
        geometry_path=tmp_path / "monitor.glb",
    )
    mislabeled_blanket = SceneObject(
        object_id=UniqueID("blanket_0"),
        object_type=ObjectType.MANIPULAND,
        name="folded_blanket",
        description="folded medical blanket",
        transform=RigidTransform(),
        geometry_path=tmp_path / "tape.glb",
        metadata={"catalog_id": "polyhaven__medical_tape"},
    )
    pillow = SceneObject(
        object_id=UniqueID("pillow_0"),
        object_type=ObjectType.MANIPULAND,
        name="pillow",
        description="soft white pillow",
        transform=RigidTransform(),
        geometry_path=tmp_path / "pillow.glb",
    )

    assert ManipulandTools._support_semantics_allow(bed, monitor)[0] is False
    assert ManipulandTools._support_semantics_allow(bed, mislabeled_blanket)[0] is False
    assert ManipulandTools._support_semantics_allow(bed, pillow) == (True, None)
