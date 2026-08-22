import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    PlacementInfo,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)


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
