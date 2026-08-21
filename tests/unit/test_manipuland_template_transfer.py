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
        objects={source.object_id: source, target.object_id: target, pillow.object_id: pillow},
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
            "ontology_path": (
                "polyhaven/Office & Stationery/Books & Documents/Books"
            ),
            "catalog_semantics": "Book Encyclopedia Set 01",
        },
    )
    placements = []

    class FakeTools:
        support_surfaces = {
            str(surface.surface_id): surface for surface in furniture.support_surfaces
        }

        def _place_manipuland_on_surface_impl(self, **kwargs):
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

    with pytest.raises(ModelBehaviorError, match=r"4\.000m.*0.*4"):
        StatefulManipulandAgent._validate_dense_library_book_rows(scene)


def test_dense_library_book_row_gate_accepts_four_rows_per_story(tmp_path: Path):
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

    assert StatefulManipulandAgent._validate_dense_library_book_rows(scene) == 12


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
    assert (
        ManipulandTools._support_semantics_allow(bed, mislabeled_blanket)[0] is False
    )
    assert ManipulandTools._support_semantics_allow(bed, pillow) == (True, None)
