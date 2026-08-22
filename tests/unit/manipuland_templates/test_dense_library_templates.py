import math

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.exceptions import ModelBehaviorError
from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.physics.validation.models import CollisionPair
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    PlacementInfo,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.experiments.indoor.runtime_support import (
    _validate_final_dense_library_book_rows,
)
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools


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
