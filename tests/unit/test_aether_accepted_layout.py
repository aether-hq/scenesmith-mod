"""Accepted Aether topology is compiled exactly or refused explicitly."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydrake.all import RigidTransform

from scenesmith.aether.accepted_layout import (
    AcceptedLayoutError,
    accepted_wall_thickness,
    build_accepted_house_layout,
)
from scenesmith.aether.locked_inventory import (
    CensusError,
    placements_for_stage,
    seed_locked_inventory,
)
from scenesmith.agent_utils.house import OpeningType, WallDirection
from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID


def _stage_input() -> dict:
    return {
        "realization_engine": "scenesmith",
        "pipeline_profile": "full",
        "people_allowed": False,
        "room_prompt": "A complete approved private meeting bar with contextual dressing.",
        "request": {
            "theme": {
                "timeline": "near future",
                "genre": "cyberpunk hospitality",
                "craft_character": "repaired industrial construction",
                "wear_intent": "heavy service wear",
            },
            "shell": {
                "room_id": "meeting-bar",
                "label": "Meeting bar",
                "dimensions_m": [14.0, 3.8, 10.0],
                "wall_thickness_m": 0.14,
                "openings": [
                    {
                        "opening_id": "public-entry",
                        "boundary": "south",
                        "shape": "rectangle",
                        "offset_m": 1.2,
                        "sill_m": 0.0,
                        "width_m": 1.4,
                        "height_m": 2.4,
                        "mechanism": "double-hinged",
                    },
                    {
                        "opening_id": "street-window",
                        "boundary": "north",
                        "shape": "rectangle",
                        "offset_m": 4.0,
                        "sill_m": 0.9,
                        "width_m": 3.0,
                        "height_m": 1.8,
                        "mechanism": "fixed-glazing",
                    },
                ],
            },
        },
    }


class AcceptedLayoutTests(unittest.TestCase):
    def test_builds_exact_dimensions_openings_and_height(self) -> None:
        stage_input = _stage_input()
        layout = build_accepted_house_layout(stage_input, house_dir=Path("/tmp/job"))

        self.assertEqual(layout.wall_height, 3.8)
        self.assertEqual(layout.room_specs[0].length, 14.0)
        self.assertEqual(layout.room_specs[0].width, 10.0)
        self.assertEqual(layout.placed_rooms[0].width, 14.0)
        self.assertEqual(layout.placed_rooms[0].depth, 10.0)
        walls = {wall.direction: wall for wall in layout.placed_rooms[0].walls}
        self.assertEqual(walls[WallDirection.SOUTH].openings[0].opening_type, OpeningType.DOOR)
        self.assertEqual(walls[WallDirection.NORTH].openings[0].opening_type, OpeningType.WINDOW)
        self.assertEqual(accepted_wall_thickness(stage_input), 0.14)

    def test_rejects_people_and_partial_pipeline(self) -> None:
        for field, value in (("people_allowed", True), ("pipeline_profile", "partial")):
            stage_input = _stage_input()
            stage_input[field] = value
            with self.assertRaises(AcceptedLayoutError):
                build_accepted_house_layout(stage_input, house_dir=Path("/tmp/job"))

    def test_rejects_curved_opening_instead_of_flattening_it(self) -> None:
        stage_input = _stage_input()
        stage_input["request"]["shell"]["openings"][0]["shape"] = "arch"
        with self.assertRaisesRegex(AcceptedLayoutError, "unsupported exact shape"):
            build_accepted_house_layout(stage_input, house_dir=Path("/tmp/job"))

    def test_rejects_opening_outside_authored_wall(self) -> None:
        stage_input = _stage_input()
        stage_input["request"]["shell"]["openings"][0]["offset_m"] = 13.0
        stage_input["request"]["shell"]["openings"][0]["width_m"] = 2.0
        with self.assertRaisesRegex(AcceptedLayoutError, "extends past"):
            build_accepted_house_layout(stage_input, house_dir=Path("/tmp/job"))


class LockedInventoryTests(unittest.TestCase):
    def _inventory_input(self, *, sources=("objaverse",)) -> dict:
        return {
            **_stage_input(),
            "job_id": "accepted-job",
            "locked_placements": [
                {
                    "object_id": "private-booth-001",
                    "role_id": "private-booth",
                    "functional_zone_ids": ["private-meeting-zone"],
                    "stage": "furniture",
                    "placement_kind": "againstWall",
                    "description": "A high-backed private meeting booth",
                    "centre_origin_m": [2.0, -1.5, 0.7],
                    "floor_origin_m": [2.0, -1.5, 0.0],
                    "size_m": [2.4, 1.4, 1.4],
                    "yaw_radians": 0.5,
                    "route": {"sources": list(sources)},
                    "accepts_dressing": True,
                    "locked_fields": ["position", "orientation", "dimensions"],
                    "provenance": ["approved-layout"],
                }
            ],
        }

    def test_stage_filter_preserves_original_placement_kind(self) -> None:
        placements = placements_for_stage(self._inventory_input(), "furniture")
        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0]["placement_kind"], "againstWall")
        self.assertEqual(placements_for_stage(self._inventory_input(), "manipuland"), ())

    def test_cpu_retrieval_seeds_exact_identity_pose_and_provenance(self) -> None:
        geometry = Path(self.id().replace(".", "-") + ".gltf")
        geometry.write_bytes(b"immutable-test-geometry")
        self.addCleanup(geometry.unlink, missing_ok=True)
        acquired = SceneObject(
            object_id=UniqueID("retrieved-template-0"),
            object_type=ObjectType.FURNITURE,
            name="private-booth-001",
            description="retrieved booth",
            transform=RigidTransform(),
            geometry_path=geometry,
            bbox_min=np.array([-1.2, -0.7, 0.0]),
            bbox_max=np.array([1.2, 0.7, 1.4]),
            metadata={"asset_source": "objaverse", "objaverse_mesh_id": "asset-7"},
        )

        class Scene:
            def __init__(self):
                self.objects = {}

            def add_object(self, item):
                self.objects[item.object_id] = item

        class Manager:
            general_asset_source = "objaverse"

            def __init__(self):
                self.request = None

            def generate_assets(self, request):
                self.request = request
                return SimpleNamespace(
                    has_failures=False,
                    failed_assets=[],
                    successful_assets=[acquired],
                )

        scene = Scene()
        manager = Manager()
        placed_ids = seed_locked_inventory(
            scene=scene,
            asset_manager=manager,
            stage_input=self._inventory_input(),
            stage="furniture",
        )

        self.assertEqual(placed_ids, ("private-booth-001",))
        self.assertEqual(manager.request.desired_dimensions, [[2.4, 1.4, 1.4]])
        placed = scene.objects[UniqueID("private-booth-001")]
        local_center = (placed.bbox_min + placed.bbox_max) / 2
        np.testing.assert_allclose(placed.transform @ local_center, [2.0, -1.5, 0.7])
        self.assertTrue(placed.immutable)
        self.assertEqual(placed.metadata["aether_placement_kind"], "againstWall")
        self.assertEqual(placed.metadata["aether_asset_id"], "objaverse:asset-7")
        self.assertEqual(placed.metadata["aether_provenance"], ["approved-layout"])
        self.assertEqual(placed.metadata["aether_measured_size_m"], [2.4, 1.4, 1.4])

    def test_locked_asset_refuses_a_bad_dimension_fit(self) -> None:
        geometry = Path(self.id().replace(".", "-") + ".gltf")
        geometry.write_bytes(b"bad-dimension-geometry")
        self.addCleanup(geometry.unlink, missing_ok=True)
        acquired = SceneObject(
            object_id=UniqueID("retrieved-template-0"),
            object_type=ObjectType.FURNITURE,
            name="private-booth-001",
            description="retrieved booth with the wrong envelope",
            transform=RigidTransform(),
            geometry_path=geometry,
            bbox_min=np.array([-0.5, -0.2, 0.0]),
            bbox_max=np.array([0.5, 0.2, 0.6]),
            metadata={"asset_source": "objaverse", "objaverse_mesh_id": "asset-8"},
        )

        class Scene:
            def __init__(self) -> None:
                self.objects = {}

            def add_object(self, item):
                self.objects[item.object_id] = item

        class Manager:
            general_asset_source = "objaverse"

            def generate_assets(self, _request):
                return SimpleNamespace(
                    has_failures=False,
                    failed_assets=[],
                    successful_assets=[acquired],
                )

        with self.assertRaisesRegex(CensusError, "cannot honor locked dimensions"):
            seed_locked_inventory(
                scene=Scene(),
                asset_manager=Manager(),
                stage_input=self._inventory_input(),
                stage="furniture",
            )

    def test_cpu_worker_refuses_selected_cuda_only_route(self) -> None:
        manager = SimpleNamespace(general_asset_source="objaverse")
        scene = SimpleNamespace(objects={})
        with self.assertRaisesRegex(CensusError, "requires one of.*sam3d"):
            seed_locked_inventory(
                scene=scene,
                asset_manager=manager,
                stage_input=self._inventory_input(sources=("sam3d",)),
                stage="furniture",
            )


if __name__ == "__main__":
    unittest.main()
