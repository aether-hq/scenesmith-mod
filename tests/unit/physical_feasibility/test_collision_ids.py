"""Unit tests for physical feasibility post-processing module."""

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path
from unittest.mock import patch

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.physics.feasibility.projection import (
    _get_colliding_object_ids,
)
from scenesmith.agent_utils.physics.physical_feasibility import (
    apply_non_penetration_projection,
)
from scenesmith.agent_utils.physics.validation.models import CollisionPair
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)

# Path to test data.
TEST_DATA_DIR = Path(__file__).parents[2] / "test_data"


class PhysicalFeasibilityTestCase(unittest.TestCase):
    """Base test case with shared fixtures for physical feasibility tests."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up shared test fixtures."""
        floor_plan_sdf_path = TEST_DATA_DIR / "simple_room_geometry.sdf"
        cls.room_geometry = RoomGeometry(
            sdf_tree=ET.parse(floor_plan_sdf_path),
            sdf_path=floor_plan_sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=10.0,
            length=10.0,
        )

    def _create_overlapping_boxes_scene(self, scene_dir: Path) -> RoomScene:
        """Create a scene with two overlapping boxes.

        Uses simple_box.sdf (0.5x0.5x0.5m boxes). Boxes overlap when centers
        are less than 0.5m apart.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with overlapping boxes",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        box1 = SceneObject(
            object_id=UniqueID("box_1"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 1",
            transform=RigidTransform(p=[0.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        box2 = SceneObject(
            object_id=UniqueID("box_2"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 2",
            transform=RigidTransform(p=[0.3, 0.01, 0.25]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(box1)
        scene.add_object(box2)

        return scene

    def _create_non_overlapping_boxes_scene(self, scene_dir: Path) -> RoomScene:
        """Create a scene with two non-overlapping boxes (2m apart)."""
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with non-overlapping boxes",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        box1 = SceneObject(
            object_id=UniqueID("box_1"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 1",
            transform=RigidTransform(p=[1.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        box2 = SceneObject(
            object_id=UniqueID("box_2"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Test box 2",
            transform=RigidTransform(p=[-1.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(box1)
        scene.add_object(box2)

        return scene

    def _create_scene_with_manipuland(self, scene_dir: Path) -> RoomScene:
        """Create a scene with furniture and a manipuland (sphere on box)."""
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with furniture and manipuland",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        sphere_sdf_path = TEST_DATA_DIR / "simple_sphere.sdf"

        furniture = SceneObject(
            object_id=UniqueID("table_0"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(p=[0.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        manipuland = SceneObject(
            object_id=UniqueID("ball_0"),
            object_type=ObjectType.MANIPULAND,
            name="ball",
            description="Test ball",
            transform=RigidTransform(p=[0.0, 0.0, 0.7]),
            sdf_path=sphere_sdf_path,
        )

        scene.add_object(furniture)
        scene.add_object(manipuland)

        return scene

    def _create_scene_with_stack(self, scene_dir: Path) -> RoomScene:
        """Create a scene with a stack (composite object).

        Uses simple_box.sdf (0.5x0.5x0.5m boxes) to create a two-box stack.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with stack",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        # Create a stack with 2 boxes stacked vertically.
        stack = SceneObject(
            object_id=UniqueID("stack_0"),
            object_type=ObjectType.FURNITURE,
            name="stack",
            description="Test stack",
            transform=RigidTransform(p=[0.0, 0.0, 0.0]),
            sdf_path=None,  # Composite objects don't have their own SDF.
            metadata={
                "composite_type": "stack",
                "member_assets": [
                    {
                        "name": "bottom_box",
                        "asset_id": "asset_bottom123",
                        "sdf_path": str(box_sdf_path),
                        "transform": {
                            "translation": [0.0, 0.0, 0.25],
                            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    },
                    {
                        "name": "top_box",
                        "asset_id": "asset_top456",
                        "sdf_path": str(box_sdf_path),
                        "transform": {
                            "translation": [0.0, 0.0, 0.75],
                            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    },
                ],
            },
        )
        scene.add_object(stack)

        return scene


class TestGetCollidingObjectIds(PhysicalFeasibilityTestCase):
    """Tests for _get_colliding_object_ids helper function."""

    def test_no_collisions_returns_empty_set(self) -> None:
        """Scene with no collisions should return empty set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))
            colliding_ids = _get_colliding_object_ids(scene)
            self.assertEqual(colliding_ids, set())

    def test_two_penetrating_objects_returns_both_ids(self) -> None:
        """Two overlapping objects should both be in result."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))
            colliding_ids = _get_colliding_object_ids(scene)

            self.assertEqual(len(colliding_ids), 2)
            self.assertIn(UniqueID("box_1"), colliding_ids)
            self.assertIn(UniqueID("box_2"), colliding_ids)

    def test_owner_bound_book_row_contact_is_not_a_projection_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="large multi-level library with thousands of books",
            )
            owner = SceneObject(
                object_id=UniqueID("bookcase_0"),
                object_type=ObjectType.FURNITURE,
                name="bookcase",
                description="library bookcase",
                transform=RigidTransform(),
            )
            row = SceneObject(
                object_id=UniqueID("book_row_0"),
                object_type=ObjectType.MANIPULAND,
                name="book_row",
                description="row of books",
                transform=RigidTransform(),
                metadata={"dense_library_owner_bound": "bookcase_0"},
            )
            scene.add_object(owner)
            scene.add_object(row)
            collisions = [
                CollisionPair(
                    owner.name,
                    str(owner.object_id),
                    row.name,
                    str(row.object_id),
                    0.03,
                )
            ]

            with patch(
                "scenesmith.agent_utils.physics.physical_feasibility."
                "compute_scene_collisions",
                return_value=collisions,
            ):
                colliding_ids = _get_colliding_object_ids(scene)

            self.assertEqual(colliding_ids, set())

    def test_static_owner_bound_scene_is_a_successful_projection_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="large multi-level library with thousands of books",
            )
            scene.add_object(
                SceneObject(
                    object_id=UniqueID("bookcase_0"),
                    object_type=ObjectType.FURNITURE,
                    name="bookcase",
                    description="library bookcase",
                    transform=RigidTransform(),
                )
            )
            scene.add_object(
                SceneObject(
                    object_id=UniqueID("book_row_0"),
                    object_type=ObjectType.MANIPULAND,
                    name="book_row",
                    description="row of books",
                    transform=RigidTransform(),
                    metadata={"dense_library_owner_bound": "bookcase_0"},
                )
            )
            scene.add_object(
                SceneObject(
                    object_id=UniqueID("wall_0"),
                    object_type=ObjectType.WALL,
                    name="wall",
                    description="structural wall panel",
                    transform=RigidTransform(),
                )
            )

            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "_get_colliding_object_ids",
                    return_value=set(),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "_create_drake_plant_for_ik"
                ) as create_plant,
            ):
                projected, success = apply_non_penetration_projection(
                    scene,
                    weld_furniture=True,
                )

            self.assertIs(projected, scene)
            self.assertTrue(success)
            create_plant.assert_not_called()

    def test_chain_collision_returns_all_involved(self) -> None:
        """A-B collision and B-C collision should return A, B, C."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Chain collision test",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Box A at origin.
            box_a = SceneObject(
                object_id=UniqueID("box_a"),
                object_type=ObjectType.FURNITURE,
                name="box_a",
                description="Box A",
                transform=RigidTransform(p=[0.0, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # Box B overlapping with A (shifted 0.3m in X).
            box_b = SceneObject(
                object_id=UniqueID("box_b"),
                object_type=ObjectType.FURNITURE,
                name="box_b",
                description="Box B",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # Box C overlapping with B but not A (shifted 0.6m in X).
            box_c = SceneObject(
                object_id=UniqueID("box_c"),
                object_type=ObjectType.FURNITURE,
                name="box_c",
                description="Box C",
                transform=RigidTransform(p=[0.6, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            scene.add_object(box_a)
            scene.add_object(box_b)
            scene.add_object(box_c)

            colliding_ids = _get_colliding_object_ids(scene)

            # A-B collide, B-C collide → all three should be in the set.
            self.assertEqual(len(colliding_ids), 3)
            self.assertIn(UniqueID("box_a"), colliding_ids)
            self.assertIn(UniqueID("box_b"), colliding_ids)
            self.assertIn(UniqueID("box_c"), colliding_ids)

    def test_isolated_non_colliding_object_not_included(self) -> None:
        """Object not in collision should not be in the result."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Mixed collision test",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Two overlapping boxes.
            box1 = SceneObject(
                object_id=UniqueID("box_1"),
                object_type=ObjectType.FURNITURE,
                name="box_1",
                description="Box 1",
                transform=RigidTransform(p=[0.0, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )
            box2 = SceneObject(
                object_id=UniqueID("box_2"),
                object_type=ObjectType.FURNITURE,
                name="box_2",
                description="Box 2",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # One isolated box far away (but not near walls - room is 10x10).
            box_isolated = SceneObject(
                object_id=UniqueID("box_isolated"),
                object_type=ObjectType.FURNITURE,
                name="box_isolated",
                description="Isolated box",
                transform=RigidTransform(p=[3.0, 3.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            scene.add_object(box1)
            scene.add_object(box2)
            scene.add_object(box_isolated)

            colliding_ids = _get_colliding_object_ids(scene)

            # Only the overlapping pair should be in result (isolated box not colliding).
            self.assertEqual(len(colliding_ids), 2)
            self.assertIn(UniqueID("box_1"), colliding_ids)
            self.assertIn(UniqueID("box_2"), colliding_ids)
            self.assertNotIn(UniqueID("box_isolated"), colliding_ids)


class TestLargeSceneOptimization(PhysicalFeasibilityTestCase):
    """Tests for threshold-based DOF reduction optimization."""

    def test_small_scene_uses_all_free_objects(self) -> None:
        """Scene below threshold uses original path (all objects free)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            # Capture initial positions before projection.
            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            # 2 objects, threshold is 100 → small scene path.
            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="snopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                    large_scene_optimization_threshold=100,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            # At least one box should have moved to resolve collision.
            self.assertTrue(success)
            box1_after = projected_scene.get_object(UniqueID("box_1"))
            box2_after = projected_scene.get_object(UniqueID("box_2"))

            pos1_changed = not np.allclose(
                box1_after.transform.translation(),
                initial_pos1,
                atol=0.01,
            )
            pos2_changed = not np.allclose(
                box2_after.transform.translation(),
                initial_pos2,
                atol=0.01,
            )
            self.assertTrue(pos1_changed or pos2_changed)

    def test_large_scene_no_collisions_skips_projection(self) -> None:
        """Large scene with no collisions returns early with success."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            # Set threshold to 1 so 2 objects triggers large scene path.
            result_scene, success = apply_non_penetration_projection(
                scene=scene,
                influence_distance=0.03,
                solver_name="snopt",
                iteration_limit=5000,
                weld_furniture=False,
                xy_only=False,
                fix_rotation=True,
                large_scene_optimization_threshold=1,  # 2 objects > 1 threshold.
            )

            # Should succeed with no changes (early return, no collisions).
            self.assertTrue(success)

            box1_after = result_scene.get_object(UniqueID("box_1"))
            box2_after = result_scene.get_object(UniqueID("box_2"))

            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos1, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(box2_after.transform.translation(), initial_pos2, atol=1e-6)
            )

    def test_large_scene_only_colliding_objects_move(self) -> None:
        """Large scene optimization only allows colliding objects to move."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Large scene optimization test",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Two overlapping boxes.
            box1 = SceneObject(
                object_id=UniqueID("box_1"),
                object_type=ObjectType.FURNITURE,
                name="box_1",
                description="Box 1",
                transform=RigidTransform(p=[0.0, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )
            box2 = SceneObject(
                object_id=UniqueID("box_2"),
                object_type=ObjectType.FURNITURE,
                name="box_2",
                description="Box 2",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            # One isolated box far away (but not near walls - room is 10x10).
            box_isolated = SceneObject(
                object_id=UniqueID("box_isolated"),
                object_type=ObjectType.FURNITURE,
                name="box_isolated",
                description="Isolated box",
                transform=RigidTransform(p=[3.0, 3.0, 0.25]),
                sdf_path=box_sdf_path,
            )

            scene.add_object(box1)
            scene.add_object(box2)
            scene.add_object(box_isolated)

            initial_isolated_pos = box_isolated.transform.translation().copy()

            # Set threshold to 1 so 3 objects triggers large scene path.
            try:
                result_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="snopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                    large_scene_optimization_threshold=1,  # 3 objects > 1.
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            self.assertTrue(success)

            # Isolated box should NOT have moved (welded in optimization).
            isolated_after = result_scene.get_object(UniqueID("box_isolated"))
            self.assertTrue(
                np.allclose(
                    isolated_after.transform.translation(),
                    initial_isolated_pos,
                    atol=1e-6,
                ),
                f"Isolated box should not move. Initial: {initial_isolated_pos}, "
                f"Final: {isolated_after.transform.translation()}",
            )


if __name__ == "__main__":
    unittest.main()
