"""Unit tests for physical feasibility post-processing module."""

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path
from unittest.mock import patch

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.physics.physical_feasibility import (
    apply_physical_feasibility_postprocessing,
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


class TestApplyPhysicalFeasibilityPostprocessing(PhysicalFeasibilityTestCase):
    """Tests for the combined post-processing pipeline."""

    def test_projection_followed_by_simulation(self) -> None:
        """Test applying projection followed by simulation (full pipeline)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            # Use SNOPT instead of IPOPT - IPOPT has numerical issues with
            # Drake's box-box gradient computation in edge cases.
            try:
                processed_scene, success, removed_ids = (
                    apply_physical_feasibility_postprocessing(
                        scene=scene,
                        weld_furniture=False,
                        projection_enabled=True,
                        projection_influence_distance=0.03,
                        projection_solver_name="snopt",
                        projection_iteration_limit=5000,
                        projection_xy_only=False,
                        projection_fix_rotation=True,
                        simulation_enabled=True,
                        simulation_time_s=0.5,
                        simulation_time_step_s=1e-3,
                        simulation_timeout_s=30.0,
                    )
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            self.assertTrue(success)
            self.assertEqual(removed_ids, [])

            box1_after = processed_scene.get_object(UniqueID("box_1"))
            box2_after = processed_scene.get_object(UniqueID("box_2"))

            pos1_changed = not np.allclose(
                box1_after.transform.translation(), initial_pos1, atol=0.01
            )
            pos2_changed = not np.allclose(
                box2_after.transform.translation(), initial_pos2, atol=0.01
            )
            self.assertTrue(pos1_changed or pos2_changed)

    def test_disabled_projection(self) -> None:
        """Test that disabled projection skips projection stage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            initial_pos = box1.transform.translation().copy()

            processed_scene, success, _ = apply_physical_feasibility_postprocessing(
                scene=scene,
                weld_furniture=False,
                projection_enabled=False,
                simulation_enabled=False,
            )

            self.assertTrue(success)

            box1_after = processed_scene.get_object(UniqueID("box_1"))
            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos)
            )

    def test_failed_projection_accepts_only_clean_bounded_simulation(self) -> None:
        """A repair-solver failure is not a dirty-scene publication verdict."""

        def settle_scene(*, scene, **_kwargs):
            box = scene.get_object(UniqueID("box_1"))
            position = box.transform.translation().copy()
            position[0] += 0.03
            box.transform = RigidTransform(p=position)
            return scene, []

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))
            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=lambda *, scene, **_kwargs: (scene, False),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "_apply_floor_penetration_fallback",
                    side_effect=lambda scene, **_kwargs: (scene, 0),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation",
                    side_effect=settle_scene,
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "compute_scene_collisions",
                    return_value=[],
                ),
            ):
                _, success, removed = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

        self.assertTrue(success)
        self.assertEqual(removed, [])

    def test_failed_welded_projection_accepts_clean_simulation_evidence(self) -> None:
        """A clean simulation can certify fixed furniture and settled objects."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))
            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=lambda *, scene, **_kwargs: (scene, False),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation",
                    side_effect=lambda *, scene, **_kwargs: (scene, []),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "compute_scene_collisions",
                    return_value=[],
                ) as collisions,
            ):
                _, success, removed = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=True,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

        self.assertTrue(success)
        self.assertEqual(removed, [])
        collisions.assert_called_once()

    def test_failed_welded_projection_rejects_remaining_collision(self) -> None:
        """A solver failure remains blocking when simulation leaves a collision."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))
            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=lambda *, scene, **_kwargs: (scene, False),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation",
                    side_effect=lambda *, scene, **_kwargs: (scene, []),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "compute_scene_collisions",
                    return_value=[object()],
                ),
            ):
                _, success, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=True,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

        self.assertFalse(success)

    def test_failed_projection_rejects_dirty_or_excessive_simulation(self) -> None:
        """Fallback recovery stays false for collisions or unstable motion."""

        for displacement, collisions in ((0.03, [object()]), (0.06, [])):
            with self.subTest(displacement=displacement, collisions=collisions):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

                    def settle_scene(*, scene, **_kwargs):
                        box = scene.get_object(UniqueID("box_1"))
                        position = box.transform.translation().copy()
                        position[0] += displacement
                        box.transform = RigidTransform(p=position)
                        return scene, []

                    with (
                        patch(
                            "scenesmith.agent_utils.physics.physical_feasibility."
                            "apply_non_penetration_projection",
                            side_effect=lambda *, scene, **_kwargs: (scene, False),
                        ),
                        patch(
                            "scenesmith.agent_utils.physics.physical_feasibility."
                            "_apply_floor_penetration_fallback",
                            side_effect=lambda scene, **_kwargs: (scene, 0),
                        ),
                        patch(
                            "scenesmith.agent_utils.physics.physical_feasibility."
                            "apply_forward_simulation",
                            side_effect=settle_scene,
                        ),
                        patch(
                            "scenesmith.agent_utils.physics.physical_feasibility."
                            "compute_scene_collisions",
                            return_value=collisions,
                        ),
                    ):
                        _, success, _ = apply_physical_feasibility_postprocessing(
                            scene=scene,
                            weld_furniture=False,
                            projection_enabled=True,
                            simulation_enabled=True,
                        )

                self.assertFalse(success)

    def test_successful_projection_restores_new_post_simulation_collision(
        self,
    ) -> None:
        """Simulation cannot publish a collision absent from its clean input."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))
            box = scene.get_object(UniqueID("box_1"))
            original_position = box.transform.translation().copy()
            collision = CollisionPair(
                "floor",
                "room_geometry",
                box.name,
                str(box.object_id),
                0.0877,
            )

            def sink_table(*, scene, **_kwargs):
                moved = scene.get_object(UniqueID("box_1"))
                position = moved.transform.translation().copy()
                position[2] -= 0.08
                moved.transform = RigidTransform(p=position)
                return scene, []

            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=lambda *, scene, **_kwargs: (scene, True),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation",
                    side_effect=sink_table,
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "compute_scene_collisions",
                    side_effect=([collision], []),
                ) as collisions,
            ):
                processed, success, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

        self.assertTrue(success)
        self.assertTrue(
            np.allclose(
                processed.get_object(UniqueID("box_1")).transform.translation(),
                original_position,
            )
        )
        self.assertEqual(collisions.call_count, 2)

    def test_owner_bound_decor_preserves_full_relative_pose_across_postprocessing(
        self,
    ) -> None:
        """Projection and simulation cannot detach decor welded to its owner."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))
            owner = scene.get_object(UniqueID("table_0"))
            row = scene.get_object(UniqueID("ball_0"))
            owner.transform = RigidTransform(
                RollPitchYaw(0.0, 0.0, 0.3), [1.0, 2.0, 0.25]
            )
            relative = RigidTransform(RollPitchYaw(0.1, -0.05, -0.2), [0.2, -0.1, 0.4])
            row.transform = owner.transform @ relative
            row.metadata["dense_library_owner_bound"] = "table_0"
            projection_owner = RigidTransform(
                RollPitchYaw(0.0, 0.0, 1.1), [2.5, -1.0, 0.3]
            )
            simulation_owner = RigidTransform(
                RollPitchYaw(0.05, -0.02, -0.7), [-1.5, 0.75, 0.28]
            )
            attached_before_simulation = []

            def detach_in_projection(*, scene, **_kwargs):
                scene.get_object(UniqueID("table_0")).transform = projection_owner
                scene.get_object(UniqueID("ball_0")).transform = RigidTransform(
                    RollPitchYaw(0.4, 0.2, 0.8), [-4.0, 5.0, 7.0]
                )
                return scene, True

            def move_owner_in_simulation(*, scene, **_kwargs):
                current_row = scene.get_object(UniqueID("ball_0"))
                attached_before_simulation.append(
                    np.allclose(
                        current_row.transform.GetAsMatrix4(),
                        (projection_owner @ relative).GetAsMatrix4(),
                    )
                )
                scene.get_object(UniqueID("table_0")).transform = simulation_owner
                return scene, []

            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=detach_in_projection,
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation",
                    side_effect=move_owner_in_simulation,
                ),
            ):
                processed, success, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=True,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

        self.assertTrue(success)
        self.assertEqual(attached_before_simulation, [True])
        self.assertTrue(
            np.allclose(
                processed.get_object(UniqueID("ball_0")).transform.GetAsMatrix4(),
                (simulation_owner @ relative).GetAsMatrix4(),
            )
        )

    def test_successful_projection_rejects_dirty_restored_scene(self) -> None:
        """A targeted restore must still pass an authoritative full recheck."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))
            box = scene.get_object(UniqueID("box_1"))
            collision = CollisionPair(
                "floor",
                "room_geometry",
                box.name,
                str(box.object_id),
                0.0877,
            )

            def sink_table(*, scene, **_kwargs):
                moved = scene.get_object(UniqueID("box_1"))
                position = moved.transform.translation().copy()
                position[2] -= 0.08
                moved.transform = RigidTransform(p=position)
                return scene, []

            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=lambda *, scene, **_kwargs: (scene, True),
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation",
                    side_effect=sink_table,
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "compute_scene_collisions",
                    return_value=[collision],
                ),
            ):
                _, success, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

        self.assertFalse(success)

    def test_ejected_projection_is_restored_and_simulation_is_skipped(self) -> None:
        """A closed room collider cannot replace the valid placement checkpoint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))
            scene.room_geometry.wall_height = 3.0
            scene.room_geometry.floor = SceneObject(
                object_id=UniqueID("floor"),
                object_type=ObjectType.FLOOR,
                name="floor",
                description="Test floor",
                transform=RigidTransform(),
                bbox_min=np.array([-5.0, -5.0, -0.1]),
                bbox_max=np.array([5.0, 5.0, 0.0]),
                immutable=True,
            )
            for obj in scene.objects.values():
                obj.bbox_min = np.array([-0.25, -0.25, -0.25])
                obj.bbox_max = np.array([0.25, 0.25, 0.25])

            original = {
                object_id: obj.transform.translation().copy()
                for object_id, obj in scene.objects.items()
            }

            def eject_furniture(*, scene, **_kwargs):
                for obj in scene.objects.values():
                    obj.transform = RigidTransform(p=[0.0, 0.0, -20.0])
                return scene, True

            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=eject_furniture,
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation"
                ) as simulation,
            ):
                processed, success, removed = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

            self.assertFalse(success)
            self.assertEqual(removed, [])
            simulation.assert_not_called()
            for object_id, position in original.items():
                self.assertTrue(
                    np.allclose(
                        processed.get_object(object_id).transform.translation(),
                        position,
                    )
                )

    def test_isolated_ejected_projection_item_is_removed(self) -> None:
        """One ejected item cannot discard an otherwise valid dense layout."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))
            scene.room_geometry.wall_height = 3.0
            scene.room_geometry.floor = SceneObject(
                object_id=UniqueID("floor"),
                object_type=ObjectType.FLOOR,
                name="floor",
                description="Test floor",
                transform=RigidTransform(),
                bbox_min=np.array([-5.0, -5.0, -0.1]),
                bbox_max=np.array([5.0, 5.0, 0.0]),
                immutable=True,
            )
            for index in range(11):
                scene.add_object(
                    SceneObject(
                        object_id=UniqueID(f"extra_{index}"),
                        object_type=ObjectType.FURNITURE,
                        name="extra box",
                        description="Test furniture",
                        transform=RigidTransform(p=[0.0, 0.0, 0.25]),
                        bbox_min=np.array([-0.25, -0.25, -0.25]),
                        bbox_max=np.array([0.25, 0.25, 0.25]),
                    )
                )
            for obj in scene.objects.values():
                obj.bbox_min = np.array([-0.25, -0.25, -0.25])
                obj.bbox_max = np.array([0.25, 0.25, 0.25])

            def eject_one(*, scene, **_kwargs):
                scene.get_object(UniqueID("box_1")).transform = RigidTransform(
                    p=[0.0, 0.0, -20.0]
                )
                return scene, True

            with (
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_non_penetration_projection",
                    side_effect=eject_one,
                ),
                patch(
                    "scenesmith.agent_utils.physics.physical_feasibility."
                    "apply_forward_simulation",
                    side_effect=lambda *, scene, **_kwargs: (scene, []),
                ) as simulation,
            ):
                processed, success, removed = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    simulation_enabled=True,
                )

            self.assertTrue(success)
            self.assertEqual(removed, [UniqueID("box_1")])
            self.assertIsNone(processed.get_object(UniqueID("box_1")))
            self.assertIsNotNone(processed.get_object(UniqueID("box_2")))
            simulation.assert_called_once()

    def test_disabled_simulation(self) -> None:
        """Test that disabled simulation skips simulation stage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            try:
                processed_scene, _, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    projection_solver_name="ipopt",
                    projection_iteration_limit=100,
                    simulation_enabled=False,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            self.assertIsNotNone(processed_scene)

    def test_weld_furniture_in_pipeline(self) -> None:
        """Test weld_furniture flag in full pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_pos = furniture.transform.translation().copy()

            try:
                processed_scene, _, _ = apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=True,
                    projection_enabled=True,
                    projection_solver_name="ipopt",
                    projection_iteration_limit=1000,
                    simulation_enabled=True,
                    simulation_time_s=0.5,
                    simulation_time_step_s=1e-3,
                    simulation_timeout_s=30.0,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            furniture_after = processed_scene.get_object(UniqueID("table_0"))
            self.assertTrue(
                np.allclose(
                    furniture_after.transform.translation(),
                    initial_furniture_pos,
                    atol=1e-6,
                )
            )
