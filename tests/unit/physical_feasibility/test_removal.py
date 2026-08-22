"""Unit tests for physical feasibility post-processing module."""

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix

from scenesmith.agent_utils.physics.feasibility.ik import compute_tilt_angle_degrees
from scenesmith.agent_utils.physics.feasibility.projection import (
    _apply_floor_penetration_fallback,
)
from scenesmith.agent_utils.physics.feasibility.simulation import (
    apply_forward_simulation,
)
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


class TestComputeTiltAngle(unittest.TestCase):
    """Tests for compute_tilt_angle_degrees function."""

    def test_upright_object_zero_tilt(self) -> None:
        """Test that an upright object has zero tilt angle."""
        transform = RigidTransform(p=[1.0, 2.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 0.0, places=5)

    def test_yaw_rotation_zero_tilt(self) -> None:
        """Test that yaw rotation (turning in place) gives zero tilt."""
        # Rotate 90 degrees around Z-axis (yaw).
        rotation = RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2))
        transform = RigidTransform(rotation, [1.0, 2.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 0.0, places=5)

    def test_45_degree_pitch_tilt(self) -> None:
        """Test that 45 degree pitch gives 45 degree tilt."""
        # Rotate 45 degrees around Y-axis (pitch).
        rotation = RotationMatrix(RollPitchYaw(0.0, np.pi / 4, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 45.0, places=3)

    def test_45_degree_roll_tilt(self) -> None:
        """Test that 45 degree roll gives 45 degree tilt."""
        # Rotate 45 degrees around X-axis (roll).
        rotation = RotationMatrix(RollPitchYaw(np.pi / 4, 0.0, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 45.0, places=3)

    def test_90_degree_tilt_horizontal(self) -> None:
        """Test that 90 degree pitch gives horizontal object (90 degree tilt)."""
        # Rotate 90 degrees around Y-axis.
        rotation = RotationMatrix(RollPitchYaw(0.0, np.pi / 2, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        self.assertAlmostEqual(tilt, 90.0, places=3)

    def test_combined_roll_pitch_tilt(self) -> None:
        """Test combined roll and pitch gives correct tilt angle."""
        # Small roll and pitch should give a combined tilt.
        rotation = RotationMatrix(RollPitchYaw(np.pi / 6, np.pi / 6, 0.0))
        transform = RigidTransform(rotation, [0.0, 0.0, 0.5])
        tilt = compute_tilt_angle_degrees(transform)
        # Combined tilt should be greater than either individual angle.
        self.assertGreater(tilt, 30.0)
        self.assertLess(tilt, 60.0)


class TestFallenFurnitureRemoval(PhysicalFeasibilityTestCase):
    """Tests for fallen furniture removal functionality."""

    def _create_tilted_furniture_scene(
        self, scene_dir: Path, tilt_degrees: float
    ) -> RoomScene:
        """Create a scene with a tilted furniture piece."""
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with tilted furniture",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        # Create a tilted box (tilt around Y-axis).
        tilt_rad = np.radians(tilt_degrees)
        rotation = RotationMatrix(RollPitchYaw(0.0, tilt_rad, 0.0))

        tilted_box = SceneObject(
            object_id=UniqueID("tilted_box"),
            object_type=ObjectType.FURNITURE,
            name="tilted_box",
            description="A tilted box",
            transform=RigidTransform(rotation, [0.0, 0.0, 0.5]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(tilted_box)
        return scene

    def test_fallen_furniture_removed_above_threshold(self) -> None:
        """Test that furniture tilted above threshold is removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with 50 degree tilt (above 45 degree threshold).
            scene = self._create_tilted_furniture_scene(Path(tmp_dir), tilt_degrees=50)

            # Verify object exists before.
            self.assertIsNotNone(scene.get_object(UniqueID("tilted_box")))

            # Run simulation with fallen furniture removal enabled.
            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
                remove_fallen_furniture=True,
                fallen_tilt_threshold_degrees=45.0,
            )

            # Object should be removed.
            self.assertEqual(len(removed_ids), 1)
            self.assertEqual(removed_ids[0], UniqueID("tilted_box"))
            self.assertIsNone(simulated_scene.get_object(UniqueID("tilted_box")))

    def test_upright_furniture_not_removed(self) -> None:
        """Test that upright furniture is not removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with slight tilt (below threshold).
            scene = self._create_tilted_furniture_scene(Path(tmp_dir), tilt_degrees=20)

            # Run simulation with fallen furniture removal enabled.
            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
                remove_fallen_furniture=True,
                fallen_tilt_threshold_degrees=45.0,
            )

            # Object should NOT be removed.
            self.assertEqual(len(removed_ids), 0)
            self.assertIsNotNone(simulated_scene.get_object(UniqueID("tilted_box")))

    def test_fallen_removal_disabled_by_default(self) -> None:
        """Test that fallen furniture removal is disabled by default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with 50 degree tilt.
            scene = self._create_tilted_furniture_scene(Path(tmp_dir), tilt_degrees=50)

            # Run simulation WITHOUT enabling fallen furniture removal.
            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
                # remove_fallen_furniture defaults to False
            )

            # Object should NOT be removed (feature disabled).
            self.assertEqual(len(removed_ids), 0)
            self.assertIsNotNone(simulated_scene.get_object(UniqueID("tilted_box")))


class TestFallenManipulandRemoval(PhysicalFeasibilityTestCase):
    """Tests for fallen manipuland removal functionality."""

    def _create_manipuland_scene(
        self,
        scene_dir: Path,
        manipuland_z: float,
        pre_sim_z: float | None = None,
        manipuland_xy: tuple[float, float] = (2.0, 0.0),
    ) -> RoomScene:
        """Create a scene with furniture and a manipuland at specified Z.

        Args:
            scene_dir: Directory for scene files.
            manipuland_z: Z position for the manipuland (post-simulation).
            pre_sim_z: If provided, the Z position before simulation (for z_delta).
                       If None, defaults to manipuland_z (no displacement).
            manipuland_xy: XY position for manipuland. Default (2, 0) places it
                away from the table at origin to allow free falling.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with manipuland",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        sphere_sdf_path = TEST_DATA_DIR / "simple_sphere.sdf"

        # Add furniture (will be welded during simulation).
        furniture = SceneObject(
            object_id=UniqueID("table_0"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(p=[0.0, 0.0, 0.25]),
            sdf_path=box_sdf_path,
        )

        # Add manipuland at specified Z and XY.
        # Use pre_sim_z for initial position if testing z_delta.
        initial_z = pre_sim_z if pre_sim_z is not None else manipuland_z
        manipuland = SceneObject(
            object_id=UniqueID("ball_0"),
            object_type=ObjectType.MANIPULAND,
            name="ball",
            description="Test ball",
            transform=RigidTransform(p=[manipuland_xy[0], manipuland_xy[1], initial_z]),
            sdf_path=sphere_sdf_path,
            bbox_min=np.array([-0.2, -0.2, -0.2]),
            bbox_max=np.array([0.2, 0.2, 0.2]),
        )

        scene.add_object(furniture)
        scene.add_object(manipuland)

        # If we need different post-sim Z, update transform after adding.
        # This simulates what happens during simulation.
        if pre_sim_z is not None and pre_sim_z != manipuland_z:
            manipuland.transform = RigidTransform(
                p=[manipuland_xy[0], manipuland_xy[1], manipuland_z]
            )

        return scene

    def test_floor_penetration_removed(self) -> None:
        """Test that manipuland below floor_z threshold is removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with manipuland at z=-1.0 (below -0.5 threshold).
            scene = self._create_manipuland_scene(
                Path(tmp_dir), manipuland_z=-1.0, pre_sim_z=0.7
            )

            self.assertIsNotNone(scene.get_object(UniqueID("ball_0")))

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.01,  # Short sim, object already positioned.
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=True,
                remove_fallen_manipulands=True,
                fallen_manipuland_floor_z=-0.5,
                fallen_manipuland_near_floor_z=0.02,
                fallen_manipuland_z_displacement=0.3,
            )

            # Object should be removed (fell through floor).
            self.assertIn(UniqueID("ball_0"), removed_ids)
            self.assertIsNone(simulated_scene.get_object(UniqueID("ball_0")))

    def test_fell_to_floor_removed(self) -> None:
        """Test that manipuland that fell to floor (big z_delta) is removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene with manipuland high in the air (z=0.7).
            # During simulation, it will fall to the floor (z~0).
            scene = self._create_manipuland_scene(
                Path(tmp_dir), manipuland_z=0.7, pre_sim_z=None
            )

            # Verify starting position.
            ball = scene.get_object(UniqueID("ball_0"))
            self.assertIsNotNone(ball)
            self.assertAlmostEqual(ball.transform.translation()[2], 0.7, places=2)

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=2.0,  # Enough time for object to fall.
                time_step_s=1e-3,
                timeout_s=30.0,
                weld_furniture=True,
                remove_fallen_manipulands=True,
                fallen_manipuland_floor_z=-0.5,
                fallen_manipuland_near_floor_z=0.1,  # Object on floor after falling.
                fallen_manipuland_z_displacement=0.3,  # Will have delta < -0.3.
            )

            # Object should be removed (fell from z=0.7 to floor).
            self.assertIn(UniqueID("ball_0"), removed_ids)
            self.assertIsNone(simulated_scene.get_object(UniqueID("ball_0")))

    def test_floor_placed_not_removed(self) -> None:
        """Test that floor-placed manipuland (no z_delta) is NOT removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create scene: started at z=0.05, still at z=0.05 (no displacement).
            scene = self._create_manipuland_scene(
                Path(tmp_dir), manipuland_z=0.05, pre_sim_z=0.05
            )

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.01,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=True,
                remove_fallen_manipulands=True,
                fallen_manipuland_floor_z=-0.5,
                fallen_manipuland_near_floor_z=0.1,  # On floor (bottom_z=0).
                fallen_manipuland_z_displacement=0.3,  # delta=0, not < -0.3.
            )

            # Object should NOT be removed (floor-placed intentionally).
            self.assertNotIn(UniqueID("ball_0"), removed_ids)
            self.assertIsNotNone(simulated_scene.get_object(UniqueID("ball_0")))


class TestApplyFloorPenetrationFallback(PhysicalFeasibilityTestCase):
    """Tests for _apply_floor_penetration_fallback function."""

    def _create_floor_penetrating_scene(
        self, scene_dir: Path, penetration_depth: float
    ) -> RoomScene:
        """Create a scene with furniture penetrating the floor.

        Args:
            scene_dir: Directory for scene files.
            penetration_depth: How far below Z=0 the bottom of the box should be.
                               Positive values mean penetration.
        """
        scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=scene_dir,
            text_description="Test scene with floor-penetrating furniture",
        )

        box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

        # Box is 0.5x0.5x0.5m. Center at z=0.25 places bottom at z=0 (on floor).
        # Center at z=(0.25 - penetration_depth) places bottom at z=-penetration_depth.
        box_z = 0.25 - penetration_depth

        penetrating_box = SceneObject(
            object_id=UniqueID("box_0"),
            object_type=ObjectType.FURNITURE,
            name="box",
            description="Floor-penetrating box",
            transform=RigidTransform(p=[0.0, 0.0, box_z]),
            sdf_path=box_sdf_path,
        )

        scene.add_object(penetrating_box)
        return scene

    def test_penetrating_furniture_lifted(self) -> None:
        """Test that furniture penetrating the floor is lifted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            penetration = 0.05  # 5cm penetration.
            scene = self._create_floor_penetrating_scene(
                Path(tmp_dir), penetration_depth=penetration
            )

            box = scene.get_object(UniqueID("box_0"))
            initial_z = box.transform.translation()[2]

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            self.assertEqual(lifted_count, 1)

            box_after = updated_scene.get_object(UniqueID("box_0"))
            final_z = box_after.transform.translation()[2]

            # Box should be lifted by at least the penetration depth.
            lift_amount = final_z - initial_z
            self.assertGreaterEqual(lift_amount, penetration)

    def test_non_penetrating_furniture_unchanged(self) -> None:
        """Test that furniture not penetrating the floor is unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_z1 = box1.transform.translation()[2]
            initial_z2 = box2.transform.translation()[2]

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            self.assertEqual(lifted_count, 0)

            box1_after = updated_scene.get_object(UniqueID("box_1"))
            box2_after = updated_scene.get_object(UniqueID("box_2"))

            self.assertAlmostEqual(
                box1_after.transform.translation()[2], initial_z1, places=6
            )
            self.assertAlmostEqual(
                box2_after.transform.translation()[2], initial_z2, places=6
            )

    def test_only_furniture_processed(self) -> None:
        """Test that only furniture objects are processed, not manipulands."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            # Move manipuland to penetrate floor (this should NOT be lifted).
            manipuland = scene.get_object(UniqueID("ball_0"))
            manipuland.transform = RigidTransform(p=[0.0, 0.0, -0.1])

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_z = furniture.transform.translation()[2]
            initial_manipuland_z = manipuland.transform.translation()[2]

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            # No furniture penetrating floor, so nothing lifted.
            self.assertEqual(lifted_count, 0)

            # Manipuland should be unchanged (function ignores non-furniture).
            manipuland_after = updated_scene.get_object(UniqueID("ball_0"))
            self.assertAlmostEqual(
                manipuland_after.transform.translation()[2],
                initial_manipuland_z,
                places=6,
            )

            # Furniture should also be unchanged.
            furniture_after = updated_scene.get_object(UniqueID("table_0"))
            self.assertAlmostEqual(
                furniture_after.transform.translation()[2],
                initial_furniture_z,
                places=6,
            )

    def test_empty_scene_returns_zero_lifted(self) -> None:
        """Test that empty scene returns zero lifted objects."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Empty test scene",
            )

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            self.assertEqual(lifted_count, 0)
            self.assertIsNotNone(updated_scene)

    def test_wall_penetration_ignored(self) -> None:
        """Test that furniture penetrating walls is NOT lifted (only floor matters)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Test scene with wall-penetrating furniture",
            )

            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"

            # Place box penetrating wall_1 (at y=5) but NOT the floor.
            # Box center at y=4.8 with size 0.5 means edge at y=5.05 (penetrating wall).
            wall_penetrating_box = SceneObject(
                object_id=UniqueID("box_0"),
                object_type=ObjectType.FURNITURE,
                name="box",
                description="Wall-penetrating box",
                transform=RigidTransform(p=[0.0, 4.8, 0.25]),  # On floor, near wall.
                sdf_path=box_sdf_path,
            )

            scene.add_object(wall_penetrating_box)

            box = scene.get_object(UniqueID("box_0"))
            initial_pos = box.transform.translation().copy()

            updated_scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )

            # Wall penetration should be ignored - nothing lifted.
            self.assertEqual(lifted_count, 0)

            # Position should be unchanged.
            box_after = updated_scene.get_object(UniqueID("box_0"))
            self.assertTrue(
                np.allclose(box_after.transform.translation(), initial_pos, atol=1e-6)
            )
