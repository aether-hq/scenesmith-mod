"""Unit tests for physical feasibility post-processing module."""

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.physics.feasibility.simulation import (
    apply_forward_simulation,
)
from scenesmith.agent_utils.physics.physical_feasibility import (
    apply_non_penetration_projection,
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


class TestApplyNonPenetrationProjection(PhysicalFeasibilityTestCase):
    """Tests for apply_non_penetration_projection function."""

    def test_overlapping_boxes_separated_snopt(self) -> None:
        """Test that overlapping boxes are separated by projection using SNOPT."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="snopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("SNOPT solver not available")
                raise

            self.assertTrue(success)

            box1_after = projected_scene.get_object(UniqueID("box_1"))
            box2_after = projected_scene.get_object(UniqueID("box_2"))

            pos1_changed = not np.allclose(
                box1_after.transform.translation(), initial_pos1, atol=0.01
            )
            pos2_changed = not np.allclose(
                box2_after.transform.translation(), initial_pos2, atol=0.01
            )
            self.assertTrue(pos1_changed or pos2_changed)

    def test_non_overlapping_boxes_unchanged(self) -> None:
        """Test that non-overlapping boxes remain roughly unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_pos1 = box1.transform.translation().copy()
            initial_pos2 = box2.transform.translation().copy()

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            self.assertTrue(success)

            box1_after = projected_scene.get_object(UniqueID("box_1"))
            box2_after = projected_scene.get_object(UniqueID("box_2"))

            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos1, atol=0.1)
            )
            self.assertTrue(
                np.allclose(box2_after.transform.translation(), initial_pos2, atol=0.1)
            )

    def test_fix_rotation_constraint(self) -> None:
        """Test that fix_rotation=True keeps rotations unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_rot1 = box1.transform.rotation().ToQuaternion().wxyz()
            initial_rot2 = box2.transform.rotation().ToQuaternion().wxyz()

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            if success:
                box1_after = projected_scene.get_object(UniqueID("box_1"))
                box2_after = projected_scene.get_object(UniqueID("box_2"))

                final_rot1 = box1_after.transform.rotation().ToQuaternion().wxyz()
                final_rot2 = box2_after.transform.rotation().ToQuaternion().wxyz()

                self.assertTrue(np.allclose(final_rot1, initial_rot1, atol=1e-3))
                self.assertTrue(np.allclose(final_rot2, initial_rot2, atol=1e-3))

    def test_xy_only_constraint(self) -> None:
        """Test that xy_only=True keeps Z position fixed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            box2 = scene.get_object(UniqueID("box_2"))
            initial_z1 = box1.transform.translation()[2]
            initial_z2 = box2.transform.translation()[2]

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=False,
                    xy_only=True,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            if success:
                box1_after = projected_scene.get_object(UniqueID("box_1"))
                box2_after = projected_scene.get_object(UniqueID("box_2"))

                self.assertTrue(
                    np.isclose(
                        box1_after.transform.translation()[2], initial_z1, atol=1e-3
                    )
                )
                self.assertTrue(
                    np.isclose(
                        box2_after.transform.translation()[2], initial_z2, atol=1e-3
                    )
                )

    def test_weld_furniture_keeps_furniture_fixed(self) -> None:
        """Test that weld_furniture=True keeps furniture fixed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_pos = furniture.transform.translation().copy()

            try:
                projected_scene, _ = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=1000,
                    weld_furniture=True,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            furniture_after = projected_scene.get_object(UniqueID("table_0"))
            self.assertTrue(
                np.allclose(
                    furniture_after.transform.translation(),
                    initial_furniture_pos,
                    atol=1e-6,
                )
            )

    def test_empty_scene_returns_success(self) -> None:
        """Test that empty scene returns success."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Empty test scene",
            )

            _, success = apply_non_penetration_projection(
                scene=scene,
                influence_distance=0.03,
                solver_name="snopt",
                iteration_limit=100,
                weld_furniture=False,
                xy_only=True,
                fix_rotation=True,
            )

            self.assertTrue(success)

    def test_stack_members_maintain_relative_positions(self) -> None:
        """Test that stack members maintain relative positions during projection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_stack(Path(tmp_dir))

            # Add overlapping box to force projection.
            box_sdf_path = TEST_DATA_DIR / "simple_box.sdf"
            overlapping_box = SceneObject(
                object_id=UniqueID("box_0"),
                object_type=ObjectType.FURNITURE,
                name="box",
                description="Overlapping box",
                transform=RigidTransform(p=[0.3, 0.0, 0.25]),
                sdf_path=box_sdf_path,
            )
            scene.add_object(overlapping_box)

            stack = scene.get_object(UniqueID("stack_0"))
            members_before = stack.metadata["member_assets"]
            initial_bottom_pos = np.array(members_before[0]["transform"]["translation"])
            initial_top_pos = np.array(members_before[1]["transform"]["translation"])
            initial_z_diff = initial_top_pos[2] - initial_bottom_pos[2]

            try:
                projected_scene, success = apply_non_penetration_projection(
                    scene=scene,
                    influence_distance=0.03,
                    solver_name="ipopt",
                    iteration_limit=5000,
                    weld_furniture=False,
                    xy_only=False,
                    fix_rotation=True,
                )
            except ValueError as e:
                if "not available" in str(e):
                    self.skipTest("IPOPT solver not available")
                raise

            self.assertTrue(success)

            stack_after = projected_scene.get_object(UniqueID("stack_0"))
            members_after = stack_after.metadata["member_assets"]
            final_bottom_pos = np.array(members_after[0]["transform"]["translation"])
            final_top_pos = np.array(members_after[1]["transform"]["translation"])
            final_z_diff = final_top_pos[2] - final_bottom_pos[2]

            # Verify stack actually moved to resolve collision.
            bottom_moved = not np.allclose(
                final_bottom_pos, initial_bottom_pos, atol=0.01
            )
            top_moved = not np.allclose(final_top_pos, initial_top_pos, atol=0.01)
            self.assertTrue(
                bottom_moved and top_moved,
                f"Stack should have moved to resolve collision. "
                f"Bottom moved: {bottom_moved}, Top moved: {top_moved}",
            )

            # Stack members should maintain their relative Z distance.
            self.assertAlmostEqual(initial_z_diff, final_z_diff, places=3)


class TestApplyForwardSimulation(PhysicalFeasibilityTestCase):
    """Tests for apply_forward_simulation function."""

    def test_persisted_upper_platform_supports_furniture(self) -> None:
        """Furniture remains on collision geometry paired with support metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_dir = Path(tmp_dir)
            platform_sidecar = scene_dir / "upper_floor.surfaces.json"
            platform_sdf = scene_dir / "upper_floor.sdf"
            platform_sidecar.write_text("{}", encoding="utf-8")
            platform_sdf.write_text(
                """<sdf version="1.7">
  <model name="upper_floor">
    <link name="structure_link">
      <visual name="visual">
        <pose>0 0 1 0 0 0</pose>
        <geometry><box><size>4 4 0.2</size></box></geometry>
      </visual>
      <collision name="collision">
        <pose>0 0 1 0 0 0</pose>
        <geometry><box><size>4 4 0.2</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>
""",
                encoding="utf-8",
            )
            room_geometry = RoomGeometry(
                sdf_tree=ET.parse(TEST_DATA_DIR / "simple_room_geometry.sdf"),
                sdf_path=TEST_DATA_DIR / "simple_room_geometry.sdf",
                additional_structural_surface_paths=[platform_sidecar],
            )
            scene = RoomScene(
                room_geometry=room_geometry,
                scene_dir=scene_dir,
                text_description="Upper platform support test",
            )
            scene.add_object(
                SceneObject(
                    object_id=UniqueID("upper_box"),
                    object_type=ObjectType.FURNITURE,
                    name="upper box",
                    description="Box on upper floor",
                    transform=RigidTransform(p=[0.0, 0.0, 1.35]),
                    sdf_path=TEST_DATA_DIR / "simple_box.sdf",
                )
            )

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.5,
                time_step_s=0.01,
                timeout_s=10.0,
                weld_furniture=False,
            )

            self.assertEqual(removed_ids, [])
            upper_box = simulated_scene.get_object(UniqueID("upper_box"))
            self.assertIsNotNone(upper_box)
            self.assertAlmostEqual(
                float(upper_box.transform.translation()[2]), 1.35, delta=0.03
            )

    def test_simulation_runs_without_error(self) -> None:
        """Test that simulation runs without errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.5,
                time_step_s=1e-3,
                timeout_s=30.0,
                weld_furniture=False,
            )

            self.assertIsNotNone(simulated_scene)
            self.assertEqual(removed_ids, [])

    def test_simulation_with_timeout(self) -> None:
        """Test that simulation respects timeout and returns scene unchanged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_non_overlapping_boxes_scene(Path(tmp_dir))

            box1 = scene.get_object(UniqueID("box_1"))
            initial_pos1 = box1.transform.translation().copy()

            simulated_scene, _ = apply_forward_simulation(
                scene=scene,
                simulation_time_s=10.0,
                time_step_s=1e-3,
                timeout_s=1e-16,
                weld_furniture=False,
            )

            box1_after = simulated_scene.get_object(UniqueID("box_1"))
            self.assertTrue(
                np.allclose(box1_after.transform.translation(), initial_pos1, atol=0.1)
            )

    def test_simulation_with_welded_furniture(self) -> None:
        """Test that welded furniture doesn't move during simulation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = self._create_scene_with_manipuland(Path(tmp_dir))

            furniture = scene.get_object(UniqueID("table_0"))
            initial_furniture_pos = furniture.transform.translation().copy()

            simulated_scene, _ = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.5,
                time_step_s=1e-3,
                timeout_s=30.0,
                weld_furniture=True,
            )

            furniture_after = simulated_scene.get_object(UniqueID("table_0"))
            self.assertTrue(
                np.allclose(
                    furniture_after.transform.translation(),
                    initial_furniture_pos,
                    atol=1e-6,
                )
            )

    def test_empty_scene_simulation(self) -> None:
        """Test that simulation on empty scene succeeds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene = RoomScene(
                room_geometry=self.room_geometry,
                scene_dir=Path(tmp_dir),
                text_description="Empty test scene",
            )

            simulated_scene, removed_ids = apply_forward_simulation(
                scene=scene,
                simulation_time_s=0.1,
                time_step_s=1e-3,
                timeout_s=10.0,
                weld_furniture=False,
            )

            self.assertIsNotNone(simulated_scene)
            self.assertEqual(removed_ids, [])
