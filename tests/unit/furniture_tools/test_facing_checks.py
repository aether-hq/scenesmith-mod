import json
import math
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

import numpy as np

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.furniture_agents.tools.scene_tools import SceneTools


class BaseAgentToolsTest(unittest.TestCase):
    """Base class for agent tools tests with common setup/teardown."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_scene(self, objects=None, description="Test scene"):
        """Create a standard mock scene for testing."""
        mock_scene = Mock(spec=RoomScene)
        mock_scene.objects = objects or {}
        mock_scene.text_description = description
        mock_scene.action_log_path = None
        return mock_scene

    def create_mock_scene_object(
        self,
        name: str,
        object_type: ObjectType,
        position: list[float] = None,
        rotation: list[float] = None,
        sdf_path: str = None,
        geometry_path: str = None,
    ):
        """Create a minimal mock scene object with transform data.

        Args:
            name: Object name (e.g., "Table")
            object_type: ObjectType enum value
            position: [x, y, z] coordinates (defaults to [0, 0, 0])
            rotation: [roll, pitch, yaw] angles in radians (defaults to [0, 0, 0])
            sdf_path: Optional SDF file path
            geometry_path: Optional geometry file path

        Returns:
            Mock object with real Drake transform for testing.
        """
        position = position or [0.0, 0.0, 0.0]
        rotation = rotation or [0.0, 0.0, 0.0]

        mock = Mock()
        mock.name = name
        mock.object_type = object_type
        mock.description = f"Test {name.lower()}"
        mock.sdf_path = Path(sdf_path) if sdf_path else None
        mock.geometry_path = Path(geometry_path) if geometry_path else None

        # Use real Drake RigidTransform for compatibility with SimplifiedFurnitureInfo.
        rpy = RollPitchYaw(rotation[0], rotation[1], rotation[2])
        mock.transform = RigidTransform(rpy, position)

        # Add bounding box fields.
        mock.bbox_min = None
        mock.bbox_max = None

        # Add immutable field (defaults to False for regular furniture).
        mock.immutable = False

        return mock


class TestFacingCheck(BaseAgentToolsTest):
    """Test facing check tool for spatial relationships between objects."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.mock_scene = self.create_mock_scene()

        # Mock get_object to return objects from the dict.
        def mock_get_object(obj_id):
            return self.mock_scene.objects.get(obj_id)

        self.mock_scene.get_object = mock_get_object

        # Load base configuration from actual config file.
        config_path = (
            Path(__file__).parents[3]
            / "configurations/furniture_agent/base_furniture_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)

        self.scene_tools = SceneTools(scene=self.mock_scene, cfg=self.cfg)

    def create_scene_object_with_bbox(
        self,
        name: str,
        position: list[float],
        yaw_degrees: float = 0.0,
        bbox_min: list[float] = None,
        bbox_max: list[float] = None,
    ) -> SceneObject:
        """Create a SceneObject with real transform and bounding box.

        Args:
            name: Object name.
            position: [x, y, z] position in world frame.
            yaw_degrees: Yaw rotation in degrees (around z-axis).
            bbox_min: Object-frame bbox minimum [x, y, z].
            bbox_max: Object-frame bbox maximum [x, y, z].

        Returns:
            SceneObject with real RigidTransform and bounding box.
        """
        # Default bounding box: 1m cube centered at origin.
        if bbox_min is None:
            bbox_min = [-0.5, -0.5, -0.5]
        if bbox_max is None:
            bbox_max = [0.5, 0.5, 0.5]

        # Create real RigidTransform.
        yaw_rad = math.radians(yaw_degrees)
        transform = RigidTransform(
            rpy=RollPitchYaw(roll=0.0, pitch=0.0, yaw=yaw_rad),
            p=position,
        )

        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.FURNITURE,
            name=name,
            description=f"Test {name}",
            transform=transform,
            bbox_min=np.array(bbox_min),
            bbox_max=np.array(bbox_max),
        )

    def test_facing_at_zero_degrees(self):
        """Test object A facing object B at 0° yaw (aligned with +y)."""
        # Object A at origin facing +y direction.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B directly in front of A (along +y).
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Chair should be facing table when aligned at 0°",
        )
        # Optimal rotation should be close to 0° (already facing).
        self.assertLess(
            abs(result["optimal_rotation_degrees"]),
            5.0,
            "Optimal rotation should be near 0° when already facing",
        )

    def test_not_facing_at_90_degrees(self):
        """Test object A not facing object B when rotated 90° away."""
        # Object A at origin rotated 90° (facing +x instead of +y).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=90.0,
        )
        # Object B still at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertFalse(
            result["is_facing"],
            "Chair rotated 90° should not be facing table at +y",
        )
        # Optimal rotation should be 0° (absolute rotation to face +y direction).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            0.0,
            delta=5.0,
            msg="Should need 0° absolute rotation to face table at +y",
        )

    def test_not_facing_at_180_degrees(self):
        """Test object A not facing object B when rotated 180° (facing away)."""
        # Object A facing -y direction (180° away from +y).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=180.0,
        )
        # Object B at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertFalse(
            result["is_facing"],
            "Chair rotated 180° should not be facing table",
        )
        # Optimal rotation should be 0° (absolute rotation to face +y direction).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            0.0,
            delta=5.0,
            msg="Should need 0° absolute rotation to face table at +y",
        )

    def test_facing_at_45_degrees(self):
        """Test object A partially facing object B at 45° angle."""
        # Object A rotated 45° from +y axis.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=45.0,
        )
        # Object B at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        # At 45° the ray might still intersect depending on bbox size.
        # The important part is the optimal rotation.
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            0.0,
            delta=5.0,
            msg="Should need 0° absolute rotation to optimally face table at +y",
        )

    def test_facing_with_arbitrary_positions(self):
        """Test facing check with objects at arbitrary positions."""
        # Object A at (1, 1, 0) facing +y.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[1.0, 1.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B at (1, 3, 0) - still along A's +y axis.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[1.0, 3.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertTrue(
            result["is_facing"],
            "Chair should be facing table when aligned along +y",
        )

    def test_facing_with_offset_target(self):
        """Test facing check when target is offset to the side."""
        # Object A at origin facing +y.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B offset to the side (+x direction).
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[2.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        # Should indicate absolute rotation needed to face the offset target.
        # Target at (2, 2) from origin requires -45° (45° clockwise) rotation.
        # At yaw=-45°, local +y points to world (sin(45°), cos(45°)) = northeast.
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            -45.0,
            delta=5.0,
            msg="Should need -45° absolute rotation to face offset target",
        )

    def test_object_without_bbox(self):
        """Test error handling when object lacks bounding box."""
        # Object A with bbox.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
        )
        # Object B without bbox.
        obj_b = SceneObject(
            object_id=UniqueID("table"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(),
            bbox_min=None,
            bbox_max=None,
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertFalse(result["success"], "Should fail when object lacks bbox")
        self.assertIn("bounding box", result["message"].lower())

    def test_invalid_object_id(self):
        """Test error handling when object ID doesn't exist."""
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id="nonexistent_id",
        )

        result = json.loads(result_str)

        self.assertFalse(result["success"], "Should fail with invalid object ID")
        self.assertIn("not found", result["message"].lower())

    def test_optimal_rotation_achieves_facing(self):
        """Test that applying optimal rotation results in facing relationship.

        This is a round-trip test that verifies the mathematical consistency
        of the facing check implementation. It ensures that:
        1. The optimal rotation is an absolute value (not a delta)
        2. Applying the optimal rotation achieves a facing relationship
        3. The optimal rotation remains consistent (doesn't change to ~0)
        """
        # Start with chair facing wrong direction (perpendicular).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=90.0,  # Facing +x instead of +y
        )
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],  # Directly in front (+y direction)
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        # First check - should NOT be facing.
        result1_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result1 = json.loads(result1_str)

        self.assertTrue(result1["success"])
        self.assertFalse(
            result1["is_facing"], "Chair at 90° should not be facing table at +y"
        )

        optimal_rotation = result1["optimal_rotation_degrees"]

        # Apply the optimal rotation (absolute value).
        obj_a_rotated = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=optimal_rotation,  # Use absolute value directly
        )

        # Update scene with rotated object.
        self.mock_scene.objects = {
            obj_a_rotated.object_id: obj_a_rotated,
            obj_b.object_id: obj_b,
        }

        # Second check - should NOW be facing.
        result2_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a_rotated.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result2 = json.loads(result2_str)

        self.assertTrue(result2["success"])
        self.assertTrue(
            result2["is_facing"],
            f"Chair at optimal rotation ({optimal_rotation:.1f}°) should be facing table",
        )

        # Since optimal_rotation is absolute, it should remain approximately
        # the same value (not become ~0).
        self.assertAlmostEqual(
            result2["optimal_rotation_degrees"],
            optimal_rotation,
            delta=5.0,
            msg="Optimal rotation should be consistent (absolute value, not delta)",
        )

    def test_facing_away_from_wall(self):
        """Test object facing away from wall with direction='away'."""
        # Furniture at origin rotated 180° (facing -y, back toward +y).
        obj_a = self.create_scene_object_with_bbox(
            name="desk",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=180.0,
        )
        # Wall at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="wall",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="away",
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Desk rotated 180° should be facing away from wall at +y",
        )
        # Optimal rotation should be close to 180° (already facing away).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            180.0,
            delta=5.0,
            msg="Optimal rotation should be near 180° when already facing away",
        )

    def test_not_facing_away(self):
        """Test object facing toward wall with direction='away' returns False."""
        # Furniture at origin rotated 0° (facing +y, front toward wall).
        obj_a = self.create_scene_object_with_bbox(
            name="desk",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Wall at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="wall",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="away",
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertFalse(
            result["is_facing"],
            "Desk rotated 0° should NOT be facing away from wall at +y",
        )
        # Optimal rotation should be 180° (absolute rotation to face away).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            180.0,
            delta=5.0,
            msg="Should need 180° absolute rotation to face away from wall at +y",
        )

    def test_facing_toward_with_explicit_direction(self):
        """Test explicit direction='toward' works same as default."""
        # Object A at origin facing +y.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="toward",
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertTrue(
            result["is_facing"],
            "Chair should be facing toward table when aligned at 0°",
        )

    def test_invalid_direction_parameter(self):
        """Test error handling for invalid direction parameter."""
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
        )
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="invalid_direction",
        )

        result = json.loads(result_str)

        self.assertFalse(result["success"], "Should fail with invalid direction")
        self.assertIn("Invalid direction parameter", result["message"])
        self.assertIn("toward", result["message"].lower())
        self.assertIn("away", result["message"].lower())

    def test_optimal_rotation_for_facing_away(self):
        """Test that optimal rotation for facing away is 180° offset."""
        # Object A at origin rotated 90° (facing +x).
        obj_a = self.create_scene_object_with_bbox(
            name="desk",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=90.0,
        )
        # Wall at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="wall",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        # Check facing toward (should suggest 0°).
        result_toward_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="toward",
        )

        result_toward = json.loads(result_toward_str)

        # Check facing away (should suggest 180°, which is 180° from toward).
        result_away_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="away",
        )

        result_away = json.loads(result_away_str)

        self.assertTrue(result_toward["success"])
        self.assertTrue(result_away["success"])

        # The difference should be approximately 180°.
        rotation_diff = abs(
            result_away["optimal_rotation_degrees"]
            - result_toward["optimal_rotation_degrees"]
        )
        # Account for wrapping (e.g., 170° to -170° is 340°, but should be 20°).
        if rotation_diff > 180.0:
            rotation_diff = 360.0 - rotation_diff

        self.assertAlmostEqual(
            rotation_diff,
            180.0,
            delta=5.0,
            msg="Optimal rotation for 'away' should be 180° from 'toward'",
        )

    def test_facing_with_z_height_difference(self):
        """Test facing check works with chair at different Z-height than table.

        Regression test for bug where 3D ray-AABB intersection failed when
        chair's center was at different Z-height than table's thin AABB.
        The fix uses 2D ray-rectangle intersection in XY plane.
        """
        # Chair at seat height (Z=0.5m).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.5],
            yaw_degrees=0.0,
            bbox_min=[-0.3, -0.3, -0.3],
            bbox_max=[0.3, 0.3, 0.3],
        )
        # Thin round table near ground (Z=0.05m, height=0.1m).
        obj_b = self.create_scene_object_with_bbox(
            name="table_round",
            position=[0.0, 1.5, 0.05],
            bbox_min=[-0.5, -0.5, -0.05],
            bbox_max=[0.5, 0.5, 0.05],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Chair at different Z-height should still be facing table in XY plane",
        )
        # Optimal rotation should be close to 0° (already aligned in XY plane).
        self.assertLess(
            abs(result["optimal_rotation_degrees"]),
            5.0,
            "Optimal rotation should be near 0° when aligned in XY plane",
        )

    def test_facing_with_small_misalignment_round_table(self):
        """Test chair close to round table with small misalignment and Z-height difference.

        Regression test for Z-height mismatch bug. The 2D ray-rectangle
        intersection ensures Z-height differences don't cause false negatives.
        """
        # Chair at seat height (Z=0.5m), positioned in front of table.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.8, 0.5],  # 0.8m in front (+Y), elevated
            yaw_degrees=2.0,  # Slightly misaligned (2° off from 0°)
            bbox_min=[-0.3, -0.3, -0.3],
            bbox_max=[0.3, 0.3, 0.3],
        )
        # Round table at ground level (different Z than chair).
        obj_b = self.create_scene_object_with_bbox(
            name="table_round",
            position=[0.0, 2.0, 0.0],  # 2m away from origin
            bbox_min=[-0.7, -0.7, -0.05],  # Large enough to catch the ray
            bbox_max=[0.7, 0.7, 0.05],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Chair with 2° misalignment at different Z-height should be "
            "facing table in XY plane (2D ray-rectangle intersection)",
        )
        # Optimal rotation should be close to 0° (facing +Y direction).
        self.assertLess(
            abs(result["optimal_rotation_degrees"]),
            5.0,
            "Optimal rotation should be near 0° for chair facing +Y toward table",
        )


if __name__ == "__main__":
    unittest.main()
