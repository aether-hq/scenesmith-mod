import math
import unittest

from pathlib import Path

import lxml.etree as ET
import numpy as np

from pydrake.all import RigidTransform
from pydrake.math import RollPitchYaw

from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.physics.validation.thin_coverings import (
    ThinCoveringBoundaryViolation,
    compute_thin_covering_boundary_violations,
)
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.manipuland_agents.tools.physics_utils import (
    load_collision_bounds_for_scene_object,
)
from scenesmith.manipuland_agents.tools.stacking_tools.stacking import (
    compute_initial_stack_transforms,
    simulate_stack_stability,
)
from scenesmith.utils.geometry.sdf_utils import serialize_rigid_transform


class TestCollisionFiltering(unittest.TestCase):
    """Test collision filtering for false positives."""

    def setUp(self):
        """Set up test fixtures with floor plan that has walls."""
        test_data_dir = Path(__file__).parents[2] / "test_data"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        self.box_sdf_path = test_data_dir / "simple_box.sdf"

        # Create room geometry with walls.
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

    def test_wall_to_wall_collisions_filtered(self):
        """Test that wall-to-wall collisions are filtered out.

        Adjacent walls in a floor plan naturally intersect at corners.
        These should not be reported as collisions.
        """
        # Test with just the floor plan (no furniture).
        collisions = compute_scene_collisions(self.scene)

        # Filter for wall-to-wall collisions.
        wall_collisions = [
            c
            for c in collisions
            if ("wall" in c.object_a_name.lower() and "wall" in c.object_b_name.lower())
        ]

        # Should not detect wall-to-wall collisions.
        self.assertEqual(
            len(wall_collisions),
            0,
            f"Wall-to-wall collisions should be filtered out, but found: {wall_collisions}",
        )

    def test_self_collisions_filtered(self):
        """Test that self-collisions are filtered out.

        Objects with multiple collision geometries should not report
        collisions between their own geometries.
        """
        # Add a furniture object with multiple collision geometries.
        multi_collision_sdf_path = (
            Path(__file__).parents[2] / "test_data" / "multi_collision_object.sdf"
        )

        chair = SceneObject(
            object_id=UniqueID("office_chair"),
            object_type=ObjectType.FURNITURE,
            name="Office Chair",
            description="Chair with multiple collision geometries",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=multi_collision_sdf_path,
        )
        self.scene.add_object(chair)

        collisions = compute_scene_collisions(self.scene)

        # Filter for self-collisions (same object ID on both sides).
        self_collisions = [
            c
            for c in collisions
            if c.object_a_id == c.object_b_id and c.object_a_id != "room_geometry"
        ]

        # Should not detect self-collisions.
        self.assertEqual(
            len(self_collisions),
            0,
            f"Self-collisions should be filtered out, but found: {self_collisions}",
        )

    def test_legitimate_furniture_collisions_preserved(self):
        """Test that legitimate furniture-to-furniture collisions are still detected.

        Real collisions between different furniture objects should not be filtered.
        """
        # Add two overlapping furniture objects.
        desk1 = SceneObject(
            object_id=UniqueID("desk1"),
            object_type=ObjectType.FURNITURE,
            name="Modern Office Desk",
            description="First desk",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=self.box_sdf_path,
        )
        desk2 = SceneObject(
            object_id=UniqueID("desk2"),
            object_type=ObjectType.FURNITURE,
            name="Modern Office Desk",
            description="Second desk",
            transform=RigidTransform(np.array([0.3, 0.0, 0.5])),  # Overlapping
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(desk1)
        self.scene.add_object(desk2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for desk-to-desk collisions.
        desk_collisions = [
            c
            for c in collisions
            if (
                "Modern Office Desk" in c.object_a_name
                and "Modern Office Desk" in c.object_b_name
            )
            and c.object_a_id != c.object_b_id  # Different objects
        ]

        # Should detect the legitimate collision.
        self.assertGreater(
            len(desk_collisions),
            0,
            "Legitimate furniture-to-furniture collisions should be preserved",
        )

        # Verify penetration depth is reasonable.
        if desk_collisions:
            penetration = desk_collisions[0].penetration_depth
            self.assertGreater(penetration, 0.1, "Should detect significant overlap")


class TestStackCollisionFiltering(unittest.TestCase):
    """Test that intra-stack collisions are correctly filtered."""

    def setUp(self):
        """Set up test fixtures with real stacking assets."""
        test_data_dir = Path(__file__).parents[2] / "test_data"
        stacking_assets_dir = test_data_dir / "stacking_assets"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        self.bread_plate_sdf = stacking_assets_dir / "bread_plate" / "bread_plate_2.sdf"

        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

    def test_no_intra_stack_collisions_reported(self):
        """Test that collisions between members of the same stack are NOT reported.

        This tests the fix for the bug where XY proximity matching incorrectly
        mapped stack members to different parent stacks, causing false collision
        reports between members of the same physical stack.
        """
        # Skip if test asset not available.
        if not self.bread_plate_sdf.exists():
            self.skipTest(f"Test asset not found: {self.bread_plate_sdf}")

        # Create stack members as SceneObjects for simulation.
        members = []
        for i in range(3):
            member = SceneObject(
                object_id=UniqueID(f"plate_{i:08x}"),
                object_type=ObjectType.MANIPULAND,
                name=f"Bread Plate",
                description="Test plate",
                transform=RigidTransform(),
                sdf_path=self.bread_plate_sdf,
            )
            members.append(member)

        # Get collision bounds and compute stack transforms.
        bounds_list = [load_collision_bounds_for_scene_object(m) for m in members]
        base_transform = RigidTransform(np.array([0.0, 0.0, 0.0]))
        initial_transforms = compute_initial_stack_transforms(
            bounds_list, base_transform
        )

        # Simulate to get final transforms.
        sim_result = simulate_stack_stability(
            scene_objects=members,
            initial_transforms=initial_transforms,
            ground_xyz=(0.0, 0.0, 0.0),
            simulation_time=1.0,
            simulation_time_step=0.001,
            position_threshold=0.1,
        )
        self.assertTrue(sim_result.is_stable, "Stack should be stable")

        # Build member_assets metadata (same structure as manipuland_tools.py).
        member_assets = []
        for i, (member, final_transform) in enumerate(
            zip(members, sim_result.final_transforms)
        ):
            member_assets.append(
                {
                    "asset_id": str(member.object_id),
                    "name": member.name,
                    "transform": serialize_rigid_transform(final_transform),
                    "sdf_path": str(member.sdf_path.absolute()),
                    "geometry_path": str(member.sdf_path.absolute()),
                }
            )

        # Create the stack scene object with proper metadata structure.
        # Use realistic stack ID like "stack_1" (real stacks use incrementing counters).
        # This gives suffix "1" (1 char), not "test" (4 chars).
        stack = SceneObject(
            object_id=UniqueID("stack_1"),
            object_type=ObjectType.MANIPULAND,
            name="stack_3",
            description="Stack of 3 plates",
            transform=sim_result.final_transforms[0],
            sdf_path=self.bread_plate_sdf,  # Not used for stacks.
            metadata={
                "composite_type": "stack",
                "member_assets": member_assets,
                "num_members": len(members),
            },
        )
        self.scene.add_object(stack)

        # Run collision detection.
        collisions = compute_scene_collisions(self.scene)

        # Filter for collisions involving the stack.
        stack_collisions = [
            c
            for c in collisions
            if "stack" in c.object_a_id.lower() or "stack" in c.object_b_id.lower()
        ]

        # Filter for intra-stack collisions (same stack ID on both sides).
        intra_stack_collisions = [
            c for c in stack_collisions if c.object_a_id == c.object_b_id
        ]

        self.assertEqual(
            len(intra_stack_collisions),
            0,
            f"Should not report intra-stack collisions, but found: {intra_stack_collisions}",
        )


class TestThinCoveringBoundaryViolation(unittest.TestCase):
    """Test ThinCoveringBoundaryViolation dataclass."""

    def test_to_description_single_boundary(self):
        """Test description formatting for single boundary violation."""
        violation = ThinCoveringBoundaryViolation(
            covering_id="rug_12345678",
            exceeded_boundaries=["west"],
        )
        expected = "Thin covering [rug_12345678] extends beyond west boundary"
        self.assertEqual(violation.to_description(), expected)

    def test_to_description_multiple_boundaries(self):
        """Test description formatting for multiple boundary violations."""
        violation = ThinCoveringBoundaryViolation(
            covering_id="rug_87654321",
            exceeded_boundaries=["east", "north"],
        )
        expected = "Thin covering [rug_87654321] extends beyond east, north boundaries"
        self.assertEqual(violation.to_description(), expected)


class TestComputeThinCoveringBoundaryViolations(unittest.TestCase):
    """Test compute_thin_covering_boundary_violations function."""

    def setUp(self):
        """Set up test fixtures with a room geometry."""
        test_data_dir = Path(__file__).parents[2] / "test_data"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"

        # Create room geometry with 5m x 5m room.
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
            length=5.0,  # x-dimension
            width=5.0,  # y-dimension
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)
        self.wall_thickness = 0.05  # 5cm walls

    def _create_thin_covering(
        self,
        object_id: str,
        x: float,
        y: float,
        width_m: float,
        depth_m: float,
        shape: str = "rectangular",
        yaw: float = 0.0,
    ) -> SceneObject:
        """Helper to create a thin covering SceneObject.

        Uses FURNITURE type with asset_source="thin_covering" metadata,
        matching how thin coverings are created in production.
        """
        transform = RigidTransform(
            RollPitchYaw(0, 0, yaw).ToRotationMatrix(), np.array([x, y, 0.01])
        )
        return SceneObject(
            object_id=UniqueID(object_id),
            object_type=ObjectType.FURNITURE,  # Thin coverings keep agent's type.
            name=f"Test Rug {object_id}",
            description="Test rug",
            transform=transform,
            sdf_path=Path(
                "/fake/path.sdf"
            ),  # Not used for thin covering boundary check.
            metadata={
                "asset_source": "thin_covering",  # Identified via metadata.
                "width_m": width_m,
                "depth_m": depth_m,
                "shape": shape,
            },
        )

    def test_thin_covering_within_bounds_no_violation(self):
        """Test thin covering fully within bounds reports no violation."""
        # Room is 5m x 5m, wall takes 0.025m on each side, so usable is ~4.95m x 4.95m.
        # Center a 2m x 2m thin covering at origin - should be well within bounds.
        covering = self._create_thin_covering(
            object_id="rug_00000001",
            x=0.0,
            y=0.0,
            width_m=2.0,
            depth_m=2.0,
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(
            len(violations), 0, "Thin covering within bounds should have no violations"
        )

    def test_thin_covering_exceeds_west_boundary(self):
        """Test thin covering extending beyond west boundary."""
        # Room x ranges from -2.5 to 2.5. With wall_thickness=0.05, inner bounds
        # are -2.475 to 2.475. Place a 2m wide thin covering centered at x=-2.0.
        # Left edge at x=-3.0 < -2.475 -> west violation.
        covering = self._create_thin_covering(
            object_id="rug_00000002",
            x=-2.0,
            y=0.0,
            width_m=2.0,
            depth_m=1.0,
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].covering_id, "rug_00000002")
        self.assertIn("west", violations[0].exceeded_boundaries)

    def test_thin_covering_exceeds_multiple_boundaries(self):
        """Test thin covering at corner extending beyond two boundaries."""
        # Place a 2m x 2m thin covering at corner position where it exceeds NE corner.
        # With inner bounds at ±2.475, a 2m thin covering centered at (2.0, 2.0)
        # has edges at x=3.0 > 2.475 and y=3.0 > 2.475.
        covering = self._create_thin_covering(
            object_id="rug_00000003",
            x=2.0,
            y=2.0,
            width_m=2.0,
            depth_m=2.0,
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].covering_id, "rug_00000003")
        # Should have both east and north (sorted alphabetically).
        self.assertIn("east", violations[0].exceeded_boundaries)
        self.assertIn("north", violations[0].exceeded_boundaries)

    def test_circular_thin_covering_within_bounds(self):
        """Test circular thin covering within bounds."""
        covering = self._create_thin_covering(
            object_id="rug_circular_01",
            x=0.0,
            y=0.0,
            width_m=2.0,
            depth_m=2.0,
            shape="circular",
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 0)

    def test_circular_thin_covering_exceeds_boundary(self):
        """Test circular thin covering extending beyond boundary."""
        # Circular thin covering with radius=1.0, centered at x=2.0.
        # Right edge at x=3.0 > 2.475 -> east violation.
        covering = self._create_thin_covering(
            object_id="rug_circular_02",
            x=2.0,
            y=0.0,
            width_m=2.0,  # radius = 1.0
            depth_m=2.0,
            shape="circular",
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        self.assertEqual(len(violations), 1)
        self.assertIn("east", violations[0].exceeded_boundaries)

    def test_rotated_thin_covering_boundary_check(self):
        """Test rotated rectangular thin covering boundary check using OBB corners."""
        # A 3m x 1m thin covering rotated 45 degrees at origin.
        # After rotation, corners extend further than the unrotated extents.
        # At 45 degrees, a 3x1 thin covering's effective bounding box is approximately:
        # diagonal extent = sqrt((1.5)^2 + (0.5)^2) ≈ 1.58m from center.
        # Place at (1.5, 1.5) with 45 degree rotation.
        covering = self._create_thin_covering(
            object_id="rug_rotated_01",
            x=1.5,
            y=1.5,
            width_m=3.0,
            depth_m=1.0,
            yaw=math.pi / 4,  # 45 degrees
        )
        self.scene.add_object(covering)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )

        # The rotated thin covering extends to ~3.08 from center along the diagonal.
        # From (1.5, 1.5), corners reach beyond 2.475 on north/east.
        self.assertEqual(len(violations), 1)
        self.assertTrue(
            len(violations[0].exceeded_boundaries) >= 1,
            "Rotated thin covering should exceed at least one boundary",
        )

    def test_non_thin_covering_objects_ignored(self):
        """Test that objects without asset_source=thin_covering are ignored."""
        # Create a furniture object (not a thin covering).
        furniture = SceneObject(
            object_id=UniqueID("sofa_001"),
            object_type=ObjectType.FURNITURE,
            name="Sofa",
            description="A sofa",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=Path("/fake/path.sdf"),
            metadata={},  # No asset_source metadata.
        )
        self.scene.add_object(furniture)

        violations = compute_thin_covering_boundary_violations(
            scene=self.scene,
            wall_thickness=self.wall_thickness,
        )
        # Furniture without thin_covering metadata should be ignored.
        self.assertEqual(len(violations), 0)
