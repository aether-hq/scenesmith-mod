import unittest

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.physics.validation.collision_filtering import (
    _get_furniture_id_for_manipuland,
    filter_collisions_by_agent,
)
from scenesmith.agent_utils.physics.validation.models import CollisionPair
from scenesmith.agent_utils.physics.validation.scene_violations import (
    filter_door_violations_by_agent,
    filter_open_connection_violations_by_agent,
    filter_wall_height_violations_by_agent,
    filter_window_violations_by_agent,
)
from scenesmith.agent_utils.physics.validation.thin_coverings import (
    ThinCoveringBoundaryViolation,
    ThinCoveringOverlap,
    filter_thin_covering_boundary_violations_by_agent,
    filter_thin_covering_overlaps_by_agent,
)
from scenesmith.agent_utils.scene.clearance_zones import (
    DoorClearanceViolation,
    OpenConnectionBlockedViolation,
    WallHeightExceededViolation,
    WindowClearanceViolation,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    AgentType,
    ObjectType,
    PlacementInfo,
    SceneObject,
    SupportSurface,
    UniqueID,
)


class TestClearanceViolationFiltering(unittest.TestCase):
    """Test clearance violation filtering by agent type."""

    def setUp(self):
        """Set up test fixtures with mock scene containing different object types."""
        test_data_dir = Path(__file__).parents[2] / "test_data"
        self.scene = RoomScene(room_geometry=None, scene_dir=test_data_dir)

        # Use an existing SDF file for tests.
        sdf_path = test_data_dir / "test_box.sdf"

        # Add furniture object.
        self.furniture_id = UniqueID("sofa_12345678")
        furniture = SceneObject(
            object_id=self.furniture_id,
            object_type=ObjectType.FURNITURE,
            name="Sofa",
            description="Test sofa",
            transform=RigidTransform(np.array([1.0, 0.0, 0.0])),
            sdf_path=sdf_path,
        )
        self.scene.add_object(furniture)

        # Add ceiling-mounted object.
        self.ceiling_id = UniqueID("chandelier_87654321")
        ceiling_obj = SceneObject(
            object_id=self.ceiling_id,
            object_type=ObjectType.CEILING_MOUNTED,
            name="Chandelier",
            description="Test chandelier",
            transform=RigidTransform(np.array([2.0, 0.0, 2.5])),
            sdf_path=sdf_path,
        )
        self.scene.add_object(ceiling_obj)

        # Add wall-mounted object.
        self.wall_id = UniqueID("mirror_11112222")
        wall_obj = SceneObject(
            object_id=self.wall_id,
            object_type=ObjectType.WALL_MOUNTED,
            name="Mirror",
            description="Test mirror",
            transform=RigidTransform(np.array([0.0, 2.0, 1.5])),
            sdf_path=sdf_path,
        )
        self.scene.add_object(wall_obj)

    def test_filter_door_violations_furniture_agent(self):
        """FurnitureAgent sees only door violations from FURNITURE objects."""
        violations = [
            DoorClearanceViolation(
                furniture_id=str(self.furniture_id),
                door_label="door_1",
                penetration_depth=0.1,
            ),
            DoorClearanceViolation(
                furniture_id=str(self.ceiling_id),
                door_label="door_2",
                penetration_depth=0.2,
            ),
        ]

        filtered = filter_door_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].furniture_id, str(self.furniture_id))

    def test_filter_door_violations_ceiling_agent(self):
        """CeilingAgent sees only door violations from CEILING_MOUNTED objects."""
        violations = [
            DoorClearanceViolation(
                furniture_id=str(self.furniture_id),
                door_label="door_1",
                penetration_depth=0.1,
            ),
            DoorClearanceViolation(
                furniture_id=str(self.ceiling_id),
                door_label="door_2",
                penetration_depth=0.2,
            ),
        ]

        filtered = filter_door_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].furniture_id, str(self.ceiling_id))

    def test_filter_door_violations_floor_plan_agent_gets_none(self):
        """FloorPlanAgent sees no door violations (can't move objects)."""
        violations = [
            DoorClearanceViolation(
                furniture_id=str(self.furniture_id),
                door_label="door_1",
                penetration_depth=0.1,
            ),
        ]

        filtered = filter_door_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FLOOR_PLAN,
        )

        self.assertEqual(len(filtered), 0)

    def test_filter_window_violations_by_agent(self):
        """Window violations are filtered by object type."""
        violations = [
            WindowClearanceViolation(
                furniture_id=str(self.furniture_id),
                window_label="window_1",
                furniture_top_height=1.5,
                sill_height=1.0,
            ),
            WindowClearanceViolation(
                furniture_id=str(self.wall_id),
                window_label="window_2",
                furniture_top_height=2.0,
                sill_height=1.2,
            ),
        ]

        # FurnitureAgent sees furniture violations.
        furniture_filtered = filter_window_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].furniture_id, str(self.furniture_id))

        # WallAgent sees wall-mounted violations.
        wall_filtered = filter_window_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].furniture_id, str(self.wall_id))

    def test_filter_open_connection_violations_by_agent(self):
        """Open connection violations filter by any blocking furniture of agent type."""
        violations = [
            OpenConnectionBlockedViolation(
                opening_label="open_living_kitchen",
                blocking_furniture_ids=[str(self.furniture_id), str(self.ceiling_id)],
                required_passage_size=0.8,
            ),
            OpenConnectionBlockedViolation(
                opening_label="open_hallway",
                blocking_furniture_ids=[str(self.wall_id)],
                required_passage_size=0.8,
            ),
        ]

        # FurnitureAgent sees first violation (has furniture in blocking list).
        furniture_filtered = filter_open_connection_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].opening_label, "open_living_kitchen")

        # CeilingAgent also sees first violation (has ceiling object in list).
        ceiling_filtered = filter_open_connection_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 1)
        self.assertEqual(ceiling_filtered[0].opening_label, "open_living_kitchen")

        # WallAgent sees second violation (has wall-mounted object).
        wall_filtered = filter_open_connection_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].opening_label, "open_hallway")

    def test_filter_wall_height_violations_by_agent(self):
        """Wall height violations filter by object type matching agent."""
        violations = [
            WallHeightExceededViolation(
                object_id=str(self.furniture_id),
                object_top_height=3.2,
                wall_height=3.0,
            ),
            WallHeightExceededViolation(
                object_id=str(self.ceiling_id),
                object_top_height=3.5,
                wall_height=3.0,
            ),
            WallHeightExceededViolation(
                object_id=str(self.wall_id),
                object_top_height=3.1,
                wall_height=3.0,
            ),
        ]

        # FurnitureAgent sees only furniture height violation.
        furniture_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].object_id, str(self.furniture_id))

        # CeilingAgent sees only ceiling object height violation.
        ceiling_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 1)
        self.assertEqual(ceiling_filtered[0].object_id, str(self.ceiling_id))

        # WallAgent sees only wall-mounted object height violation.
        wall_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].object_id, str(self.wall_id))

        # FloorPlanAgent sees none (no object type).
        floor_plan_filtered = filter_wall_height_violations_by_agent(
            violations=violations,
            scene=self.scene,
            agent_type=AgentType.FLOOR_PLAN,
        )
        self.assertEqual(len(floor_plan_filtered), 0)

    def test_filter_collisions_by_agent(self):
        """Collisions filter to show only those involving agent's object type."""
        collisions = [
            CollisionPair(
                object_a_name="Sofa",
                object_a_id=str(self.furniture_id),
                object_b_name="Chandelier",
                object_b_id=str(self.ceiling_id),
                penetration_depth=0.05,
            ),
            CollisionPair(
                object_a_name="Mirror",
                object_a_id=str(self.wall_id),
                object_b_name="Chandelier",
                object_b_id=str(self.ceiling_id),
                penetration_depth=0.03,
            ),
        ]

        # FurnitureAgent sees first collision (involves furniture).
        furniture_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)
        self.assertEqual(furniture_filtered[0].object_a_id, str(self.furniture_id))

        # CeilingAgent sees both (chandelier in both).
        ceiling_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 2)

        # WallAgent sees second collision (involves mirror).
        wall_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 1)
        self.assertEqual(wall_filtered[0].object_a_id, str(self.wall_id))

        # FloorPlanAgent sees none.
        floor_plan_filtered = filter_collisions_by_agent(
            collisions=collisions,
            scene=self.scene,
            agent_type=AgentType.FLOOR_PLAN,
        )
        self.assertEqual(len(floor_plan_filtered), 0)

    def test_filter_thin_covering_overlaps_by_agent(self):
        """Thin covering overlaps filter by owner agent type."""
        # Both furniture_id objects are FURNITURE type, so both overlaps
        # are owned by FurnitureAgent.
        overlaps = [
            ThinCoveringOverlap(
                covering_a_name="Rug",
                covering_a_id=str(self.furniture_id),
                covering_b_name="Carpet",
                covering_b_id=str(self.furniture_id),
            ),
        ]

        # FurnitureAgent sees overlap (furniture-owned thin coverings).
        furniture_filtered = filter_thin_covering_overlaps_by_agent(
            overlaps=overlaps,
            scene=self.scene,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)

        # CeilingAgent sees none (no ceiling-owned thin coverings).
        ceiling_filtered = filter_thin_covering_overlaps_by_agent(
            overlaps=overlaps,
            scene=self.scene,
            agent_type=AgentType.CEILING_MOUNTED,
        )
        self.assertEqual(len(ceiling_filtered), 0)

        # WallAgent sees none (no wall-owned thin coverings).
        wall_filtered = filter_thin_covering_overlaps_by_agent(
            overlaps=overlaps,
            scene=self.scene,
            agent_type=AgentType.WALL_MOUNTED,
        )
        self.assertEqual(len(wall_filtered), 0)

    def test_filter_thin_covering_boundary_violations_by_agent(self):
        """Thin covering boundary violations only shown to FurnitureAgent."""
        violations = [
            ThinCoveringBoundaryViolation(
                covering_id="rug_123",
                exceeded_boundaries=["north", "east"],
            ),
        ]

        # FurnitureAgent sees floor covering boundary violations.
        furniture_filtered = filter_thin_covering_boundary_violations_by_agent(
            violations=violations,
            agent_type=AgentType.FURNITURE,
        )
        self.assertEqual(len(furniture_filtered), 1)

        # Other agents see none (only floor coverings have boundary constraints).
        for agent_type in [
            AgentType.CEILING_MOUNTED,
            AgentType.WALL_MOUNTED,
            AgentType.MANIPULAND,
            AgentType.FLOOR_PLAN,
        ]:
            filtered = filter_thin_covering_boundary_violations_by_agent(
                violations=violations,
                agent_type=agent_type,
            )
            self.assertEqual(len(filtered), 0)


class TestWallMountedParentLookup(unittest.TestCase):
    """Test that manipulands on wall-mounted objects are correctly handled."""

    def setUp(self):
        """Set up test fixtures with a wall-mounted object and manipuland."""
        test_data_dir = Path(__file__).parents[2] / "test_data"
        self.scene = RoomScene(room_geometry=None, scene_dir=test_data_dir)

        # Create a wall-mounted shelf with a support surface.
        self.wall_shelf_id = UniqueID("wall_shelf_0")
        self.surface_id = UniqueID("S_5")

        wall_shelf = SceneObject(
            object_id=self.wall_shelf_id,
            object_type=ObjectType.WALL_MOUNTED,
            name="Wall Shelf",
            description="A floating wall shelf",
            transform=RigidTransform(np.array([2.0, 0.0, 1.5])),
            sdf_path=Path("/fake/wall_shelf.sdf"),
            support_surfaces=[
                SupportSurface(
                    surface_id=self.surface_id,
                    bounding_box_min=np.array([-0.5, -0.12, 0.0]),
                    bounding_box_max=np.array([0.5, 0.12, 0.05]),
                    transform=RigidTransform(np.array([2.0, 0.0, 1.55])),
                )
            ],
        )
        self.scene.add_object(wall_shelf)

        # Create a manipuland placed on the wall shelf.
        self.manipuland_id = UniqueID("book_0")
        manipuland = SceneObject(
            object_id=self.manipuland_id,
            object_type=ObjectType.MANIPULAND,
            name="Book",
            description="A book on the shelf",
            transform=RigidTransform(np.array([2.0, 0.0, 1.6])),
            sdf_path=Path("/fake/book.sdf"),
            placement_info=PlacementInfo(
                parent_surface_id=self.surface_id,
                position_2d=np.array([0.0, 0.0]),
                rotation_2d=0.0,
            ),
        )
        self.scene.add_object(manipuland)

    def test_wall_mounted_parent_found(self):
        """Test that manipulands on wall-mounted objects find their parent.

        This is a regression test for the bug where _get_furniture_id_for_manipuland
        only checked for FURNITURE and FLOOR types, missing WALL_MOUNTED.
        """
        parent_id = _get_furniture_id_for_manipuland(
            manipuland_id=str(self.manipuland_id),
            scene=self.scene,
        )

        self.assertEqual(
            parent_id,
            str(self.wall_shelf_id),
            "Manipuland on wall-mounted object should find its parent",
        )

    def test_manipuland_without_placement_info_returns_none(self):
        """Test that manipulands without placement_info return None."""
        # Create manipuland without placement_info.
        orphan_id = UniqueID("orphan_book")
        orphan = SceneObject(
            object_id=orphan_id,
            object_type=ObjectType.MANIPULAND,
            name="Orphan Book",
            description="A book without placement info",
            transform=RigidTransform(np.array([0.0, 0.0, 0.0])),
            sdf_path=Path("/fake/book.sdf"),
        )
        self.scene.add_object(orphan)

        parent_id = _get_furniture_id_for_manipuland(
            manipuland_id=str(orphan_id),
            scene=self.scene,
        )

        self.assertIsNone(parent_id)


class TestFloorManipulandParentLookup(unittest.TestCase):
    """Test that manipulands on the floor find their parent via room_geometry.floor."""

    def setUp(self):
        """Set up test fixtures with a floor and manipuland."""
        from unittest.mock import Mock

        test_data_dir = Path(__file__).parents[2] / "test_data"

        # Create mock room_geometry with a floor object.
        self.floor_id = UniqueID("floor_bedroom")
        self.floor_surface_id = "S_floor"

        floor_obj = SceneObject(
            object_id=self.floor_id,
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Floor surface",
            transform=RigidTransform(),
            sdf_path=None,
            support_surfaces=[
                SupportSurface(
                    surface_id=self.floor_surface_id,
                    bounding_box_min=np.array([-2.0, -2.0, 0.0]),
                    bounding_box_max=np.array([2.0, 2.0, 0.1]),
                    transform=RigidTransform(np.array([0.0, 0.0, 0.01])),
                )
            ],
        )

        room_geometry = Mock()
        room_geometry.floor = floor_obj

        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Create a manipuland placed on the floor.
        self.manipuland_id = UniqueID("backpack_0")
        manipuland = SceneObject(
            object_id=self.manipuland_id,
            object_type=ObjectType.MANIPULAND,
            name="Backpack",
            description="A backpack on the floor",
            transform=RigidTransform(np.array([1.0, 0.5, 0.1])),
            sdf_path=Path("/fake/backpack.sdf"),
            placement_info=PlacementInfo(
                parent_surface_id=self.floor_surface_id,
                position_2d=np.array([1.0, 0.5]),
                rotation_2d=0.0,
            ),
        )
        self.scene.add_object(manipuland)

    def test_floor_manipuland_parent_found_via_room_geometry(self):
        """Test that manipulands on floor find their parent via room_geometry.floor.

        This is a regression test for the bug where _get_furniture_id_for_manipuland
        only searched scene.objects but the floor is in room_geometry.floor.
        """
        parent_id = _get_furniture_id_for_manipuland(
            manipuland_id=str(self.manipuland_id),
            scene=self.scene,
        )

        self.assertEqual(
            parent_id,
            str(self.floor_id),
            "Manipuland on floor should find its parent via room_geometry.floor",
        )


if __name__ == "__main__":
    unittest.main()
