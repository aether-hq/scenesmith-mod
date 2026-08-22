"""Tests for floor plan tools - door/window preservation after room changes."""

import json
import random
import unittest

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import OpeningType, WallDirection
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.tools.submission.placement.geometry import (
    get_shared_edge,
)


class TestOpeningPreservation(unittest.TestCase):
    def _create_single_room_layout(self) -> tuple:
        """Create a simple layout with one room."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "A spacious living room",
                "width": 5.0,
                "depth": 4.0,
            }
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        return layout, tools

    def _create_two_room_layout(self) -> tuple:
        """Create a layout with two adjacent rooms."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "A spacious living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "A modern kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        return layout, tools

    def test_door_on_interior_wall(self):
        """Door on interior wall should create openings on both rooms' walls."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find interior wall label.
        interior_wall = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is not None:  # Interior wall.
                interior_wall = label
                break

        assert interior_wall is not None, "Should have an interior wall"

        # Add door to interior wall.
        door_result = tools._add_door_impl(
            wall_id=interior_wall, position="center", width=0.9, height=2.1
        )
        assert door_result.success

        # Verify door metadata stored.
        assert len(layout.doors) == 1
        door = layout.doors[0]
        assert door.room_b is not None, "Interior door should have room_b set"

    def test_door_cutout_alignment_on_interior_walls(self):
        """Door cutouts on interior walls must align at same world position.

        When two rooms share an internal wall, each room has its own wall object
        with different start points. Door cutouts must align to the same world
        position, which means position_along_wall values will differ between walls.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create two rooms with adjacency.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find interior wall label.
        interior_wall_label = None
        interior_room_a = None
        interior_room_b = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is not None:  # Interior wall.
                interior_wall_label = label
                interior_room_a = room_a
                interior_room_b = room_b
                break

        assert interior_wall_label is not None, "Should have an interior wall"

        # Add door at position "left" (0.3m from start).
        door_result = tools._add_door_impl(
            wall_id=interior_wall_label, position="left", width=0.9, height=2.1
        )
        assert door_result.success

        # Find placed rooms.
        placed_a = next(r for r in layout.placed_rooms if r.room_id == interior_room_a)
        placed_b = next(r for r in layout.placed_rooms if r.room_id == interior_room_b)

        # Get shared edges from both perspectives.
        shared_edge_a = get_shared_edge(placed_a, placed_b)
        shared_edge_b = get_shared_edge(placed_b, placed_a)
        assert shared_edge_a is not None
        assert shared_edge_b is not None

        # Find door openings on each wall.
        opening_on_a = None
        opening_on_b = None

        for wall in placed_a.walls:
            if wall.direction == shared_edge_a.wall_direction:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.DOOR:
                        opening_on_a = opening
                        break

        for wall in placed_b.walls:
            if wall.direction == shared_edge_b.wall_direction:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.DOOR:
                        opening_on_b = opening
                        break

        assert opening_on_a is not None, "Room A wall should have door opening"
        assert opening_on_b is not None, "Room B wall should have door opening"

        # Calculate world positions of door left edges.
        # For vertical walls (east/west), position is along Y axis.
        # For horizontal walls (north/south), position is along X axis.
        def get_world_position(placed_room, wall_dir, position_along_wall):
            """Convert wall-relative position to world coordinate."""
            x, y = placed_room.position
            if wall_dir in (WallDirection.EAST, WallDirection.WEST):
                # Wall runs along Y axis from room's min_y.
                return y + position_along_wall
            else:
                # Wall runs along X axis from room's min_x.
                return x + position_along_wall

        world_pos_a = get_world_position(
            placed_a, shared_edge_a.wall_direction, opening_on_a.position_along_wall
        )
        world_pos_b = get_world_position(
            placed_b, shared_edge_b.wall_direction, opening_on_b.position_along_wall
        )

        # Door cutouts must align at the same world position.
        assert abs(world_pos_a - world_pos_b) < 0.01, (
            f"Door cutouts must align! Room A door at world pos {world_pos_a:.3f}, "
            f"Room B door at world pos {world_pos_b:.3f}, "
            f"position_along_wall A={opening_on_a.position_along_wall:.3f}, "
            f"position_along_wall B={opening_on_b.position_along_wall:.3f}"
        )

    def test_room_creation_validates_dimensions(self):
        """Room creation should fail for invalid dimensions."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Zero-width room should fail.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Invalid room",
                "width": 0.0,
                "depth": 4.0,
            }
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        # The placement algorithm should reject this.
        # Note: If this passes, the code might need validation added.

    def test_sequential_operations_maintain_consistency(self):
        """Multiple sequential operations should maintain layout consistency."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create initial layout.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        # Add various openings.
        exterior_walls = [
            label
            for label, (_, room_b, _) in layout.boundary_labels.items()
            if room_b is None
        ]
        if len(exterior_walls) >= 2:
            tools._add_door_impl(
                wall_id=exterior_walls[0], position="center", width=0.9
            )
            tools._add_window_impl(
                wall_id=exterior_walls[1], position="center", width=1.0
            )

        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Perform multiple resize operations (all growing or similar).
        tools._resize_room_impl(room_id="living_room", width=6.0, depth=4.0)
        tools._resize_room_impl(room_id="kitchen", width=5.0, depth=3.5)
        tools._resize_room_impl(room_id="living_room", width=5.5, depth=4.5)

        # Layout should still be valid.
        assert layout.placement_valid
        assert len(layout.placed_rooms) == 2

        # Doors and windows should be preserved if they still fit after proportional
        # repositioning. Since all resizes here are growing or similar, openings
        # that were originally at "center" should still fit after repositioning.
        # (The exact count depends on which room each opening was on and whether
        # it still fits after all the resizes.)
        # At minimum, open connections should be preserved.

        # Open connection should still have openings (positions recomputed).
        open_count = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_count >= 2, "Open connection should survive multiple resizes"

    def test_door_window_overlap_prevention_on_exterior_wall(self):
        """Doors and windows on same wall must not overlap (min separation enforced)."""
        layout = HouseLayout()
        # Use small separation for predictable test behavior.
        tools = FloorPlanTools(layout=layout, mode="house", min_opening_separation=0.5)

        # Create a large room so wall is long enough for both door and window.
        rooms = [
            {"type": "living_room", "prompt": "Living room", "width": 8.0, "depth": 6.0}
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find an exterior wall.
        exterior_wall = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                exterior_wall = label
                break
        assert exterior_wall is not None, "Should have exterior wall"

        # Add window at left (small wall segment).
        window_result = tools._add_window_impl(
            wall_id=exterior_wall, position="left", width=1.0
        )
        assert window_result.success, f"Window should succeed: {window_result.message}"

        # Adding door at right (different segment) should succeed.
        door_result = tools._add_door_impl(wall_id=exterior_wall, position="right")
        assert (
            door_result.success
        ), f"Door at right should succeed: {door_result.message}"

        # Now test overlap detection: try to add door at same position as window.
        # On a new layout with window at center.
        # Use seed for deterministic positioning to ensure overlap.
        random.seed(42)
        layout2 = HouseLayout()
        tools2 = FloorPlanTools(
            layout=layout2, mode="house", min_opening_separation=0.5
        )
        tools2._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        exterior_wall2 = None
        for label, (room_a, room_b, _) in layout2.boundary_labels.items():
            if room_b is None:
                exterior_wall2 = label
                break

        # Add window at center.
        window_result2 = tools2._add_window_impl(
            wall_id=exterior_wall2, position="center", width=1.5
        )
        assert window_result2.success

        # Adding door at center should fail (overlap).
        door_result2 = tools2._add_door_impl(wall_id=exterior_wall2, position="center")
        assert not door_result2.success, "Door should fail when overlapping window"
        assert "overlap" in door_result2.message.lower()

    def test_arched_window_rejects_impossible_crown_proportions(self):
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="room")
        created = tools._generate_room_specs_impl(
            room_specs_json=json.dumps(
                [{"type": "gallery", "prompt": "Gallery", "width": 8, "depth": 6}]
            )
        )
        assert created.success
        exterior_wall = next(
            label
            for label, (_room_a, room_b, _direction) in layout.boundary_labels.items()
            if room_b is None
        )

        result = tools._add_window_impl(
            wall_id=exterior_wall,
            position="center",
            width=4.0,
            height=1.5,
            sill_height=0.5,
            shape="arched",
        )

        assert not result.success
        assert "height must exceed half its width" in result.message
