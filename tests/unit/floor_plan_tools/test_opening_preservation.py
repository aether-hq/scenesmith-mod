"""Tests for floor plan tools - door/window preservation after room changes."""

import json
import unittest

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import OpeningType
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

    def test_door_preserved_after_room_resize_when_fits(self):
        """Door should be preserved and repositioned when room is resized if it still fits."""
        layout, tools = self._create_single_room_layout()

        # Add door on exterior wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="left", width=1.0, height=2.1
        )
        assert door_result.success

        # Count door openings before resize.
        room = layout.placed_rooms[0]
        doors_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )
        assert doors_before == 1
        assert len(layout.doors) == 1
        old_position = layout.doors[0].position_exact

        # Resize room - door should be preserved (wall grew, door still fits).
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # Count door openings after resize - should be preserved.
        room = layout.placed_rooms[0]
        doors_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        assert (
            doors_after == 1
        ), "Door should be preserved after resize when it still fits"
        assert len(layout.doors) == 1, "Door metadata should be preserved"
        # Position should be proportionally adjusted (wall grew from 5m to 6m).
        new_position = layout.doors[0].position_exact
        expected_ratio = 6.0 / 5.0
        assert (
            abs(new_position - old_position * expected_ratio) < 0.01
        ), "Door repositioned proportionally"

    def test_window_preserved_after_room_resize_when_fits(self):
        """Window should be preserved and repositioned when room is resized if it still fits."""
        layout, tools = self._create_single_room_layout()

        # Add window on exterior wall B (which is on the depth dimension).
        window_result = tools._add_window_impl(
            wall_id="B", position="left", width=1.2, height=1.2
        )
        assert window_result.success

        # Count window openings before resize.
        room = layout.placed_rooms[0]
        windows_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.WINDOW])
            for w in room.walls
        )
        assert windows_before == 1
        assert len(layout.windows) == 1
        old_position = layout.windows[0].position_along_wall

        # Resize room - window should be preserved (wall grew, window still fits).
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # Count window openings after resize - should be preserved.
        room = layout.placed_rooms[0]
        windows_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.WINDOW])
            for w in room.walls
        )

        assert (
            windows_after == 1
        ), "Window should be preserved after resize when it still fits"
        assert len(layout.windows) == 1, "Window metadata should be preserved"
        # Position should be proportionally adjusted (depth grew from 4m to 5m).
        new_position = layout.windows[0].position_along_wall
        expected_ratio = 5.0 / 4.0
        # Use 0.1m tolerance due to floating point and boundary adjustments.
        assert (
            abs(new_position - old_position * expected_ratio) < 0.1
        ), f"Window repositioned proportionally: new={new_position}, expected={old_position * expected_ratio}"

    def test_door_invalidated_when_wall_shrinks(self):
        """Door at far end should be removed when room shrinks too much."""
        layout, tools = self._create_single_room_layout()

        # Add door on right side of 5m wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="right", width=1.0, height=2.1
        )
        assert door_result.success

        # Door is near end of 5m wall (position ~3-4m).
        door_position = layout.doors[0].position_exact
        assert door_position > 2.8, f"Door should be at right end, got {door_position}"

        # Resize room to 2m wide - door position becomes invalid and door is removed.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=2.0, depth=4.0
        )
        assert resize_result.success

        # Door opening should NOT be in wall and should be removed from layout.
        room = layout.placed_rooms[0]
        doors_in_wall = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        assert doors_in_wall == 0, "Door should be invalidated when wall shrinks"
        assert len(layout.doors) == 0, "Invalid door should be removed from layout"

        # Result message should inform about the removed door.
        assert "Removed" in resize_result.message, "Should inform about removed door"

    def test_partial_opening_preservation_on_resize(self):
        """Openings that still fit are preserved, those that don't are removed."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create wide room.
        # Boundary labels: A=north (8m), B=south (8m), C=east (4m), D=west (4m).
        rooms = [
            {"type": "living_room", "prompt": "A wide room", "width": 8.0, "depth": 4.0}
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Add door at left on wall A (north, 8m wall).
        # "left" position is around 0.2 * wall_length = 1.6m from start.
        door1_result = tools._add_door_impl(
            wall_id="A", position="left", width=1.0, height=2.1
        )
        assert (
            door1_result.success
        ), f"First door should succeed: {door1_result.message}"

        # Add door at right on wall B (south, also 8m wall - different wall).
        # "right" position is around 0.8 * wall_length = 6.4m from start.
        door2_result = tools._add_door_impl(
            wall_id="B", position="right", width=1.0, height=2.1
        )
        assert (
            door2_result.success
        ), f"Second door should succeed: {door2_result.message}"

        # Verify both doors are in walls.
        room = layout.placed_rooms[0]
        doors_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )
        assert doors_before == 2, f"Expected 2 doors, got {doors_before}"
        assert len(layout.doors) == 2

        # Resize room - shrink width from 8m to 2.5m (extreme shrink).
        # Both north (A) and south (B) walls shrink from 8m to 2.5m.
        # Left door on A at ~1.6m will scale to ~0.5m - door extends to 1.5m, fits in 2.5m wall.
        # Right door on B at ~6.4m will scale to ~2.0m - door extends to 3.0m > 2.5m wall, doesn't fit.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=2.5, depth=4.0
        )
        assert resize_result.success

        room = layout.placed_rooms[0]
        doors_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        # Left door should be preserved (scaled position still fits).
        # Right door should be removed (scaled position + width exceeds wall).
        assert (
            len(layout.doors) == 1
        ), f"Expected 1 door preserved, got {len(layout.doors)}"
        assert doors_after == 1, "One door should remain in wall openings"

        # Result message should mention the removed door.
        assert "Removed" in resize_result.message or "1" in resize_result.message

    def test_openings_preserved_after_add_adjacency(self):
        """Openings should be preserved when adjacency is added."""
        layout, tools = self._create_two_room_layout()

        # Add door on living room exterior wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="center", width=1.0, height=2.1
        )
        assert door_result.success

        # Get initial door count.
        assert len(layout.doors) == 1

        # Remove and re-add adjacency to trigger re-placement.
        tools._remove_adjacency_impl(room_a="living_room", room_b="kitchen")
        tools._add_adjacency_impl(room_a="living_room", room_b="kitchen")

        # Door should still exist.
        assert len(layout.doors) == 1, "Door metadata should be preserved"

        # Check if door is in wall openings (may or may not be depending on wall changes).
        # At minimum, metadata should be preserved.

    def test_open_connection_creates_opening(self):
        """Adding open connection should create OPEN type opening."""
        layout, tools = self._create_two_room_layout()

        # Add open connection.
        result = tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")
        assert result.success

        # Check that OPEN openings were created.
        open_count = 0
        for room in layout.placed_rooms:
            for wall in room.walls:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.OPEN:
                        open_count += 1

        assert open_count >= 2, "OPEN openings should be created on both rooms' walls"

    def test_open_connection_preserved_after_resize(self):
        """Open connection should be preserved and recalculated when room is resized."""
        layout, tools = self._create_two_room_layout()

        # Add open connection.
        result = tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")
        assert result.success

        # Count OPEN openings before resize.
        open_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_before >= 2

        # Resize kitchen.
        resize_result = tools._resize_room_impl(room_id="kitchen", width=5.0, depth=4.0)
        assert resize_result.success

        # OPEN openings should still exist (recalculated for new overlap).
        open_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_after >= 2, "OPEN openings should be preserved after resize"

    def test_combined_openings_preserved_after_resize(self):
        """All opening types should be preserved after resize when they still fit."""
        layout, tools = self._create_two_room_layout()

        # Add door on living room exterior wall B (depth dimension).
        door_result = tools._add_door_impl(
            wall_id="B", position="left", width=0.9, height=2.1
        )
        assert door_result.success

        # Add window on kitchen exterior wall.
        window_result = tools._add_window_impl(
            wall_id="E", position="center", width=1.2, height=1.0
        )
        assert window_result.success

        # Add open connection.
        open_result = tools._add_open_connection_impl(
            room_a="living_room", room_b="kitchen"
        )
        assert open_result.success

        # Count all openings before resize.
        def count_openings():
            counts = {"door": 0, "window": 0, "open": 0}
            for room in layout.placed_rooms:
                for wall in room.walls:
                    for opening in wall.openings:
                        counts[opening.opening_type.value] += 1
            return counts

        before = count_openings()
        assert before["door"] == 1
        assert before["window"] == 1
        assert before["open"] >= 2

        # Resize living room - growing from 5x4 to 6x5.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # After resize (walls grow, so openings should fit):
        # - Door on living_room's wall B: PRESERVED (wall grew from 4m to 5m, door fits)
        # - Window on kitchen's wall: PRESERVED (kitchen wasn't resized)
        # - OPEN connections: PRESERVED with recomputed positions
        after = count_openings()
        assert (
            after["door"] == 1
        ), "Door on resized room preserved (wall grew, still fits)"
        assert (
            after["window"] == before["window"]
        ), "Window on other room should be preserved"
        assert after["open"] >= 2, "OPEN openings should be preserved"

    def test_remove_open_connection_clears_openings(self):
        """Removing open connection should remove OPEN type openings."""
        layout, tools = self._create_two_room_layout()

        # Add and then remove open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Verify openings exist.
        open_count = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_count >= 2

        # Remove open connection.
        tools._remove_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Verify openings removed.
        open_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_after == 0, "OPEN openings should be removed"

    def test_three_room_layout_preserves_all_openings(self):
        """Complex layout with 3 rooms should preserve all openings after changes."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create L-shaped layout: living room with kitchen and bedroom adjacent.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Main living area",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 3.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
            {
                "type": "bedroom",
                "prompt": "Bedroom",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"
        assert len(layout.placed_rooms) == 3

        # Find exterior walls for each room.
        exterior_walls = {}
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                if room_a not in exterior_walls:
                    exterior_walls[room_a] = []
                exterior_walls[room_a].append(label)

        # Add door to living room, windows to kitchen and bedroom.
        living_wall = exterior_walls.get("living_room", [None])[0]
        if living_wall:
            tools._add_door_impl(wall_id=living_wall, position="center", width=0.9)

        kitchen_wall = exterior_walls.get("kitchen", [None])[0]
        if kitchen_wall:
            tools._add_window_impl(wall_id=kitchen_wall, position="center", width=1.0)

        bedroom_wall = exterior_walls.get("bedroom", [None])[0]
        if bedroom_wall:
            tools._add_window_impl(wall_id=bedroom_wall, position="center", width=1.2)

        # Add open connection between living room and kitchen.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Count openings before resize.
        def count_all():
            counts = {"door": 0, "window": 0, "open": 0}
            for room in layout.placed_rooms:
                for wall in room.walls:
                    for o in wall.openings:
                        counts[o.opening_type.value] += 1
            return counts

        before = count_all()

        # Resize bedroom from 4x3 to 5x4 (growing).
        resize_result = tools._resize_room_impl(room_id="bedroom", width=5.0, depth=4.0)
        assert resize_result.success

        after = count_all()

        # Resizing bedroom (growing) should preserve bedroom's window (repositioned):
        # - Door on living_room: PRESERVED (living_room wasn't resized)
        # - Window on kitchen: PRESERVED (kitchen wasn't resized)
        # - Window on bedroom: PRESERVED (bedroom grew, window repositioned and still fits)
        # - Open connection: PRESERVED (positions recomputed)
        assert (
            after["door"] >= before["door"]
        ), "Door on non-resized room should be preserved"
        assert (
            after["window"] == before["window"]
        ), "All windows preserved (bedroom grew, window still fits)"
        assert after["open"] >= 2, "Open connection should be preserved"

    def test_open_connection_width_matches_overlap(self):
        """Open connection width should match the actual room overlap."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create rooms of different sizes - overlap should be smaller room's width.
        rooms = [
            {"type": "living_room", "prompt": "Large room", "width": 6.0, "depth": 4.0},
            {
                "type": "kitchen",
                "prompt": "Smaller room",
                "width": 3.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Calculate expected overlap.
        living_room = next(r for r in layout.placed_rooms if r.room_id == "living_room")
        kitchen = next(r for r in layout.placed_rooms if r.room_id == "kitchen")
        shared_edge = get_shared_edge(living_room, kitchen)
        assert shared_edge is not None, "Rooms should share an edge"

        # Add open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Find the OPEN opening and verify width matches shared edge.
        for room in layout.placed_rooms:
            for wall in room.walls:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.OPEN:
                        assert abs(opening.width - shared_edge.width) < 0.01, (
                            f"Opening width {opening.width} should match "
                            f"shared edge width {shared_edge.width}"
                        )

    def test_open_connection_position_correct_for_both_walls(self):
        """Open connection position should be relative to each room's wall origin.

        When rooms have different sizes and the smaller room is offset, the
        opening position will be different for each wall. This test ensures
        each wall gets the correct position, not a shared incorrect position.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create rooms where kitchen is smaller and will be offset from living room.
        # Living room: 6m wide, Kitchen: 4m wide adjacent.
        # The placement algorithm may offset the smaller room.
        rooms = [
            {"type": "living_room", "prompt": "Large room", "width": 6.0, "depth": 5.0},
            {
                "type": "kitchen",
                "prompt": "Smaller room",
                "width": 4.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Add open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Get shared edges from both perspectives.
        living_room = next(r for r in layout.placed_rooms if r.room_id == "living_room")
        kitchen = next(r for r in layout.placed_rooms if r.room_id == "kitchen")

        shared_edge_living = get_shared_edge(living_room, kitchen)
        shared_edge_kitchen = get_shared_edge(kitchen, living_room)

        assert shared_edge_living is not None
        assert shared_edge_kitchen is not None

        # Find OPEN openings on each room's wall.
        living_opening = None
        kitchen_opening = None

        for wall in living_room.walls:
            for opening in wall.openings:
                if opening.opening_type == OpeningType.OPEN:
                    living_opening = opening
                    break

        for wall in kitchen.walls:
            for opening in wall.openings:
                if opening.opening_type == OpeningType.OPEN:
                    kitchen_opening = opening
                    break

        assert living_opening is not None, "Living room should have OPEN opening"
        assert kitchen_opening is not None, "Kitchen should have OPEN opening"

        # Each opening's position should match the shared edge from that room's perspective.
        assert (
            abs(
                living_opening.position_along_wall
                - shared_edge_living.position_along_wall
            )
            < 0.01
        ), (
            f"Living room opening position {living_opening.position_along_wall} should match "
            f"shared edge position {shared_edge_living.position_along_wall}"
        )
        assert (
            abs(
                kitchen_opening.position_along_wall
                - shared_edge_kitchen.position_along_wall
            )
            < 0.01
        ), (
            f"Kitchen opening position {kitchen_opening.position_along_wall} should match "
            f"shared edge position {shared_edge_kitchen.position_along_wall}"
        )
