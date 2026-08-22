"""Tests for floor plan tools - door/window preservation after room changes."""

import json
import unittest

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools


class TestLayoutCheckpointRestore(unittest.TestCase):
    """Test HouseLayout checkpoint/restore for reset functionality."""

    def test_layout_round_trip_preserves_all_state(self):
        """HouseLayout.from_dict(layout.to_dict()) should preserve all state.

        This test ensures the checkpoint/reset mechanism works correctly.
        If this test fails, _perform_checkpoint_reset would restore corrupted state.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create a complex layout with rooms, adjacencies, open connections.
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
                "connections": {"living_room": "OPEN"},
            },
            {
                "type": "bedroom",
                "prompt": "Bedroom",
                "width": 4.0,
                "depth": 3.5,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        # Add doors and windows.
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                tools._add_window_impl(wall_id=label, position="center", width=1.2)
            elif room_b == "bedroom":  # Interior door to bedroom.
                tools._add_door_impl(wall_id=label, position="center")

        # Capture state before serialization.
        original_room_ids = [s.room_id for s in layout.room_specs]
        original_door_count = len(layout.doors)
        original_window_count = len(layout.windows)
        original_placed_room_count = len(layout.placed_rooms)
        original_connections = layout.room_specs[
            1
        ].connections  # Kitchen's connections.

        # Serialize and restore.
        state_dict = layout.to_dict()
        restored = HouseLayout.from_dict(state_dict)

        # Verify all state was preserved.
        restored_room_ids = [s.room_id for s in restored.room_specs]
        assert restored_room_ids == original_room_ids, "Room IDs should match"

        assert (
            len(restored.doors) == original_door_count
        ), f"Door count should match: {len(restored.doors)} vs {original_door_count}"
        assert (
            len(restored.windows) == original_window_count
        ), f"Window count should match: {len(restored.windows)} vs {original_window_count}"
        assert (
            len(restored.placed_rooms) == original_placed_room_count
        ), f"Placed room count should match: {len(restored.placed_rooms)} vs {original_placed_room_count}"

        # Verify connections preserved.
        restored_kitchen = next(
            s for s in restored.room_specs if s.room_id == "kitchen"
        )
        assert (
            restored_kitchen.connections == original_connections
        ), f"connections should match: {restored_kitchen.connections} vs {original_connections}"

        # Verify placement_valid flag.
        assert restored.placement_valid == layout.placement_valid

    def test_layout_restore_after_modifications(self):
        """Restoring from checkpoint should undo subsequent modifications.

        Simulates the reset workflow: create checkpoint, make changes, restore.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create initial layout.
        rooms = [
            {"type": "living_room", "prompt": "Living room", "width": 5.0, "depth": 4.0}
        ]
        tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        # Add a door on north wall (this is our checkpoint state).
        exterior_walls = []
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                exterior_walls.append((label, direction))

        # Use first wall for door.
        door_wall = exterior_walls[0][0]
        tools._add_door_impl(wall_id=door_wall, position="center")

        # Create checkpoint.
        checkpoint = layout.to_dict()
        checkpoint_door_count = len(layout.doors)
        checkpoint_window_count = len(layout.windows)

        # Make modifications on a DIFFERENT wall (to avoid overlap issues).
        # Find a wall without the door.
        window_wall = exterior_walls[1][0] if len(exterior_walls) > 1 else door_wall
        window_result = tools._add_window_impl(
            wall_id=window_wall, position="center", width=1.0
        )
        assert window_result.success, f"Window should be added: {window_result.message}"
        assert (
            len(layout.windows) == checkpoint_window_count + 1
        ), "Window should be added"

        # Restore from checkpoint (simulating reset).
        restored = HouseLayout.from_dict(checkpoint)

        # Verify modifications were undone.
        assert len(restored.doors) == checkpoint_door_count
        assert (
            len(restored.windows) == checkpoint_window_count
        ), f"Window count should be restored: {len(restored.windows)} vs {checkpoint_window_count}"


if __name__ == "__main__":
    unittest.main()
