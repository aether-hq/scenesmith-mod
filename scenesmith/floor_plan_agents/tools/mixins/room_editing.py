"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import json
import logging

from scenesmith.agent_utils.scene.house_parts.openings import (
    ConnectionType,
    Wall,
    WallDirection,
)
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenesmith.floor_plan_agents.tools.floor_plan_models import Result, RoomSpecsResult
from scenesmith.floor_plan_agents.tools.submission.placement.geometry import (
    get_shared_edge,
)
from scenesmith.floor_plan_agents.tools.submission.placement.models import (
    PlacementConfig,
    PlacementError,
)
from scenesmith.floor_plan_agents.tools.submission.placement.search import place_rooms

console_logger = logging.getLogger(__name__)


class FloorPlanRoomEditingMixin:
    """Room creation, resizing, adjacency, height, and wall lookup operations."""

    def _generate_room_specs_impl(self, room_specs_json: str) -> RoomSpecsResult:
        """Create rooms with the specified dimensions and adjacencies.

        MUST be called first. Room mode: fails if >1 room specified.

        Args:
            room_specs_json: JSON string with list of room specifications.
                Each room must have: type and prompt (required).
                Optional fields: width, depth, connections.
                connections: dict mapping room_id to "DOOR" or "OPEN".
                Example:
                '[{"type": "living_room", "width": 5.0, "depth": 4.0,
                   "prompt": "A cozy modern living room with large windows."},
                  {"type": "kitchen", "width": 3.0, "depth": 4.0,
                   "prompt": "A bright kitchen with white cabinets.",
                   "connections": {"living_room": "DOOR"}}]'

        Returns:
            RoomSpecsResult with placed rooms and wall segment labels.
        """
        # Format JSON for readable logging.
        try:
            parsed_for_log = json.loads(room_specs_json)
            formatted_json = json.dumps(parsed_for_log, indent=2)
        except json.JSONDecodeError:
            formatted_json = room_specs_json  # Use raw string if invalid JSON.
        console_logger.info(
            f"Tool called: generate_room_specs(room_specs_json=\n{formatted_json})"
        )
        # Parse JSON input.
        try:
            room_specs = json.loads(room_specs_json)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON: {e}"
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        if not isinstance(room_specs, list):
            msg = "room_specs_json must be a JSON array of room specifications."
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Mode check.
        if self.mode == "room" and len(room_specs) > 1:
            msg = "Room mode: only 1 room allowed. Use house mode for multiple rooms."
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Convert to RoomSpec objects.
        specs = []
        room_type_counts: dict[str, int] = {}
        for spec_dict in room_specs:
            room_type = spec_dict.get("type", "room")

            # Generate room_id from type.
            room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            if room_type_counts[room_type] == 1:
                room_id = room_type
            else:
                room_id = f"{room_type}_{room_type_counts[room_type]}"

            # In room mode, always use house prompt directly (ignore agent's prompt).
            # In house mode, prompt is required for each room.
            if self.mode == "room":
                prompt = self.layout.house_prompt
            else:
                prompt = spec_dict.get("prompt", "")
                if not prompt:
                    msg = (
                        f"Room '{room_type}' is missing required 'prompt' field. "
                        f"Each room must have a descriptive prompt."
                    )
                    console_logger.info(f"Tool failed: {msg}")
                    return RoomSpecsResult(success=False, message=msg)

            # Validate room dimensions.
            room_width = spec_dict.get("width", 5.0)
            room_depth = spec_dict.get("depth", 4.0)
            if not (self.room_dim_min <= room_width <= self.room_dim_max):
                msg = (
                    f"Room '{room_type}' width must be {self.room_dim_min}-"
                    f"{self.room_dim_max}m. Got: {room_width}"
                )
                console_logger.info(f"Tool failed: {msg}")
                return RoomSpecsResult(success=False, message=msg)
            if not (self.room_dim_min <= room_depth <= self.room_dim_max):
                msg = (
                    f"Room '{room_type}' depth must be {self.room_dim_min}-"
                    f"{self.room_dim_max}m. Got: {room_depth}"
                )
                console_logger.info(f"Tool failed: {msg}")
                return RoomSpecsResult(success=False, message=msg)

            # Parse connections from JSON.
            connections_raw = spec_dict.get("connections", {})
            connections = {k: ConnectionType(v) for k, v in connections_raw.items()}

            # Parse exterior_walls constraint from JSON.
            exterior_walls_raw = spec_dict.get("exterior_walls", [])
            exterior_walls = {WallDirection(w) for w in exterior_walls_raw}

            explicit_cover = spec_dict.get(
                "has_overhead_cover",
                spec_dict.get(
                    "has_roof", spec_dict.get("has_ceiling", spec_dict.get("covered"))
                ),
            )
            if isinstance(explicit_cover, bool):
                has_overhead_cover = explicit_cover
            else:
                outdoor_terms = {
                    "garden",
                    "courtyard",
                    "patio",
                    "terrace",
                    "yard",
                    "rooftop",
                }
                covered_terms = {"covered", "roofed", "indoor", "enclosed"}
                semantic_text = f"{room_type} {prompt}".casefold()
                has_overhead_cover = not (
                    any(term in semantic_text for term in outdoor_terms)
                    and not any(term in semantic_text for term in covered_terms)
                )

            specs.append(
                RoomSpec(
                    room_id=room_id,
                    room_type=room_type,
                    prompt=prompt,
                    width=room_depth,  # Y dimension.
                    length=room_width,  # X dimension.
                    connections=connections,
                    exterior_walls=exterior_walls,
                    has_overhead_cover=has_overhead_cover,
                )
            )

        # Run placement algorithm.
        try:
            placed_rooms = place_rooms(room_specs=specs, config=self.placement_config)
        except PlacementError as e:
            msg = f"Room placement failed: {e}"
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Update layout.
        self.layout.room_specs = specs
        self.layout.placed_rooms = placed_rooms
        self.layout.placement_valid = True

        # Clear openings from previous layout (wall labels may have changed).
        self.layout.doors.clear()
        self.layout.windows.clear()

        # Invalidate all cached geometry (room specs completely changed).
        invalidated = self.layout.invalidate_all_room_geometries()
        console_logger.info(f"Invalidated {invalidated} room geometries")

        # Generate ASCII floor plan.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Log ASCII for visibility during runs.
        console_logger.info(
            "Floor plan layout:\n%s\n%s", ascii_result.ascii_art, ascii_result.legend
        )

        # Build wall segment labels description.
        labels_desc = {}
        for label, (room_a, room_b, direction) in ascii_result.boundary_labels.items():
            if room_b:
                labels_desc[label] = f"Interior: {room_a} <-> {room_b}"
            else:
                dir_str = f" ({direction})" if direction else ""
                labels_desc[label] = f"Exterior: {room_a}{dir_str}"

        return RoomSpecsResult(
            success=True,
            message=f"Created {len(specs)} room(s) successfully.",
            ascii_floor_plan=ascii_result.ascii_art,
            wall_segment_labels=labels_desc,
        )

    def _resize_room_impl(self, room_id: str, width: float, depth: float) -> Result:
        """Change a room's dimensions with layout stability.

        Doors and windows are preserved when possible:
        - Openings on walls whose length changed are proportionally repositioned
        - Openings that no longer fit after repositioning are removed
        - Open connections are preserved (positions recomputed from shared edges)

        Other rooms stay in approximately the same positions unless adjacency
        constraints require movement.

        Args:
            room_id: Room to resize (e.g., "living_room").
            width: New width in meters (within configured range).
            depth: New depth in meters (within configured range).

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: resize_room(room_id={room_id}, width={width}, depth={depth})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        # Validate dimensions.
        if not (self.room_dim_min <= width <= self.room_dim_max):
            return self._fail(
                f"Width must be {self.room_dim_min}-{self.room_dim_max}m. Got: {width}"
            )
        if not (self.room_dim_min <= depth <= self.room_dim_max):
            return self._fail(
                f"Depth must be {self.room_dim_min}-{self.room_dim_max}m. Got: {depth}"
            )

        # Find room spec.
        spec = self.layout.get_room_spec(room_id)
        if not spec:
            return self._fail(f"Room '{room_id}' not found.")

        # Store old state for rollback on failure.
        old_width = spec.length  # X dimension.
        old_depth = spec.width  # Y dimension.
        old_placed_rooms = self.layout.placed_rooms

        # Temporarily update dimensions for placement attempt.
        spec.length = width  # X dimension.
        spec.width = depth  # Y dimension.

        # Re-run placement with layout stability (other rooms stay in place).
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={r.room_id: r.position for r in old_placed_rooms},
                free_room_ids={room_id},
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: restore old dimensions, keep layout valid.
            spec.length = old_width
            spec.width = old_depth
            # Note: placed_rooms unchanged since assignment only happens on success.
            return self._fail(f"Resize failed (layout unchanged): {e}")

        # Invalidate geometry for resized room (dimensions changed).
        if self.layout.invalidate_room_geometry(room_id):
            console_logger.debug(f"Invalidated geometry for resized room: {room_id}")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Proportionally adjust opening positions for walls whose length changed.
        self._adjust_opening_positions_for_resize(
            room_id=room_id,
            old_width=old_width,
            old_depth=old_depth,
            new_width=width,
            new_depth=depth,
        )

        # Reapply all doors/windows and open connections. This validates positions
        # and removes openings that no longer fit after resize.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        # Build result message with removal info.
        msg = f"Room '{room_id}' resized to {width}m x {depth}m."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _add_adjacency_impl(self, room_a: str, room_b: str) -> Result:
        """Require two rooms to share a wall.

        Room mode: fails (single room has no adjacencies).

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: add_adjacency(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no adjacencies for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # Track whether we need to add connections (for rollback on failure).
        added_b_to_a = room_b not in spec_a.connections
        added_a_to_b = room_a not in spec_b.connections

        # Add connection with DOOR type.
        if added_b_to_a:
            spec_a.connections[room_b] = ConnectionType.DOOR
        if added_a_to_b:
            spec_b.connections[room_a] = ConnectionType.DOOR

        # If rooms are already placed and already adjacent, no re-placement needed.
        if self.layout.placed_rooms:
            placed_a = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_a), None
            )
            placed_b = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_b), None
            )
            if placed_a and placed_b:
                shared_edge = get_shared_edge(placed_a, placed_b)
                if shared_edge:
                    # Already adjacent - constraint already satisfied.
                    msg = f"Added adjacency: {room_a} <-> {room_b}."
                    return Result(success=True, message=msg)

        # Rooms not yet placed or not currently adjacent - run placement with stability.
        # Both rooms being made adjacent should have freedom to move.
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={
                    r.room_id: r.position for r in self.layout.placed_rooms
                },
                free_room_ids={room_a, room_b},
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: remove connections we added.
            if added_b_to_a:
                del spec_a.connections[room_b]
            if added_a_to_b:
                del spec_b.connections[room_a]
            return self._fail(f"Add adjacency failed (layout unchanged): {e}")

        # Invalidate all geometry (adjacency affects positions and wall labels).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore openings (doors, windows, open connections) to new walls.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Added adjacency: {room_a} <-> {room_b}."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _remove_adjacency_impl(self, room_a: str, room_b: str) -> Result:
        """Remove requirement for two rooms to share a wall.

        Room mode: fails.

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: remove_adjacency(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no adjacencies for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # Track connections we remove (for rollback on failure).
        removed_b_from_a = spec_a.connections.pop(room_b, None)
        removed_a_from_b = spec_b.connections.pop(room_a, None)

        # If rooms are already placed, no re-placement needed.
        # Removing a constraint doesn't invalidate existing placement.
        if self.layout.placed_rooms:
            msg = f"Removed adjacency: {room_a} <-> {room_b}."
            return Result(success=True, message=msg)

        # Rooms not yet placed - run placement with stability.
        # No special rooms need freedom since we're just removing a constraint.
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={
                    r.room_id: r.position for r in self.layout.placed_rooms
                },
                free_room_ids=set(),
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: restore connections we removed.
            if removed_b_from_a is not None:
                spec_a.connections[room_b] = removed_b_from_a
            if removed_a_from_b is not None:
                spec_b.connections[room_a] = removed_a_from_b
            return self._fail(f"Remove adjacency failed (layout unchanged): {e}")

        # Invalidate all geometry (adjacency affects positions and wall labels).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore openings (doors, windows, open connections) to new walls.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Removed adjacency: {room_a} <-> {room_b}."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _set_wall_height_impl(self, height_meters: float) -> Result:
        """Set wall/ceiling height for entire house.

        Args:
            height_meters: Height in valid range (from config).

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: set_wall_height(height_meters={height_meters})"
        )
        if not (self.wall_height_min <= height_meters <= self.wall_height_max):
            return self._fail(
                f"Height must be between {self.wall_height_min} and "
                f"{self.wall_height_max}m. Got: {height_meters}"
            )

        self.layout.wall_height = height_meters

        # Invalidate all geometry (wall height affects all rooms).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        return Result(success=True, message=f"Wall height set to {height_meters}m.")

    def _get_wall_by_boundary(self, wall_label: str, room_id: str) -> Wall:
        """Get a Wall object by boundary label and room ID.

        Args:
            wall_label: Boundary label (e.g., "A", "B") from boundary_labels.
            room_id: Room ID owning the wall.

        Returns:
            The Wall object.

        Raises:
            ValueError: If wall cannot be found (fail-fast per CLAUDE.md).
        """
        # Look up what this boundary label refers to.
        boundary_info = self.layout.boundary_labels.get(wall_label)
        if not boundary_info:
            raise ValueError(
                f"Unknown wall label '{wall_label}'. "
                f"Available labels: {list(self.layout.boundary_labels.keys())}"
            )

        room_a_label, room_b_label, direction = boundary_info

        # Determine which room we're looking for on the wall.
        # If room_id is room_a, look for wall facing room_b.
        # If room_id is room_b, look for wall facing room_a.
        if room_id == room_a_label:
            target_room = room_b_label
        elif room_id == room_b_label:
            target_room = room_a_label
        else:
            # room_id doesn't match either end of this boundary.
            raise ValueError(
                f"Room '{room_id}' is not part of boundary '{wall_label}' "
                f"which connects {room_a_label} <-> {room_b_label}."
            )

        # Find the placed room.
        placed_room = None
        for placed in self.layout.placed_rooms:
            if placed.room_id == room_id:
                placed_room = placed
                break

        if not placed_room:
            raise ValueError(f"Room '{room_id}' not found in placed_rooms.")

        # Find the wall matching this boundary.
        for wall in placed_room.walls:
            if target_room is None:
                # Exterior wall - match by direction.
                if wall.is_exterior and wall.direction.value == direction:
                    return wall
            else:
                # Interior wall - check if this wall faces the target room.
                if target_room in wall.faces_rooms:
                    return wall

        raise ValueError(
            f"Wall for boundary '{wall_label}' not found in room '{room_id}'."
        )

    def _get_wall_length(self, room_id: str, wall_label: str) -> float:
        """Get the length of a wall by room and boundary label.

        Args:
            room_id: Room ID owning the wall.
            wall_label: Boundary label (e.g., "A", "B") from boundary_labels.

        Returns:
            Wall length in meters.

        Raises:
            ValueError: If wall cannot be found (fail-fast per CLAUDE.md).
        """
        wall = self._get_wall_by_boundary(wall_label=wall_label, room_id=room_id)
        return wall.length

    def _wall_faces_nearby_room(
        self, room_id: str, wall_label: str, threshold: float = 0.5
    ) -> str | None:
        """Check if an exterior wall faces another room within threshold distance.

        Used to prevent windows on walls that technically exterior but face another
        room across a small gap (e.g., 10cm gap between non-adjacent rooms).

        Args:
            room_id: Room owning the wall.
            wall_label: Boundary label from boundary_labels.
            threshold: Maximum distance (meters) to consider as "facing".

        Returns:
            Room ID of nearby room if found, None otherwise.
        """
        wall = self._get_wall_by_boundary(wall_label=wall_label, room_id=room_id)
        placed_room = next(
            (r for r in self.layout.placed_rooms if r.room_id == room_id), None
        )
        if not placed_room:
            return None

        # Get wall position based on direction.
        for other in self.layout.placed_rooms:
            if other.room_id == room_id:
                continue

            other_min_x = other.position[0]
            other_max_x = other.position[0] + other.width
            other_min_y = other.position[1]
            other_max_y = other.position[1] + other.depth

            # Check if wall faces this room within threshold.
            if wall.direction == WallDirection.NORTH:
                # Wall at y = placed_room.position[1] + depth, facing +Y.
                wall_y = placed_room.position[1] + placed_room.depth
                # Check if other room's south edge is within threshold.
                if 0 < other_min_y - wall_y <= threshold:
                    # Check x overlap.
                    if max(wall.start_point[0], other_min_x) < min(
                        wall.end_point[0], other_max_x
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.SOUTH:
                # Wall at y = placed_room.position[1], facing -Y.
                wall_y = placed_room.position[1]
                # Check if other room's north edge is within threshold.
                if 0 < wall_y - other_max_y <= threshold:
                    if max(wall.start_point[0], other_min_x) < min(
                        wall.end_point[0], other_max_x
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.EAST:
                # Wall at x = placed_room.position[0] + width, facing +X.
                wall_x = placed_room.position[0] + placed_room.width
                # Check if other room's west edge is within threshold.
                if 0 < other_min_x - wall_x <= threshold:
                    if max(wall.start_point[1], other_min_y) < min(
                        wall.end_point[1], other_max_y
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.WEST:
                # Wall at x = placed_room.position[0], facing -X.
                wall_x = placed_room.position[0]
                # Check if other room's east edge is within threshold.
                if 0 < wall_x - other_max_x <= threshold:
                    if max(wall.start_point[1], other_min_y) < min(
                        wall.end_point[1], other_max_y
                    ):
                        return other.room_id

        return None
