"""Window placement and removal behavior for floor-plan submission tools."""

import logging
import random

from typing import TYPE_CHECKING

from scenesmith.agent_utils.scene.house_parts.openings import (
    Opening,
    OpeningType,
    WallDirection,
    Window,
    WindowShape,
)
from scenesmith.floor_plan_agents.tools.submission.opening_constants import (
    SEGMENT_LEFT_END,
    SEGMENT_RIGHT_START,
    WINDOW_EDGE_INSET,
)
from scenesmith.floor_plan_agents.tools.submission.placement.layout import (
    validate_connectivity,
)

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.house import HouseLayout

console_logger = logging.getLogger(__name__)


class WindowSubmissionMixin:
    layout: "HouseLayout"
    min_opening_separation: float

    def _apply_window_to_wall(self, window: Window) -> str | None:
        """Apply a window's Opening to the appropriate wall.

        Creates an Opening from the Window and adds it to the wall.

        Args:
            window: The Window to apply.

        Returns:
            None if successful, error message string if window doesn't fit.
        """
        # Look up wall by stable (room_id, direction) if available.
        wall = None
        if window.wall_direction:
            placed_room = next(
                (r for r in self.layout.placed_rooms if r.room_id == window.room_id),
                None,
            )
            if placed_room:
                wall = next(
                    (
                        w
                        for w in placed_room.walls
                        if w.direction == window.wall_direction
                    ),
                    None,
                )
            if not wall:
                msg = (
                    f"Wall {window.wall_direction.value} not found for room "
                    f"'{window.room_id}'"
                )
                console_logger.warning(f"Window {window.id}: {msg}")
                return msg
        else:
            # Fallback to boundary_label lookup (legacy windows without wall_direction).
            if window.boundary_label not in self.layout.boundary_labels:
                msg = f"boundary '{window.boundary_label}' no longer exists"
                console_logger.warning(f"Window {window.id}: {msg}")
                return msg
            try:
                wall = self._get_wall_by_boundary(window.boundary_label, window.room_id)
            except ValueError as e:
                console_logger.warning(f"Window {window.id}: {e}")
                return str(e)

        # Validate position fits in wall.
        # Window position is LEFT EDGE (matches door/OPEN convention).
        if window.position_along_wall < 0:
            msg = (
                f"position {window.position_along_wall:.2f}m is before wall start "
                f"for {window.width:.2f}m window"
            )
            console_logger.warning(f"Window {window.id}: {msg}")
            return msg
        if window.position_along_wall + window.width > wall.length:
            msg = (
                f"position {window.position_along_wall:.2f}m too close to wall end "
                f"for {window.width:.2f}m window (wall is {wall.length:.2f}m)"
            )
            console_logger.warning(f"Window {window.id}: {msg}")
            return msg

        # Create Opening.
        opening = Opening(
            opening_id=window.id,
            opening_type=OpeningType.WINDOW,
            position_along_wall=window.position_along_wall,
            width=window.width,
            height=window.height,
            sill_height=window.sill_height,
            shape=window.shape,
        )

        # Add to wall if not already present.
        if not any(o.opening_id == window.id for o in wall.openings):
            wall.openings.append(opening)

        return None

    def _remove_door_impl(self, door_id: str):
        """Remove a door.

        Fails if removal breaks room connectivity.

        Args:
            door_id: Door identifier to remove.

        Returns:
            Result indicating success or failure.
        """
        from scenesmith.floor_plan_agents.tools.floor_plan_tools import Result

        console_logger.info(f"Tool called: remove_door(door_id={door_id})")
        error = self._check_rooms_exist()
        if error:
            return error

        # Find door.
        door_idx = None
        for i, door in enumerate(self.layout.doors):
            if door.id == door_id:
                door_idx = i
                break

        if door_idx is None:
            return self._fail(f"Door '{door_id}' not found.")

        # Check connectivity without this door.
        doors_without = self.layout.doors[:door_idx] + self.layout.doors[door_idx + 1 :]
        is_valid, msg = validate_connectivity(
            self.layout.placed_rooms, doors_without, self.layout.room_specs
        )

        if not is_valid:
            return self._fail(f"Cannot remove door: {msg}")

        # Get door info before removing.
        door = self.layout.doors[door_idx]

        # Remove opening from wall(s).
        wall_a = self._get_wall_by_boundary(door.boundary_label, door.room_a)
        wall_a.openings = [o for o in wall_a.openings if o.opening_id != door_id]
        if door.room_b:
            wall_b = self._get_wall_by_boundary(door.boundary_label, door.room_b)
            wall_b.openings = [o for o in wall_b.openings if o.opening_id != door_id]

        # Remove door.
        self.layout.doors.pop(door_idx)

        # Invalidate geometry for affected rooms (wall openings changed).
        if self.layout.invalidate_room_geometry(door.room_a):
            console_logger.debug(f"Invalidated geometry for room: {door.room_a}")
        if door.room_b and self.layout.invalidate_room_geometry(door.room_b):
            console_logger.debug(f"Invalidated geometry for room: {door.room_b}")

        return Result(success=True, message=f"Removed door '{door_id}'.")

    def _add_window_impl(
        self,
        wall_id: str,
        position: str = "center",
        width: float | None = None,
        height: float | None = None,
        sill_height: float | None = None,
        shape: str = "rectangular",
    ):
        """Add a window to an exterior wall.

        Fails if wall is interior or has a door.

        Args:
            wall_id: Exterior wall segment ID.
            position: "left" | "center" | "right".
            width: Window width in valid range (uses config default if not specified).
            height: Window height in valid range (uses config default if not specified).
            sill_height: Height from floor to window bottom (uses config default).
            shape: Window silhouette: "rectangular" or "arched".

        Returns:
            Result indicating success or failure.
        """
        from scenesmith.floor_plan_agents.tools.floor_plan_tools import Result

        # Apply defaults from config.
        cfg = self.door_window_config
        if width is None:
            width = cfg.window_default_width
        if height is None:
            height = cfg.window_default_height
        if sill_height is None:
            sill_height = cfg.window_default_sill_height
        try:
            window_shape = WindowShape(shape)
        except ValueError:
            return self._fail(
                f"Window shape must be 'rectangular' or 'arched'. Got: {shape}"
            )

        console_logger.info(
            f"Tool called: add_window(wall_id={wall_id}, position={position}, "
            f"width={width}, sill_height={sill_height}, shape={window_shape.value})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        if position not in {"left", "center", "right"}:
            return self._fail(
                f"Position must be 'left', 'center', or 'right'. Got: {position}"
            )

        # Validate dimensions.
        if not (cfg.window_width_min <= width <= cfg.window_width_max):
            return self._fail(
                f"Window width must be {cfg.window_width_min}-{cfg.window_width_max}m. "
                f"Got: {width}"
            )

        if not (cfg.window_height_min <= height <= cfg.window_height_max):
            return self._fail(
                f"Window height must be {cfg.window_height_min}-{cfg.window_height_max}m. "
                f"Got: {height}"
            )
        if window_shape == WindowShape.ARCHED and height <= width / 2.0:
            return self._fail(
                "Arched window height must exceed half its width so the curved "
                f"crown fits. Got width={width}, height={height}."
            )

        # Check wall exists and is exterior.
        if wall_id not in self.layout.boundary_labels:
            available = ", ".join(sorted(self.layout.boundary_labels.keys()))
            return self._fail(f"Wall '{wall_id}' not found. Available: {available}")

        room_a, room_b, _direction = self.layout.boundary_labels[wall_id]
        if room_b is not None:
            return self._fail(
                f"Wall '{wall_id}' is interior. Windows only on exterior walls."
            )

        # Check if wall faces another room across a small gap.
        nearby_room = self._wall_faces_nearby_room(room_a, wall_id, threshold=0.5)
        if nearby_room:
            return self._fail(
                f"Wall '{wall_id}' faces {nearby_room} across a small gap. "
                f"Windows only allowed on true exterior walls."
            )

        # Check for duplicate window on same wall/position.
        for existing in self.layout.windows:
            if existing.boundary_label == wall_id:
                # Get position segment for existing window.
                wall_length = self._get_wall_length(room_a, wall_id)
                existing_pct = existing.position_along_wall / wall_length
                if position == "left" and existing_pct < SEGMENT_LEFT_END:
                    return self._fail(
                        f"Wall '{wall_id}' already has a window on the left."
                    )
                if (
                    position == "center"
                    and SEGMENT_LEFT_END <= existing_pct <= SEGMENT_RIGHT_START
                ):
                    return self._fail(
                        f"Wall '{wall_id}' already has a window in the center."
                    )
                if position == "right" and existing_pct > SEGMENT_RIGHT_START:
                    return self._fail(
                        f"Wall '{wall_id}' already has a window on the right."
                    )

        # Calculate position with randomization.
        # Windows are inset from wall edges to avoid corners.
        segment_ranges = {
            "left": (WINDOW_EDGE_INSET, SEGMENT_LEFT_END),
            "center": (SEGMENT_LEFT_END, SEGMENT_RIGHT_START),
            "right": (SEGMENT_RIGHT_START, 1.0 - WINDOW_EDGE_INSET),
        }
        start_pct, end_pct = segment_ranges[position]

        wall_length = self._get_wall_length(room_a, wall_id)

        # Calculate position using center-based sampling.
        # Window center must be within segment, but edges must respect wall margins.
        margin = self.door_window_config.window_segment_margin
        half_width = width / 2

        # Valid center range based on wall boundaries (window must fit with margins).
        wall_center_min = margin + half_width
        wall_center_max = wall_length - margin - half_width

        # Segment boundaries for window center.
        segment_center_min = wall_length * start_pct
        segment_center_max = wall_length * end_pct

        # Intersection: center must be in segment AND within wall bounds.
        center_min = max(wall_center_min, segment_center_min)
        center_max = min(wall_center_max, segment_center_max)

        if center_max < center_min:
            return self._fail(
                f"Cannot place {width:.2f}m window in '{position}' segment of "
                f"{wall_length:.2f}m wall. Try a different segment or narrower window."
            )

        # Sample window center, convert to left edge position.
        window_center = random.uniform(center_min, center_max)
        position_along = window_center - half_width

        # Check for overlap with existing doors on this wall.
        window_start = position_along
        window_end = position_along + width
        for door in self.layout.doors:
            if door.boundary_label == wall_id:
                door_start = door.position_exact
                door_end = door.position_exact + door.width
                # Check overlap with separation margin.
                if (
                    window_start < door_end + self.min_opening_separation
                    and window_end > door_start - self.min_opening_separation
                ):
                    return self._fail(
                        f"Window would overlap with door on wall '{wall_id}'. "
                        f"Window: {window_start:.2f}-{window_end:.2f}m, "
                        f"Door: {door_start:.2f}-{door_end:.2f}m. "
                        f"Min separation: {self.min_opening_separation}m. "
                        f"Try a different position."
                    )

        # Check for overlap with existing windows on this wall.
        for existing_window in self.layout.windows:
            if existing_window.boundary_label == wall_id:
                existing_start = existing_window.position_along_wall
                existing_end = (
                    existing_window.position_along_wall + existing_window.width
                )
                # Check overlap with separation margin.
                if (
                    window_start < existing_end + self.min_opening_separation
                    and window_end > existing_start - self.min_opening_separation
                ):
                    return self._fail(
                        f"Window would overlap with existing window "
                        f"'{existing_window.id}' on wall '{wall_id}'. "
                        f"New window: {window_start:.2f}-{window_end:.2f}m, "
                        f"Existing: {existing_start:.2f}-{existing_end:.2f}m. "
                        f"Min separation: {self.min_opening_separation}m. "
                        f"Try a different position or remove the existing window."
                    )

        # Create window.
        # Get wall direction from boundary info for stable lookup.
        _, _, direction_str = self.layout.boundary_labels[wall_id]
        wall_direction = WallDirection(direction_str) if direction_str else None

        window_id = self._next_window_id()
        window = Window(
            id=window_id,
            boundary_label=wall_id,
            position_along_wall=position_along,
            room_id=room_a,
            wall_direction=wall_direction,
            width=width,
            height=height,
            sill_height=sill_height,
            shape=window_shape,
        )

        # Add Opening to wall for rendering/ASCII.
        error = self._apply_window_to_wall(window)
        if error:
            return self._fail(f"Failed to add window at wall {wall_id}: {error}")

        self.layout.windows.append(window)

        # Invalidate geometry for affected room (wall openings changed).
        if self.layout.invalidate_room_geometry(room_a):
            console_logger.debug(f"Invalidated geometry for room: {room_a}")

        return Result(
            success=True,
            message=f"Added window '{window_id}' at wall {wall_id} ({position}).",
        )

    def _remove_window_impl(self, window_id: str):
        """Remove a window.

        Args:
            window_id: Window identifier to remove.

        Returns:
            Result indicating success or failure.
        """
        from scenesmith.floor_plan_agents.tools.floor_plan_tools import Result

        console_logger.info(f"Tool called: remove_window(window_id={window_id})")
        error = self._check_rooms_exist()
        if error:
            return error

        # Find window.
        window_idx = None
        for i, window in enumerate(self.layout.windows):
            if window.id == window_id:
                window_idx = i
                break

        if window_idx is None:
            return self._fail(f"Window '{window_id}' not found.")

        # Get window info before removing.
        window = self.layout.windows[window_idx]

        # Remove opening from wall using stable lookup.
        if window.wall_direction:
            placed_room = next(
                (r for r in self.layout.placed_rooms if r.room_id == window.room_id),
                None,
            )
            if placed_room:
                wall = next(
                    (
                        w
                        for w in placed_room.walls
                        if w.direction == window.wall_direction
                    ),
                    None,
                )
                if wall:
                    wall.openings = [
                        o for o in wall.openings if o.opening_id != window_id
                    ]
        elif window.boundary_label in self.layout.boundary_labels:
            # Fallback to boundary_label lookup.
            try:
                wall = self._get_wall_by_boundary(window.boundary_label, window.room_id)
                wall.openings = [o for o in wall.openings if o.opening_id != window_id]
            except ValueError:
                pass  # Wall no longer exists, just remove window.

        # Remove window.
        self.layout.windows.pop(window_idx)

        # Invalidate geometry for affected room (wall openings changed).
        if self.layout.invalidate_room_geometry(window.room_id):
            console_logger.debug(f"Invalidated geometry for room: {window.room_id}")

        return Result(success=True, message=f"Removed window '{window_id}'.")
