"""Slot-based room placement algorithm with backtracking search.

This module implements an optimal room placement algorithm that satisfies
adjacency constraints while minimizing bounding box size and maintaining
layout stability across iterative edits.

Algorithm Overview
------------------
1. **Topological Sort**: Rooms sorted by adjacency dependencies. Anchor rooms
   (no adjacencies) placed first, then connector rooms that link them.

2. **First Room at Origin**: The first sorted room is placed at (0, 0).

3. **Slot-Based Attachment**: Each placed room exposes 4 edge slots (N/S/E/W).
   New rooms attach to slots of rooms they must be adjacent to.

4. **Backtracking Search**: Explores all valid placements recursively:
   - For each unplaced room, generate all valid candidate positions
   - Sort candidates by local score (adjacency satisfaction, compactness)
   - Recurse on each candidate, tracking the best complete layout found
   - Prune branches on timeout (anytime algorithm behavior)

5. **Global Scoring**: Complete layouts scored by:
   - Compactness: ratio of room area to bounding box area (higher = better)
   - Stability: proximity to previous positions (for iterative editing)

Properties
----------
- **Optimality**: Finds optimal layout within *fixed room ordering* and *discrete
  position space*. Two limitations:
  1. Room order fixed by topological sort (anchor rooms first, then connectors).
     Different orderings could yield different layouts, but exploring all O(n!)
     orderings is intractable.
  2. Positions sampled at 11 evenly-spaced points per slot edge (0%, 10%, ..., 100%).
  With ≤10 rooms and 5s timeout, typically explores all valid positions for the
  fixed ordering.

- **Anytime Behavior**: Returns best-found-so-far if timeout exceeded. Search
  is best-first (candidates sorted by score), so early layouts are good.

- **Soundness**: All returned layouts satisfy adjacency constraints. Candidates
  that don't satisfy required adjacencies to placed rooms are rejected during
  scoring (score=0), ensuring only valid placements are explored.

- **Completeness**: If a valid layout exists for the fixed room ordering and
  discrete position space, the algorithm will find it (given sufficient time).
  Returns PlacementError only when no valid layout exists within these constraints.

Features
--------
- 90° room rotation: Automatically tries rotated orientation for better fit
- Multi-adjacency: Corner-aligned positions for rooms adjacent to 2+ others
- Layout stability: Configurable preference for positions near previous layout
"""

import logging

from scenesmith.agent_utils.scene.house_parts.openings import (
    PlacedRoom,
    Wall,
    WallDirection,
)
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec

console_logger = logging.getLogger(__name__)

from scenesmith.floor_plan_agents.tools.submission.placement.models import Slot


def create_placed_room(
    spec: RoomSpec,
    position: tuple[float, float],
    room_width: float | None = None,
    room_depth: float | None = None,
) -> PlacedRoom:
    """Create a PlacedRoom from spec at given position.

    Args:
        spec: Room specification.
        position: (x, y) min corner position.
        room_width: Explicit X dimension (overrides spec.length if provided).
        room_depth: Explicit Y dimension (overrides spec.width if provided).

    Returns:
        PlacedRoom with walls.
    """
    # Create walls for this room.
    # Using length as x-dimension (width in PlacedRoom) and width as y-dimension (depth).
    if room_width is None:
        room_width = spec.length  # X dimension.
    if room_depth is None:
        room_depth = spec.width  # Y dimension.

    walls = [
        Wall(
            wall_id=f"{spec.room_id}_north",
            room_id=spec.room_id,
            direction=WallDirection.NORTH,
            start_point=(position[0], position[1] + room_depth),
            end_point=(position[0] + room_width, position[1] + room_depth),
            length=room_width,
        ),
        Wall(
            wall_id=f"{spec.room_id}_south",
            room_id=spec.room_id,
            direction=WallDirection.SOUTH,
            start_point=(position[0], position[1]),
            end_point=(position[0] + room_width, position[1]),
            length=room_width,
        ),
        Wall(
            wall_id=f"{spec.room_id}_east",
            room_id=spec.room_id,
            direction=WallDirection.EAST,
            start_point=(position[0] + room_width, position[1]),
            end_point=(position[0] + room_width, position[1] + room_depth),
            length=room_depth,
        ),
        Wall(
            wall_id=f"{spec.room_id}_west",
            room_id=spec.room_id,
            direction=WallDirection.WEST,
            start_point=(position[0], position[1]),
            end_point=(position[0], position[1] + room_depth),
            length=room_depth,
        ),
    ]

    return PlacedRoom(
        room_id=spec.room_id,
        position=position,
        width=room_width,
        depth=room_depth,
        walls=walls,
        level_id=spec.level_id,
        elevation=spec.elevation,
        yaw=spec.yaw,
        footprint=spec.footprint,
    )


def _get_room_slots(room: PlacedRoom) -> list[Slot]:
    """Get available attachment slots for a room.

    Args:
        room: Placed room.

    Returns:
        List of slots (one per edge).
    """
    slots = []

    # North slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.NORTH,
            start=room.position[0],
            end=room.position[0] + room.width,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    # South slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.SOUTH,
            start=room.position[0],
            end=room.position[0] + room.width,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    # East slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.EAST,
            start=room.position[1],
            end=room.position[1] + room.depth,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    # West slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.WEST,
            start=room.position[1],
            end=room.position[1] + room.depth,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    return slots


def update_wall_connectivity(placed_rooms: list[PlacedRoom]) -> None:
    """Update wall is_exterior and faces_rooms based on placement.

    Args:
        placed_rooms: All placed rooms (modified in place).
    """
    for room in placed_rooms:
        for wall in room.walls:
            # Check each other room to see if they share this wall.
            wall.is_exterior = True
            wall.faces_rooms = []

            for other_room in placed_rooms:
                if other_room.room_id == room.room_id:
                    continue
                if other_room.level_id != room.level_id:
                    continue

                # Check if this wall touches the other room.
                if _wall_touches_room(wall, other_room):
                    wall.is_exterior = False
                    if other_room.room_id not in wall.faces_rooms:
                        wall.faces_rooms.append(other_room.room_id)


def _wall_touches_room(wall: Wall, room: PlacedRoom) -> bool:
    """Check if a wall touches (is adjacent to) a room.

    Args:
        wall: Wall to check.
        room: Room to check against.

    Returns:
        True if wall touches room's boundary.
    """
    # Wall endpoints.
    w_start = wall.start_point
    w_end = wall.end_point

    # Room bounds.
    r_min_x = room.position[0]
    r_max_x = room.position[0] + room.width
    r_min_y = room.position[1]
    r_max_y = room.position[1] + room.depth

    # Check based on wall direction.
    if wall.direction == WallDirection.NORTH:
        # Wall is horizontal at y = w_start[1].
        # Check if it touches room's south edge.
        if abs(w_start[1] - r_min_y) < 0.001:
            # Check x overlap.
            x_overlap = max(w_start[0], r_min_x) < min(w_end[0], r_max_x)
            return x_overlap
    elif wall.direction == WallDirection.SOUTH:
        # Check if it touches room's north edge.
        if abs(w_start[1] - r_max_y) < 0.001:
            x_overlap = max(w_start[0], r_min_x) < min(w_end[0], r_max_x)
            return x_overlap
    elif wall.direction == WallDirection.EAST:
        # Wall is vertical at x = w_start[0].
        # Check if it touches room's west edge.
        if abs(w_start[0] - r_min_x) < 0.001:
            y_overlap = max(w_start[1], r_min_y) < min(w_end[1], r_max_y)
            return y_overlap
    elif wall.direction == WallDirection.WEST:
        # Check if it touches room's east edge.
        if abs(w_start[0] - r_max_x) < 0.001:
            y_overlap = max(w_start[1], r_min_y) < min(w_end[1], r_max_y)
            return y_overlap

    return False


def find_room(rooms: list[PlacedRoom], room_id: str) -> PlacedRoom:
    """Find a room by ID.

    Args:
        rooms: List of placed rooms.
        room_id: Room ID to find.

    Returns:
        The room with matching ID.

    Raises:
        ValueError: If room not found.
    """
    for room in rooms:
        if room.room_id == room_id:
            return room
    raise ValueError(f"Room '{room_id}' not found")


def get_shared_boundary(room_a: PlacedRoom, room_b: PlacedRoom) -> Wall | None:
    """Get the wall segment shared between two rooms.

    Args:
        room_a: First room.
        room_b: Second room.

    Returns:
        Wall from room_a that faces room_b, or None if not adjacent.
    """
    for wall in room_a.walls:
        if room_b.room_id in wall.faces_rooms:
            return wall
    return None


def validate_connectivity(
    placed_rooms: list[PlacedRoom],
    doors: list,  # list[Door] but avoiding circular import
    room_specs: list | None = None,  # list[RoomSpec] for open connections
) -> tuple[bool, str]:
    """Validate that all rooms are reachable from exterior via doors or open connections.

    Uses BFS from rooms with exterior doors.

    Args:
        placed_rooms: All placed rooms.
        doors: All doors in the house.
        room_specs: Room specifications containing open connections.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not placed_rooms:
        return True, ""

    if not doors:
        return False, "No doors defined. At least one exterior door required."

    # Find rooms with exterior doors.
    rooms_with_exterior_door = set()
    for door in doors:
        if door.door_type == "exterior":
            rooms_with_exterior_door.add(door.room_a)

    if not rooms_with_exterior_door:
        return False, "No exterior door found. At least one exterior door required."

    # Build adjacency graph from interior doors AND open connections.
    connections: dict[str, set[str]] = {r.room_id: set() for r in placed_rooms}

    # Add interior door connections.
    for door in doors:
        if door.door_type == "interior" and door.room_b:
            connections[door.room_a].add(door.room_b)
            connections[door.room_b].add(door.room_a)

    # Add open connections (rooms with no wall between them).
    if room_specs:
        for spec in room_specs:
            for other_room, conn_type in spec.connections.items():
                if conn_type.value == "OPEN":
                    if spec.room_id in connections and other_room in connections:
                        connections[spec.room_id].add(other_room)
                        connections[other_room].add(spec.room_id)

    # BFS from exterior doors.
    reachable = set()
    queue = list(rooms_with_exterior_door)
    reachable.update(queue)

    while queue:
        current = queue.pop(0)
        for neighbor in connections.get(current, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    # Check all rooms are reachable.
    all_room_ids = {r.room_id for r in placed_rooms}
    unreachable = all_room_ids - reachable

    if unreachable:
        return (
            False,
            f"Rooms not reachable from exterior: {', '.join(sorted(unreachable))}",
        )

    return True, ""
