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
import math

from dataclasses import dataclass

from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom, WallDirection
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.structure.geometry_models.surface_models import Footprint2D

console_logger = logging.getLogger(__name__)


def _has_overlap(room: PlacedRoom, placed_rooms: list[PlacedRoom]) -> bool:
    """Check if room overlaps with any placed room.

    Args:
        room: Room to check.
        placed_rooms: Existing rooms.

    Returns:
        True if any overlap detected.
    """
    for other in placed_rooms:
        if rooms_overlap(room_a=room, room_b=other):
            return True
    return False


def rooms_overlap(room_a: PlacedRoom, room_b: PlacedRoom) -> bool:
    """Check if two rooms have overlapping interiors.

    Args:
        room_a: First room.
        room_b: Second room.

    Returns:
        True if rooms overlap (share interior space).
    """
    if room_a.level_id != room_b.level_id:
        return False

    if (
        room_a.footprint is not None
        or room_b.footprint is not None
        or abs(room_a.yaw) > 1e-9
        or abs(room_b.yaw) > 1e-9
    ):
        from shapely.geometry import Polygon

        def world_polygon(room: PlacedRoom) -> Polygon:
            footprint = room.footprint or Footprint2D.rectangle(room.width, room.depth)
            centered = footprint.centered_on_bounds()
            cosine, sine = math.cos(room.yaw), math.sin(room.yaw)
            center_x = room.position[0] + room.width / 2.0
            center_y = room.position[1] + room.depth / 2.0

            def transform(loop):
                return tuple(
                    (
                        center_x + cosine * x - sine * y,
                        center_y + sine * x + cosine * y,
                    )
                    for x, y in loop
                )

            return Polygon(
                transform(centered.outer),
                [transform(hole) for hole in centered.holes],
            )

        overlap_area = world_polygon(room_a).intersection(world_polygon(room_b)).area
        return overlap_area > 1e-9

    # Room A bounds.
    a_min_x = room_a.position[0]
    a_max_x = room_a.position[0] + room_a.width
    a_min_y = room_a.position[1]
    a_max_y = room_a.position[1] + room_a.depth

    # Room B bounds.
    b_min_x = room_b.position[0]
    b_max_x = room_b.position[0] + room_b.width
    b_min_y = room_b.position[1]
    b_max_y = room_b.position[1] + room_b.depth

    # Check for overlap (strict inequality - touching edges are OK).
    x_overlap = a_min_x < b_max_x and a_max_x > b_min_x
    y_overlap = a_min_y < b_max_y and a_max_y > b_min_y

    return x_overlap and y_overlap


def _get_exterior_clearance_zones(
    room: PlacedRoom, spec: RoomSpec, clearance: float
) -> list[tuple[float, float, float, float]]:
    """Get rectangular clearance zones for exterior_walls.

    Each exterior_walls direction creates a forbidden zone extending outward.
    This prevents rooms from blocking exterior access either by direct adjacency
    or by wrapping around.

    Args:
        room: Placed room to compute zones for.
        spec: Room specification with exterior_walls constraints.
        clearance: Clearance distance in meters.

    Returns:
        List of (min_x, min_y, max_x, max_y) tuples representing forbidden zones.
    """
    zones: list[tuple[float, float, float, float]] = []
    for direction in spec.exterior_walls:
        if direction == WallDirection.WEST:
            # Zone extends clearance meters to the west.
            zones.append(
                (
                    room.position[0] - clearance,
                    room.position[1],
                    room.position[0],
                    room.position[1] + room.depth,
                )
            )
        elif direction == WallDirection.EAST:
            # Zone extends clearance meters to the east.
            zones.append(
                (
                    room.position[0] + room.width,
                    room.position[1],
                    room.position[0] + room.width + clearance,
                    room.position[1] + room.depth,
                )
            )
        elif direction == WallDirection.SOUTH:
            # Zone extends clearance meters to the south.
            zones.append(
                (
                    room.position[0],
                    room.position[1] - clearance,
                    room.position[0] + room.width,
                    room.position[1],
                )
            )
        elif direction == WallDirection.NORTH:
            # Zone extends clearance meters to the north.
            zones.append(
                (
                    room.position[0],
                    room.position[1] + room.depth,
                    room.position[0] + room.width,
                    room.position[1] + room.depth + clearance,
                )
            )
    return zones


def _overlaps_zone(room: PlacedRoom, zone: tuple[float, float, float, float]) -> bool:
    """Check if room overlaps with a clearance zone.

    Args:
        room: Room to check.
        zone: Clearance zone as (min_x, min_y, max_x, max_y).

    Returns:
        True if room overlaps with the zone.
    """
    z_min_x, z_min_y, z_max_x, z_max_y = zone
    r_min_x = room.position[0]
    r_max_x = room.position[0] + room.width
    r_min_y = room.position[1]
    r_max_y = room.position[1] + room.depth

    # Overlap if ranges intersect in both X and Y (strict inequality).
    x_overlap = r_min_x < z_max_x and r_max_x > z_min_x
    y_overlap = r_min_y < z_max_y and r_max_y > z_min_y
    return x_overlap and y_overlap


def _violates_exterior_clearance(
    candidate: PlacedRoom,
    candidate_spec: RoomSpec,
    placed_rooms: list[PlacedRoom],
    room_spec_map: dict[str, RoomSpec],
    clearance: float,
) -> bool:
    """Check if placement violates any exterior_walls clearance zones.

    Checks bidirectionally:
    - Candidate cannot be in any placed room's clearance zones.
    - Placed rooms cannot be in candidate's clearance zones.

    Args:
        candidate: Room placement candidate to check.
        candidate_spec: Specification for the candidate room.
        placed_rooms: Already placed rooms.
        room_spec_map: Map of room_id to RoomSpec for all rooms.
        clearance: Clearance distance in meters.

    Returns:
        True if placement violates exterior_walls constraints.
    """
    # Check if candidate is in any placed room's clearance zones.
    for placed in placed_rooms:
        if placed.level_id != candidate.level_id:
            continue
        placed_spec = room_spec_map.get(placed.room_id)
        if placed_spec and placed_spec.exterior_walls:
            zones = _get_exterior_clearance_zones(
                room=placed, spec=placed_spec, clearance=clearance
            )
            for zone in zones:
                if _overlaps_zone(room=candidate, zone=zone):
                    return True

    # Check if placed rooms are in candidate's clearance zones.
    if candidate_spec.exterior_walls:
        zones = _get_exterior_clearance_zones(
            room=candidate, spec=candidate_spec, clearance=clearance
        )
        for zone in zones:
            for placed in placed_rooms:
                if placed.level_id != candidate.level_id:
                    continue
                if _overlaps_zone(room=placed, zone=zone):
                    return True

    return False


def rooms_share_edge(
    room_a: PlacedRoom, room_b: PlacedRoom, min_overlap: float = 0.0
) -> bool:
    """Check if two rooms share an edge with at least min_overlap length.

    Args:
        room_a: First room.
        room_b: Second room.
        min_overlap: Minimum shared edge length.

    Returns:
        True if rooms share sufficient edge.
    """
    if room_a.level_id != room_b.level_id:
        return False

    # Room A bounds.
    a_min_x = room_a.position[0]
    a_max_x = room_a.position[0] + room_a.width
    a_min_y = room_a.position[1]
    a_max_y = room_a.position[1] + room_a.depth

    # Room B bounds.
    b_min_x = room_b.position[0]
    b_max_x = room_b.position[0] + room_b.width
    b_min_y = room_b.position[1]
    b_max_y = room_b.position[1] + room_b.depth

    # Check for shared vertical edge (A's east = B's west or vice versa).
    # Also check that rooms actually overlap in Y (not just touching corners).
    if abs(a_max_x - b_min_x) < 0.001 or abs(b_max_x - a_min_x) < 0.001:
        # Y ranges must have positive overlap (not just touching at a point).
        y_overlap_start = max(a_min_y, b_min_y)
        y_overlap_end = min(a_max_y, b_max_y)
        overlap = y_overlap_end - y_overlap_start
        if overlap > 0.001 and overlap >= min_overlap:
            return True

    # Check for shared horizontal edge (A's north = B's south or vice versa).
    # Also check that rooms actually overlap in X (not just touching corners).
    if abs(a_max_y - b_min_y) < 0.001 or abs(b_max_y - a_min_y) < 0.001:
        # X ranges must have positive overlap (not just touching at a point).
        x_overlap_start = max(a_min_x, b_min_x)
        x_overlap_end = min(a_max_x, b_max_x)
        overlap = x_overlap_end - x_overlap_start
        if overlap > 0.001 and overlap >= min_overlap:
            return True

    return False


@dataclass
class SharedEdge:
    """Describes the shared edge between two rooms."""

    wall_direction: WallDirection
    """Direction of the shared wall from room_a's perspective."""

    position_along_wall: float
    """Distance from wall start to where overlap begins (meters)."""

    width: float
    """Length of the overlapping segment (meters)."""


def get_shared_edge(room_a: PlacedRoom, room_b: PlacedRoom) -> SharedEdge | None:
    """Compute the shared edge segment between two rooms.

    Args:
        room_a: First room (perspective room for wall direction).
        room_b: Second room.

    Returns:
        SharedEdge describing the overlap, or None if rooms don't share an edge.
    """
    # Room A bounds.
    a_min_x = room_a.position[0]
    a_max_x = room_a.position[0] + room_a.width
    a_min_y = room_a.position[1]
    a_max_y = room_a.position[1] + room_a.depth

    # Room B bounds.
    b_min_x = room_b.position[0]
    b_max_x = room_b.position[0] + room_b.width
    b_min_y = room_b.position[1]
    b_max_y = room_b.position[1] + room_b.depth

    # Check for shared vertical edge.
    # A's east = B's west (room_b is to the east of room_a).
    if abs(a_max_x - b_min_x) < 0.001:
        y_overlap_start = max(a_min_y, b_min_y)
        y_overlap_end = min(a_max_y, b_max_y)
        overlap = y_overlap_end - y_overlap_start
        if overlap > 0.001:
            # Room A's east wall. Wall runs from a_min_y to a_max_y (south to north).
            position_along_wall = y_overlap_start - a_min_y
            return SharedEdge(
                wall_direction=WallDirection.EAST,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    # B's east = A's west (room_b is to the west of room_a).
    if abs(b_max_x - a_min_x) < 0.001:
        y_overlap_start = max(a_min_y, b_min_y)
        y_overlap_end = min(a_max_y, b_max_y)
        overlap = y_overlap_end - y_overlap_start
        if overlap > 0.001:
            # Room A's west wall. Wall runs from a_min_y to a_max_y (south to north).
            position_along_wall = y_overlap_start - a_min_y
            return SharedEdge(
                wall_direction=WallDirection.WEST,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    # Check for shared horizontal edge.
    # A's north = B's south (room_b is to the north of room_a).
    if abs(a_max_y - b_min_y) < 0.001:
        x_overlap_start = max(a_min_x, b_min_x)
        x_overlap_end = min(a_max_x, b_max_x)
        overlap = x_overlap_end - x_overlap_start
        if overlap > 0.001:
            # Room A's north wall. Wall runs from a_min_x to a_max_x (west to east).
            position_along_wall = x_overlap_start - a_min_x
            return SharedEdge(
                wall_direction=WallDirection.NORTH,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    # B's north = A's south (room_b is to the south of room_a).
    if abs(b_max_y - a_min_y) < 0.001:
        x_overlap_start = max(a_min_x, b_min_x)
        x_overlap_end = min(a_max_x, b_max_x)
        overlap = x_overlap_end - x_overlap_start
        if overlap > 0.001:
            # Room A's south wall. Wall runs from a_min_x to a_max_x (west to east).
            position_along_wall = x_overlap_start - a_min_x
            return SharedEdge(
                wall_direction=WallDirection.SOUTH,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    return None
