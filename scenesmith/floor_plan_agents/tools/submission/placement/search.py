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
import time

from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom, WallDirection
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec

console_logger = logging.getLogger(__name__)

from scenesmith.floor_plan_agents.tools.submission.placement.geometry import (
    _has_overlap,
    _violates_exterior_clearance,
)
from scenesmith.floor_plan_agents.tools.submission.placement.layout import (
    _get_room_slots,
    create_placed_room,
    update_wall_connectivity,
)
from scenesmith.floor_plan_agents.tools.submission.placement.models import (
    PlacementConfig,
    PlacementError,
    Slot,
    _SearchState,
)
from scenesmith.floor_plan_agents.tools.submission.placement.scoring import (
    _global_layout_score,
    _score_placement,
    _topological_sort_rooms,
)


def place_rooms(
    room_specs: list[RoomSpec], config: PlacementConfig | None = None
) -> list[PlacedRoom]:
    """Place rooms to satisfy adjacency constraints.

    Uses backtracking search with timeout to find globally optimal (or near-optimal)
    room layouts. Returns the best layout found within the timeout.

    For layout stability during iterative edits, pass previous_positions and
    free_room_ids in config. The algorithm will prefer positions close to
    previous locations for non-free rooms.

    Args:
        room_specs: List of room specifications with dimensions and adjacencies.
        config: Placement configuration. If None, uses default PlacementConfig.

    Returns:
        List of PlacedRoom with computed positions and walls.

    Raises:
        PlacementError: If no valid layout can be found.
    """
    if config is None:
        config = PlacementConfig()

    if not room_specs:
        return []

    # v2 layouts place each level independently in XY.  This permits stacked
    # rooms with identical footprints while retaining the proven legacy
    # rectangle search within each level.  Cross-level adjacency must be
    # represented by a ConnectorSpec rather than a planar door/open edge.
    spec_lookup = {spec.room_id: spec for spec in room_specs}
    for spec in room_specs:
        for connected_room_id in spec.connections:
            connected = spec_lookup.get(connected_room_id)
            if connected is not None and connected.level_id != spec.level_id:
                raise PlacementError(
                    f"Rooms '{spec.room_id}' and '{connected_room_id}' are on "
                    "different levels; use a structural connector instead of "
                    "a planar room connection"
                )

    level_groups: dict[str, list[RoomSpec]] = {}
    for spec in room_specs:
        if spec.footprint is not None:
            raise PlacementError(
                f"Room '{spec.room_id}' has an arbitrary footprint. Polygon "
                "placement is not available in the legacy rectangle placer."
            )
        if abs(spec.yaw) > 1e-9:
            raise PlacementError(
                f"Room '{spec.room_id}' has yaw={spec.yaw}. Rotated room "
                "placement is not available in the legacy rectangle placer."
            )
        level_groups.setdefault(spec.level_id, []).append(spec)

    if len(level_groups) > 1:
        placed: list[PlacedRoom] = []
        for level_specs in level_groups.values():
            if len(level_specs) == 1:
                placed.append(
                    create_placed_room(spec=level_specs[0], position=(0.0, 0.0))
                )
            else:
                placed.extend(
                    _place_rooms_attempt(room_specs=level_specs, config=config)
                )
        update_wall_connectivity(placed)
        return placed

    if len(room_specs) == 1:
        # Single room: place at origin.
        spec = room_specs[0]
        return [create_placed_room(spec=spec, position=(0.0, 0.0))]

    return _place_rooms_attempt(room_specs=room_specs, config=config)


def _place_rooms_attempt(
    room_specs: list[RoomSpec], config: PlacementConfig
) -> list[PlacedRoom]:
    """Find optimal room placement using backtracking with timeout.

    Args:
        room_specs: Room specifications.
        config: Placement configuration.

    Returns:
        List of placed rooms (best layout found within timeout).

    Raises:
        PlacementError: If no valid layout found.
    """
    # Topological sort: place rooms with fewer/no dependencies first.
    sorted_specs = _topological_sort_rooms(room_specs)

    # Build reverse adjacency map: which rooms require each room.
    reverse_adj_map: dict[str, list[str]] = {s.room_id: [] for s in room_specs}
    for s in room_specs:
        for adj_id in s.connections:
            if adj_id in reverse_adj_map:
                reverse_adj_map[adj_id].append(s.room_id)

    # Build room spec map for exterior_walls constraint checking.
    room_spec_map: dict[str, RoomSpec] = {s.room_id: s for s in room_specs}

    # Place first room at origin (should have no/fewest adjacencies).
    first_spec = sorted_specs[0]
    first_room = create_placed_room(first_spec, (0.0, 0.0))
    initial_placed = [first_room]
    initial_slots = _get_room_slots(first_room)

    # Initialize search state.
    start_time = time.time()
    state = _SearchState(start_time=start_time, timeout_seconds=config.timeout_seconds)

    # Run backtracking search.
    _place_rooms_backtrack(
        remaining_specs=sorted_specs[1:],
        placed_rooms=initial_placed,
        available_slots=initial_slots,
        config=config,
        reverse_adj_map=reverse_adj_map,
        room_spec_map=room_spec_map,
        state=state,
    )

    # Calculate search duration.
    elapsed_time = time.time() - start_time

    if state.best_layout is None:
        console_logger.warning(
            f"Room placement failed after {elapsed_time:.2f}s. "
            f"Rooms: {[s.room_id for s in room_specs]}. Timed out: {state.timed_out}"
        )
        raise PlacementError(
            f"Room placement failed: no valid layout found. "
            f"Rooms: {[s.room_id for s in room_specs]}. "
            f"Timed out: {state.timed_out}"
        )

    # Log search results with score breakdown.
    score_breakdown = _global_layout_score(
        placed_rooms=state.best_layout, config=config
    )
    timeout_msg = "timed out" if state.timed_out else "completed"
    console_logger.info(
        f"Room placement {timeout_msg} in {elapsed_time:.2f}s "
        f"(compactness={score_breakdown.compactness:.1f}, "
        f"stability={score_breakdown.stability:.1f}, "
        f"total={score_breakdown.total:.1f})"
    )

    # Update wall connectivity.
    update_wall_connectivity(state.best_layout)

    return state.best_layout


def _get_all_candidates(
    spec: RoomSpec,
    placed_rooms: list[PlacedRoom],
    available_slots: list[Slot],
    config: PlacementConfig,
    reverse_adj_map: dict[str, list[str]] | None = None,
    room_spec_map: dict[str, RoomSpec] | None = None,
) -> list[tuple[PlacedRoom, float]]:
    """Get all valid placement candidates for a room.

    Tries both orientations (original and 90° rotated) to maximize placement options.

    Args:
        spec: Room specification to place.
        placed_rooms: Already placed rooms.
        available_slots: Available attachment slots.
        config: Placement configuration.
        reverse_adj_map: Map of room_id to list of rooms that require it.
        room_spec_map: Map of room_id to RoomSpec for exterior_walls checking.

    Returns:
        List of (PlacedRoom, score) tuples for all valid placements.
    """
    candidates: list[tuple[PlacedRoom, float]] = []

    # Determine if this is a multi-adjacency case (needs corner placement).
    required_placed = [r for r in placed_rooms if r.room_id in spec.connections]
    is_multi_adjacency = len(required_placed) >= 2

    # Try both orientations: original and 90° rotated.
    # Original: X=spec.length, Y=spec.width
    # Rotated:  X=spec.width, Y=spec.length
    orientations = [
        (spec.length, spec.width),  # Original orientation.
    ]
    # Only add rotated if dimensions differ (avoid duplicate work for square rooms).
    if abs(spec.length - spec.width) > 0.001:
        orientations.append((spec.width, spec.length))  # 90° rotated.

    for room_x, room_y in orientations:
        for slot in available_slots:
            # Check if this slot's owner is in spec's connections.
            if spec.connections and slot.room_id not in spec.connections:
                # If room has specific adjacency requirements, only consider matching
                # slots.
                continue

            # Try placing room at this slot with current orientation.
            # For multi-adjacency, also consider corner-aligned positions.
            positions = _get_candidate_positions(
                spec=spec,
                slot=slot,
                min_shared_edge=config.min_shared_edge,
                placed_rooms=placed_rooms if is_multi_adjacency else None,
                room_x=room_x,
                room_y=room_y,
            )

            for pos in positions:
                room = create_placed_room(
                    spec=spec, position=pos, room_width=room_x, room_depth=room_y
                )

                # Check for overlaps with existing rooms.
                if _has_overlap(room=room, placed_rooms=placed_rooms):
                    continue

                # Check exterior_walls clearance constraints.
                if room_spec_map and _violates_exterior_clearance(
                    candidate=room,
                    candidate_spec=spec,
                    placed_rooms=placed_rooms,
                    room_spec_map=room_spec_map,
                    clearance=config.exterior_wall_clearance_m,
                ):
                    continue

                # Check adjacencies with placed rooms.
                score = _score_placement(
                    room=room,
                    spec=spec,
                    placed_rooms=placed_rooms,
                    config=config,
                    reverse_adj_map=reverse_adj_map,
                )
                # Only add valid placements to candidates.
                if score > 0:
                    candidates.append((room, score))

    return candidates


def _place_rooms_backtrack(
    remaining_specs: list[RoomSpec],
    placed_rooms: list[PlacedRoom],
    available_slots: list[Slot],
    config: PlacementConfig,
    reverse_adj_map: dict[str, list[str]],
    room_spec_map: dict[str, RoomSpec],
    state: _SearchState,
) -> None:
    """Recursive backtracking search for optimal layout.

    Explores all valid layouts, updating state.best_layout when a better complete
    layout is found. Stops early if timeout is exceeded.

    Args:
        remaining_specs: Room specs still to be placed.
        placed_rooms: Rooms placed so far in this branch.
        available_slots: Available attachment slots.
        config: Placement configuration.
        reverse_adj_map: Map of room_id to list of rooms that require it.
        room_spec_map: Map of room_id to RoomSpec for exterior_walls checking.
        state: Mutable search state (best layout, score, timeout tracking).
    """
    # Check timeout.
    if time.time() - state.start_time > state.timeout_seconds:
        state.timed_out = True
        return

    # Base case: all rooms placed - score and possibly update best.
    if not remaining_specs:
        score = _global_layout_score(placed_rooms=placed_rooms, config=config)
        if score.total > state.best_score:
            state.best_score = score.total
            state.best_layout = list(placed_rooms)  # Copy the list.
        return

    # Get next room to place.
    spec = remaining_specs[0]
    rest = remaining_specs[1:]

    # Get all valid candidates for this room.
    candidates = _get_all_candidates(
        spec=spec,
        placed_rooms=placed_rooms,
        available_slots=available_slots,
        config=config,
        reverse_adj_map=reverse_adj_map,
        room_spec_map=room_spec_map,
    )

    # Sort candidates by score (highest first) for best-first exploration.
    candidates.sort(key=lambda x: -x[1])

    # Recurse on each candidate.
    for room, _ in candidates:
        if state.timed_out:
            return

        # Extend placed rooms and slots for this branch.
        new_placed = placed_rooms + [room]
        new_slots = available_slots + _get_room_slots(room)

        _place_rooms_backtrack(
            remaining_specs=rest,
            placed_rooms=new_placed,
            available_slots=new_slots,
            config=config,
            reverse_adj_map=reverse_adj_map,
            room_spec_map=room_spec_map,
            state=state,
        )


def _get_candidate_positions(
    spec: RoomSpec,
    slot: Slot,
    min_shared_edge: float,
    placed_rooms: list[PlacedRoom] | None = None,
    room_x: float | None = None,
    room_y: float | None = None,
) -> list[tuple[float, float]]:
    """Get candidate positions for placing a room at a slot.

    The room must share at least min_shared_edge with the slot's anchor room.
    Valid position range along the slot:
    - Horizontal slots (N/S): x varies from (anchor_x - room_x + min_shared_edge)
      to (anchor_x + anchor_width - min_shared_edge)
    - Vertical slots (E/W): y varies similarly

    Args:
        spec: Room to place.
        slot: Slot to attach to.
        min_shared_edge: Minimum shared edge length required.
        placed_rooms: If provided, also generate corner-aligned positions
            that could touch other placed rooms (for multi-adjacency).
        room_x: Explicit X dimension (overrides spec.length if provided).
        room_y: Explicit Y dimension (overrides spec.width if provided).

    Returns:
        List of candidate (x, y) positions.
    """
    positions = []

    # Room dimensions using length as x-dimension and width as y-dimension.
    if room_x is None:
        room_x = spec.length
    if room_y is None:
        room_y = spec.width

    # Calculate positions based on slot direction.
    if slot.direction == WallDirection.NORTH:
        # Slot is on north edge of anchor room.
        # New room attaches from north (its south edge touches slot).
        y = slot.anchor_pos[1] + slot.anchor_depth
        # Valid x range for sufficient edge sharing.
        x_min = slot.anchor_pos[0] - room_x + min_shared_edge
        x_max = slot.anchor_pos[0] + slot.anchor_width - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=x_min, var_max=x_max, fixed_coord=y, vary_axis="x"
            )
        )

    elif slot.direction == WallDirection.SOUTH:
        # New room attaches from south (its north edge touches slot).
        y = slot.anchor_pos[1] - room_y
        x_min = slot.anchor_pos[0] - room_x + min_shared_edge
        x_max = slot.anchor_pos[0] + slot.anchor_width - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=x_min, var_max=x_max, fixed_coord=y, vary_axis="x"
            )
        )

    elif slot.direction == WallDirection.EAST:
        # New room attaches from east (its west edge touches slot).
        x = slot.anchor_pos[0] + slot.anchor_width
        y_min = slot.anchor_pos[1] - room_y + min_shared_edge
        y_max = slot.anchor_pos[1] + slot.anchor_depth - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=y_min, var_max=y_max, fixed_coord=x, vary_axis="y"
            )
        )

    elif slot.direction == WallDirection.WEST:
        # New room attaches from west (its east edge touches slot).
        x = slot.anchor_pos[0] - room_x
        y_min = slot.anchor_pos[1] - room_y + min_shared_edge
        y_max = slot.anchor_pos[1] + slot.anchor_depth - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=y_min, var_max=y_max, fixed_coord=x, vary_axis="y"
            )
        )

    # Add corner-aligned positions for multi-adjacency.
    if placed_rooms:
        positions.extend(
            _get_corner_aligned_positions(
                spec=spec,
                slot=slot,
                placed_rooms=placed_rooms,
                room_x=room_x,
                room_y=room_y,
            )
        )

    return positions


def _generate_positions_in_range(
    var_min: float,
    var_max: float,
    fixed_coord: float,
    vary_axis: str,
) -> list[tuple[float, float]]:
    """Generate evenly spaced positions within a valid range.

    Args:
        var_min: Minimum value for the varying coordinate.
        var_max: Maximum value for the varying coordinate.
        fixed_coord: Fixed coordinate value.
        vary_axis: "x" if x varies, "y" if y varies.

    Returns:
        List of (x, y) positions.
    """
    positions = []

    # Generate positions at different points in the range.
    if var_max >= var_min:
        # Valid range exists.
        range_size = var_max - var_min
        # Generate 11 evenly spaced positions (0%, 10%, 20%, ..., 100%).
        for i in range(11):
            fraction = i / 10.0
            var_val = var_min + fraction * range_size
            if vary_axis == "x":
                positions.append((var_val, fixed_coord))
            else:
                positions.append((fixed_coord, var_val))

    return positions


def _get_corner_aligned_positions(
    spec: RoomSpec,
    slot: Slot,
    placed_rooms: list[PlacedRoom],
    room_x: float | None = None,
    room_y: float | None = None,
) -> list[tuple[float, float]]:
    """Get positions aligned with corners of other placed rooms.

    For multi-adjacency cases, tries to position the new room so it touches
    both the slot owner and other required adjacent rooms.

    Args:
        spec: Room to place.
        slot: Slot to attach to.
        placed_rooms: All placed rooms.
        room_x: Explicit X dimension (overrides spec.length if provided).
        room_y: Explicit Y dimension (overrides spec.width if provided).

    Returns:
        List of corner-aligned (x, y) positions.
    """
    positions = []
    if room_x is None:
        room_x = spec.length
    if room_y is None:
        room_y = spec.width

    # Find other rooms we need to be adjacent to.
    other_required = [
        r
        for r in placed_rooms
        if r.room_id in spec.connections and r.room_id != slot.room_id
    ]

    for other in other_required:
        # Calculate positions that would align our room's edges with the other room.
        other_left = other.position[0]
        other_right = other.position[0] + other.width
        other_bottom = other.position[1]
        other_top = other.position[1] + other.depth

        if slot.direction == WallDirection.NORTH:
            y = slot.anchor_pos[1] + slot.anchor_depth
            # Align left edge with other room's right edge.
            positions.append((other_right, y))
            # Align right edge with other room's left edge.
            positions.append((other_left - room_x, y))
            # Align left edge with other room's left edge.
            positions.append((other_left, y))
            # Align right edge with other room's right edge.
            positions.append((other_right - room_x, y))

        elif slot.direction == WallDirection.SOUTH:
            y = slot.anchor_pos[1] - room_y
            positions.append((other_right, y))
            positions.append((other_left - room_x, y))
            positions.append((other_left, y))
            positions.append((other_right - room_x, y))

        elif slot.direction == WallDirection.EAST:
            x = slot.anchor_pos[0] + slot.anchor_width
            # Align top edge with other room's bottom edge.
            positions.append((x, other_bottom - room_y))
            # Align bottom edge with other room's top edge.
            positions.append((x, other_top))
            # Align top edge with other room's top edge.
            positions.append((x, other_top - room_y))
            # Align bottom edge with other room's bottom edge.
            positions.append((x, other_bottom))

        elif slot.direction == WallDirection.WEST:
            x = slot.anchor_pos[0] - room_x
            positions.append((x, other_bottom - room_y))
            positions.append((x, other_top))
            positions.append((x, other_top - room_y))
            positions.append((x, other_bottom))

    return positions
