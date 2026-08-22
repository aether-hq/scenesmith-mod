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

from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec

console_logger = logging.getLogger(__name__)

from scenesmith.floor_plan_agents.tools.submission.placement.geometry import (
    rooms_share_edge,
)
from scenesmith.floor_plan_agents.tools.submission.placement.models import (
    LayoutScoreBreakdown,
    PlacementConfig,
)


def _global_layout_score(
    placed_rooms: list[PlacedRoom], config: PlacementConfig
) -> LayoutScoreBreakdown:
    """Score a complete layout (higher = better).

    Considers:
    - Compactness: ratio of total room area to bounding box area
    - Stability: distance from previous positions (if provided)

    Args:
        placed_rooms: Complete room layout to score.
        config: Placement configuration with scoring weights and previous positions.

    Returns:
        Score breakdown with compactness, stability, and total scores.
    """
    if not placed_rooms:
        return LayoutScoreBreakdown(compactness=0.0, stability=0.0, total=0.0)

    weights = config.scoring_weights

    # Compactness score: room_area / bounding_box_area.
    total_room_area = sum(r.width * r.depth for r in placed_rooms)
    min_x = min(r.position[0] for r in placed_rooms)
    max_x = max(r.position[0] + r.width for r in placed_rooms)
    min_y = min(r.position[1] for r in placed_rooms)
    max_y = max(r.position[1] + r.depth for r in placed_rooms)
    bounding_box_area = (max_x - min_x) * (max_y - min_y)

    # Avoid division by zero (shouldn't happen with valid rooms).
    if bounding_box_area < 0.001:
        compactness_ratio = 1.0
    else:
        compactness_ratio = total_room_area / bounding_box_area

    # Scale scores to comparable ranges so neither dominates when weights are equal.
    # Compactness: ratio in [0, 1] -> scaled to [0, 100].
    compactness_score = compactness_ratio * 100.0 * weights.compactness

    # Stability: sum of per-room bonuses. Each room contributes up to 50 points
    # when at its previous position, decaying exponentially with distance.
    stability_score = 0.0
    if config.previous_positions and weights.stability > 0:
        for room in placed_rooms:
            if room.room_id in config.free_room_ids:
                continue  # Free rooms don't contribute to stability.
            if room.room_id in config.previous_positions:
                prev_pos = config.previous_positions[room.room_id]
                # Distance between room centers.
                curr_center = (
                    room.position[0] + room.width / 2,
                    room.position[1] + room.depth / 2,
                )
                prev_center = (
                    prev_pos[0] + room.width / 2,
                    prev_pos[1] + room.depth / 2,
                )
                distance = math.sqrt(
                    (curr_center[0] - prev_center[0]) ** 2
                    + (curr_center[1] - prev_center[1]) ** 2
                )
                # Exponential decay: full bonus (50) at distance=0, ~37% at distance=2m.
                scale = 2.0
                stability_score += (
                    math.exp(-distance / scale) * 50.0 * weights.stability
                )

    return LayoutScoreBreakdown(
        compactness=compactness_score,
        stability=stability_score,
        total=compactness_score + stability_score,
    )


def _topological_sort_rooms(room_specs: list[RoomSpec]) -> list[RoomSpec]:
    """Sort rooms for optimal placement order.

    Strategy for linear layouts like A-B-C where B connects A and C:
    1. Place anchor rooms (no adjacencies, not blocking connectors) first
    2. Place connector rooms (rooms with adjacencies, at least one satisfied) next
    3. Place remaining anchor rooms that were blocked by connectors last

    This ensures connectors like B are placed before their unmet deps (like C),
    so C can be placed adjacent to B rather than adjacent to A.

    Args:
        room_specs: List of room specifications.

    Returns:
        Sorted list optimized for successful placement.
    """
    if not room_specs:
        return []

    # Build adjacency lookup.
    sorted_list: list[RoomSpec] = []
    placed_ids: set[str] = set()
    remaining = list(room_specs)

    while remaining:
        # Capture remaining specs before sort (list.sort() empties list temporarily).
        remaining_snapshot = list(remaining)

        # Score each remaining room for placement priority.
        # Lower score = higher priority.
        def placement_score(spec: RoomSpec) -> tuple[int, int, float]:
            # Count how many adjacencies are already placed.
            placed_adj = sum(1 for a in spec.connections if a in placed_ids)
            total_adj = len(spec.connections)
            unmet = total_adj - placed_adj

            # Find reverse dependencies: unplaced rooms that require this spec.
            # Use snapshot because 'remaining' is empty during sort.
            reverse_deps = [
                s for s in remaining_snapshot if spec.room_id in s.connections
            ]

            # For rooms with no forward adjacencies.
            if total_adj == 0:
                # Check if any unplaced connector room requires us and is ready.
                # A connector is "ready" if it has at least one dep already placed.
                for r_spec in reverse_deps:
                    r_placed = sum(1 for a in r_spec.connections if a in placed_ids)
                    if r_placed > 0:
                        # This connector is ready - it should be placed before us.
                        # We wait so the connector can be positioned first.
                        return (2, 0, -(spec.width * spec.length))
                # No ready connectors blocking us - we're a true anchor.
                return (0, 0, -(spec.width * spec.length))

            # Priority 1: At least one adjacency placed (can attach to graph).
            if placed_adj > 0:
                return (1, unmet, -(spec.width * spec.length))

            # Priority 3: All adjacencies unplaced.
            # These need their deps placed first.
            return (3, total_adj, -(spec.width * spec.length))

        # Sort remaining by placement score.
        remaining.sort(key=placement_score)

        # Place the best candidate.
        chosen = remaining[0]
        sorted_list.append(chosen)
        placed_ids.add(chosen.room_id)
        remaining.remove(chosen)

    return sorted_list


def _score_placement(
    room: PlacedRoom,
    spec: RoomSpec,
    placed_rooms: list[PlacedRoom],
    config: PlacementConfig,
    reverse_adj_map: dict[str, list[str]] | None = None,
) -> float:
    """Score a placement based on adjacency satisfaction and compactness.

    Args:
        room: Proposed room placement.
        spec: Room specification with requirements.
        placed_rooms: Already placed rooms.
        config: Placement configuration.
        reverse_adj_map: Map of room_id to list of rooms that require it.

    Returns:
        Score (higher is better), 0 if invalid.
    """
    score = 100.0  # Base score.

    # Check required adjacencies (all connections require physical adjacency).
    required_adjacent = list(spec.connections.keys())

    if required_adjacent:
        satisfied = 0
        for adj_id in required_adjacent:
            adj_room = next((r for r in placed_rooms if r.room_id == adj_id), None)
            if adj_room and rooms_share_edge(room, adj_room, config.min_shared_edge):
                satisfied += 1
                score += 50.0  # Bonus for each satisfied adjacency.

        # If any required adjacency is not satisfied, reject.
        if satisfied < len(required_adjacent):
            # Check if the unmatched adjacencies are to unplaced rooms.
            placed_ids = {r.room_id for r in placed_rooms}
            unplaced_adjacencies = [a for a in required_adjacent if a not in placed_ids]
            # If all unsatisfied adjacencies are to unplaced rooms, still valid.
            if satisfied < len(required_adjacent) - len(unplaced_adjacencies):
                return 0.0  # Required adjacency to placed room not satisfied.

    # Check reverse adjacencies: placed rooms that require this room.
    # Give bonus for satisfying these (e.g., C placed adjacent to B when B needs C).
    if reverse_adj_map:
        requiring_rooms = reverse_adj_map.get(spec.room_id, [])
        for req_id in requiring_rooms:
            req_room = next((r for r in placed_rooms if r.room_id == req_id), None)
            if req_room and rooms_share_edge(room, req_room, config.min_shared_edge):
                score += 75.0  # Higher bonus for satisfying reverse adjacency.

    # Compactness bonus: prefer positions closer to center of mass.
    # This is a local heuristic for candidate ordering; global compactness is
    # evaluated by _global_layout_score on complete layouts.
    if placed_rooms:
        center_x = sum(r.position[0] + r.width / 2 for r in placed_rooms) / len(
            placed_rooms
        )
        center_y = sum(r.position[1] + r.depth / 2 for r in placed_rooms) / len(
            placed_rooms
        )
        room_center_x = room.position[0] + room.width / 2
        room_center_y = room.position[1] + room.depth / 2
        distance = math.sqrt(
            (room_center_x - center_x) ** 2 + (room_center_y - center_y) ** 2
        )
        score -= distance * 2  # Penalize distance from center.

    return score
