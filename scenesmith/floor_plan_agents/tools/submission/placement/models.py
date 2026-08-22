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

from dataclasses import dataclass, field

from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom, WallDirection

console_logger = logging.getLogger(__name__)


class PlacementError(Exception):
    """Raised when room placement fails."""


@dataclass
class ScoringWeights:
    """Weights for global layout scoring."""

    compactness: float = 1.0
    """Weight for bounding box minimization (higher = prefer compact layouts)."""

    stability: float = 1.0
    """Weight for staying near previous positions (higher = more stable)."""


@dataclass
class LayoutScoreBreakdown:
    """Breakdown of layout scoring components (all values are weighted)."""

    compactness: float
    stability: float
    total: float


@dataclass
class Slot:
    """An attachment slot on a placed room's edge.

    A slot represents an available edge where a new room can attach.
    """

    room_id: str
    """ID of the room that owns this slot."""

    direction: WallDirection
    """Direction of this slot (N/S/E/W edge of the room)."""

    start: float
    """Start coordinate along the slot edge."""

    end: float
    """End coordinate along the slot edge."""

    anchor_pos: tuple[float, float]
    """Position of the room that owns this slot (for reference)."""

    anchor_width: float
    """Width of the room that owns this slot."""

    anchor_depth: float
    """Depth of the room that owns this slot."""


@dataclass
class PlacementConfig:
    """Configuration for room placement algorithm."""

    min_shared_edge: float = 1.0
    """Minimum shared edge length for adjacency (meters)."""

    timeout_seconds: float = 5.0
    """Timeout for backtracking search. Returns best layout found when exceeded."""

    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    """Weights for global layout scoring (compactness, stability)."""

    previous_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Previous room positions for layout stability. Map of room_id to (x, y) position."""

    free_room_ids: set[str] = field(default_factory=set)
    """Room IDs that should have no position bias (can move freely).
    Typically includes rooms being resized or made adjacent."""

    exterior_wall_clearance_m: float = 20.0
    """Clearance zone for exterior_walls constraint (meters).

    Rooms with exterior_walls specified will have clearance zones created
    extending this distance outward from the specified walls. No other room
    can be placed within these zones, ensuring the walls remain accessible
    for exterior doors.
    """


@dataclass
class _SearchState:
    """Mutable state for backtracking search."""

    best_layout: list[PlacedRoom] | None = None
    """Best complete layout found so far."""

    best_score: float = float("-inf")
    """Score of the best layout."""

    start_time: float = 0.0
    """Search start time (from time.time())."""

    timed_out: bool = False
    """Whether the search has exceeded the timeout."""

    timeout_seconds: float = 5.0
    """Maximum search time before returning best-found-so-far."""
