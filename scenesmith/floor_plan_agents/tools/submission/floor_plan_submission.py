"""Provider-neutral normalization for one-shot floor-plan submissions.

Model tool arguments are design intent, not trusted internal state.  This module
accepts common OpenAI/Anthropic/open-model shapes and converts them to the small
canonical boundary consumed by :class:`FloorPlanTools`.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
        SceneBlueprint,
    )


_MULTILEVEL_PATTERN = re.compile(
    r"\b(multi[- ]?level|multi[- ]?story|multiple levels?|two[- ]?story|"
    r"three[- ]?story|two[- ]?levels?|three[- ]?levels?|four[- ]?levels?|"
    r"mezzanine|platforms?|upper floor|spiral stairs?|stairs?|staircases?|ramp)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BlueprintOpeningPlacement:
    """One blueprint aperture assigned to a concrete rectangular wall edge."""

    opening_id: str
    kind: Literal["door", "window", "open_connection"]
    host_space_id: str
    connects_to_space_id: str | None
    boundary_edge_index: int
    position_along_m: float
    width_m: float
    height_m: float
    sill_height_m: float
    shape: Literal["rectangular", "arched"]


def opening_placements_from_blueprint(
    blueprint: "SceneBlueprint",
) -> tuple[BlueprintOpeningPlacement, ...]:
    """Assign immutable blueprint openings to non-overlapping wall intervals.

    Metric scene scale is intentionally not capped here. An opening is rejected
    only when the accepted space itself is too small to contain it.
    """

    spaces = {space.space_id: space for space in blueprint.spaces}
    intervals_by_edge: dict[tuple[str, int], list[tuple[float, float]]] = {}
    placements: list[BlueprintOpeningPlacement] = []
    separation_m = 0.5

    def preferred_edges(kind: str) -> tuple[int, ...]:
        if kind == "open_connection":
            return (0, 2, 1, 3)
        if kind == "door":
            return (2, 1, 3, 0)
        return (1, 3, 2, 0)

    def available_center(
        *, space_id: str, edge_index: int, edge_length: float, width: float
    ) -> float | None:
        occupied = sorted(intervals_by_edge.get((space_id, edge_index), ()))
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in occupied:
            gap_end = max(cursor, start - separation_m)
            if gap_end > cursor:
                gaps.append((cursor, gap_end))
            cursor = max(cursor, end + separation_m)
        if cursor < edge_length:
            gaps.append((cursor, edge_length))
        feasible = [gap for gap in gaps if gap[1] - gap[0] + 1e-9 >= width]
        if not feasible:
            return None
        wall_center = edge_length / 2.0
        gap_start, gap_end = min(
            feasible,
            key=lambda gap: abs((gap[0] + gap[1]) / 2.0 - wall_center),
        )
        return max(gap_start + width / 2.0, min(wall_center, gap_end - width / 2.0))

    ordered = sorted(
        blueprint.openings,
        key=lambda opening: (
            opening.kind != "open_connection",
            -(opening.width_m * opening.height_m),
            opening.opening_id,
        ),
    )
    for opening in ordered:
        space = spaces[opening.host_space_id]
        edge_lengths = (
            float(space.dimensions_m[0]),
            float(space.dimensions_m[1]),
            float(space.dimensions_m[0]),
            float(space.dimensions_m[1]),
        )
        selected: tuple[int, float] | None = None
        for edge_index in preferred_edges(opening.kind):
            center = available_center(
                space_id=space.space_id,
                edge_index=edge_index,
                edge_length=edge_lengths[edge_index],
                width=float(opening.width_m),
            )
            if center is not None:
                selected = (edge_index, center)
                break
        if selected is None:
            raise ValueError(
                f"space {space.space_id!r} cannot fit blueprint opening "
                f"{opening.opening_id!r} ({opening.width_m:g}m wide) on any wall"
            )
        edge_index, center = selected
        intervals_by_edge.setdefault((space.space_id, edge_index), []).append(
            (center - opening.width_m / 2.0, center + opening.width_m / 2.0)
        )
        placements.append(
            BlueprintOpeningPlacement(
                opening_id=opening.opening_id,
                kind=opening.kind,
                host_space_id=opening.host_space_id,
                connects_to_space_id=opening.connects_to_space_id,
                boundary_edge_index=edge_index,
                position_along_m=center,
                width_m=opening.width_m,
                height_m=opening.height_m,
                sill_height_m=opening.sill_height_m,
                shape=opening.shape,
            )
        )
    return tuple(placements)
