"""Compile Aether's accepted single-room contract into native SceneSmith topology.

The semantic design and layout solver have already done the architectural work by
the time this module runs.  Re-prompting the floor-plan agent would permit an
approved shell or opening to drift, so this boundary constructs the same
``HouseLayout`` objects the agent tools produce and lets the remaining SceneSmith
stages complete the scenic design.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.house import (
    HouseLayout,
    Opening,
    OpeningType,
    RoomSpec,
    WallDirection,
)
from scenesmith.floor_plan_agents.tools.room_placement import create_placed_room


class AcceptedLayoutError(ValueError):
    """The accepted contract cannot be represented without changing its meaning."""


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptedLayoutError(f"{label} must be an object")
    return value


def _opening_type(opening: Mapping[str, Any]) -> OpeningType:
    mechanism = opening.get("mechanism")
    sill = float(opening.get("sill_m", 0.0))
    if mechanism == "fixed-glazing" or sill > 0:
        return OpeningType.WINDOW
    # Legacy OPEN means "remove this wall from floor to ceiling" and ignores
    # the authored opening height.  A mechanism-free doorway is therefore a
    # DOOR aperture with no leaf asset; this preserves its exact lintel.
    return OpeningType.DOOR


def _validate_contract(stage_input: Mapping[str, Any]) -> None:
    if stage_input.get("realization_engine") != "scenesmith":
        raise AcceptedLayoutError("accepted layout requires realization_engine=scenesmith")
    if stage_input.get("pipeline_profile") != "full":
        raise AcceptedLayoutError("accepted layout requires the full SceneSmith profile")
    if stage_input.get("people_allowed") is not False:
        raise AcceptedLayoutError("scenic realization cannot place people")


def build_accepted_house_layout(
    stage_input: Mapping[str, Any], *, house_dir: Path
) -> HouseLayout:
    """Build exact native room topology from one validated Aether stage input.

    SceneSmith's current wall compiler represents rectangular apertures.  Curved
    apertures fail loudly instead of being silently flattened into rectangles.
    """
    _validate_contract(stage_input)
    request = _required_mapping(stage_input.get("request"), "request")
    shell = _required_mapping(request.get("shell"), "request.shell")
    dimensions = shell.get("dimensions_m")
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3:
        raise AcceptedLayoutError("request.shell.dimensions_m must contain width, height, depth")
    width, height, depth = (float(value) for value in dimensions)
    if min(width, height, depth) <= 0:
        raise AcceptedLayoutError("accepted shell dimensions must be positive")

    room_id = str(shell.get("room_id", "")).strip()
    if not room_id:
        raise AcceptedLayoutError("request.shell.room_id is required")
    prompt = str(stage_input.get("room_prompt", "")).strip()
    if not prompt:
        raise AcceptedLayoutError("room_prompt is required")

    room_spec = RoomSpec(
        room_id=room_id,
        room_type="scenic_environment",
        prompt=prompt,
        position=(0.0, 0.0),
        width=depth,
        length=width,
        exterior_walls=set(WallDirection),
    )
    placed_room = create_placed_room(room_spec, (0.0, 0.0))
    walls = {wall.direction.value: wall for wall in placed_room.walls}
    seen_opening_ids: set[str] = set()
    for raw_opening in shell.get("openings", []):
        opening = _required_mapping(raw_opening, "request.shell.openings[]")
        opening_id = str(opening.get("opening_id", "")).strip()
        if not opening_id or opening_id in seen_opening_ids:
            raise AcceptedLayoutError("accepted opening ids must be present and unique")
        seen_opening_ids.add(opening_id)
        shape = opening.get("shape", "rectangle")
        if shape != "rectangle":
            raise AcceptedLayoutError(
                f"opening {opening_id} uses unsupported exact shape {shape!r}; "
                "the native SceneSmith wall compiler must gain that shape before realization"
            )
        boundary = str(opening.get("boundary", ""))
        wall = walls.get(boundary)
        if wall is None:
            raise AcceptedLayoutError(f"opening {opening_id} has unknown boundary {boundary!r}")
        width_m = float(opening["width_m"])
        height_m = float(opening["height_m"])
        offset_m = float(opening["offset_m"])
        sill_m = float(opening.get("sill_m", 0.0))
        if offset_m < 0 or offset_m + width_m > wall.length + 1e-6:
            raise AcceptedLayoutError(f"opening {opening_id} extends past the {boundary} wall")
        if sill_m < 0 or sill_m + height_m > height + 1e-6:
            raise AcceptedLayoutError(f"opening {opening_id} extends above the accepted shell")
        wall.openings.append(
            Opening(
                opening_id=opening_id,
                opening_type=_opening_type(opening),
                position_along_wall=offset_m,
                width=width_m,
                height=height_m,
                sill_height=sill_m,
            )
        )

    return HouseLayout(
        wall_height=height,
        house_prompt=prompt,
        room_specs=[room_spec],
        placed_rooms=[placed_room],
        house_dir=house_dir,
        placement_valid=True,
        connectivity_valid=True,
    )


def accepted_wall_thickness(stage_input: Mapping[str, Any]) -> float:
    """Return the authored wall thickness for the geometry compiler."""
    request = _required_mapping(stage_input.get("request"), "request")
    shell = _required_mapping(request.get("shell"), "request.shell")
    value = float(shell.get("wall_thickness_m", 0.0))
    if not 0.02 < value <= 1.0:
        raise AcceptedLayoutError("accepted wall thickness must be in (0.02, 1.0] metres")
    return value
