"""Provider-neutral normalization for one-shot floor-plan submissions.

Model tool arguments are design intent, not trusted internal state.  This module
accepts common OpenAI/Anthropic/open-model shapes and converts them to the small
canonical boundary consumed by :class:`FloorPlanTools`.
"""

from __future__ import annotations

import ast
import json
import math
import re

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


_MULTILEVEL_PATTERN = re.compile(
    r"\b(multi[- ]?level|multi[- ]?story|multiple levels?|two[- ]?story|"
    r"three[- ]?story|two[- ]?levels?|three[- ]?levels?|four[- ]?levels?|"
    r"mezzanine|platforms?|upper floor|spiral stairs?|stairs?|staircases?|ramp)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizedFloorPlanSubmission:
    room_specs: list[dict[str, Any]]
    wall_height_meters: float
    structural: dict[str, Any] | None
    windows_per_room: int
    floor_material_description: str
    wall_material_description: str
    exterior_material_description: str
    exterior_door_room_id: str
    repairs: tuple[str, ...] = ()

    def tool_kwargs(self) -> dict[str, Any]:
        """Return only arguments accepted by the deterministic executor."""

        return {
            "room_specs": self.room_specs,
            "wall_height_meters": self.wall_height_meters,
            "structural": self.structural,
            "windows_per_room": self.windows_per_room,
            "floor_material_description": self.floor_material_description,
            "wall_material_description": self.wall_material_description,
            "exterior_material_description": self.exterior_material_description,
            "exterior_door_room_id": self.exterior_door_room_id,
        }


def _snake_key(value: Any) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _jsonish(value: Any) -> Any:
    """Parse JSON/Python-literal wrappers without invoking a model repair turn."""

    if not isinstance(value, str):
        return value
    candidate = value.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    if not candidate or candidate[0] not in "[{(":
        return value
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return value


def _mapping(value: Any) -> dict[str, Any] | None:
    parsed = _jsonish(value)
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _keyed(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {_snake_key(key): value for key, value in mapping.items()}


def _pick(mapping: Mapping[str, Any], *aliases: str, default: Any = None) -> Any:
    keyed = _keyed(mapping)
    for alias in aliases:
        key = _snake_key(alias)
        if key in keyed and keyed[key] is not None:
            return keyed[key]
    return default


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        candidate = float(value)
    else:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or ""))
        if not match:
            return default
        candidate = float(match.group())
    return candidate if math.isfinite(candidate) else default


def _bounded_number(value: Any, default: float, low: float, high: float) -> float:
    return max(low, min(high, _number(value, default)))


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    normalized = str(value).strip()
    return normalized or default


def _infer_room_type(prompt: str) -> str:
    categories = (
        "library",
        "bedroom",
        "bathroom",
        "kitchen",
        "office",
        "classroom",
        "laboratory",
        "hospital_room",
        "restaurant",
        "bar",
        "gallery",
        "warehouse",
        "living_room",
    )
    normalized = prompt.lower().replace(" ", "_")
    return next((category for category in categories if category in normalized), "room")


def _normalize_dimensions(
    room: Mapping[str, Any], *, default_width: float, default_depth: float
) -> tuple[Any, Any]:
    dimensions = _jsonish(_pick(room, "dimensions", "dimension", "size", "floor_size"))
    width = _pick(room, "width", "length", "x_size", "size_x")
    depth = _pick(room, "depth", "y_size", "size_y")
    if isinstance(dimensions, Mapping):
        width = (
            width if width is not None else _pick(dimensions, "width", "x", "length")
        )
        depth = depth if depth is not None else _pick(dimensions, "depth", "y")
    elif isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2:
        width = width if width is not None else dimensions[0]
        depth = depth if depth is not None else dimensions[1]
    elif isinstance(dimensions, str):
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", dimensions)
        if len(matches) >= 2:
            width = width if width is not None else matches[0]
            depth = depth if depth is not None else matches[1]
    return (
        width if width is not None else default_width,
        depth if depth is not None else default_depth,
    )


def _normalize_connections(value: Any) -> dict[str, str]:
    parsed = _jsonish(value)
    if isinstance(parsed, list):
        parsed = {str(room): "DOOR" for room in parsed}
    if not isinstance(parsed, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for room_id, connection in parsed.items():
        raw = _snake_key(connection)
        normalized[str(room_id)] = (
            "OPEN"
            if raw
            in {
                "open",
                "open_plan",
                "opening",
                "archway",
            }
            else "DOOR"
        )
    return normalized


def _normalize_exterior_walls(value: Any) -> list[str]:
    parsed = _jsonish(value)
    if isinstance(parsed, str):
        parsed = re.split(r"[,\s]+", parsed)
    if not isinstance(parsed, (list, tuple, set)):
        return []
    aliases = {
        "n": "north",
        "s": "south",
        "e": "east",
        "w": "west",
        "top": "north",
        "bottom": "south",
        "right": "east",
        "left": "west",
    }
    result = []
    for direction in parsed:
        key = _snake_key(direction)
        canonical = aliases.get(key, key)
        if canonical in {"north", "south", "east", "west"} and canonical not in result:
            result.append(canonical)
    return result


def _normalize_rooms(
    value: Any,
    *,
    prompt: str,
    mode: Literal["room", "house"],
    dim_min: float,
    dim_max: float,
    repairs: list[str],
) -> list[dict[str, Any]]:
    parsed = _jsonish(value)
    if isinstance(parsed, Mapping):
        keyed = _keyed(parsed)
        room_fields = {"type", "room_type", "width", "depth", "dimensions", "size"}
        if room_fields.intersection(keyed):
            parsed = [dict(parsed)]
        else:
            parsed = [
                {"type": room_id, **(dict(spec) if isinstance(spec, Mapping) else {})}
                for room_id, spec in parsed.items()
            ]
            repairs.append("converted room map to room array")
    elif isinstance(parsed, str):
        parsed = [{"type": _infer_room_type(parsed), "prompt": parsed}]
        repairs.append("converted room description to room specification")
    elif not isinstance(parsed, list):
        parsed = []

    default_width, default_depth = (
        (12.0, 10.0) if "large" in prompt.lower() else (8.0, 6.0)
    )
    rooms: list[dict[str, Any]] = []
    for index, raw_room in enumerate(parsed):
        if isinstance(raw_room, str):
            raw_room = {"type": _infer_room_type(raw_room), "prompt": raw_room}
        if not isinstance(raw_room, Mapping):
            repairs.append(f"ignored non-object room entry {index + 1}")
            continue
        authored_room_type = _pick(
            raw_room, "type", "room_type", "category", "id", "name"
        )
        if not isinstance(authored_room_type, str) or not re.search(
            r"[A-Za-z]", authored_room_type
        ):
            authored_room_type = _infer_room_type(prompt)
            repairs.append(f"repaired invalid room type at entry {index + 1}")
        room_type = _text(authored_room_type, _infer_room_type(prompt))
        room_type = _snake_key(room_type) or f"room_{index + 1}"
        width, depth = _normalize_dimensions(
            raw_room,
            default_width=default_width,
            default_depth=default_depth,
        )
        room = {
            "type": room_type,
            "width": _bounded_number(width, default_width, dim_min, dim_max),
            "depth": _bounded_number(depth, default_depth, dim_min, dim_max),
            "prompt": _text(
                _pick(raw_room, "prompt", "description", "details", "contents"),
                prompt,
            ),
        }
        overhead_value = _pick(
            raw_room,
            "has_overhead_cover",
            "has_roof",
            "has_ceiling",
            "covered",
        )
        if isinstance(overhead_value, bool):
            room["has_overhead_cover"] = overhead_value
        connections = _normalize_connections(
            _pick(raw_room, "connections", "adjacencies", "connected_to")
        )
        if connections:
            room["connections"] = connections
        exterior_walls = _normalize_exterior_walls(
            _pick(raw_room, "exterior_walls", "external_walls", "outside_walls")
        )
        if exterior_walls:
            room["exterior_walls"] = exterior_walls
        rooms.append(room)

    if not rooms:
        rooms = [
            {
                "type": _infer_room_type(prompt),
                "width": max(dim_min, min(dim_max, default_width)),
                "depth": max(dim_min, min(dim_max, default_depth)),
                "prompt": prompt,
            }
        ]
        repairs.append("synthesized missing room specification from the scene prompt")
    if mode == "room" and len(rooms) > 1:
        primary = max(rooms, key=lambda room: room["width"] * room["depth"])
        primary["prompt"] = prompt
        rooms = [primary]
        repairs.append("collapsed multiple room-mode entries into one logical space")
    return rooms


def synthesize_structural_layout(
    prompt: str,
    room_specs: list[dict[str, Any]],
    storey_height: float,
    *,
    max_total_height: float = 12.0,
    level_count_hint: int | None = None,
) -> dict[str, Any] | None:
    """Build conservative walkable levels, slabs, and stairs from prompt intent.

    ``storey_height`` is often authored as a per-storey height while the legacy
    room shell validates one total height. Fit the requested number of levels
    into that envelope before emitting geometry; otherwise the deterministic
    fallback would repeat the provider's invalid height and silently flatten.
    """

    if not _MULTILEVEL_PATTERN.search(prompt):
        return None
    lowered = prompt.lower()
    level_count = (
        4
        if "four" in lowered
        else (
            3
            if any(token in lowered for token in ("three", "multi", "multiple"))
            or re.search(
                r"\btwo\s+(?:raised\s+)?(?:mezzanine\s+)?platforms?\b",
                lowered,
            )
            else 2
        )
    )
    if level_count_hint is not None:
        level_count = max(level_count, int(level_count_hint))
    level_count = max(2, min(4, level_count))
    requested_level_count = level_count
    requested_storey_height = float(storey_height)
    diagnostics: list[str] = []
    minimum_storey_height = 2.8
    maximum_storey_height = 4.5
    max_total_height = max(2 * minimum_storey_height, float(max_total_height))
    max_feasible_levels = max(2, int(max_total_height // minimum_storey_height))
    level_count = min(level_count, max_feasible_levels)
    height = max(
        minimum_storey_height,
        min(
            maximum_storey_height, float(storey_height), max_total_height / level_count
        ),
    )
    if level_count < requested_level_count:
        diagnostics.append(
            f"Reduced the requested {requested_level_count} levels to {level_count} "
            f"because the {max_total_height:g}m shell cannot provide safe headroom."
        )
    elif requested_storey_height * level_count > max_total_height + 1e-9:
        diagnostics.append(
            f"Reduced each storey from {requested_storey_height:g}m to {height:g}m "
            f"to fit {level_count} usable levels inside the {max_total_height:g}m shell."
        )
    room = room_specs[0]
    room_id = str(room["type"])
    width = float(room["width"])
    depth = float(room["depth"])
    replacement_target = re.search(
        r"\b(?:replace|swap|change)\b.*?\b(?:with|to)\b(?P<target>.+)$",
        lowered,
    )
    family_text = replacement_target.group("target") if replacement_target else lowered
    requested_family = (
        "stairs_spiral"
        if "spiral" in family_text
        else (
            "stairs_u"
            if any(
                token in family_text
                for token in (
                    "switchback",
                    "switch-back",
                    "u-shaped",
                    "u shaped",
                    "u-stair",
                    "u stair",
                    "double-back",
                    "double back",
                )
            )
            else (
                "stairs_l"
                if any(
                    token in family_text
                    for token in (
                        "l-shaped",
                        "l shaped",
                        "l-stair",
                        "l stair",
                        "quarter-turn",
                        "quarter turn",
                    )
                )
                else "stairs_straight"
            )
        )
    )
    stair_width = 1.1
    riser_count = max(12, round(height / 0.18))
    tread_depth = 0.28
    clearance_padding = 0.2
    # A landing must overlap the surrounding slab by enough to read and behave
    # as a real destination. The stairwell includes tread-width clearance past
    # each endpoint, so a short 0.9 m landing left only a ~0.15 m connection.
    landing_depth = 1.5

    # Author stair paths in a local long/short-axis frame, then map them back to
    # room XY.  This keeps ordinary stairs viable in portrait and landscape rooms.
    swap_axes = depth > width
    long_extent, short_extent = (depth, width) if swap_axes else (width, depth)

    def room_xy(along: float, across: float) -> list[float]:
        return [across, along] if swap_axes else [along, across]

    def point(along: float, across: float, elevation: float) -> list[float]:
        return [*room_xy(along, across), elevation]

    # Connector centerlines are authored in the house frame, while platforms
    # are welded to the room's center frame. Platform footprints therefore use
    # the same centered-local convention as compiled room geometry.
    def platform_xy(point_xy: Sequence[float]) -> list[float]:
        return [point_xy[0] - width / 2.0, point_xy[1] - depth / 2.0]

    margin = landing_depth + stair_width / 2.0 + clearance_padding
    family = requested_family
    half_risers = riser_count // 2
    remaining_risers = riser_count - half_risers
    first_run = half_risers * tread_depth
    second_run = remaining_risers * tread_depth
    straight_run = riser_count * tread_depth
    switchback_gap = stair_width + 0.35

    if family == "stairs_straight" and long_extent - 2 * margin < riser_count * 0.22:
        family = "stairs_spiral"
    elif family == "stairs_l" and (
        long_extent - 2 * margin < first_run or short_extent - 2 * margin < second_run
    ):
        family = "stairs_u"
    if family == "stairs_u" and (
        long_extent - 2 * margin < max(first_run, second_run)
        or short_extent - 2 * margin < switchback_gap
    ):
        family = "stairs_spiral"
    if family != requested_family:
        diagnostics.append(
            f"Changed {requested_family} to {family} because the room footprint "
            "cannot fit safe tread and landing geometry for the requested form."
        )

    spiral_radius = max(
        stair_width / 2.0 + 0.25,
        riser_count * 0.22 / (2.0 * math.pi),
    )
    spiral_outer_radius = spiral_radius + stair_width / 2.0
    spiral_center = [
        max(
            spiral_outer_radius + clearance_padding,
            min(
                width - spiral_outer_radius - clearance_padding,
                width * 0.25,
            ),
        ),
        max(
            spiral_outer_radius + clearance_padding,
            min(
                depth - spiral_outer_radius - clearance_padding,
                depth * 0.25,
            ),
        ),
    ]
    levels = [
        {"id": f"level_{index}", "elevation": index * height, "nominal_height": height}
        for index in range(level_count)
    ]
    connectors = []
    clearance_points: list[list[float]] = []
    landing_platforms: list[dict[str, Any]] = []
    for index in range(level_count - 1):
        bottom = index * height
        top = (index + 1) * height
        mid = bottom + height * half_risers / riser_count
        start = point(margin, margin, bottom)
        end = point(margin + straight_run, margin, top)
        parameters: dict[str, Any] = {"riser_count": riser_count}

        if family == "stairs_spiral":
            start = [
                spiral_center[0] + spiral_radius,
                spiral_center[1],
                bottom,
            ]
            end = [
                spiral_center[0] + spiral_radius,
                spiral_center[1],
                top,
            ]
            parameters.update(
                {
                    "center": spiral_center,
                    "radius": spiral_radius,
                    "turns": 1.0,
                    "direction": "cw",
                }
            )
            clearance_points.extend(
                [
                    [
                        spiral_center[0] - spiral_outer_radius,
                        spiral_center[1] - spiral_outer_radius,
                    ],
                    [
                        spiral_center[0] + spiral_outer_radius,
                        spiral_center[1] + spiral_outer_radius,
                    ],
                ]
            )
        elif family == "stairs_l":
            turn = point(margin + first_run, margin, mid)
            end = point(margin + first_run, margin + second_run, top)
            parameters = {
                "waypoints": [turn],
                "riser_counts": [half_risers, remaining_risers],
                "landing_length": stair_width,
            }
            clearance_points.extend([start[:2], turn[:2], end[:2]])
        elif family == "stairs_u":
            first_turn = point(margin + first_run, margin, mid)
            second_turn = point(
                margin + first_run,
                margin + switchback_gap,
                mid,
            )
            end = point(margin, margin + switchback_gap, top)
            parameters = {
                "waypoints": [first_turn, second_turn],
                "riser_counts": [half_risers, remaining_risers],
            }
            clearance_points.extend(
                [start[:2], first_turn[:2], second_turn[:2], end[:2]]
            )
        else:
            clearance_points.extend([start[:2], end[:2]])

        if index % 2 == 1 and family != "stairs_spiral":
            previous_start = start
            start = [*end[:2], bottom]
            end = [*previous_start[:2], top]
            if family == "stairs_l":
                parameters["waypoints"] = [[*parameters["waypoints"][0][:2], mid]]
                parameters["riser_counts"] = list(reversed(parameters["riser_counts"]))
            elif family == "stairs_u":
                parameters["waypoints"] = [
                    [*waypoint[:2], mid]
                    for waypoint in reversed(parameters["waypoints"])
                ]
                parameters["riser_counts"] = list(reversed(parameters["riser_counts"]))

        last_waypoint = parameters.get("waypoints", ())[-1:] or [start]
        approach_from = last_waypoint[0]
        if family == "stairs_spiral":
            direction_x = end[0] - spiral_center[0]
            direction_y = end[1] - spiral_center[1]
        else:
            direction_x = end[0] - approach_from[0]
            direction_y = end[1] - approach_from[1]
        direction_length = max(math.hypot(direction_x, direction_y), 1e-9)
        direction_x /= direction_length
        direction_y /= direction_length
        perpendicular_x, perpendicular_y = -direction_y, direction_x
        landing_half_width = stair_width / 2.0 + 0.1
        landing_start = [
            end[0] - direction_x * 0.08,
            end[1] - direction_y * 0.08,
        ]
        landing_end = [
            end[0] + direction_x * landing_depth,
            end[1] + direction_y * landing_depth,
        ]
        landing_outer = [
            [
                landing_start[0] - perpendicular_x * landing_half_width,
                landing_start[1] - perpendicular_y * landing_half_width,
            ],
            [
                landing_end[0] - perpendicular_x * landing_half_width,
                landing_end[1] - perpendicular_y * landing_half_width,
            ],
            [
                landing_end[0] + perpendicular_x * landing_half_width,
                landing_end[1] + perpendicular_y * landing_half_width,
            ],
            [
                landing_start[0] + perpendicular_x * landing_half_width,
                landing_start[1] + perpendicular_y * landing_half_width,
            ],
        ]
        landing_platforms.append(
            {
                "id": f"level_{index + 1}_stair_landing",
                "space_id": room_id,
                "footprint": {
                    "outer": [platform_xy(point_xy) for point_xy in landing_outer],
                    "holes": [],
                },
                "elevation": top,
                "thickness": 0.2,
                "traversable": True,
            }
        )

        connector: dict[str, Any] = {
            "id": f"fallback_stairs_{index + 1}",
            "type": family,
            "start": {
                "space_id": room_id,
                "level_id": f"level_{index}",
                "position": start,
            },
            "end": {
                "space_id": room_id,
                "level_id": f"level_{index + 1}",
                "position": end,
            },
            "width": stair_width,
            "parameters": parameters,
        }
        connectors.append(connector)

    # Each upper datum needs actual walkable geometry. Use a full floor slab with
    # a stairwell void rather than treating the level list as metadata or adding a
    # decorative furniture staircase. Clockwise hole winding is intentional.
    inset = min(0.15, width * 0.01, depth * 0.01)
    outer_house = [
        [inset, inset],
        [width - inset, inset],
        [width - inset, depth - inset],
        [inset, depth - inset],
    ]
    outer = [platform_xy(point_xy) for point_xy in outer_house]
    # Spiral clearance points already describe the outer tread edge. Straight,
    # L, and U paths describe their centerlines and still need half-width added.
    footprint_half_width = 0.0 if family == "stairs_spiral" else stair_width / 2.0
    min_x = max(
        inset + 0.02,
        min(point[0] for point in clearance_points)
        - footprint_half_width
        - clearance_padding,
    )
    max_x = min(
        width - inset - 0.02,
        max(point[0] for point in clearance_points)
        + footprint_half_width
        + clearance_padding,
    )
    min_y = max(
        inset + 0.02,
        min(point[1] for point in clearance_points)
        - footprint_half_width
        - clearance_padding,
    )
    max_y = min(
        depth - inset - 0.02,
        max(point[1] for point in clearance_points)
        + footprint_half_width
        + clearance_padding,
    )
    hole_house = [
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ]
    hole = [platform_xy(point_xy) for point_xy in hole_house]
    platforms = [
        {
            "id": f"level_{index}_walkable_slab",
            "space_id": room_id,
            "footprint": {"outer": outer, "holes": [hole]},
            "elevation": index * height,
            "thickness": 0.2,
            "traversable": True,
        }
        for index in range(1, level_count)
    ] + landing_platforms
    return {
        "levels": levels,
        "rooms": [{"id": room_id, "level_id": "level_0"}],
        "connectors": connectors,
        "platforms": platforms,
        "_diagnostics": diagnostics,
    }


def normalize_floor_plan_submission(
    raw: Mapping[str, Any] | None,
    *,
    prompt: str,
    mode: Literal["room", "house"],
    room_dim_min: float,
    room_dim_max: float,
    wall_height_min: float,
    wall_height_max: float,
) -> NormalizedFloorPlanSubmission:
    """Normalize arbitrary provider tool arguments into the canonical submission."""

    repairs: list[str] = []
    source = dict(raw or {})
    for envelope_name in (
        "design",
        "plan",
        "floor_plan",
        "floorplan",
        "scene",
        "arguments",
    ):
        envelope = _mapping(_pick(source, envelope_name))
        if envelope is not None:
            source = {**envelope, **source}
            repairs.append(f"unwrapped {envelope_name} envelope")
            break

    rooms_value = _pick(source, "room_specs", "rooms", "spaces", "room_specifications")
    rooms = _normalize_rooms(
        rooms_value,
        prompt=prompt,
        mode=mode,
        dim_min=room_dim_min,
        dim_max=room_dim_max,
        repairs=repairs,
    )

    authored_height = _pick(
        source,
        "wall_height_meters",
        "wall_height",
        "ceiling_height",
        "storey_height",
        "story_height",
    )
    wall_height = _bounded_number(
        authored_height, 3.2, wall_height_min, wall_height_max
    )

    structural_value = _pick(
        source,
        "structural",
        "structural_layout",
        "structure",
        "architecture",
    )
    structural = _mapping(structural_value)
    if structural is None:
        structural_fields = {
            key: value
            for key, value in source.items()
            if _snake_key(key)
            in {
                "levels",
                "connectors",
                "platforms",
                "portals",
                "heightfields",
                "structural_meshes",
                "semantic_environment",
            }
        }
        structural = structural_fields or None
    if structural is None:
        structural = synthesize_structural_layout(
            prompt,
            rooms,
            wall_height,
            max_total_height=wall_height_max,
        )
        if structural is not None:
            repairs.append("synthesized missing multi-level structure from the prompt")
            repairs.extend(structural.get("_diagnostics", ()))

    materials = _mapping(_pick(source, "materials", "material", "finishes")) or {}
    floor_material = _text(
        _pick(
            source,
            "floor_material_description",
            "floor_material",
            "floor_finish",
            default=_pick(materials, "floor", "floor_material"),
        ),
        "warm wood floor",
    )
    wall_material = _text(
        _pick(
            source,
            "wall_material_description",
            "wall_material",
            "wall_finish",
            default=_pick(materials, "wall", "walls", "wall_material"),
        ),
        "neutral plaster wall",
    )
    exterior_material = _text(
        _pick(
            source,
            "exterior_material_description",
            "exterior_material",
            "facade_material",
            default=_pick(materials, "exterior", "facade"),
        ),
        "neutral exterior plaster",
    )
    windows = round(
        _bounded_number(
            _pick(source, "windows_per_room", "window_count", "windows"),
            2,
            0,
            8,
        )
    )
    exterior_door_room = _text(
        _pick(
            source,
            "exterior_door_room_id",
            "entrance_room",
            "entry_room",
        ),
        str(rooms[0]["type"]),
    )

    return NormalizedFloorPlanSubmission(
        room_specs=rooms,
        wall_height_meters=wall_height,
        structural=structural,
        windows_per_room=windows,
        floor_material_description=floor_material,
        wall_material_description=wall_material,
        exterior_material_description=exterior_material,
        exterior_door_room_id=exterior_door_room,
        repairs=tuple(dict.fromkeys(repairs)),
    )
