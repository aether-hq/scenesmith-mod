"""Normalization of loosely shaped model tool arguments."""

from __future__ import annotations

import ast
import json
import math
import re

from typing import Any, Literal, Mapping

from scenesmith.floor_plan_agents.tools.submission.structural_submission import (
    synthesize_structural_layout,
)
from scenesmith.floor_plan_agents.tools.submission.submission_models import (
    NormalizedFloorPlanSubmission,
)


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
    folded_prompt = prompt.casefold()
    prompt_arched = bool(re.search(r"\barch[a-z]{0,5}\s+windows?\b", folded_prompt))
    prompt_huge = bool(
        re.search(
            r"\b(?:huge|grand|monumental|oversized)\b.{0,30}\bwindows?\b",
            folded_prompt,
        )
    )
    default_window_count = (
        3
        if (prompt_arched or prompt_huge) and re.search(r"\bwindows\b", folded_prompt)
        else 2
    )
    windows = round(
        _bounded_number(
            _pick(source, "windows_per_room", "window_count", "windows"),
            default_window_count,
            0,
            8,
        )
    )
    raw_window_shape = _snake_key(
        _pick(source, "window_shape", "windows_shape", default="")
    )
    window_shape: Literal["rectangular", "arched"] = (
        "arched"
        if raw_window_shape in {"arch", "arched"} or prompt_arched
        else "rectangular"
    )
    window_width = _bounded_number(
        _pick(source, "window_width_m", "window_width"),
        4.0 if prompt_huge else 1.2,
        0.6,
        4.0,
    )
    window_height = _bounded_number(
        _pick(source, "window_height_m", "window_height"),
        3.5 if prompt_huge else 1.2,
        0.6,
        4.0,
    )
    window_sill_height = _bounded_number(
        _pick(source, "window_sill_height_m", "window_sill_height", "sill_height"),
        0.35 if prompt_huge else 0.9,
        0.0,
        2.0,
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
        window_shape=window_shape,
        window_width_m=window_width,
        window_height_m=window_height,
        window_sill_height_m=window_sill_height,
        floor_material_description=floor_material,
        wall_material_description=wall_material,
        exterior_material_description=exterior_material,
        exterior_door_room_id=exterior_door_room,
        repairs=tuple(dict.fromkeys(repairs)),
    )
