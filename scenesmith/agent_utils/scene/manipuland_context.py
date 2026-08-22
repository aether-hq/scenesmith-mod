"""Deterministic manipuland context and furniture proximity analysis."""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID


@dataclass(frozen=True)
class _ManipulandDesignContext:
    """Canonical scene intent used by deterministic detail selection."""

    rich_collection: bool
    prompt_constraints: str
    style_notes: str


_MANIPULAND_SURFACE_RULES: tuple[tuple[tuple[str, ...], str, int], ...] = (
    (
        ("medical bed", "med bed", "hospital bed", "treatment bed"),
        "one pillow",
        100,
    ),
    (("nightstand", "bedside table"), "reading light, book, personal item", 90),
    (("desk", "workstation"), "task light, notebook, compact work accessories", 85),
    (
        ("dining table", "coffee table", "side table", "table"),
        "a sparse set of context-appropriate tabletop objects",
        80,
    ),
    (
        ("counter", "worktop", "island"),
        "a sparse set of functional counter objects",
        75,
    ),
    (
        ("shelf", "bookcase", "cabinet", "dresser", "console"),
        "a few context-appropriate display or storage objects",
        70,
    ),
    (("bench",), "one or two context-appropriate loose objects", 50),
)

_MANIPULAND_SURFACE_EXCLUSIONS = (
    "chair",
    "stool",
    "sofa",
    "couch",
    "lamp",
    "light",
    "monitor",
    "screen",
    "television",
    "plant",
    "door",
    "window",
    "painting",
    "mirror",
)


def deterministic_manipuland_assignment(obj: Any) -> tuple[str, int] | None:
    """Return a stable small-object assignment for a likely support surface."""
    searchable = f"{getattr(obj, 'name', '')} {getattr(obj, 'description', '')}".lower()
    if any(keyword in searchable for keyword in _MANIPULAND_SURFACE_EXCLUSIONS):
        return None
    for keywords, suggested_items, priority in _MANIPULAND_SURFACE_RULES:
        if any(keyword in searchable for keyword in keywords):
            return suggested_items, priority
    return None


def _load_manipuland_design_context(
    scene: RoomScene,
) -> _ManipulandDesignContext | None:
    """Load the persisted scene blueprint adjacent to a room checkpoint."""

    room_dir = Path(scene.scene_dir)
    blueprint_paths = (
        room_dir / "scene_blueprint.json",
        room_dir.parent / "scene_blueprint.json",
    )
    blueprint: dict[str, Any] | None = None
    for blueprint_path in blueprint_paths:
        try:
            blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
            break
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            continue
    if blueprint is None:
        return None

    source_prompt = str(blueprint.get("source_prompt") or scene.text_description)
    furniture_groups = blueprint.get("furniture_groups") or []
    rich_collection = "thousands of books" in source_prompt.casefold()
    for group in furniture_groups:
        roles = group.get("roles") or {}
        if (
            int(roles.get("bookshelf", 0)) >= 12
            and str(group.get("density", "")).casefold() == "layered"
        ):
            rich_collection = True

    tokens = blueprint.get("design_tokens") or {}
    style_keywords = [str(value) for value in tokens.get("style_keywords") or []]
    palette = [str(value) for value in tokens.get("palette") or []]
    material_roles = {
        str(role): str(value)
        for role, value in (tokens.get("material_roles") or {}).items()
    }
    focal_hierarchy = [str(value) for value in tokens.get("focal_hierarchy") or []]
    lighting_mood = str(tokens.get("lighting_mood") or "")
    ornate = any(
        keyword in " ".join(style_keywords).casefold()
        for keyword in ("renaissance", "ornate", "grand")
    )

    sections = []
    if style_keywords:
        sections.append(f"style: {', '.join(style_keywords)}")
    if palette:
        sections.append(f"palette: {', '.join(palette)}")
    if material_roles:
        sections.append(
            "materials: "
            + ", ".join(f"{role}={value}" for role, value in material_roles.items())
        )
    if lighting_mood:
        sections.append(f"lighting: {lighting_mood}")
    if focal_hierarchy:
        sections.append(f"focal hierarchy: {', '.join(focal_hierarchy)}")
    if not sections:
        sections.append(f"room prompt: {source_prompt[:240]}")
    style_notes = "Canonical scene design: " + "; ".join(sections)
    if ornate:
        style_notes = (
            "Richly layered canonical scene design. Avoid sparse functional "
            "treatment; reinforce ornate authored details. " + "; ".join(sections)
        )
    return _ManipulandDesignContext(
        rich_collection=rich_collection,
        prompt_constraints=f"Explicit scene requirements: {source_prompt}",
        style_notes=style_notes,
    )


def _select_surfaces_across_levels(
    candidates: list[tuple[int, str, Any, str]],
    *,
    limit: int,
) -> list[tuple[int, str, Any, str]]:
    """Round-robin deterministic support surfaces across authored stories."""

    by_level: dict[float, list[tuple[int, str, Any, str]]] = {}
    for candidate in candidates:
        obj = candidate[2]
        try:
            elevation = round(float(obj.transform.translation()[2]), 3)
        except (AttributeError, IndexError, TypeError, ValueError):
            elevation = 0.0
        by_level.setdefault(elevation, []).append(candidate)

    selected: list[tuple[int, str, Any, str]] = []
    while len(selected) < limit and any(by_level.values()):
        for elevation in sorted(by_level):
            if by_level[elevation]:
                selected.append(by_level[elevation].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def _compute_aabb_edge_distance(
    bounds_a: tuple[np.ndarray, np.ndarray],
    bounds_b: tuple[np.ndarray, np.ndarray],
) -> float:
    """Compute minimum XY-plane distance between two AABBs (edge-to-edge).

    Returns 0.0 if boxes overlap in XY plane.
    """
    min_a, max_a = bounds_a
    min_b, max_b = bounds_b

    # For each axis, compute gap (negative means overlap).
    dx = max(min_a[0] - max_b[0], min_b[0] - max_a[0], 0.0)
    dy = max(min_a[1] - max_b[1], min_b[1] - max_a[1], 0.0)

    return float(np.sqrt(dx**2 + dy**2))


def _compute_direction(from_center: np.ndarray, to_center: np.ndarray) -> str:
    """Compute 8-way direction from XY delta.

    Y+ = NORTH, X+ = EAST (room coordinates).
    """
    dx = to_center[0] - from_center[0]
    dy = to_center[1] - from_center[1]

    angle = np.degrees(np.arctan2(dy, dx))  # 0° = EAST, 90° = NORTH.
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    index = int((angle + 22.5) % 360 / 45)
    return directions[index]


def compute_nearby_furniture_candidates(
    scene: RoomScene,
    target_furniture_id: UniqueID,
    distance_threshold_m: float = 2.0,
    max_candidates: int = 30,
) -> list[dict]:
    """Compute nearby furniture with distance and direction.

    Uses edge-to-edge AABB distance (XY plane only) for proximity.

    Args:
        scene: RoomScene with all furniture.
        target_furniture_id: Furniture to find candidates around.
        distance_threshold_m: Max distance to consider.
        max_candidates: Max candidates to return.

    Returns:
        List of candidate dicts sorted by distance:
        {
            "furniture_id": str,
            "name": str,
            "description": str,
            "distance_m": float,
            "direction": str,  # "N", "NE", "E", etc.
        }
    """
    target = scene.get_object(target_furniture_id)
    if target is None:
        return []

    target_bounds = target.compute_world_bounds()
    if target_bounds is None:
        return []

    target_center = (target_bounds[0] + target_bounds[1]) / 2

    candidates = []
    placeable_types = (ObjectType.FURNITURE, ObjectType.WALL_MOUNTED)

    for obj in scene.objects.values():
        # Skip self and non-furniture.
        if obj.object_id == target_furniture_id:
            continue
        if obj.object_type not in placeable_types:
            continue
        if obj.immutable:
            continue

        obj_bounds = obj.compute_world_bounds()
        if obj_bounds is None:
            continue

        distance = _compute_aabb_edge_distance(target_bounds, obj_bounds)
        if distance > distance_threshold_m:
            continue

        obj_center = (obj_bounds[0] + obj_bounds[1]) / 2
        direction = _compute_direction(target_center, obj_center)

        candidates.append(
            {
                "furniture_id": str(obj.object_id),
                "name": obj.name,
                "description": obj.description or "",
                "distance_m": round(distance, 2),
                "direction": direction,
            }
        )

    # Sort by distance, limit count.
    candidates.sort(key=lambda c: c["distance_m"])
    return candidates[:max_candidates]
