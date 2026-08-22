"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import logging
import math
import re

from collections.abc import Callable
from typing import Any

from scenesmith.agent_utils.assets.asset_semantics import (
    catalog_candidate_is_compatible,
    catalog_candidate_satisfies_request_details,
    tall_furniture_dimensions_are_compatible,
)
from scenesmith.agent_utils.design.room_kits import RoomKitSelection
from scenesmith.agent_utils.scene.clearance_zones import (
    compute_door_clearance_violations,
    compute_window_clearance_violations,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType

console_logger = logging.getLogger(__name__)


def _object_matches_room_kit_slot(obj: Any, slot: Any) -> bool:
    """Whether one furniture object satisfies a semantic room-kit role."""

    object_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                (
                    str(getattr(obj, "name", "")),
                    str(getattr(obj, "description", "")),
                    str(getattr(obj, "object_id", "")),
                )
            )
            .casefold()
            .replace("_", " "),
        )
    )
    role_match = False
    for role_name in (slot.role, *getattr(slot, "aliases", ())):
        role_tokens = set(
            re.findall(r"[a-z0-9]+", str(role_name).casefold().replace("_", " "))
        )
        if role_tokens and role_tokens <= object_tokens:
            role_match = True
            break
    if not role_match:
        return False

    metadata = getattr(obj, "metadata", None) or {}
    catalog_text = str(
        metadata.get("catalog_semantics") or metadata.get("ontology_path") or ""
    )
    if catalog_text:
        compatible, _ = catalog_candidate_is_compatible(
            request_text=str(getattr(slot, "query", slot.role)),
            candidate_text=catalog_text,
            quality_score=(
                float(metadata["asset_quality_score"])
                if metadata.get("asset_quality_score") is not None
                else None
            ),
        )
        if not compatible:
            return False
        compatible, _ = catalog_candidate_satisfies_request_details(
            request_text=str(getattr(slot, "query", slot.role)),
            candidate_text=catalog_text,
            supports_detail_fill=bool(metadata.get("support_zones")),
        )
        if not compatible:
            return False

    compatible_dimensions, _ = tall_furniture_dimensions_are_compatible(
        request_text=str(getattr(slot, "query", slot.role)),
        desired_dimensions=getattr(slot, "nominal_dimensions_m", None),
        bbox_min=getattr(obj, "bbox_min", None),
        bbox_max=getattr(obj, "bbox_max", None),
    )
    return compatible_dimensions


def _required_room_kit_level_coverage(
    scene_text: str,
    room_kit: RoomKitSelection,
    support_elevations: tuple[float, ...],
) -> dict[str, int]:
    """Return per-story role minimums for explicit dense multilevel libraries."""

    normalized = str(scene_text).casefold().replace("_", " ")
    explicit_multilevel = bool(
        re.search(
            r"\bmulti[ -]?level\b|\bmultiple (?:levels|floors|stories)\b",
            normalized,
        )
    )
    dense_library = (
        str(getattr(room_kit, "kit_id", "")) == "library-reading-hall-v1"
        and bool(re.search(r"\blarge\b", normalized))
        and bool(re.search(r"\bthousands?\b", normalized))
    )
    if not dense_library or not explicit_multilevel or len(support_elevations) < 2:
        return {}
    return {"bookshelf": 5, "reading_table": 1, "reading_chair": 3}


def _room_kit_role_level_counts(
    objects: Any,
    slot: Any,
    support_elevations: tuple[float, ...],
) -> dict[float, int]:
    """Count suitable role instances at their nearest authored support level."""

    counts = {elevation: 0 for elevation in support_elevations}
    if not counts:
        return counts
    for obj in objects:
        if getattr(
            obj, "object_type", None
        ) != ObjectType.FURNITURE or not _object_matches_room_kit_slot(obj, slot):
            continue
        try:
            object_elevation = float(obj.transform.translation()[2])
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        nearest = min(
            support_elevations,
            key=lambda elevation: abs(elevation - object_elevation),
        )
        counts[nearest] += 1
    return counts


def _nearest_level(obj: Any, support_elevations: tuple[float, ...]) -> float | None:
    if not support_elevations:
        return None
    try:
        object_elevation = float(obj.transform.translation()[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return min(
        support_elevations,
        key=lambda elevation: abs(elevation - object_elevation),
    )


def _stable_chair_faces_table(chair: Any, table: Any) -> bool:
    """Whether an active upright chair occupies the table's usable annulus."""

    if (
        str((getattr(chair, "metadata", None) or {}).get("placement_context", "active"))
        != "active"
    ):
        return False
    try:
        chair_position = chair.transform.translation()
        table_position = table.transform.translation()
        rotation = chair.transform.rotation().matrix()
        forward_x = float(rotation[0, 1])
        forward_y = float(rotation[1, 1])
        upright = float(rotation[2, 2]) >= math.cos(math.radians(15.0))
        dx = float(table_position[0]) - float(chair_position[0])
        dy = float(table_position[1]) - float(chair_position[1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    distance = math.hypot(dx, dy)
    if not upright or distance < 0.75 or distance > 2.25:
        return False
    alignment = (forward_x * dx + forward_y * dy) / distance
    return alignment >= 0.35


def _patron_ensemble_level_counts(
    objects: Any,
    table_slot: Any,
    chair_slot: Any,
    support_elevations: tuple[float, ...],
) -> dict[float, int]:
    """Count the largest coherent table/chair ensemble on every story."""

    tables_by_level = {elevation: [] for elevation in support_elevations}
    chairs_by_level = {elevation: [] for elevation in support_elevations}
    for obj in objects:
        if getattr(obj, "object_type", None) != ObjectType.FURNITURE:
            continue
        level = _nearest_level(obj, support_elevations)
        if level is None:
            continue
        if _object_matches_room_kit_slot(obj, table_slot):
            tables_by_level[level].append(obj)
        if _object_matches_room_kit_slot(obj, chair_slot):
            chairs_by_level[level].append(obj)
    return {
        elevation: max(
            (
                sum(
                    _stable_chair_faces_table(chair, table)
                    for chair in chairs_by_level[elevation]
                )
                for table in tables_by_level[elevation]
            ),
            default=0,
        )
        for elevation in support_elevations
    }


def _bookcase_wall_run_level_counts(
    objects: Any,
    bookshelf_slot: Any,
    support_elevations: tuple[float, ...],
) -> dict[float, int]:
    """Return the largest contiguous, consistently oriented bookcase run/story."""

    by_level = {elevation: [] for elevation in support_elevations}
    for obj in objects:
        if getattr(
            obj, "object_type", None
        ) != ObjectType.FURNITURE or not _object_matches_room_kit_slot(
            obj, bookshelf_slot
        ):
            continue
        level = _nearest_level(obj, support_elevations)
        if level is not None:
            metadata = getattr(obj, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                obj.metadata = metadata
            metadata.pop("dense_library_grouped_run", None)
            by_level[level].append(obj)

    def pose_and_footprint(obj: Any) -> tuple[float, float, float, float, float] | None:
        try:
            translation = obj.transform.translation()
            yaw = float(obj.transform.rotation().ToRollPitchYaw().yaw_angle())
            x_size = abs(float(obj.bbox_max[0]) - float(obj.bbox_min[0]))
            y_size = abs(float(obj.bbox_max[1]) - float(obj.bbox_min[1]))
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        return (
            float(translation[0]),
            float(translation[1]),
            yaw,
            max(x_size, y_size, 0.5),
            max(min(x_size, y_size), 0.15),
        )

    largest_runs: dict[float, int] = {}
    for elevation, level_objects in by_level.items():
        posed = [
            (obj, pose)
            for obj in level_objects
            if (pose := pose_and_footprint(obj)) is not None
        ]
        adjacency: dict[int, set[int]] = {index: set() for index in range(len(posed))}
        for left_index, (_left_obj, left) in enumerate(posed):
            for right_index in range(left_index + 1, len(posed)):
                _right_obj, right = posed[right_index]
                yaw_delta = abs(
                    (left[2] - right[2] + math.pi) % (2.0 * math.pi) - math.pi
                )
                if yaw_delta > math.radians(15.0):
                    continue
                dx = right[0] - left[0]
                dy = right[1] - left[1]
                along = abs(math.cos(left[2]) * dx + math.sin(left[2]) * dy)
                across = abs(-math.sin(left[2]) * dx + math.cos(left[2]) * dy)
                if (
                    along >= 0.45 * min(left[3], right[3])
                    and along <= (left[3] + right[3]) / 2.0 + 0.35
                    and across <= (left[4] + right[4]) / 2.0 + 0.15
                ):
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)

        largest = 0
        largest_component: list[int] = []
        remaining = set(adjacency)
        while remaining:
            stack = [remaining.pop()]
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = adjacency[current] & remaining
                remaining.difference_update(neighbors)
                stack.extend(neighbors)
            if len(component) > largest:
                largest = len(component)
                largest_component = component
        if largest >= 3:
            for index in largest_component:
                posed[index][0].metadata["dense_library_grouped_run"] = elevation
        largest_runs[elevation] = largest
    return largest_runs


def _required_room_kit_role_count(room_kit: Any, slot: Any) -> int:
    """Return the room-sized required count, falling back to the slot minimum."""

    slot_counts = getattr(room_kit, "slot_counts", None)
    if isinstance(slot_counts, dict) and slot.role in slot_counts:
        return max(int(slot.minimum_count), int(slot_counts[slot.role]))
    return int(slot.minimum_count)


def _required_room_kit_exact_level_counts(
    scene: RoomScene,
    room_kit: RoomKitSelection,
    slot: Any,
    support_elevations: tuple[float, ...],
) -> dict[float, int]:
    """Distribute an aggregate role target without weakening per-story minima."""

    if not support_elevations:
        return {}
    level_requirements = _required_room_kit_level_coverage(
        str(getattr(scene, "text_description", "")),
        room_kit,
        support_elevations,
    )
    minimum_per_level = level_requirements.get(slot.role)
    if minimum_per_level is None:
        return {}

    aggregate_target = _required_room_kit_role_count(room_kit, slot)
    covered_target = minimum_per_level * len(support_elevations)
    surplus, remainder = divmod(
        max(0, aggregate_target - covered_target),
        len(support_elevations),
    )
    return {
        elevation: minimum_per_level + surplus + (index < remainder)
        for index, elevation in enumerate(support_elevations)
    }


def _normalize_dense_library_bookcases(
    scene: RoomScene,
    room_kit: RoomKitSelection,
    support_elevations: tuple[float, ...],
    *,
    remove_object: Callable[[str], Any],
) -> int:
    """Prune explicit rich-library shelving to its canonical story counts."""

    level_requirements = _required_room_kit_level_coverage(
        str(getattr(scene, "text_description", "")),
        room_kit,
        support_elevations,
    )
    bookshelf_slot = next(
        (slot for slot in room_kit.slots if slot.role == "bookshelf"), None
    )
    if "bookshelf" not in level_requirements or bookshelf_slot is None:
        return 0
    targets_by_level = _required_room_kit_exact_level_counts(
        scene,
        room_kit,
        bookshelf_slot,
        support_elevations,
    )

    try:
        window_blockers = {
            str(violation.furniture_id)
            for violation in compute_window_clearance_violations(scene)
        }
    except (AttributeError, TypeError, ValueError):
        window_blockers = set()
    try:
        path_blockers = {
            str(violation.furniture_id)
            for violation in compute_door_clearance_violations(scene)
        }
    except (AttributeError, TypeError, ValueError):
        path_blockers = set()

    by_level = {elevation: [] for elevation in support_elevations}
    for obj in scene.objects.values():
        if not _object_matches_room_kit_slot(obj, bookshelf_slot):
            continue
        metadata = getattr(obj, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            obj.metadata = metadata
        metadata.pop("dense_library_populated_case", None)
        level = _nearest_level(obj, support_elevations)
        if level is not None:
            by_level[level].append(obj)

    room_geometry = getattr(scene, "room_geometry", None)
    half_x = max(0.0, float(getattr(room_geometry, "length", 0.0)) / 2.0)
    half_y = max(0.0, float(getattr(room_geometry, "width", 0.0)) / 2.0)

    removed = 0
    for elevation, candidates in by_level.items():
        target_per_level = targets_by_level[elevation]
        if len(candidates) <= target_per_level:
            retained = list(candidates)
        else:
            grouped = sorted(
                (
                    obj
                    for obj in candidates
                    if (getattr(obj, "metadata", None) or {}).get(
                        "dense_library_grouped_run"
                    )
                    == elevation
                ),
                key=lambda obj: str(obj.object_id),
            )
            retained = grouped[:target_per_level]
            retained_ids = {str(obj.object_id) for obj in retained}
            grouped_points: list[tuple[float, float]] = []
            for obj in retained:
                try:
                    translation = obj.transform.translation()
                    grouped_points.append(
                        (float(translation[0]), float(translation[1]))
                    )
                except (AttributeError, IndexError, TypeError, ValueError):
                    continue

            def retention_priority(obj: Any) -> tuple[bool, bool, float, float, str]:
                object_id = str(obj.object_id)
                try:
                    translation = obj.transform.translation()
                    x = float(translation[0])
                    y = float(translation[1])
                    wall_gap = min(abs(half_x - abs(x)), abs(half_y - abs(y)))
                    run_gap = min(
                        (math.hypot(x - px, y - py) for px, py in grouped_points),
                        default=0.0,
                    )
                except (AttributeError, IndexError, TypeError, ValueError):
                    wall_gap = math.inf
                    run_gap = math.inf
                return (
                    object_id in window_blockers,
                    object_id in path_blockers,
                    wall_gap,
                    run_gap,
                    object_id,
                )

            extras = sorted(
                (obj for obj in candidates if str(obj.object_id) not in retained_ids),
                key=retention_priority,
            )
            retained.extend(extras[: max(0, target_per_level - len(retained))])
            retained_ids = {str(obj.object_id) for obj in retained}
            for obj in candidates:
                if str(obj.object_id) in retained_ids:
                    continue
                remove_object(str(obj.object_id))
                removed += 1

        for obj in retained:
            metadata = getattr(obj, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                obj.metadata = metadata
            metadata["dense_library_populated_case"] = elevation

    # The room-kit brief defines exact expanded counts. Keep the model free to
    # compose a good first pass, then deterministically remove role surplus so
    # successful dense scenes retain their bounded object/export contract.
    for slot in room_kit.slots:
        if slot.role == "bookshelf" or slot.placement_class == "surface":
            continue
        candidates = [
            obj
            for obj in scene.objects.values()
            if getattr(obj, "object_type", None) == ObjectType.FURNITURE
            and _object_matches_room_kit_slot(obj, slot)
        ]
        aggregate_target = _required_room_kit_role_count(room_kit, slot)
        targets_by_level = _required_room_kit_exact_level_counts(
            scene,
            room_kit,
            slot,
            support_elevations,
        )

        def partner_distance(obj: Any) -> float:
            if slot.role != "reading_chair":
                return 0.0
            table_slot = next(
                (
                    candidate_slot
                    for candidate_slot in room_kit.slots
                    if candidate_slot.role == "reading_table"
                ),
                None,
            )
            if table_slot is None:
                return math.inf
            tables = [
                candidate
                for candidate in scene.objects.values()
                if getattr(candidate, "object_type", None) == ObjectType.FURNITURE
                and _object_matches_room_kit_slot(candidate, table_slot)
            ]
            try:
                position = obj.transform.translation()
                return min(
                    math.hypot(
                        float(position[0]) - float(table.transform.translation()[0]),
                        float(position[1]) - float(table.transform.translation()[1]),
                    )
                    for table in tables
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                return math.inf

        def role_priority(obj: Any) -> tuple[bool, bool, float, str]:
            object_id = str(obj.object_id)
            return (
                object_id in window_blockers,
                object_id in path_blockers,
                partner_distance(obj),
                object_id,
            )

        retained_ids: set[str] = set()
        if targets_by_level:
            for elevation in support_elevations:
                level_candidates = sorted(
                    (
                        obj
                        for obj in candidates
                        if _nearest_level(obj, support_elevations) == elevation
                    ),
                    key=role_priority,
                )
                retained_ids.update(
                    str(obj.object_id)
                    for obj in level_candidates[: targets_by_level[elevation]]
                )
        else:
            retained_ids.update(
                str(obj.object_id)
                for obj in sorted(candidates, key=role_priority)[:aggregate_target]
            )
        for obj in candidates:
            if str(obj.object_id) in retained_ids:
                continue
            remove_object(str(obj.object_id))
            removed += 1

    return removed


def _chair_cluster_poses(
    table: Any, chair_asset: Any
) -> list[tuple[float, float, float]]:
    """Return bounded deterministic chair poses around one table anchor."""

    translation = table.transform.translation()

    def footprint_size(obj: Any, fallback: tuple[float, float]) -> list[float]:
        try:
            return [
                float(obj.bbox_max[index]) - float(obj.bbox_min[index])
                for index in (0, 1)
            ]
        except (AttributeError, IndexError, TypeError, ValueError):
            return list(fallback)

    table_size = footprint_size(table, (1.5, 0.8))
    chair_size = footprint_size(chair_asset, (0.6, 0.6))
    radius = min(1.9, max(0.9, max(table_size) / 2 + max(chair_size) / 2 + 0.25))
    poses: list[tuple[float, float, float]] = []
    for multiplier in (1.0, 1.15, 1.5, 2.0):
        candidate_radius = min(2.2, radius * multiplier)
        for angle_degrees in (-90.0, 0.0, 90.0, 180.0, -45.0, 45.0, 135.0, 225.0):
            angle = math.radians(angle_degrees)
            x = float(translation[0]) + math.cos(angle) * candidate_radius
            y = float(translation[1]) + math.sin(angle) * candidate_radius
            dx = float(translation[0]) - x
            dy = float(translation[1]) - y
            yaw = math.degrees(math.atan2(-dx, dy))
            poses.append((x, y, yaw))
    return poses
