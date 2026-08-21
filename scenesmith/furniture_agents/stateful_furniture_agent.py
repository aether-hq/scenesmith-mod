"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool
from agents.exceptions import ModelBehaviorError
from omegaconf import DictConfig

from scenesmith.agent_utils.asset_semantics import (
    catalog_candidate_is_compatible,
    catalog_candidate_satisfies_request_details,
    tall_furniture_dimensions_are_compatible,
)
from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.clearance_zones import (
    compute_door_clearance_violations,
    compute_window_clearance_violations,
)
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.reachability import (
    compute_reachability,
    format_reachability_for_critic,
)
from scenesmith.agent_utils.room import AgentType, ObjectType, RoomScene
from scenesmith.agent_utils.room_kits import (
    RoomKitSelection,
    persist_room_kit,
    select_room_kit,
)
from scenesmith.agent_utils.scoring import FurnitureCritiqueWithScores
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
from scenesmith.furniture_agents.tools.scene_tools import SceneTools
from scenesmith.furniture_agents.tools.vision_tools import VisionTools
from scenesmith.prompts.registry import FurnitureAgentPrompts
from scenesmith.utils.logging import BaseLogger

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


def _normalize_dense_library_bookcases(
    scene: RoomScene,
    room_kit: RoomKitSelection,
    support_elevations: tuple[float, ...],
    *,
    remove_object: Callable[[str], Any],
) -> int:
    """Prune explicit rich-library shelving to its authored per-story count."""

    level_requirements = _required_room_kit_level_coverage(
        str(getattr(scene, "text_description", "")),
        room_kit,
        support_elevations,
    )
    target_per_level = level_requirements.get("bookshelf")
    bookshelf_slot = next(
        (slot for slot in room_kit.slots if slot.role == "bookshelf"), None
    )
    if target_per_level is None or bookshelf_slot is None:
        return 0

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


def _validate_room_kit_completion(
    scene: RoomScene,
    room_kit: RoomKitSelection | None,
    *,
    support_elevations: tuple[float, ...] = (),
    enforce_exact_level_counts: bool = False,
) -> int:
    """Reject a matched semantic room kit that did not place required furniture."""

    furniture_count = sum(
        obj.object_type == ObjectType.FURNITURE for obj in scene.objects.values()
    )
    if room_kit is None:
        return furniture_count

    required_minimum = sum(
        _required_room_kit_role_count(room_kit, slot)
        for slot in room_kit.slots
        if slot.required
    )
    if furniture_count < required_minimum:
        raise ModelBehaviorError(
            f"Semantic room kit {room_kit.kit_id} placed {furniture_count} "
            f"furniture objects; required minimum is {required_minimum}. "
            "The furniture stage cannot publish this checkpoint."
        )

    furniture = [
        obj for obj in scene.objects.values() if obj.object_type == ObjectType.FURNITURE
    ]
    role_counts = {
        slot.role: sum(_object_matches_room_kit_slot(obj, slot) for obj in furniture)
        for slot in room_kit.slots
        if slot.required
    }
    deficits = [
        (
            slot.role,
            role_counts[slot.role],
            _required_room_kit_role_count(room_kit, slot),
        )
        for slot in room_kit.slots
        if slot.required
        and role_counts[slot.role] < _required_room_kit_role_count(room_kit, slot)
    ]
    if deficits:
        details = "; ".join(
            f"{role} placed {placed}, required {required}"
            for role, placed, required in deficits
        )
        raise ModelBehaviorError(
            f"Semantic room kit {room_kit.kit_id} has required role deficits: "
            f"{details}. The furniture stage cannot publish this checkpoint."
        )

    level_requirements = _required_room_kit_level_coverage(
        str(getattr(scene, "text_description", "")),
        room_kit,
        support_elevations,
    )
    level_deficits: list[tuple[str, float, int, int]] = []
    for slot in room_kit.slots:
        required_per_level = level_requirements.get(slot.role)
        if required_per_level is None:
            continue
        counts = _room_kit_role_level_counts(
            furniture,
            slot,
            support_elevations,
        )
        level_deficits.extend(
            (slot.role, elevation, counts[elevation], required_per_level)
            for elevation in support_elevations
            if counts[elevation] < required_per_level
        )
    if level_deficits:
        details = "; ".join(
            f"{role} at {elevation:.3f}m placed {placed}, required {required}"
            for role, elevation, placed, required in level_deficits
        )
        raise ModelBehaviorError(
            f"Semantic room kit {room_kit.kit_id} has required level coverage "
            f"deficits: {details}. The furniture stage cannot publish this checkpoint."
        )
    if enforce_exact_level_counts and "bookshelf" in level_requirements:
        bookshelf_slot = next(
            (slot for slot in room_kit.slots if slot.role == "bookshelf"), None
        )
        if bookshelf_slot is not None:
            counts = _room_kit_role_level_counts(
                furniture,
                bookshelf_slot,
                support_elevations,
            )
            target = level_requirements["bookshelf"]
            mismatches = [
                (elevation, counts[elevation])
                for elevation in support_elevations
                if counts[elevation] != target
            ]
            if mismatches:
                details = "; ".join(
                    f"bookshelf at {elevation:.3f}m placed {placed}, required "
                    f"exactly {target}"
                    for elevation, placed in mismatches
                )
                raise ModelBehaviorError(
                    f"Semantic room kit {room_kit.kit_id} has noncanonical "
                    f"bookshelf density: {details}. The furniture stage cannot "
                    "publish this checkpoint."
                )
    bookshelf_slot = next(
        (slot for slot in room_kit.slots if slot.role == "bookshelf"), None
    )
    if bookshelf_slot is not None and "bookshelf" in level_requirements:
        wall_run_counts = _bookcase_wall_run_level_counts(
            furniture,
            bookshelf_slot,
            support_elevations,
        )
        wall_run_deficits = [
            (elevation, wall_run_counts[elevation])
            for elevation in support_elevations
            if wall_run_counts[elevation] < 3
        ]
        if wall_run_deficits:
            details = "; ".join(
                f"bookshelf wall run at {elevation:.3f}m has {placed}, required 3"
                for elevation, placed in wall_run_deficits
            )
            raise ModelBehaviorError(
                f"Semantic room kit {room_kit.kit_id} has sparse bookcase "
                f"grouping: {details}. The furniture stage cannot publish this "
                "checkpoint."
            )
    table_slot = next(
        (slot for slot in room_kit.slots if slot.role == "reading_table"), None
    )
    chair_slot = next(
        (slot for slot in room_kit.slots if slot.role == "reading_chair"), None
    )
    required_chairs = level_requirements.get("reading_chair")
    if (
        table_slot is not None
        and chair_slot is not None
        and required_chairs is not None
    ):
        ensemble_counts = _patron_ensemble_level_counts(
            furniture,
            table_slot,
            chair_slot,
            support_elevations,
        )
        ensemble_deficits = [
            (elevation, ensemble_counts[elevation])
            for elevation in support_elevations
            if ensemble_counts[elevation] < required_chairs
        ]
        if ensemble_deficits:
            details = "; ".join(
                f"patron ensemble at {elevation:.3f}m has {placed} stable "
                f"inward-facing chairs, required {required_chairs}"
                for elevation, placed in ensemble_deficits
            )
            raise ModelBehaviorError(
                f"Semantic room kit {room_kit.kit_id} has incoherent patron "
                f"ensembles: {details}. The furniture stage cannot publish this "
                "checkpoint."
            )
    console_logger.info(
        "Semantic room kit %s completion gate passed: %d furniture objects "
        "(minimum %d; roles %s)",
        room_kit.kit_id,
        furniture_count,
        required_minimum,
        role_counts,
    )
    return furniture_count


def _validate_furniture_collision_free(scene: RoomScene, physics_cfg: Any) -> None:
    """Reject hard collisions left by the complete model-authored furniture batch."""
    furniture_ids = {
        str(getattr(obj, "object_id", object_id))
        for object_id, obj in scene.objects.items()
        if obj.object_type == ObjectType.FURNITURE
    }
    collisions = compute_scene_collisions(
        scene=scene,
        penetration_threshold=physics_cfg.object_penetration_threshold_m,
        floor_penetration_tolerance=physics_cfg.floor_penetration_tolerance_m,
        manipuland_furniture_tolerance_m=(physics_cfg.manipuland_furniture_tolerance_m),
    )
    furniture_collisions = [
        collision
        for collision in collisions
        if collision.object_a_id in furniture_ids
        or collision.object_b_id in furniture_ids
    ]
    if furniture_collisions:
        details = "; ".join(
            collision.to_description() for collision in furniture_collisions
        )
        raise ModelBehaviorError(
            "Furniture workflow left hard collisions after its final tool batch: "
            f"{details}. The furniture stage cannot run recovery or publish this "
            "checkpoint."
        )
    console_logger.info("Furniture workflow batch collision gate passed")


class StatefulFurnitureAgent(BaseStatefulAgent, BaseFurnitureAgent):
    """Natural conversation between persistent agents with proper image injection."""

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.FURNITURE

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        num_workers: int = 1,
        render_allocation: RenderAllocation | None = None,
    ):
        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
        )
        # Initialize furniture-specific base class.
        BaseFurnitureAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
            articulated_server_host=articulated_server_host,
            articulated_server_port=articulated_server_port,
            materials_server_host=materials_server_host,
            materials_server_port=materials_server_port,
            num_workers=num_workers,
            render_allocation=render_allocation,
        )

        # Create persistent agent sessions using base class method.
        self.designer_session, self.critic_session = self._create_sessions()

        # Context image for designer initialization (furniture-specific).
        self.context_image_path: Path | None = None
        self.room_kit_brief = (
            "No semantic room kit matched; use the scene requirements."
        )

    def _create_designer_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create designer agent with tools.

        Args:
            tools: Tools to provide to the designer

        Returns:
            Configured designer agent
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = FurnitureAgentPrompts[designer_config.prompt]
        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            has_reference_image=self.context_image_path is not None,
        )

    def _create_critic_tools(self) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Returns:
            List of tools for the critic (read-only scene validation tools)
        """
        vision_tools = VisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )
        scene_tools = SceneTools(scene=self.scene, cfg=self.cfg)

        # Return vision tools + read-only scene tools.
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            scene_tools.tools["get_current_scene_state"],
            scene_tools.tools["check_facing_tool"],
        ]

    def _create_critic_agent(
        self, scene: RoomScene, tools: list[FunctionTool]
    ) -> Agent:
        """Create critic agent with scene context.

        Args:
            scene: RoomScene to provide context for the critic
            tools: Tools to provide to the critic

        Returns:
            Configured critic agent with structured output
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = FurnitureAgentPrompts[critic_config.prompt]
        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=FurnitureCritiqueWithScores,
            scene_description=scene.text_description,
        )

    def _create_planner_agent(
        self, scene: RoomScene, tools: list[FunctionTool]
    ) -> Agent:
        """Create planner agent with scene-specific context.

        Args:
            scene: RoomScene to provide context for the planner
            tools: Tools to provide to the planner

        Returns:
            Configured planner agent
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = FurnitureAgentPrompts[planner_config.prompt]
        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            scene_prompt=scene.text_description,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=self.cfg.reset_single_category_threshold,
            reset_total_sum_threshold=self.cfg.reset_total_sum_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _create_designer_tools(self) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Returns:
            List of tools for the designer agent.
        """
        vision_tools = VisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )
        self.furniture_tools = FurnitureTools(
            scene=self.scene, asset_manager=self.asset_manager, cfg=self.cfg
        )
        scene_tools = SceneTools(scene=self.scene, cfg=self.cfg)
        workflow_tools = WorkflowTools()

        return [
            *vision_tools.tools.values(),
            *self.furniture_tools.tools.values(),
            *scene_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]

    def _render_empty_room(self) -> Path:
        """Render top-down view of empty room showing doors/windows.

        Uses furniture_selection mode which disables coordinate grid/frame.
        Pass annotate_object_types=[] to disable all labels and bounding boxes.
        Result: clean room geometry with doors/windows visible but unlabeled.

        Returns:
            Path to directory containing rendered image.
        """
        return self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=[],  # Empty room only
            render_name="empty_room_context",
            rendering_mode="furniture_selection",  # Disables grid/frame
            annotate_object_types=[],  # Disables all labels/bboxes
        )

    def _generate_and_save_context_image(self, scene: RoomScene) -> Path:
        """Generate and save context image for design guidance.

        Renders an empty room showing doors/windows, then uses image editing
        to add suggested furniture placement.

        Args:
            scene: RoomScene to generate context image for.

        Returns:
            Path to saved context image.
        """
        console_logger.info("Generating context image for scene...")

        # Render empty room showing doors/windows.
        room_render_dir = self._render_empty_room()
        # Get the top-down image from the render directory.
        room_render = room_render_dir / "0_top.png"

        # Generate context image using the render as reference.
        # Save alongside the input render for easy association.
        output_path = room_render_dir / "context_edited.png"
        image_path = (
            self.asset_manager.image_generator.generate_furniture_context_image(
                reference_image_path=room_render,
                scene_description=scene.text_description,
                width_m=scene.room_geometry.width,
                length_m=scene.room_geometry.length,
                output_path=output_path,
            )
        )

        console_logger.info(f"Context image saved to: {image_path}")
        return image_path

    @staticmethod
    def _semantic_tokens(value: str) -> set[str]:
        """Normalize a short role or asset label for deterministic matching."""

        stop_words = {"a", "an", "and", "for", "of", "the", "with"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
            if token not in stop_words
        }

    @classmethod
    def _slot_relevance(cls, asset: Any, slot: Any) -> tuple[int, int, float, str]:
        """Rank one cached asset against a semantic room-kit slot."""

        role_names = (slot.role, *getattr(slot, "aliases", ()))
        normalized_roles = {
            " ".join(sorted(cls._semantic_tokens(role))) for role in role_names
        }
        asset_name = " ".join(sorted(cls._semantic_tokens(str(asset.name))))
        asset_tokens = cls._semantic_tokens(
            f"{asset.name} {getattr(asset, 'description', '')}"
        )
        role_tokens = set().union(*(cls._semantic_tokens(role) for role in role_names))
        exact = int(asset_name in normalized_roles)
        overlap = len(asset_tokens & role_tokens)
        metadata = getattr(asset, "metadata", None) or {}
        quality = float(metadata.get("asset_quality_score", 0.0))
        catalog_text = str(
            metadata.get("catalog_semantics")
            or metadata.get("ontology_path")
            or f"{asset.name} {getattr(asset, 'description', '')}"
        )
        detail_text = f"{asset.name} {getattr(asset, 'description', '')} {catalog_text}"
        compatible, _ = catalog_candidate_is_compatible(
            request_text=str(getattr(slot, "query", slot.role)),
            candidate_text=catalog_text,
            quality_score=quality,
        )
        if not compatible:
            return (-1, 0, quality, str(asset.object_id))
        compatible, _ = catalog_candidate_satisfies_request_details(
            request_text=str(getattr(slot, "query", slot.role)),
            candidate_text=catalog_text,
            supports_detail_fill=bool(metadata.get("support_zones")),
        )
        if not compatible:
            return (-1, 0, quality, str(asset.object_id))
        compatible_dimensions, _ = tall_furniture_dimensions_are_compatible(
            request_text=str(getattr(slot, "query", slot.role)),
            desired_dimensions=getattr(slot, "nominal_dimensions_m", None),
            bbox_min=getattr(asset, "bbox_min", None),
            bbox_max=getattr(asset, "bbox_max", None),
        )
        if not compatible_dimensions:
            return (-1, 0, quality, str(asset.object_id))
        query_tokens = cls._semantic_tokens(str(getattr(slot, "query", slot.role)))
        candidate_tokens = asset_tokens | cls._semantic_tokens(detail_text)
        query_overlap = len(query_tokens & candidate_tokens)
        return (
            query_overlap * 1000 + exact * 100 + overlap,
            query_overlap,
            quality,
            str(asset.object_id),
        )

    def _deterministic_room_positions(
        self, *, wall: bool
    ) -> list[tuple[float, float, float]]:
        """Return conservative unique SE(2) poses inside the room envelope."""

        half_x = max(0.5, float(self.scene.room_geometry.length) / 2.0 - 0.65)
        half_y = max(0.5, float(self.scene.room_geometry.width) / 2.0 - 0.65)
        if wall:
            return [
                (-0.55 * half_x, 0.88 * half_y, 180.0),
                (0.55 * half_x, 0.88 * half_y, 180.0),
                (-0.55 * half_x, -0.88 * half_y, 0.0),
                (0.55 * half_x, -0.88 * half_y, 0.0),
                (-0.88 * half_x, 0.45 * half_y, -90.0),
                (-0.88 * half_x, -0.45 * half_y, -90.0),
                (0.88 * half_x, 0.45 * half_y, 90.0),
                (0.88 * half_x, -0.45 * half_y, 90.0),
            ]
        return [
            (0.0, -0.18 * half_y, 0.0),
            (-0.42 * half_x, -0.18 * half_y, -90.0),
            (0.42 * half_x, -0.18 * half_y, 90.0),
            (-0.42 * half_x, 0.38 * half_y, -135.0),
            (0.42 * half_x, 0.38 * half_y, 135.0),
            (0.0, 0.58 * half_y, 180.0),
            (-0.68 * half_x, -0.58 * half_y, -45.0),
            (0.68 * half_x, -0.58 * half_y, 45.0),
            (-0.68 * half_x, 0.68 * half_y, -135.0),
            (0.68 * half_x, 0.68 * half_y, 135.0),
            (0.0, -0.72 * half_y, 0.0),
            (-0.72 * half_x, 0.0, -90.0),
            (0.72 * half_x, 0.0, 90.0),
        ]

    def _bookcase_wall_run_candidates(
        self, asset: Any
    ) -> list[list[tuple[float, float, float]]]:
        """Return bounded three-case runs along each wall, away from corners."""

        try:
            width = max(
                abs(float(asset.bbox_max[0]) - float(asset.bbox_min[0])),
                abs(float(asset.bbox_max[1]) - float(asset.bbox_min[1])),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            width = 1.0
        spacing = min(1.45, max(0.8, width + 0.12))
        half_x = max(0.5, float(self.scene.room_geometry.length) / 2.0 - 0.65)
        half_y = max(0.5, float(self.scene.room_geometry.width) / 2.0 - 0.65)
        wall_x = 0.88 * half_x
        wall_y = 0.88 * half_y
        centers = (-2.0 * spacing, 0.0, 2.0 * spacing)
        runs: list[list[tuple[float, float, float]]] = []
        for center in centers:
            offsets = (center - spacing, center, center + spacing)
            if max(abs(offset) for offset in offsets) <= half_x - 0.4:
                runs.append([(offset, wall_y, 180.0) for offset in offsets])
                runs.append([(offset, -wall_y, 0.0) for offset in offsets])
            if max(abs(offset) for offset in offsets) <= half_y - 0.4:
                runs.append([(-wall_x, offset, -90.0) for offset in offsets])
                runs.append([(wall_x, offset, 90.0) for offset in offsets])
        return runs

    def _place_bookcase_wall_run_deterministically(
        self,
        asset: Any,
        slot: Any,
        elevation: float,
        support_elevations: tuple[float, ...],
    ) -> int:
        """Place one complete collision-validated wall run or leave no partial run."""

        if (
            _bookcase_wall_run_level_counts(
                self.scene.objects.values(), slot, support_elevations
            )[elevation]
            >= 3
        ):
            return 0
        for candidate_run in self._bookcase_wall_run_candidates(asset):
            added_ids: list[str] = []
            complete = True
            for x, y, yaw in candidate_run:
                before_ids = set(self.scene.objects)
                raw_result = self.furniture_tools._add_furniture_to_scene_impl(
                    asset_id=str(asset.object_id),
                    x=x,
                    y=y,
                    z=elevation,
                    roll=0.0,
                    pitch=0.0,
                    yaw=yaw,
                )
                try:
                    result_payload = json.loads(raw_result)
                    success = bool(result_payload.get("success"))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    result_payload = {}
                    success = False
                object_id = str(result_payload.get("object_id") or "")
                if success and not object_id:
                    new_ids = sorted(set(self.scene.objects) - before_ids)
                    object_id = new_ids[0] if len(new_ids) == 1 else ""
                placed_object = self.scene.objects.get(object_id)
                actual_level = (
                    _nearest_level(placed_object, support_elevations)
                    if placed_object is not None
                    else None
                )
                if not success or not object_id or actual_level != elevation:
                    if object_id and object_id in self.scene.objects:
                        self.furniture_tools._remove_furniture_impl(object_id)
                    complete = False
                    break
                added_ids.append(object_id)
            if complete and len(added_ids) == 3:
                run_size = _bookcase_wall_run_level_counts(
                    self.scene.objects.values(), slot, support_elevations
                )[elevation]
                if run_size >= 3:
                    console_logger.info(
                        "Deterministic recovery placed a contiguous 3-case "
                        "bookshelf wall run at %.3fm",
                        elevation,
                    )
                    return 3
            for object_id in reversed(added_ids):
                self.furniture_tools._remove_furniture_impl(object_id)
        console_logger.warning(
            "Deterministic recovery could not place a complete bookshelf wall "
            "run at %.3fm without violating placement constraints",
            elevation,
        )
        return 0

    def _place_room_kit_minimums_deterministically(
        self, room_kit: RoomKitSelection
    ) -> int:
        """Recover required kit roles from acquired assets without another model call.

        Every attempt goes through ``FurnitureTools`` so structural support,
        enclosure, contextual, and collision validation remain authoritative.
        """

        assets = [
            asset
            for asset in self.asset_manager.list_available_assets()
            if asset.object_type == ObjectType.FURNITURE
        ]

        self.furniture_tools.set_noise_profile(PlacementNoiseMode.PERFECT)
        support_elevations = self.furniture_tools._major_support_elevations()
        level_requirements = _required_room_kit_level_coverage(
            str(getattr(self.scene, "text_description", "")),
            room_kit,
            support_elevations,
        )
        table_slot = next(
            (slot for slot in room_kit.slots if slot.role == "reading_table"), None
        )
        chair_slot = next(
            (slot for slot in room_kit.slots if slot.role == "reading_chair"), None
        )
        attempted_positions: set[tuple[float, float, float]] = set()
        level_counts = {elevation: 0 for elevation in support_elevations}
        for scene_object in self.scene.objects.values():
            if scene_object.object_type != ObjectType.FURNITURE:
                continue
            try:
                object_elevation = float(scene_object.transform.translation()[2])
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            nearest = min(
                support_elevations,
                key=lambda elevation: abs(elevation - object_elevation),
            )
            level_counts[nearest] += 1
        placed = 0

        for slot in room_kit.slots:
            if not slot.required:
                continue
            existing = sum(
                obj.object_type == ObjectType.FURNITURE
                and _object_matches_room_kit_slot(obj, slot)
                for obj in self.scene.objects.values()
            )
            aggregate_missing = max(
                0,
                _required_room_kit_role_count(room_kit, slot) - existing,
            )
            level_targets: list[float] = []
            wall_run_targets: list[float] = []
            required_per_level = level_requirements.get(slot.role)
            if required_per_level is not None:
                if (
                    slot.role == "reading_chair"
                    and table_slot is not None
                    and chair_slot is not None
                ):
                    role_level_counts = _patron_ensemble_level_counts(
                        self.scene.objects.values(),
                        table_slot,
                        chair_slot,
                        support_elevations,
                    )
                else:
                    role_level_counts = _room_kit_role_level_counts(
                        self.scene.objects.values(),
                        slot,
                        support_elevations,
                    )
                for elevation in support_elevations:
                    level_targets.extend(
                        [elevation]
                        * max(0, required_per_level - role_level_counts[elevation])
                    )
                if slot.role == "bookshelf":
                    wall_run_counts = _bookcase_wall_run_level_counts(
                        self.scene.objects.values(), slot, support_elevations
                    )
                    wall_run_targets = [
                        elevation
                        for elevation in support_elevations
                        if wall_run_counts[elevation] < 3
                    ]
            missing = max(
                aggregate_missing,
                len(level_targets),
                3 * len(wall_run_targets),
            )
            if missing == 0:
                continue

            ranked = sorted(
                assets,
                key=lambda asset: self._slot_relevance(asset, slot),
                reverse=True,
            )
            if not ranked or self._slot_relevance(ranked[0], slot)[0] <= 0:
                try:
                    result = self.asset_manager.generate_assets(
                        AssetGenerationRequest(
                            object_descriptions=[
                                str(getattr(slot, "query", slot.role))
                            ],
                            short_names=[slot.role],
                            object_type=ObjectType.FURNITURE,
                            desired_dimensions=[
                                list(
                                    getattr(
                                        slot,
                                        "nominal_dimensions_m",
                                        (1.0, 1.0, 1.0),
                                    )
                                )
                            ],
                            style_context=str(
                                getattr(self.scene, "text_description", "")
                            ),
                            scene_id=getattr(
                                getattr(self.scene, "scene_dir", None),
                                "name",
                                None,
                            ),
                        )
                    )
                    console_logger.info(
                        "Deterministic recovery acquired %d asset(s) for missing "
                        "room-kit role %s",
                        len(result.successful_assets),
                        slot.role,
                    )
                except Exception as exc:
                    console_logger.warning(
                        "Deterministic recovery could not acquire missing room-kit "
                        "role %s: %s",
                        slot.role,
                        exc,
                    )
                assets = [
                    asset
                    for asset in self.asset_manager.list_available_assets()
                    if asset.object_type == ObjectType.FURNITURE
                ]
                ranked = sorted(
                    assets,
                    key=lambda asset: self._slot_relevance(asset, slot),
                    reverse=True,
                )
            if not ranked or self._slot_relevance(ranked[0], slot)[0] <= 0:
                console_logger.warning(
                    "No cached furniture asset matched required room-kit role %s",
                    slot.role,
                )
                continue
            asset = ranked[0]
            if slot.role == "bookshelf" and wall_run_targets:
                for elevation in wall_run_targets:
                    placed += self._place_bookcase_wall_run_deterministically(
                        asset,
                        slot,
                        elevation,
                        support_elevations,
                    )
                existing = sum(
                    obj.object_type == ObjectType.FURNITURE
                    and _object_matches_room_kit_slot(obj, slot)
                    for obj in self.scene.objects.values()
                )
                aggregate_missing = max(
                    0,
                    _required_room_kit_role_count(room_kit, slot) - existing,
                )
                role_level_counts = _room_kit_role_level_counts(
                    self.scene.objects.values(),
                    slot,
                    support_elevations,
                )
                level_targets = []
                if required_per_level is not None:
                    for elevation in support_elevations:
                        level_targets.extend(
                            [elevation]
                            * max(
                                0,
                                required_per_level - role_level_counts[elevation],
                            )
                        )
                missing = max(aggregate_missing, len(level_targets))
                if missing == 0:
                    continue
            positions = self._deterministic_room_positions(
                wall=getattr(slot, "placement_class", "floor") == "wall"
            )
            if slot.role == "reading_table" and level_targets:
                role_anchors: list[tuple[float, float, float]] = []
                for scene_object in self.scene.objects.values():
                    if not _object_matches_room_kit_slot(scene_object, slot):
                        continue
                    try:
                        translation = scene_object.transform.translation()
                        yaw = math.degrees(
                            scene_object.transform.rotation()
                            .ToRollPitchYaw()
                            .yaw_angle()
                        )
                    except (AttributeError, IndexError, TypeError, ValueError):
                        try:
                            translation = scene_object.transform.translation()
                        except (AttributeError, IndexError, TypeError, ValueError):
                            continue
                        yaw = 0.0
                    anchor = (float(translation[0]), float(translation[1]), yaw)
                    if anchor not in role_anchors:
                        role_anchors.append(anchor)
                positions = [*role_anchors, *positions]

            cluster_ids: dict[float, list[str]] = {
                elevation: [] for elevation in set(level_targets)
            }
            for recovery_index in range(missing):
                success = False
                target_elevation = (
                    level_targets[recovery_index]
                    if recovery_index < len(level_targets)
                    else None
                )
                candidate_elevations = (
                    (target_elevation,)
                    if target_elevation is not None
                    else tuple(
                        sorted(
                            support_elevations,
                            key=lambda value: (level_counts[value], value),
                        )
                    )
                )
                for elevation in candidate_elevations:
                    candidate_positions = positions
                    if slot.role == "reading_chair" and table_slot is not None:
                        tables = sorted(
                            (
                                obj
                                for obj in self.scene.objects.values()
                                if _object_matches_room_kit_slot(obj, table_slot)
                                and _nearest_level(obj, support_elevations) == elevation
                            ),
                            key=lambda obj: str(getattr(obj, "object_id", "")),
                        )
                        if tables:
                            chairs = [
                                obj
                                for obj in self.scene.objects.values()
                                if _object_matches_room_kit_slot(obj, slot)
                                and _nearest_level(obj, support_elevations) == elevation
                            ]
                            anchor_table = min(
                                tables,
                                key=lambda table: (
                                    -sum(
                                        _stable_chair_faces_table(chair, table)
                                        for chair in chairs
                                    ),
                                    str(getattr(table, "object_id", "")),
                                ),
                            )
                            candidate_positions = _chair_cluster_poses(
                                anchor_table, asset
                            )
                    for x, y, yaw in candidate_positions:
                        position_key = (
                            round(x, 4),
                            round(y, 4),
                            round(elevation, 4),
                        )
                        if position_key in attempted_positions:
                            continue
                        attempted_positions.add(position_key)
                        raw_result = self.furniture_tools._add_furniture_to_scene_impl(
                            asset_id=str(asset.object_id),
                            x=x,
                            y=y,
                            z=elevation,
                            roll=0.0,
                            pitch=0.0,
                            yaw=yaw,
                        )
                        try:
                            result_payload = json.loads(raw_result)
                            success = bool(result_payload.get("success"))
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            result_payload = {}
                            success = False
                        object_id = str(result_payload.get("object_id") or "")
                        placed_object = self.scene.objects.get(object_id)
                        actual_level = (
                            _nearest_level(placed_object, support_elevations)
                            if placed_object is not None
                            else None
                        )
                        if (
                            success
                            and target_elevation is not None
                            and actual_level is not None
                            and actual_level != elevation
                        ):
                            self.furniture_tools._remove_furniture_impl(object_id)
                            console_logger.warning(
                                "Deterministic recovery rejected %s at %.3fm: "
                                "support resolution placed it at %.3fm",
                                slot.role,
                                elevation,
                                actual_level,
                            )
                            success = False
                        if success:
                            level_counts[elevation] += 1
                            if (
                                slot.role == "reading_chair"
                                and elevation in cluster_ids
                            ):
                                if object_id:
                                    cluster_ids[elevation].append(object_id)
                            break
                    if success:
                        break
                if success:
                    placed += 1
                if not success:
                    console_logger.warning(
                        "Deterministic recovery exhausted valid poses for room-kit "
                        "role %s after placing %d of %d missing instances",
                        slot.role,
                        placed,
                        missing,
                    )
                    break

            if slot.role == "reading_chair" and cluster_ids:
                required_by_level = Counter(level_targets)
                for elevation, object_ids in cluster_ids.items():
                    if len(object_ids) >= required_by_level[elevation]:
                        continue
                    for object_id in reversed(object_ids):
                        self.furniture_tools._remove_furniture_impl(object_id)
                    placed -= len(object_ids)
                    level_counts[elevation] -= len(object_ids)
                    console_logger.warning(
                        "Rolled back incomplete patron chair cluster at %.3fm: "
                        "placed %d of %d required chairs",
                        elevation,
                        len(object_ids),
                        required_by_level[elevation],
                    )

        return placed

    async def add_furniture(self, scene: RoomScene) -> None:
        """Add furniture to a scene.

        Args:
            scene: RoomScene to add furniture to (mutated in place)
        """
        self._reset_workflow_budget()

        # Store everything as instance variables for closure access.
        self.scene = scene

        room_area_m2 = float(scene.room_geometry.width) * float(
            scene.room_geometry.length
        )
        room_kit = select_room_kit(scene.text_description, room_area_m2=room_area_m2)
        if room_kit is not None:
            self.room_kit_brief = room_kit.to_prompt_brief()
            persist_room_kit(room_kit, scene.scene_dir / "room_kit.json")
            console_logger.info(
                "Selected semantic room kit %s with counts %s",
                room_kit.kit_id,
                room_kit.slot_counts,
            )
        else:
            self.room_kit_brief = (
                "No semantic room kit matched; infer a compact functional grouping "
                "from the scene requirements."
            )

        # Generate context image if configured. If generation fails, continue without it.
        if self.cfg.context_image_generation.enabled:
            try:
                self.context_image_path = self._generate_and_save_context_image(scene)
            except Exception as e:
                console_logger.warning(
                    f"Context image generation failed, continuing without it: {e}"
                )
                self.context_image_path = None

        # Create designer, critic, and planner with tools once for this scene.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)
        critic_tools = self._create_critic_tools()
        self.critic = self._create_critic_agent(scene=scene, tools=critic_tools)
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent(scene=scene, tools=planner_tools)

        # Get runner instruction from prompt registry.
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_PLANNER_RUNNER_INSTRUCTION,
        )

        # Run the furniture placement workflow.
        result = await self._run_planner_with_partial_recovery(
            runner_instruction=runner_instruction,
            agent_name="PLANNER (FURNITURE)",
            state_hash=self.scene.content_hash,
        )

        _validate_furniture_collision_free(
            self.scene,
            self.cfg.physics_validation,
        )

        if room_kit is not None:
            recovered = self._place_room_kit_minimums_deterministically(room_kit)
            if recovered:
                console_logger.warning(
                    "Deterministic room-kit recovery placed %d required furniture "
                    "objects from cached assets",
                    recovered,
                )

        # Compute final critique and scores for completed scene.
        # Check if scene changed since last checkpoint to avoid redundant critique.
        current_scene_hash = self.scene.content_hash()

        if (
            self.cfg.max_critique_rounds <= 0
            or self._workflow_limit_reached
            or self._critique_calls >= int(self.cfg.max_critique_rounds)
        ):
            console_logger.info("Final critique skipped: critique budget unavailable")
            self.final_render_dir = self.rendering_manager.last_render_dir
        elif (
            self.checkpoint_scene_hash is not None
            and current_scene_hash == self.checkpoint_scene_hash
        ):
            console_logger.info(
                "Scene unchanged since last critique, skipping final critique"
            )
        else:
            console_logger.info(
                "Scene changed since last critique, computing final critique"
            )
            # Pass update_checkpoint=False to preserve N-1 checkpoint for reset check.
            await self._request_critique_bounded(update_checkpoint=False)

        support_elevations = self.furniture_tools._major_support_elevations()
        if room_kit is not None:
            pruned = _normalize_dense_library_bookcases(
                self.scene,
                room_kit,
                support_elevations,
                remove_object=self.furniture_tools._remove_furniture_impl,
            )
            if pruned:
                console_logger.info(
                    "Pruned %d surplus bookcases from explicit dense library",
                    pruned,
                )
            _validate_furniture_collision_free(
                self.scene,
                self.cfg.physics_validation,
            )

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()
        _validate_room_kit_completion(
            self.scene,
            room_kit,
            support_elevations=support_elevations,
            enforce_exact_level_counts=True,
        )

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving final furniture placement state.

        Returns:
            Path to scene_states/furniture directory.
        """
        return self.logger.output_dir / "scene_states" / "furniture"

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Furniture-specific critic instruction prompt.
        """
        return FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Furniture-specific initial design instruction prompt.
        """
        return FurnitureAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dict with scene description and reference image flag.
        """
        return {
            "scene_description": self.scene.text_description,
            "has_reference_image": self.context_image_path is not None,
            "room_kit_brief": self.room_kit_brief,
        }

    def _get_context_image_path(self) -> Path | None:
        """Get the AI-generated context image for initial design.

        Returns:
            Path to context image if available, None otherwise.
        """
        return self.context_image_path

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Furniture-specific design change instruction prompt.
        """
        return FurnitureAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION_STATEFUL

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for furniture tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        self.furniture_tools.set_noise_profile(mode)

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        """Get extra kwargs for critic prompt (reachability context).

        Computes room reachability and formats it for critic context injection.
        This allows the critic to score reachability based on computed metrics.

        Returns:
            Dict with reachability_context and robot_width for prompt template.
        """
        robot_width = self.cfg.reachability.robot_width
        result = compute_reachability(scene=self.scene, robot_width=robot_width)
        reachability_context = format_reachability_for_critic(result)

        return {
            "reachability_context": reachability_context,
            "robot_width": robot_width,
        }
