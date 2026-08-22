"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import logging

from typing import Any

from agents.exceptions import ModelBehaviorError

from scenesmith.agent_utils.design.room_kits import RoomKitSelection
from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType

console_logger = logging.getLogger(__name__)

from scenesmith.furniture_agents.room_kit.planning import (
    _bookcase_wall_run_level_counts,
    _object_matches_room_kit_slot,
    _patron_ensemble_level_counts,
    _required_room_kit_exact_level_counts,
    _required_room_kit_level_coverage,
    _required_room_kit_role_count,
    _room_kit_role_level_counts,
)


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
    if furniture_count < required_minimum:
        raise ModelBehaviorError(
            f"Semantic room kit {room_kit.kit_id} placed {furniture_count} "
            f"furniture objects; required minimum is {required_minimum}. "
            "The furniture stage cannot publish this checkpoint."
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
            targets = _required_room_kit_exact_level_counts(
                scene,
                room_kit,
                bookshelf_slot,
                support_elevations,
            )
            mismatches = [
                (elevation, counts[elevation], targets[elevation])
                for elevation in support_elevations
                if counts[elevation] != targets[elevation]
            ]
            if mismatches:
                details = "; ".join(
                    f"bookshelf at {elevation:.3f}m placed {placed}, required "
                    f"exactly {target}"
                    for elevation, placed, target in mismatches
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
