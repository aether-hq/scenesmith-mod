"""Project locked semantic furniture groups into surviving procedural geometry."""

from __future__ import annotations

import logging
import math
import os
import re

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenesmith.agent_utils.scene_blueprint import (
    BlueprintConstraint,
    FurnitureGroupBlueprint,
    SceneBlueprint,
)

console_logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_GROUP_OBJECT_BUDGET = 128
_LARGE_HERO_DIMENSION_M = 5.0
_DENSE_GROUP_COUNT = 5


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_") or "object"


def _constraint_for_role(
    blueprint: SceneBlueprint,
    group: FurnitureGroupBlueprint,
    role: str,
) -> BlueprintConstraint | None:
    matches = [
        constraint
        for constraint in blueprint.constraints
        if group.group_id in constraint.target_ids
        and str(constraint.parameters.get("role_key") or "") == role
        and constraint.strength == "hard"
    ]
    return matches[0] if matches else None


def _dimensions(
    constraint: BlueprintConstraint | None,
    role: str,
) -> tuple[float, float, float]:
    if constraint is not None:
        raw = constraint.parameters.get("preferred_dimensions_m")
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            values = tuple(float(value) for value in raw)
            if min(values) > 0:
                return values
    role_words = role.casefold()
    if "machine" in role_words:
        return (3.0, 2.5, 3.2)
    if any(word in role_words for word in ("rack", "shelf", "parts")):
        return (2.4, 0.8, 3.0)
    if any(word in role_words for word in ("fighter", "craft", "vehicle")):
        return (18.0, 8.0, 5.0)
    if any(word in role_words for word in ("bay", "zone", "booth")):
        return (8.0, 6.0, 4.0)
    return (1.4, 1.0, 1.4)


def _requires_materialization(
    blueprint: SceneBlueprint,
    group: FurnitureGroupBlueprint,
    requirement_ids: frozenset[str] | None = None,
) -> bool:
    if group.group_id not in blueprint.locked_ids:
        return False
    if requirement_ids is not None:
        return any(
            constraint is not None
            and str(constraint.parameters.get("requirement_id") or "")
            in requirement_ids
            for role in group.roles
            for constraint in (_constraint_for_role(blueprint, group, role),)
        )
    if any(count >= _DENSE_GROUP_COUNT for count in group.roles.values()):
        return True
    for role in group.roles:
        constraint = _constraint_for_role(blueprint, group, role)
        if constraint is None:
            continue
        if constraint.kind == "semantic_repeated_zone":
            return True
        if (
            constraint.kind == "semantic_hero_object"
            and max(_dimensions(constraint, role)) >= _LARGE_HERO_DIMENSION_M
        ):
            return True
    return False


def _material(
    rgba: tuple[float, float, float, float],
) -> str:
    values = " ".join(str(value) for value in rgba)
    return (
        f"<material><ambient>{values}</ambient><diffuse>{values}</diffuse></material>"
    )


def _box(
    name: str,
    size: tuple[float, float, float],
    pose: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    *,
    collision: bool = True,
) -> str:
    size_text = " ".join(f"{value:.6g}" for value in size)
    pose_text = " ".join(f"{value:.6g}" for value in (*pose, 0.0, 0.0, 0.0))
    collision_xml = (
        f"<collision name='{name}_collision'><pose>{pose_text}</pose>"
        f"<geometry><box><size>{size_text}</size></box></geometry></collision>"
        if collision
        else ""
    )
    return (
        f"<visual name='{name}_visual'><pose>{pose_text}</pose>"
        f"<geometry><box><size>{size_text}</size></box></geometry>"
        f"{_material(rgba)}</visual>{collision_xml}"
    )


def _zone_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    steel = (0.12, 0.16, 0.20, 1.0)
    hazard = (0.95, 0.55, 0.04, 1.0)
    pieces = [
        _box(
            "service_pad",
            (x, y, 0.16),
            (0.0, 0.0, 0.08),
            steel,
            collision=False,
        ),
        _box("back_wall", (x, 0.30, z), (0.0, -y / 2.0 + 0.15, z / 2.0), steel),
        _box(
            "left_wall", (0.30, y, z * 0.55), (-x / 2.0 + 0.15, 0.0, z * 0.275), steel
        ),
        _box(
            "right_wall", (0.30, y, z * 0.55), (x / 2.0 - 0.15, 0.0, z * 0.275), steel
        ),
        _box(
            "left_pylon",
            (0.65, 0.65, z),
            (-x / 2.0 + 0.325, y / 2.0 - 0.325, z / 2.0),
            hazard,
        ),
        _box(
            "right_pylon",
            (0.65, 0.65, z),
            (x / 2.0 - 0.325, y / 2.0 - 0.325, z / 2.0),
            hazard,
        ),
        _box("header", (x, 0.65, 0.65), (0.0, y / 2.0 - 0.325, z - 0.325), hazard),
    ]
    return "".join(pieces)


def _fighter_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    hull = (0.24, 0.31, 0.38, 1.0)
    dark = (0.05, 0.07, 0.10, 1.0)
    accent = (0.92, 0.42, 0.04, 1.0)
    canopy = (0.08, 0.35, 0.50, 1.0)
    return "".join(
        (
            _box(
                "fuselage", (x * 0.74, y * 0.34, z * 0.24), (0.0, 0.0, z * 0.30), hull
            ),
            _box(
                "nose",
                (x * 0.23, y * 0.22, z * 0.17),
                (x * 0.43, 0.0, z * 0.30),
                accent,
            ),
            _box("wings", (x * 0.42, y, z * 0.07), (-x * 0.06, 0.0, z * 0.28), dark),
            _box(
                "canopy",
                (x * 0.22, y * 0.28, z * 0.17),
                (x * 0.16, 0.0, z * 0.48),
                canopy,
                collision=False,
            ),
            _box(
                "tail", (x * 0.20, y * 0.48, z * 0.30), (-x * 0.39, 0.0, z * 0.42), hull
            ),
            _box(
                "vertical_fin",
                (x * 0.12, y * 0.08, z * 0.62),
                (-x * 0.37, 0.0, z * 0.69),
                accent,
            ),
            _box(
                "landing_skid_left",
                (x * 0.35, y * 0.07, z * 0.06),
                (0.0, -y * 0.26, z * 0.03),
                dark,
            ),
            _box(
                "landing_skid_right",
                (x * 0.35, y * 0.07, z * 0.06),
                (0.0, y * 0.26, z * 0.03),
                dark,
            ),
        )
    )


def _machine_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    steel = (0.18, 0.22, 0.25, 1.0)
    hazard = (0.95, 0.55, 0.04, 1.0)
    screen = (0.04, 0.50, 0.62, 1.0)
    return "".join(
        (
            _box("base", (x, y, z * 0.18), (0.0, 0.0, z * 0.09), steel),
            _box(
                "column",
                (x * 0.22, y * 0.34, z * 0.72),
                (-x * 0.28, 0.0, z * 0.54),
                steel,
            ),
            _box(
                "service_arm",
                (x * 0.72, y * 0.18, z * 0.14),
                (x * 0.08, 0.0, z * 0.82),
                hazard,
            ),
            _box(
                "tool_head",
                (x * 0.18, y * 0.45, z * 0.28),
                (x * 0.37, 0.0, z * 0.66),
                steel,
            ),
            _box(
                "control_screen",
                (x * 0.22, y * 0.06, z * 0.20),
                (-x * 0.05, -y * 0.20, z * 0.48),
                screen,
                collision=False,
            ),
        )
    )


def _rack_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    steel = (0.18, 0.21, 0.24, 1.0)
    parts = (0.78, 0.32, 0.08, 1.0)
    pieces = []
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            pieces.append(
                _box(
                    f"post_{x_sign}_{y_sign}",
                    (0.10, 0.10, z),
                    (x_sign * (x / 2.0 - 0.05), y_sign * (y / 2.0 - 0.05), z / 2.0),
                    steel,
                )
            )
    for index, height in enumerate((0.18, 0.45, 0.72, 0.96)):
        pieces.append(
            _box(f"shelf_{index}", (x, y, 0.10), (0.0, 0.0, z * height), steel)
        )
    pieces.append(
        _box("parts_bins", (x * 0.82, y * 0.78, z * 0.18), (0.0, 0.0, z * 0.58), parts)
    )
    return "".join(pieces)


def _equipment_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    steel = (0.20, 0.24, 0.28, 1.0)
    hazard = (0.92, 0.45, 0.04, 1.0)
    return "".join(
        (
            _box("equipment_case", (x, y, z * 0.72), (0.0, 0.0, z * 0.36), steel),
            _box(
                "hazard_band",
                (x * 1.01, y * 1.01, z * 0.10),
                (0.0, 0.0, z * 0.48),
                hazard,
                collision=False,
            ),
            _box(
                "top_module",
                (x * 0.62, y * 0.62, z * 0.28),
                (0.0, 0.0, z * 0.86),
                steel,
            ),
        )
    )


def _write_role_sdf(
    output_dir: Path,
    role: str,
    dimensions: tuple[float, float, float],
    kind: str,
) -> Path:
    role_dir = output_dir / _slug(role)
    role_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = role_dir / "model.sdf"
    role_words = role.casefold()
    if kind == "semantic_repeated_zone" or any(
        word in role_words for word in ("bay", "zone", "booth")
    ):
        geometry = _zone_geometry(dimensions)
    elif kind == "semantic_hero_object" or any(
        word in role_words for word in ("fighter", "craft", "vehicle")
    ):
        geometry = _fighter_geometry(dimensions)
    elif "machine" in role_words:
        geometry = _machine_geometry(dimensions)
    elif any(word in role_words for word in ("rack", "shelf", "parts")):
        geometry = _rack_geometry(dimensions)
    else:
        geometry = _equipment_geometry(dimensions)
    model_name = _slug(role)
    sdf_path.write_text(
        "<?xml version='1.0'?>"
        "<sdf version='1.9'>"
        f"<model name='{model_name}'><static>true</static>"
        f"<link name='base_link'>{geometry}</link></model></sdf>",
        encoding="utf-8",
    )
    return sdf_path


def _zone_poses(
    count: int,
    dimensions: tuple[float, float, float],
    room_length: float,
    room_width: float,
) -> list[tuple[float, float, float]]:
    length, depth, _ = dimensions
    gap = max(1.0, min(length, depth) * 0.12)
    per_row = math.ceil(count / 2)
    occupied = per_row * length + max(0, per_row - 1) * gap
    if occupied > room_length or depth + 1.0 > room_width / 2.0:
        raise RuntimeError(
            "Locked repeated zones do not fit the accepted room footprint without "
            f"shrinking them: {count} zones at {dimensions}m in "
            f"{room_length}x{room_width}m. Expand the semantic space or reduce the "
            "source-requested count; SceneSmith will not silently clamp dimensions."
        )
    x_start = -occupied / 2.0 + length / 2.0
    edge_y = room_width / 2.0 - depth / 2.0 - 0.5
    poses = []
    for index in range(count):
        north = index >= per_row
        row_index = index - per_row if north else index
        row_count = count - per_row if north else per_row
        row_occupied = row_count * length + max(0, row_count - 1) * gap
        row_start = -row_occupied / 2.0 + length / 2.0
        x = row_start + row_index * (length + gap)
        poses.append((x, edge_y if north else -edge_y, math.pi if north else 0.0))
    return poses


def _grid_poses(
    count: int,
    dimensions: tuple[float, float, float],
    room_length: float,
    room_width: float,
) -> list[tuple[float, float, float]]:
    x_size, y_size, _ = dimensions
    columns = max(1, math.ceil(math.sqrt(count * room_length / room_width)))
    rows = math.ceil(count / columns)
    x_step = room_length / max(columns, 1)
    y_step = room_width / max(rows, 1)
    if x_size > x_step or y_size > y_step:
        raise RuntimeError(
            f"Locked semantic objects at {dimensions}m do not fit a collision-free "
            f"{columns}x{rows} construction grid inside {room_length}x{room_width}m. "
            "SceneSmith will not silently shrink them."
        )
    return [
        (
            -room_length / 2.0 + x_step * (column + 0.5),
            -room_width / 2.0 + y_step * (row + 0.5),
            0.0,
        )
        for row in range(rows)
        for column in range(columns)
    ][:count]


def _equipment_poses(
    role: str,
    count: int,
    zone_poses: list[tuple[float, float, float]],
    zone_dimensions: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    poses = []
    zone_length, zone_depth, _ = zone_dimensions
    for index in range(count):
        zone_x, zone_y, zone_yaw = zone_poses[index % len(zone_poses)]
        north = zone_y > 0
        inward = -1.0 if north else 1.0
        lane = index // len(zone_poses)
        if "machine" in role.casefold():
            x_offset = 0.0
            y_offset = -inward * zone_depth * 0.15
        elif any(word in role.casefold() for word in ("rack", "shelf", "parts")):
            x_offset = (-0.30 if lane % 2 == 0 else 0.30) * zone_length
            y_offset = inward * zone_depth * 0.08
        else:
            x_offset = ((index % 3) - 1) * zone_length * 0.18
            y_offset = inward * zone_depth * (0.30 + 0.08 * lane)
        poses.append((zone_x + x_offset, zone_y + y_offset, zone_yaw))
    return poses


def _description(
    group: FurnitureGroupBlueprint,
    role: str,
    dimensions: tuple[float, float, float],
    constraint: BlueprintConstraint | None,
) -> str:
    relationships = constraint.parameters.get("relationships", ()) if constraint else ()
    relation_text = "; ".join(
        f"{item.get('predicate', 'related_to').replace('_', ' ')} "
        f"{item.get('target', 'the scene')}"
        for item in relationships
        if isinstance(item, dict)
    )
    dimensions_text = " x ".join(f"{value:g}" for value in dimensions)
    return (
        f"Concrete {role.replace('_', ' ')} in {group.name}; authored dimensions "
        f"{dimensions_text} meters. {relation_text}".strip()
    )


def materialize_locked_semantic_groups(
    scene: Any,
    blueprint: SceneBlueprint,
    output_dir: Path,
    *,
    max_objects: int | None = None,
    requirement_ids: frozenset[str] | None = None,
) -> tuple[UniqueID, ...]:
    """Add high-impact locked blueprint groups as immutable procedural objects.

    Metric dimensions are never used as a safety budget. The only default bound is
    an explicit object-count budget, which limits downstream solver and rendering
    work without changing the requested real-world scale.
    """

    groups = [
        group
        for group in blueprint.furniture_groups
        if _requires_materialization(blueprint, group, requirement_ids)
    ]
    requested = sum(sum(group.roles.values()) for group in groups)
    budget = max_objects
    if budget is None:
        budget = int(
            os.environ.get(
                "SCENESMITH_SEMANTIC_GROUP_OBJECT_BUDGET",
                DEFAULT_SEMANTIC_GROUP_OBJECT_BUDGET,
            )
        )
    if requested > budget:
        raise RuntimeError(
            "Locked semantic groups request "
            f"{requested} concrete objects, exceeding the explicit construction "
            f"budget of {budget}. Increase SCENESMITH_SEMANTIC_GROUP_OBJECT_BUDGET; "
            "SceneSmith will not reduce counts or dimensions silently."
        )

    existing = {
        (
            str(obj.metadata.get("semantic_blueprint_group_id")),
            str(obj.metadata.get("role")),
            int(obj.metadata.get("semantic_instance", -1)),
        )
        for obj in scene.objects.values()
        if obj.metadata.get("semantic_blueprint_group_id")
    }
    room_length = float(scene.room_geometry.length)
    room_width = float(scene.room_geometry.width)
    generated_dir = output_dir / "generated_assets" / "semantic_groups"
    created: list[UniqueID] = []
    zone_anchors: list[tuple[float, float, float]] = []
    zone_dimensions: tuple[float, float, float] | None = None

    def priority(group: FurnitureGroupBlueprint) -> int:
        kinds = {
            (
                _constraint_for_role(blueprint, group, role)
                or BlueprintConstraint(constraint_id="fallback", kind="fallback")
            ).kind
            for role in group.roles
        }
        if "semantic_hero_object" in kinds:
            return 0
        if "semantic_repeated_zone" in kinds:
            return 1
        return 2

    for group in sorted(groups, key=priority):
        for role, count in group.roles.items():
            constraint = _constraint_for_role(blueprint, group, role)
            kind = (
                constraint.kind if constraint is not None else "semantic_object_group"
            )
            dimensions = _dimensions(constraint, role)
            if dimensions[0] > room_length or dimensions[1] > room_width:
                raise RuntimeError(
                    f"Locked semantic role {role!r} has dimensions {dimensions}m, "
                    f"larger than its accepted {room_length}x{room_width}m room. "
                    "SceneSmith will not clamp the role to fit."
                )
            if kind == "semantic_hero_object":
                poses = (
                    [(0.0, 0.0, 0.0)]
                    if count == 1
                    else _grid_poses(count, dimensions, room_length, room_width)
                )
            elif kind == "semantic_repeated_zone":
                poses = _zone_poses(count, dimensions, room_length, room_width)
                zone_anchors = poses
                zone_dimensions = dimensions
            elif group.focal_target and zone_anchors and zone_dimensions is not None:
                poses = _equipment_poses(role, count, zone_anchors, zone_dimensions)
            else:
                poses = _grid_poses(count, dimensions, room_length, room_width)
            sdf_path = _write_role_sdf(generated_dir, role, dimensions, kind)
            for index, (x, y, yaw) in enumerate(poses):
                key = (group.group_id, role, index)
                if key in existing:
                    continue
                object_id = UniqueID(f"semantic_{_slug(role)}_{index:03d}")
                if object_id in scene.objects:
                    object_id = scene.generate_unique_id(f"semantic_{_slug(role)}")
                description = _description(group, role, dimensions, constraint)
                scene.add_object(
                    SceneObject(
                        object_id=object_id,
                        object_type=ObjectType.FURNITURE,
                        name=role,
                        description=description,
                        transform=RigidTransform(
                            rpy=RollPitchYaw(roll=0.0, pitch=0.0, yaw=yaw),
                            p=[x, y, 0.0],
                        ),
                        sdf_path=sdf_path,
                        bbox_min=np.array(
                            [-dimensions[0] / 2.0, -dimensions[1] / 2.0, 0.0]
                        ),
                        bbox_max=np.array(
                            [dimensions[0] / 2.0, dimensions[1] / 2.0, dimensions[2]]
                        ),
                        immutable=True,
                        metadata={
                            "asset_source": "procedural_semantic_group",
                            "generated_from": "locked_scene_blueprint",
                            "semantic_blueprint_group_id": group.group_id,
                            "semantic_instance": float(index),
                            "role": role,
                            "construction_kind": kind,
                        },
                    )
                )
                created.append(object_id)

    if created:
        console_logger.info(
            "Materialized %d locked semantic group objects within explicit budget %d",
            len(created),
            budget,
        )
    return tuple(created)


def load_and_materialize_locked_semantic_groups(
    scene: Any,
    scene_dir: Path,
    *,
    requirement_ids: frozenset[str] | None = None,
) -> tuple[UniqueID, ...]:
    """Load an adjacent blueprint and project its locked groups if present."""

    blueprint_path = scene_dir / "scene_blueprint.json"
    if not blueprint_path.is_file():
        return ()
    blueprint = SceneBlueprint.model_validate_json(
        blueprint_path.read_text(encoding="utf-8")
    )
    return materialize_locked_semantic_groups(
        scene,
        blueprint,
        scene_dir,
        requirement_ids=requirement_ids,
    )
