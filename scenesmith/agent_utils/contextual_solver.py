"""Fast deterministic spatial constraints for scene composition.

The solver operates on semantic zones rather than triangle meshes.  It is a
bounded first line of defense for obviously unusable layouts; physics remains
the final collision authority.
"""

from __future__ import annotations

import json
import math
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from scenesmith.agent_utils.asset_semantics import semantic_families


class ZoneKind(str, Enum):
    OCCUPANCY = "occupancy"
    SUPPORT = "support"
    ACCESS = "access"
    INTERACTION = "interaction"
    CIRCULATION = "circulation"
    VISIBILITY = "visibility"


@dataclass(frozen=True)
class OrientedZone:
    zone_id: str
    owner_id: str
    kind: ZoneKind
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    yaw_radians: float = 0.0
    z_interval: tuple[float, float] = (0.0, 0.0)
    hard: bool = True
    target_ids: tuple[str, ...] = ()

    def corners(self) -> np.ndarray:
        hx, hy = self.half_extents_xy
        local = np.array(((-hx, -hy), (-hx, hy), (hx, hy), (hx, -hy)))
        cosine, sine = math.cos(self.yaw_radians), math.sin(self.yaw_radians)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        return local @ rotation.T + np.array(self.center_xy)


@dataclass(frozen=True)
class SpatialEntity:
    entity_id: str
    name: str
    families: frozenset[str]
    center_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]
    yaw_radians: float
    context: Literal["active", "stacked", "stored"] = "active"
    zones: tuple[OrientedZone, ...] = ()

    @property
    def forward_xy(self) -> np.ndarray:
        # SceneSmith's canonical local front is +Y.
        return np.array((-math.sin(self.yaw_radians), math.cos(self.yaw_radians)))


@dataclass(frozen=True)
class SolverViolation:
    code: str
    message: str
    object_ids: tuple[str, ...]
    severity: Literal["hard", "soft"] = "hard"
    repair: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "object_ids": list(self.object_ids),
            "severity": self.severity,
            "repair": self.repair,
        }


@dataclass(frozen=True)
class SolverResult:
    valid: bool
    violations: tuple[SolverViolation, ...] = ()
    evaluations: int = 0
    elapsed_ms: float = 0.0
    selected_pose: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [violation.to_dict() for violation in self.violations],
            "evaluations": self.evaluations,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "selected_pose": list(self.selected_pose) if self.selected_pose else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def zones_intersect(first: OrientedZone, second: OrientedZone) -> bool:
    """Use the separating axis theorem for two oriented rectangles."""

    if (
        first.z_interval[1] < second.z_interval[0]
        or second.z_interval[1] < first.z_interval[0]
    ):
        return False
    first_corners, second_corners = first.corners(), second.corners()
    axes: list[np.ndarray] = []
    for corners in (first_corners, second_corners):
        for index in (0, 1):
            edge = corners[(index + 1) % 4] - corners[index]
            axis = np.array((-edge[1], edge[0]))
            norm = np.linalg.norm(axis)
            if norm > 1e-9:
                axes.append(axis / norm)
    epsilon = 1e-6
    for axis in axes:
        first_projection = first_corners @ axis
        second_projection = second_corners @ axis
        if (
            first_projection.max() <= second_projection.min() + epsilon
            or second_projection.max() <= first_projection.min() + epsilon
        ):
            return False
    return True


def _scene_object_yaw(scene_object: Any) -> float:
    rotation = np.asarray(scene_object.transform.rotation().matrix(), dtype=float)
    local_front = rotation @ np.array((0.0, 1.0, 0.0))
    return math.atan2(-float(local_front[0]), float(local_front[1]))


def scene_object_to_spatial_entity(scene_object: Any) -> SpatialEntity | None:
    """Convert a SceneObject into cheap semantic zones."""

    try:
        bounds = scene_object.compute_world_bounds()
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if bounds is None:
        return None
    world_min, world_max = (np.asarray(value, dtype=float) for value in bounds)
    if world_min.shape != (3,) or world_max.shape != (3,):
        return None
    center = (world_min + world_max) / 2.0
    dimensions = np.maximum(world_max - world_min, 0.001)
    object_id = str(scene_object.object_id)
    text = f"{scene_object.name} {scene_object.description}"
    families = semantic_families(text)
    context_value = str(
        getattr(scene_object, "metadata", {}).get("placement_context", "active")
    )
    context: Literal["active", "stacked", "stored"] = (
        context_value if context_value in {"active", "stacked", "stored"} else "active"
    )
    yaw = _scene_object_yaw(scene_object)
    z_interval = (float(world_min[2]), float(world_max[2]))
    zones: list[OrientedZone] = [
        OrientedZone(
            zone_id=f"{object_id}:occupancy",
            owner_id=object_id,
            kind=ZoneKind.OCCUPANCY,
            center_xy=(float(center[0]), float(center[1])),
            half_extents_xy=(float(dimensions[0] / 2), float(dimensions[1] / 2)),
            yaw_radians=yaw,
            z_interval=z_interval,
        )
    ]
    if context == "active" and families & {"chair", "stool", "sofa", "bench"}:
        forward = np.array((-math.sin(yaw), math.cos(yaw)))
        access_depth = max(0.55, float(dimensions[1]) * 0.8)
        access_center = center[:2] + forward * (
            float(dimensions[1]) / 2 + access_depth / 2
        )
        zones.append(
            OrientedZone(
                zone_id=f"{object_id}:access",
                owner_id=object_id,
                kind=ZoneKind.ACCESS,
                center_xy=(float(access_center[0]), float(access_center[1])),
                half_extents_xy=(
                    max(0.35, float(dimensions[0]) / 2),
                    access_depth / 2,
                ),
                yaw_radians=yaw,
                z_interval=(float(world_min[2]), min(float(world_max[2]), 1.2)),
                hard=False,
            )
        )
    if context == "active" and families & {"storage", "appliance"}:
        forward = np.array((-math.sin(yaw), math.cos(yaw)))
        access_center = center[:2] + forward * (float(dimensions[1]) / 2 + 0.4)
        zones.append(
            OrientedZone(
                zone_id=f"{object_id}:access",
                owner_id=object_id,
                kind=ZoneKind.ACCESS,
                center_xy=(float(access_center[0]), float(access_center[1])),
                half_extents_xy=(float(dimensions[0]) / 2, 0.4),
                yaw_radians=yaw,
                z_interval=z_interval,
                hard=True,
            )
        )
    if families & {"table", "storage"}:
        zones.append(
            OrientedZone(
                zone_id=f"{object_id}:support",
                owner_id=object_id,
                kind=ZoneKind.SUPPORT,
                center_xy=(float(center[0]), float(center[1])),
                half_extents_xy=(
                    float(dimensions[0]) * 0.45,
                    float(dimensions[1]) * 0.45,
                ),
                yaw_radians=yaw,
                z_interval=(float(world_max[2]) - 0.03, float(world_max[2]) + 0.03),
                hard=False,
            )
        )
    return SpatialEntity(
        entity_id=object_id,
        name=str(scene_object.name),
        families=families,
        center_xyz=(float(center[0]), float(center[1]), float(center[2])),
        dimensions_xyz=(
            float(dimensions[0]),
            float(dimensions[1]),
            float(dimensions[2]),
        ),
        yaw_radians=yaw,
        context=context,
        zones=tuple(zones),
    )


def _nearest_family(
    subject: SpatialEntity,
    entities: Sequence[SpatialEntity],
    family: str,
    maximum_distance_m: float,
    maximum_vertical_gap_m: float = 0.5,
) -> SpatialEntity | None:
    choices = [
        entity
        for entity in entities
        if entity.entity_id != subject.entity_id
        and family in entity.families
        and max(
            0.0,
            abs(subject.center_xyz[2] - entity.center_xyz[2])
            - (subject.dimensions_xyz[2] + entity.dimensions_xyz[2]) / 2,
        )
        <= maximum_vertical_gap_m
    ]
    if not choices:
        return None
    choices.sort(
        key=lambda entity: (
            math.dist(subject.center_xyz[:2], entity.center_xyz[:2]),
            entity.entity_id,
        )
    )
    nearest = choices[0]
    return (
        nearest
        if math.dist(subject.center_xyz[:2], nearest.center_xyz[:2])
        <= maximum_distance_m
        else None
    )


def evaluate_spatial_entities(entities: Sequence[SpatialEntity]) -> SolverResult:
    """Evaluate occupancy, access, facing, and semantic support constraints."""

    started = time.monotonic()
    violations: list[SolverViolation] = []
    for index, first in enumerate(entities):
        first_occupancy = next(
            zone for zone in first.zones if zone.kind == ZoneKind.OCCUPANCY
        )
        for second in entities[index + 1 :]:
            second_occupancy = next(
                zone for zone in second.zones if zone.kind == ZoneKind.OCCUPANCY
            )
            if zones_intersect(first_occupancy, second_occupancy):
                violations.append(
                    SolverViolation(
                        code="occupancy_overlap",
                        message=f"{first.name} overlaps {second.name}",
                        object_ids=(first.entity_id, second.entity_id),
                        repair={"action": "separate", "minimum_gap_m": 0.05},
                    )
                )
            for access_owner, other_occupancy in (
                (first, second_occupancy),
                (second, first_occupancy),
            ):
                for access in (
                    zone for zone in access_owner.zones if zone.kind == ZoneKind.ACCESS
                ):
                    if not zones_intersect(access, other_occupancy):
                        continue
                    violations.append(
                        SolverViolation(
                            code="access_zone_blocked",
                            message=f"{access_owner.name} has a blocked access zone",
                            object_ids=(
                                access_owner.entity_id,
                                other_occupancy.owner_id,
                            ),
                            severity="hard" if access.hard else "soft",
                            repair={
                                "action": "clear_zone",
                                "zone_id": access.zone_id,
                            },
                        )
                    )

    for chair in (
        entity
        for entity in entities
        if entity.context == "active" and entity.families & {"chair", "stool"}
    ):
        table = _nearest_family(chair, entities, "table", 2.25)
        if table is None:
            continue
        direction = np.asarray(table.center_xyz[:2]) - np.asarray(chair.center_xyz[:2])
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            continue
        alignment = float(np.dot(chair.forward_xy, direction / distance))
        if alignment < 0.35:
            target_yaw = math.atan2(-float(direction[0]), float(direction[1]))
            violations.append(
                SolverViolation(
                    code="seat_faces_away_from_table",
                    message=f"{chair.name} faces away from its nearest table",
                    object_ids=(chair.entity_id, table.entity_id),
                    repair={
                        "action": "rotate",
                        "yaw_degrees": round(math.degrees(target_yaw), 2),
                    },
                )
            )

    for device in (
        entity
        for entity in entities
        if "medical" in entity.name.casefold()
        and not entity.families.intersection({"bed"})
    ):
        bed = _nearest_family(device, entities, "bed", 2.5)
        if bed is None:
            continue
        device_bottom = device.center_xyz[2] - device.dimensions_xyz[2] / 2
        bed_top = bed.center_xyz[2] + bed.dimensions_xyz[2] / 2
        if device_bottom >= bed_top - 0.05:
            violations.append(
                SolverViolation(
                    code="medical_device_on_bed",
                    message=f"{device.name} is supported by the patient bed instead of floor",
                    object_ids=(device.entity_id, bed.entity_id),
                    repair={"action": "place_beside", "clearance_m": 0.25},
                )
            )

    return SolverResult(
        valid=not any(item.severity == "hard" for item in violations),
        violations=tuple(violations),
        elapsed_ms=(time.monotonic() - started) * 1000.0,
    )


def validate_scene_object_placement(
    candidate: Any, existing: Iterable[Any]
) -> SolverResult:
    """Validate a proposed object against the complete current context."""

    candidate_id = str(candidate.object_id)
    entities = [
        entity
        for scene_object in (*tuple(existing), candidate)
        if (entity := scene_object_to_spatial_entity(scene_object)) is not None
    ]
    result = evaluate_spatial_entities(entities)
    candidate_violations = tuple(
        violation
        for violation in result.violations
        if candidate_id in violation.object_ids
    )
    return SolverResult(
        valid=not any(
            violation.severity == "hard" for violation in candidate_violations
        ),
        violations=candidate_violations,
        evaluations=result.evaluations,
        elapsed_ms=result.elapsed_ms,
        selected_pose=result.selected_pose,
    )


def validate_hosted_object(
    *,
    object_id: str,
    kind: Literal["wall_art", "ceiling_fixture"],
    center_xyz: tuple[float, float, float],
    dimensions_xyz: tuple[float, float, float],
    host_bounds_min: tuple[float, float, float],
    host_bounds_max: tuple[float, float, float],
    normal_alignment: float,
) -> SolverResult:
    """Validate wall/ceiling objects against their semantic host surface."""

    violations: list[SolverViolation] = []
    half = np.asarray(dimensions_xyz) / 2
    lower = np.asarray(center_xyz) - half
    upper = np.asarray(center_xyz) + half
    host_min, host_max = np.asarray(host_bounds_min), np.asarray(host_bounds_max)
    if np.any(lower < host_min - 1e-3) or np.any(upper > host_max + 1e-3):
        violations.append(
            SolverViolation(
                code="host_bounds_exceeded",
                message=f"{kind} extends beyond its host surface",
                object_ids=(object_id,),
                repair={"action": "fit_inside_host"},
            )
        )
    required_alignment = 0.95
    if normal_alignment < required_alignment:
        violations.append(
            SolverViolation(
                code="host_orientation_mismatch",
                message=f"{kind} is not parallel to its host surface",
                object_ids=(object_id,),
                repair={"action": "align_to_host_normal"},
            )
        )
    return SolverResult(
        valid=not violations,
        violations=tuple(violations),
    )


def validate_blueprint_topology(blueprint: Any) -> SolverResult:
    """Check connector endpoint reachability, slope, and basic stair headroom."""

    violations: list[SolverViolation] = []
    levels = {level.level_id: level for level in blueprint.levels}
    for connector in blueprint.connectors:
        rise = connector.end.position_m[2] - connector.start.position_m[2]
        run = math.dist(connector.start.position_m[:2], connector.end.position_m[:2])
        expected = (
            levels[connector.end.level_id].elevation_m
            - levels[connector.start.level_id].elevation_m
        )
        if abs(rise - expected) > 0.15:
            violations.append(
                SolverViolation(
                    code="connector_misses_landing",
                    message=f"{connector.connector_id} does not terminate at its level elevation",
                    object_ids=(connector.connector_id,),
                    repair={"action": "snap_endpoints_to_levels"},
                )
            )
        if connector.kind == "ramp" and run > 0 and abs(rise / run) > 1 / 8:
            violations.append(
                SolverViolation(
                    code="ramp_too_steep",
                    message=f"{connector.connector_id} exceeds a 1:8 slope",
                    object_ids=(connector.connector_id,),
                    repair={"action": "increase_run", "minimum_run_m": abs(rise) * 8},
                )
            )
        lower_clear_height = levels[connector.start.level_id].clear_height_m
        if lower_clear_height < 2.05:
            violations.append(
                SolverViolation(
                    code="connector_headroom",
                    message=f"{connector.connector_id} has less than 2.05m headroom",
                    object_ids=(connector.connector_id,),
                    repair={"action": "increase_clear_height", "minimum_m": 2.05},
                )
            )
    return SolverResult(
        valid=not violations,
        violations=tuple(violations),
    )


def solve_candidate_poses(
    candidate_factory: Any,
    existing: Sequence[Any],
    candidate_poses: Sequence[tuple[float, float, float]],
    *,
    max_evaluations: int = 128,
    timeout_ms: float = 50.0,
) -> SolverResult:
    """Choose the first minimum-penalty pose within strict deterministic bounds."""

    started = time.monotonic()
    best: (
        tuple[tuple[int, int, float, int], SolverResult, tuple[float, float, float]]
        | None
    ) = None
    evaluations = 0
    for index, pose in enumerate(candidate_poses[:max_evaluations]):
        if (time.monotonic() - started) * 1000.0 >= timeout_ms:
            break
        evaluations += 1
        candidate = candidate_factory(pose)
        result = validate_scene_object_placement(candidate, existing)
        hard = sum(item.severity == "hard" for item in result.violations)
        soft = sum(item.severity == "soft" for item in result.violations)
        distance = math.hypot(pose[0], pose[1])
        score = (hard, soft, distance, index)
        if best is None or score < best[0]:
            best = (score, result, pose)
            if hard == 0 and soft == 0:
                break
    elapsed = (time.monotonic() - started) * 1000.0
    if best is None:
        return SolverResult(
            valid=False,
            violations=(
                SolverViolation(
                    code="solver_budget_exhausted",
                    message="No candidate pose was evaluated within the solver budget",
                    object_ids=(),
                    repair={"action": "provide_candidate_poses"},
                ),
            ),
            evaluations=0,
            elapsed_ms=elapsed,
        )
    _, result, selected_pose = best
    return SolverResult(
        valid=result.valid,
        violations=result.violations,
        evaluations=evaluations,
        elapsed_ms=elapsed,
        selected_pose=selected_pose if result.valid else None,
    )
