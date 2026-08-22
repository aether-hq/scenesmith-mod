"""Implicit primitive construction for semantic environment compilation."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Callable

from scenesmith.agent_utils.semantics.environment.models.chambers import (
    EnvironmentRegionSpec,
)
from scenesmith.agent_utils.semantics.environment.models.common import (
    OpeningShape,
    PassageFloorMode,
    PassageProfile,
)
from scenesmith.agent_utils.semantics.environment.models.features import (
    EnvironmentOpeningSpec,
)
from scenesmith.agent_utils.semantics.environment.models.passages import (
    PassageSegmentSpec,
)
from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point3,
)


def _add(first: Point3, second: Point3) -> Point3:
    return tuple(first[axis] + second[axis] for axis in range(3))  # type: ignore[return-value]


def _subtract(first: Point3, second: Point3) -> Point3:
    return tuple(first[axis] - second[axis] for axis in range(3))  # type: ignore[return-value]


def _scale(vector: Point3, amount: float) -> Point3:
    return tuple(component * amount for component in vector)  # type: ignore[return-value]


def _dot(first: Point3, second: Point3) -> float:
    return sum(first[axis] * second[axis] for axis in range(3))


def _cross(first: Point3, second: Point3) -> Point3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _length(vector: Point3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Point3) -> Point3:
    length = _length(vector)
    if length <= GEOMETRY_TOLERANCE:
        raise ValueError("cannot normalize a zero-length vector")
    return _scale(vector, 1.0 / length)


def _rotate_rpy(point: Point3, rotation_rpy: Point3) -> Point3:
    roll, pitch, yaw = rotation_rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y, z = point
    rolled = (x, cr * y - sr * z, sr * y + cr * z)
    pitched = (
        cp * rolled[0] + sp * rolled[2],
        rolled[1],
        -sp * rolled[0] + cp * rolled[2],
    )
    return (
        cy * pitched[0] - sy * pitched[1],
        sy * pitched[0] + cy * pitched[1],
        pitched[2],
    )


def _transform_point(point: Point3, region: EnvironmentRegionSpec) -> Point3:
    return _add(
        _rotate_rpy(point, region.transform.rotation_rpy),
        region.transform.translation,
    )


@dataclass(frozen=True)
class _ImplicitPrimitive:
    source_id: str
    source_kind: str
    floor_mode: PassageFloorMode | None
    minimum: Point3
    maximum: Point3
    evaluate: Callable[[Point3], float]
    open_planes: tuple[tuple[Point3, Point3], ...] = ()


def _opening_primitive(
    opening: EnvironmentOpeningSpec, region: EnvironmentRegionSpec
) -> _ImplicitPrimitive:
    """Create an outward aperture volume unioned with its source chamber."""

    center = _transform_point(opening.center, region)
    normal = _normalize(_rotate_rpy(opening.normal, region.transform.rotation_rpy))
    reference = (0.0, 0.0, 1.0)
    if abs(_dot(normal, reference)) > 0.95:
        reference = (1.0, 0.0, 0.0)
    across = _normalize(_cross(reference, normal))
    vertical = _normalize(_cross(normal, across))
    width, height = opening.size
    half_depth = opening.depth / 2.0
    volume_center = _add(center, _scale(normal, half_depth))
    extents = tuple(
        abs(across[axis]) * width / 2.0
        + abs(vertical[axis]) * height / 2.0
        + abs(normal[axis]) * half_depth
        for axis in range(3)
    )
    minimum = tuple(volume_center[axis] - extents[axis] for axis in range(3))
    maximum = tuple(volume_center[axis] + extents[axis] for axis in range(3))

    def evaluate(point: Point3) -> float:
        relative = _subtract(point, volume_center)
        u = _dot(relative, across) / (width / 2.0)
        v = _dot(relative, vertical) / (height / 2.0)
        depth = _dot(relative, normal) / half_depth
        if opening.shape == OpeningShape.ELLIPSE:
            radial = u * u + v * v - 1.0
        else:
            radial = max(abs(u), abs(v)) - 1.0
        return max(radial, abs(depth) - 1.0)

    return _ImplicitPrimitive(
        source_id=opening.opening_id,
        source_kind="environment_opening",
        floor_mode=None,
        minimum=minimum,  # type: ignore[arg-type]
        maximum=maximum,  # type: ignore[arg-type]
        evaluate=evaluate,
        open_planes=(
            (_add(center, _scale(normal, opening.depth)), _scale(normal, -1)),
        ),
    )


def _interpolate_cross_section(
    segment: PassageSegmentSpec, station: float
) -> tuple[float, float]:
    sections = segment.cross_sections
    for first, second in zip(sections, sections[1:]):
        if station <= second.station + GEOMETRY_TOLERANCE:
            span = second.station - first.station
            amount = min(1.0, max(0.0, (station - first.station) / span))
            return (
                first.width + (second.width - first.width) * amount,
                first.height + (second.height - first.height) * amount,
            )
    return sections[-1].width, sections[-1].height


def _passage_profile_value(
    profile: PassageProfile,
    across_distance: float,
    vertical_distance: float,
) -> float:
    """Evaluate normalized profiles with floor at -1 and crown at +1."""

    if profile == PassageProfile.ELLIPSE:
        return across_distance**2 + vertical_distance**2 - 1.0
    if profile == PassageProfile.SLOT:
        return abs(across_distance) ** 6 + abs(vertical_distance) ** 6 - 1.0
    if profile == PassageProfile.ARCHED:
        lower_rectangle = max(
            abs(across_distance) - 1.0,
            -vertical_distance - 1.0,
            vertical_distance,
        )
        upper_ellipse = max(
            across_distance**2 + vertical_distance**2 - 1.0,
            -vertical_distance,
        )
        return min(lower_rectangle, upper_ellipse)
    if profile == PassageProfile.KEYHOLE:
        lower_slot = max(
            abs(across_distance) / 0.55 - 1.0,
            -vertical_distance - 1.0,
            vertical_distance - 0.15,
        )
        crown = max(
            across_distance**2 + ((vertical_distance - 0.15) / 0.85) ** 2 - 1.0,
            0.15 - vertical_distance,
        )
        return min(lower_slot, crown)
