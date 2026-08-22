"""Versioned, LLM-authorable primitives for semantic 3D environments.

The declarations in this module describe navigable voids rather than mesh
topology.  They are dependency-light, deterministic, and deliberately keep
tessellation choices out of authored scene data.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from scenesmith.agent_utils.structure.geometry_models.common import Point3
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    require_safe_identifier,
)

SEMANTIC_ENVIRONMENT_SCHEMA_VERSION = 1
_REFERENCE_TOLERANCE = 1e-6
MAX_DETAIL_INSTANCES_PER_FIELD = 10_000
MAX_DETAIL_INSTANCES_PER_SCENE = 50_000
MAX_PASSAGE_SEGMENTS_PER_SCENE = 10_000
MAX_PATH_POINTS_PER_SEGMENT = 10_000


@dataclass(frozen=True)
class SemanticTransform3D:
    """Strict JSON-facing rigid transform for semantic authoring."""

    translation: Point3 = (0.0, 0.0, 0.0)
    rotation_rpy: Point3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _point(self.translation, "translation"))
        object.__setattr__(
            self, "rotation_rpy", _point(self.rotation_rpy, "rotation_rpy")
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "translation": list(self.translation),
            "rotation_rpy": list(self.rotation_rpy),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SemanticTransform3D":
        if data is None:
            return cls()
        _reject_unknown_fields(data, {"translation", "rotation_rpy"})
        return cls(
            translation=tuple(data.get("translation", (0.0, 0.0, 0.0))),
            rotation_rpy=tuple(data.get("rotation_rpy", (0.0, 0.0, 0.0))),
        )


def _identifier(value: Any, label: str) -> str:
    return require_safe_identifier(value, label)


def _strict_bool(value: Any, label: str, *, entity_id: str | None = None) -> bool:
    if type(value) is not bool:
        raise GeometryValidationError(
            "invalid_boolean", f"{label} must be a JSON boolean", entity_id=entity_id
        )
    return value


def _strict_int(value: Any, label: str, *, entity_id: str | None = None) -> int:
    if type(value) is not int:
        raise GeometryValidationError(
            "invalid_integer", f"{label} must be a JSON integer", entity_id=entity_id
        )
    return value


def _finite(value: Any, label: str, *, entity_id: str | None = None) -> float:
    if type(value) not in {int, float}:
        raise GeometryValidationError(
            "invalid_number", f"{label} must be a JSON number", entity_id=entity_id
        )
    number = float(value)
    if not math.isfinite(number):
        raise GeometryValidationError(
            "invalid_number", f"{label} must be finite", entity_id=entity_id
        )
    return number


def _point(value: Sequence[Any], label: str, *, entity_id: str | None = None) -> Point3:
    if len(value) != 3:
        raise GeometryValidationError(
            "invalid_point",
            f"{label} must contain exactly three coordinates",
            entity_id=entity_id,
        )
    return tuple(
        _finite(component, f"{label}[{axis}]", entity_id=entity_id)
        for axis, component in enumerate(value)
    )  # type: ignore[return-value]


def _positive_point(
    value: Sequence[Any], label: str, *, entity_id: str | None = None
) -> Point3:
    point = _point(value, label, entity_id=entity_id)
    if any(component <= 0.0 for component in point):
        raise GeometryValidationError(
            "invalid_size",
            f"{label} components must all be positive",
            entity_id=entity_id,
        )
    return point


def _rotate_rpy(point: Point3, rotation_rpy: Point3) -> Point3:
    """Apply intrinsic XYZ roll/pitch/yaw as Rz * Ry * Rx."""

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


def _unique_ids(items: Sequence[Any], attribute: str, label: str) -> None:
    identifiers = [getattr(item, attribute) for item in items]
    duplicates = sorted(
        identifier
        for identifier in set(identifiers)
        if identifiers.count(identifier) > 1
    )
    if duplicates:
        raise GeometryValidationError(
            f"duplicate_{label}",
            f"duplicate {label} IDs: {', '.join(duplicates)}",
        )


def _reject_unknown_fields(
    data: Mapping[str, Any], allowed: set[str], *, entity_id: str | None = None
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise GeometryValidationError(
            "unknown_semantic_field",
            "unknown fields: " + ", ".join(unknown),
            entity_id=entity_id,
        )


class EnvironmentKind(str, Enum):
    CONSTRUCTED = "constructed"
    SUBTERRANEAN = "subterranean"
    EXTERIOR = "exterior"
    HYBRID = "hybrid"


class CavernShape(str, Enum):
    ELLIPSOID = "ellipsoid"
    SUPERELLIPSOID = "superellipsoid"
    LOFT = "loft"
    VAULTED = "vaulted"
    MESH = "mesh"


class PassageProfile(str, Enum):
    ELLIPSE = "ellipse"
    KEYHOLE = "keyhole"
    SLOT = "slot"
    ARCHED = "arched"


class PassageFloorMode(str, Enum):
    NATURAL = "natural"
    GRADED = "graded"
    STEPS = "steps"
    NON_TRAVERSABLE = "non_traversable"


class OpeningShape(str, Enum):
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"


class OpeningTarget(str, Enum):
    SKY = "sky"
    EXTERIOR = "exterior"


class FormationType(str, Enum):
    STALACTITE = "stalactite"
    STALAGMITE = "stalagmite"
    COLUMN = "column"
    FLOWSTONE = "flowstone"
    BOULDER = "boulder"
    RUBBLE = "rubble"
    SCREE = "scree"


class DetailSurfaceRole(str, Enum):
    OVERHEAD = "overhead"
    SUPPORT = "support"
    BOUNDARY = "boundary"


class DetailCollisionPolicy(str, Enum):
    VISUAL_ONLY = "visual_only"
    COARSE = "coarse"
    FULL = "full"


FORMATION_SURFACE_ROLES: Mapping[FormationType, frozenset[DetailSurfaceRole]] = {
    FormationType.STALACTITE: frozenset({DetailSurfaceRole.OVERHEAD}),
    FormationType.STALAGMITE: frozenset({DetailSurfaceRole.SUPPORT}),
    FormationType.COLUMN: frozenset(
        {DetailSurfaceRole.OVERHEAD, DetailSurfaceRole.SUPPORT}
    ),
    FormationType.FLOWSTONE: frozenset(
        {DetailSurfaceRole.SUPPORT, DetailSurfaceRole.BOUNDARY}
    ),
    FormationType.BOULDER: frozenset({DetailSurfaceRole.SUPPORT}),
    FormationType.RUBBLE: frozenset({DetailSurfaceRole.SUPPORT}),
    FormationType.SCREE: frozenset({DetailSurfaceRole.SUPPORT}),
}


class HeroFeatureType(str, Enum):
    ROCK_SPIRE = "rock_spire"
    BOULDER = "boulder"
