"""Versioned, LLM-authorable primitives for semantic 3D environments.

The declarations in this module describe navigable voids rather than mesh
topology.  They are dependency-light, deterministic, and deliberately keep
tessellation choices out of authored scene data.
"""

from __future__ import annotations

import hashlib
import json
import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from scenesmith.agent_utils.structural_geometry import (
    GEOMETRY_TOLERANCE,
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
    GeometryValidationError,
    Point2,
    Point3,
    require_safe_identifier,
    validate_global_identifiers,
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


@dataclass(frozen=True)
class Bounds3D:
    """Conservative axis-aligned compilation bounds in scene coordinates."""

    minimum: Point3
    maximum: Point3

    def __post_init__(self) -> None:
        minimum = _point(self.minimum, "minimum")
        maximum = _point(self.maximum, "maximum")
        if any(
            maximum[axis] - minimum[axis] <= GEOMETRY_TOLERANCE for axis in range(3)
        ):
            raise GeometryValidationError(
                "invalid_bounds", "bounds maximum must exceed minimum on every axis"
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(
        self, point: Point3, *, tolerance: float = _REFERENCE_TOLERANCE
    ) -> bool:
        return all(
            self.minimum[axis] - tolerance
            <= point[axis]
            <= self.maximum[axis] + tolerance
            for axis in range(3)
        )

    def contains_box(
        self,
        minimum: Point3,
        maximum: Point3,
        *,
        tolerance: float = _REFERENCE_TOLERANCE,
    ) -> bool:
        return self.contains(minimum, tolerance=tolerance) and self.contains(
            maximum, tolerance=tolerance
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"minimum": list(self.minimum), "maximum": list(self.maximum)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Bounds3D":
        _reject_unknown_fields(data, {"minimum", "maximum", "min", "max"})
        minimum = data.get("minimum", data.get("min"))
        maximum = data.get("maximum", data.get("max"))
        if minimum is None or maximum is None:
            raise GeometryValidationError(
                "invalid_bounds", "bounds require minimum/min and maximum/max"
            )
        return cls(tuple(minimum), tuple(maximum))


@dataclass(frozen=True)
class EnvironmentRegionSpec:
    """Owning region and compilation policy for related semantic geometry."""

    region_id: str
    kind: EnvironmentKind
    bounds: Bounds3D
    transform: SemanticTransform3D = field(default_factory=SemanticTransform3D)
    material_context: Mapping[str, str] = field(default_factory=dict)
    detail_seed: int = 0
    chunk_policy: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(self, "kind", EnvironmentKind(self.kind))
        object.__setattr__(self, "material_context", dict(self.material_context))
        object.__setattr__(self, "chunk_policy", dict(self.chunk_policy))
        object.__setattr__(
            self,
            "detail_seed",
            _strict_int(self.detail_seed, "detail_seed", entity_id=self.region_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "kind": self.kind.value,
            "bounds": self.bounds.to_dict(),
            "transform": self.transform.to_dict(),
            "material_context": dict(sorted(self.material_context.items())),
            "detail_seed": self.detail_seed,
            "chunk_policy": dict(sorted(self.chunk_policy.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentRegionSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "kind",
                "bounds",
                "transform",
                "material_context",
                "detail_seed",
                "chunk_policy",
            },
            entity_id=data.get("id"),
        )
        return cls(
            region_id=data["id"],
            kind=EnvironmentKind(data["kind"]),
            bounds=Bounds3D.from_dict(data["bounds"]),
            transform=SemanticTransform3D.from_dict(data.get("transform")),
            material_context=data.get("material_context", {}),
            detail_seed=data.get("detail_seed", 0),
            chunk_policy=data.get("chunk_policy", {}),
        )


@dataclass(frozen=True)
class CavernChamberSpec:
    """One authored open volume within a surrounding substrate."""

    chamber_id: str
    region_id: str
    center: Point3
    size: Point3
    shape: CavernShape = CavernShape.ELLIPSOID
    orientation_rpy: Point3 = (0.0, 0.0, 0.0)
    substrate_id: str | None = None
    semantic_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        chamber_id = _identifier(self.chamber_id, "chamber_id")
        object.__setattr__(self, "chamber_id", chamber_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self, "center", _point(self.center, "center", entity_id=chamber_id)
        )
        object.__setattr__(
            self, "size", _positive_point(self.size, "size", entity_id=chamber_id)
        )
        object.__setattr__(self, "shape", CavernShape(self.shape))
        object.__setattr__(
            self,
            "orientation_rpy",
            _point(self.orientation_rpy, "orientation_rpy", entity_id=chamber_id),
        )
        object.__setattr__(
            self,
            "semantic_tags",
            frozenset(
                str(tag).strip() for tag in self.semantic_tags if str(tag).strip()
            ),
        )
        if self.substrate_id is not None:
            object.__setattr__(
                self, "substrate_id", _identifier(self.substrate_id, "substrate_id")
            )

    @property
    def bounds(self) -> Bounds3D:
        half = tuple(component / 2.0 for component in self.size)
        axes = tuple(
            _rotate_rpy(axis, self.orientation_rpy)
            for axis in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        extents = tuple(
            sum(
                abs(axes[local_axis][world_axis]) * half[local_axis]
                for local_axis in range(3)
            )
            for world_axis in range(3)
        )
        return Bounds3D(
            tuple(self.center[axis] - extents[axis] for axis in range(3)),
            tuple(self.center[axis] + extents[axis] for axis in range(3)),
        )

    def contains(self, point: Point3, *, tolerance: float = 1e-6) -> bool:
        """Return whether a region-local point lies inside the chamber volume."""

        offset = tuple(point[axis] - self.center[axis] for axis in range(3))
        axes = tuple(
            _rotate_rpy(axis, self.orientation_rpy)
            for axis in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        local = tuple(
            sum(offset[axis] * basis[axis] for axis in range(3)) for basis in axes
        )
        exponent = 2 if self.shape == CavernShape.ELLIPSOID else 4
        value = sum(
            abs(local[axis] / (self.size[axis] / 2.0)) ** exponent for axis in range(3)
        )
        return value <= 1.0 + tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chamber_id,
            "region_id": self.region_id,
            "center": list(self.center),
            "size": list(self.size),
            "shape": self.shape.value,
            "orientation_rpy": list(self.orientation_rpy),
            "substrate_id": self.substrate_id,
            "semantic_tags": sorted(self.semantic_tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CavernChamberSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "center",
                "size",
                "shape",
                "orientation_rpy",
                "substrate_id",
                "semantic_tags",
            },
            entity_id=data.get("id"),
        )
        return cls(
            chamber_id=data["id"],
            region_id=data["region_id"],
            center=tuple(data["center"]),
            size=tuple(data["size"]),
            shape=CavernShape(data.get("shape", "ellipsoid")),
            orientation_rpy=tuple(data.get("orientation_rpy", (0.0, 0.0, 0.0))),
            substrate_id=data.get("substrate_id"),
            semantic_tags=frozenset(data.get("semantic_tags", [])),
        )


@dataclass(frozen=True)
class PassageCrossSectionSpec:
    """Width and height at one normalized station along a passage path."""

    station: float
    width: float
    height: float

    def __post_init__(self) -> None:
        station = _finite(self.station, "station")
        width = _finite(self.width, "width")
        height = _finite(self.height, "height")
        if not 0.0 <= station <= 1.0:
            raise GeometryValidationError(
                "invalid_cross_section", "station must be inside [0, 1]"
            )
        if width <= 0.0 or height <= 0.0:
            raise GeometryValidationError(
                "invalid_cross_section", "width and height must be positive"
            )
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def to_dict(self) -> dict[str, float]:
        return {"station": self.station, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageCrossSectionSpec":
        _reject_unknown_fields(data, {"station", "width", "height"})
        return cls(data["station"], data["width"], data["height"])


@dataclass(frozen=True)
class PassageJunctionSpec:
    """A graph node at a chamber, opening, bound space, or free 3D position."""

    junction_id: str
    position: Point3
    chamber_id: str | None = None
    opening_id: str | None = None
    space_id: str | None = None
    level_id: str | None = None
    open_boundary: bool = False
    semantic_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        junction_id = _identifier(self.junction_id, "junction_id")
        object.__setattr__(self, "junction_id", junction_id)
        object.__setattr__(
            self, "position", _point(self.position, "position", entity_id=junction_id)
        )
        for attribute in ("chamber_id", "opening_id", "space_id", "level_id"):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(self, attribute, _identifier(value, attribute))
        if (self.space_id is None) != (self.level_id is None):
            raise GeometryValidationError(
                "incomplete_junction_binding",
                "space_id and level_id must be provided together",
                entity_id=junction_id,
            )
        object.__setattr__(
            self,
            "open_boundary",
            _strict_bool(self.open_boundary, "open_boundary", entity_id=junction_id),
        )
        object.__setattr__(
            self,
            "semantic_tags",
            frozenset(
                str(tag).strip() for tag in self.semantic_tags if str(tag).strip()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.junction_id,
            "position": list(self.position),
            "chamber_id": self.chamber_id,
            "opening_id": self.opening_id,
            "space_id": self.space_id,
            "level_id": self.level_id,
            "open_boundary": self.open_boundary,
            "semantic_tags": sorted(self.semantic_tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageJunctionSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "position",
                "chamber_id",
                "opening_id",
                "space_id",
                "level_id",
                "open_boundary",
                "semantic_tags",
            },
            entity_id=data.get("id"),
        )
        return cls(
            junction_id=data["id"],
            position=tuple(data["position"]),
            chamber_id=data.get("chamber_id"),
            opening_id=data.get("opening_id"),
            space_id=data.get("space_id"),
            level_id=data.get("level_id"),
            open_boundary=data.get("open_boundary", False),
            semantic_tags=frozenset(data.get("semantic_tags", [])),
        )


@dataclass(frozen=True)
class PassageSegmentSpec:
    """A variable-profile semantic edge between two passage junctions."""

    segment_id: str
    start_junction_id: str
    end_junction_id: str
    path: tuple[Point3, ...]
    cross_sections: tuple[PassageCrossSectionSpec, ...]
    profile: PassageProfile = PassageProfile.ELLIPSE
    floor_mode: PassageFloorMode = PassageFloorMode.NATURAL
    capabilities: frozenset[str] = frozenset({"walk"})
    roughness_seed: int | None = None

    def __post_init__(self) -> None:
        segment_id = _identifier(self.segment_id, "segment_id")
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(
            self,
            "start_junction_id",
            _identifier(self.start_junction_id, "start_junction_id"),
        )
        object.__setattr__(
            self,
            "end_junction_id",
            _identifier(self.end_junction_id, "end_junction_id"),
        )
        if self.start_junction_id == self.end_junction_id:
            raise GeometryValidationError(
                "invalid_passage_segment",
                "start and end junctions must differ",
                entity_id=segment_id,
            )
        path = tuple(
            _point(item, f"path[{index}]", entity_id=segment_id)
            for index, item in enumerate(self.path)
        )
        if len(path) < 2:
            raise GeometryValidationError(
                "invalid_passage_path",
                "path requires at least two points",
                entity_id=segment_id,
            )
        for index, (start, end) in enumerate(zip(path, path[1:])):
            if math.dist(start, end) <= GEOMETRY_TOLERANCE:
                raise GeometryValidationError(
                    "invalid_passage_path",
                    f"path span {index} has zero length",
                    entity_id=segment_id,
                )
        sections = tuple(self.cross_sections)
        if len(sections) < 2:
            raise GeometryValidationError(
                "invalid_cross_section",
                "at least two cross-sections are required",
                entity_id=segment_id,
            )
        stations = [section.station for section in sections]
        if any(second <= first for first, second in zip(stations, stations[1:])):
            raise GeometryValidationError(
                "invalid_cross_section",
                "cross-section stations must be strictly increasing",
                entity_id=segment_id,
            )
        if (
            abs(stations[0]) > _REFERENCE_TOLERANCE
            or abs(stations[-1] - 1.0) > _REFERENCE_TOLERANCE
        ):
            raise GeometryValidationError(
                "invalid_cross_section",
                "cross-sections must include stations 0 and 1",
                entity_id=segment_id,
            )
        capabilities = frozenset(
            str(item).strip() for item in self.capabilities if str(item).strip()
        )
        allowed_capabilities = {"walk", "crawl", "climb", "swim", "fly"}
        if not capabilities or not capabilities <= allowed_capabilities:
            raise GeometryValidationError(
                "invalid_capabilities",
                "capabilities must use walk, crawl, climb, swim, or fly",
                entity_id=segment_id,
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "cross_sections", sections)
        object.__setattr__(self, "profile", PassageProfile(self.profile))
        object.__setattr__(self, "floor_mode", PassageFloorMode(self.floor_mode))
        object.__setattr__(self, "capabilities", capabilities)
        if self.roughness_seed is not None:
            object.__setattr__(
                self,
                "roughness_seed",
                _strict_int(
                    self.roughness_seed, "roughness_seed", entity_id=segment_id
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.segment_id,
            "start_junction_id": self.start_junction_id,
            "end_junction_id": self.end_junction_id,
            "path": [list(point) for point in self.path],
            "cross_sections": [section.to_dict() for section in self.cross_sections],
            "profile": self.profile.value,
            "floor_mode": self.floor_mode.value,
            "capabilities": sorted(self.capabilities),
            "roughness_seed": self.roughness_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageSegmentSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "start_junction_id",
                "end_junction_id",
                "path",
                "cross_sections",
                "profile",
                "floor_mode",
                "capabilities",
                "roughness_seed",
            },
            entity_id=data.get("id"),
        )
        return cls(
            segment_id=data["id"],
            start_junction_id=data["start_junction_id"],
            end_junction_id=data["end_junction_id"],
            path=tuple(tuple(point) for point in data["path"]),
            cross_sections=tuple(
                PassageCrossSectionSpec.from_dict(item)
                for item in data["cross_sections"]
            ),
            profile=PassageProfile(data.get("profile", "ellipse")),
            floor_mode=PassageFloorMode(data.get("floor_mode", "natural")),
            capabilities=frozenset(data.get("capabilities", ["walk"])),
            roughness_seed=data.get("roughness_seed"),
        )


@dataclass(frozen=True)
class PassageNetworkSpec:
    """A topology-preserving graph of passage junctions and segments."""

    network_id: str
    region_id: str
    junctions: tuple[PassageJunctionSpec, ...]
    segments: tuple[PassageSegmentSpec, ...]

    def __post_init__(self) -> None:
        network_id = _identifier(self.network_id, "network_id")
        object.__setattr__(self, "network_id", network_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        junctions = tuple(sorted(self.junctions, key=lambda item: item.junction_id))
        segments = tuple(sorted(self.segments, key=lambda item: item.segment_id))
        _unique_ids(junctions, "junction_id", "junction")
        _unique_ids(segments, "segment_id", "passage_segment")
        junction_by_id = {item.junction_id: item for item in junctions}
        for segment in segments:
            if segment.start_junction_id not in junction_by_id:
                raise GeometryValidationError(
                    "unknown_passage_junction",
                    f"unknown start junction '{segment.start_junction_id}'",
                    entity_id=segment.segment_id,
                )
            if segment.end_junction_id not in junction_by_id:
                raise GeometryValidationError(
                    "unknown_passage_junction",
                    f"unknown end junction '{segment.end_junction_id}'",
                    entity_id=segment.segment_id,
                )
            start = junction_by_id[segment.start_junction_id].position
            end = junction_by_id[segment.end_junction_id].position
            if math.dist(segment.path[0], start) > _REFERENCE_TOLERANCE:
                raise GeometryValidationError(
                    "passage_endpoint_mismatch",
                    "path start does not match its start junction position",
                    entity_id=segment.segment_id,
                )
            if math.dist(segment.path[-1], end) > _REFERENCE_TOLERANCE:
                raise GeometryValidationError(
                    "passage_endpoint_mismatch",
                    "path end does not match its end junction position",
                    entity_id=segment.segment_id,
                )
        object.__setattr__(self, "junctions", junctions)
        object.__setattr__(self, "segments", segments)

    def degree(self, junction_id: str) -> int:
        if junction_id not in {item.junction_id for item in self.junctions}:
            raise KeyError(junction_id)
        return sum(
            junction_id in {segment.start_junction_id, segment.end_junction_id}
            for segment in self.segments
        )

    def reachable(self, start_junction_id: str) -> frozenset[str]:
        known = {item.junction_id for item in self.junctions}
        if start_junction_id not in known:
            raise KeyError(start_junction_id)
        adjacency: dict[str, set[str]] = {identifier: set() for identifier in known}
        for segment in self.segments:
            adjacency[segment.start_junction_id].add(segment.end_junction_id)
            adjacency[segment.end_junction_id].add(segment.start_junction_id)
        visited: set[str] = set()
        pending = [start_junction_id]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(sorted(adjacency[current] - visited, reverse=True))
        return frozenset(visited)

    def to_connector_spec(
        self, segment_id: str, *, connector_id: str | None = None
    ) -> ConnectorSpec:
        """Adapt one bound passage edge to SceneSmith's shared topology contract."""

        segment_by_id = {item.segment_id: item for item in self.segments}
        if segment_id not in segment_by_id:
            raise KeyError(segment_id)
        segment = segment_by_id[segment_id]
        junction_by_id = {item.junction_id: item for item in self.junctions}
        start = junction_by_id[segment.start_junction_id]
        end = junction_by_id[segment.end_junction_id]
        if any(
            value is None
            for value in (start.space_id, start.level_id, end.space_id, end.level_id)
        ):
            raise GeometryValidationError(
                "unbound_passage_connector",
                "connector endpoints require junction space_id and level_id bindings",
                entity_id=segment.segment_id,
            )
        return ConnectorSpec(
            connector_id=connector_id or segment.segment_id,
            connector_type=ConnectorType.NATURAL_PASSAGE,
            start=ConnectorEndpoint(
                start.space_id, start.level_id, start.position  # type: ignore[arg-type]
            ),
            end=ConnectorEndpoint(
                end.space_id, end.level_id, end.position  # type: ignore[arg-type]
            ),
            width=min(item.width for item in segment.cross_sections),
            clearance_height=min(item.height for item in segment.cross_sections),
            required_capabilities=segment.capabilities,
            parameters={
                "geometry_embedded": True,
                "waypoints": [list(point) for point in segment.path[1:-1]],
            },
        )

    @property
    def dead_ends(self) -> frozenset[str]:
        return frozenset(
            junction.junction_id
            for junction in self.junctions
            if self.degree(junction.junction_id) == 1
        )

    @property
    def cycle_rank(self) -> int:
        if not self.junctions:
            return 0
        unseen = {item.junction_id for item in self.junctions}
        components = 0
        while unseen:
            start = min(unseen)
            reached = set(self.reachable(start))
            unseen -= reached
            components += 1
        return max(0, len(self.segments) - len(self.junctions) + components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.network_id,
            "region_id": self.region_id,
            "junctions": [item.to_dict() for item in self.junctions],
            "segments": [item.to_dict() for item in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageNetworkSpec":
        _reject_unknown_fields(
            data,
            {"id", "region_id", "junctions", "segments"},
            entity_id=data.get("id"),
        )
        return cls(
            network_id=data["id"],
            region_id=data["region_id"],
            junctions=tuple(
                PassageJunctionSpec.from_dict(item)
                for item in data.get("junctions", [])
            ),
            segments=tuple(
                PassageSegmentSpec.from_dict(item) for item in data.get("segments", [])
            ),
        )


@dataclass(frozen=True)
class EnvironmentOpeningSpec:
    """A physical aperture connecting a chamber to sky or exterior."""

    opening_id: str
    region_id: str
    source_chamber_id: str
    target: OpeningTarget
    center: Point3
    normal: Point3
    size: Point2
    depth: float
    shape: OpeningShape = OpeningShape.ELLIPSE
    passable: bool = False
    visible: bool = True
    weather_exposed: bool = False

    def __post_init__(self) -> None:
        opening_id = _identifier(self.opening_id, "opening_id")
        object.__setattr__(self, "opening_id", opening_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self,
            "source_chamber_id",
            _identifier(self.source_chamber_id, "source_chamber_id"),
        )
        object.__setattr__(self, "target", OpeningTarget(self.target))
        object.__setattr__(
            self, "center", _point(self.center, "center", entity_id=opening_id)
        )
        normal = _point(self.normal, "normal", entity_id=opening_id)
        normal_length = math.sqrt(sum(component * component for component in normal))
        if normal_length <= GEOMETRY_TOLERANCE:
            raise GeometryValidationError(
                "invalid_opening_normal",
                "normal must have non-zero length",
                entity_id=opening_id,
            )
        object.__setattr__(
            self, "normal", tuple(component / normal_length for component in normal)
        )
        if len(self.size) != 2:
            raise GeometryValidationError(
                "invalid_opening_size",
                "size must contain width and height",
                entity_id=opening_id,
            )
        size = tuple(
            _finite(value, f"size[{axis}]", entity_id=opening_id)
            for axis, value in enumerate(self.size)
        )
        if any(value <= 0.0 for value in size):
            raise GeometryValidationError(
                "invalid_opening_size",
                "opening size must be positive",
                entity_id=opening_id,
            )
        object.__setattr__(self, "size", size)
        depth = _finite(self.depth, "depth", entity_id=opening_id)
        if depth <= 0.0:
            raise GeometryValidationError(
                "invalid_opening_depth",
                "opening depth must be positive",
                entity_id=opening_id,
            )
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "shape", OpeningShape(self.shape))
        object.__setattr__(
            self,
            "passable",
            _strict_bool(self.passable, "passable", entity_id=opening_id),
        )
        object.__setattr__(
            self,
            "visible",
            _strict_bool(self.visible, "visible", entity_id=opening_id),
        )
        object.__setattr__(
            self,
            "weather_exposed",
            _strict_bool(self.weather_exposed, "weather_exposed", entity_id=opening_id),
        )

    @property
    def sky_exposed(self) -> bool:
        return self.target == OpeningTarget.SKY

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.opening_id,
            "region_id": self.region_id,
            "source_chamber_id": self.source_chamber_id,
            "target": self.target.value,
            "center": list(self.center),
            "normal": list(self.normal),
            "size": list(self.size),
            "depth": self.depth,
            "shape": self.shape.value,
            "passable": self.passable,
            "visible": self.visible,
            "weather_exposed": self.weather_exposed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentOpeningSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "source_chamber_id",
                "target",
                "center",
                "normal",
                "size",
                "depth",
                "shape",
                "passable",
                "visible",
                "weather_exposed",
            },
            entity_id=data.get("id"),
        )
        return cls(
            opening_id=data["id"],
            region_id=data["region_id"],
            source_chamber_id=data["source_chamber_id"],
            target=OpeningTarget(data["target"]),
            center=tuple(data["center"]),
            normal=tuple(data["normal"]),
            size=tuple(data["size"]),
            depth=data["depth"],
            shape=OpeningShape(data.get("shape", "ellipse")),
            passable=data.get("passable", False),
            visible=data.get("visible", True),
            weather_exposed=data.get("weather_exposed", False),
        )


@dataclass(frozen=True)
class DetailFieldSpec:
    """A deterministic repeated geological-detail recipe."""

    field_id: str
    region_id: str
    target_chamber_id: str
    formation_type: FormationType
    surface_role: DetailSurfaceRole
    count: int
    min_size: Point3
    max_size: Point3
    seed: int
    protect_passage_network_ids: tuple[str, ...] = ()
    route_clearance: float = 1.0
    collision_policy: DetailCollisionPolicy = DetailCollisionPolicy.COARSE

    def __post_init__(self) -> None:
        field_id = _identifier(self.field_id, "field_id")
        object.__setattr__(self, "field_id", field_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self,
            "target_chamber_id",
            _identifier(self.target_chamber_id, "target_chamber_id"),
        )
        object.__setattr__(self, "formation_type", FormationType(self.formation_type))
        object.__setattr__(self, "surface_role", DetailSurfaceRole(self.surface_role))
        if self.surface_role not in FORMATION_SURFACE_ROLES[self.formation_type]:
            allowed = ", ".join(
                sorted(
                    role.value for role in FORMATION_SURFACE_ROLES[self.formation_type]
                )
            )
            raise GeometryValidationError(
                "unsupported_formation_surface",
                f"formation '{self.formation_type.value}' supports surface roles: "
                + allowed,
                entity_id=field_id,
            )
        count = _strict_int(self.count, "count", entity_id=field_id)
        if count <= 0:
            raise GeometryValidationError(
                "invalid_detail_count", "count must be positive", entity_id=field_id
            )
        object.__setattr__(self, "count", count)
        minimum = _positive_point(self.min_size, "min_size", entity_id=field_id)
        maximum = _positive_point(self.max_size, "max_size", entity_id=field_id)
        if any(minimum[axis] > maximum[axis] for axis in range(3)):
            raise GeometryValidationError(
                "invalid_detail_size",
                "min_size must not exceed max_size",
                entity_id=field_id,
            )
        object.__setattr__(self, "min_size", minimum)
        object.__setattr__(self, "max_size", maximum)
        object.__setattr__(
            self, "seed", _strict_int(self.seed, "seed", entity_id=field_id)
        )
        protected = tuple(
            sorted(
                _identifier(identifier, "protect_passage_network_id")
                for identifier in self.protect_passage_network_ids
            )
        )
        if len(protected) != len(set(protected)):
            raise GeometryValidationError(
                "duplicate_protected_network",
                "protected passage network IDs must be unique",
                entity_id=field_id,
            )
        object.__setattr__(self, "protect_passage_network_ids", protected)
        clearance = _finite(self.route_clearance, "route_clearance", entity_id=field_id)
        if clearance < 0.0:
            raise GeometryValidationError(
                "invalid_route_clearance",
                "route_clearance must not be negative",
                entity_id=field_id,
            )
        object.__setattr__(self, "route_clearance", clearance)
        object.__setattr__(
            self, "collision_policy", DetailCollisionPolicy(self.collision_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.field_id,
            "region_id": self.region_id,
            "target_chamber_id": self.target_chamber_id,
            "formation_type": self.formation_type.value,
            "surface_role": self.surface_role.value,
            "count": self.count,
            "min_size": list(self.min_size),
            "max_size": list(self.max_size),
            "seed": self.seed,
            "protect_passage_network_ids": list(self.protect_passage_network_ids),
            "route_clearance": self.route_clearance,
            "collision_policy": self.collision_policy.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetailFieldSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "target_chamber_id",
                "formation_type",
                "surface_role",
                "count",
                "min_size",
                "max_size",
                "seed",
                "protect_passage_network_ids",
                "route_clearance",
                "collision_policy",
            },
            entity_id=data.get("id"),
        )
        return cls(
            field_id=data["id"],
            region_id=data["region_id"],
            target_chamber_id=data["target_chamber_id"],
            formation_type=FormationType(data["formation_type"]),
            surface_role=DetailSurfaceRole(data["surface_role"]),
            count=data["count"],
            min_size=tuple(data["min_size"]),
            max_size=tuple(data["max_size"]),
            seed=data["seed"],
            protect_passage_network_ids=tuple(
                data.get("protect_passage_network_ids", [])
            ),
            route_clearance=data.get("route_clearance", 1.0),
            collision_policy=DetailCollisionPolicy(
                data.get("collision_policy", "coarse")
            ),
        )


@dataclass(frozen=True)
class HeroFeatureSpec:
    """One intentionally placed geological landmark."""

    feature_id: str
    region_id: str
    target_chamber_id: str
    feature_type: HeroFeatureType
    anchor: Point3
    size: Point3
    collision_policy: DetailCollisionPolicy = DetailCollisionPolicy.FULL
    semantic_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        feature_id = _identifier(self.feature_id, "feature_id")
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self,
            "target_chamber_id",
            _identifier(self.target_chamber_id, "target_chamber_id"),
        )
        object.__setattr__(self, "feature_type", HeroFeatureType(self.feature_type))
        object.__setattr__(
            self, "anchor", _point(self.anchor, "anchor", entity_id=feature_id)
        )
        object.__setattr__(
            self, "size", _positive_point(self.size, "size", entity_id=feature_id)
        )
        object.__setattr__(
            self, "collision_policy", DetailCollisionPolicy(self.collision_policy)
        )
        object.__setattr__(
            self,
            "semantic_tags",
            frozenset(
                str(tag).strip() for tag in self.semantic_tags if str(tag).strip()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.feature_id,
            "region_id": self.region_id,
            "target_chamber_id": self.target_chamber_id,
            "feature_type": self.feature_type.value,
            "anchor": list(self.anchor),
            "size": list(self.size),
            "collision_policy": self.collision_policy.value,
            "semantic_tags": sorted(self.semantic_tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeroFeatureSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "target_chamber_id",
                "feature_type",
                "anchor",
                "size",
                "collision_policy",
                "semantic_tags",
            },
            entity_id=data.get("id"),
        )
        return cls(
            feature_id=data["id"],
            region_id=data["region_id"],
            target_chamber_id=data["target_chamber_id"],
            feature_type=HeroFeatureType(data["feature_type"]),
            anchor=tuple(data["anchor"]),
            size=tuple(data["size"]),
            collision_policy=DetailCollisionPolicy(
                data.get("collision_policy", "full")
            ),
            semantic_tags=frozenset(data.get("semantic_tags", [])),
        )


@dataclass(frozen=True)
class SemanticEnvironmentSpec:
    """Canonical semantic graph for one or more environment regions."""

    regions: tuple[EnvironmentRegionSpec, ...]
    chambers: tuple[CavernChamberSpec, ...] = ()
    passage_networks: tuple[PassageNetworkSpec, ...] = ()
    openings: tuple[EnvironmentOpeningSpec, ...] = ()
    detail_fields: tuple[DetailFieldSpec, ...] = ()
    hero_features: tuple[HeroFeatureSpec, ...] = ()
    schema_version: int = SEMANTIC_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _strict_int(self.schema_version, "schema_version")
        if schema_version != SEMANTIC_ENVIRONMENT_SCHEMA_VERSION:
            raise GeometryValidationError(
                "unsupported_environment_schema",
                f"schema_version must be {SEMANTIC_ENVIRONMENT_SCHEMA_VERSION}",
            )
        regions = tuple(sorted(self.regions, key=lambda item: item.region_id))
        chambers = tuple(sorted(self.chambers, key=lambda item: item.chamber_id))
        networks = tuple(
            sorted(self.passage_networks, key=lambda item: item.network_id)
        )
        openings = tuple(sorted(self.openings, key=lambda item: item.opening_id))
        detail_fields = tuple(
            sorted(self.detail_fields, key=lambda item: item.field_id)
        )
        hero_features = tuple(
            sorted(self.hero_features, key=lambda item: item.feature_id)
        )
        if not regions:
            raise GeometryValidationError(
                "empty_environment", "an environment requires at least one region"
            )
        _unique_ids(regions, "region_id", "environment_region")
        _unique_ids(chambers, "chamber_id", "cavern_chamber")
        _unique_ids(networks, "network_id", "passage_network")
        _unique_ids(openings, "opening_id", "environment_opening")
        _unique_ids(detail_fields, "field_id", "detail_field")
        _unique_ids(hero_features, "feature_id", "hero_feature")
        identifiers: list[tuple[str, str]] = [
            *((item.region_id, "environment_region") for item in regions),
            *((item.chamber_id, "cavern_chamber") for item in chambers),
            *((item.network_id, "passage_network") for item in networks),
            *((item.opening_id, "environment_opening") for item in openings),
            *((item.field_id, "detail_field") for item in detail_fields),
            *((item.feature_id, "hero_feature") for item in hero_features),
            *(
                (item.junction_id, "passage_junction")
                for network in networks
                for item in network.junctions
            ),
            *(
                (item.segment_id, "passage_segment")
                for network in networks
                for item in network.segments
            ),
            *(
                (f"{item.field_id}_{index:04d}", "detail_instance")
                for item in detail_fields
                for index in range(item.count)
            ),
        ]
        try:
            validate_global_identifiers(identifiers)
        except GeometryValidationError as exc:
            if exc.code != "duplicate_scene_id":
                raise
            raise GeometryValidationError(
                "duplicate_semantic_id",
                str(exc).split(": ", 1)[-1],
            ) from exc
        segment_count = sum(len(network.segments) for network in networks)
        if segment_count > MAX_PASSAGE_SEGMENTS_PER_SCENE:
            raise GeometryValidationError(
                "semantic_budget_exceeded",
                f"scene has {segment_count} passage segments; budget is "
                f"{MAX_PASSAGE_SEGMENTS_PER_SCENE}",
            )
        for network in networks:
            for segment in network.segments:
                if len(segment.path) > MAX_PATH_POINTS_PER_SEGMENT:
                    raise GeometryValidationError(
                        "semantic_budget_exceeded",
                        f"passage path has {len(segment.path)} points; budget is "
                        f"{MAX_PATH_POINTS_PER_SEGMENT}",
                        entity_id=segment.segment_id,
                    )
        for field_spec in detail_fields:
            if field_spec.count > MAX_DETAIL_INSTANCES_PER_FIELD:
                raise GeometryValidationError(
                    "semantic_budget_exceeded",
                    f"detail count {field_spec.count} exceeds per-field budget "
                    f"{MAX_DETAIL_INSTANCES_PER_FIELD}",
                    entity_id=field_spec.field_id,
                )
        total_details = sum(field_spec.count for field_spec in detail_fields)
        if total_details > MAX_DETAIL_INSTANCES_PER_SCENE:
            raise GeometryValidationError(
                "semantic_budget_exceeded",
                f"scene requests {total_details} detail instances; budget is "
                f"{MAX_DETAIL_INSTANCES_PER_SCENE}",
            )
        region_by_id = {item.region_id: item for item in regions}
        chamber_by_id = {item.chamber_id: item for item in chambers}
        network_by_id = {item.network_id: item for item in networks}
        opening_by_id = {item.opening_id: item for item in openings}
        for chamber in chambers:
            if chamber.shape not in {
                CavernShape.ELLIPSOID,
                CavernShape.SUPERELLIPSOID,
            }:
                raise GeometryValidationError(
                    "unsupported_chamber_shape",
                    f"chamber shape '{chamber.shape.value}' is not supported by the "
                    "semantic compiler",
                    entity_id=chamber.chamber_id,
                )
            region = region_by_id.get(chamber.region_id)
            if region is None:
                raise GeometryValidationError(
                    "unknown_environment_region",
                    f"unknown region '{chamber.region_id}'",
                    entity_id=chamber.chamber_id,
                )
            chamber_bounds = chamber.bounds
            if not region.bounds.contains_box(
                chamber_bounds.minimum, chamber_bounds.maximum
            ):
                raise GeometryValidationError(
                    "environment_bounds_exceeded",
                    "chamber exceeds its region's conservative bounds",
                    entity_id=chamber.chamber_id,
                )
        for network in networks:
            region = region_by_id.get(network.region_id)
            if region is None:
                raise GeometryValidationError(
                    "unknown_environment_region",
                    f"unknown region '{network.region_id}'",
                    entity_id=network.network_id,
                )
            for junction in network.junctions:
                if not region.bounds.contains(junction.position):
                    raise GeometryValidationError(
                        "environment_bounds_exceeded",
                        "junction is outside its region bounds",
                        entity_id=junction.junction_id,
                    )
                if junction.chamber_id is not None:
                    chamber = chamber_by_id.get(junction.chamber_id)
                    if chamber is None:
                        raise GeometryValidationError(
                            "unknown_cavern_chamber",
                            f"unknown chamber '{junction.chamber_id}'",
                            entity_id=junction.junction_id,
                        )
                    if chamber.region_id != network.region_id:
                        raise GeometryValidationError(
                            "cross_region_junction",
                            "junction and referenced chamber must share a region",
                            entity_id=junction.junction_id,
                        )
                    if chamber.shape in {
                        CavernShape.ELLIPSOID,
                        CavernShape.SUPERELLIPSOID,
                    } and not chamber.contains(junction.position):
                        raise GeometryValidationError(
                            "disjoint_chamber_junction",
                            "junction position does not overlap its referenced chamber",
                            entity_id=junction.junction_id,
                        )
                if junction.opening_id is not None:
                    opening = opening_by_id.get(junction.opening_id)
                    if opening is None:
                        raise GeometryValidationError(
                            "unknown_environment_opening",
                            f"unknown opening '{junction.opening_id}'",
                            entity_id=junction.junction_id,
                        )
                    if opening.region_id != network.region_id:
                        raise GeometryValidationError(
                            "cross_region_junction",
                            "junction and referenced opening must share a region",
                            entity_id=junction.junction_id,
                        )
            for segment in network.segments:
                if segment.floor_mode == PassageFloorMode.STEPS:
                    raise GeometryValidationError(
                        "unsupported_passage_floor_mode",
                        "stepped passage floors are not supported by the semantic compiler",
                        entity_id=segment.segment_id,
                    )
                if any(not region.bounds.contains(point) for point in segment.path):
                    raise GeometryValidationError(
                        "environment_bounds_exceeded",
                        "passage path is outside its region bounds",
                        entity_id=segment.segment_id,
                    )

        for opening in openings:
            chamber = chamber_by_id.get(opening.source_chamber_id)
            if opening.region_id not in region_by_id:
                raise GeometryValidationError(
                    "unknown_environment_region",
                    f"unknown region '{opening.region_id}'",
                    entity_id=opening.opening_id,
                )
            if chamber is None:
                raise GeometryValidationError(
                    "unknown_cavern_chamber",
                    f"unknown chamber '{opening.source_chamber_id}'",
                    entity_id=opening.opening_id,
                )
            if chamber.region_id != opening.region_id:
                raise GeometryValidationError(
                    "cross_region_opening",
                    "opening and source chamber must share a region",
                    entity_id=opening.opening_id,
                )
            if chamber.shape in {
                CavernShape.ELLIPSOID,
                CavernShape.SUPERELLIPSOID,
            } and not chamber.contains(opening.center, tolerance=0.1):
                raise GeometryValidationError(
                    "disjoint_chamber_opening",
                    "opening center does not overlap its source chamber",
                    entity_id=opening.opening_id,
                )
        for field_spec in detail_fields:
            chamber = chamber_by_id.get(field_spec.target_chamber_id)
            if chamber is None or chamber.region_id != field_spec.region_id:
                raise GeometryValidationError(
                    "unknown_detail_target",
                    "detail target chamber is missing or belongs to another region",
                    entity_id=field_spec.field_id,
                )
            for network_id in field_spec.protect_passage_network_ids:
                network = network_by_id.get(network_id)
                if network is None or network.region_id != field_spec.region_id:
                    raise GeometryValidationError(
                        "unknown_protected_network",
                        f"protected network '{network_id}' is missing or cross-region",
                        entity_id=field_spec.field_id,
                    )
        for feature in hero_features:
            chamber = chamber_by_id.get(feature.target_chamber_id)
            if chamber is None or chamber.region_id != feature.region_id:
                raise GeometryValidationError(
                    "unknown_hero_target",
                    "hero target chamber is missing or belongs to another region",
                    entity_id=feature.feature_id,
                )
            half_size = tuple(value / 2.0 for value in feature.size)
            envelope_points = tuple(
                tuple(
                    feature.anchor[axis] + signs[axis] * half_size[axis]
                    for axis in range(3)
                )
                for signs in (
                    (-1, -1, 0),
                    (-1, 1, 0),
                    (1, -1, 0),
                    (1, 1, 0),
                    (-1, -1, 1),
                    (-1, 1, 1),
                    (1, -1, 1),
                    (1, 1, 1),
                )
            )
            if not all(chamber.contains(point) for point in envelope_points):
                raise GeometryValidationError(
                    "hero_outside_chamber",
                    "the complete hero envelope must lie inside its target chamber",
                    entity_id=feature.feature_id,
                )

        for region in regions:
            if region.kind == EnvironmentKind.SUBTERRANEAN and not (
                any(item.region_id == region.region_id for item in chambers)
                or any(item.region_id == region.region_id for item in networks)
            ):
                raise GeometryValidationError(
                    "empty_subterranean_region",
                    "a subterranean region requires a chamber or passage network",
                    entity_id=region.region_id,
                )
        object.__setattr__(
            self,
            "schema_version",
            schema_version,
        )
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "chambers", chambers)
        object.__setattr__(self, "passage_networks", networks)
        object.__setattr__(self, "openings", openings)
        object.__setattr__(self, "detail_fields", detail_fields)
        object.__setattr__(self, "hero_features", hero_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "regions": [item.to_dict() for item in self.regions],
            "chambers": [item.to_dict() for item in self.chambers],
            "passage_networks": [item.to_dict() for item in self.passage_networks],
            "openings": [item.to_dict() for item in self.openings],
            "detail_fields": [item.to_dict() for item in self.detail_fields],
            "hero_features": [item.to_dict() for item in self.hero_features],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticEnvironmentSpec":
        _reject_unknown_fields(
            data,
            {
                "schema_version",
                "regions",
                "chambers",
                "passage_networks",
                "openings",
                "detail_fields",
                "hero_features",
            },
        )
        return cls(
            schema_version=data.get(
                "schema_version", SEMANTIC_ENVIRONMENT_SCHEMA_VERSION
            ),
            regions=tuple(
                EnvironmentRegionSpec.from_dict(item)
                for item in data.get("regions", [])
            ),
            chambers=tuple(
                CavernChamberSpec.from_dict(item) for item in data.get("chambers", [])
            ),
            passage_networks=tuple(
                PassageNetworkSpec.from_dict(item)
                for item in data.get("passage_networks", [])
            ),
            openings=tuple(
                EnvironmentOpeningSpec.from_dict(item)
                for item in data.get("openings", [])
            ),
            detail_fields=tuple(
                DetailFieldSpec.from_dict(item)
                for item in data.get("detail_fields", [])
            ),
            hero_features=tuple(
                HeroFeatureSpec.from_dict(item)
                for item in data.get("hero_features", [])
            ),
        )

    def content_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate_layout_bindings(
        self, *, space_level_ids: Mapping[str, str], level_ids: Sequence[str]
    ) -> None:
        """Validate optional passage junction bindings against a HouseLayout."""

        known_levels = set(level_ids)
        for network in self.passage_networks:
            for junction in network.junctions:
                if junction.space_id is None:
                    continue
                assert junction.level_id is not None
                if junction.space_id not in space_level_ids:
                    raise GeometryValidationError(
                        "unknown_junction_space",
                        f"space '{junction.space_id}' is not defined",
                        entity_id=junction.junction_id,
                    )
                if junction.level_id not in known_levels:
                    raise GeometryValidationError(
                        "unknown_junction_level",
                        f"level '{junction.level_id}' is not defined",
                        entity_id=junction.junction_id,
                    )
                expected_level = space_level_ids[junction.space_id]
                if junction.level_id != expected_level:
                    raise GeometryValidationError(
                        "junction_level_mismatch",
                        f"space '{junction.space_id}' belongs to level "
                        f"'{expected_level}', not '{junction.level_id}'",
                        entity_id=junction.junction_id,
                    )
