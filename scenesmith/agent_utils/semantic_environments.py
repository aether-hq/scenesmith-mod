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
    Point3,
    Transform3D,
)

SEMANTIC_ENVIRONMENT_SCHEMA_VERSION = 1
_REFERENCE_TOLERANCE = 1e-6


def _identifier(value: Any, label: str) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise GeometryValidationError("missing_id", f"{label} must not be empty")
    return identifier


def _finite(value: Any, label: str, *, entity_id: str | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(
            "invalid_number", f"{label} must be numeric", entity_id=entity_id
        ) from exc
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
    transform: Transform3D = field(default_factory=Transform3D)
    material_context: Mapping[str, str] = field(default_factory=dict)
    detail_seed: int = 0
    chunk_policy: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(self, "kind", EnvironmentKind(self.kind))
        object.__setattr__(self, "material_context", dict(self.material_context))
        object.__setattr__(self, "chunk_policy", dict(self.chunk_policy))
        object.__setattr__(self, "detail_seed", int(self.detail_seed))

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
            transform=Transform3D.from_dict(data.get("transform")),
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
        object.__setattr__(self, "open_boundary", bool(self.open_boundary))
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
            open_boundary=bool(data.get("open_boundary", False)),
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
            object.__setattr__(self, "roughness_seed", int(self.roughness_seed))

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
class SemanticEnvironmentSpec:
    """Canonical semantic graph for one or more environment regions."""

    regions: tuple[EnvironmentRegionSpec, ...]
    chambers: tuple[CavernChamberSpec, ...] = ()
    passage_networks: tuple[PassageNetworkSpec, ...] = ()
    schema_version: int = SEMANTIC_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != SEMANTIC_ENVIRONMENT_SCHEMA_VERSION:
            raise GeometryValidationError(
                "unsupported_environment_schema",
                f"schema_version must be {SEMANTIC_ENVIRONMENT_SCHEMA_VERSION}",
            )
        regions = tuple(sorted(self.regions, key=lambda item: item.region_id))
        chambers = tuple(sorted(self.chambers, key=lambda item: item.chamber_id))
        networks = tuple(
            sorted(self.passage_networks, key=lambda item: item.network_id)
        )
        if not regions:
            raise GeometryValidationError(
                "empty_environment", "an environment requires at least one region"
            )
        _unique_ids(regions, "region_id", "environment_region")
        _unique_ids(chambers, "chamber_id", "cavern_chamber")
        _unique_ids(networks, "network_id", "passage_network")
        region_by_id = {item.region_id: item for item in regions}
        chamber_by_id = {item.chamber_id: item for item in chambers}
        for chamber in chambers:
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
            for segment in network.segments:
                if any(not region.bounds.contains(point) for point in segment.path):
                    raise GeometryValidationError(
                        "environment_bounds_exceeded",
                        "passage path is outside its region bounds",
                        entity_id=segment.segment_id,
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
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "chambers", chambers)
        object.__setattr__(self, "passage_networks", networks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "regions": [item.to_dict() for item in self.regions],
            "chambers": [item.to_dict() for item in self.chambers],
            "passage_networks": [item.to_dict() for item in self.passage_networks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticEnvironmentSpec":
        _reject_unknown_fields(
            data, {"schema_version", "regions", "chambers", "passage_networks"}
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
