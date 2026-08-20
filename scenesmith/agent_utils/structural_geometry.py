"""Versioned semantic primitives for general structural scene geometry.

This module deliberately depends only on the Python standard library.  The
semantic layout can therefore be validated before starting Blender, Drake, or
any generative model service.  Mesh compilation lives in separate modules.
"""

from __future__ import annotations

import math
import re

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 2
GEOMETRY_TOLERANCE = 1e-9

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
Loop2 = tuple[Point2, ...]


class GeometryValidationError(ValueError):
    """Base error for structural-geometry validation failures."""

    def __init__(
        self, code: str, message: str, *, entity_id: str | None = None
    ) -> None:
        self.code = code
        self.entity_id = entity_id
        prefix = f"[{code}]"
        if entity_id is not None:
            prefix += f" {entity_id}:"
        super().__init__(f"{prefix} {message}")


class UnknownLevelError(GeometryValidationError):
    """A space or connector references a level that does not exist."""

    def __init__(self, level_id: str, *, entity_id: str | None = None) -> None:
        super().__init__(
            "unknown_level",
            f"level '{level_id}' is not defined",
            entity_id=entity_id,
        )


class UnknownConnectorEndpointError(GeometryValidationError):
    """A connector references a space that does not exist."""

    def __init__(self, space_id: str, *, connector_id: str) -> None:
        super().__init__(
            "unknown_connector_endpoint",
            f"space '{space_id}' is not defined",
            entity_id=connector_id,
        )


class InvalidFootprintError(GeometryValidationError):
    """A footprint loop is degenerate, intersects itself, or has invalid holes."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("invalid_footprint", message, entity_id=entity_id)


class InvalidTransformError(GeometryValidationError):
    """A structural transform contains non-finite coordinates."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("invalid_transform", message, entity_id=entity_id)


class UnsafeConnectorError(GeometryValidationError):
    """A connector violates its declared geometric or ergonomic constraints."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("unsafe_connector", message, entity_id=entity_id)


class UnsupportedGeometryError(GeometryValidationError):
    """The semantic input is valid but its compiler is not implemented yet."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("unsupported_geometry", message, entity_id=entity_id)


def require_safe_identifier(value: str, label: str) -> str:
    """Normalize one globally usable file/model/semantic identifier."""

    if type(value) is not str:
        raise GeometryValidationError(
            "invalid_identifier", f"{label} must be a JSON string"
        )
    normalized = value.strip()
    if not normalized:
        raise GeometryValidationError("missing_id", f"{label} must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", normalized):
        raise GeometryValidationError(
            "invalid_identifier",
            f"{label} must use only letters, numbers, '.', '_', or '-' and must "
            "start with a letter or number",
            entity_id=normalized,
        )
    return normalized


_require_id = require_safe_identifier


def validate_global_identifiers(identifiers: Iterable[tuple[str, str]]) -> None:
    """Reject cross-category collisions in one authoritative scene namespace."""

    uses: dict[str, list[str]] = {}
    for identifier, category in identifiers:
        normalized = require_safe_identifier(identifier, f"{category}_id")
        uses.setdefault(normalized, []).append(category)
    collisions = {
        identifier: categories
        for identifier, categories in uses.items()
        if len(categories) > 1
    }
    if collisions:
        details = ", ".join(
            f"{identifier} ({'/'.join(categories)})"
            for identifier, categories in sorted(collisions.items())
        )
        raise GeometryValidationError(
            "duplicate_scene_id",
            "scene IDs must be globally unique; duplicates: " + details,
        )


def _finite(value: Any, label: str, *, entity_id: str | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidTransformError(
            f"{label} must be numeric; got {value!r}", entity_id=entity_id
        ) from exc
    if not math.isfinite(number):
        raise InvalidTransformError(
            f"{label} must be finite; got {value!r}", entity_id=entity_id
        )
    return number


def _point2(value: Sequence[Any], label: str) -> Point2:
    if len(value) != 2:
        raise InvalidFootprintError(f"{label} must have exactly 2 coordinates")
    return (_finite(value[0], f"{label}.x"), _finite(value[1], f"{label}.y"))


def _point3(
    value: Sequence[Any], label: str, *, entity_id: str | None = None
) -> Point3:
    if len(value) != 3:
        raise InvalidTransformError(
            f"{label} must have exactly 3 coordinates", entity_id=entity_id
        )
    return (
        _finite(value[0], f"{label}.x", entity_id=entity_id),
        _finite(value[1], f"{label}.y", entity_id=entity_id),
        _finite(value[2], f"{label}.z", entity_id=entity_id),
    )


def _signed_area(loop: Loop2) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(loop, loop[1:] + loop[:1])
    )


def _orientation(a: Point2, b: Point2, c: Point2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point: Point2, start: Point2, end: Point2) -> bool:
    if abs(_orientation(start, end, point)) > GEOMETRY_TOLERANCE:
        return False
    return (
        min(start[0], end[0]) - GEOMETRY_TOLERANCE
        <= point[0]
        <= max(start[0], end[0]) + GEOMETRY_TOLERANCE
        and min(start[1], end[1]) - GEOMETRY_TOLERANCE
        <= point[1]
        <= max(start[1], end[1]) + GEOMETRY_TOLERANCE
    )


def _segments_intersect(a: Point2, b: Point2, c: Point2, d: Point2) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if (
        (o1 > GEOMETRY_TOLERANCE and o2 < -GEOMETRY_TOLERANCE)
        or (o1 < -GEOMETRY_TOLERANCE and o2 > GEOMETRY_TOLERANCE)
    ) and (
        (o3 > GEOMETRY_TOLERANCE and o4 < -GEOMETRY_TOLERANCE)
        or (o3 < -GEOMETRY_TOLERANCE and o4 > GEOMETRY_TOLERANCE)
    ):
        return True

    return (
        (abs(o1) <= GEOMETRY_TOLERANCE and _point_on_segment(c, a, b))
        or (abs(o2) <= GEOMETRY_TOLERANCE and _point_on_segment(d, a, b))
        or (abs(o3) <= GEOMETRY_TOLERANCE and _point_on_segment(a, c, d))
        or (abs(o4) <= GEOMETRY_TOLERANCE and _point_on_segment(b, c, d))
    )


def _loop_edges(loop: Loop2) -> Iterable[tuple[Point2, Point2]]:
    return zip(loop, loop[1:] + loop[:1])


def _validate_simple_loop(loop: Loop2, label: str) -> None:
    if len(loop) < 3:
        raise InvalidFootprintError(f"{label} needs at least 3 vertices")

    for index, (start, end) in enumerate(_loop_edges(loop)):
        if math.dist(start, end) <= GEOMETRY_TOLERANCE:
            raise InvalidFootprintError(
                f"{label} edge {index} has zero length or repeated vertices"
            )

    edge_count = len(loop)
    edges = list(_loop_edges(loop))
    for first_index, (a, b) in enumerate(edges):
        for second_index in range(first_index + 1, edge_count):
            # Adjacent edges meet at a vertex by definition and are valid.
            if second_index in {
                first_index,
                (first_index + 1) % edge_count,
                (first_index - 1) % edge_count,
            }:
                continue
            c, d = edges[second_index]
            if _segments_intersect(a, b, c, d):
                raise InvalidFootprintError(
                    f"{label} self-intersects at edges {first_index} and "
                    f"{second_index}"
                )

    if abs(_signed_area(loop)) <= GEOMETRY_TOLERANCE:
        raise InvalidFootprintError(f"{label} has zero area")


def _loops_intersect(first: Loop2, second: Loop2) -> bool:
    return any(
        _segments_intersect(a, b, c, d)
        for a, b in _loop_edges(first)
        for c, d in _loop_edges(second)
    )


def _point_in_loop(point: Point2, loop: Loop2, *, include_boundary: bool) -> bool:
    for start, end in _loop_edges(loop):
        if _point_on_segment(point, start, end):
            return include_boundary

    inside = False
    x, y = point
    previous = loop[-1]
    for current in loop:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            x_crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_crossing > x:
                inside = not inside
        previous = current
    return inside


def _normalize_loop(
    points: Sequence[Sequence[Any]], *, counter_clockwise: bool, label: str
) -> Loop2:
    loop = tuple(
        _point2(point, f"{label}[{index}]") for index, point in enumerate(points)
    )
    # A repeated closing point is common in interchange formats. Normalize it
    # before strict repeated-edge validation.
    if len(loop) >= 2 and math.dist(loop[0], loop[-1]) <= GEOMETRY_TOLERANCE:
        loop = loop[:-1]
    _validate_simple_loop(loop, label)
    is_counter_clockwise = _signed_area(loop) > 0
    if is_counter_clockwise != counter_clockwise:
        loop = tuple(reversed(loop))
    return loop


@dataclass(frozen=True)
class Transform3D:
    """Rigid transform described by translation and roll/pitch/yaw radians."""

    translation: Point3 = (0.0, 0.0, 0.0)
    rotation_rpy: Point3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation", _point3(self.translation, "translation")
        )
        object.__setattr__(
            self, "rotation_rpy", _point3(self.rotation_rpy, "rotation_rpy")
        )

    @property
    def yaw(self) -> float:
        return self.rotation_rpy[2]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "translation": list(self.translation),
            "rotation_rpy": list(self.rotation_rpy),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Transform3D":
        if not data:
            return cls()
        return cls(
            translation=tuple(data.get("translation", (0.0, 0.0, 0.0))),
            rotation_rpy=tuple(data.get("rotation_rpy", (0.0, 0.0, 0.0))),
        )


@dataclass(frozen=True)
class LevelSpec:
    """A stable vertical datum used to group spaces and connector endpoints."""

    level_id: str
    elevation: float = 0.0
    nominal_height: float = 2.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "level_id", _require_id(self.level_id, "level_id"))
        object.__setattr__(
            self,
            "elevation",
            _finite(self.elevation, "elevation", entity_id=self.level_id),
        )
        height = _finite(self.nominal_height, "nominal_height", entity_id=self.level_id)
        if height <= 0:
            raise GeometryValidationError(
                "invalid_level_height",
                f"nominal_height must be positive; got {height}",
                entity_id=self.level_id,
            )
        object.__setattr__(self, "nominal_height", height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.level_id,
            "elevation": self.elevation,
            "nominal_height": self.nominal_height,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LevelSpec":
        return cls(
            level_id=data["id"],
            elevation=data.get("elevation", 0.0),
            nominal_height=data.get("nominal_height", 2.5),
        )


def default_ground_level() -> LevelSpec:
    """Return the compatibility level for v1 flat layouts."""

    return LevelSpec(level_id="ground", elevation=0.0, nominal_height=2.5)


@dataclass(frozen=True)
class Footprint2D:
    """A simple outer polygon with zero or more non-overlapping holes.

    The outer loop is normalized counter-clockwise and holes clockwise.  Loops
    may be concave but may not self-intersect, touch, overlap, or nest.
    """

    outer: Loop2
    holes: tuple[Loop2, ...] = ()

    def __post_init__(self) -> None:
        outer = _normalize_loop(self.outer, counter_clockwise=True, label="outer")
        holes = tuple(
            _normalize_loop(hole, counter_clockwise=False, label=f"hole[{index}]")
            for index, hole in enumerate(self.holes)
        )

        for index, hole in enumerate(holes):
            if _loops_intersect(outer, hole) or not _point_in_loop(
                hole[0], outer, include_boundary=False
            ):
                raise InvalidFootprintError(
                    f"hole[{index}] must be strictly inside the outer loop"
                )

        for first_index, first in enumerate(holes):
            for second_index in range(first_index + 1, len(holes)):
                second = holes[second_index]
                if (
                    _loops_intersect(first, second)
                    or _point_in_loop(first[0], second, include_boundary=True)
                    or _point_in_loop(second[0], first, include_boundary=True)
                ):
                    raise InvalidFootprintError(
                        f"hole[{first_index}] and hole[{second_index}] overlap, "
                        "touch, or nest"
                    )

        object.__setattr__(self, "outer", outer)
        object.__setattr__(self, "holes", holes)

    @classmethod
    def rectangle(cls, width: float, depth: float) -> "Footprint2D":
        width = _finite(width, "width")
        depth = _finite(depth, "depth")
        if width <= 0 or depth <= 0:
            raise InvalidFootprintError(
                f"rectangle dimensions must be positive; got {width} × {depth}"
            )
        return cls(outer=((0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)))

    @classmethod
    def circle(
        cls,
        radius: float,
        chord_tolerance: float,
        *,
        center: Point2 = (0.0, 0.0),
        max_segments: int = 4096,
    ) -> "Footprint2D":
        """Create a deterministic circular footprint within a chord-error bound."""

        radius = _finite(radius, "radius")
        tolerance = _finite(chord_tolerance, "chord_tolerance")
        center = _point2(center, "center")
        if radius <= 0 or tolerance <= 0:
            raise InvalidFootprintError("radius and chord_tolerance must be positive")
        if max_segments < 3:
            raise InvalidFootprintError("max_segments must be at least 3")
        if tolerance >= radius:
            segment_count = 3
        else:
            maximum_angle = 2.0 * math.acos(1.0 - tolerance / radius)
            segment_count = max(3, math.ceil(2.0 * math.pi / maximum_angle))
        if segment_count > max_segments:
            raise GeometryValidationError(
                "geometry_budget_exceeded",
                f"circle requires {segment_count} segments for tolerance "
                f"{tolerance:g}; budget is {max_segments}",
            )
        return cls(
            outer=tuple(
                (
                    center[0]
                    + radius * math.cos(2.0 * math.pi * index / segment_count),
                    center[1]
                    + radius * math.sin(2.0 * math.pi * index / segment_count),
                )
                for index in range(segment_count)
            )
        )

    @property
    def area(self) -> float:
        return abs(_signed_area(self.outer)) - sum(
            abs(_signed_area(hole)) for hole in self.holes
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            min(point[0] for point in self.outer),
            min(point[1] for point in self.outer),
            max(point[0] for point in self.outer),
            max(point[1] for point in self.outer),
        )

    def contains(self, point: Sequence[Any], *, include_boundary: bool = True) -> bool:
        candidate = _point2(point, "point")
        if not _point_in_loop(candidate, self.outer, include_boundary=include_boundary):
            return False
        return not any(
            _point_in_loop(candidate, hole, include_boundary=True)
            for hole in self.holes
        )

    def translated(self, offset: Sequence[Any]) -> "Footprint2D":
        """Return a copy translated by an XY offset."""

        dx, dy = _point2(offset, "offset")

        def move(loop: Loop2) -> Loop2:
            return tuple((x + dx, y + dy) for x, y in loop)

        return Footprint2D(
            outer=move(self.outer), holes=tuple(move(hole) for hole in self.holes)
        )

    def centered_on_bounds(self) -> "Footprint2D":
        """Return the footprint with its bounding-box center at the origin."""

        min_x, min_y, max_x, max_y = self.bounds
        return self.translated((-(min_x + max_x) / 2.0, -(min_y + max_y) / 2.0))

    def to_dict(self) -> dict[str, list[list[list[float]]] | list[list[float]]]:
        return {
            "outer": [list(point) for point in self.outer],
            "holes": [[list(point) for point in hole] for hole in self.holes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Footprint2D":
        return cls(
            outer=tuple(tuple(point) for point in data["outer"]),
            holes=tuple(
                tuple(tuple(point) for point in hole) for hole in data.get("holes", [])
            ),
        )


class ElevationProfileType(str, Enum):
    PLANAR = "planar"
    SLOPED = "sloped"
    STEPPED = "stepped"
    HEIGHTFIELD = "heightfield"
    MESH = "mesh"


@dataclass(frozen=True)
class ElevationProfile:
    """Semantic description of a floor or ceiling elevation surface."""

    profile_type: ElevationProfileType = ElevationProfileType.PLANAR
    base_elevation: float = 0.0
    gradient: Point2 = (0.0, 0.0)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        profile_type = ElevationProfileType(self.profile_type)
        base = _finite(self.base_elevation, "base_elevation")
        gradient = _point2(self.gradient, "gradient")
        if profile_type == ElevationProfileType.PLANAR and any(
            abs(value) > GEOMETRY_TOLERANCE for value in gradient
        ):
            raise GeometryValidationError(
                "invalid_elevation_profile",
                "planar profiles must have a zero gradient; use 'sloped'",
            )
        object.__setattr__(self, "profile_type", profile_type)
        object.__setattr__(self, "base_elevation", base)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "parameters", dict(self.parameters))

    def height_at(self, point: Sequence[Any]) -> float:
        """Evaluate planar/sloped profiles; other tiers require a compiler."""

        x, y = _point2(point, "point")
        if self.profile_type not in {
            ElevationProfileType.PLANAR,
            ElevationProfileType.SLOPED,
        }:
            raise GeometryValidationError(
                "unsupported_profile_query",
                f"height_at is not implemented for '{self.profile_type.value}'",
            )
        return self.base_elevation + self.gradient[0] * x + self.gradient[1] * y

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.profile_type.value,
            "base_elevation": self.base_elevation,
            "gradient": list(self.gradient),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ElevationProfile":
        if not data:
            return cls()
        return cls(
            profile_type=ElevationProfileType(data.get("type", "planar")),
            base_elevation=data.get("base_elevation", 0.0),
            gradient=tuple(data.get("gradient", (0.0, 0.0))),
            parameters=data.get("parameters", {}),
        )


class SurfaceRole(str, Enum):
    SUPPORT = "support"
    TRAVERSABLE = "traversable"
    ATTACHMENT = "attachment"
    OVERHEAD = "overhead"
    BOUNDARY = "boundary"
    OPEN_EDGE = "open_edge"
    NON_INTERACTIVE = "non_interactive"


@dataclass(frozen=True)
class StructuralSurface:
    """Semantic surface patch consumed by placement and navigation stages."""

    surface_id: str
    roles: frozenset[SurfaceRole]
    source_id: str
    transform: Transform3D = field(default_factory=Transform3D)
    geometry_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "surface_id", _require_id(self.surface_id, "surface_id")
        )
        object.__setattr__(self, "source_id", _require_id(self.source_id, "source_id"))
        roles = frozenset(SurfaceRole(role) for role in self.roles)
        if not roles:
            raise GeometryValidationError(
                "missing_surface_role",
                "at least one semantic role is required",
                entity_id=self.surface_id,
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.surface_id,
            "roles": sorted(role.value for role in self.roles),
            "source_id": self.source_id,
            "transform": self.transform.to_dict(),
            "geometry_ref": self.geometry_ref,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuralSurface":
        return cls(
            surface_id=data["id"],
            roles=frozenset(SurfaceRole(role) for role in data.get("roles", [])),
            source_id=data["source_id"],
            transform=Transform3D.from_dict(data.get("transform")),
            geometry_ref=data.get("geometry_ref"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class PlatformSpec:
    """Raised/sunken slab, mezzanine, balcony, bridge, or catwalk."""

    platform_id: str
    space_id: str
    footprint: Footprint2D
    elevation: float
    thickness: float = 0.15
    open_edge_indices: tuple[int, ...] = ()
    guarded_hole_indices: tuple[int, ...] = ()
    traversable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "platform_id", _require_id(self.platform_id, "platform_id")
        )
        object.__setattr__(self, "space_id", _require_id(self.space_id, "space_id"))
        elevation = _finite(self.elevation, "elevation", entity_id=self.platform_id)
        thickness = _finite(self.thickness, "thickness", entity_id=self.platform_id)
        if thickness <= 0:
            raise GeometryValidationError(
                "invalid_platform_thickness",
                f"thickness must be positive; got {thickness}",
                entity_id=self.platform_id,
            )
        open_edges = tuple(int(index) for index in self.open_edge_indices)
        edge_count = len(self.footprint.outer)
        if len(open_edges) != len(set(open_edges)) or any(
            index < 0 or index >= edge_count for index in open_edges
        ):
            raise GeometryValidationError(
                "invalid_open_edge",
                f"open_edge_indices must be unique and inside [0, {edge_count})",
                entity_id=self.platform_id,
            )
        guarded_holes = tuple(int(index) for index in self.guarded_hole_indices)
        hole_count = len(self.footprint.holes)
        if len(guarded_holes) != len(set(guarded_holes)) or any(
            index < 0 or index >= hole_count for index in guarded_holes
        ):
            raise GeometryValidationError(
                "invalid_guarded_hole",
                "guarded_hole_indices must be unique and inside "
                f"[0, {hole_count})",
                entity_id=self.platform_id,
            )
        object.__setattr__(self, "elevation", elevation)
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "open_edge_indices", open_edges)
        object.__setattr__(self, "guarded_hole_indices", guarded_holes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.platform_id,
            "space_id": self.space_id,
            "footprint": self.footprint.to_dict(),
            "elevation": self.elevation,
            "thickness": self.thickness,
            "open_edge_indices": list(self.open_edge_indices),
            "guarded_hole_indices": list(self.guarded_hole_indices),
            "traversable": self.traversable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlatformSpec":
        return cls(
            platform_id=data["id"],
            space_id=data["space_id"],
            footprint=Footprint2D.from_dict(data["footprint"]),
            elevation=data["elevation"],
            thickness=data.get("thickness", 0.15),
            open_edge_indices=tuple(data.get("open_edge_indices", [])),
            guarded_hole_indices=tuple(data.get("guarded_hole_indices", [])),
            traversable=bool(data.get("traversable", True)),
        )


@dataclass(frozen=True)
class HeightfieldSpec:
    """Regular sampled elevation grid for terrain and organic floors."""

    heightfield_id: str
    space_id: str
    heights: tuple[tuple[float, ...], ...]
    cell_size: Point2 = (1.0, 1.0)
    origin: Point3 = (0.0, 0.0, 0.0)
    replaces_floor: bool = False
    """Suppress the room's default slab when this grid is the structural floor."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "heightfield_id",
            _require_id(self.heightfield_id, "heightfield_id"),
        )
        object.__setattr__(self, "space_id", _require_id(self.space_id, "space_id"))
        rows = tuple(
            tuple(
                _finite(value, f"heights[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            )
            for row_index, row in enumerate(self.heights)
        )
        if len(rows) < 2 or any(len(row) < 2 for row in rows):
            raise GeometryValidationError(
                "invalid_heightfield",
                "heightfield requires at least a 2 × 2 grid",
                entity_id=self.heightfield_id,
            )
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise GeometryValidationError(
                "invalid_heightfield",
                "heightfield rows must have equal length",
                entity_id=self.heightfield_id,
            )
        cell_size = _point2(self.cell_size, "cell_size")
        if cell_size[0] <= 0 or cell_size[1] <= 0:
            raise GeometryValidationError(
                "invalid_heightfield",
                "cell_size values must be positive",
                entity_id=self.heightfield_id,
            )
        object.__setattr__(self, "heights", rows)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(
            self,
            "origin",
            _point3(self.origin, "origin", entity_id=self.heightfield_id),
        )
        object.__setattr__(self, "replaces_floor", bool(self.replaces_floor))

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.heights), len(self.heights[0]))

    def height_at(self, x: float, y: float) -> float:
        """Interpolate height using the same diagonal split as the compiler."""

        x = _finite(x, "x", entity_id=self.heightfield_id)
        y = _finite(y, "y", entity_id=self.heightfield_id)
        local_x = (x - self.origin[0]) / self.cell_size[0]
        local_y = (y - self.origin[1]) / self.cell_size[1]
        rows, columns = self.shape
        if not 0 <= local_x <= columns - 1 or not 0 <= local_y <= rows - 1:
            raise GeometryValidationError(
                "heightfield_query_out_of_bounds",
                f"point ({x}, {y}) is outside the sampled heightfield",
                entity_id=self.heightfield_id,
            )
        column = min(int(math.floor(local_x)), columns - 2)
        row = min(int(math.floor(local_y)), rows - 2)
        u = local_x - column
        v = local_y - row
        lower_left = self.heights[row][column]
        lower_right = self.heights[row][column + 1]
        upper_left = self.heights[row + 1][column]
        upper_right = self.heights[row + 1][column + 1]
        # Compiler triangles are LL/LR/UR and LL/UR/UL.
        if v <= u:
            local_height = (
                lower_left
                + u * (lower_right - lower_left)
                + v * (upper_right - lower_right)
            )
        else:
            local_height = (
                lower_left
                + u * (upper_right - upper_left)
                + v * (upper_left - lower_left)
            )
        return self.origin[2] + local_height

    def normal_at(self, x: float, y: float) -> Point3:
        """Return the compiler-consistent upward triangle normal at XY."""

        self.height_at(x, y)  # Bounds and finite-value validation.
        local_x = (x - self.origin[0]) / self.cell_size[0]
        local_y = (y - self.origin[1]) / self.cell_size[1]
        rows, columns = self.shape
        column = min(int(math.floor(local_x)), columns - 2)
        row = min(int(math.floor(local_y)), rows - 2)
        u = local_x - column
        v = local_y - row
        lower_left = self.heights[row][column]
        lower_right = self.heights[row][column + 1]
        upper_left = self.heights[row + 1][column]
        upper_right = self.heights[row + 1][column + 1]
        if v <= u:
            gradient_x = (lower_right - lower_left) / self.cell_size[0]
            gradient_y = (upper_right - lower_right) / self.cell_size[1]
        else:
            gradient_x = (upper_right - upper_left) / self.cell_size[0]
            gradient_y = (upper_left - lower_left) / self.cell_size[1]
        normal = (-gradient_x, -gradient_y, 1.0)
        length = math.sqrt(sum(component * component for component in normal))
        return tuple(component / length for component in normal)  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.heightfield_id,
            "space_id": self.space_id,
            "heights": [list(row) for row in self.heights],
            "cell_size": list(self.cell_size),
            "origin": list(self.origin),
            "replaces_floor": self.replaces_floor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeightfieldSpec":
        return cls(
            heightfield_id=data["id"],
            space_id=data["space_id"],
            heights=tuple(tuple(row) for row in data["heights"]),
            cell_size=tuple(data.get("cell_size", (1.0, 1.0))),
            origin=tuple(data.get("origin", (0.0, 0.0, 0.0))),
            replaces_floor=bool(data.get("replaces_floor", False)),
        )


@dataclass(frozen=True)
class MeshSurfaceAnnotation:
    """Semantic roles authored for selected triangles of a structural mesh."""

    annotation_id: str
    triangle_indices: tuple[int, ...]
    roles: frozenset[SurfaceRole]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_id",
            _require_id(self.annotation_id, "annotation_id"),
        )
        indices = tuple(int(index) for index in self.triangle_indices)
        if not indices or any(index < 0 for index in indices):
            raise GeometryValidationError(
                "invalid_mesh_annotation",
                "triangle_indices must contain non-negative indices",
                entity_id=self.annotation_id,
            )
        if len(indices) != len(set(indices)):
            raise GeometryValidationError(
                "invalid_mesh_annotation",
                "triangle_indices must not contain duplicates",
                entity_id=self.annotation_id,
            )
        roles = frozenset(SurfaceRole(role) for role in self.roles)
        if not roles:
            raise GeometryValidationError(
                "missing_surface_role",
                "at least one semantic role is required",
                entity_id=self.annotation_id,
            )
        object.__setattr__(self, "triangle_indices", indices)
        object.__setattr__(self, "roles", roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.annotation_id,
            "triangle_indices": list(self.triangle_indices),
            "roles": sorted(role.value for role in self.roles),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshSurfaceAnnotation":
        return cls(
            annotation_id=data["id"],
            triangle_indices=tuple(data["triangle_indices"]),
            roles=frozenset(SurfaceRole(role) for role in data["roles"]),
        )


@dataclass(frozen=True)
class StructuralMeshSpec:
    """Imported/freeform structural mesh with explicit units and semantics."""

    mesh_id: str
    space_id: str
    mesh_path: str
    unit_scale: float
    transform: Transform3D = field(default_factory=Transform3D)
    annotations: tuple[MeshSurfaceAnnotation, ...] = ()
    require_watertight: bool = False
    normal_orientation: str = "unspecified"
    """Expected watertight winding: exterior, interior (cavern), or unspecified."""
    replaces_room_shell: bool = False
    """Use this mesh as the room shell instead of generating a flat box room."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_id", _require_id(self.mesh_id, "mesh_id"))
        object.__setattr__(self, "space_id", _require_id(self.space_id, "space_id"))
        path = str(self.mesh_path).strip()
        if not path:
            raise GeometryValidationError(
                "missing_mesh_path",
                "mesh_path must not be empty",
                entity_id=self.mesh_id,
            )
        scale = _finite(self.unit_scale, "unit_scale", entity_id=self.mesh_id)
        if scale <= 0:
            raise GeometryValidationError(
                "invalid_mesh_scale",
                f"unit_scale must be positive; got {scale}",
                entity_id=self.mesh_id,
            )
        annotations = tuple(self.annotations)
        normal_orientation = str(self.normal_orientation).strip().lower()
        if normal_orientation not in {"unspecified", "exterior", "interior"}:
            raise GeometryValidationError(
                "invalid_normal_orientation",
                "normal_orientation must be exterior, interior, or unspecified",
                entity_id=self.mesh_id,
            )
        annotation_ids = [annotation.annotation_id for annotation in annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise GeometryValidationError(
                "duplicate_mesh_annotation",
                "annotation IDs must be unique",
                entity_id=self.mesh_id,
            )
        object.__setattr__(self, "mesh_path", path)
        object.__setattr__(self, "unit_scale", scale)
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "normal_orientation", normal_orientation)
        object.__setattr__(self, "replaces_room_shell", bool(self.replaces_room_shell))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.mesh_id,
            "space_id": self.space_id,
            "mesh_path": self.mesh_path,
            "unit_scale": self.unit_scale,
            "transform": self.transform.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "require_watertight": self.require_watertight,
            "normal_orientation": self.normal_orientation,
            "replaces_room_shell": self.replaces_room_shell,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuralMeshSpec":
        if "unit_scale" not in data:
            raise GeometryValidationError(
                "missing_mesh_units",
                "freeform structural meshes require explicit unit_scale",
                entity_id=data.get("id"),
            )
        return cls(
            mesh_id=data["id"],
            space_id=data["space_id"],
            mesh_path=data["mesh_path"],
            unit_scale=data["unit_scale"],
            transform=Transform3D.from_dict(data.get("transform")),
            annotations=tuple(
                MeshSurfaceAnnotation.from_dict(annotation)
                for annotation in data.get("annotations", [])
            ),
            require_watertight=bool(data.get("require_watertight", False)),
            normal_orientation=data.get("normal_orientation", "unspecified"),
            replaces_room_shell=bool(data.get("replaces_room_shell", False)),
        )


class PortalType(str, Enum):
    DOOR = "door"
    OPEN = "open"
    WINDOW = "window"
    ARCH = "arch"
    CAVE_MOUTH = "cave_mouth"


@dataclass(frozen=True)
class PortalSpec:
    """An aperture connecting two spaces or one space to the exterior."""

    portal_id: str
    portal_type: PortalType
    source_space_id: str
    target_space_id: str | None = None
    transform: Transform3D = field(default_factory=Transform3D)
    width: float = 1.0
    height: float = 2.1
    boundary_loop_index: int | None = None
    boundary_edge_index: int | None = None
    position_along: float | None = None
    sill_height: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "portal_id", _require_id(self.portal_id, "portal_id"))
        object.__setattr__(
            self,
            "source_space_id",
            _require_id(self.source_space_id, "source_space_id"),
        )
        if self.target_space_id is not None:
            object.__setattr__(
                self,
                "target_space_id",
                _require_id(self.target_space_id, "target_space_id"),
            )
        object.__setattr__(self, "portal_type", PortalType(self.portal_type))
        width = _finite(self.width, "width", entity_id=self.portal_id)
        height = _finite(self.height, "height", entity_id=self.portal_id)
        sill_height = _finite(self.sill_height, "sill_height", entity_id=self.portal_id)
        if width <= 0 or height <= 0:
            raise GeometryValidationError(
                "invalid_portal_size",
                f"width and height must be positive; got {width} × {height}",
                entity_id=self.portal_id,
            )
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        if (self.boundary_loop_index is None) != (self.boundary_edge_index is None):
            raise GeometryValidationError(
                "invalid_portal_boundary",
                "boundary_loop_index and boundary_edge_index must be set together",
                entity_id=self.portal_id,
            )
        if self.boundary_loop_index is not None:
            loop_index = int(self.boundary_loop_index)
            edge_index = int(self.boundary_edge_index)
            if loop_index < 0 or edge_index < 0 or self.position_along is None:
                raise GeometryValidationError(
                    "invalid_portal_boundary",
                    "boundary indices and position_along must be non-negative",
                    entity_id=self.portal_id,
                )
            position_along = _finite(
                self.position_along, "position_along", entity_id=self.portal_id
            )
            if position_along < 0:
                raise GeometryValidationError(
                    "invalid_portal_boundary",
                    "position_along must be non-negative",
                    entity_id=self.portal_id,
                )
            object.__setattr__(self, "boundary_loop_index", loop_index)
            object.__setattr__(self, "boundary_edge_index", edge_index)
            object.__setattr__(self, "position_along", position_along)
        object.__setattr__(self, "sill_height", sill_height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.portal_id,
            "type": self.portal_type.value,
            "source_space_id": self.source_space_id,
            "target_space_id": self.target_space_id,
            "transform": self.transform.to_dict(),
            "width": self.width,
            "height": self.height,
            "boundary_loop_index": self.boundary_loop_index,
            "boundary_edge_index": self.boundary_edge_index,
            "position_along": self.position_along,
            "sill_height": self.sill_height,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortalSpec":
        return cls(
            portal_id=data["id"],
            portal_type=PortalType(data["type"]),
            source_space_id=data["source_space_id"],
            target_space_id=data.get("target_space_id"),
            transform=Transform3D.from_dict(data.get("transform")),
            width=data.get("width", 1.0),
            height=data.get("height", 2.1),
            boundary_loop_index=data.get("boundary_loop_index"),
            boundary_edge_index=data.get("boundary_edge_index"),
            position_along=data.get("position_along"),
            sill_height=data.get("sill_height", 0.0),
        )


class ConnectorType(str, Enum):
    STAIRS_STRAIGHT = "stairs_straight"
    STAIRS_L = "stairs_l"
    STAIRS_U = "stairs_u"
    STAIRS_SPIRAL = "stairs_spiral"
    RAMP = "ramp"
    LADDER = "ladder"
    ELEVATOR = "elevator"
    SHAFT = "shaft"
    NATURAL_PASSAGE = "natural_passage"


_AUTO_CONNECTOR_CAPABILITIES = frozenset({"__scenesmith_auto__"})


@dataclass(frozen=True)
class ConnectorEndpoint:
    """One end of a vertical or non-planar connection."""

    space_id: str
    level_id: str
    position: Point3

    def __post_init__(self) -> None:
        object.__setattr__(self, "space_id", _require_id(self.space_id, "space_id"))
        object.__setattr__(self, "level_id", _require_id(self.level_id, "level_id"))
        object.__setattr__(
            self,
            "position",
            _point3(self.position, "position", entity_id=self.space_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "level_id": self.level_id,
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConnectorEndpoint":
        return cls(
            space_id=data["space_id"],
            level_id=data["level_id"],
            position=tuple(data["position"]),
        )


@dataclass(frozen=True)
class ConnectorSpec:
    """Semantic connector between two structural spaces or elevation patches."""

    connector_id: str
    connector_type: ConnectorType
    start: ConnectorEndpoint
    end: ConnectorEndpoint
    width: float = 1.0
    clearance_height: float = 2.1
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required_capabilities: frozenset[str] = field(
        default_factory=lambda: _AUTO_CONNECTOR_CAPABILITIES
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "connector_id", _require_id(self.connector_id, "connector_id")
        )
        connector_type = ConnectorType(self.connector_type)
        width = _finite(self.width, "width", entity_id=self.connector_id)
        clearance = _finite(
            self.clearance_height,
            "clearance_height",
            entity_id=self.connector_id,
        )
        if width <= 0 or clearance <= 0:
            raise UnsafeConnectorError(
                f"width and clearance_height must be positive; got {width} and "
                f"{clearance}",
                entity_id=self.connector_id,
            )
        if math.dist(self.start.position, self.end.position) <= GEOMETRY_TOLERANCE:
            raise UnsafeConnectorError(
                "start and end positions must differ", entity_id=self.connector_id
            )
        raw_capabilities = self.required_capabilities
        if raw_capabilities == _AUTO_CONNECTOR_CAPABILITIES:
            raw_capabilities = (
                frozenset({"climb"})
                if connector_type in {ConnectorType.LADDER, ConnectorType.SHAFT}
                else frozenset({"walk"})
            )
        capabilities = frozenset(
            _require_id(capability, "required capability")
            for capability in raw_capabilities
        )
        if not capabilities:
            raise UnsafeConnectorError(
                "at least one required capability is required",
                entity_id=self.connector_id,
            )
        parameters = dict(self.parameters)
        if "geometry_embedded" in parameters:
            embedded = parameters["geometry_embedded"]
            if not isinstance(embedded, bool):
                raise UnsafeConnectorError(
                    "parameters.geometry_embedded must be a boolean",
                    entity_id=self.connector_id,
                )
            if connector_type not in {
                ConnectorType.NATURAL_PASSAGE,
                ConnectorType.SHAFT,
            }:
                raise UnsafeConnectorError(
                    "parameters.geometry_embedded is only valid for natural_passage "
                    "or shaft connectors",
                    entity_id=self.connector_id,
                )
        if parameters.get("geometry_embedded"):
            waypoints = parameters.get("waypoints", ())
            if not isinstance(waypoints, (list, tuple)):
                raise UnsafeConnectorError(
                    "embedded connector parameters.waypoints must be an array",
                    entity_id=self.connector_id,
                )
            for index, waypoint in enumerate(waypoints):
                _point3(
                    waypoint,
                    f"parameters.waypoints[{index}]",
                    entity_id=self.connector_id,
                )
        object.__setattr__(self, "connector_type", connector_type)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "clearance_height", clearance)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "required_capabilities", capabilities)

    @property
    def rise(self) -> float:
        return self.end.position[2] - self.start.position[2]

    @property
    def horizontal_run(self) -> float:
        return math.dist(self.start.position[:2], self.end.position[:2])

    def validate_straight_access(
        self,
        *,
        max_ramp_slope: float = 1.0 / 12.0,
        min_stair_rise: float = 0.10,
        max_stair_rise: float = 0.20,
        min_stair_run: float = 0.22,
    ) -> None:
        """Validate basic straight-stair/ramp parameters before compilation."""

        rise = abs(self.rise)
        run = self.horizontal_run
        if self.connector_type == ConnectorType.RAMP:
            if run <= GEOMETRY_TOLERANCE:
                raise UnsafeConnectorError(
                    "a ramp must have non-zero horizontal run",
                    entity_id=self.connector_id,
                )
            slope = rise / run
            if slope > max_ramp_slope + GEOMETRY_TOLERANCE:
                raise UnsafeConnectorError(
                    f"ramp slope {slope:.6g} exceeds maximum {max_ramp_slope:.6g}",
                    entity_id=self.connector_id,
                )
        elif self.connector_type == ConnectorType.STAIRS_STRAIGHT:
            riser_count = self.parameters.get("riser_count")
            if not isinstance(riser_count, int) or isinstance(riser_count, bool):
                raise UnsafeConnectorError(
                    "straight stairs require integer parameters.riser_count",
                    entity_id=self.connector_id,
                )
            if riser_count <= 0:
                raise UnsafeConnectorError(
                    "riser_count must be positive", entity_id=self.connector_id
                )
            per_riser = rise / riser_count
            per_tread = run / riser_count
            if not min_stair_rise <= per_riser <= max_stair_rise:
                raise UnsafeConnectorError(
                    f"riser height {per_riser:.6g} is outside "
                    f"[{min_stair_rise}, {max_stair_rise}]",
                    entity_id=self.connector_id,
                )
            if per_tread < min_stair_run:
                raise UnsafeConnectorError(
                    f"tread run {per_tread:.6g} is below minimum {min_stair_run}",
                    entity_id=self.connector_id,
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.connector_id,
            "type": self.connector_type.value,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "width": self.width,
            "clearance_height": self.clearance_height,
            "parameters": dict(self.parameters),
            "required_capabilities": sorted(self.required_capabilities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConnectorSpec":
        connector_type = ConnectorType(data["type"])
        default_capabilities = (
            frozenset({"climb"})
            if connector_type in {ConnectorType.LADDER, ConnectorType.SHAFT}
            else frozenset({"walk"})
        )
        return cls(
            connector_id=data["id"],
            connector_type=connector_type,
            start=ConnectorEndpoint.from_dict(data["start"]),
            end=ConnectorEndpoint.from_dict(data["end"]),
            width=data.get("width", 1.0),
            clearance_height=data.get("clearance_height", 2.1),
            parameters=data.get("parameters", {}),
            required_capabilities=frozenset(
                data.get("required_capabilities", default_capabilities)
            ),
        )


def validate_structural_references(
    *,
    levels: Sequence[LevelSpec],
    space_level_ids: Mapping[str, str],
    connectors: Sequence[ConnectorSpec] = (),
    portals: Sequence[PortalSpec] = (),
    structural_meshes: Sequence[StructuralMeshSpec] = (),
    platforms: Sequence[PlatformSpec] = (),
    heightfields: Sequence[HeightfieldSpec] = (),
) -> None:
    """Validate unique IDs and all level/space references in a semantic layout."""

    level_ids = [level.level_id for level in levels]
    if len(level_ids) != len(set(level_ids)):
        raise GeometryValidationError("duplicate_level", "level IDs must be unique")
    known_levels = set(level_ids)
    if not known_levels:
        raise GeometryValidationError("missing_level", "at least one level is required")

    known_spaces = set(space_level_ids)
    for space_id, level_id in space_level_ids.items():
        _require_id(space_id, "space_id")
        if level_id not in known_levels:
            raise UnknownLevelError(level_id, entity_id=space_id)

    mesh_ids: set[str] = set()
    for mesh in structural_meshes:
        if mesh.mesh_id in mesh_ids:
            raise GeometryValidationError(
                "duplicate_structural_mesh",
                "structural mesh IDs must be unique",
                entity_id=mesh.mesh_id,
            )
        mesh_ids.add(mesh.mesh_id)
        if mesh.space_id not in known_spaces:
            raise GeometryValidationError(
                "unknown_mesh_space",
                f"space '{mesh.space_id}' is not defined",
                entity_id=mesh.mesh_id,
            )

    platform_ids: set[str] = set()
    for platform in platforms:
        if platform.platform_id in platform_ids:
            raise GeometryValidationError(
                "duplicate_platform",
                "platform IDs must be unique",
                entity_id=platform.platform_id,
            )
        platform_ids.add(platform.platform_id)
        if platform.space_id not in known_spaces:
            raise GeometryValidationError(
                "unknown_platform_space",
                f"space '{platform.space_id}' is not defined",
                entity_id=platform.platform_id,
            )

    heightfield_ids: set[str] = set()
    for heightfield in heightfields:
        if heightfield.heightfield_id in heightfield_ids:
            raise GeometryValidationError(
                "duplicate_heightfield",
                "heightfield IDs must be unique",
                entity_id=heightfield.heightfield_id,
            )
        heightfield_ids.add(heightfield.heightfield_id)
        if heightfield.space_id not in known_spaces:
            raise GeometryValidationError(
                "unknown_heightfield_space",
                f"space '{heightfield.space_id}' is not defined",
                entity_id=heightfield.heightfield_id,
            )

    connector_ids: set[str] = set()
    for connector in connectors:
        if connector.connector_id in connector_ids:
            raise GeometryValidationError(
                "duplicate_connector",
                "connector IDs must be unique",
                entity_id=connector.connector_id,
            )
        connector_ids.add(connector.connector_id)
        # One tall semantic space (an atrium, library hall, stage, etc.) may
        # contain several structural datums. Internal stairs therefore retain
        # their authored endpoint levels even though the enclosing room has one
        # canonical topology level. Cross-space connectors remain strict.
        is_internal_multilevel_connector = (
            connector.start.space_id == connector.end.space_id
            and connector.start.level_id != connector.end.level_id
        )
        for endpoint in (connector.start, connector.end):
            if endpoint.space_id not in known_spaces:
                raise UnknownConnectorEndpointError(
                    endpoint.space_id, connector_id=connector.connector_id
                )
            if endpoint.level_id not in known_levels:
                raise UnknownLevelError(
                    endpoint.level_id, entity_id=connector.connector_id
                )
            expected_level = space_level_ids[endpoint.space_id]
            if (
                endpoint.level_id != expected_level
                and not is_internal_multilevel_connector
            ):
                raise GeometryValidationError(
                    "connector_level_mismatch",
                    f"endpoint space '{endpoint.space_id}' belongs to level "
                    f"'{expected_level}', not '{endpoint.level_id}'",
                    entity_id=connector.connector_id,
                )

    portal_ids: set[str] = set()
    for portal in portals:
        if portal.portal_id in portal_ids:
            raise GeometryValidationError(
                "duplicate_portal",
                "portal IDs must be unique",
                entity_id=portal.portal_id,
            )
        portal_ids.add(portal.portal_id)
        for space_id in (portal.source_space_id, portal.target_space_id):
            if space_id is not None and space_id not in known_spaces:
                raise GeometryValidationError(
                    "unknown_portal_endpoint",
                    f"space '{space_id}' is not defined",
                    entity_id=portal.portal_id,
                )
