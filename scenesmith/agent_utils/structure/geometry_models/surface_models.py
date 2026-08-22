"""Levels, footprints, surfaces, platforms, and heightfields."""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Loop2,
    Point2,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    InvalidFootprintError,
    _finite,
    _loops_intersect,
    _normalize_loop,
    _point2,
    _point3,
    _point_in_loop,
    _require_id,
    _signed_area,
)


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
                "guarded_hole_indices must be unique and inside " f"[0, {hole_count})",
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
