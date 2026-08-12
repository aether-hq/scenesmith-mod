"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from scenesmith.agent_utils.structural_geometry import (
    GEOMETRY_TOLERANCE,
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    GeometryValidationError,
    HeightfieldSpec,
    PlatformSpec,
    Point2,
    Point3,
    PortalSpec,
    StructuralMeshSpec,
    StructuralSurface,
    SurfaceRole,
    Transform3D,
    UnsafeConnectorError,
    UnsupportedGeometryError,
)

Triangle = tuple[int, int, int]


def _subtract(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point3, b: Point3) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Point3, b: Point3) -> float:
    return sum(first * second for first, second in zip(a, b))


def _normalize(vector: Point3) -> Point3:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= GEOMETRY_TOLERANCE:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


@dataclass(frozen=True)
class TriangleMesh:
    """Minimal indexed triangle mesh with deterministic validation/export."""

    vertices: tuple[Point3, ...]
    triangles: tuple[Triangle, ...]

    def __post_init__(self) -> None:
        if not self.vertices:
            raise ValueError("mesh requires at least one vertex")
        if not self.triangles:
            raise ValueError("mesh requires at least one triangle")
        for vertex_index, vertex in enumerate(self.vertices):
            if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
                raise ValueError(f"vertex {vertex_index} must contain 3 finite values")
        for triangle_index, triangle in enumerate(self.triangles):
            if len(triangle) != 3 or len(set(triangle)) != 3:
                raise ValueError(
                    f"triangle {triangle_index} must contain 3 distinct indices"
                )
            if min(triangle) < 0 or max(triangle) >= len(self.vertices):
                raise ValueError(
                    f"triangle {triangle_index} references an invalid vertex"
                )
            a, b, c = (self.vertices[index] for index in triangle)
            normal = _cross(_subtract(b, a), _subtract(c, a))
            if math.sqrt(sum(value * value for value in normal)) <= GEOMETRY_TOLERANCE:
                raise ValueError(f"triangle {triangle_index} has zero area")

    @property
    def bounds(self) -> tuple[Point3, Point3]:
        return (
            tuple(min(vertex[axis] for vertex in self.vertices) for axis in range(3)),
            tuple(max(vertex[axis] for vertex in self.vertices) for axis in range(3)),
        )  # type: ignore[return-value]

    def triangle_normal(self, triangle_index: int) -> Point3:
        triangle = self.triangles[triangle_index]
        a, b, c = (self.vertices[index] for index in triangle)
        return _normalize(_cross(_subtract(b, a), _subtract(c, a)))

    def to_obj(self, *, object_name: str = "structure") -> str:
        """Return a dependency-free Wavefront OBJ representation."""

        lines = [f"o {object_name}"]
        lines.extend(f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in self.vertices)
        lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.triangles)
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CompiledSurfacePatch:
    """A semantic surface with an explicit ordered 3D boundary polygon."""

    surface: StructuralSurface
    boundary: tuple[Point3, ...]
    normal: Point3

    def __post_init__(self) -> None:
        if len(self.boundary) < 3:
            raise ValueError("surface patch boundary needs at least 3 points")
        object.__setattr__(self, "normal", _normalize(self.normal))


@dataclass(frozen=True)
class CollisionPrimitive:
    """Analytic collision primitive retained alongside the triangle mesh."""

    primitive_id: str
    primitive_type: str
    transform: Transform3D
    dimensions: Point3
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledStructure:
    """Simulator-agnostic result of compiling one semantic structure."""

    structure_id: str
    visual_mesh: TriangleMesh
    collision_mesh: TriangleMesh
    surfaces: tuple[CompiledSurfacePatch, ...]
    collision_primitives: tuple[CollisionPrimitive, ...] = ()
    triangle_groups: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledStructurePaths:
    """Files emitted for a compiled structural element."""

    mesh_path: Path
    sdf_path: Path
    surfaces_path: Path


class _MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[Point3] = []
        self.triangles: list[Triangle] = []

    def add_prism(self, vertices: Sequence[Point3]) -> None:
        """Add a hexahedron using bottom 0–3 and top 4–7 vertex ordering."""

        if len(vertices) != 8:
            raise ValueError("a hexahedron needs exactly 8 vertices")
        offset = len(self.vertices)
        self.vertices.extend(vertices)
        faces = (
            (0, 2, 1),
            (0, 3, 2),  # bottom
            (4, 5, 6),
            (4, 6, 7),  # top
            (0, 1, 5),
            (0, 5, 4),  # -local Y
            (1, 2, 6),
            (1, 6, 5),  # +local X
            (2, 3, 7),
            (2, 7, 6),  # +local Y
            (3, 0, 4),
            (3, 4, 7),  # -local X
        )
        self.triangles.extend(
            tuple(offset + index for index in triangle)  # type: ignore[arg-type]
            for triangle in faces
        )

    def add_triangle(self, a: Point3, b: Point3, c: Point3) -> int:
        offset = len(self.vertices)
        self.vertices.extend((a, b, c))
        self.triangles.append((offset, offset + 1, offset + 2))
        return len(self.triangles) - 1

    def build(self) -> TriangleMesh:
        return TriangleMesh(tuple(self.vertices), tuple(self.triangles))


def _horizontal_basis(start: Point3, end: Point3) -> tuple[Point3, Point3, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    run = math.hypot(dx, dy)
    if run <= GEOMETRY_TOLERANCE:
        raise ValueError("connector must have non-zero horizontal run")
    along = (dx / run, dy / run, 0.0)
    across = (-along[1], along[0], 0.0)
    return along, across, run


def _offset(point: Point3, *vectors: tuple[Point3, float]) -> Point3:
    return tuple(
        point[axis] + sum(vector[axis] * scale for vector, scale in vectors)
        for axis in range(3)
    )  # type: ignore[return-value]


def _box_vertices(
    *,
    origin: Point3,
    along: Point3,
    across: Point3,
    length: float,
    width: float,
    bottom_z: float,
    top_start_z: float,
    top_end_z: float,
) -> tuple[Point3, ...]:
    start_center = (origin[0], origin[1], 0.0)
    end_center = _offset(start_center, (along, length))
    half_width = width / 2.0
    start_left = _offset(start_center, (across, -half_width))
    start_right = _offset(start_center, (across, half_width))
    end_left = _offset(end_center, (across, -half_width))
    end_right = _offset(end_center, (across, half_width))
    return (
        (start_left[0], start_left[1], bottom_z),
        (end_left[0], end_left[1], bottom_z),
        (end_right[0], end_right[1], bottom_z),
        (start_right[0], start_right[1], bottom_z),
        (start_left[0], start_left[1], top_start_z),
        (end_left[0], end_left[1], top_end_z),
        (end_right[0], end_right[1], top_end_z),
        (start_right[0], start_right[1], top_start_z),
    )


def compile_straight_stairs(connector: ConnectorSpec) -> CompiledStructure:
    """Compile a validated straight stair into step boxes and tread surfaces."""

    if connector.connector_type != ConnectorType.STAIRS_STRAIGHT:
        raise UnsupportedGeometryError(
            "compile_straight_stairs requires type 'stairs_straight'",
            entity_id=connector.connector_id,
        )
    connector.validate_straight_access()
    riser_count = int(connector.parameters["riser_count"])

    # Compile from the lower endpoint upward, regardless of semantic direction.
    low, high = (
        (connector.start, connector.end)
        if connector.start.position[2] <= connector.end.position[2]
        else (connector.end, connector.start)
    )
    along, across, total_run = _horizontal_basis(low.position, high.position)
    total_rise = high.position[2] - low.position[2]
    tread_run = total_run / riser_count
    riser_height = total_rise / riser_count
    builder = _MeshBuilder()
    surfaces: list[CompiledSurfacePatch] = []
    primitives: list[CollisionPrimitive] = []

    for index in range(riser_count):
        step_origin = _offset(low.position, (along, index * tread_run))
        top_z = low.position[2] + (index + 1) * riser_height
        vertices = _box_vertices(
            origin=step_origin,
            along=along,
            across=across,
            length=tread_run,
            width=connector.width,
            bottom_z=low.position[2],
            top_start_z=top_z,
            top_end_z=top_z,
        )
        builder.add_prism(vertices)
        surface_id = f"{connector.connector_id}_tread_{index + 1:03d}"
        surfaces.append(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=surface_id,
                    roles=frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
                    source_id=connector.connector_id,
                    metadata={
                        "connector_type": connector.connector_type.value,
                        "step_index": index,
                    },
                ),
                boundary=(vertices[4], vertices[5], vertices[6], vertices[7]),
                normal=(0.0, 0.0, 1.0),
            )
        )
        center = _offset(step_origin, (along, tread_run / 2.0))
        primitives.append(
            CollisionPrimitive(
                primitive_id=f"{connector.connector_id}_step_{index + 1:03d}",
                primitive_type="box",
                transform=Transform3D(
                    translation=(
                        center[0],
                        center[1],
                        low.position[2] + (top_z - low.position[2]) / 2.0,
                    ),
                    rotation_rpy=(0.0, 0.0, math.atan2(along[1], along[0])),
                ),
                dimensions=(
                    tread_run,
                    connector.width,
                    top_z - low.position[2],
                ),
            )
        )

    mesh = builder.build()
    return CompiledStructure(
        structure_id=connector.connector_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
        collision_primitives=tuple(primitives),
    )


def compile_straight_ramp(
    connector: ConnectorSpec, *, slab_thickness: float = 0.15
) -> CompiledStructure:
    """Compile a straight ramp as a sloped hexahedral slab."""

    if connector.connector_type != ConnectorType.RAMP:
        raise UnsupportedGeometryError(
            "compile_straight_ramp requires type 'ramp'",
            entity_id=connector.connector_id,
        )
    if slab_thickness <= 0 or not math.isfinite(slab_thickness):
        raise ValueError("slab_thickness must be finite and positive")
    connector.validate_straight_access()
    along, across, run = _horizontal_basis(
        connector.start.position, connector.end.position
    )
    bottom_z = (
        min(connector.start.position[2], connector.end.position[2]) - slab_thickness
    )
    vertices = _box_vertices(
        origin=connector.start.position,
        along=along,
        across=across,
        length=run,
        width=connector.width,
        bottom_z=bottom_z,
        top_start_z=connector.start.position[2],
        top_end_z=connector.end.position[2],
    )
    builder = _MeshBuilder()
    builder.add_prism(vertices)
    mesh = builder.build()
    top_normal = _normalize(
        _cross(_subtract(vertices[5], vertices[4]), _subtract(vertices[7], vertices[4]))
    )
    surface = CompiledSurfacePatch(
        surface=StructuralSurface(
            surface_id=f"{connector.connector_id}_top",
            roles=frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
            source_id=connector.connector_id,
            metadata={"connector_type": connector.connector_type.value},
        ),
        boundary=(vertices[4], vertices[5], vertices[6], vertices[7]),
        normal=top_normal,
    )
    return CompiledStructure(
        structure_id=connector.connector_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=(surface,),
    )


def compile_connector(connector: ConnectorSpec) -> CompiledStructure:
    """Compile a supported connector or fail with an explicit diagnostic."""

    if connector.connector_type == ConnectorType.STAIRS_STRAIGHT:
        return compile_straight_stairs(connector)
    if connector.connector_type in {ConnectorType.STAIRS_L, ConnectorType.STAIRS_U}:
        return compile_multisegment_stairs(connector)
    if connector.connector_type == ConnectorType.STAIRS_SPIRAL:
        return compile_spiral_stairs(connector)
    if connector.connector_type == ConnectorType.LADDER:
        return compile_ladder(connector)
    if connector.connector_type == ConnectorType.RAMP:
        if connector.parameters.get("waypoints"):
            return compile_multisegment_ramp(connector)
        return compile_straight_ramp(connector)
    raise UnsupportedGeometryError(
        f"no compiler is implemented for connector type "
        f"'{connector.connector_type.value}'",
        entity_id=connector.connector_id,
    )


def compile_spiral_stairs(connector: ConnectorSpec) -> CompiledStructure:
    """Compile a segmented spiral stair around an explicit center point."""

    if connector.connector_type != ConnectorType.STAIRS_SPIRAL:
        raise UnsupportedGeometryError(
            "compile_spiral_stairs requires type stairs_spiral",
            entity_id=connector.connector_id,
        )
    raw_center = connector.parameters.get("center")
    if not isinstance(raw_center, (list, tuple)) or len(raw_center) != 2:
        raise UnsafeConnectorError(
            "spiral stairs require parameters.center [x, y]",
            entity_id=connector.connector_id,
        )
    center_x, center_y = (float(value) for value in raw_center)
    if not all(math.isfinite(value) for value in (center_x, center_y)):
        raise UnsafeConnectorError(
            "spiral stair center must be finite", entity_id=connector.connector_id
        )
    riser_count = connector.parameters.get("riser_count")
    if (
        not isinstance(riser_count, int)
        or isinstance(riser_count, bool)
        or riser_count <= 0
    ):
        raise UnsafeConnectorError(
            "spiral stairs require positive integer parameters.riser_count",
            entity_id=connector.connector_id,
        )
    turns = float(connector.parameters.get("turns", 1.0))
    direction = str(connector.parameters.get("direction", "ccw")).lower()
    if not math.isfinite(turns) or turns <= 0:
        raise UnsafeConnectorError(
            "spiral stair turns must be finite and positive",
            entity_id=connector.connector_id,
        )
    if direction not in {"ccw", "cw"}:
        raise UnsafeConnectorError(
            "spiral stair direction must be 'ccw' or 'cw'",
            entity_id=connector.connector_id,
        )

    low, high = (
        (connector.start, connector.end)
        if connector.start.position[2] <= connector.end.position[2]
        else (connector.end, connector.start)
    )
    centerline_radius = math.hypot(
        low.position[0] - center_x, low.position[1] - center_y
    )
    inner_radius = centerline_radius - connector.width / 2.0
    outer_radius = centerline_radius + connector.width / 2.0
    if inner_radius <= GEOMETRY_TOLERANCE:
        raise UnsafeConnectorError(
            "spiral stair width reaches or crosses its center; increase radius",
            entity_id=connector.connector_id,
        )
    rise = high.position[2] - low.position[2]
    riser_height = rise / riser_count
    if not 0.10 <= riser_height <= 0.20:
        raise UnsafeConnectorError(
            f"riser height {riser_height:.6g} is outside [0.1, 0.2]",
            entity_id=connector.connector_id,
        )
    signed_angle = (1.0 if direction == "ccw" else -1.0) * math.tau * turns
    tread_depth = centerline_radius * abs(signed_angle) / riser_count
    if tread_depth < 0.22:
        raise UnsafeConnectorError(
            f"spiral centerline tread depth {tread_depth:.6g} is below 0.22",
            entity_id=connector.connector_id,
        )
    start_angle = math.atan2(low.position[1] - center_y, low.position[0] - center_x)
    expected_high = (
        center_x + centerline_radius * math.cos(start_angle + signed_angle),
        center_y + centerline_radius * math.sin(start_angle + signed_angle),
    )
    endpoint_tolerance = float(connector.parameters.get("endpoint_tolerance", 0.02))
    if math.dist(expected_high, high.position[:2]) > endpoint_tolerance:
        raise UnsafeConnectorError(
            "spiral stair end XY does not match center/radius/turns/direction",
            entity_id=connector.connector_id,
        )
    slab_thickness = float(connector.parameters.get("tread_thickness", 0.1))
    if not math.isfinite(slab_thickness) or slab_thickness <= 0:
        raise UnsafeConnectorError(
            "spiral stair tread_thickness must be finite and positive",
            entity_id=connector.connector_id,
        )

    builder = _MeshBuilder()
    surfaces: list[CompiledSurfacePatch] = []
    for index in range(riser_count):
        angle_start = start_angle + signed_angle * index / riser_count
        angle_end = start_angle + signed_angle * (index + 1) / riser_count
        top_z = low.position[2] + rise * (index + 1) / riser_count

        def radial_point(radius: float, angle: float, z: float) -> Point3:
            return (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
                z,
            )

        bottom = top_z - min(slab_thickness, riser_height)
        vertices = (
            radial_point(inner_radius, angle_start, bottom),
            radial_point(inner_radius, angle_end, bottom),
            radial_point(outer_radius, angle_end, bottom),
            radial_point(outer_radius, angle_start, bottom),
            radial_point(inner_radius, angle_start, top_z),
            radial_point(inner_radius, angle_end, top_z),
            radial_point(outer_radius, angle_end, top_z),
            radial_point(outer_radius, angle_start, top_z),
        )
        if direction == "cw":
            vertices = tuple(vertices[index] for index in (3, 2, 1, 0, 7, 6, 5, 4))
        builder.add_prism(vertices)
        surfaces.append(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=f"{connector.connector_id}_tread_{index + 1:03d}",
                    roles=frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
                    source_id=connector.connector_id,
                    metadata={
                        "connector_type": connector.connector_type.value,
                        "step_index": index,
                        "center": [center_x, center_y],
                        "inner_radius": inner_radius,
                        "outer_radius": outer_radius,
                    },
                ),
                boundary=(vertices[4], vertices[5], vertices[6], vertices[7]),
                normal=(0.0, 0.0, 1.0),
            )
        )
        surfaces.append(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=f"{connector.connector_id}_underside_{index + 1:03d}",
                    roles=frozenset({SurfaceRole.OVERHEAD}),
                    source_id=connector.connector_id,
                    metadata={
                        "connector_type": connector.connector_type.value,
                        "step_index": index,
                    },
                ),
                boundary=(vertices[0], vertices[3], vertices[2], vertices[1]),
                normal=(0.0, 0.0, -1.0),
            )
        )
    mesh = builder.build()
    return CompiledStructure(
        structure_id=connector.connector_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
    )


def compile_ladder(connector: ConnectorSpec) -> CompiledStructure:
    """Compile a vertical ladder with two rails and capability-gated rungs."""

    if connector.connector_type != ConnectorType.LADDER:
        raise UnsupportedGeometryError(
            "compile_ladder requires type ladder", entity_id=connector.connector_id
        )
    if "climb" not in connector.required_capabilities:
        raise UnsafeConnectorError(
            "ladder connectors must require the 'climb' capability",
            entity_id=connector.connector_id,
        )
    if connector.horizontal_run > 0.05:
        raise UnsafeConnectorError(
            "the ladder compiler currently requires vertically aligned endpoints",
            entity_id=connector.connector_id,
        )
    low, high = (
        (connector.start, connector.end)
        if connector.start.position[2] <= connector.end.position[2]
        else (connector.end, connector.start)
    )
    rise = high.position[2] - low.position[2]
    rung_count = connector.parameters.get("rung_count", max(2, math.ceil(rise / 0.3)))
    if (
        not isinstance(rung_count, int)
        or isinstance(rung_count, bool)
        or rung_count < 2
    ):
        raise UnsafeConnectorError(
            "ladder rung_count must be an integer of at least 2",
            entity_id=connector.connector_id,
        )
    rung_spacing = rise / (rung_count - 1)
    if not 0.2 <= rung_spacing <= 0.4:
        raise UnsafeConnectorError(
            f"ladder rung spacing {rung_spacing:.6g} is outside [0.2, 0.4]",
            entity_id=connector.connector_id,
        )
    yaw = math.radians(float(connector.parameters.get("yaw_degrees", 0.0)))
    rail_thickness = float(connector.parameters.get("rail_thickness", 0.06))
    rung_depth = float(connector.parameters.get("rung_depth", 0.05))
    if not all(
        math.isfinite(value) and value > 0 for value in (rail_thickness, rung_depth)
    ):
        raise UnsafeConnectorError(
            "ladder rail_thickness and rung_depth must be finite and positive",
            entity_id=connector.connector_id,
        )
    across = (math.cos(yaw), math.sin(yaw), 0.0)
    depth = (-math.sin(yaw), math.cos(yaw), 0.0)
    builder = _MeshBuilder()
    center = (
        (low.position[0] + high.position[0]) / 2.0,
        (low.position[1] + high.position[1]) / 2.0,
        0.0,
    )
    for side in (-1.0, 1.0):
        rail_center = _offset(center, (across, side * connector.width / 2.0))
        rail_origin = _offset(rail_center, (depth, -rail_thickness / 2.0))
        builder.add_prism(
            _box_vertices(
                origin=rail_origin,
                along=depth,
                across=across,
                length=rail_thickness,
                width=rail_thickness,
                bottom_z=low.position[2],
                top_start_z=high.position[2],
                top_end_z=high.position[2],
            )
        )
    surfaces: list[CompiledSurfacePatch] = []
    for index in range(rung_count):
        z = low.position[2] + index * rung_spacing
        rung_origin = _offset(center, (across, -connector.width / 2.0))
        vertices = _box_vertices(
            origin=rung_origin,
            along=across,
            across=depth,
            length=connector.width,
            width=rung_depth,
            bottom_z=z - rail_thickness / 2.0,
            top_start_z=z + rail_thickness / 2.0,
            top_end_z=z + rail_thickness / 2.0,
        )
        builder.add_prism(vertices)
        surfaces.append(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=f"{connector.connector_id}_rung_{index + 1:03d}",
                    roles=frozenset({SurfaceRole.SUPPORT}),
                    source_id=connector.connector_id,
                    metadata={
                        "connector_type": connector.connector_type.value,
                        "rung_index": index,
                        "required_capabilities": ["climb"],
                    },
                ),
                boundary=(vertices[4], vertices[5], vertices[6], vertices[7]),
                normal=(0.0, 0.0, 1.0),
            )
        )
    mesh = builder.build()
    return CompiledStructure(
        structure_id=connector.connector_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
    )


def _parameter_point3(value: Sequence[float], *, connector_id: str) -> Point3:
    if len(value) != 3:
        raise UnsafeConnectorError(
            "each stair waypoint must contain x, y, z",
            entity_id=connector_id,
        )
    point = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in point):
        raise UnsafeConnectorError(
            "stair waypoints must be finite", entity_id=connector_id
        )
    return point  # type: ignore[return-value]


def _compile_landing_segment(
    *,
    structure_id: str,
    start: Point3,
    end: Point3,
    width: float,
    thickness: float = 0.15,
) -> CompiledStructure:
    along, across, run = _horizontal_basis(start, end)
    if abs(start[2] - end[2]) > GEOMETRY_TOLERANCE:
        raise UnsafeConnectorError(
            "a landing segment must be horizontal", entity_id=structure_id
        )
    vertices = _box_vertices(
        origin=start,
        along=along,
        across=across,
        length=run,
        width=width,
        bottom_z=start[2] - thickness,
        top_start_z=start[2],
        top_end_z=end[2],
    )
    builder = _MeshBuilder()
    builder.add_prism(vertices)
    center = _offset(start, (along, run / 2.0))
    surface = CompiledSurfacePatch(
        surface=StructuralSurface(
            surface_id=f"{structure_id}_top",
            roles=frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
            source_id=structure_id,
            metadata={"segment_type": "landing"},
        ),
        boundary=(vertices[4], vertices[5], vertices[6], vertices[7]),
        normal=(0.0, 0.0, 1.0),
    )
    primitive = CollisionPrimitive(
        primitive_id=f"{structure_id}_box",
        primitive_type="box",
        transform=Transform3D(
            translation=(center[0], center[1], start[2] - thickness / 2.0),
            rotation_rpy=(0.0, 0.0, math.atan2(along[1], along[0])),
        ),
        dimensions=(run, width, thickness),
    )
    mesh = builder.build()
    return CompiledStructure(
        structure_id=structure_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=(surface,),
        collision_primitives=(primitive,),
    )


def compile_multisegment_stairs(connector: ConnectorSpec) -> CompiledStructure:
    """Compile L/U stairs from explicit centerline waypoints and flight risers."""

    if connector.connector_type not in {ConnectorType.STAIRS_L, ConnectorType.STAIRS_U}:
        raise UnsupportedGeometryError(
            "compile_multisegment_stairs requires stairs_l or stairs_u",
            entity_id=connector.connector_id,
        )
    raw_waypoints = connector.parameters.get("waypoints")
    expected_waypoints = 1 if connector.connector_type == ConnectorType.STAIRS_L else 2
    if (
        not isinstance(raw_waypoints, (list, tuple))
        or len(raw_waypoints) != expected_waypoints
    ):
        raise UnsafeConnectorError(
            f"{connector.connector_type.value} requires exactly "
            f"{expected_waypoints} parameters.waypoints",
            entity_id=connector.connector_id,
        )
    waypoints = tuple(
        _parameter_point3(waypoint, connector_id=connector.connector_id)
        for waypoint in raw_waypoints
    )
    points = (connector.start.position, *waypoints, connector.end.position)
    rises = tuple(end[2] - start[2] for start, end in zip(points, points[1:]))
    nonzero_rises = [rise for rise in rises if abs(rise) > GEOMETRY_TOLERANCE]
    if not nonzero_rises or any(
        math.copysign(1.0, rise) != math.copysign(1.0, nonzero_rises[0])
        for rise in nonzero_rises
    ):
        raise UnsafeConnectorError(
            "stair flights must rise or descend monotonically",
            entity_id=connector.connector_id,
        )
    raw_riser_counts = connector.parameters.get("riser_counts")
    if not isinstance(raw_riser_counts, (list, tuple)) or len(raw_riser_counts) != len(
        nonzero_rises
    ):
        raise UnsafeConnectorError(
            f"parameters.riser_counts must contain {len(nonzero_rises)} values",
            entity_id=connector.connector_id,
        )
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count <= 0
        for count in raw_riser_counts
    ):
        raise UnsafeConnectorError(
            "all riser_counts must be positive integers",
            entity_id=connector.connector_id,
        )

    first_direction, _, _ = _horizontal_basis(points[0], points[1])
    last_direction, _, _ = _horizontal_basis(points[-2], points[-1])
    direction_dot = _dot(first_direction, last_direction)
    if connector.connector_type == ConnectorType.STAIRS_L and abs(direction_dot) > 0.71:
        raise UnsafeConnectorError(
            "L stairs must turn approximately 90 degrees",
            entity_id=connector.connector_id,
        )
    if connector.connector_type == ConnectorType.STAIRS_U:
        if abs(rises[1]) > GEOMETRY_TOLERANCE:
            raise UnsafeConnectorError(
                "U-stair middle segment must be a horizontal landing",
                entity_id=connector.connector_id,
            )
        if direction_dot > -0.71:
            raise UnsafeConnectorError(
                "U-stair flights must run in approximately opposite directions",
                entity_id=connector.connector_id,
            )

    pieces: list[CompiledStructure] = []
    riser_index = 0
    for segment_index, (start, end, rise) in enumerate(
        zip(points, points[1:], rises), start=1
    ):
        segment_id = f"{connector.connector_id}_segment_{segment_index:02d}"
        if abs(rise) <= GEOMETRY_TOLERANCE:
            pieces.append(
                _compile_landing_segment(
                    structure_id=segment_id,
                    start=start,
                    end=end,
                    width=connector.width,
                )
            )
            continue
        synthetic = ConnectorSpec(
            connector_id=segment_id,
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint(
                connector.start.space_id, connector.start.level_id, start
            ),
            end=ConnectorEndpoint(connector.end.space_id, connector.end.level_id, end),
            width=connector.width,
            clearance_height=connector.clearance_height,
            parameters={"riser_count": raw_riser_counts[riser_index]},
            required_capabilities=connector.required_capabilities,
        )
        pieces.append(compile_straight_stairs(synthetic))
        riser_index += 1

    if connector.connector_type == ConnectorType.STAIRS_L:
        turn = waypoints[0]
        next_direction, _, _ = _horizontal_basis(turn, connector.end.position)
        landing_length = float(
            connector.parameters.get("landing_length", connector.width)
        )
        if not math.isfinite(landing_length) or landing_length <= 0:
            raise UnsafeConnectorError(
                "landing_length must be finite and positive",
                entity_id=connector.connector_id,
            )
        landing_start = _offset(turn, (next_direction, -landing_length / 2.0))
        landing_end = _offset(turn, (next_direction, landing_length / 2.0))
        pieces.append(
            _compile_landing_segment(
                structure_id=f"{connector.connector_id}_landing",
                start=landing_start,
                end=landing_end,
                width=connector.width,
            )
        )

    surfaces: list[CompiledSurfacePatch] = []
    primitives: list[CollisionPrimitive] = []
    for piece_index, piece in enumerate(pieces):
        primitives.extend(piece.collision_primitives)
        for patch in piece.surfaces:
            surfaces.append(
                CompiledSurfacePatch(
                    surface=StructuralSurface(
                        surface_id=patch.surface.surface_id,
                        roles=patch.surface.roles,
                        source_id=connector.connector_id,
                        transform=patch.surface.transform,
                        geometry_ref=patch.surface.geometry_ref,
                        metadata={
                            **patch.surface.metadata,
                            "connector_type": connector.connector_type.value,
                            "path_piece": piece_index,
                        },
                    ),
                    boundary=patch.boundary,
                    normal=patch.normal,
                )
            )
    mesh = combine_meshes(piece.visual_mesh for piece in pieces)
    return CompiledStructure(
        structure_id=connector.connector_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
        collision_primitives=tuple(primitives),
    )


def compile_multisegment_ramp(connector: ConnectorSpec) -> CompiledStructure:
    """Compile a turning/switchback ramp from sloped runs and flat landings."""

    if connector.connector_type != ConnectorType.RAMP:
        raise UnsupportedGeometryError(
            "compile_multisegment_ramp requires type ramp",
            entity_id=connector.connector_id,
        )
    raw_waypoints = connector.parameters.get("waypoints")
    if not isinstance(raw_waypoints, (list, tuple)) or not raw_waypoints:
        raise UnsafeConnectorError(
            "multisegment ramps require parameters.waypoints",
            entity_id=connector.connector_id,
        )
    waypoints = tuple(
        _parameter_point3(waypoint, connector_id=connector.connector_id)
        for waypoint in raw_waypoints
    )
    points = (connector.start.position, *waypoints, connector.end.position)
    rises = tuple(end[2] - start[2] for start, end in zip(points, points[1:]))
    nonzero_rises = [rise for rise in rises if abs(rise) > GEOMETRY_TOLERANCE]
    if not nonzero_rises or any(
        math.copysign(1.0, rise) != math.copysign(1.0, nonzero_rises[0])
        for rise in nonzero_rises
    ):
        raise UnsafeConnectorError(
            "ramp runs must rise or descend monotonically",
            entity_id=connector.connector_id,
        )

    pieces: list[CompiledStructure] = []
    for segment_index, (start, end, rise) in enumerate(
        zip(points, points[1:], rises), start=1
    ):
        segment_id = f"{connector.connector_id}_segment_{segment_index:02d}"
        if abs(rise) <= GEOMETRY_TOLERANCE:
            pieces.append(
                _compile_landing_segment(
                    structure_id=segment_id,
                    start=start,
                    end=end,
                    width=connector.width,
                )
            )
            continue
        synthetic = ConnectorSpec(
            connector_id=segment_id,
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint(
                connector.start.space_id, connector.start.level_id, start
            ),
            end=ConnectorEndpoint(connector.end.space_id, connector.end.level_id, end),
            width=connector.width,
            clearance_height=connector.clearance_height,
            required_capabilities=connector.required_capabilities,
        )
        pieces.append(compile_straight_ramp(synthetic))

    surfaces: list[CompiledSurfacePatch] = []
    for piece_index, piece in enumerate(pieces):
        for patch in piece.surfaces:
            surfaces.append(
                CompiledSurfacePatch(
                    surface=StructuralSurface(
                        surface_id=patch.surface.surface_id,
                        roles=patch.surface.roles,
                        source_id=connector.connector_id,
                        transform=patch.surface.transform,
                        geometry_ref=patch.surface.geometry_ref,
                        metadata={
                            **patch.surface.metadata,
                            "connector_type": "ramp_multisegment",
                            "path_piece": piece_index,
                        },
                    ),
                    boundary=patch.boundary,
                    normal=patch.normal,
                )
            )
    mesh = combine_meshes(piece.visual_mesh for piece in pieces)
    # Keep the combined collision representation uniformly mesh-based. Mixing
    # landing boxes with ramp meshes would cause SDF exporters to omit one tier.
    return CompiledStructure(
        structure_id=connector.connector_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
    )


def _triangulate_footprint(
    footprint: Footprint2D,
) -> tuple[tuple[Point2, Point2, Point2], ...]:
    """Constrained-Delaunay triangulate a validated polygon with holes."""

    try:
        from shapely import constrained_delaunay_triangles
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise UnsupportedGeometryError(
            "polygon compilation requires the project dependency 'shapely>=2.1.2'"
        ) from exc

    polygon = Polygon(footprint.outer, holes=footprint.holes)
    triangles_geometry = constrained_delaunay_triangles(polygon)
    triangles: list[tuple[Point2, Point2, Point2]] = []
    for geometry in triangles_geometry.geoms:
        coordinates = tuple(
            (float(x), float(y)) for x, y in list(geometry.exterior.coords)[:-1]
        )
        if len(coordinates) != 3:
            raise UnsupportedGeometryError(
                "constrained triangulation produced a non-triangle polygon"
            )
        if _signed_area_2d(coordinates) < 0:
            coordinates = (coordinates[0], coordinates[2], coordinates[1])
        triangles.append(coordinates)

    triangles.sort(
        key=lambda triangle: (
            sum(point[0] for point in triangle) / 3.0,
            sum(point[1] for point in triangle) / 3.0,
            triangle,
        )
    )
    if not triangles:
        raise UnsupportedGeometryError("polygon triangulation produced no triangles")
    return tuple(triangles)


def _signed_area_2d(points: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )


def _profile_height(profile: ElevationProfile, point: tuple[float, float]) -> float:
    if profile.profile_type not in {
        ElevationProfileType.PLANAR,
        ElevationProfileType.SLOPED,
    }:
        raise UnsupportedGeometryError(
            f"polygon compiler does not yet support elevation profile "
            f"'{profile.profile_type.value}'"
        )
    return profile.height_at(point)


def _add_slab(
    *,
    builder: _MeshBuilder,
    triangles: Sequence[tuple[Point2, Point2, Point2]],
    boundary_loops: Sequence[Sequence[Point2]],
    surface_height: Callable[[Point2], float],
    thickness: float,
    surface_is_top: bool,
    group_prefix: str,
    groups: dict[str, list[int]],
) -> None:
    surface_group = groups.setdefault(
        f"{group_prefix}_{'top' if surface_is_top else 'bottom'}", []
    )
    opposite_group = groups.setdefault(
        f"{group_prefix}_{'bottom' if surface_is_top else 'top'}", []
    )
    side_group = groups.setdefault(f"{group_prefix}_sides", [])
    direction = -1.0 if surface_is_top else 1.0

    for triangle in triangles:
        surface_vertices = tuple((x, y, surface_height((x, y))) for x, y in triangle)
        opposite_vertices = tuple(
            (x, y, z + direction * thickness) for x, y, z in surface_vertices
        )
        if surface_is_top:
            surface_group.append(builder.add_triangle(*surface_vertices))
            opposite_group.append(
                builder.add_triangle(
                    opposite_vertices[0],
                    opposite_vertices[2],
                    opposite_vertices[1],
                )
            )
        else:
            surface_group.append(
                builder.add_triangle(
                    surface_vertices[0], surface_vertices[2], surface_vertices[1]
                )
            )
            opposite_group.append(builder.add_triangle(*opposite_vertices))

    for loop in boundary_loops:
        for start, end in zip(loop, loop[1:] + loop[:1]):
            start_surface = (*start, surface_height(start))
            end_surface = (*end, surface_height(end))
            start_opposite = (
                start_surface[0],
                start_surface[1],
                start_surface[2] + direction * thickness,
            )
            end_opposite = (
                end_surface[0],
                end_surface[1],
                end_surface[2] + direction * thickness,
            )
            if surface_is_top:
                side_group.extend(
                    (
                        builder.add_triangle(start_opposite, end_opposite, end_surface),
                        builder.add_triangle(
                            start_opposite, end_surface, start_surface
                        ),
                    )
                )
            else:
                side_group.extend(
                    (
                        builder.add_triangle(start_surface, end_surface, end_opposite),
                        builder.add_triangle(
                            start_surface, end_opposite, start_opposite
                        ),
                    )
                )


def compile_platform(platform: PlatformSpec) -> CompiledStructure:
    """Compile a raised/sunken platform, mezzanine, bridge, or catwalk slab."""

    triangles = _triangulate_footprint(platform.footprint)
    loops: tuple[Sequence[Point2], ...] = (
        platform.footprint.outer,
        *platform.footprint.holes,
    )
    builder = _MeshBuilder()
    mutable_groups: dict[str, list[int]] = {}

    def top_height(_: Point2) -> float:
        return platform.elevation

    _add_slab(
        builder=builder,
        triangles=triangles,
        boundary_loops=loops,
        surface_height=top_height,
        thickness=platform.thickness,
        surface_is_top=True,
        group_prefix="platform",
        groups=mutable_groups,
    )
    top_roles = {SurfaceRole.SUPPORT}
    if platform.traversable:
        top_roles.add(SurfaceRole.TRAVERSABLE)
    surfaces: list[CompiledSurfacePatch] = [
        CompiledSurfacePatch(
            surface=StructuralSurface(
                surface_id=f"{platform.platform_id}_top",
                roles=frozenset(top_roles),
                source_id=platform.platform_id,
                metadata={
                    "space_id": platform.space_id,
                    "holes": [
                        [list(point) for point in hole]
                        for hole in platform.footprint.holes
                    ],
                    "structure_type": "platform",
                },
            ),
            boundary=tuple(
                (point[0], point[1], platform.elevation)
                for point in platform.footprint.outer
            ),
            normal=(0.0, 0.0, 1.0),
        )
    ]
    surfaces.append(
        CompiledSurfacePatch(
            surface=StructuralSurface(
                surface_id=f"{platform.platform_id}_underside",
                roles=frozenset({SurfaceRole.OVERHEAD}),
                source_id=platform.platform_id,
                metadata={
                    "space_id": platform.space_id,
                    "holes": [
                        [list(point) for point in hole]
                        for hole in platform.footprint.holes
                    ],
                    "structure_type": "platform_underside",
                },
            ),
            boundary=tuple(
                (point[0], point[1], platform.elevation - platform.thickness)
                for point in reversed(platform.footprint.outer)
            ),
            normal=(0.0, 0.0, -1.0),
        )
    )
    for loop_index, loop in enumerate(loops):
        for edge_index, (start, end) in enumerate(zip(loop, loop[1:] + loop[:1])):
            roles = {SurfaceRole.BOUNDARY}
            if loop_index == 0 and edge_index in platform.open_edge_indices:
                roles.add(SurfaceRole.OPEN_EDGE)
            dx, dy = end[0] - start[0], end[1] - start[1]
            surfaces.append(
                CompiledSurfacePatch(
                    surface=StructuralSurface(
                        surface_id=(
                            f"{platform.platform_id}_side_{loop_index:02d}_"
                            f"{edge_index:03d}"
                        ),
                        roles=frozenset(roles),
                        source_id=platform.platform_id,
                        metadata={
                            "space_id": platform.space_id,
                            "loop_index": loop_index,
                            "edge_index": edge_index,
                            "structure_type": "platform_side",
                        },
                    ),
                    boundary=(
                        (start[0], start[1], platform.elevation - platform.thickness),
                        (end[0], end[1], platform.elevation - platform.thickness),
                        (end[0], end[1], platform.elevation),
                        (start[0], start[1], platform.elevation),
                    ),
                    normal=_normalize((dy, -dx, 0.0)),
                )
            )
    mesh = builder.build()
    return CompiledStructure(
        structure_id=platform.platform_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
        triangle_groups={
            name: tuple(indices) for name, indices in mutable_groups.items()
        },
    )


def compile_heightfield(
    heightfield: HeightfieldSpec,
    *,
    max_triangles: int = 250_000,
    max_traversable_slope_degrees: float = 35.0,
) -> CompiledStructure:
    """Compile a sampled terrain/floor grid into queryable surface triangles."""

    row_count, column_count = heightfield.shape
    triangle_count = (row_count - 1) * (column_count - 1) * 2
    if triangle_count > max_triangles:
        raise GeometryValidationError(
            "geometry_budget_exceeded",
            f"heightfield requires {triangle_count} triangles; budget is "
            f"{max_triangles}",
            entity_id=heightfield.heightfield_id,
        )
    origin_x, origin_y, origin_z = heightfield.origin
    cell_x, cell_y = heightfield.cell_size
    vertices = tuple(
        (
            origin_x + column_index * cell_x,
            origin_y + row_index * cell_y,
            origin_z + heightfield.heights[row_index][column_index],
        )
        for row_index in range(row_count)
        for column_index in range(column_count)
    )
    triangles: list[Triangle] = []
    for row_index in range(row_count - 1):
        for column_index in range(column_count - 1):
            lower_left = row_index * column_count + column_index
            lower_right = lower_left + 1
            upper_left = lower_left + column_count
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    mesh = TriangleMesh(vertices=vertices, triangles=tuple(triangles))
    groups: dict[str, list[int]] = {}
    surfaces: list[CompiledSurfacePatch] = []
    for triangle_index, triangle in enumerate(mesh.triangles):
        normal = mesh.triangle_normal(triangle_index)
        roles = _auto_surface_roles(
            normal, max_traversable_slope_degrees=max_traversable_slope_degrees
        )
        group_name = (
            "traversable" if SurfaceRole.TRAVERSABLE in roles else "steep_or_overhead"
        )
        groups.setdefault(group_name, []).append(triangle_index)
        surfaces.append(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=(
                        f"{heightfield.heightfield_id}_triangle_{triangle_index:06d}"
                    ),
                    roles=roles,
                    source_id=heightfield.heightfield_id,
                    geometry_ref=f"triangle:{triangle_index}",
                    metadata={
                        "space_id": heightfield.space_id,
                        "structure_type": "heightfield",
                    },
                ),
                boundary=tuple(mesh.vertices[index] for index in triangle),
                normal=normal,
            )
        )
    return CompiledStructure(
        structure_id=heightfield.heightfield_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
        triangle_groups={name: tuple(indices) for name, indices in groups.items()},
    )


def compile_polygon_space(
    *,
    structure_id: str,
    footprint: Footprint2D,
    floor_footprint: Footprint2D | None = None,
    ceiling_footprint: Footprint2D | None = None,
    floor_profile: ElevationProfile | None = None,
    ceiling_profile: ElevationProfile | None = None,
    wall_height: float = 2.5,
    floor_thickness: float = 0.1,
    ceiling_thickness: float = 0.1,
    include_floor: bool = True,
    include_ceiling: bool = True,
    portals: Sequence[PortalSpec] = (),
) -> CompiledStructure:
    """Compile a polygon-with-holes room shell with arbitrary boundary walls."""

    if not structure_id.strip():
        raise ValueError("structure_id must not be empty")
    if wall_height <= 0 or not math.isfinite(wall_height):
        raise ValueError("wall_height must be finite and positive")
    if floor_thickness <= 0 or not math.isfinite(floor_thickness):
        raise ValueError("floor_thickness must be finite and positive")
    if ceiling_thickness <= 0 or not math.isfinite(ceiling_thickness):
        raise ValueError("ceiling_thickness must be finite and positive")

    floor_profile = floor_profile or ElevationProfile()
    floor_footprint = floor_footprint or footprint
    ceiling_footprint = ceiling_footprint or footprint
    for label, slab_footprint in (
        ("floor", floor_footprint),
        ("ceiling", ceiling_footprint),
    ):
        if slab_footprint.outer != footprint.outer:
            raise GeometryValidationError(
                "slab_boundary_mismatch",
                f"{label} footprint outer loop must match the room boundary; "
                "use holes for slab openings",
                entity_id=structure_id,
            )
    floor_triangles = _triangulate_footprint(floor_footprint)
    ceiling_triangles = _triangulate_footprint(ceiling_footprint)
    loops: tuple[Sequence[Point2], ...] = (footprint.outer, *footprint.holes)
    floor_loops: tuple[Sequence[Point2], ...] = (
        floor_footprint.outer,
        *floor_footprint.holes,
    )
    ceiling_loops: tuple[Sequence[Point2], ...] = (
        ceiling_footprint.outer,
        *ceiling_footprint.holes,
    )
    portals_by_edge: dict[tuple[int, int], list[PortalSpec]] = {}
    for portal in portals:
        if portal.boundary_loop_index is None:
            continue
        loop_index = portal.boundary_loop_index
        edge_index = portal.boundary_edge_index
        assert edge_index is not None and portal.position_along is not None
        if loop_index >= len(loops) or edge_index >= len(loops[loop_index]):
            raise GeometryValidationError(
                "invalid_portal_boundary",
                f"boundary edge ({loop_index}, {edge_index}) does not exist",
                entity_id=portal.portal_id,
            )
        edge_start = loops[loop_index][edge_index]
        edge_end = loops[loop_index][(edge_index + 1) % len(loops[loop_index])]
        edge_length = math.dist(edge_start, edge_end)
        interval_start = portal.position_along - portal.width / 2.0
        interval_end = portal.position_along + portal.width / 2.0
        if (
            interval_start < -GEOMETRY_TOLERANCE
            or interval_end > edge_length + GEOMETRY_TOLERANCE
        ):
            raise GeometryValidationError(
                "portal_outside_boundary",
                f"opening interval [{interval_start:g}, {interval_end:g}] exceeds "
                f"edge length {edge_length:g}",
                entity_id=portal.portal_id,
            )
        portals_by_edge.setdefault((loop_index, edge_index), []).append(portal)
    for edge_portals in portals_by_edge.values():
        edge_portals.sort(key=lambda portal: portal.position_along or 0.0)
        for first, second in zip(edge_portals, edge_portals[1:]):
            assert (
                first.position_along is not None and second.position_along is not None
            )
            if (
                first.position_along + first.width / 2.0
                > second.position_along - second.width / 2.0 + GEOMETRY_TOLERANCE
            ):
                raise GeometryValidationError(
                    "overlapping_portals",
                    f"portals '{first.portal_id}' and '{second.portal_id}' overlap",
                    entity_id=second.portal_id,
                )

    def floor_height(point: Point2) -> float:
        return _profile_height(floor_profile, point)

    if ceiling_profile is None:

        def ceiling_height(point: Point2) -> float:
            return floor_height(point) + wall_height

    else:

        def ceiling_height(point: Point2) -> float:
            return _profile_height(ceiling_profile, point)

    clearance_points = {
        point for loop in (*loops, *floor_loops, *ceiling_loops) for point in loop
    }
    for point in clearance_points:
        clearance = ceiling_height(point) - floor_height(point)
        if clearance <= GEOMETRY_TOLERANCE:
            raise ValueError(
                f"ceiling must be above floor at {point}; clearance={clearance}"
            )

    builder = _MeshBuilder()
    mutable_groups: dict[str, list[int]] = {}
    if include_floor:
        _add_slab(
            builder=builder,
            triangles=floor_triangles,
            boundary_loops=floor_loops,
            surface_height=floor_height,
            thickness=floor_thickness,
            surface_is_top=True,
            group_prefix="floor",
            groups=mutable_groups,
        )
    if include_ceiling:
        _add_slab(
            builder=builder,
            triangles=ceiling_triangles,
            boundary_loops=ceiling_loops,
            surface_height=ceiling_height,
            thickness=ceiling_thickness,
            surface_is_top=False,
            group_prefix="ceiling",
            groups=mutable_groups,
        )

    surfaces: list[CompiledSurfacePatch] = []
    floor_normal = _normalize(
        (-floor_profile.gradient[0], -floor_profile.gradient[1], 1.0)
    )
    floor_surface = CompiledSurfacePatch(
        surface=StructuralSurface(
            surface_id=f"{structure_id}_floor",
            roles=frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
            source_id=structure_id,
            metadata={
                "holes": [
                    [list(point) for point in hole] for hole in floor_footprint.holes
                ],
                "profile": floor_profile.to_dict(),
            },
        ),
        boundary=tuple(
            (*point, floor_height(point)) for point in floor_footprint.outer
        ),
        normal=floor_normal,
    )
    if include_floor:
        surfaces.append(floor_surface)

    ceiling_gradient = (
        ceiling_profile.gradient
        if ceiling_profile is not None
        else floor_profile.gradient
    )
    ceiling_surface = CompiledSurfacePatch(
        surface=StructuralSurface(
            surface_id=f"{structure_id}_ceiling",
            roles=frozenset({SurfaceRole.OVERHEAD, SurfaceRole.ATTACHMENT}),
            source_id=structure_id,
            metadata={
                "holes": [
                    [list(point) for point in hole] for hole in ceiling_footprint.holes
                ],
                "profile": (
                    ceiling_profile.to_dict()
                    if ceiling_profile is not None
                    else {"derived_from_floor": True, "wall_height": wall_height}
                ),
            },
        ),
        boundary=tuple(
            (*point, ceiling_height(point)) for point in ceiling_footprint.outer
        ),
        normal=_normalize((ceiling_gradient[0], ceiling_gradient[1], -1.0)),
    )
    if include_ceiling:
        surfaces.append(ceiling_surface)
    floor_underside = CompiledSurfacePatch(
        surface=StructuralSurface(
            surface_id=f"{structure_id}_floor_underside",
            roles=frozenset({SurfaceRole.OVERHEAD}),
            source_id=structure_id,
            metadata={
                "holes": [
                    [list(point) for point in hole] for hole in floor_footprint.holes
                ],
                "derived_from": f"{structure_id}_floor",
            },
        ),
        boundary=tuple(
            (*point, floor_height(point) - floor_thickness)
            for point in floor_footprint.outer
        ),
        normal=tuple(-component for component in floor_normal),
    )
    if include_floor:
        surfaces.append(floor_underside)

    wall_group = mutable_groups.setdefault("walls", [])
    for loop_index, loop in enumerate(loops):
        loop_kind = "outer" if loop_index == 0 else f"hole_{loop_index - 1}"
        for edge_index, (start, end) in enumerate(zip(loop, loop[1:] + loop[:1])):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            edge_length = math.hypot(dx, dy)
            inward = _normalize((-dy, dx, 0.0))
            edge_portals = portals_by_edge.get((loop_index, edge_index), [])
            cuts = {0.0, edge_length}
            for portal in edge_portals:
                assert portal.position_along is not None
                cuts.update(
                    {
                        portal.position_along - portal.width / 2.0,
                        portal.position_along + portal.width / 2.0,
                    }
                )
            sorted_cuts = sorted(cuts)

            def edge_point(distance: float) -> Point2:
                ratio = distance / edge_length
                return (start[0] + dx * ratio, start[1] + dy * ratio)

            panel_index = 0
            for distance_start, distance_end in zip(sorted_cuts, sorted_cuts[1:]):
                midpoint = (distance_start + distance_end) / 2.0
                opening = next(
                    (
                        portal
                        for portal in edge_portals
                        if portal.position_along is not None
                        and portal.position_along - portal.width / 2.0
                        < midpoint
                        < portal.position_along + portal.width / 2.0
                    ),
                    None,
                )
                segment_start = edge_point(distance_start)
                segment_end = edge_point(distance_end)
                floor_start = floor_height(segment_start)
                floor_end = floor_height(segment_end)
                ceiling_start = ceiling_height(segment_start)
                ceiling_end = ceiling_height(segment_end)
                vertical_bands = [(floor_start, floor_end, ceiling_start, ceiling_end)]
                if opening is not None:
                    opening_bottom_start = floor_start + opening.sill_height
                    opening_bottom_end = floor_end + opening.sill_height
                    opening_top_start = opening_bottom_start + opening.height
                    opening_top_end = opening_bottom_end + opening.height
                    if (
                        opening_top_start > ceiling_start + GEOMETRY_TOLERANCE
                        or opening_top_end > ceiling_end + GEOMETRY_TOLERANCE
                    ):
                        raise GeometryValidationError(
                            "portal_outside_boundary",
                            "opening height exceeds the local floor-to-ceiling clearance",
                            entity_id=opening.portal_id,
                        )
                    vertical_bands = []
                    if opening.sill_height > GEOMETRY_TOLERANCE:
                        vertical_bands.append(
                            (
                                floor_start,
                                floor_end,
                                opening_bottom_start,
                                opening_bottom_end,
                            )
                        )
                    if (
                        opening_top_start < ceiling_start - GEOMETRY_TOLERANCE
                        or opening_top_end < ceiling_end - GEOMETRY_TOLERANCE
                    ):
                        vertical_bands.append(
                            (
                                opening_top_start,
                                opening_top_end,
                                ceiling_start,
                                ceiling_end,
                            )
                        )
                for lower_start, lower_end, upper_start, upper_end in vertical_bands:
                    start_lower = (*segment_start, lower_start)
                    end_lower = (*segment_end, lower_end)
                    start_upper = (*segment_start, upper_start)
                    end_upper = (*segment_end, upper_end)
                    wall_group.extend(
                        (
                            builder.add_triangle(start_lower, end_lower, end_upper),
                            builder.add_triangle(start_lower, end_upper, start_upper),
                        )
                    )
                    surface_id = f"{structure_id}_{loop_kind}_wall_{edge_index:03d}"
                    if edge_portals:
                        surface_id += f"_panel_{panel_index:03d}"
                    surfaces.append(
                        CompiledSurfacePatch(
                            surface=StructuralSurface(
                                surface_id=surface_id,
                                roles=frozenset(
                                    {SurfaceRole.BOUNDARY, SurfaceRole.ATTACHMENT}
                                ),
                                source_id=structure_id,
                                metadata={
                                    "loop_kind": loop_kind,
                                    "edge_index": edge_index,
                                    "adjacent_portal_id": (
                                        opening.portal_id if opening else None
                                    ),
                                },
                            ),
                            boundary=(
                                end_lower,
                                start_lower,
                                start_upper,
                                end_upper,
                            ),
                            normal=inward,
                        )
                    )
                    panel_index += 1

    mesh = builder.build()
    return CompiledStructure(
        structure_id=structure_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
        triangle_groups={
            group_name: tuple(indices) for group_name, indices in mutable_groups.items()
        },
    )


def combine_meshes(meshes: Iterable[TriangleMesh]) -> TriangleMesh:
    """Combine meshes without welding, preserving deterministic index order."""

    vertices: list[Point3] = []
    triangles: list[Triangle] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        triangles.extend(
            (a + offset, b + offset, c + offset) for a, b, c in mesh.triangles
        )
    return TriangleMesh(tuple(vertices), tuple(triangles))


def _rotate_rpy(point: Point3, rotation_rpy: Point3) -> Point3:
    """Apply intrinsic XYZ roll/pitch/yaw as Rz * Ry * Rx."""

    roll, pitch, yaw = rotation_rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y, z = point
    roll_point = (x, cr * y - sr * z, sr * y + cr * z)
    pitch_point = (
        cp * roll_point[0] + sp * roll_point[2],
        roll_point[1],
        -sp * roll_point[0] + cp * roll_point[2],
    )
    return (
        cy * pitch_point[0] - sy * pitch_point[1],
        sy * pitch_point[0] + cy * pitch_point[1],
        pitch_point[2],
    )


def _transform_mesh_vertex(vertex: Sequence[float], spec: StructuralMeshSpec) -> Point3:
    scaled = tuple(float(value) * spec.unit_scale for value in vertex)
    rotated = _rotate_rpy(scaled, spec.transform.rotation_rpy)  # type: ignore[arg-type]
    return tuple(
        rotated[axis] + spec.transform.translation[axis] for axis in range(3)
    )  # type: ignore[return-value]


def _auto_surface_roles(
    normal: Point3, *, max_traversable_slope_degrees: float
) -> frozenset[SurfaceRole]:
    threshold = math.cos(math.radians(max_traversable_slope_degrees))
    if normal[2] >= threshold:
        return frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE})
    if normal[2] <= -threshold:
        return frozenset({SurfaceRole.OVERHEAD, SurfaceRole.ATTACHMENT})
    return frozenset({SurfaceRole.BOUNDARY, SurfaceRole.ATTACHMENT})


def compile_structural_mesh(
    spec: StructuralMeshSpec,
    *,
    repair: bool = False,
    max_triangles: int = 250_000,
    max_traversable_slope_degrees: float = 35.0,
) -> CompiledStructure:
    """Validate and compile an imported cavern/freeform structural mesh.

    Authored annotations take precedence. Unannotated triangles are classified
    from their transformed normals so every collision triangle has explicit
    semantics. Repair mode only removes duplicate/degenerate faces and fixes
    winding for watertight meshes; it never guesses units or transforms.
    """

    if max_triangles <= 0:
        raise ValueError("max_triangles must be positive")
    if not 0.0 < max_traversable_slope_degrees < 90.0:
        raise ValueError("max_traversable_slope_degrees must be between 0 and 90")
    mesh_path = Path(spec.mesh_path)
    if not mesh_path.is_file():
        raise GeometryValidationError(
            "missing_mesh_file",
            f"mesh file does not exist: {mesh_path}",
            entity_id=spec.mesh_id,
        )
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise UnsupportedGeometryError(
            "freeform structural compilation requires trimesh and numpy",
            entity_id=spec.mesh_id,
        ) from exc

    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise GeometryValidationError(
            "invalid_mesh_file",
            "file did not contain a triangle mesh",
            entity_id=spec.mesh_id,
        )
    vertices = np.asarray(loaded.vertices, dtype=float)
    faces = np.asarray(loaded.faces, dtype=int)
    if len(vertices) == 0 or len(faces) == 0:
        raise GeometryValidationError(
            "empty_mesh", "mesh has no vertices or faces", entity_id=spec.mesh_id
        )
    if len(faces) > max_triangles:
        raise GeometryValidationError(
            "geometry_budget_exceeded",
            f"mesh has {len(faces)} triangles; budget is {max_triangles}",
            entity_id=spec.mesh_id,
        )
    if not np.isfinite(vertices).all():
        raise GeometryValidationError(
            "nonfinite_mesh",
            "mesh contains NaN or Inf vertices",
            entity_id=spec.mesh_id,
        )

    working = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    original_triangle_count = len(faces)
    face_index_map = {index: index for index in range(original_triangle_count)}
    sorted_faces = np.sort(np.asarray(working.faces), axis=1)
    _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
    duplicate_count = len(faces) - len(unique_indices)
    areas = np.asarray(working.area_faces)
    degenerate_mask = ~np.isfinite(areas) | (areas <= GEOMETRY_TOLERANCE)
    if duplicate_count or degenerate_mask.any():
        if not repair:
            raise GeometryValidationError(
                "invalid_mesh_topology",
                f"mesh has {duplicate_count} duplicate and "
                f"{int(degenerate_mask.sum())} degenerate faces",
                entity_id=spec.mesh_id,
            )
        keep = np.zeros(len(faces), dtype=bool)
        keep[unique_indices] = True
        keep &= ~degenerate_mask
        face_index_map = {
            int(original_index): new_index
            for new_index, original_index in enumerate(np.flatnonzero(keep))
        }
        working.update_faces(keep)
        working.remove_unreferenced_vertices()

    if spec.require_watertight and not working.is_watertight:
        raise GeometryValidationError(
            "nonmanifold_mesh",
            "mesh is not watertight/manifold but require_watertight is true",
            entity_id=spec.mesh_id,
        )
    if working.is_watertight and spec.normal_orientation != "unspecified":
        actual_orientation = "exterior" if working.volume >= 0 else "interior"
        if actual_orientation != spec.normal_orientation:
            if not repair:
                raise GeometryValidationError(
                    "inverted_mesh_normals",
                    f"mesh winding is {actual_orientation}, expected "
                    f"{spec.normal_orientation}",
                    entity_id=spec.mesh_id,
                )
            working.invert()

    transformed_mesh = TriangleMesh(
        vertices=tuple(
            _transform_mesh_vertex(vertex, spec) for vertex in working.vertices
        ),
        triangles=tuple(tuple(int(index) for index in face) for face in working.faces),
    )
    triangle_count = len(transformed_mesh.triangles)
    roles_by_triangle: dict[int, frozenset[SurfaceRole]] = {}
    annotation_by_triangle: dict[int, str] = {}
    for annotation in spec.annotations:
        for authored_triangle_index in annotation.triangle_indices:
            if authored_triangle_index >= original_triangle_count:
                raise GeometryValidationError(
                    "invalid_mesh_annotation",
                    f"triangle index {authored_triangle_index} is outside "
                    f"[0, {original_triangle_count})",
                    entity_id=annotation.annotation_id,
                )
            if authored_triangle_index not in face_index_map:
                raise GeometryValidationError(
                    "removed_mesh_annotation",
                    f"annotated triangle {authored_triangle_index} was removed "
                    "during repair",
                    entity_id=annotation.annotation_id,
                )
            triangle_index = face_index_map[authored_triangle_index]
            if triangle_index in roles_by_triangle:
                raise GeometryValidationError(
                    "overlapping_mesh_annotation",
                    f"triangle {triangle_index} is assigned by both "
                    f"'{annotation_by_triangle[triangle_index]}' and "
                    f"'{annotation.annotation_id}'",
                    entity_id=spec.mesh_id,
                )
            roles_by_triangle[triangle_index] = annotation.roles
            annotation_by_triangle[triangle_index] = annotation.annotation_id

    groups: dict[str, list[int]] = {}
    surfaces: list[CompiledSurfacePatch] = []
    for triangle_index, triangle in enumerate(transformed_mesh.triangles):
        normal = transformed_mesh.triangle_normal(triangle_index)
        roles = roles_by_triangle.get(triangle_index)
        annotation_id = annotation_by_triangle.get(triangle_index)
        if roles is None:
            roles = _auto_surface_roles(
                normal,
                max_traversable_slope_degrees=max_traversable_slope_degrees,
            )
            annotation_id = "auto_" + "_".join(sorted(role.value for role in roles))
        assert annotation_id is not None
        groups.setdefault(annotation_id, []).append(triangle_index)
        surfaces.append(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=f"{spec.mesh_id}_{annotation_id}_{triangle_index:06d}",
                    roles=roles,
                    source_id=spec.mesh_id,
                    geometry_ref=f"triangle:{triangle_index}",
                    metadata={
                        "space_id": spec.space_id,
                        "annotation_id": annotation_id,
                        "auto_classified": triangle_index not in roles_by_triangle,
                    },
                ),
                boundary=tuple(transformed_mesh.vertices[index] for index in triangle),
                normal=normal,
            )
        )

    return CompiledStructure(
        structure_id=spec.mesh_id,
        visual_mesh=transformed_mesh,
        collision_mesh=transformed_mesh,
        surfaces=tuple(surfaces),
        triangle_groups={name: tuple(indices) for name, indices in groups.items()},
    )


def _add_mesh_geometry(parent: ET.Element, mesh_name: str) -> None:
    geometry = ET.SubElement(parent, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "uri").text = mesh_name


def write_compiled_structure(
    compiled: CompiledStructure,
    output_dir: Path | str,
    *,
    model_name: str | None = None,
    link_name: str = "structure_link",
) -> CompiledStructurePaths:
    """Write OBJ, static SDF, and semantic-surface sidecar files.

    Collision boxes are kept analytic for stairs.  Structures without analytic
    primitives (currently ramps) use the closed collision triangle mesh.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mesh_path = output_path / f"{compiled.structure_id}.obj"
    sdf_path = output_path / f"{compiled.structure_id}.sdf"
    surfaces_path = output_path / f"{compiled.structure_id}.surfaces.json"

    mesh_path.write_text(
        compiled.visual_mesh.to_obj(object_name=compiled.structure_id),
        encoding="utf-8",
    )

    sdf = ET.Element("sdf", {"version": "1.12"})
    model = ET.SubElement(sdf, "model", {"name": model_name or compiled.structure_id})
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", {"name": link_name})
    visual = ET.SubElement(link, "visual", {"name": "structure_visual"})
    _add_mesh_geometry(visual, mesh_path.name)

    if compiled.collision_primitives:
        for primitive in compiled.collision_primitives:
            collision = ET.SubElement(
                link, "collision", {"name": primitive.primitive_id}
            )
            tx, ty, tz = primitive.transform.translation
            roll, pitch, yaw = primitive.transform.rotation_rpy
            ET.SubElement(collision, "pose").text = (
                f"{tx:.12g} {ty:.12g} {tz:.12g} " f"{roll:.12g} {pitch:.12g} {yaw:.12g}"
            )
            geometry = ET.SubElement(collision, "geometry")
            if primitive.primitive_type != "box":
                raise UnsupportedGeometryError(
                    f"SDF export does not support collision primitive "
                    f"'{primitive.primitive_type}'",
                    entity_id=primitive.primitive_id,
                )
            box = ET.SubElement(geometry, "box")
            ET.SubElement(box, "size").text = " ".join(
                f"{value:.12g}" for value in primitive.dimensions
            )
    else:
        collision = ET.SubElement(link, "collision", {"name": "structure_collision"})
        _add_mesh_geometry(collision, mesh_path.name)

    ET.indent(sdf, space="  ")
    ET.ElementTree(sdf).write(sdf_path, encoding="utf-8", xml_declaration=True)

    surface_data: dict[str, object] = {
        "schema_version": 1,
        "structure_id": compiled.structure_id,
        "mesh": mesh_path.name,
        "bounds": [list(bound) for bound in compiled.visual_mesh.bounds],
        "visual_triangles": len(compiled.visual_mesh.triangles),
        "collision_triangles": len(compiled.collision_mesh.triangles),
        "triangle_groups": {
            group_name: list(indices)
            for group_name, indices in compiled.triangle_groups.items()
        },
    }
    triangle_group_by_index = {
        triangle_index: group_name
        for group_name, indices in compiled.triangle_groups.items()
        for triangle_index in indices
    }
    compact_triangle_surfaces = len(compiled.surfaces) == len(
        compiled.visual_mesh.triangles
    ) and all(
        patch.surface.source_id == compiled.structure_id
        and patch.surface.transform == Transform3D()
        and patch.surface.geometry_ref == f"triangle:{triangle_index}"
        and patch.boundary
        == tuple(
            compiled.visual_mesh.vertices[index]
            for index in compiled.visual_mesh.triangles[triangle_index]
        )
        and triangle_group_by_index.get(triangle_index) is not None
        and patch.surface.surface_id
        == (
            f"{compiled.structure_id}_"
            f"{triangle_group_by_index[triangle_index]}_{triangle_index:06d}"
        )
        for triangle_index, patch in enumerate(compiled.surfaces)
    )
    if compact_triangle_surfaces:
        metadata_runs: list[dict[str, object]] = []
        for triangle_index, patch in enumerate(compiled.surfaces):
            metadata = dict(patch.surface.metadata)
            if metadata_runs and metadata_runs[-1]["metadata"] == metadata:
                metadata_runs[-1]["end"] = triangle_index + 1
            else:
                metadata_runs.append(
                    {
                        "start": triangle_index,
                        "end": triangle_index + 1,
                        "metadata": metadata,
                    }
                )
        surface_data.update(
            {
                "schema_version": 2,
                "surface_encoding": "triangle_mesh_v1",
                "surface_mesh": {
                    "vertices": [
                        list(vertex) for vertex in compiled.visual_mesh.vertices
                    ],
                    "triangles": [
                        list(triangle) for triangle in compiled.visual_mesh.triangles
                    ],
                },
                "surface_roles": {
                    group_name: sorted(
                        role.value
                        for role in compiled.surfaces[indices[0]].surface.roles
                    )
                    for group_name, indices in compiled.triangle_groups.items()
                    if indices
                },
                "surface_metadata_runs": metadata_runs,
            }
        )
    else:
        surface_data["surfaces"] = [
            {
                **patch.surface.to_dict(),
                "boundary": [list(point) for point in patch.boundary],
                "normal": list(patch.normal),
            }
            for patch in compiled.surfaces
        ]
    surfaces_path.write_text(
        json.dumps(surface_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CompiledStructurePaths(
        mesh_path=mesh_path,
        sdf_path=sdf_path,
        surfaces_path=surfaces_path,
    )
