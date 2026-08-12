"""Deterministic implicit-union compiler for semantic environment voids.

Passage sweeps and chambers are evaluated as one continuous empty volume and
extracted with a dependency-light marching-tetrahedra backend.  This makes
branch junctions and chamber joins physical unions rather than overlapping
props, while preserving semantic provenance on every output triangle.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Callable, Iterable

from scenesmith.agent_utils.semantic_environments import (
    CavernChamberSpec,
    CavernShape,
    EnvironmentOpeningSpec,
    EnvironmentRegionSpec,
    OpeningShape,
    PassageFloorMode,
    PassageNetworkSpec,
    PassageProfile,
    PassageSegmentSpec,
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structural_compiler import (
    CompiledStructure,
    CompiledSurfacePatch,
    Triangle,
    TriangleMesh,
)
from scenesmith.agent_utils.structural_geometry import (
    GEOMETRY_TOLERANCE,
    GeometryValidationError,
    MeshSurfaceAnnotation,
    Point3,
    StructuralSurface,
    SurfaceRole,
    UnsupportedGeometryError,
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
class SemanticCompileOptions:
    """Derived-geometry controls that are intentionally outside scene recipes."""

    voxel_size: float = 0.5
    padding_voxels: int = 2
    max_cells: int = 2_000_000
    max_triangles: int = 500_000
    max_traversable_slope_degrees: float = 35.0
    structure_id: str = "semantic_environment"

    def __post_init__(self) -> None:
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be finite and positive")
        if self.padding_voxels < 1:
            raise ValueError("padding_voxels must be at least 1")
        if self.max_cells <= 0 or self.max_triangles <= 0:
            raise ValueError("geometry budgets must be positive")
        if not 0.0 < self.max_traversable_slope_degrees < 90.0:
            raise ValueError("max_traversable_slope_degrees must be between 0 and 90")
        if not str(self.structure_id).strip():
            raise ValueError("structure_id must not be empty")


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
    raise AssertionError(profile)


def _passage_primitives(
    network: PassageNetworkSpec, region: EnvironmentRegionSpec
) -> Iterable[_ImplicitPrimitive]:
    junction_by_id = {item.junction_id: item for item in network.junctions}
    region_up = _normalize(_rotate_rpy((0.0, 0.0, 1.0), region.transform.rotation_rpy))
    for segment in network.segments:
        if segment.floor_mode == PassageFloorMode.STEPS:
            raise UnsupportedGeometryError(
                "semantic compiler does not yet support stepped passage floors",
                entity_id=segment.segment_id,
            )
        path = tuple(_transform_point(point, region) for point in segment.path)
        span_lengths = tuple(
            math.dist(first, second) for first, second in zip(path, path[1:])
        )
        total_length = sum(span_lengths)
        cumulative = 0.0
        start_open = junction_by_id[segment.start_junction_id].open_boundary
        end_open = junction_by_id[segment.end_junction_id].open_boundary
        maximum_extent = max(
            max(section.width / 2.0, section.height)
            for section in segment.cross_sections
        )
        for index, (start, end, span_length) in enumerate(
            zip(path, path[1:], span_lengths)
        ):
            tangent = _normalize(_subtract(end, start))
            across_raw = _cross(region_up, tangent)
            if _length(across_raw) <= GEOMETRY_TOLERANCE:
                fallback = _rotate_rpy((1.0, 0.0, 0.0), region.transform.rotation_rpy)
                across_raw = _cross(fallback, tangent)
            across = _normalize(across_raw)
            vertical = _normalize(_cross(tangent, across))
            if _dot(vertical, region_up) < 0.0:
                across = _scale(across, -1.0)
                vertical = _scale(vertical, -1.0)
            station_start = cumulative / total_length
            cumulative += span_length
            station_end = cumulative / total_length
            span_minimum = tuple(
                min(start[axis], end[axis]) - maximum_extent for axis in range(3)
            )
            span_maximum = tuple(
                max(start[axis], end[axis]) + maximum_extent for axis in range(3)
            )

            def evaluate(
                point: Point3,
                *,
                start: Point3 = start,
                tangent: Point3 = tangent,
                across: Point3 = across,
                vertical: Point3 = vertical,
                span_length: float = span_length,
                station_start: float = station_start,
                station_end: float = station_end,
                segment: PassageSegmentSpec = segment,
            ) -> float:
                offset = _subtract(point, start)
                raw_distance = _dot(offset, tangent)
                clamped_distance = min(span_length, max(0.0, raw_distance))
                local_amount = min(1.0, max(0.0, clamped_distance / span_length))
                station = station_start + (station_end - station_start) * local_amount
                width, height = _interpolate_cross_section(segment, station)
                floor_center = _add(start, _scale(tangent, clamped_distance))
                section_center = _add(floor_center, _scale(vertical, height / 2.0))
                relative = _subtract(point, section_center)
                across_distance = _dot(relative, across) / (width / 2.0)
                vertical_distance = _dot(relative, vertical) / (height / 2.0)
                cap_distance = raw_distance - clamped_distance
                cap_radius = min(width, height) / 2.0
                return (
                    _passage_profile_value(
                        segment.profile, across_distance, vertical_distance
                    )
                    + (cap_distance / cap_radius) ** 2
                )

            yield _ImplicitPrimitive(
                source_id=segment.segment_id,
                source_kind="passage_segment",
                floor_mode=segment.floor_mode,
                minimum=span_minimum,
                maximum=span_maximum,
                evaluate=evaluate,
                open_planes=tuple(
                    plane
                    for plane in (
                        (start, tangent) if index == 0 and start_open else None,
                        (
                            (end, _scale(tangent, -1.0))
                            if index == len(span_lengths) - 1 and end_open
                            else None
                        ),
                    )
                    if plane is not None
                ),
            )


def _chamber_primitive(
    chamber: CavernChamberSpec, region: EnvironmentRegionSpec
) -> _ImplicitPrimitive:
    if chamber.shape not in {CavernShape.ELLIPSOID, CavernShape.SUPERELLIPSOID}:
        raise UnsupportedGeometryError(
            f"semantic compiler does not yet support chamber shape "
            f"'{chamber.shape.value}'",
            entity_id=chamber.chamber_id,
        )
    center = _transform_point(chamber.center, region)
    local_axes = tuple(
        _rotate_rpy(axis, chamber.orientation_rpy)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    axes = tuple(
        _normalize(_rotate_rpy(axis, region.transform.rotation_rpy))
        for axis in local_axes
    )
    half_size = tuple(component / 2.0 for component in chamber.size)
    extents = tuple(
        sum(
            abs(axes[local_axis][world_axis]) * half_size[local_axis]
            for local_axis in range(3)
        )
        for world_axis in range(3)
    )
    minimum = tuple(center[axis] - extents[axis] for axis in range(3))
    maximum = tuple(center[axis] + extents[axis] for axis in range(3))

    def evaluate(point: Point3) -> float:
        offset = _subtract(point, center)
        exponent = 2 if chamber.shape == CavernShape.ELLIPSOID else 4
        return (
            sum(
                abs(_dot(offset, axes[axis]) / half_size[axis]) ** exponent
                for axis in range(3)
            )
            - 1.0
        )

    return _ImplicitPrimitive(
        source_id=chamber.chamber_id,
        source_kind="cavern_chamber",
        floor_mode=None,
        minimum=minimum,  # type: ignore[arg-type]
        maximum=maximum,  # type: ignore[arg-type]
        evaluate=evaluate,
    )


def _build_primitives(
    environment: SemanticEnvironmentSpec,
) -> tuple[_ImplicitPrimitive, ...]:
    region_by_id = {item.region_id: item for item in environment.regions}
    primitives: list[_ImplicitPrimitive] = []
    for chamber in environment.chambers:
        primitives.append(_chamber_primitive(chamber, region_by_id[chamber.region_id]))
    for network in environment.passage_networks:
        primitives.extend(_passage_primitives(network, region_by_id[network.region_id]))
    for opening in environment.openings:
        primitives.append(_opening_primitive(opening, region_by_id[opening.region_id]))
    if not primitives:
        raise GeometryValidationError(
            "empty_environment_geometry",
            "environment has no chamber or passage geometry to compile",
        )
    return tuple(
        sorted(primitives, key=lambda item: (item.source_id, item.source_kind))
    )


def _union_value(primitives: tuple[_ImplicitPrimitive, ...], point: Point3) -> float:
    candidates = (
        primitive
        for primitive in primitives
        if all(
            primitive.minimum[axis] - 1e-8
            <= point[axis]
            <= primitive.maximum[axis] + 1e-8
            for axis in range(3)
        )
    )
    return min((primitive.evaluate(point) for primitive in candidates), default=1.0)


def _nearest_primitive(
    primitives: tuple[_ImplicitPrimitive, ...], point: Point3
) -> _ImplicitPrimitive:
    candidates = tuple(
        primitive
        for primitive in primitives
        if all(
            primitive.minimum[axis] - 1e-8
            <= point[axis]
            <= primitive.maximum[axis] + 1e-8
            for axis in range(3)
        )
    )
    return min(
        candidates or primitives,
        key=lambda item: (item.evaluate(point), item.source_id),
    )


_CUBE_OFFSETS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)
_CUBE_TETRAHEDRA = (
    (0, 5, 1, 6),
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
)


def _triangle_orientation(
    triangle: Triangle,
    vertices: list[Point3],
    inside_hint: Point3,
    outside_hint: Point3,
) -> Triangle:
    a, b, c = (vertices[index] for index in triangle)
    normal = _cross(_subtract(b, a), _subtract(c, a))
    toward_inside = _subtract(inside_hint, outside_hint)
    if _dot(normal, toward_inside) < 0.0:
        return (triangle[0], triangle[2], triangle[1])
    return triangle


def _extract_mesh(
    primitives: tuple[_ImplicitPrimitive, ...],
    options: SemanticCompileOptions,
) -> TriangleMesh:
    padding = options.voxel_size * options.padding_voxels
    geometry_minimum = tuple(
        min(primitive.minimum[axis] for primitive in primitives) for axis in range(3)
    )
    geometry_maximum = tuple(
        max(primitive.maximum[axis] for primitive in primitives) for axis in range(3)
    )
    grid_minimum = tuple(component - padding for component in geometry_minimum)
    cell_counts = tuple(
        max(
            2,
            math.ceil(
                (geometry_maximum[axis] - geometry_minimum[axis] + 2.0 * padding)
                / options.voxel_size
            ),
        )
        for axis in range(3)
    )
    cell_total = math.prod(cell_counts)
    if cell_total > options.max_cells:
        raise GeometryValidationError(
            "geometry_budget_exceeded",
            f"implicit grid needs {cell_total} cells; budget is {options.max_cells}",
            entity_id=options.structure_id,
        )
    point_counts = tuple(count + 1 for count in cell_counts)

    def grid_index(x: int, y: int, z: int) -> int:
        return (z * point_counts[1] + y) * point_counts[0] + x

    def grid_point(x: int, y: int, z: int) -> Point3:
        return (
            grid_minimum[0] + x * options.voxel_size,
            grid_minimum[1] + y * options.voxel_size,
            grid_minimum[2] + z * options.voxel_size,
        )

    values: list[float] = [0.0] * math.prod(point_counts)
    points: list[Point3] = [(0.0, 0.0, 0.0)] * math.prod(point_counts)
    for z in range(point_counts[2]):
        for y in range(point_counts[1]):
            for x in range(point_counts[0]):
                index = grid_index(x, y, z)
                point = grid_point(x, y, z)
                points[index] = point
                values[index] = _union_value(primitives, point)

    vertices: list[Point3] = []
    triangles: list[Triangle] = []
    edge_vertices: dict[tuple[int, int], int] = {}

    def intersection(first: int, second: int) -> int:
        key = tuple(sorted((first, second)))
        if key in edge_vertices:
            return edge_vertices[key]
        first_value, second_value = values[first], values[second]
        denominator = first_value - second_value
        amount = (
            0.5 if abs(denominator) <= GEOMETRY_TOLERANCE else first_value / denominator
        )
        amount = min(1.0, max(0.0, amount))
        point = _add(
            points[first], _scale(_subtract(points[second], points[first]), amount)
        )
        edge_vertices[key] = len(vertices)
        vertices.append(point)
        return len(vertices) - 1

    def append_triangle(
        triangle: Triangle, inside_indices: list[int], outside_indices: list[int]
    ) -> None:
        if len(set(triangle)) != 3:
            return
        a, b, c = (vertices[index] for index in triangle)
        if _length(_cross(_subtract(b, a), _subtract(c, a))) <= GEOMETRY_TOLERANCE:
            return
        inside_hint = tuple(
            sum(points[index][axis] for index in inside_indices) / len(inside_indices)
            for axis in range(3)
        )
        outside_hint = tuple(
            sum(points[index][axis] for index in outside_indices) / len(outside_indices)
            for axis in range(3)
        )
        triangles.append(
            _triangle_orientation(
                triangle, vertices, inside_hint, outside_hint  # type: ignore[arg-type]
            )
        )
        if len(triangles) > options.max_triangles:
            raise GeometryValidationError(
                "geometry_budget_exceeded",
                f"surface extraction exceeded {options.max_triangles} triangles",
                entity_id=options.structure_id,
            )

    for z in range(cell_counts[2]):
        for y in range(cell_counts[1]):
            for x in range(cell_counts[0]):
                cube = tuple(
                    grid_index(x + dx, y + dy, z + dz) for dx, dy, dz in _CUBE_OFFSETS
                )
                for tetrahedron in _CUBE_TETRAHEDRA:
                    tetra = [cube[index] for index in tetrahedron]
                    inside = [index for index in tetra if values[index] <= 0.0]
                    outside = [index for index in tetra if values[index] > 0.0]
                    if not inside or not outside:
                        continue
                    if len(inside) == 1:
                        point_indices = tuple(
                            intersection(inside[0], item) for item in outside
                        )
                        append_triangle(point_indices, inside, outside)  # type: ignore[arg-type]
                    elif len(inside) == 3:
                        point_indices = tuple(
                            intersection(item, outside[0]) for item in inside
                        )
                        append_triangle(point_indices, inside, outside)  # type: ignore[arg-type]
                    else:
                        first_inside, second_inside = inside
                        first_outside, second_outside = outside
                        p00 = intersection(first_inside, first_outside)
                        p01 = intersection(first_inside, second_outside)
                        p10 = intersection(second_inside, first_outside)
                        p11 = intersection(second_inside, second_outside)
                        append_triangle((p00, p01, p11), inside, outside)
                        append_triangle((p00, p11, p10), inside, outside)

    if not vertices or not triangles:
        raise GeometryValidationError(
            "empty_compiled_environment",
            "surface extraction produced no triangles",
            entity_id=options.structure_id,
        )
    # Explicit open-boundary junctions remove their rounded sweep caps.  The
    # side surface remains and terminates at a collision-visible open edge.
    filtered_triangles: list[Triangle] = []
    for triangle in triangles:
        centroid = tuple(
            sum(vertices[index][axis] for index in triangle) / 3.0 for axis in range(3)
        )
        primitive = _nearest_primitive(primitives, centroid)  # type: ignore[arg-type]
        clipped_by_endpoint = any(
            _dot(_subtract(centroid, origin), inward) < 0.0
            for origin, inward in primitive.open_planes
        )
        clipped_by_aperture = any(
            candidate.source_kind == "environment_opening"
            and candidate.evaluate(centroid) <= 0.25
            and _dot(_subtract(centroid, origin), inward) <= options.voxel_size * 0.55
            for candidate in primitives
            for origin, inward in candidate.open_planes
        )
        if clipped_by_endpoint or clipped_by_aperture:
            continue
        filtered_triangles.append(triangle)
    if not filtered_triangles:
        raise GeometryValidationError(
            "empty_compiled_environment",
            "open-boundary clipping removed every triangle",
            entity_id=options.structure_id,
        )
    return TriangleMesh(tuple(vertices), tuple(filtered_triangles))


def _surface_roles(
    normal: Point3,
    primitive: _ImplicitPrimitive,
    *,
    max_traversable_slope_degrees: float,
) -> frozenset[SurfaceRole]:
    threshold = math.cos(math.radians(max_traversable_slope_degrees))
    if normal[2] >= threshold:
        roles = {SurfaceRole.SUPPORT}
        if primitive.floor_mode != PassageFloorMode.NON_TRAVERSABLE:
            roles.add(SurfaceRole.TRAVERSABLE)
        return frozenset(roles)
    if normal[2] <= -threshold:
        return frozenset({SurfaceRole.OVERHEAD, SurfaceRole.ATTACHMENT})
    return frozenset({SurfaceRole.BOUNDARY, SurfaceRole.ATTACHMENT})


def compile_semantic_environment(
    environment: SemanticEnvironmentSpec,
    *,
    options: SemanticCompileOptions | None = None,
) -> CompiledStructure:
    """Compile chamber and passage voids into one collision-ready interior shell."""

    options = options or SemanticCompileOptions()
    primitives = _build_primitives(environment)
    mesh = _extract_mesh(primitives, options)
    groups: dict[str, list[int]] = {}
    surfaces: list[CompiledSurfacePatch] = []
    for triangle_index, triangle in enumerate(mesh.triangles):
        normal = mesh.triangle_normal(triangle_index)
        centroid = tuple(
            sum(mesh.vertices[index][axis] for index in triangle) / 3.0
            for axis in range(3)
        )
        primitive = _nearest_primitive(primitives, centroid)  # type: ignore[arg-type]
        roles = _surface_roles(
            normal,
            primitive,
            max_traversable_slope_degrees=options.max_traversable_slope_degrees,
        )
        group_name = "auto_" + "_".join(sorted(role.value for role in roles))
        groups.setdefault(group_name, []).append(triangle_index)
        surfaces.append(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=(
                        f"{options.structure_id}_{group_name}_{triangle_index:06d}"
                    ),
                    roles=roles,
                    source_id=options.structure_id,
                    geometry_ref=f"triangle:{triangle_index}",
                    metadata={
                        "semantic_source_id": primitive.source_id,
                        "semantic_source_kind": primitive.source_kind,
                        **(
                            {
                                "target": opening.target.value,
                                "sky_exposed": opening.sky_exposed,
                                "weather_exposed": opening.weather_exposed,
                                "passable": opening.passable,
                                "visible": opening.visible,
                            }
                            if (
                                opening := next(
                                    (
                                        item
                                        for item in environment.openings
                                        if item.opening_id == primitive.source_id
                                    ),
                                    None,
                                )
                            )
                            is not None
                            else {}
                        ),
                        "auto_classified": True,
                    },
                ),
                boundary=tuple(mesh.vertices[index] for index in triangle),
                normal=normal,
            )
        )
    return CompiledStructure(
        structure_id=options.structure_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
        triangle_groups={
            name: tuple(indices) for name, indices in sorted(groups.items())
        },
    )


def semantic_mesh_annotations(
    compiled: CompiledStructure,
) -> tuple[MeshSurfaceAnnotation, ...]:
    """Convert compiler role groups into exhaustive imported-mesh annotations."""

    roles_by_group: dict[str, frozenset[SurfaceRole]] = {}
    for patch in compiled.surfaces:
        reference = patch.surface.geometry_ref
        if reference is None or not reference.startswith("triangle:"):
            continue
        triangle_index = int(reference.split(":", 1)[1])
        for group_name, indices in compiled.triangle_groups.items():
            if triangle_index in indices:
                roles_by_group[group_name] = patch.surface.roles
                break
    return tuple(
        MeshSurfaceAnnotation(
            annotation_id=group_name,
            triangle_indices=tuple(indices),
            roles=roles_by_group[group_name],
        )
        for group_name, indices in sorted(compiled.triangle_groups.items())
    )
