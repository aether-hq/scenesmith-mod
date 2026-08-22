"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import math

from typing import Callable, Sequence

from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point2,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    HeightfieldSpec,
    PlatformSpec,
    StructuralSurface,
    SurfaceRole,
    Transform3D,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    UnsupportedGeometryError,
)

Triangle = tuple[int, int, int]

from scenesmith.agent_utils.structure.compiler.connector_primitives import (
    _box_vertices,
    _offset,
)
from scenesmith.agent_utils.structure.compiler.mesh_assembly import _auto_surface_roles
from scenesmith.agent_utils.structure.compiler.models import (
    CollisionPrimitive,
    CompiledStructure,
    CompiledSurfacePatch,
    TriangleMesh,
    _MeshBuilder,
    _normalize,
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


def _axis_aligned_rectangle_bounds(
    footprint: Footprint2D,
) -> tuple[float, float, float, float] | None:
    """Return bounds when ``footprint`` is one unholed axis-aligned rectangle."""

    if footprint.holes or len(footprint.outer) != 4:
        return None
    min_x, min_y, max_x, max_y = footprint.bounds
    expected_corners = (
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    )
    if all(
        any(
            math.isclose(point[0], corner[0], abs_tol=GEOMETRY_TOLERANCE)
            and math.isclose(point[1], corner[1], abs_tol=GEOMETRY_TOLERANCE)
            for corner in expected_corners
        )
        for point in footprint.outer
    ):
        return (min_x, min_y, max_x, max_y)
    return None


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
    slab_mesh = builder.build()
    guard_primitives: list[CollisionPrimitive] = []
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

    guard_height = 1.1
    rail_width = 0.1

    def add_guard_member(
        *,
        member_id: str,
        origin: Point3,
        along: Point3,
        across: Point3,
        length: float,
        bottom_z: float,
        top_z: float,
        width: float,
    ) -> None:
        builder.add_prism(
            _box_vertices(
                origin=origin,
                along=along,
                across=across,
                length=length,
                width=width,
                bottom_z=bottom_z,
                top_start_z=top_z,
                top_end_z=top_z,
            )
        )
        center = _offset(origin, (along, length / 2.0))
        guard_primitives.append(
            CollisionPrimitive(
                primitive_id=member_id,
                primitive_type="box",
                transform=Transform3D(
                    translation=(
                        center[0],
                        center[1],
                        (bottom_z + top_z) / 2.0,
                    ),
                    rotation_rpy=(0.0, 0.0, math.atan2(along[1], along[0])),
                ),
                dimensions=(length, width, top_z - bottom_z),
            )
        )

    for hole_index in platform.guarded_hole_indices:
        loop = platform.footprint.holes[hole_index]
        for edge_index, (start, end) in enumerate(zip(loop, loop[1:] + loop[:1])):
            dx, dy = end[0] - start[0], end[1] - start[1]
            edge_length = math.hypot(dx, dy)
            along = (dx / edge_length, dy / edge_length, 0.0)
            across = (-along[1], along[0], 0.0)
            origin = (start[0], start[1], 0.0)
            edge_prefix = (
                f"{platform.platform_id}_guard_{hole_index:02d}_{edge_index:03d}"
            )
            add_guard_member(
                member_id=f"{edge_prefix}_base_rail",
                origin=origin,
                along=along,
                across=across,
                length=edge_length,
                bottom_z=platform.elevation + 0.12,
                top_z=platform.elevation + 0.22,
                width=rail_width,
            )
            add_guard_member(
                member_id=f"{edge_prefix}_top_rail",
                origin=origin,
                along=along,
                across=across,
                length=edge_length,
                bottom_z=platform.elevation + 0.98,
                top_z=platform.elevation + guard_height,
                width=rail_width,
            )
            interval_count = max(1, math.ceil(edge_length / 0.75))
            for post_index in range(interval_count + 1):
                distance = edge_length * post_index / interval_count
                post_center = _offset(origin, (along, distance))
                post_origin = _offset(post_center, (along, -0.035))
                add_guard_member(
                    member_id=f"{edge_prefix}_baluster_{post_index:03d}",
                    origin=post_origin,
                    along=along,
                    across=across,
                    length=0.07,
                    bottom_z=platform.elevation + 0.18,
                    top_z=platform.elevation + 1.02,
                    width=0.07,
                )
            surfaces.append(
                CompiledSurfacePatch(
                    surface=StructuralSurface(
                        surface_id=f"{edge_prefix}_surface",
                        roles=frozenset({SurfaceRole.BOUNDARY}),
                        source_id=platform.platform_id,
                        metadata={
                            "space_id": platform.space_id,
                            "hole_index": hole_index,
                            "edge_index": edge_index,
                            "structure_type": "platform_guard",
                            "guard_style": "Renaissance posts and rails",
                            "guard_height_m": guard_height,
                        },
                    ),
                    boundary=(
                        (start[0], start[1], platform.elevation),
                        (end[0], end[1], platform.elevation),
                        (end[0], end[1], platform.elevation + guard_height),
                        (start[0], start[1], platform.elevation + guard_height),
                    ),
                    normal=_normalize((dy, -dx, 0.0)),
                )
            )
    mesh = builder.build()
    return CompiledStructure(
        structure_id=platform.platform_id,
        visual_mesh=mesh,
        collision_mesh=slab_mesh,
        surfaces=tuple(surfaces),
        collision_primitives=tuple(guard_primitives),
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
