"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import math

from typing import Sequence

from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point2,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    ElevationProfile,
    Footprint2D,
    StructuralSurface,
    SurfaceRole,
    Transform3D,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import PortalSpec
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)

Triangle = tuple[int, int, int]

from scenesmith.agent_utils.structure.compiler.models import (
    CollisionPrimitive,
    CompiledStructure,
    CompiledSurfacePatch,
    _MeshBuilder,
    _normalize,
)
from scenesmith.agent_utils.structure.compiler.surfaces import (
    _add_slab,
    _axis_aligned_rectangle_bounds,
    _profile_height,
    _triangulate_footprint,
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

    rectangle_bounds = _axis_aligned_rectangle_bounds(footprint)
    floor_rectangle_bounds = _axis_aligned_rectangle_bounds(floor_footprint)
    ceiling_rectangle_bounds = _axis_aligned_rectangle_bounds(ceiling_footprint)
    floor_heights = tuple(floor_height(point) for point in footprint.outer)
    ceiling_heights = tuple(ceiling_height(point) for point in footprint.outer)
    use_analytic_shell_collision = (
        rectangle_bounds is not None
        and (not include_floor or floor_rectangle_bounds == rectangle_bounds)
        and (not include_ceiling or ceiling_rectangle_bounds == rectangle_bounds)
        and max(floor_heights) - min(floor_heights) <= GEOMETRY_TOLERANCE
        and max(ceiling_heights) - min(ceiling_heights) <= GEOMETRY_TOLERANCE
    )
    collision_primitives: list[CollisionPrimitive] = []
    if use_analytic_shell_collision:
        assert rectangle_bounds is not None
        min_x, min_y, max_x, max_y = rectangle_bounds
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        if include_floor:
            floor_z = floor_heights[0]
            collision_primitives.append(
                CollisionPrimitive(
                    primitive_id=f"{structure_id}_floor_collision",
                    primitive_type="box",
                    transform=Transform3D(
                        translation=(
                            center_x,
                            center_y,
                            floor_z - floor_thickness / 2.0,
                        )
                    ),
                    dimensions=(max_x - min_x, max_y - min_y, floor_thickness),
                )
            )
        if include_ceiling:
            ceiling_z = ceiling_heights[0]
            collision_primitives.append(
                CollisionPrimitive(
                    primitive_id=f"{structure_id}_ceiling_collision",
                    primitive_type="box",
                    transform=Transform3D(
                        translation=(
                            center_x,
                            center_y,
                            ceiling_z + ceiling_thickness / 2.0,
                        )
                    ),
                    dimensions=(
                        max_x - min_x,
                        max_y - min_y,
                        ceiling_thickness,
                    ),
                )
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
                    if use_analytic_shell_collision:
                        panel_length = distance_end - distance_start
                        panel_height = (
                            upper_start + upper_end - lower_start - lower_end
                        ) / 2.0
                        panel_center_z = (
                            lower_start + lower_end + upper_start + upper_end
                        ) / 4.0
                        collision_primitives.append(
                            CollisionPrimitive(
                                primitive_id=(
                                    f"{structure_id}_{loop_kind}_wall_"
                                    f"{edge_index:03d}_panel_{panel_index:03d}_collision"
                                ),
                                primitive_type="box",
                                transform=Transform3D(
                                    translation=(
                                        (segment_start[0] + segment_end[0]) / 2.0,
                                        (segment_start[1] + segment_end[1]) / 2.0,
                                        panel_center_z,
                                    ),
                                    rotation_rpy=(0.0, 0.0, math.atan2(dy, dx)),
                                ),
                                dimensions=(panel_length, 0.05, panel_height),
                            )
                        )
                    panel_index += 1

    mesh = builder.build()
    return CompiledStructure(
        structure_id=structure_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=tuple(surfaces),
        collision_primitives=tuple(collision_primitives),
        triangle_groups={
            group_name: tuple(indices) for group_name, indices in mutable_groups.items()
        },
    )
