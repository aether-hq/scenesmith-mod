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
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    StructuralSurface,
    SurfaceRole,
    Transform3D,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    UnsafeConnectorError,
    UnsupportedGeometryError,
)

Triangle = tuple[int, int, int]

from scenesmith.agent_utils.structure.compiler.connector_primitives import (
    _box_vertices,
    _horizontal_basis,
    _offset,
    compile_straight_ramp,
    compile_straight_stairs,
)
from scenesmith.agent_utils.structure.compiler.mesh_assembly import combine_meshes
from scenesmith.agent_utils.structure.compiler.models import (
    CollisionPrimitive,
    CompiledStructure,
    CompiledSurfacePatch,
    _dot,
    _MeshBuilder,
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
