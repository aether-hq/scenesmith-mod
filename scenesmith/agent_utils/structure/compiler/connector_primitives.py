"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import math

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
    ConnectorSpec,
    ConnectorType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    UnsafeConnectorError,
    UnsupportedGeometryError,
)

Triangle = tuple[int, int, int]

from scenesmith.agent_utils.structure.compiler.models import (
    CollisionPrimitive,
    CompiledStructure,
    CompiledSurfacePatch,
    _cross,
    _MeshBuilder,
    _normalize,
    _subtract,
)


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
