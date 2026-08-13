"""Deterministic geological detail and hero-feature compilation."""

from __future__ import annotations

import math

from dataclasses import dataclass

from scenesmith.agent_utils.semantic_environments import (
    CavernChamberSpec,
    CavernShape,
    DetailCollisionPolicy,
    DetailFieldSpec,
    DetailSurfaceRole,
    EnvironmentRegionSpec,
    FormationType,
    HeroFeatureSpec,
    HeroFeatureType,
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structural_compiler import (
    CompiledStructure,
    CompiledSurfacePatch,
    Triangle,
    TriangleMesh,
    audit_triangle_mesh,
)
from scenesmith.agent_utils.structural_geometry import (
    GeometryValidationError,
    Point3,
    StructuralSurface,
    SurfaceRole,
    UnsupportedGeometryError,
)

DETAIL_SAMPLER_VERSION = 1
_MASK_64 = (1 << 64) - 1

FORMATION_MESH_FAMILY = {
    FormationType.STALACTITE: "cone",
    FormationType.STALAGMITE: "cone",
    FormationType.COLUMN: "cone",
    FormationType.FLOWSTONE: "rounded",
    FormationType.BOULDER: "rounded",
    FormationType.RUBBLE: "rounded",
    FormationType.SCREE: "rounded",
}

HERO_FORMATION_TYPE = {
    HeroFeatureType.ROCK_SPIRE: FormationType.STALAGMITE,
    HeroFeatureType.BOULDER: FormationType.BOULDER,
}


def _add(first: Point3, second: Point3) -> Point3:
    return tuple(first[axis] + second[axis] for axis in range(3))  # type: ignore[return-value]


def _subtract(first: Point3, second: Point3) -> Point3:
    return tuple(first[axis] - second[axis] for axis in range(3))  # type: ignore[return-value]


def _scale(vector: Point3, amount: float) -> Point3:
    return tuple(component * amount for component in vector)  # type: ignore[return-value]


def _dot(first: Point3, second: Point3) -> float:
    return sum(first[axis] * second[axis] for axis in range(3))


def _length(vector: Point3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Point3) -> Point3:
    length = _length(vector)
    if length <= 1e-12:
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


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return value ^ (value >> 31)


def _sample_unit(seed: int, candidate: int, channel: int) -> float:
    value = (
        (int(seed) & _MASK_64)
        ^ ((candidate + 1) * 0xD1B54A32D192ED03)
        ^ ((channel + 1) * 0x94D049BB133111EB)
        ^ DETAIL_SAMPLER_VERSION
    )
    return (_mix64(value & _MASK_64) >> 11) / float(1 << 53)


def _distance_to_segment(point: Point3, start: Point3, end: Point3) -> float:
    span = _subtract(end, start)
    length_squared = _dot(span, span)
    if length_squared <= 1e-12:
        return math.dist(point, start)
    amount = min(1.0, max(0.0, _dot(_subtract(point, start), span) / length_squared))
    return math.dist(point, _add(start, _scale(span, amount)))


@dataclass(frozen=True)
class DetailInstance:
    instance_id: str
    source_id: str
    formation_type: FormationType
    anchor: Point3
    axis: Point3
    size: Point3
    surface_role: DetailSurfaceRole
    collision_policy: DetailCollisionPolicy
    sampler_version: int = DETAIL_SAMPLER_VERSION


@dataclass(frozen=True)
class CompiledEnvironmentDetails:
    structures: tuple[CompiledStructure, ...]
    instances: tuple[DetailInstance, ...]
    dropped_candidates: int


def _sample_chamber_surface(
    chamber: CavernChamberSpec,
    region: EnvironmentRegionSpec,
    role: DetailSurfaceRole,
    radial: float,
    angle: float,
) -> tuple[Point3, Point3]:
    if chamber.shape not in {CavernShape.ELLIPSOID, CavernShape.SUPERELLIPSOID}:
        raise UnsupportedGeometryError(
            "detail sampling currently requires an ellipsoid or superellipsoid",
            entity_id=chamber.chamber_id,
        )
    exponent = 2 if chamber.shape == CavernShape.ELLIPSOID else 4
    half = tuple(component / 2.0 for component in chamber.size)
    local_x = half[0] * radial * math.cos(angle)
    local_y = half[1] * radial * math.sin(angle)
    horizontal = abs(local_x / half[0]) ** exponent + abs(local_y / half[1]) ** exponent
    local_z_extent = half[2] * max(0.0, 1.0 - horizontal) ** (1.0 / exponent)
    if role == DetailSurfaceRole.OVERHEAD:
        local = (local_x, local_y, local_z_extent)
    elif role == DetailSurfaceRole.SUPPORT:
        local = (local_x, local_y, -local_z_extent)
    else:
        boundary_angle = angle
        vertical = radial * 2.0 - 1.0
        radius_scale = max(0.0, 1.0 - abs(vertical) ** exponent) ** (1.0 / exponent)
        local = (
            half[0] * radius_scale * math.cos(boundary_angle),
            half[1] * radius_scale * math.sin(boundary_angle),
            half[2] * vertical,
        )
    chamber_point = _add(chamber.center, _rotate_rpy(local, chamber.orientation_rpy))
    anchor = _transform_point(chamber_point, region)
    chamber_center = _transform_point(chamber.center, region)
    return anchor, _normalize(_subtract(chamber_center, anchor))


def _masked(
    anchor: Point3,
    axis: Point3,
    size: Point3,
    field_spec: DetailFieldSpec,
    environment: SemanticEnvironmentSpec,
    region: EnvironmentRegionSpec,
    accepted_instances: tuple[DetailInstance, ...] = (),
) -> str | None:
    lateral_radius = max(size[0], size[1]) / 2.0
    half_height = size[2] / 2.0
    instance_center = _add(anchor, _scale(axis, half_height))
    instance_radius = math.hypot(lateral_radius, half_height)
    protected = set(field_spec.protect_passage_network_ids)
    clearance = field_spec.route_clearance + instance_radius
    for network in environment.passage_networks:
        if network.network_id not in protected:
            continue
        for segment in network.segments:
            path = tuple(_transform_point(point, region) for point in segment.path)
            spans = tuple(zip(path, path[1:]))
            span_lengths = tuple(math.dist(start, end) for start, end in spans)
            total_length = sum(span_lengths)
            cumulative = 0.0
            for (start, end), span_length in zip(spans, span_lengths):
                span = _subtract(end, start)
                amount = min(
                    1.0,
                    max(
                        0.0,
                        _dot(_subtract(instance_center, start), span)
                        / (span_length * span_length),
                    ),
                )
                station = (cumulative + amount * span_length) / total_length
                width, height = _interpolate_segment_cross_section(segment, station)
                passage_radius = max(width / 2.0, height)
                if _distance_to_segment(instance_center, start, end) < (
                    clearance + passage_radius
                ):
                    return "passage"
                cumulative += span_length
    for opening in environment.openings:
        if opening.region_id != field_spec.region_id:
            continue
        center = _transform_point(opening.center, region)
        opening_clearance = max(opening.size) / 2.0 + instance_radius
        if math.dist(instance_center, center) < opening_clearance:
            return "opening"
    for feature in environment.hero_features:
        if feature.region_id != field_spec.region_id:
            continue
        center = _transform_point(feature.anchor, region)
        feature_clearance = max(feature.size) / 2.0 + instance_radius
        if math.dist(instance_center, center) < feature_clearance:
            return "hero"
    for accepted in accepted_instances:
        accepted_lateral = max(accepted.size[0], accepted.size[1]) / 2.0
        accepted_half_height = accepted.size[2] / 2.0
        accepted_center = _add(
            accepted.anchor, _scale(accepted.axis, accepted_half_height)
        )
        accepted_radius = math.hypot(accepted_lateral, accepted_half_height)
        if (
            math.dist(instance_center, accepted_center)
            < instance_radius + accepted_radius
        ):
            return "detail_conflict"
    return None


def _interpolate_segment_cross_section(segment, station: float) -> tuple[float, float]:
    for first, second in zip(segment.cross_sections, segment.cross_sections[1:]):
        if station <= second.station + 1e-12:
            span = second.station - first.station
            amount = min(1.0, max(0.0, (station - first.station) / span))
            return (
                first.width + (second.width - first.width) * amount,
                first.height + (second.height - first.height) * amount,
            )
    last = segment.cross_sections[-1]
    return last.width, last.height


def sample_detail_field(
    field_spec: DetailFieldSpec,
    environment: SemanticEnvironmentSpec,
    *,
    accepted_instances: tuple[DetailInstance, ...] = (),
) -> tuple[tuple[DetailInstance, ...], int]:
    """Sample a detail field deterministically while enforcing semantic masks."""

    chamber_by_id = {item.chamber_id: item for item in environment.chambers}
    region_by_id = {item.region_id: item for item in environment.regions}
    chamber = chamber_by_id[field_spec.target_chamber_id]
    region = region_by_id[field_spec.region_id]
    instances: list[DetailInstance] = []
    dropped = 0
    mask_causes: dict[str, int] = {}
    maximum_candidates = max(field_spec.count * 256, field_spec.count)
    for candidate in range(maximum_candidates):
        if len(instances) == field_spec.count:
            break
        radial = math.sqrt(_sample_unit(field_spec.seed, candidate, 0)) * 0.94
        angle = 2.0 * math.pi * _sample_unit(field_spec.seed, candidate, 1)
        anchor, axis = _sample_chamber_surface(
            chamber, region, field_spec.surface_role, radial, angle
        )
        size = tuple(
            field_spec.min_size[axis]
            + (field_spec.max_size[axis] - field_spec.min_size[axis])
            * _sample_unit(field_spec.seed, candidate, axis + 2)
            for axis in range(3)
        )
        mask_cause = _masked(
            anchor,
            axis,
            size,
            field_spec,
            environment,
            region,
            (*accepted_instances, *instances),
        )
        if mask_cause is not None:
            dropped += 1
            mask_causes[mask_cause] = mask_causes.get(mask_cause, 0) + 1
            continue
        instances.append(
            DetailInstance(
                instance_id=f"{field_spec.field_id}_{len(instances):04d}",
                source_id=field_spec.field_id,
                formation_type=field_spec.formation_type,
                anchor=anchor,
                axis=axis,
                size=size,  # type: ignore[arg-type]
                surface_role=field_spec.surface_role,
                collision_policy=field_spec.collision_policy,
            )
        )
    if len(instances) != field_spec.count:
        cause_summary = ", ".join(
            f"{cause}={count}" for cause, count in sorted(mask_causes.items())
        )
        raise GeometryValidationError(
            "no_legal_detail_samples",
            f"placed {len(instances)} of {field_spec.count} requested samples after "
            f"{maximum_candidates} deterministic attempts; mask causes: "
            f"{cause_summary or 'none'}",
            entity_id=field_spec.field_id,
        )
    return tuple(instances), dropped


def _formation_mesh(
    instance: DetailInstance, radial_segments: int = 10
) -> TriangleMesh:
    anchor = instance.anchor
    axis = _normalize(instance.axis)
    reference = (0.0, 0.0, 1.0)
    if abs(_dot(axis, reference)) > 0.95:
        reference = (1.0, 0.0, 0.0)
    across = _normalize(
        (
            reference[1] * axis[2] - reference[2] * axis[1],
            reference[2] * axis[0] - reference[0] * axis[2],
            reference[0] * axis[1] - reference[1] * axis[0],
        )
    )
    vertical = _normalize(
        (
            axis[1] * across[2] - axis[2] * across[1],
            axis[2] * across[0] - axis[0] * across[2],
            axis[0] * across[1] - axis[1] * across[0],
        )
    )

    def world(local_x: float, local_y: float, local_z: float) -> Point3:
        return _add(
            anchor,
            _add(
                _scale(across, local_x),
                _add(_scale(vertical, local_y), _scale(axis, local_z)),
            ),
        )

    radius_x, radius_y, height = (
        instance.size[0] / 2.0,
        instance.size[1] / 2.0,
        instance.size[2],
    )
    mesh_family = FORMATION_MESH_FAMILY[instance.formation_type]
    if mesh_family == "rounded":
        rings = 5
        vertices: list[Point3] = [world(0.0, 0.0, 0.0)]
        for ring in range(1, rings):
            latitude = -math.pi / 2.0 + math.pi * ring / rings
            for segment in range(radial_segments):
                angle = 2.0 * math.pi * segment / radial_segments
                vertices.append(
                    world(
                        radius_x * math.cos(latitude) * math.cos(angle),
                        radius_y * math.cos(latitude) * math.sin(angle),
                        height / 2.0 + height / 2.0 * math.sin(latitude),
                    )
                )
        top_index = len(vertices)
        vertices.append(world(0.0, 0.0, height))
        triangles: list[Triangle] = []
        first_ring = 1
        for segment in range(radial_segments):
            following = (segment + 1) % radial_segments
            triangles.append((0, first_ring + following, first_ring + segment))
        for ring in range(rings - 2):
            for segment in range(radial_segments):
                following = (segment + 1) % radial_segments
                current = first_ring + ring * radial_segments + segment
                next_ring = current + radial_segments
                triangles.extend(
                    (
                        (
                            current,
                            first_ring + ring * radial_segments + following,
                            next_ring,
                        ),
                        (
                            first_ring + ring * radial_segments + following,
                            first_ring + (ring + 1) * radial_segments + following,
                            next_ring,
                        ),
                    )
                )
        last_ring = first_ring + (rings - 2) * radial_segments
        for segment in range(radial_segments):
            following = (segment + 1) % radial_segments
            triangles.append((last_ring + segment, last_ring + following, top_index))
        return TriangleMesh(tuple(vertices), tuple(triangles))

    base = tuple(
        world(
            radius_x * math.cos(2.0 * math.pi * index / radial_segments),
            radius_y * math.sin(2.0 * math.pi * index / radial_segments),
            0.0,
        )
        for index in range(radial_segments)
    )
    vertices = (*base, world(0.0, 0.0, height), anchor)
    tip_index = radial_segments
    center_index = radial_segments + 1
    triangles = []
    for index in range(radial_segments):
        following = (index + 1) % radial_segments
        triangles.extend(
            ((index, following, tip_index), (index, center_index, following))
        )
    return TriangleMesh(tuple(vertices), tuple(triangles))


def _combine_meshes(meshes: tuple[TriangleMesh, ...]) -> TriangleMesh:
    vertices: list[Point3] = []
    triangles: list[Triangle] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        triangles.extend(
            tuple(offset + index for index in triangle)  # type: ignore[arg-type]
            for triangle in mesh.triangles
        )
    return TriangleMesh(tuple(vertices), tuple(triangles))


def _compile_instances(
    source_id: str,
    instances: tuple[DetailInstance, ...],
    collision_policy: DetailCollisionPolicy,
) -> CompiledStructure:
    mesh = _combine_meshes(tuple(_formation_mesh(instance) for instance in instances))
    collision_mesh = mesh
    if collision_policy == DetailCollisionPolicy.COARSE:
        collision_mesh = _combine_meshes(
            tuple(
                _formation_mesh(instance, radial_segments=6) for instance in instances
            )
        )
    for label, audited_mesh in (
        ("visual", mesh),
        ("collision", collision_mesh),
    ):
        audit = audit_triangle_mesh(audited_mesh)
        if (
            not audit.is_closed
            or not audit.is_winding_consistent
            or audit.signed_volume <= 0.0
        ):
            raise GeometryValidationError(
                "invalid_compiled_detail_mesh",
                f"{label} detail mesh must be closed, consistently wound, and outward",
                entity_id=source_id,
            )
    authored_roles = {instance.surface_role for instance in instances}
    structural_roles = frozenset(
        {
            {
                DetailSurfaceRole.OVERHEAD: SurfaceRole.OVERHEAD,
                DetailSurfaceRole.SUPPORT: SurfaceRole.SUPPORT,
                DetailSurfaceRole.BOUNDARY: SurfaceRole.BOUNDARY,
            }[role]
            for role in authored_roles
        }
        | (
            {SurfaceRole.NON_INTERACTIVE}
            if collision_policy == DetailCollisionPolicy.VISUAL_ONLY
            else {SurfaceRole.BOUNDARY}
        )
    )
    surfaces = tuple(
        CompiledSurfacePatch(
            surface=StructuralSurface(
                surface_id=f"{source_id}_triangle_{triangle_index:06d}",
                roles=structural_roles,
                source_id=source_id,
                geometry_ref=f"triangle:{triangle_index}",
                metadata={
                    "detail_source_id": source_id,
                    "collision_policy": collision_policy.value,
                    "sampler_version": DETAIL_SAMPLER_VERSION,
                },
            ),
            boundary=tuple(mesh.vertices[index] for index in triangle),
            normal=mesh.triangle_normal(triangle_index),
        )
        for triangle_index, triangle in enumerate(mesh.triangles)
    )
    collision_surfaces = surfaces
    if collision_policy == DetailCollisionPolicy.COARSE:
        collision_surfaces = tuple(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=f"{source_id}_collision_triangle_{triangle_index:06d}",
                    roles=structural_roles,
                    source_id=source_id,
                    geometry_ref=f"collision_triangle:{triangle_index}",
                    metadata={
                        "detail_source_id": source_id,
                        "collision_policy": collision_policy.value,
                        "sampler_version": DETAIL_SAMPLER_VERSION,
                    },
                ),
                boundary=tuple(collision_mesh.vertices[index] for index in triangle),
                normal=collision_mesh.triangle_normal(triangle_index),
            )
            for triangle_index, triangle in enumerate(collision_mesh.triangles)
        )
    return CompiledStructure(
        structure_id=source_id,
        visual_mesh=mesh,
        collision_mesh=collision_mesh,
        surfaces=surfaces,
        triangle_groups={"details": tuple(range(len(mesh.triangles)))},
        collision_enabled=collision_policy != DetailCollisionPolicy.VISUAL_ONLY,
        collision_surfaces=collision_surfaces,
    )


def _hero_instance(
    feature: HeroFeatureSpec, region: EnvironmentRegionSpec
) -> DetailInstance:
    formation_type = HERO_FORMATION_TYPE[feature.feature_type]
    return DetailInstance(
        instance_id=feature.feature_id,
        source_id=feature.feature_id,
        formation_type=formation_type,
        anchor=_transform_point(feature.anchor, region),
        axis=_normalize(_rotate_rpy((0.0, 0.0, 1.0), region.transform.rotation_rpy)),
        size=feature.size,
        surface_role=DetailSurfaceRole.SUPPORT,
        collision_policy=feature.collision_policy,
    )


def compile_environment_details(
    environment: SemanticEnvironmentSpec,
) -> CompiledEnvironmentDetails:
    """Compile all repeated detail fields and hero geological landmarks."""

    structures: list[CompiledStructure] = []
    instances: list[DetailInstance] = []
    dropped = 0
    for field_spec in environment.detail_fields:
        sampled, field_dropped = sample_detail_field(
            field_spec, environment, accepted_instances=tuple(instances)
        )
        structures.append(
            _compile_instances(
                field_spec.field_id, sampled, field_spec.collision_policy
            )
        )
        instances.extend(sampled)
        dropped += field_dropped
    region_by_id = {item.region_id: item for item in environment.regions}
    for feature in environment.hero_features:
        instance = _hero_instance(feature, region_by_id[feature.region_id])
        structures.append(
            _compile_instances(
                feature.feature_id, (instance,), feature.collision_policy
            )
        )
        instances.append(instance)
    return CompiledEnvironmentDetails(tuple(structures), tuple(instances), dropped)
