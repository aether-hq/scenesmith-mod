"""Single derived contract for semantic topology, collision, and artifacts."""

from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path

from scenesmith.agent_utils.semantic_environment_compiler import (
    SEMANTIC_ENVIRONMENT_COMPILER_VERSION,
    SemanticCompileOptions,
    compile_semantic_environment,
)
from scenesmith.agent_utils.semantic_environment_details import (
    CompiledEnvironmentDetails,
    DetailInstance,
    compile_environment_details,
)
from scenesmith.agent_utils.semantic_environments import (
    DetailCollisionPolicy,
    EnvironmentRegionSpec,
)
from scenesmith.agent_utils.structural_compiler import (
    ArtifactRef,
    CompiledStructure,
    write_compiled_structure,
)
from scenesmith.agent_utils.structural_surfaces import StructuralSurfaceIndex
from scenesmith.agent_utils.structural_topology import (
    EXTERIOR_NODE,
    StructuralTopology,
    TopologyEdge,
)

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class DerivedSceneProduct:
    """One compiled runtime product with source provenance and identity."""

    source_id: str
    source_kind: str
    artifact: ArtifactRef


@dataclass(frozen=True)
class DerivedSceneContract:
    """Products that may be consumed only after one successful derivation."""

    source_content_hash: str
    topology: StructuralTopology
    surface_index: StructuralSurfaceIndex
    collision_source_ids: frozenset[str]
    blocked_edge_ids: frozenset[str]
    products: tuple[DerivedSceneProduct, ...]


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


def _normalize(vector: Point3) -> Point3:
    length = math.sqrt(_dot(vector, vector))
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


def _interpolate_cross_section(segment, station: float) -> tuple[float, float]:
    for first, second in zip(segment.cross_sections, segment.cross_sections[1:]):
        if station <= second.station + 1e-12:
            amount = (station - first.station) / (second.station - first.station)
            amount = min(1.0, max(0.0, amount))
            return (
                first.width + (second.width - first.width) * amount,
                first.height + (second.height - first.height) * amount,
            )
    last = segment.cross_sections[-1]
    return last.width, last.height


def _instance_envelope(instance: DetailInstance) -> tuple[Point3, float]:
    half_height = instance.size[2] / 2.0
    center = _add(instance.anchor, _scale(instance.axis, half_height))
    radius = math.hypot(max(instance.size[0], instance.size[1]) / 2.0, half_height)
    return center, radius


def _blocked_semantic_edges(environment, details: CompiledEnvironmentDetails):
    """Conservatively intersect colliding detail envelopes with passage sweeps."""

    colliders = tuple(
        _instance_envelope(instance)
        for instance in details.instances
        if instance.collision_policy != DetailCollisionPolicy.VISUAL_ONLY
    )
    if not colliders:
        return frozenset()
    region_by_id = {region.region_id: region for region in environment.regions}
    blocked: set[str] = set()
    for network in environment.passage_networks:
        region = region_by_id[network.region_id]
        region_up = _normalize(
            _rotate_rpy((0.0, 0.0, 1.0), region.transform.rotation_rpy)
        )
        junctions = {item.junction_id: item for item in network.junctions}
        for segment in network.segments:
            if (
                junctions[segment.start_junction_id].space_id is None
                or junctions[segment.end_junction_id].space_id is None
            ):
                continue
            path = tuple(_transform_point(point, region) for point in segment.path)
            spans = tuple(zip(path, path[1:]))
            lengths = tuple(math.dist(start, end) for start, end in spans)
            total_length = sum(lengths)
            cumulative = 0.0
            for (start, end), span_length in zip(spans, lengths):
                tangent = _normalize(_subtract(end, start))
                across_raw = _cross(region_up, tangent)
                if math.sqrt(_dot(across_raw, across_raw)) <= 1e-12:
                    fallback = _rotate_rpy(
                        (1.0, 0.0, 0.0), region.transform.rotation_rpy
                    )
                    across_raw = _cross(fallback, tangent)
                across = _normalize(across_raw)
                vertical = _normalize(_cross(tangent, across))
                if _dot(vertical, region_up) < 0.0:
                    vertical = _scale(vertical, -1.0)
                for collider_center, collider_radius in colliders:
                    amount = min(
                        1.0,
                        max(
                            0.0,
                            _dot(_subtract(collider_center, start), tangent)
                            / span_length,
                        ),
                    )
                    station = (cumulative + amount * span_length) / total_length
                    width, height = _interpolate_cross_section(segment, station)
                    floor_center = _add(start, _scale(tangent, amount * span_length))
                    passage_center = _add(floor_center, _scale(vertical, height / 2.0))
                    passage_radius = math.hypot(width / 2.0, height / 2.0)
                    if math.dist(collider_center, passage_center) <= (
                        collider_radius + passage_radius
                    ):
                        blocked.add(segment.segment_id)
                        break
                if segment.segment_id in blocked:
                    break
                cumulative += span_length
    return frozenset(blocked)


def _semantic_topology(layout, *, shell_compiled: bool) -> StructuralTopology:
    """Derive semantic edges only after their physical shell exists."""

    base = StructuralTopology.build(
        space_ids=layout.room_ids,
        portals=layout.portals,
        connectors=layout.connectors,
    )
    if not shell_compiled or layout.semantic_environment is None:
        return base
    nodes = set(base.nodes)
    edges = list(base.edges)
    for network in layout.semantic_environment.passage_networks:
        junctions = {item.junction_id: item for item in network.junctions}
        for segment in network.segments:
            start = junctions[segment.start_junction_id]
            end = junctions[segment.end_junction_id]
            if start.space_id is None or end.space_id is None:
                continue
            nodes.update((start.space_id, end.space_id))
            if any(edge.edge_id == segment.segment_id for edge in edges):
                continue
            edges.append(
                TopologyEdge(
                    edge_id=segment.segment_id,
                    source=start.space_id,
                    target=end.space_id,
                    kind="semantic:passage",
                    required_capabilities=segment.capabilities,
                )
            )
    for opening in layout.semantic_environment.openings:
        if not opening.passable:
            continue
        source = opening.source_chamber_id
        bound_spaces = {
            junction.space_id
            for network in layout.semantic_environment.passage_networks
            for junction in network.junctions
            if junction.chamber_id == source and junction.space_id is not None
        }
        if not bound_spaces:
            continue
        target = EXTERIOR_NODE
        nodes.add(target)
        for bound_space in sorted(bound_spaces):
            edges.append(
                TopologyEdge(
                    edge_id=opening.opening_id,
                    source=bound_space,
                    target=target,
                    kind=f"semantic:opening:{opening.target.value}",
                    required_capabilities=frozenset({"walk"}),
                )
            )
    return StructuralTopology(nodes=frozenset(nodes), edges=tuple(edges))


def _collision_patches(structures: tuple[CompiledStructure, ...]):
    for structure in structures:
        if structure.collision_enabled:
            yield from (structure.collision_surfaces or structure.surfaces)


def derive_scene_contract(
    layout,
    output_dir: Path,
    *,
    voxel_size: float = 0.5,
    max_cells: int = 2_000_000,
    max_triangles: int = 500_000,
) -> DerivedSceneContract:
    """Compile and atomically expose all products from one semantic source."""

    layout.validate_structure()
    if layout.semantic_environment is None:
        return DerivedSceneContract(
            source_content_hash="",
            topology=_semantic_topology(layout, shell_compiled=False),
            surface_index=layout.build_structural_surface_index(),
            collision_source_ids=frozenset(),
            blocked_edge_ids=frozenset(),
            products=(),
        )
    environment = layout.semantic_environment
    source_hash = environment.content_hash()
    options = SemanticCompileOptions(
        voxel_size=voxel_size,
        max_cells=max_cells,
        max_triangles=max_triangles,
    )
    shell = compile_semantic_environment(environment, options=options)
    details = compile_environment_details(environment)
    shell_paths = write_compiled_structure(
        shell,
        output_dir / "shell",
        source_content_hash=source_hash,
        compiler_version=SEMANTIC_ENVIRONMENT_COMPILER_VERSION,
        compile_options={
            "max_cells": max_cells,
            "max_triangles": max_triangles,
            "voxel_size": voxel_size,
        },
    )
    products = [
        DerivedSceneProduct(
            shell.structure_id, "semantic_shell", shell_paths.artifact_ref
        )
    ]
    for structure in details.structures:
        paths = write_compiled_structure(
            structure,
            output_dir / "details" / structure.structure_id,
            source_content_hash=source_hash,
            compiler_version="semantic-environment-details-v1",
            compile_options={"sampler_version": 1},
        )
        products.append(
            DerivedSceneProduct(
                structure.structure_id, "semantic_detail", paths.artifact_ref
            )
        )
    layout.semantic_environment_geometry_path = shell_paths.sdf_path
    layout.semantic_environment_source_hash = source_hash
    layout.semantic_detail_geometry_paths = {
        product.source_id: product.artifact.sdf_path
        for product in products
        if product.source_kind == "semantic_detail"
    }
    layout.semantic_detail_source_hash = source_hash
    collision_structures = (shell,) + tuple(
        structure for structure in details.structures if structure.collision_enabled
    )
    return DerivedSceneContract(
        source_content_hash=source_hash,
        topology=_semantic_topology(layout, shell_compiled=True),
        surface_index=StructuralSurfaceIndex(_collision_patches(collision_structures)),
        collision_source_ids=frozenset(
            structure.structure_id
            for structure in collision_structures
            if structure.collision_enabled
        ),
        blocked_edge_ids=_blocked_semantic_edges(environment, details),
        products=tuple(products),
    )
