"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from scenesmith.utils.gltf_generation import (
    create_glb_from_mesh_data,
    zup_to_yup_transform,
)
from scenesmith.utils.material import Material

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
        accumulated = [[0.0, 0.0, 0.0] for _ in self.vertices]
        for triangle_index, triangle in enumerate(self.triangles):
            normal = self.triangle_normal(triangle_index)
            for vertex_index in triangle:
                for axis in range(3):
                    accumulated[vertex_index][axis] += normal[axis]
        vertex_normals = [
            (
                _normalize(tuple(normal))
                if any(abs(value) > GEOMETRY_TOLERANCE for value in normal)
                else (0.0, 0.0, 1.0)
            )
            for normal in accumulated
        ]
        lines.extend(f"vn {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertex_normals)
        lines.extend(
            f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}"
            for a, b, c in self.triangles
        )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class MeshAudit:
    """Topology/orientation facts shared by every geometry backend."""

    boundary_edges: tuple[tuple[int, int], ...]
    nonmanifold_edges: tuple[tuple[int, int], ...]
    inconsistent_edges: tuple[tuple[int, int], ...]
    signed_volume: float

    @property
    def is_closed(self) -> bool:
        return not self.boundary_edges and not self.nonmanifold_edges

    @property
    def is_winding_consistent(self) -> bool:
        return not self.nonmanifold_edges and not self.inconsistent_edges


def audit_triangle_mesh(mesh: TriangleMesh) -> MeshAudit:
    """Return dependency-free manifold, winding, and signed-volume evidence."""

    # Independent generators often emit equal coordinates with separate OBJ
    # indices.  Topology is audited on an exact deterministic positional weld,
    # matching the serialization precision used by this compiler.
    canonical_by_vertex: list[int] = []
    representative_by_position: dict[Point3, int] = {}
    for index, vertex in enumerate(mesh.vertices):
        representative_by_position.setdefault(vertex, index)
        canonical_by_vertex.append(representative_by_position[vertex])
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    signed_volume = 0.0
    for triangle in mesh.triangles:
        a, b, c = (mesh.vertices[index] for index in triangle)
        signed_volume += _dot(a, _cross(b, c)) / 6.0
        for raw_directed in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            directed = (
                canonical_by_vertex[raw_directed[0]],
                canonical_by_vertex[raw_directed[1]],
            )
            edge_uses.setdefault(tuple(sorted(directed)), []).append(directed)
    boundary_edges = tuple(
        sorted(edge for edge, uses in edge_uses.items() if len(uses) == 1)
    )
    nonmanifold_edges = tuple(
        sorted(edge for edge, uses in edge_uses.items() if len(uses) > 2)
    )
    inconsistent_edges = tuple(
        sorted(
            edge
            for edge, uses in edge_uses.items()
            if len(uses) == 2 and uses[0] != (uses[1][1], uses[1][0])
        )
    )
    return MeshAudit(
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        inconsistent_edges=inconsistent_edges,
        signed_volume=signed_volume,
    )


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
    collision_enabled: bool = True
    collision_surfaces: tuple[CompiledSurfacePatch, ...] = ()


@dataclass(frozen=True)
class CompiledStructurePaths:
    """Files emitted for a compiled structural element."""

    mesh_path: Path
    sdf_path: Path
    surfaces_path: Path
    collision_mesh_path: Path | None = None

    @property
    def artifact_hash(self) -> str:
        return self.sdf_path.parent.name

    @property
    def artifact_ref(self) -> "ArtifactRef":
        return ArtifactRef(
            mesh_path=self.mesh_path,
            sdf_path=self.sdf_path,
            surfaces_path=self.surfaces_path,
            collision_mesh_path=self.collision_mesh_path,
        )


@dataclass(frozen=True)
class ArtifactRef:
    """Authenticated reference to one atomically published artifact bundle."""

    mesh_path: Path
    sdf_path: Path
    surfaces_path: Path
    collision_mesh_path: Path | None = None

    @property
    def artifact_hash(self) -> str:
        return self.sdf_path.parent.name

    def verify(
        self,
        *,
        expected_source_hash: str | None = None,
        expected_compiler_version: str | None = None,
    ) -> Mapping[str, object]:
        """Authenticate identity and every byte before a consumer uses it."""

        try:
            manifest = json.loads(self.surfaces_path.read_text(encoding="utf-8"))
            product_hashes = manifest["product_hashes"]
            compilation = manifest["compilation"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise GeometryValidationError(
                "invalid_artifact_manifest",
                f"cannot read authenticated artifact manifest {self.surfaces_path}",
            ) from exc
        source_hash = compilation.get("source_content_hash")
        compiler_version = compilation.get("compiler_version")
        if expected_source_hash is not None and source_hash != expected_source_hash:
            raise GeometryValidationError(
                "artifact_source_mismatch", "artifact source content is stale"
            )
        if (
            expected_compiler_version is not None
            and compiler_version != expected_compiler_version
        ):
            raise GeometryValidationError(
                "artifact_compiler_mismatch", "artifact compiler version is stale"
            )
        products = {
            "mesh_sha256": self.mesh_path,
            "sdf_sha256": self.sdf_path,
        }
        collision_name = manifest.get("collision_mesh")
        if collision_name is not None:
            collision_path = self.surfaces_path.parent / str(collision_name)
            products["collision_mesh_sha256"] = collision_path
            if (
                self.collision_mesh_path is not None
                and collision_path != self.collision_mesh_path
            ):
                raise GeometryValidationError(
                    "artifact_path_mismatch",
                    "collision product path disagrees with manifest",
                )
        for hash_name, path in products.items():
            expected = product_hashes.get(hash_name)
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise GeometryValidationError(
                    "artifact_product_mismatch",
                    f"artifact product hash mismatch: {path}",
                )
        semantic_payload = dict(manifest)
        for key in (
            "artifact_hash",
            "compiler_version",
            "source_content_hash",
            "compilation",
            "product_hashes",
        ):
            semantic_payload.pop(key, None)
        semantic_hash = hashlib.sha256(
            json.dumps(
                semantic_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if semantic_hash != product_hashes.get("surface_semantics_sha256"):
            raise GeometryValidationError(
                "artifact_product_mismatch", "artifact surface semantics hash mismatch"
            )
        identity = _artifact_identity_payload(
            structure_id=str(manifest["structure_id"]),
            compilation=compilation,
            product_hashes=product_hashes,
        )
        authenticated_hash = hashlib.sha256(identity).hexdigest()
        if (
            manifest.get("artifact_hash") != authenticated_hash
            or self.artifact_hash != authenticated_hash
        ):
            raise GeometryValidationError(
                "artifact_identity_mismatch", "artifact identity is not authenticated"
            )
        if (
            manifest.get("compiler_version") != compiler_version
            or manifest.get("source_content_hash") != source_hash
        ):
            raise GeometryValidationError(
                "artifact_identity_mismatch",
                "legacy manifest identity fields disagree with authenticated compilation",
            )
        return manifest


def _artifact_identity_payload(
    *,
    structure_id: str,
    compilation: Mapping[str, object],
    product_hashes: Mapping[str, object],
) -> bytes:
    return json.dumps(
        {
            "compilation": compilation,
            "product_hashes": product_hashes,
            "structure_id": structure_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
        for edge_index, (start, end) in enumerate(
            zip(loop, loop[1:] + loop[:1])
        ):
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


def _textured_mesh_glb_bytes(
    mesh: TriangleMesh,
    material: Material,
    *,
    texture_scale: float,
) -> bytes:
    """Serialize one structural mesh as a tiled, single-material PBR GLB."""

    if texture_scale <= 0:
        raise ValueError("texture_scale must be positive")
    vertices_zup: list[Point3] = []
    normals_zup: list[Point3] = []
    uvs: list[Point2] = []
    for triangle_index, triangle in enumerate(mesh.triangles):
        normal = mesh.triangle_normal(triangle_index)
        dominant_axis = max(range(3), key=lambda axis: abs(normal[axis]))
        for vertex_index in triangle:
            x, y, z = mesh.vertices[vertex_index]
            vertices_zup.append((x, y, z))
            normals_zup.append(normal)
            if dominant_axis == 2:
                uvs.append((x / texture_scale, y / texture_scale))
            elif dominant_axis == 0:
                uvs.append((y / texture_scale, z / texture_scale))
            else:
                uvs.append((x / texture_scale, z / texture_scale))

    vertices = zup_to_yup_transform(np.asarray(vertices_zup, dtype=np.float32))
    normals = zup_to_yup_transform(np.asarray(normals_zup, dtype=np.float32))
    indices = np.arange(len(vertices_zup), dtype=np.uint32)
    textures = material.get_all_textures()
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "structure.glb"
        create_glb_from_mesh_data(
            vertices=vertices,
            normals=normals,
            uvs=np.asarray(uvs, dtype=np.float32),
            indices=indices,
            color_texture_path=textures["color"],
            normal_texture_path=textures["normal"],
            roughness_texture_path=textures["roughness"],
            output_path=output_path,
        )
        return output_path.read_bytes()


def write_compiled_structure(
    compiled: CompiledStructure,
    output_dir: Path | str,
    *,
    model_name: str | None = None,
    link_name: str = "structure_link",
    source_content_hash: str | None = None,
    compiler_version: str = "structural-compiler-v1",
    compile_options: Mapping[str, object] | None = None,
    visual_material: Material | None = None,
    visual_texture_scale: float = 0.5,
) -> CompiledStructurePaths:
    """Write OBJ, weldable SDF, and semantic-surface sidecar files.

    Collision boxes are kept analytic for stairs and flat rectangular room
    shells. Structures without analytic primitives use the collision triangle
    mesh.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", compiled.structure_id):
        raise GeometryValidationError(
            "invalid_identifier",
            "compiled structure ID is not safe for a file or model name",
            entity_id=compiled.structure_id,
        )
    if not str(compiler_version).strip():
        raise ValueError("compiler_version must not be empty")
    mesh_name = f"{compiled.structure_id}.{'glb' if visual_material else 'obj'}"
    sdf_name = f"{compiled.structure_id}.sdf"
    surfaces_name = f"{compiled.structure_id}.surfaces.json"
    mesh_bytes = (
        _textured_mesh_glb_bytes(
            compiled.visual_mesh,
            visual_material,
            texture_scale=visual_texture_scale,
        )
        if visual_material is not None
        else compiled.visual_mesh.to_obj(object_name=compiled.structure_id).encode(
            "utf-8"
        )
    )
    collision_mesh_name: str | None = None
    collision_mesh_bytes: bytes | None = None
    if (
        compiled.collision_enabled
        and (
            compiled.collision_mesh != compiled.visual_mesh
            or visual_material is not None
        )
    ):
        collision_mesh_name = f"{compiled.structure_id}.collision.obj"
        collision_mesh_bytes = compiled.collision_mesh.to_obj(
            object_name=f"{compiled.structure_id}_collision"
        ).encode("utf-8")

    sdf = ET.Element("sdf", {"version": "1.12"})
    model = ET.SubElement(sdf, "model", {"name": model_name or compiled.structure_id})
    # HouseLayout places structural models by welding them to a room or house
    # frame. An SDF-static model is already welded to world by Drake, so adding
    # the authored weld produces a duplicate-joint failure and loses multi-room
    # transforms. Keep the model non-static; the directive supplies immobility.
    ET.SubElement(model, "static").text = "false"
    link = ET.SubElement(model, "link", {"name": link_name})
    visual = ET.SubElement(link, "visual", {"name": "structure_visual"})
    _add_mesh_geometry(visual, mesh_name)

    if not compiled.collision_enabled:
        pass
    elif compiled.collision_primitives:
        if collision_mesh_name is not None:
            collision = ET.SubElement(
                link, "collision", {"name": "structure_collision"}
            )
            _add_mesh_geometry(collision, collision_mesh_name)
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
        _add_mesh_geometry(collision, collision_mesh_name or mesh_name)

    ET.indent(sdf, space="  ")
    sdf_bytes = ET.tostring(sdf, encoding="utf-8", xml_declaration=True)

    surface_data: dict[str, object] = {
        "schema_version": 1,
        "structure_id": compiled.structure_id,
        "mesh": mesh_name,
        "collision_mesh": collision_mesh_name,
        "bounds": [list(bound) for bound in compiled.visual_mesh.bounds],
        "visual_triangles": len(compiled.visual_mesh.triangles),
        "collision_triangles": (
            len(compiled.collision_mesh.triangles) if compiled.collision_enabled else 0
        ),
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
    mesh_hash = hashlib.sha256(mesh_bytes).hexdigest()
    sdf_hash = hashlib.sha256(sdf_bytes).hexdigest()
    semantic_product_hash = hashlib.sha256(
        json.dumps(surface_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    effective_source_hash = source_content_hash or semantic_product_hash
    normalized_compile_options = dict(compile_options or {})
    compilation = {
        "compile_options": normalized_compile_options,
        "compiler_version": compiler_version,
        "link_name": link_name,
        "model_name": model_name or compiled.structure_id,
        "source_content_hash": effective_source_hash,
    }
    product_hashes = {
        "mesh_sha256": mesh_hash,
        "sdf_sha256": sdf_hash,
        "surface_semantics_sha256": semantic_product_hash,
    }
    if collision_mesh_bytes is not None:
        product_hashes["collision_mesh_sha256"] = hashlib.sha256(
            collision_mesh_bytes
        ).hexdigest()
    artifact_hash = hashlib.sha256(
        _artifact_identity_payload(
            structure_id=compiled.structure_id,
            compilation=compilation,
            product_hashes=product_hashes,
        )
    ).hexdigest()
    surface_data.update(
        {
            "artifact_hash": artifact_hash,
            "compiler_version": compiler_version,
            "source_content_hash": effective_source_hash,
            "compilation": compilation,
            "product_hashes": product_hashes,
        }
    )
    surfaces_bytes = (json.dumps(surface_data, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    output_path = Path(output_dir)
    artifact_path = output_path / artifact_hash
    mesh_path = artifact_path / mesh_name
    sdf_path = artifact_path / sdf_name
    surfaces_path = artifact_path / surfaces_name
    collision_mesh_path = (
        artifact_path / collision_mesh_name if collision_mesh_name is not None else None
    )
    products = {
        mesh_path: mesh_bytes,
        sdf_path: sdf_bytes,
        surfaces_path: surfaces_bytes,
    }
    if collision_mesh_path is not None and collision_mesh_bytes is not None:
        products[collision_mesh_path] = collision_mesh_bytes
    if artifact_path.exists():
        if not all(
            path.is_file() and path.read_bytes() == data
            for path, data in products.items()
        ):
            raise GeometryValidationError(
                "artifact_hash_collision",
                "content-addressed artifact exists with different or incomplete products",
                entity_id=compiled.structure_id,
            )
    else:
        output_parent = output_path
        output_parent.mkdir(parents=True, exist_ok=True)
        staging_path = Path(
            tempfile.mkdtemp(prefix=f".{compiled.structure_id}.", dir=output_parent)
        )
        try:
            (staging_path / mesh_name).write_bytes(mesh_bytes)
            (staging_path / sdf_name).write_bytes(sdf_bytes)
            (staging_path / surfaces_name).write_bytes(surfaces_bytes)
            if collision_mesh_name is not None and collision_mesh_bytes is not None:
                (staging_path / collision_mesh_name).write_bytes(collision_mesh_bytes)
            try:
                os.replace(staging_path, artifact_path)
            except OSError:
                # Another equal writer may win the content-addressed publish
                # race.  Accept it only after validating every product byte.
                if not all(
                    path.is_file() and path.read_bytes() == data
                    for path, data in products.items()
                ):
                    raise
                shutil.rmtree(staging_path, ignore_errors=True)
        except Exception:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise
    return CompiledStructurePaths(
        mesh_path=mesh_path,
        sdf_path=sdf_path,
        surfaces_path=surfaces_path,
        collision_mesh_path=collision_mesh_path,
    )
