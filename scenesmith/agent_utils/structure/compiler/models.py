"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import hashlib
import json
import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    StructuralSurface,
    Transform3D,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
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


def triangle_group_mesh(
    mesh: TriangleMesh,
    triangle_indices: Sequence[int],
    *,
    translation: Point3 = (0.0, 0.0, 0.0),
) -> TriangleMesh:
    """Extract a compact translated mesh for one authored triangle group."""

    selected = tuple(mesh.triangles[index] for index in triangle_indices)
    if not selected:
        raise ValueError("triangle group must contain at least one triangle")
    used_indices = sorted({index for triangle in selected for index in triangle})
    remap = {source: target for target, source in enumerate(used_indices)}
    translated_vertices = tuple(
        (
            mesh.vertices[index][0] + translation[0],
            mesh.vertices[index][1] + translation[1],
            mesh.vertices[index][2] + translation[2],
        )
        for index in used_indices
    )
    return TriangleMesh(
        vertices=translated_vertices,
        triangles=tuple(
            tuple(remap[index] for index in triangle) for triangle in selected
        ),
    )


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
