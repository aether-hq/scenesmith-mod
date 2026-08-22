"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import math

from pathlib import Path
from typing import Iterable, Sequence

from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    StructuralSurface,
    SurfaceRole,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    UnsupportedGeometryError,
)

Triangle = tuple[int, int, int]

from scenesmith.agent_utils.structure.compiler.models import (
    CompiledStructure,
    CompiledSurfacePatch,
    TriangleMesh,
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
