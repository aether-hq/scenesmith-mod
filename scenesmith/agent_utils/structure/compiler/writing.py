"""Deterministic compilers for parametric structural geometry.

The compiler output is simulator-agnostic: indexed triangle meshes plus
first-class semantic surface patches.  Exporters can convert this intermediate
form to GLTF/SDF, Blender, Drake, MuJoCo, or USD without re-deriving geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Mapping

import numpy as np

from scenesmith.agent_utils.structure.geometry_models.common import Point2, Point3
from scenesmith.agent_utils.structure.geometry_models.surface_models import Transform3D
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    UnsupportedGeometryError,
)
from scenesmith.utils.geometry.gltf_generation import (
    create_glb_from_mesh_data,
    zup_to_yup_transform,
)
from scenesmith.utils.geometry.material import Material

Triangle = tuple[int, int, int]

from scenesmith.agent_utils.structure.compiler.models import (
    CompiledStructure,
    CompiledStructurePaths,
    TriangleMesh,
    _artifact_identity_payload,
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
    visual_overlays: Mapping[str, tuple[TriangleMesh, Material, float]] | None = None,
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
    overlay_products: dict[str, bytes] = {}
    for overlay_name, (overlay_mesh, overlay_material, texture_scale) in (
        visual_overlays or {}
    ).items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", overlay_name):
            raise GeometryValidationError(
                "invalid_identifier",
                "visual overlay ID is not safe for a file name",
                entity_id=overlay_name,
            )
        overlay_products[f"{compiled.structure_id}.{overlay_name}.glb"] = (
            _textured_mesh_glb_bytes(
                overlay_mesh,
                overlay_material,
                texture_scale=texture_scale,
            )
        )
    collision_mesh_name: str | None = None
    collision_mesh_bytes: bytes | None = None
    if compiled.collision_enabled and (
        compiled.collision_mesh != compiled.visual_mesh or visual_material is not None
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
    for overlay_name in overlay_products:
        overlay_visual = ET.SubElement(
            link,
            "visual",
            {"name": f"{Path(overlay_name).stem}_visual"},
        )
        _add_mesh_geometry(overlay_visual, overlay_name)

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
    for overlay_name, overlay_bytes in overlay_products.items():
        product_hashes[f"visual_overlay_{overlay_name}_sha256"] = hashlib.sha256(
            overlay_bytes
        ).hexdigest()
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
    products.update(
        {artifact_path / name: data for name, data in overlay_products.items()}
    )
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
            for overlay_name, overlay_bytes in overlay_products.items():
                (staging_path / overlay_name).write_bytes(overlay_bytes)
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
