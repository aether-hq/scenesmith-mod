"""Evidence-based PBR qualification for real SceneSmith GLTF/GLB assets."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from ..artifact_paths import composite_geometry_paths, resolve_scene_path
from ..scene_census import canonical_digest


class PbrInspectionError(RuntimeError):
    """A scene asset could not be truthfully inspected for required PBR data."""


def _load_gltf(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if path.suffix.lower() == ".gltf":
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PbrInspectionError(f"invalid GLTF JSON: {path}") from exc
        if not isinstance(value, dict):
            raise PbrInspectionError(f"GLTF root is not an object: {path}")
        return value
    if path.suffix.lower() not in {".glb", ".vrm"}:
        raise PbrInspectionError(f"unsupported scene geometry format: {path.suffix}")
    if len(data) < 20 or data[:4] != b"glTF":
        raise PbrInspectionError(f"invalid GLB header: {path}")
    declared_length = struct.unpack_from("<I", data, 8)[0]
    if declared_length != len(data):
        raise PbrInspectionError(f"GLB length mismatch: {path}")
    offset = 12
    while offset + 8 <= len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + length]
        offset += length
        if chunk_type == 0x4E4F534A:
            try:
                value = json.loads(chunk.rstrip(b" \x00"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PbrInspectionError(f"invalid GLB JSON chunk: {path}") from exc
            if not isinstance(value, dict):
                raise PbrInspectionError(f"GLB JSON root is not an object: {path}")
            return value
    raise PbrInspectionError(f"GLB has no JSON chunk: {path}")


def _texture_present(document: dict[str, Any], value: Any, root: Path) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("index"), int):
        return False
    textures = document.get("textures", ())
    index = value["index"]
    if not isinstance(textures, list) or not 0 <= index < len(textures):
        return False
    texture = textures[index]
    if not isinstance(texture, dict) or not isinstance(texture.get("source"), int):
        return False
    images = document.get("images", ())
    source = texture["source"]
    if not isinstance(images, list) or not 0 <= source < len(images):
        return False
    image = images[source]
    if not isinstance(image, dict):
        return False
    if "bufferView" in image:
        return isinstance(image["bufferView"], int)
    uri = image.get("uri")
    if not isinstance(uri, str):
        return False
    return uri.startswith("data:") or (root / uri).is_file()


def inspect_asset(path: Path) -> dict[str, Any]:
    """Return primitive-level PBR evidence tied to exact geometry bytes."""
    if not path.is_file():
        raise PbrInspectionError(f"scene geometry is missing: {path}")
    document = _load_gltf(path)
    materials = document.get("materials", ())
    findings: list[dict[str, Any]] = []
    for mesh_index, mesh in enumerate(document.get("meshes", ())):
        if not isinstance(mesh, dict):
            continue
        for primitive_index, primitive in enumerate(mesh.get("primitives", ())):
            if not isinstance(primitive, dict):
                continue
            attributes = primitive.get("attributes") or {}
            material_index = primitive.get("material")
            material = (
                materials[material_index]
                if isinstance(materials, list)
                and isinstance(material_index, int)
                and 0 <= material_index < len(materials)
                and isinstance(materials[material_index], dict)
                else {}
            )
            pbr = material.get("pbrMetallicRoughness") or {}
            shared = _texture_present(
                document, pbr.get("metallicRoughnessTexture"), path.parent
            )
            evidence = {
                "uvs": "TEXCOORD_0" in attributes,
                "base_color": _texture_present(
                    document, pbr.get("baseColorTexture"), path.parent
                )
                or "baseColorFactor" in pbr,
                "roughness": shared or "roughnessFactor" in pbr,
                "metalness": shared or "metallicFactor" in pbr,
                "normal_or_bump": _texture_present(
                    document, material.get("normalTexture"), path.parent
                ),
            }
            missing = [key for key, present in evidence.items() if not present]
            findings.append(
                {
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    **evidence,
                    "complete": not missing,
                    "missing": missing,
                }
            )
    if not findings:
        raise PbrInspectionError(f"scene geometry has no mesh primitives: {path}")
    return {
        "geometry_path": str(path),
        "geometry_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "complete": all(item["complete"] for item in findings),
        "primitives": findings,
    }


def inspect_scene_assets(
    scene_state: dict[str, Any], scene_root: Path
) -> dict[str, Any]:
    """Inspect every non-architectural scene object without fabricating evidence."""
    results = []
    complete_ids = []
    for raw_id, item in sorted(scene_state.get("objects", {}).items()):
        if item.get("object_type") in {"wall", "floor"}:
            continue
        geometry = item.get("geometry_path")
        member_paths = composite_geometry_paths(item, scene_root)
        if isinstance(geometry, str) and geometry:
            member_paths = (resolve_scene_path(geometry, scene_root),)
        if not member_paths:
            raise PbrInspectionError(f"scene object {raw_id} has no geometry evidence")
        members = [inspect_asset(path) for path in member_paths]
        result = {
            "instance_id": str(raw_id),
            "geometry_paths": [str(path) for path in member_paths],
            "geometry_sha256": canonical_digest(
                [member["geometry_sha256"] for member in members]
            ),
            "complete": all(member["complete"] for member in members),
            "members": members,
        }
        results.append(result)
        if result["complete"]:
            complete_ids.append(str(raw_id))
    return {
        "contract_version": 1,
        "scene_object_count": len(results),
        "pbr_complete_instance_ids": complete_ids,
        "incomplete_instance_ids": [
            item["instance_id"] for item in results if not item["complete"]
        ],
        "objects": results,
    }


def write_pbr_qualification(value: dict[str, Any], output_dir: Path) -> Path:
    """Persist immutable, content-addressed qualification evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = canonical_digest(value)
    path = output_dir / f"pbr-qualification-{digest}.json"
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise PbrInspectionError(f"immutable PBR qualification changed: {path.name}")
    if not path.exists():
        path.write_text(payload)
    return path
