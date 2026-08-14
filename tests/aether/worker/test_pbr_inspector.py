from __future__ import annotations

import json
import struct

import pytest

from scenesmith.aether.worker.pbr_inspector import (
    PbrInspectionError,
    inspect_asset,
    inspect_scene_assets,
    write_pbr_qualification,
)


def _document(*, uv: bool = True, normal: bool = True) -> dict:
    attributes = {"POSITION": 0}
    if uv:
        attributes["TEXCOORD_0"] = 1
    material = {
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 1},
        }
    }
    if normal:
        material["normalTexture"] = {"index": 2}
    return {
        "asset": {"version": "2.0"},
        "images": [
            {"uri": "data:image/png;base64,AA=="},
            {"uri": "data:image/png;base64,AA=="},
            {"uri": "data:image/png;base64,AA=="},
        ],
        "textures": [{"source": 0}, {"source": 1}, {"source": 2}],
        "materials": [material],
        "meshes": [{"primitives": [{"attributes": attributes, "material": 0}]}],
    }


def _write_glb(path, document: dict) -> None:
    payload = json.dumps(document, separators=(",", ":")).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    chunk = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(chunk)) + chunk)


def test_complete_gltf_requires_all_five_material_signals(tmp_path) -> None:
    path = tmp_path / "asset.gltf"
    path.write_text(json.dumps(_document()))
    evidence = inspect_asset(path)
    assert evidence["complete"] is True
    assert evidence["primitives"][0]["missing"] == []


def test_glb_reports_missing_uv_and_normal_map(tmp_path) -> None:
    path = tmp_path / "asset.glb"
    _write_glb(path, _document(uv=False, normal=False))
    evidence = inspect_asset(path)
    assert evidence["complete"] is False
    assert evidence["primitives"][0]["missing"] == ["uvs", "normal_or_bump"]


def test_scene_inspection_never_marks_missing_geometry_complete(tmp_path) -> None:
    state = {
        "objects": {
            "chair": {"object_type": "furniture", "geometry_path": "missing.glb"}
        }
    }
    with pytest.raises(PbrInspectionError, match="scene geometry is missing"):
        inspect_scene_assets(state, tmp_path)


def test_qualification_artifact_is_content_addressed(tmp_path) -> None:
    value = {"contract_version": 1, "incomplete_instance_ids": ["chair"]}
    first = write_pbr_qualification(value, tmp_path)
    second = write_pbr_qualification(value, tmp_path)
    assert first == second
    assert first.name.startswith("pbr-qualification-")


def test_composite_qualifies_actual_container_and_fill_members(tmp_path) -> None:
    for name in ("tray.gltf", "glass.gltf"):
        (tmp_path / name).write_text(json.dumps(_document()))
    state = {
        "objects": {
            "filled-tray": {
                "object_type": "manipuland",
                "geometry_path": None,
                "metadata": {
                    "composite_type": "filled_container",
                    "container_asset": {"geometry_path": "tray.gltf"},
                    "fill_assets": [{"geometry_path": "glass.gltf"}],
                },
            }
        }
    }
    result = inspect_scene_assets(state, tmp_path)
    assert result["pbr_complete_instance_ids"] == ["filled-tray"]
    assert len(result["objects"][0]["members"]) == 2
