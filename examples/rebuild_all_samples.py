#!/usr/bin/env python3
"""Cleanly rebuild every checked-in SceneSmith sample and write an inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from examples.prison_escape.generate_scene import rebuild_scene
from examples.semantic_gallery.generate_gallery import (
    DEFAULT_CONTROL_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_TRIAL_DIRECTORY,
    rebuild_gallery,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRISON_OUTPUT = REPOSITORY_ROOT / "examples/prison_escape/generated"
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "examples/sample-build-manifest.json"
DEFAULT_EXPERIMENT_CONFIG = (
    REPOSITORY_ROOT / "configurations/experiment/base_experiment.yaml"
)
SAMPLE_BUILD_SCHEMA_VERSION = 1


def _path_for_manifest(path: Path) -> str:
    resolved = path.resolve()
    if resolved == REPOSITORY_ROOT:
        return "."
    if resolved.is_relative_to(REPOSITORY_ROOT):
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    return str(resolved)


def _output_paths(scene: dict[str, Any], gallery_output: Path) -> list[str]:
    if scene.get("scene_asset"):
        relatives = [scene["scene_asset"]["path"]]
    else:
        relatives = [
            scene["shell"]["mesh_path"],
            *(detail["mesh_path"] for detail in scene["details"]),
        ]
    return [_path_for_manifest(gallery_output / relative) for relative in relatives]


def _gallery_records(
    gallery_manifest: dict[str, Any], gallery_output: Path
) -> list[dict[str, Any]]:
    records = []
    for scene in gallery_manifest["scenes"]:
        reference_only = scene["representation"] == "full_fidelity_gltf"
        records.append(
            {
                "id": scene["id"],
                "title": scene["title"],
                "kind": "semantic_gallery_scene",
                "support_status": "reference_only" if reference_only else "runnable",
                "build": dict(scene["build"]),
                "outputs": _output_paths(scene, gallery_output),
                "render": {
                    "status": "pending",
                    "target": "semantic_gallery",
                    "scene_id": scene["id"],
                },
                "diagnostics": list(scene.get("diagnostics", [])),
            }
        )
    return records


def _prison_record(
    prison_manifest: dict[str, Any], prison_output: Path
) -> dict[str, Any]:
    relative_outputs = [
        "manifest.json",
        "preview.svg",
        prison_manifest["architecture"]["mesh_path"],
        prison_manifest["architecture"]["sdf_path"],
        prison_manifest["tunnel"]["mesh_path"],
        prison_manifest["tunnel"]["sdf_path"],
        "details/lights/ceiling_lights.obj",
        "details/prison/prison_details.obj",
    ]
    return {
        "id": "prison_escape_long_way_out",
        "title": prison_manifest["name"],
        "kind": "deterministic_architecture_demo",
        "support_status": "runnable",
        "build": dict(prison_manifest["build"]),
        "outputs": [
            _path_for_manifest(prison_output / relative)
            for relative in relative_outputs
        ],
        "render": {
            "status": "pending",
            "target": "prison_escape",
            "scene_id": "prison_escape_long_way_out",
        },
        "diagnostics": [],
    }


def _model_gated_records(config_path: Path) -> list[dict[str, Any]]:
    config = OmegaConf.load(config_path)
    prompts = list(config.prompts)
    source_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    requirements = [
        "OPENAI_API_KEY and GOOGLE_API_KEY for the configured agent/image providers",
        "SAM3/SAM3D provider checkpoints or a compatible external geometry service",
        "downloaded HSSD, Objaverse, articulated-object, and material indexes",
    ]
    return [
        {
            "id": f"default_five_agent_prompt_{index:03d}",
            "title": f"Default five-agent prompt {index}",
            "kind": "five_agent_text_to_scene_prompt",
            "support_status": "model_gated",
            "prompt": str(prompt),
            "build": {
                "status": "not_run",
                "rebuilt_from_recipe": False,
                "source_path": _path_for_manifest(config_path),
                "source_sha256": source_hash,
                "provider": "configured-at-runtime",
                "compiler_version": "SceneSmith five-agent pipeline",
            },
            "outputs": [],
            "render": {"status": "not_run", "target": "five_agent_pipeline"},
            "diagnostics": requirements,
        }
        for index, prompt in enumerate(prompts, start=1)
    ]


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def rebuild_all_samples(
    *,
    gallery_output: Path = DEFAULT_OUTPUT_DIRECTORY,
    prison_output: Path = DEFAULT_PRISON_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    trial_directory: Path = DEFAULT_TRIAL_DIRECTORY,
    control_directory: Path = DEFAULT_CONTROL_DIRECTORY,
    experiment_config: Path = DEFAULT_EXPERIMENT_CONFIG,
) -> dict[str, Any]:
    """Rebuild deterministic samples and inventory all model-gated prompts."""

    gallery_manifest = rebuild_gallery(
        gallery_output,
        trial_directory=trial_directory,
        control_directory=control_directory,
    )
    prison_manifest = rebuild_scene(prison_output)
    samples = [
        *_gallery_records(gallery_manifest, gallery_output),
        _prison_record(prison_manifest, prison_output),
        *_model_gated_records(experiment_config),
    ]
    summary = {
        status: sum(record["support_status"] == status for record in samples)
        for status in ("runnable", "reference_only", "model_gated")
    }
    manifest = {
        "schema_version": SAMPLE_BUILD_SCHEMA_VERSION,
        "inventory_strategy": {
            "gallery_trials": "docs/geometry-extension/llm-trials/results/heldout_*.json",
            "gallery_controls": "examples/semantic_gallery/sources/*.json",
            "deterministic_demos": ["examples/prison_escape/generate_scene.py"],
            "five_agent_prompts": _path_for_manifest(experiment_config),
        },
        "summary": summary,
        "sample_count": len(samples),
        "samples": samples,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def apply_render_report(manifest_path: Path, report_path: Path) -> dict[str, Any]:
    """Merge a complete browser-render report into the sample inventory."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or not isinstance(
        report.get("renders"), list
    ):
        raise ValueError("Unsupported or invalid sample render report")
    by_id: dict[str, dict[str, Any]] = {}
    for result in report["renders"]:
        if not isinstance(result, dict) or type(result.get("id")) is not str:
            raise ValueError("Every render result must have a string id")
        if result["id"] in by_id:
            raise ValueError(f"Duplicate render result: {result['id']}")
        if result.get("status") not in {"passed", "failed"}:
            raise ValueError(f"Invalid render status for {result['id']}")
        by_id[result["id"]] = result

    expected = {
        record["id"]
        for record in manifest["samples"]
        if record["render"]["status"] == "pending"
    }
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        unexpected = sorted(set(by_id) - expected)
        raise ValueError(
            f"Render report inventory mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for record in manifest["samples"]:
        if record["id"] not in by_id:
            continue
        result = by_id[record["id"]]
        record["render"] = {
            **record["render"],
            **result,
            "provider": report.get("provider", "unknown"),
        }
    statuses = [record["render"]["status"] for record in manifest["samples"]]
    manifest["summary"]["rendered"] = statuses.count("passed")
    manifest["summary"]["render_failed"] = statuses.count("failed")
    manifest["render_report"] = {
        "path": _path_for_manifest(report_path),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "provider": report.get("provider", "unknown"),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--prison-output", type=Path, default=DEFAULT_PRISON_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--apply-render-report",
        type=Path,
        help="merge a complete browser-render report after rebuilding",
    )
    args = parser.parse_args()
    if args.apply_render_report is not None:
        manifest = apply_render_report(args.manifest, args.apply_render_report)
    else:
        manifest = rebuild_all_samples(
            gallery_output=args.gallery_output,
            prison_output=args.prison_output,
            manifest_path=args.manifest,
        )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
