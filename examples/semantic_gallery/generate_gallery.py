#!/usr/bin/env python3
"""Compile every retained, passing semantic-environment trial for the gallery."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

from pathlib import Path
from typing import Iterable

# Keep the documented ``python examples/.../generate_gallery.py`` entry point
# usable without requiring an editable package install first.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scenesmith.agent_utils.semantic_environment_compiler import (
    SEMANTIC_ENVIRONMENT_COMPILER_VERSION,
    SemanticCompileOptions,
    compile_semantic_environment,
)
from scenesmith.agent_utils.semantic_environment_details import (
    DETAIL_SAMPLER_VERSION,
    compile_environment_details,
)
from scenesmith.agent_utils.semantic_environments import SemanticEnvironmentSpec
from scenesmith.agent_utils.structural_compiler import (
    CompiledStructure,
    CompiledStructurePaths,
    write_compiled_structure,
)
from scenesmith.agent_utils.structural_geometry import require_safe_identifier

DEFAULT_TRIAL_DIRECTORY = (
    REPOSITORY_ROOT / "docs" / "geometry-extension" / "llm-trials" / "results"
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "generated"
GALLERY_SCHEMA_VERSION = 1
GALLERY_COMPILER_VERSION = "semantic-gallery-v1"


def discover_trial_paths(trial_directory: Path | str) -> tuple[Path, ...]:
    """Return retained trial records, excluding aggregate files such as summary."""

    return tuple(sorted(Path(trial_directory).glob("heldout_*.json")))


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _combined_bounds(
    structures: Iterable[CompiledStructure],
) -> tuple[list[float], list[float]]:
    bounds = [structure.visual_mesh.bounds for structure in structures]
    return (
        [min(bound[0][axis] for bound in bounds) for axis in range(3)],
        [max(bound[1][axis] for bound in bounds) for axis in range(3)],
    )


def _rotate_rpy(point: Iterable[float], rotation_rpy: Iterable[float]) -> list[float]:
    """Apply the semantic schema's intrinsic XYZ rotation to a point."""

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
    return [
        cy * pitched[0] - sy * pitched[1],
        sy * pitched[0] + cy * pitched[1],
        pitched[2],
    ]


def _world_point(
    point: Iterable[float], environment: SemanticEnvironmentSpec, region_id: str
) -> list[float]:
    region = next(item for item in environment.regions if item.region_id == region_id)
    rotated = _rotate_rpy(point, region.transform.rotation_rpy)
    return [rotated[axis] + region.transform.translation[axis] for axis in range(3)]


def _camera_hint(
    environment: SemanticEnvironmentSpec,
    minimum: list[float],
    maximum: list[float],
) -> dict[str, list[float]]:
    center = [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)]
    chamber_target = (
        _world_point(
            environment.chambers[0].center,
            environment,
            environment.chambers[0].region_id,
        )
        if environment.chambers
        else center
    )
    junctions = [
        (junction, network.region_id)
        for network in environment.passage_networks
        for junction in network.junctions
    ]
    if junctions:
        spawn = _world_point(junctions[0][0].position, environment, junctions[0][1])
        target = chamber_target
        if math.dist(spawn, target) <= 1e-6 and len(junctions) > 1:
            target = _world_point(
                junctions[1][0].position, environment, junctions[1][1]
            )
    elif environment.chambers:
        chamber = environment.chambers[0]
        spawn = _world_point(chamber.center, environment, chamber.region_id)
        span = max(maximum[0] - minimum[0], maximum[1] - minimum[1])
        target_local = [
            chamber.center[0] + max(1.0, span * 0.25),
            chamber.center[1],
            chamber.center[2],
        ]
        target = _world_point(target_local, environment, chamber.region_id)
    else:
        spawn = center
        target = [center[0] + 1.0, center[1], center[2]]
    return {"position": spawn, "target": target}


def _write_structure(
    compiled: CompiledStructure,
    output_directory: Path,
    *,
    source_hash: str,
    compiler_version: str,
    compile_options: dict[str, object],
) -> CompiledStructurePaths:
    paths = write_compiled_structure(
        compiled,
        output_directory,
        source_content_hash=source_hash,
        compiler_version=compiler_version,
        compile_options=compile_options,
    )
    paths.artifact_ref.verify(
        expected_source_hash=source_hash,
        expected_compiler_version=compiler_version,
    )
    return paths


def _detail_metadata(
    environment: SemanticEnvironmentSpec,
) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for field in environment.detail_fields:
        metadata[field.field_id] = {
            "kind": "detail_field",
            "formation_type": field.formation_type.value,
            "surface_role": field.surface_role.value,
            "collision_policy": field.collision_policy.value,
            "instance_count": field.count,
        }
    for feature in environment.hero_features:
        metadata[feature.feature_id] = {
            "kind": "hero_feature",
            "formation_type": feature.feature_type.value,
            "surface_role": "support",
            "collision_policy": feature.collision_policy.value,
            "instance_count": 1,
        }
    return metadata


def _compile_trial(
    data: dict[str, object], output_directory: Path
) -> dict[str, object]:
    trial_id = require_safe_identifier(data["trial_id"], "trial_id")
    environment = SemanticEnvironmentSpec.from_dict(data["semantic_environment"])
    options_data = data["compiler_options"]
    if not isinstance(options_data, dict):
        raise ValueError(f"{trial_id}: compiler_options must be an object")
    compile_options = {
        "max_cells": options_data["max_cells"],
        "max_triangles": options_data["max_triangles"],
        "voxel_size": options_data["voxel_size"],
    }
    shell = compile_semantic_environment(
        environment,
        options=SemanticCompileOptions(
            structure_id=f"{trial_id}_shell",
            max_cells=compile_options["max_cells"],
            max_triangles=compile_options["max_triangles"],
            voxel_size=compile_options["voxel_size"],
        ),
    )
    source_hash = environment.content_hash()
    scene_root = output_directory / "scenes" / trial_id
    shell_paths = _write_structure(
        shell,
        scene_root / "shell",
        source_hash=source_hash,
        compiler_version=(
            f"{SEMANTIC_ENVIRONMENT_COMPILER_VERSION}+{GALLERY_COMPILER_VERSION}"
        ),
        compile_options=compile_options,
    )

    details = compile_environment_details(environment)
    details_by_id = _detail_metadata(environment)
    detail_entries: list[dict[str, object]] = []
    for detail in details.structures:
        paths = _write_structure(
            detail,
            scene_root / "details" / detail.structure_id,
            source_hash=source_hash,
            compiler_version=(
                f"semantic-detail-v{DETAIL_SAMPLER_VERSION}+{GALLERY_COMPILER_VERSION}"
            ),
            compile_options={"sampler_version": DETAIL_SAMPLER_VERSION},
        )
        detail_entries.append(
            {
                "id": detail.structure_id,
                "mesh_path": _relative(paths.mesh_path, output_directory),
                "artifact_hash": paths.artifact_hash,
                "triangles": len(detail.visual_mesh.triangles),
                "collision_enabled": detail.collision_enabled,
                **details_by_id[detail.structure_id],
            }
        )

    minimum, maximum = _combined_bounds((shell, *details.structures))
    prompt = data.get("prompt", "")
    metrics = data.get("metrics", {})
    diagnostics = data.get("diagnostics", [])
    return {
        "id": trial_id,
        "title": trial_id.removeprefix("heldout_")
        .removesuffix("_v1")
        .replace("_", " ")
        .title(),
        "model": data.get("model", "unknown"),
        "result": data.get("result", "UNKNOWN"),
        "repair_attempts": data.get("repair_attempts", 0),
        "prompt": prompt,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "semantic_hash": source_hash,
        "bounds": {"minimum": minimum, "maximum": maximum},
        "camera": _camera_hint(environment, minimum, maximum),
        "shell": {
            "mesh_path": _relative(shell_paths.mesh_path, output_directory),
            "artifact_hash": shell_paths.artifact_hash,
            "triangles": len(shell.visual_mesh.triangles),
        },
        "details": detail_entries,
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
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


def generate_gallery(
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    *,
    trial_directory: Path | str = DEFAULT_TRIAL_DIRECTORY,
) -> dict[str, object]:
    """Compile every retained PASS trial and atomically publish one manifest."""

    output = Path(output_directory)
    trial_paths = discover_trial_paths(trial_directory)
    if not trial_paths:
        raise ValueError(f"no held-out trials found in {Path(trial_directory)}")
    scenes: list[dict[str, object]] = []
    unavailable: list[dict[str, str]] = []
    for path in trial_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("result") != "PASS":
            unavailable.append(
                {
                    "id": str(data.get("trial_id", path.stem)),
                    "reason": f"retained result is {data.get('result', 'UNKNOWN')}",
                }
            )
            continue
        scenes.append(_compile_trial(data, output))
    if not scenes:
        raise ValueError("no passing held-out trials are available to render")
    manifest: dict[str, object] = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "gallery_compiler_version": GALLERY_COMPILER_VERSION,
        "scene_count": len(scenes),
        "scenes": scenes,
        "unavailable": unavailable,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--trials-dir", type=Path, default=DEFAULT_TRIAL_DIRECTORY)
    args = parser.parse_args()
    manifest = generate_gallery(args.output_dir, trial_directory=args.trials_dir)
    print(
        f"Compiled {manifest['scene_count']} semantic scenes into "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
