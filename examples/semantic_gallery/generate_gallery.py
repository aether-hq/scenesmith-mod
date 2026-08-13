#!/usr/bin/env python3
"""Compile every retained, passing semantic-environment trial for the gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile

from pathlib import Path
from typing import Iterable, Mapping

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
    CompiledSurfacePatch,
    TriangleMesh,
    compile_polygon_space,
    write_compiled_structure,
)
from scenesmith.agent_utils.structural_geometry import (
    Footprint2D,
    PortalSpec,
    PortalType,
    StructuralSurface,
    SurfaceRole,
    require_safe_identifier,
    validate_global_identifiers,
)

DEFAULT_TRIAL_DIRECTORY = (
    REPOSITORY_ROOT / "docs" / "geometry-extension" / "llm-trials" / "results"
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "generated"
DEFAULT_CONTROL_DIRECTORY = Path(__file__).resolve().parent / "sources"
GALLERY_SCHEMA_VERSION = 2
GALLERY_COMPILER_VERSION = "semantic-gallery-v2"


def discover_trial_paths(trial_directory: Path | str) -> tuple[Path, ...]:
    """Return retained trial records, excluding aggregate files such as summary."""

    return tuple(sorted(Path(trial_directory).glob("heldout_*.json")))


def discover_control_paths(control_directory: Path | str) -> tuple[Path, ...]:
    """Return checked-in non-trial scene controls compiled by the gallery."""

    return tuple(sorted(Path(control_directory).glob("*.json")))


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


def _move_toward(start: list[float], end: list[float], distance: float) -> list[float]:
    separation = math.dist(start, end)
    if separation <= 1e-9:
        return list(start)
    amount = min(1.0, distance / separation)
    return [start[axis] + (end[axis] - start[axis]) * amount for axis in range(3)]


def _network_camera_hint(
    environment: SemanticEnvironmentSpec, chamber_target: list[float]
) -> dict[str, list[float]] | None:
    """Choose a safe point along semantic free space, never a graph singularity."""

    if not environment.passage_networks:
        return None
    network = max(
        environment.passage_networks,
        key=lambda item: (len(item.segments), item.network_id),
    )
    chamber_junctions = [item for item in network.junctions if item.chamber_id]
    if chamber_junctions:
        junction = chamber_junctions[0]
        entry = _world_point(junction.position, environment, network.region_id)
        return {
            "position": _move_toward(entry, chamber_target, 4.0),
            "target": chamber_target,
        }

    junction = max(
        network.junctions,
        key=lambda item: (network.degree(item.junction_id), item.junction_id),
    )
    candidates = [
        segment
        for segment in network.segments
        if junction.junction_id in {segment.start_junction_id, segment.end_junction_id}
    ]
    if not candidates:
        return None
    segment = max(
        candidates,
        key=lambda item: (
            max(section.width * section.height for section in item.cross_sections),
            item.segment_id,
        ),
    )
    local_path = list(segment.path)
    if segment.end_junction_id == junction.junction_id:
        local_path.reverse()
    world_path = [
        _world_point(point, environment, network.region_id) for point in local_path
    ]
    first_span = math.dist(world_path[0], world_path[1])
    camera_distance = min(1.25, first_span * 0.2)
    target_distance = min(10.0, first_span * 0.75)
    position = _move_toward(world_path[0], world_path[1], camera_distance)
    target = _move_toward(world_path[0], world_path[1], target_distance)
    region = next(
        item for item in environment.regions if item.region_id == network.region_id
    )
    region_up = _rotate_rpy((0.0, 0.0, 1.0), region.transform.rotation_rpy)
    tangent = [
        (world_path[1][axis] - world_path[0][axis]) / first_span for axis in range(3)
    ]
    across = [
        region_up[1] * tangent[2] - region_up[2] * tangent[1],
        region_up[2] * tangent[0] - region_up[0] * tangent[2],
        region_up[0] * tangent[1] - region_up[1] * tangent[0],
    ]
    across_length = math.sqrt(sum(value * value for value in across))
    if across_length <= 1e-9:
        across = [1.0, 0.0, 0.0]
    else:
        across = [value / across_length for value in across]
    passage_vertical = [
        tangent[1] * across[2] - tangent[2] * across[1],
        tangent[2] * across[0] - tangent[0] * across[2],
        tangent[0] * across[1] - tangent[1] * across[0],
    ]
    if sum(passage_vertical[axis] * region_up[axis] for axis in range(3)) < 0.0:
        passage_vertical = [-value for value in passage_vertical]
    section = (
        segment.cross_sections[0]
        if segment.start_junction_id == junction.junction_id
        else segment.cross_sections[-1]
    )
    eye_height = min(1.6, max(0.7, section.height * 0.45))
    position = [
        position[axis] + passage_vertical[axis] * eye_height for axis in range(3)
    ]
    target = [target[axis] + passage_vertical[axis] * eye_height for axis in range(3)]
    return {"position": position, "target": target}


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
    network_hint = _network_camera_hint(environment, chamber_target)
    if network_hint is not None:
        return network_hint
    if environment.chambers:
        chamber = environment.chambers[0]
        span = max(maximum[0] - minimum[0], maximum[1] - minimum[1])
        center_world = _world_point(chamber.center, environment, chamber.region_id)
        spawn_local = [
            chamber.center[0] - max(1.0, span * 0.15),
            chamber.center[1],
            chamber.center[2],
        ]
        raw_spawn = _world_point(spawn_local, environment, chamber.region_id)
        spawn = _move_toward(center_world, raw_spawn, max(1.0, span * 0.15))
        target = center_world
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


def _summary_metrics(metrics: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {"label": label, "value": metrics.get(key, 0)}
        for label, key in (
            ("chambers", "chambers"),
            ("passages", "segments"),
            ("junctions", "junctions"),
            ("openings", "openings"),
            ("formations", "detail_instances"),
            ("heroes", "hero_features"),
        )
    ]


def _gallery_voxel_size(
    environment: SemanticEnvironmentSpec, requested: float
) -> float:
    """Resolve a view mesh fine enough for the smallest authored passage."""

    passage_dimensions = [
        min(section.width, section.height)
        for network in environment.passage_networks
        for segment in network.segments
        for section in segment.cross_sections
    ]
    if not passage_dimensions:
        return requested
    # At least roughly one sample across each radius prevents coarse marching
    # chords from visibly cutting through an otherwise valid narrow passage.
    return min(requested, max(0.35, min(passage_dimensions) * 0.9))


def _compile_trial(
    data: dict[str, object], output_directory: Path
) -> dict[str, object]:
    trial_id = require_safe_identifier(data["trial_id"], "trial_id")
    environment = SemanticEnvironmentSpec.from_dict(data["semantic_environment"])
    options_data = data["compiler_options"]
    if not isinstance(options_data, dict):
        raise ValueError(f"{trial_id}: compiler_options must be an object")
    requested_voxel_size = options_data["voxel_size"]
    display_voxel_size = _gallery_voxel_size(environment, requested_voxel_size)
    compile_options = {
        "max_cells": options_data["max_cells"],
        "max_triangles": options_data["max_triangles"],
        "voxel_size": display_voxel_size,
        "source_voxel_size": requested_voxel_size,
    }
    shell = compile_semantic_environment(
        environment,
        options=SemanticCompileOptions(
            structure_id=f"{trial_id}_shell",
            max_cells=compile_options["max_cells"],
            max_triangles=compile_options["max_triangles"],
            voxel_size=display_voxel_size,
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
        "summary_metrics": _summary_metrics(metrics),
        "source_kind": "heldout_llm_semantic_environment",
        "compiler": (
            "scenesmith.agent_utils.semantic_environment_compiler."
            "compile_semantic_environment"
        ),
        "representation": "compiled_semantic_geometry",
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


_FURNITURE_FORMS = frozenset(
    {"chair", "couch", "counter", "round-table", "stool", "table"}
)
_LIGHT_ROLES = frozenset({"general-practical-light", "task-practical-light"})


def _proxy_material_key(placement: Mapping[str, object]) -> str:
    role = placement["role"]
    form = placement["form"]
    if role in _LIGHT_ROLES:
        return "practical_light"
    if form in _FURNITURE_FORMS:
        return "furniture"
    if form in {"bin", "bottle", "canister", "screen", "switch"}:
        return "dressing"
    return "fixtures"


def _box_mesh(center: list[float], size: list[float], yaw: float) -> TriangleMesh:
    half = [value / 2.0 for value in size]
    local = (
        (-half[0], -half[1], -half[2]),
        (half[0], -half[1], -half[2]),
        (half[0], half[1], -half[2]),
        (-half[0], half[1], -half[2]),
        (-half[0], -half[1], half[2]),
        (half[0], -half[1], half[2]),
        (half[0], half[1], half[2]),
        (-half[0], half[1], half[2]),
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    vertices = tuple(
        (
            center[0] + cosine * x - sine * y,
            center[1] + sine * x + cosine * y,
            center[2] + z,
        )
        for x, y, z in local
    )
    triangles = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    return TriangleMesh(vertices, triangles)


def _combine_proxy_meshes(meshes: Iterable[TriangleMesh]) -> TriangleMesh:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        triangles.extend(
            tuple(index + offset for index in triangle) for triangle in mesh.triangles
        )
    return TriangleMesh(tuple(vertices), tuple(triangles))


def _proxy_structure(
    structure_id: str, placements: list[Mapping[str, object]]
) -> CompiledStructure:
    mesh = _combine_proxy_meshes(
        _box_mesh(item["center"], item["size"], item["yaw"]) for item in placements
    )
    roles = frozenset({SurfaceRole.BOUNDARY, SurfaceRole.NON_INTERACTIVE})
    surfaces = tuple(
        CompiledSurfacePatch(
            surface=StructuralSurface(
                surface_id=f"{structure_id}_triangle_{index:06d}",
                roles=roles,
                source_id=structure_id,
                geometry_ref=f"triangle:{index}",
                metadata={"representation": "semantic_proxy_regression"},
            ),
            boundary=tuple(mesh.vertices[vertex] for vertex in triangle),
            normal=mesh.triangle_normal(index),
        )
        for index, triangle in enumerate(mesh.triangles)
    )
    return CompiledStructure(
        structure_id=structure_id,
        visual_mesh=mesh,
        collision_mesh=mesh,
        surfaces=surfaces,
        triangle_groups={"semantic_proxies": tuple(range(len(mesh.triangles)))},
        collision_enabled=False,
    )


def _bar_portal(
    opening: Mapping[str, object], shell: Mapping[str, object]
) -> PortalSpec:
    width, _, depth = shell["dimensions_m"]
    boundary = opening["boundary"]
    edge_index = {"north": 0, "east": 1, "south": 2, "west": 3}[boundary]
    edge_length = width if boundary in {"north", "south"} else depth
    center = opening["offset_m"] + opening["width_m"] / 2.0
    if boundary in {"south", "west"}:
        center = edge_length - center
    return PortalSpec(
        portal_id=opening["opening_id"],
        portal_type=PortalType.OPEN,
        source_space_id=shell["room_id"],
        target_space_id=opening["connection_id"],
        width=opening["width_m"],
        height=opening["height_m"],
        boundary_loop_index=0,
        boundary_edge_index=edge_index,
        position_along=center,
        sill_height=opening["sill_m"],
    )


def _compile_bar_control(
    data: Mapping[str, object], output_directory: Path
) -> dict[str, object]:
    control_id = require_safe_identifier(data["id"], "control_id")
    shell_data = data["shell"]
    source = data["source"]
    width, height, depth = shell_data["dimensions_m"]
    source_hash = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    shell = compile_polygon_space(
        structure_id=f"{control_id}_shell",
        footprint=Footprint2D.rectangle(width, depth),
        wall_height=height,
        floor_thickness=shell_data["wall_thickness_m"],
        ceiling_thickness=shell_data["wall_thickness_m"],
        portals=tuple(
            _bar_portal(opening, shell_data) for opening in shell_data["openings"]
        ),
    )
    scene_root = output_directory / "scenes" / control_id
    compiler_version = f"structural-compiler-v1+{GALLERY_COMPILER_VERSION}"
    shell_paths = _write_structure(
        shell,
        scene_root / "shell",
        source_hash=source_hash,
        compiler_version=compiler_version,
        compile_options={"control_schema_version": data["schema_version"]},
    )
    placements = data["placements"]
    validate_global_identifiers(
        (placement["id"], "bar_control_placement") for placement in placements
    )
    groups: dict[str, list[Mapping[str, object]]] = {}
    for placement in placements:
        groups.setdefault(_proxy_material_key(placement), []).append(placement)
    details: list[dict[str, object]] = []
    structures: list[CompiledStructure] = []
    for material_key, items in sorted(groups.items()):
        structure = _proxy_structure(f"{control_id}_{material_key}", items)
        structures.append(structure)
        paths = _write_structure(
            structure,
            scene_root / "details" / material_key,
            source_hash=source_hash,
            compiler_version=f"gallery-semantic-proxy-v1+{GALLERY_COMPILER_VERSION}",
            compile_options={"material_key": material_key},
        )
        details.append(
            {
                "id": structure.structure_id,
                "mesh_path": _relative(paths.mesh_path, output_directory),
                "artifact_hash": paths.artifact_hash,
                "triangles": len(structure.visual_mesh.triangles),
                "collision_enabled": False,
                "kind": "semantic_proxy_layer",
                "material_key": material_key,
                "instance_count": len(items),
            }
        )
    minimum, maximum = _combined_bounds((shell, *structures))
    camera = data["cameras"][1]
    metrics = {
        "rooms": 1,
        "portals": len(shell_data["openings"]),
        "placements": len(placements),
        "cameras": len(data["cameras"]),
    }
    return {
        "id": control_id,
        "title": data["title"],
        "model": "SceneSmith original control",
        "result": "PASS",
        "repair_attempts": 0,
        "prompt": data["prompt"],
        "diagnostics": [],
        "metrics": metrics,
        "summary_metrics": [
            {"label": "rooms", "value": 1},
            {"label": "portals", "value": metrics["portals"]},
            {"label": "placements", "value": metrics["placements"]},
            {"label": "cameras", "value": metrics["cameras"]},
            {"label": "reference meshes", "value": source["reference_meshes"]},
            {"label": "reference tris", "value": source["reference_triangles"]},
        ],
        "source_kind": source["kind"],
        "compiler": (
            "scenesmith.agent_utils.structural_compiler.compile_polygon_space"
        ),
        "representation": "semantic_proxy_regression",
        "semantic_hash": source_hash,
        "reference": source,
        "bounds": {"minimum": minimum, "maximum": maximum},
        "camera": {
            "position": camera["position_m"],
            "target": camera["target_m"],
            "label": camera["label"],
        },
        "shell": {
            "mesh_path": _relative(shell_paths.mesh_path, output_directory),
            "artifact_hash": shell_paths.artifact_hash,
            "triangles": len(shell.visual_mesh.triangles),
        },
        "details": details,
    }


def _compile_control(
    data: Mapping[str, object], output_directory: Path
) -> dict[str, object]:
    source = data.get("source")
    if (
        isinstance(source, Mapping)
        and source.get("kind") == "accepted_aether_room_packet"
    ):
        return _compile_bar_control(data, output_directory)
    raise ValueError(f"unsupported gallery control source: {source!r}")


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
    control_directory: Path | str = DEFAULT_CONTROL_DIRECTORY,
) -> dict[str, object]:
    """Compile retained PASS trials and checked-in controls into one manifest."""

    output = Path(output_directory)
    trial_paths = discover_trial_paths(trial_directory)
    if not trial_paths:
        raise ValueError(f"no held-out trials found in {Path(trial_directory)}")
    control_paths = discover_control_paths(control_directory)
    if not control_paths:
        raise ValueError(f"no gallery controls found in {Path(control_directory)}")
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
    for path in control_paths:
        scenes.append(
            _compile_control(json.loads(path.read_text(encoding="utf-8")), output)
        )
    if not scenes:
        raise ValueError("no gallery scenes are available to render")
    manifest: dict[str, object] = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "gallery_compiler_version": GALLERY_COMPILER_VERSION,
        "scene_count": len(scenes),
        "trial_count": len(scenes) - len(control_paths),
        "control_count": len(control_paths),
        "scenes": scenes,
        "unavailable": unavailable,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--trials-dir", type=Path, default=DEFAULT_TRIAL_DIRECTORY)
    parser.add_argument("--controls-dir", type=Path, default=DEFAULT_CONTROL_DIRECTORY)
    args = parser.parse_args()
    manifest = generate_gallery(
        args.output_dir,
        trial_directory=args.trials_dir,
        control_directory=args.controls_dir,
    )
    print(
        f"Compiled {manifest['scene_count']} semantic scenes into "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
