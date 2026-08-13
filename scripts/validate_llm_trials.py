#!/usr/bin/env python3
"""Validate retained held-out LLM authoring evidence without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenesmith.agent_utils.semantic_environment_compiler import (
    SemanticCompileOptions,
    compile_semantic_environment,
)
from scenesmith.agent_utils.semantic_environment_details import (
    compile_environment_details,
)
from scenesmith.agent_utils.semantic_environments import SemanticEnvironmentSpec
from scenesmith.agent_utils.structural_compiler import (
    audit_triangle_mesh,
    write_compiled_structure,
)

REQUIRED_FIELDS = {
    "trial_id",
    "prompt_sha256",
    "model",
    "raw_response_sha256",
    "semantic_environment",
    "diagnostics",
    "predicates",
    "result",
    "run_timestamp_utc",
    "repair_attempts",
    "metrics",
    "compiler_options",
    "repairs",
}


def _branching_trial_predicates(
    environment: SemanticEnvironmentSpec,
) -> dict[str, bool]:
    if len(environment.passage_networks) != 1:
        return {"heldout_branching_requirements": False}
    network = environment.passage_networks[0]
    degree = {junction.junction_id: 0 for junction in network.junctions}
    for segment in network.segments:
        degree[segment.start_junction_id] += 1
        degree[segment.end_junction_id] += 1
    remaining = set(degree)
    components = 0
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        while frontier:
            current = frontier.pop()
            for segment in network.segments:
                neighbor = None
                if segment.start_junction_id == current:
                    neighbor = segment.end_junction_id
                elif segment.end_junction_id == current:
                    neighbor = segment.start_junction_id
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    sizes = {
        (section.width, section.height)
        for segment in network.segments
        for section in segment.cross_sections
    }
    elevations = [junction.position[2] for junction in network.junctions]
    return {
        "heldout_central_degree_five": max(degree.values(), default=0) == 5,
        "heldout_cycle_rank_one": len(network.segments)
        - len(network.junctions)
        + components
        == 1,
        "heldout_at_least_two_dead_ends": sum(value == 1 for value in degree.values())
        >= 2,
        "heldout_meaningful_elevation_change": max(elevations) - min(elevations)
        >= 10.0,
        "heldout_variable_cross_sections": len(sizes) > 1,
        "heldout_all_segments_walkable": all(
            "walk" in segment.capabilities for segment in network.segments
        ),
        "heldout_no_unrequested_primitives": not (
            environment.chambers
            or environment.openings
            or environment.detail_fields
            or environment.hero_features
        ),
    }


def _dragon_trial_predicates(environment: SemanticEnvironmentSpec) -> dict[str, bool]:
    large_chamber = any(
        all(
            actual >= required for actual, required in zip(chamber.size, (160, 100, 60))
        )
        for chamber in environment.chambers
    )
    sky_opening = any(
        opening.target.value == "sky" and opening.weather_exposed
        for opening in environment.openings
    )
    protected_stalactites = any(
        field.formation_type.value == "stalactite"
        and field.count == 24
        and field.collision_policy.value == "coarse"
        and field.route_clearance >= 5.0
        and bool(field.protect_passage_network_ids)
        for field in environment.detail_fields
    )
    hero_spire = any(
        feature.feature_type.value == "rock_spire"
        and feature.collision_policy.value == "full"
        for feature in environment.hero_features
    )
    approach_bound_to_chamber = any(
        "walk" in segment.capabilities
        and (
            junctions[segment.start_junction_id].chamber_id is not None
            or junctions[segment.end_junction_id].chamber_id is not None
        )
        for network in environment.passage_networks
        for junctions in (
            {junction.junction_id: junction for junction in network.junctions},
        )
        for segment in network.segments
    )
    return {
        "heldout_dragon_scale_chamber": large_chamber,
        "heldout_walkable_bound_approach": approach_bound_to_chamber,
        "heldout_weather_exposed_sky_opening": sky_opening,
        "heldout_protected_coarse_stalactites": protected_stalactites,
        "heldout_full_collision_hero_spire": hero_spire,
    }


TRIAL_PREDICATES = {
    "heldout_branching_network_v1": _branching_trial_predicates,
    "heldout_dragon_scale_cavern_v1": _dragon_trial_predicates,
}


def validate_trial(path: Path, *, update: bool = False) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS - set(data)
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
    if data["result"] not in {"PASS", "FAIL", "UNSUPPORTED", "BLOCKED_ENV"}:
        raise ValueError(f"{path}: invalid result category")
    expected_prompt_hash = hashlib.sha256(data["prompt"].encode("utf-8")).hexdigest()
    if data["prompt_sha256"] != expected_prompt_hash:
        raise ValueError(f"{path}: prompt_sha256 does not authenticate prompt")
    environment = SemanticEnvironmentSpec.from_dict(data["semantic_environment"])
    longest_axis = max(
        region.bounds.maximum[axis] - region.bounds.minimum[axis]
        for region in environment.regions
        for axis in range(3)
    )
    voxel_size = max(1.0, longest_axis / 50.0)
    compiler_options = {
        "max_cells": 500_000,
        "max_triangles": 500_000,
        "voxel_size": voxel_size,
    }
    if data["compiler_options"] != compiler_options:
        raise ValueError(f"{path}: retained compiler_options do not match validator")
    compiled = compile_semantic_environment(
        environment,
        options=SemanticCompileOptions(
            voxel_size=voxel_size,
            max_cells=compiler_options["max_cells"],
            max_triangles=compiler_options["max_triangles"],
        ),
    )
    details = compile_environment_details(environment)
    audit = audit_triangle_mesh(compiled.visual_mesh)
    predicates = {
        "mesh_winding_consistent": audit.is_winding_consistent,
        "semantic_sources_present": bool(compiled.surfaces),
        "detail_instance_count": len(details.instances)
        == sum(field.count for field in environment.detail_fields)
        + len(environment.hero_features),
    }
    predicate_builder = TRIAL_PREDICATES.get(data["trial_id"])
    if predicate_builder is None:
        raise ValueError(f"{path}: no held-out requirement oracle for trial_id")
    predicates.update(predicate_builder(environment))
    metrics = {
        "chambers": len(environment.chambers),
        "detail_fields": len(environment.detail_fields),
        "detail_instances": len(details.instances),
        "hero_features": len(environment.hero_features),
        "junctions": sum(
            len(network.junctions) for network in environment.passage_networks
        ),
        "openings": len(environment.openings),
        "passage_networks": len(environment.passage_networks),
        "segments": sum(
            len(network.segments) for network in environment.passage_networks
        ),
        "semantic_json_bytes": len(
            json.dumps(
                environment.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ),
        "visual_triangles": len(compiled.visual_mesh.triangles),
    }
    if data["metrics"] != metrics and not update:
        raise ValueError(f"{path}: retained metrics do not match current compiler")
    data["metrics"] = metrics
    if not environment.openings and not any(
        junction.open_boundary
        for network in environment.passage_networks
        for junction in network.junctions
    ):
        predicates["closed_when_no_apertures"] = audit.is_closed
    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact = write_compiled_structure(
            compiled,
            temporary_directory,
            source_content_hash=environment.content_hash(),
            compiler_version="held-out-eval-v1",
            compile_options=compiler_options,
        ).artifact_ref
        artifact.verify(
            expected_source_hash=environment.content_hash(),
            expected_compiler_version="held-out-eval-v1",
        )
        predicates["artifact_authenticated"] = True
    if data["predicates"] and data["predicates"] != predicates and not update:
        raise ValueError(f"{path}: retained predicates do not match current compiler")
    data["predicates"] = predicates
    if data["result"] == "PASS" and not all(predicates.values()):
        raise ValueError(f"{path}: PASS has a false verification predicate")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("docs/geometry-extension/llm-trials/results"),
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite retained predicates after an intentional oracle change",
    )
    args = parser.parse_args()
    paths = (
        sorted(args.directory.glob("heldout_*.json")) if args.directory.exists() else []
    )
    if not paths:
        raise SystemExit("No held-out LLM trial result files were found.")
    for path in paths:
        validate_trial(path, update=args.update)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    passed_before_repair = sum(
        result["result"] == "PASS" and result["repair_attempts"] == 0
        for result in results
    )
    passed_after_repair = sum(result["result"] == "PASS" for result in results)
    summary = {
        "after_repair_pass_rate": passed_after_repair / len(results),
        "after_repair_threshold": 0.85,
        "after_repair_threshold_met": passed_after_repair / len(results) >= 0.85,
        "before_repair_pass_rate": passed_before_repair / len(results),
        "before_repair_threshold": 0.65,
        "before_repair_threshold_met": passed_before_repair / len(results) >= 0.65,
        "executed_trials": len(results),
        "note": (
            "Initial held-out evidence only; the specification requires ten runs "
            "per prompt before treating these rates as statistically meaningful."
        ),
        "overall_threshold_met": (
            passed_before_repair / len(results) >= 0.65
            and passed_after_repair / len(results) >= 0.85
        ),
    }
    summary_path = args.directory / "summary.json"
    if summary_path.exists() and not args.update:
        retained_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if retained_summary != summary:
            raise ValueError(f"{summary_path}: retained aggregate summary is stale")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Validated {len(paths)} held-out LLM authoring trials.")


if __name__ == "__main__":
    main()
