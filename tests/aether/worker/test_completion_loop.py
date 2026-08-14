from __future__ import annotations

import json

import pytest

from scenesmith.aether.scene_census import canonical_digest
from scenesmith.aether.worker.completion_loop import (
    ArtifactLedger,
    CompletionLoopError,
    build_authoring_brief,
    derive_deficit_report,
    validate_patch,
)


def _stage_input() -> dict:
    return {
        "job_id": "bar-one",
        "room_prompt": "A fully approved bar environment with service, seating, circulation, and contextual dressing requirements. The prompt is intentionally long enough for the durable authoring contract.",
        "request": {
            "shell": {"dimensions_m": [10, 4, 10]},
            "functional_zones": [{"zone_id": "back-bar"}],
            "realization": {
                "density": {
                    "target_objects_per_50_m2": 4,
                    "minimum_completion_ratio": 0.8,
                    "role_targets": [
                        {
                            "role_id": "bottle",
                            "functional_zone_ids": ["back-bar"],
                            "target_count": 4,
                            "maximum_count": 6,
                            "support_role_ids": ["display-shelf"],
                            "arrangement": "surface-dressing",
                            "rationale": "The approved bar needs visible bottle variation on its display shelving.",
                        }
                    ],
                },
                "asset_routing": {
                    "conventional_sources": ["curated", "hssd"],
                    "articulated_source": "artvip",
                    "distinctive_source": "sam3d",
                    "generated_fallback_source": "sam3d",
                },
            },
        },
    }


def _census() -> dict:
    return {
        "contract_version": 1,
        "job_id": "bar-one",
        "round_index": 0,
        "architecture_sha256": "a" * 64,
        "objects": [
            {
                "instance_id": "shelf-one",
                "role_id": "display-shelf",
                "functional_zone_ids": ["back-bar"],
            },
            {
                "instance_id": "bottle-one",
                "role_id": "bottle",
                "functional_zone_ids": ["back-bar"],
            },
        ],
        "clear_circulation_route_ids": [],
        "clear_story_position_ids": [],
        "collision_instance_ids": [],
    }


def test_report_and_brief_preserve_approved_target_and_sources() -> None:
    report = derive_deficit_report(_stage_input(), _census())
    assert report["approved_target_count"] == 8
    assert report["observed_count"] == 2
    assert report["maximum_addition_count"] == 6
    role = next(item for item in report["deficits"] if item["role_id"] == "bottle")
    assert role["shortfall_count"] == 3
    assert role["permitted_operations"] == ["populate-surfaces"]
    brief = build_authoring_brief(_stage_input(), report)
    assert brief["report"] is report
    assert brief["permitted_asset_sources"] == ["curated", "hssd", "artvip", "sam3d"]


def test_patch_validation_rejects_stale_or_over_broad_patch() -> None:
    report = derive_deficit_report(_stage_input(), _census())
    patch = {
        "job_id": "bar-one",
        "round_index": 0,
        "base_census_sha256": "0" * 64,
        "operations": [
            {
                "operation_id": "wrong",
                "operation": "place-floor-group",
                "resolves_deficit_ids": ["role-bottle"],
                "role_id": "chair",
                "count": 3,
                "functional_zone_ids": ["unknown-zone"],
            }
        ],
    }
    with pytest.raises(CompletionLoopError, match="stale census"):
        validate_patch(_stage_input(), report, patch)


def test_artifact_ledger_is_content_addressed_and_idempotent(tmp_path) -> None:
    ledger = ArtifactLedger(tmp_path)
    value = {"round": 0, "state": "measured"}
    digest = ledger.record("scene-census", 0, value)
    assert digest == canonical_digest(value)
    assert ledger.record("scene-census", 0, value) == digest
    path = next(tmp_path.glob("00-scene-census-*.json"))
    assert json.loads(path.read_text()) == value
