"""Dependency-light tests for Aether's worker-side census bridge."""

from __future__ import annotations

import copy

import pytest

from scenesmith.aether import (
    CensusError,
    build_scene_census,
    canonical_digest,
    execute_completion_patch,
)

ARCHITECTURE = "a" * 64


def _object(
    object_id: str,
    object_type: str,
    role: str,
    *,
    locked: bool = False,
    parent_surface: str | None = None,
):
    return {
        "object_id": object_id,
        "object_type": object_type,
        "name": role,
        "description": role,
        "geometry_path": f"assets/{role}.glb",
        "support_surfaces": (
            [{"surface_id": f"{object_id}-surface"}] if role == "service-counter" else []
        ),
        "placement_info": (
            {"parent_surface_id": parent_surface} if parent_surface is not None else None
        ),
        "metadata": {
            "aether_asset_id": f"fixture:{role}",
            "aether_role_id": role,
            "aether_functional_zone_ids": ["bar-service-zone"],
            "aether_locked": locked,
        },
        "immutable": locked,
    }


def _stage_input():
    return {"job_id": "fixture-job", "locked_architecture_sha256": ARCHITECTURE}


def _evidence(ids):
    return {
        "architecture_sha256": ARCHITECTURE,
        "baseline_room_geometry_sha256": "c" * 64,
        "current_room_geometry_sha256": "c" * 64,
        "clear_circulation_route_ids": ["guest-route"],
        "clear_story_position_ids": ["story-booth"],
        "collision_instance_ids": [],
        "supported_instance_ids": list(ids),
        "pbr_complete_instance_ids": list(ids),
        "visible_view_ids_by_instance": {item: ["inspection-one"] for item in ids},
    }


def test_census_excludes_architecture_and_preserves_measured_semantics():
    state = {
        "objects": {
            "floor_0": _object("floor_0", "floor", "floor"),
            "service-counter": _object(
                "service-counter", "furniture", "service-counter", locked=True
            ),
            "bottle_0": _object(
                "bottle_0",
                "manipuland",
                "bottle",
                parent_surface="service-counter-surface",
            ),
        }
    }
    census = build_scene_census(
        _stage_input(), state, _evidence(("service-counter", "bottle-0")), round_index=0
    )
    assert [item["instance_id"] for item in census["objects"]] == [
        "bottle-0",
        "service-counter",
    ]
    bottle = census["objects"][0]
    assert bottle["supported_by_instance_id"] == "service-counter"
    assert bottle["role_id"] == "bottle"
    assert census["clear_story_position_ids"] == ["story-booth"]


def test_census_refuses_missing_physical_validation():
    evidence = _evidence(())
    del evidence["collision_instance_ids"]
    with pytest.raises(CensusError, match="incomplete"):
        build_scene_census(_stage_input(), {"objects": {}}, evidence, round_index=0)


def test_unstamped_scene_smith_names_remain_semantic_roles():
    bottle = _object("bottle_0", "manipuland", "bottle")
    bottle["metadata"].pop("aether_role_id")
    census = build_scene_census(
        _stage_input(), {"objects": {"bottle_0": bottle}}, _evidence(("bottle-0",)), round_index=0
    )
    assert census["objects"][0]["role_id"] == "bottle"


def test_census_preserves_approved_instance_id_across_internal_scene_key():
    counter = _object("scene_object_42", "furniture", "service-counter", locked=True)
    counter["metadata"]["aether_instance_id"] = "approved-service-counter"
    state = {"objects": {"scene_object_42": counter}}
    evidence = _evidence(("scene_object_42",))
    evidence["collision_instance_ids"] = ["scene_object_42"]
    census = build_scene_census(_stage_input(), state, evidence, round_index=0)
    assert census["objects"][0]["instance_id"] == "approved-service-counter"
    assert census["objects"][0]["supported"] is True
    assert census["objects"][0]["pbr_complete"] is True
    assert census["collision_instance_ids"] == ["approved-service-counter"]


class _Runtime:
    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.placed = []

    def place_asset_brief(self, operation, asset_brief, *, round_index):
        ids = []
        for index in range(asset_brief["instance_count"]):
            raw_id = f"{asset_brief['variant_id']}_{index}"
            self.state["objects"][raw_id] = _object(
                raw_id,
                "manipuland",
                "unstamped-detail",
                parent_surface="service-counter-surface",
            )
            ids.append(raw_id)
            self.placed.append(raw_id)
        return tuple(ids)

    def scene_state(self):
        return self.state

    def restore_scene_state(self, state):
        self.state = copy.deepcopy(state)

    def annotate_instances(self, raw_ids, operation, asset_brief, *, round_index):
        for raw_id in raw_ids:
            self.state["objects"][raw_id]["metadata"].update(
                {
                    "aether_role_id": operation["role_id"],
                    "aether_functional_zone_ids": operation["functional_zone_ids"],
                    "aether_completion_operation_id": operation["operation_id"],
                    "aether_completion_variant_id": asset_brief["variant_id"],
                    "aether_completion_round": round_index,
                }
            )

    def validation_evidence(self):
        return _evidence(("service-counter", *(value.replace("_", "-") for value in self.placed)))


def test_completion_bridge_places_and_stamps_semantic_patch_objects():
    state = {
        "objects": {
            "service-counter": _object(
                "service-counter", "furniture", "service-counter", locked=True
            )
        }
    }
    initial = build_scene_census(
        _stage_input(), state, _evidence(("service-counter",)), round_index=0
    )
    patch = {
        "contract_version": 1,
        "job_id": "fixture-job",
        "round_index": 0,
        "base_census_sha256": canonical_digest(initial),
        "operations": [
            {
                "operation_id": "add-bottles",
                "operation": "populate-surfaces",
                "resolves_deficit_ids": ["role-bottle"],
                "role_id": "bottle",
                "count": 2,
                "functional_zone_ids": ["bar-service-zone"],
                "minimum_asset_variants": 1,
                "support_role_ids": ["service-counter"],
                "arrangement": "surface-dressing",
                "asset_sources": ["hssd"],
                "asset_briefs": [
                    {
                        "variant_id": "amber-bottle",
                        "short_name": "amber-bottle",
                        "description": "two scuffed amber bottles",
                        "dimensions_m": [0.08, 0.08, 0.28],
                        "instance_count": 2,
                        "interaction": "static",
                        "distinctiveness": "conventional",
                        "source_order": ["hssd"],
                    }
                ],
            }
        ],
        "target_change_requested": False,
        "rationale": "resolve measured bottle shortfall",
    }
    result = execute_completion_patch(_stage_input(), patch, initial, _Runtime(state))
    assert result["placed_count"] == 2
    assert result["rejected_count"] == 0
    assert [item["role_id"] for item in result["next_census"]["objects"]].count("bottle") == 2


def test_completion_bridge_rejects_mutation_hidden_behind_equal_object_count():
    state = {
        "objects": {
            "service-counter": _object(
                "service-counter", "furniture", "service-counter", locked=True
            )
        }
    }
    initial = build_scene_census(
        _stage_input(), state, _evidence(("service-counter",)), round_index=0
    )
    patch = {
        "job_id": "fixture-job",
        "round_index": 0,
        "base_census_sha256": canonical_digest(initial),
        "operations": [
            {
                "operation_id": "add-bottle",
                "operation": "populate-surfaces",
                "role_id": "bottle",
                "count": 1,
                "functional_zone_ids": ["bar-service-zone"],
                "support_role_ids": ["service-counter"],
                "asset_briefs": [
                    {
                        "variant_id": "amber-bottle",
                        "instance_count": 1,
                    }
                ],
            }
        ],
    }

    class MutatingRuntime(_Runtime):
        def place_asset_brief(self, operation, asset_brief, *, round_index):
            ids = super().place_asset_brief(operation, asset_brief, round_index=round_index)
            self.state["objects"]["service-counter"]["metadata"]["aether_asset_id"] = "changed"
            return ids

    with pytest.raises(CensusError, match="mutated or removed"):
        runtime = MutatingRuntime(state)
        execute_completion_patch(_stage_input(), patch, initial, runtime)
    assert list(runtime.state["objects"]) == ["service-counter"]
