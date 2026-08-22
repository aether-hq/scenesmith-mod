"""Tests for generic semantic fulfillment capability preflight."""

from datetime import UTC, datetime

import pytest

from scenesmith.agent_utils.semantics.requirements.requirement_graph.literal_parsing import (
    literal_candidates_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    CompositionPlan,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementQuantity,
    SceneCompositionOpinion,
    TopologyOpinion,
    VerificationPolicy,
)
from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    merge_requirement_interpretations,
    requirement_graph_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.semantic_ledger import (
    initialize_semantic_ledger,
)
from scenesmith.agent_utils.semantics.requirements.semantic_strategies import (
    CapabilityPreflightError,
    SemanticCapabilityProfile,
    StrategyAttemptError,
    apply_capability_manifest_to_ledger,
    assert_capability_preflight_passed,
    capability_preflight,
    capability_profile_from_config,
    initialize_strategy_journal,
    load_capability_manifest,
    load_strategy_journal,
    persist_capability_manifest,
    persist_strategy_journal,
    record_strategy_attempt,
)

FIXED_TIME = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _clock():
    return FIXED_TIME


def _graph(
    *,
    kind="hero_object",
    strategy_order=("catalog", "composed", "procedural"),
    prompt="Exactly 10 velorian assemblies dominate the chamber.",
    scale=None,
    relations=(),
):
    candidates = literal_candidates_from_prompt(prompt)
    assert len(candidates) == 1
    candidate = candidates[0]
    quantity = candidate.explicit_quantities[0]
    proposal = RequirementInterpretationProposal(
        candidate_id=candidate.candidate_id,
        subject="velorian assembly",
        kind=kind,
        source_quantity_id=quantity.quantity_id,
        quantity=RequirementQuantity(
            mode=quantity.mode,
            value=quantity.value,
            source_quantity_id=quantity.quantity_id,
        ),
        scale=scale,
        relations=relations,
        topology=TopologyOpinion(
            role="dominant repeated artifact",
            enclosure="inside the primary chamber",
            circulation="preserve circulation between instances",
            rationale="Judged from the complete prompt.",
        ),
        composition=CompositionPlan(
            recommended_strategy=strategy_order[0],
            strategy_order=strategy_order,
            reusable_parts=("frame", "surface", "functional core"),
            procedural_geometry="Generate a measured envelope and visible parts.",
            arrangement="Build ten legible instances around a shared work volume.",
            rationale="The structured semantic model selected this strategy order.",
        ),
        verification=VerificationPolicy(
            stage="semantic",
            method="count_and_measure",
            measurable_criteria=("Exactly ten distinct instances are present",),
        ),
        interpretation_rationale="The LLM classified an arbitrary domain concept.",
    )
    return merge_requirement_interpretations(
        prompt,
        candidates,
        RequirementInterpretationBatch(
            composition=SceneCompositionOpinion(
                scene_type="arbitrary test chamber",
                overall_scale="large enough for ten dominant assemblies",
                preferred_dimensions_m=(30.0, 20.0, 10.0),
                composition_summary="Ten repeated artifacts define the room.",
                topology_summary="One chamber with repeated perimeter positions.",
                circulation_summary="A central circulation loop remains clear.",
                density="dense but traversable",
                focal_hierarchy=("velorian assemblies",),
            ),
            requirements=(proposal,),
            analysis_summary="The arbitrary concept was interpreted without a noun table.",
        ),
        analysis_model="fixture-semantic-model",
    )


def _profile(**updates):
    values = {
        "catalog_available": True,
        "generated_geometry_available": True,
        "reusable_composition_available": True,
        "structural_compiler_available": True,
    }
    values.update(updates)
    return SemanticCapabilityProfile(**values)


def test_llm_order_selects_first_executable_strategy_and_preserves_exact_count():
    graph = _graph(strategy_order=("procedural", "catalog", "composed"))
    manifest = capability_preflight(graph, _profile())

    assert manifest.preflight_passed
    assert manifest.plans[0].selected_strategy == "procedural"
    assert manifest.plans[0].planned_instances == 10
    assert [item.strategy for item in manifest.plans[0].ordered_strategies] == [
        "procedural",
        "catalog",
        "composed",
    ]


def test_missing_catalog_falls_through_to_model_ranked_composition():
    manifest = capability_preflight(
        _graph(),
        _profile(catalog_available=False),
    )

    plan = manifest.plans[0]
    assert manifest.preflight_passed
    assert plan.ordered_strategies[0].status == "unavailable"
    assert plan.selected_strategy == "composed"
    assert plan.selected_provider == "reusable_assembly_compiler"


def test_structural_semantics_use_topology_compiler_without_room_kit_or_catalog():
    graph = _graph(kind="repeated_zone")
    manifest = capability_preflight(
        graph,
        _profile(
            catalog_available=False,
            generated_geometry_available=False,
            reusable_composition_available=False,
        ),
    )

    plan = manifest.plans[0]
    assert manifest.preflight_passed
    assert plan.selected_strategy == "composed"
    assert plan.selected_provider == "spatial_topology_compiler"
    assert plan.planned_instances == 10


def test_no_viable_provider_fails_with_requirement_specific_diagnostic():
    graph = _graph()
    manifest = capability_preflight(
        graph,
        _profile(
            catalog_available=False,
            generated_geometry_available=False,
            reusable_composition_available=False,
            structural_compiler_available=False,
        ),
    )

    assert not manifest.preflight_passed
    assert manifest.blocking_failures == (graph.requirements[0].requirement_id,)
    with pytest.raises(CapabilityPreflightError, match="velorian assembly"):
        assert_capability_preflight_passed(manifest)


def test_unclassified_hard_requirement_never_enters_construction():
    graph = requirement_graph_from_prompt("A qelthic engine dominates the vault.")
    manifest = capability_preflight(graph, _profile())

    assert not manifest.preflight_passed
    assert all(plan.selected_strategy is None for plan in manifest.plans)


def test_forbidden_requirement_uses_absence_guard_not_construction_provider():
    graph = _graph(prompt="No 10 velorian assemblies may enter the chamber.")
    manifest = capability_preflight(
        graph,
        _profile(
            catalog_available=False,
            generated_geometry_available=False,
            reusable_composition_available=False,
            structural_compiler_available=False,
        ),
    )

    assert manifest.preflight_passed
    assert manifest.plans[0].selected_strategy == "verification_guard"
    assert manifest.plans[0].selected_provider == "semantic_absence_verifier"


def test_manifest_advances_or_fails_append_only_ledger_idempotently():
    graph = _graph()
    ledger = initialize_semantic_ledger(graph, clock=_clock)
    manifest = capability_preflight(graph, _profile())
    advanced = apply_capability_manifest_to_ledger(
        ledger,
        graph,
        manifest,
        manifest_ref="artifact:capability-manifest",
        clock=_clock,
    )
    retry = apply_capability_manifest_to_ledger(
        advanced,
        graph,
        manifest,
        manifest_ref="artifact:capability-manifest",
        clock=_clock,
    )

    assert retry == advanced
    assert advanced.entries[0].current_status == "strategy_assigned"
    assert advanced.revision == 2


def test_profile_is_derived_from_provider_configuration_not_scene_vocabulary():
    profile = capability_profile_from_config(
        {
            "experiment": {"semantic_obligations": {}},
            "furniture_agent": {"asset_manager": {"general_asset_source": "hssd"}},
            "wall_agent": {"asset_manager": {"general_asset_source": "polyhaven"}},
        }
    )

    assert profile.catalog_available
    assert not profile.generated_geometry_available
    assert profile.reusable_composition_available
    assert profile.structural_compiler_available


def test_manifest_persistence_round_trip(tmp_path):
    manifest = capability_preflight(_graph(), _profile())
    path = tmp_path / "semantic_capability_manifest.json"

    persist_capability_manifest(manifest, path)

    assert load_capability_manifest(path) == manifest


def test_strategy_attempts_are_provider_checked_evidenced_and_idempotent(tmp_path):
    manifest = capability_preflight(_graph(), _profile())
    journal = initialize_strategy_journal(manifest)
    plan = manifest.plans[0]
    recorded = record_strategy_attempt(
        journal,
        manifest,
        attempt_key="construct:velorian:1",
        requirement_id=plan.requirement_id,
        strategy=plan.selected_strategy,
        provider_id=plan.selected_provider,
        stage="construction",
        outcome="succeeded",
        evidence_refs=("artifact:velorian-0",),
        clock=_clock,
    )
    retry = record_strategy_attempt(
        recorded,
        manifest,
        attempt_key="construct:velorian:1",
        requirement_id=plan.requirement_id,
        strategy=plan.selected_strategy,
        provider_id=plan.selected_provider,
        stage="construction",
        outcome="succeeded",
        evidence_refs=("artifact:velorian-0",),
        clock=_clock,
    )
    path = tmp_path / "semantic_strategy_journal.json"
    persist_strategy_journal(retry, path)

    assert retry == recorded
    assert load_strategy_journal(path) == recorded
    with pytest.raises(StrategyAttemptError, match="not available"):
        record_strategy_attempt(
            journal,
            manifest,
            attempt_key="bad-provider",
            requirement_id=plan.requirement_id,
            strategy="catalog",
            provider_id="missing-provider",
            stage="construction",
            outcome="failed",
            diagnostic="catalog did not contain the required artifact",
            clock=_clock,
        )
