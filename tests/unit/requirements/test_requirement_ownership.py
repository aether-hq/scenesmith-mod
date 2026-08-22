"""Tests for source-bound, model-interpreted scene requirements."""

import asyncio

from types import SimpleNamespace

import pytest

import scenesmith.floor_plan_agents.stateful_floor_plan_agent as floor_plan_module

from scenesmith.agent_utils.semantics.requirements import (
    scene_requirements as requirement_module,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.literal_parsing import (
    literal_candidates_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    CompositionPlan,
    RequirementGraphValidationError,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementInterpretationWire,
    RequirementInterpretationWireBatch,
    RequirementQuantity,
    RequirementRelationWire,
    SceneCompositionOpinion,
    TopologyOpinion,
    VerificationPolicy,
    semantic_model_name,
)
from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    assert_requirement_graph_consistent,
    load_requirement_graph,
    merge_requirement_interpretations,
)
from scenesmith.agent_utils.semantics.requirements.semantic_ledger import (
    load_semantic_ledger,
)
from scenesmith.agent_utils.semantics.requirements.semantic_strategies import (
    load_capability_manifest,
)

DOCK_PROMPT = (
    "A SPACEship dock with a huge space-fighter in the middle. There are 10 "
    "repair bays, each big enough for a fighter, a giant opening into space, and "
    "then all the parts, machines, and so on are across the bay. There are "
    "massive doors into the rest of the station and many smaller doors to e.g. "
    "crew quarters, etc."
)


def test_semantic_model_override_is_reserved_for_ownership_gates(monkeypatch):
    monkeypatch.setenv("SCENESMITH_SEMANTIC_MODEL", "sonnet")
    assert semantic_model_name("haiku") == "sonnet"

    monkeypatch.delenv("SCENESMITH_SEMANTIC_MODEL")
    assert semantic_model_name("haiku") == "haiku"


def test_semantic_analyst_contract_is_explicitly_bounded():
    instructions = " ".join(requirement_module.REQUIREMENT_ANALYST_INSTRUCTIONS.split())
    assert "at most 18 words" in instructions
    assert "at most two relations" in instructions
    assert "Output only the requested schema" in instructions


def _composition():
    return SceneCompositionOpinion(
        scene_type="specialized operational interior",
        overall_scale="very large",
        preferred_dimensions_m=(45.0, 32.0, 16.0),
        composition_summary="A dominant central subject with repeated work cells.",
        topology_summary="One primary volume, perimeter cells, and external portals.",
        circulation_summary="A clear loop around the central subject.",
        density="dense perimeter with a clear center",
        focal_hierarchy=("central subject", "repeated cells", "external opening"),
    )


def _quantity_for(candidate, *, changed_value=None):
    if candidate.explicit_quantities:
        explicit = candidate.explicit_quantities[0]
        return RequirementQuantity(
            mode=explicit.mode,
            value=(changed_value if changed_value is not None else explicit.value),
            label=explicit.label,
            source_quantity_id=explicit.quantity_id,
            interpreted_minimum=(6 if explicit.mode == "qualitative" else None),
        )
    return RequirementQuantity(mode="qualitative", label="present")


def _proposal(
    candidate,
    *,
    subject=None,
    kind="object_group",
    quantity=None,
    source_quantity_id="auto",
    scale=None,
    relations=(),
):
    quantity = quantity or _quantity_for(candidate)
    if source_quantity_id == "auto":
        source_quantity_id = (
            candidate.explicit_quantities[0].quantity_id
            if candidate.explicit_quantities
            else None
        )
    return RequirementInterpretationProposal(
        candidate_id=candidate.candidate_id,
        subject=subject or candidate.evidence.text,
        kind=kind,
        source_quantity_id=source_quantity_id,
        quantity=quantity,
        scale=scale,
        relations=relations,
        topology=TopologyOpinion(
            role="model-judged role",
            enclosure="model-judged enclosure",
            circulation="maintain usable circulation",
            rationale="Derived from the full prompt.",
        ),
        composition=CompositionPlan(
            recommended_strategy="composed",
            reusable_parts=("structural frame", "surface modules"),
            procedural_geometry="Generate a measured fallback envelope.",
            arrangement="Compose the parts into the requested whole.",
            rationale="Composition preserves semantics if retrieval is insufficient.",
        ),
        verification=VerificationPolicy(
            stage="semantic",
            method="artifact_measurement",
            measurable_criteria=("The requested subject is present and measurable",),
        ),
        interpretation_rationale="The model classified this source clause in context.",
    )


def _batch(candidates, replacements=None):
    replacements = replacements or {}
    proposals = []
    for candidate in candidates:
        custom = replacements.get(candidate.evidence.text)
        if custom is None:
            proposals.append(_proposal(candidate))
        elif isinstance(custom, tuple):
            proposals.extend(custom)
        else:
            proposals.append(custom)
        # One proposal must claim each additional explicit quantity.
        claimed = {
            proposal.source_quantity_id
            for proposal in proposals
            if proposal.candidate_id == candidate.candidate_id
        }
        for explicit in candidate.explicit_quantities:
            if explicit.quantity_id not in claimed:
                proposals.append(
                    _proposal(
                        candidate,
                        subject=f"quantity-bound {candidate.evidence.text}",
                        quantity=RequirementQuantity(
                            mode=explicit.mode,
                            value=explicit.value,
                            label=explicit.label,
                            source_quantity_id=explicit.quantity_id,
                        ),
                        source_quantity_id=explicit.quantity_id,
                    )
                )
    return RequirementInterpretationBatch(
        composition=_composition(),
        requirements=tuple(proposals),
        analysis_summary="All literal candidates were interpreted.",
    )


def _wire_batch(candidates):
    batch = _batch(candidates)
    return RequirementInterpretationWireBatch(
        composition=batch.composition,
        requirements=tuple(
            RequirementInterpretationWire(
                candidate_id=proposal.candidate_id,
                subject=proposal.subject,
                kind=proposal.kind,
                source_quantity_id=proposal.source_quantity_id,
                interpreted_minimum=proposal.quantity.interpreted_minimum,
                scale_label=(
                    proposal.scale.qualitative_label if proposal.scale else ""
                ),
                scale_relative_to=(
                    proposal.scale.relative_to if proposal.scale else None
                ),
                minimum_dimensions_m=(
                    proposal.scale.minimum_dimensions_m if proposal.scale else None
                ),
                preferred_dimensions_m=(
                    proposal.scale.preferred_dimensions_m if proposal.scale else None
                ),
                clearance_m=(proposal.scale.clearance_m if proposal.scale else None),
                relations=tuple(
                    RequirementRelationWire(
                        predicate=relation.predicate,
                        target=relation.target,
                    )
                    for relation in proposal.relations
                ),
                recommended_strategy=proposal.composition.recommended_strategy,
                fallback_construction=proposal.composition.procedural_geometry,
                arrangement=proposal.composition.arrangement,
            )
            for proposal in batch.requirements
        ),
        analysis_summary=batch.analysis_summary,
    )


def _candidate(candidates, contains):
    return next(
        item for item in candidates if contains in item.evidence.text.casefold()
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "A simple bedroom with a bed, two nightstands, and a wardrobe.",
            (
                ("A simple bedroom", (1,)),
                ("with a bed", (1,)),
                ("two nightstands", (2,)),
                ("and a wardrobe", (1,)),
            ),
        ),
        (
            "A small office with one desk, one ergonomic chair, and storage.",
            (
                ("A small office", (1,)),
                ("with one desk", (1,)),
                ("one ergonomic chair", (1,)),
                ("and storage", ()),
            ),
        ),
    ],
)
def test_simple_control_prompts_create_only_literal_obligations(prompt, expected):
    candidates = literal_candidates_from_prompt(prompt)

    assert (
        tuple(
            (
                candidate.evidence.text,
                tuple(quantity.value for quantity in candidate.explicit_quantities),
            )
            for candidate in candidates
        )
        == expected
    )


def test_floor_plan_calls_semantic_llm_then_requirement_bound_spatial_compiler(
    tmp_path, monkeypatch
):
    events = []
    agent = floor_plan_module.StatefulFloorPlanAgent.__new__(
        floor_plan_module.StatefulFloorPlanAgent
    )
    agent.mode = "room"
    agent.cfg = SimpleNamespace(
        openai=SimpleNamespace(model="semantic-test-model"),
        max_floor_plan_dim_m=20.0,
        wall_height=SimpleNamespace(max=12.0),
        windows=SimpleNamespace(width_range=(0.6, 4.0), height_range=(0.6, 4.0)),
        semantic_capabilities={
            "catalog_available": True,
            "generated_geometry_available": True,
            "reusable_composition_available": True,
            "structural_compiler_available": True,
        },
    )
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent._reset_workflow_budget = lambda: None
    agent._create_run_config = lambda: None
    agent._get_model_settings = lambda **_kwargs: None
    monkeypatch.setenv("SCENESMITH_SEMANTIC_MODEL", "semantic-ownership-model")

    async def fake_analysis(prompt, candidates, **kwargs):
        events.append(("analyze", prompt, kwargs["model"]))
        return _batch(candidates), SimpleNamespace()

    class BlueprintBoundary(RuntimeError):
        pass

    async def stop_at_spatial_compiler(graph, manifest, **kwargs):
        events.append(("spatial", graph.graph_id, manifest.graph_id, kwargs["model"]))
        raise BlueprintBoundary

    monkeypatch.setattr(floor_plan_module, "load_design_system_from_env", lambda: None)
    monkeypatch.setattr(
        floor_plan_module, "analyze_requirement_candidates", fake_analysis
    )
    monkeypatch.setattr(floor_plan_module, "log_agent_usage", lambda **_kwargs: None)
    monkeypatch.setattr(
        floor_plan_module,
        "compile_requirement_blueprint",
        stop_at_spatial_compiler,
    )

    with pytest.raises(BlueprintBoundary):
        asyncio.run(agent.generate_house_layout("A novel chamber.", tmp_path / "floor"))

    assert [event[0] for event in events] == ["analyze", "spatial"]
    assert events[0][2] == "semantic-ownership-model"
    assert events[1][3] == "semantic-ownership-model"
    persisted = load_requirement_graph(tmp_path / "scene_requirement_graph.json")
    assert persisted.analysis_status == "complete"
    assert persisted.analysis_model == "semantic-ownership-model"
    assert (tmp_path / "semantic_obligation_ledger.json").is_file()
    assert (tmp_path / "semantic_obligation_summary.json").is_file()
    manifest = load_capability_manifest(tmp_path / "semantic_capability_manifest.json")
    assert manifest.preflight_passed
    assert (tmp_path / "semantic_strategy_journal.json").is_file()
    ledger = load_semantic_ledger(tmp_path / "semantic_obligation_ledger.json")
    assert all(entry.current_status == "strategy_assigned" for entry in ledger.entries)


def test_subjective_style_is_retained_but_advisory_until_calibrated():
    prompt = "A gallery with lavish iridescent decor."
    candidates = literal_candidates_from_prompt(prompt)
    style = _candidate(candidates, "lavish")
    graph = merge_requirement_interpretations(
        prompt,
        candidates,
        _batch(
            candidates,
            {
                style.evidence.text: _proposal(
                    style,
                    subject="lavish iridescent decor",
                    kind="style",
                )
            },
        ),
    )

    requirement = next(item for item in graph.requirements if item.kind == "style")
    assert requirement.strength == "hard"
    assert requirement.enforcement == "advisory"
    assert "subjective verifier" in requirement.enforcement_rationale


@pytest.mark.parametrize(
    ("prompt", "first_text", "second_text", "expected_code"),
    [
        (
            "A hall with exactly two echo plinths and exactly three echo plinths.",
            "two echo",
            "three echo",
            "exact_quantity_conflict",
        ),
        (
            "A hall with exactly two echo plinths and at least three echo plinths.",
            "two echo",
            "three echo",
            "quantity_range_conflict",
        ),
        (
            "A hall with one echo plinth. No echo plinth is allowed.",
            "one echo",
            "no echo",
            "polarity_conflict",
        ),
    ],
)
def test_conflicting_source_bound_semantics_are_explicit_graph_errors(
    prompt, first_text, second_text, expected_code
):
    candidates = literal_candidates_from_prompt(prompt)
    first = _candidate(candidates, first_text)
    second = _candidate(candidates, second_text)
    graph = merge_requirement_interpretations(
        prompt,
        candidates,
        _batch(
            candidates,
            {
                first.evidence.text: _proposal(first, subject="echo plinth"),
                second.evidence.text: _proposal(second, subject="echo plinth"),
            },
        ),
    )

    assert not graph.is_valid
    assert expected_code in {issue.code for issue in graph.validation_issues}
    with pytest.raises(RequirementGraphValidationError, match="echo plinth"):
        assert_requirement_graph_consistent(graph)
