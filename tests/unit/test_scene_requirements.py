"""Tests for source-bound, model-interpreted scene requirements."""

import asyncio

from types import SimpleNamespace

import pytest

import scenesmith.floor_plan_agents.stateful_floor_plan_agent as floor_plan_module

from scenesmith.agent_utils.scene_blueprint import blueprint_from_prompt
from scenesmith.agent_utils.scene_requirements import (
    CompositionPlan,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementMergeError,
    RequirementQuantity,
    RequirementRelation,
    RequirementScale,
    SceneCompositionOpinion,
    SceneRequirementGraph,
    TopologyOpinion,
    VerificationPolicy,
    analyze_requirement_candidates,
    audit_requirement_graph,
    literal_candidates_from_prompt,
    load_requirement_graph,
    merge_requirement_interpretations,
    persist_requirement_graph,
    persist_shadow_audit,
    requirement_graph_from_prompt,
)


DOCK_PROMPT = (
    "A SPACEship dock with a huge space-fighter in the middle. There are 10 "
    "repair bays, each big enough for a fighter, a giant opening into space, and "
    "then all the parts, machines, and so on are across the bay. There are "
    "massive doors into the rest of the station and many smaller doors to e.g. "
    "crew quarters, etc."
)


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


def _candidate(candidates, contains):
    return next(
        item for item in candidates if contains in item.evidence.text.casefold()
    )


def test_llm_semantics_supply_size_composition_topology_and_strategies():
    candidates = literal_candidates_from_prompt(DOCK_PROMPT)
    scene_type = _candidate(candidates, "spaceship dock")
    fighter = _candidate(candidates, "space-fighter")
    bays = _candidate(candidates, "10 repair bays")
    sized_for = _candidate(candidates, "big enough for")
    opening = _candidate(candidates, "giant opening")
    parts = _candidate(candidates, "all the parts")
    machines = _candidate(candidates, "machines")
    distribution = _candidate(candidates, "across the bay")
    station_doors = _candidate(candidates, "massive doors")
    crew_doors = _candidate(candidates, "smaller doors")

    batch = _batch(
        candidates,
        {
            scene_type.evidence.text: _proposal(
                scene_type,
                subject="spaceship dock",
                kind="scene_type",
            ),
            fighter.evidence.text: _proposal(
                fighter,
                subject="space fighter",
                kind="hero_object",
                scale=RequirementScale(
                    qualitative_label="huge",
                    relative_to="repair bay",
                    minimum_dimensions_m=(12.0, 8.0, 3.0),
                    preferred_dimensions_m=(18.0, 12.0, 4.5),
                    clearance_m=2.0,
                    rationale="A dominant serviceable fighter needs operational clearance.",
                ),
                relations=(
                    RequirementRelation(
                        predicate="centered_in",
                        target="primary dock volume",
                        rationale="The prompt makes it the central focal object.",
                    ),
                ),
            ),
            bays.evidence.text: _proposal(
                bays,
                subject="repair bay",
                kind="repeated_zone",
            ),
            sized_for.evidence.text: _proposal(
                sized_for,
                subject="repair bay capacity",
                kind="spatial_constraint",
                relations=(
                    RequirementRelation(
                        predicate="sized_for",
                        target="space fighter",
                        rationale="Each repeated bay must accommodate the fighter.",
                    ),
                ),
            ),
            opening.evidence.text: _proposal(
                opening,
                subject="space opening",
                kind="opening",
                scale=RequirementScale(
                    qualitative_label="giant",
                    relative_to="space fighter",
                    minimum_dimensions_m=(20.0, 1.0, 10.0),
                    preferred_dimensions_m=(28.0, 1.0, 14.0),
                    clearance_m=3.0,
                    rationale="The portal must visually and physically serve the fighter.",
                ),
            ),
            parts.evidence.text: _proposal(
                parts,
                subject="fighter parts",
                kind="object_group",
            ),
            machines.evidence.text: _proposal(
                machines,
                subject="repair machines",
                kind="object_group",
            ),
            distribution.evidence.text: _proposal(
                distribution,
                subject="support equipment distribution",
                kind="spatial_constraint",
                relations=(
                    RequirementRelation(
                        predicate="distributed_across",
                        target="dock bay",
                        rationale="Support equipment belongs throughout the bay.",
                    ),
                ),
            ),
            station_doors.evidence.text: _proposal(
                station_doors,
                subject="station doors",
                kind="opening",
                quantity=RequirementQuantity(mode="minimum", value=2),
                source_quantity_id=None,
                scale=RequirementScale(
                    qualitative_label="massive",
                    relative_to="personnel doors",
                    rationale="These are the primary station circulation portals.",
                ),
                relations=(
                    RequirementRelation(
                        predicate="connects_to",
                        target="rest of station",
                    ),
                ),
            ),
            crew_doors.evidence.text: _proposal(
                crew_doors,
                subject="crew-quarter doors",
                kind="opening",
                scale=RequirementScale(
                    qualitative_label="smaller",
                    relative_to="station doors",
                    rationale="They are subordinate personnel-scale openings.",
                ),
                relations=(
                    RequirementRelation(
                        predicate="connects_to",
                        target="crew quarters",
                    ),
                ),
            ),
        },
    )
    graph = merge_requirement_interpretations(
        DOCK_PROMPT, candidates, batch, analysis_model="test-model"
    )

    assert graph.analysis_status == "complete"
    assert graph.composition.preferred_dimensions_m == (45.0, 32.0, 16.0)
    fighter_requirement = next(
        item for item in graph.requirements if item.subject == "space fighter"
    )
    assert fighter_requirement.kind == "hero_object"
    assert fighter_requirement.scale.preferred_dimensions_m == (18.0, 12.0, 4.5)
    assert fighter_requirement.composition.strategy_order == (
        "catalog",
        "composed",
        "procedural",
    )
    repair_bays = next(
        item for item in graph.requirements if item.subject == "repair bay"
    )
    assert repair_bays.quantity.value == 10
    assert {
        "spaceship dock",
        "space fighter",
        "repair bay",
        "space opening",
        "fighter parts",
        "repair machines",
        "station doors",
        "crew-quarter doors",
    } <= {item.subject for item in graph.requirements}
    assert all(item.strength == "hard" for item in graph.requirements)


def test_merge_rejects_model_quantity_downgrade():
    prompt = "A test volume with 10 calibration cradles."
    candidates = literal_candidates_from_prompt(prompt)
    counted = _candidate(candidates, "10 calibration")
    bad = _proposal(
        counted,
        subject="calibration cradle",
        quantity=_quantity_for(counted, changed_value=7),
    )

    with pytest.raises(RequirementMergeError, match="altered explicit quantity"):
        merge_requirement_interpretations(
            prompt,
            candidates,
            RequirementInterpretationBatch(
                composition=_composition(),
                requirements=(bad,),
                analysis_summary="Incorrectly reduced the count.",
            ),
        )


def test_model_omission_is_retained_unclassified_or_rejected_strictly():
    prompt = "A chamber with a central phase lattice and seven service alcoves."
    candidates = literal_candidates_from_prompt(prompt)
    partial = RequirementInterpretationBatch(
        composition=_composition(),
        requirements=(_proposal(candidates[0], kind="scene_type"),),
        analysis_summary="Incomplete output.",
    )

    graph = merge_requirement_interpretations(prompt, candidates, partial)
    assert graph.analysis_status == "partial"
    assert any(item.kind == "unclassified" for item in graph.requirements)
    assert all(
        candidate.candidate_id in {r.source_candidate_id for r in graph.requirements}
        for candidate in candidates
    )
    with pytest.raises(RequirementMergeError, match="uninterpreted"):
        merge_requirement_interpretations(
            prompt, candidates, partial, allow_partial=False
        )


def test_modality_is_literal_and_cannot_be_softened_by_model():
    prompt = "A clean laboratory without five flux cradles. It might include a prism."
    candidates = literal_candidates_from_prompt(prompt)
    forbidden = _candidate(candidates, "without five")
    optional = _candidate(candidates, "might include")
    graph = merge_requirement_interpretations(prompt, candidates, _batch(candidates))
    by_candidate = {item.source_candidate_id: item for item in graph.requirements}

    assert forbidden.modality == "forbidden"
    assert by_candidate[forbidden.candidate_id].polarity == "forbidden"
    assert by_candidate[forbidden.candidate_id].strength == "hard"
    assert optional.modality == "optional"
    assert by_candidate[optional.candidate_id].strength == "soft"


def test_abbreviation_periods_do_not_split_literal_obligations():
    prompt = "A station with many doors to e.g. crew quarters, etc."
    evidence = [
        candidate.evidence.text for candidate in literal_candidates_from_prompt(prompt)
    ]

    assert any("e.g. crew quarters" in text for text in evidence)
    assert "g" not in evidence


def test_graph_is_stable_frozen_hashable_and_round_trips(tmp_path):
    candidates = literal_candidates_from_prompt(DOCK_PROMPT)
    first = merge_requirement_interpretations(
        DOCK_PROMPT, candidates, _batch(candidates)
    )
    second = merge_requirement_interpretations(
        DOCK_PROMPT, candidates, _batch(candidates)
    )
    output = tmp_path / "scene_requirement_graph.json"

    assert first == second
    assert first.content_hash == second.content_hash
    persist_requirement_graph(first, output)
    restored = load_requirement_graph(output)
    assert restored == first
    assert SceneRequirementGraph.model_validate_json(first.model_dump_json()) == first


def test_shadow_audit_marks_unclassified_fallback_ambiguous(tmp_path):
    graph = requirement_graph_from_prompt(DOCK_PROMPT, analysis_error="route failed")
    blueprint = blueprint_from_prompt(DOCK_PROMPT)
    scene = SimpleNamespace(objects={})
    house_layout = SimpleNamespace(doors=[], portals=[])

    audit = audit_requirement_graph(
        graph,
        blueprint=blueprint,
        scene=scene,
        house_layout=house_layout,
    )
    assert audit.ambiguous_count == len(graph.requirements)
    assert graph.analysis_status == "unavailable"
    output = tmp_path / "semantic_shadow_audit.json"
    persist_shadow_audit(audit, output)
    assert '"mode": "shadow"' in output.read_text()


def test_structured_analyst_is_the_semantic_authority():
    prompt = "A broad test volume with three unfamiliar assemblies."
    candidates = literal_candidates_from_prompt(prompt)
    batch = _batch(candidates)

    class FakeResult:
        def final_output_as(self, output_type):
            assert output_type is RequirementInterpretationBatch
            return batch

    class FakeRunner:
        call = None

        @staticmethod
        async def run(**kwargs):
            FakeRunner.call = kwargs
            return FakeResult()

    observed, result = asyncio.run(
        analyze_requirement_candidates(
            prompt,
            candidates,
            model="test-semantic-model",
            runner=FakeRunner,
        )
    )

    assert observed == batch
    assert isinstance(result, FakeResult)
    assert (
        FakeRunner.call["starting_agent"].output_type is RequirementInterpretationBatch
    )
    assert "preferred_dimensions_m" in str(
        RequirementInterpretationBatch.model_json_schema()
    )
    assert "immutable_candidates" in FakeRunner.call["input"]


def test_floor_plan_calls_semantic_llm_before_blueprint_compilation(
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
    )
    agent.logger = SimpleNamespace(output_dir=tmp_path)
    agent._reset_workflow_budget = lambda: None
    agent._create_run_config = lambda: None
    agent._get_model_settings = lambda **_kwargs: None

    async def fake_analysis(prompt, candidates, **kwargs):
        events.append(("analyze", prompt, kwargs["model"]))
        return _batch(candidates), SimpleNamespace()

    class BlueprintBoundary(RuntimeError):
        pass

    def stop_at_blueprint(*_args, **_kwargs):
        events.append(("blueprint",))
        raise BlueprintBoundary

    monkeypatch.setattr(floor_plan_module, "load_design_system_from_env", lambda: None)
    monkeypatch.setattr(
        floor_plan_module, "analyze_requirement_candidates", fake_analysis
    )
    monkeypatch.setattr(floor_plan_module, "log_agent_usage", lambda **_kwargs: None)
    monkeypatch.setattr(floor_plan_module, "blueprint_from_prompt", stop_at_blueprint)

    with pytest.raises(BlueprintBoundary):
        asyncio.run(agent.generate_house_layout("A novel chamber.", tmp_path / "floor"))

    assert [event[0] for event in events] == ["analyze", "blueprint"]
    persisted = load_requirement_graph(tmp_path / "scene_requirement_graph.json")
    assert persisted.analysis_status == "complete"
    assert persisted.analysis_model == "semantic-test-model"
