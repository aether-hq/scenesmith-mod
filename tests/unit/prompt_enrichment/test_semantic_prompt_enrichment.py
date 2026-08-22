import asyncio
import json

from pathlib import Path
from types import SimpleNamespace

import pytest

from pydantic import ValidationError

from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_application import (
    apply_prompt_enrichment,
    load_prompt_enrichment,
    persist_prompt_enrichment,
)
from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_models import (
    InferredSceneElement,
    InstancePromptWire,
    RepeatedEnrichmentWireBatch,
    RepeatedRoleWire,
    RequirementPromptWire,
    SceneEnrichmentWire,
    blueprint_content_hash,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirementGraph,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    BlueprintConstraint,
    FurnitureGroupBlueprint,
    LevelBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
)
from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    requirement_graph_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.semantic_prompt_enrichment import (
    analyze_scene_enrichment,
    collect_repeated_role_targets,
    fallback_scene_enrichment,
    finalize_prompt_enrichment,
    merge_scene_enrichment,
)

DOCK_PROMPT = (
    "A SPACEship dock with a huge space-fighter in the middle. There are 10 "
    "repair bays, each big enough for a fighter, a giant opening into space, and "
    "then all the parts, machines, and so on are across the bay. There are "
    "massive doors into the rest of the station and many smaller doors to e.g. "
    "crew quarters, etc."
)


def _graph() -> SceneRequirementGraph:
    return requirement_graph_from_prompt(DOCK_PROMPT)


def _requirement_id(graph: SceneRequirementGraph, evidence: str) -> str:
    return next(
        requirement.requirement_id
        for requirement in graph.requirements
        if evidence in requirement.evidence.text
    )


def _blueprint(
    graph: SceneRequirementGraph,
    *,
    role: str = "repair_bay",
    count: int = 10,
    evidence: str = "repair bays",
) -> SceneBlueprint:
    level = LevelBlueprint(
        level_id="level-dock",
        name="Dock level",
        clear_height_m=40.0,
    )
    space = SpaceBlueprint(
        space_id="space-dock",
        name="Orbital fighter dock",
        room_type="fighter dock",
        level_id=level.level_id,
        dimensions_m=(220.0, 120.0),
    )
    group = FurnitureGroupBlueprint(
        group_id=f"group-{role}",
        name=role.replace("_", " ").title(),
        space_id=space.space_id,
        roles={role: count},
        density="layered",
    )
    constraint = BlueprintConstraint(
        constraint_id=f"constraint-{role}",
        kind="semantic_repeated_zone",
        target_ids=(group.group_id,),
        parameters={
            "requirement_id": _requirement_id(graph, evidence),
            "role_key": role,
            "preferred_dimensions_m": [25.0, 15.0, 20.0],
        },
        source="user",
    )
    return SceneBlueprint(
        blueprint_id=f"fighter-dock-{role}",
        source_prompt=graph.source_prompt,
        levels=(level,),
        spaces=(space,),
        furniture_groups=(group,),
        constraints=(constraint,),
        locked_ids=(group.group_id, constraint.constraint_id),
    )


def _scene_wire(graph: SceneRequirementGraph) -> SceneEnrichmentWire:
    return SceneEnrichmentWire(
        domain_context="front-line orbital carrier maintenance and launch deck",
        scene_purpose=(
            "Turn fighters around under combat tempo while separating fuel, "
            "ordnance, diagnostics, and crew circulation."
        ),
        operational_logic=(
            "Fighters enter through the shielded space aperture, cross a marked "
            "launch corridor, receive service in radial bays, and exit nose-first."
        ),
        spatial_logic=(
            "A central hero hardstand anchors ten asymmetric repair cells, with an "
            "oversight deck above and station access behind blast doors."
        ),
        visual_language=(
            "Layer soot-darkened armored ribs, numbered gantries, amber utility "
            "light, exposed fuel trunks, and hard-worn deck markings."
        ),
        enriched_prompt=(
            "Build a colossal pressure-zoned orbital fighter dock around one huge "
            "hero interceptor. Ten fighter-sized repair bays form an operational "
            "service ring, each connected to fuel, coolant, power, diagnostics, "
            "parts handling, and safe crew paths. Preserve a clear launch corridor "
            "to a giant shielded opening into space; put massive pressure doors to "
            "the station at the opposite end, smaller crew doors along protected "
            "side galleries, and an oversight deck above the traffic spine."
        ),
        inferred_elements=(
            InferredSceneElement(
                category="circulation",
                description="A full-width launch and recovery corridor reaches space.",
                rationale="The fighters need an unobstructed way to enter and leave.",
            ),
            InferredSceneElement(
                category="infrastructure",
                description="Overhead fuel, coolant, power, and extraction trunks serve bays.",
                rationale="A military repair dock requires visible service infrastructure.",
            ),
            InferredSceneElement(
                category="operations",
                description="A glazed oversight deck controls traffic and repair work.",
                rationale="Supervisors need protected sightlines over the full deck.",
            ),
        ),
        requirement_prompts=tuple(
            RequirementPromptWire(
                requirement_id=requirement.requirement_id,
                operational_role=f"Preserve the source-bound {requirement.subject} role.",
                visual_identity=f"Make {requirement.subject} immediately recognizable.",
                construction_prompt=(
                    f"Construct {requirement.subject} as a concrete part of the "
                    "fighter turnaround, launch, or station-support workflow."
                ),
            )
            for requirement in graph.requirements
        ),
    )


def _instance(index: int) -> InstancePromptWire:
    functions = (
        "battle-damage intake",
        "engine hot-section service",
        "avionics diagnostics",
        "weapons safing",
        "landing-gear alignment",
        "canopy and life-support service",
        "fuel-system isolation",
        "coolant pressure testing",
        "rapid rearm turnaround",
        "final flight-line release",
    )
    function = functions[index]
    return InstancePromptWire(
        instance_index=index,
        name=f"Bay {chr(ord('A') + index)} — {function.title()}",
        function=function,
        description=f"A fighter workcell purpose-built for {function}.",
        geometry_cues=(f"asymmetric {function} gantry", "fighter-scale service cradle"),
        equipment_cues=(f"dedicated {function} console", "retractable utility hoses"),
        material_cues=("heat-blued steel", "worn amber hazard markings"),
        operational_relationship="Feeds the central launch corridor without cross-traffic.",
    )


def test_two_pass_dock_enrichment_authors_ten_stable_unique_bays(tmp_path: Path):
    graph = _graph()
    blueprint = _blueprint(graph)
    scene = merge_scene_enrichment(graph, _scene_wire(graph), analysis_model="sonnet")
    target = collect_repeated_role_targets(graph, blueprint, scene)[0]
    repeated = RepeatedEnrichmentWireBatch(
        targets=(
            RepeatedRoleWire(
                target_id=target.target_id,
                shared_design_language="Armored carrier workcells with common utility spines.",
                instances=tuple(_instance(index) for index in range(10)),
            ),
        )
    )

    enrichment = finalize_prompt_enrichment(graph, blueprint, scene, repeated)

    assert enrichment.analysis_status == "complete"
    assert enrichment.source_prompt == DOCK_PROMPT
    assert enrichment.blueprint_hash == blueprint_content_hash(blueprint)
    assert len(enrichment.repeated_roles) == 1
    bays = enrichment.repeated_roles[0].instances
    assert len(bays) == 10
    assert len({bay.name for bay in bays}) == 10
    assert len({bay.construction_prompt for bay in bays}) == 10
    assert "launch corridor" in enrichment.complete_prompt
    assert "oversight deck" in enrichment.complete_prompt
    assert "ORIGINAL USER REQUEST" in enrichment.complete_prompt
    assert (
        "Preserve exactly 1 total a giant opening into space" in scene.enriched_prompt
    )
    assert "override every inference" in enrichment.complete_prompt

    applied = apply_prompt_enrichment(blueprint, enrichment, graph)
    assert blueprint.spaces[0].prompt == ""
    assert "SEMANTIC PROMPT ENRICHMENT" in applied.spaces[0].prompt
    prompts = applied.constraints[0].parameters["instance_prompts"]
    assert [prompt["instance_index"] for prompt in prompts] == list(range(10))
    assert len({prompt["construction_prompt"] for prompt in prompts}) == 10

    output_path = tmp_path / "semantic_prompt_enrichment.json"
    persist_prompt_enrichment(enrichment, output_path)
    assert load_prompt_enrichment(output_path) == enrichment


def test_scene_analysis_sends_explicit_hard_guardrails_to_the_model():
    graph = _graph()
    calls = []

    class _Runner:
        @staticmethod
        async def run(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(final_output_as=lambda _: _scene_wire(graph))

    scene, _ = asyncio.run(
        analyze_scene_enrichment(graph, model="sonnet", runner=_Runner)
    )

    payload = json.loads(calls[0]["input"])
    opening = next(
        item
        for item in payload["hard_requirement_guardrails"]
        if "opening into space" in item["subject"]
    )
    assert opening["quantity_mode"] == "exact"
    assert opening["quantity_value"] == 1
    assert "exactly 1 total" in opening["guardrail"]
    assert "exactly 1 total" in scene.enriched_prompt


def test_malformed_large_role_degrades_to_unique_deterministic_variants():
    graph = _graph()
    blueprint = _blueprint(
        graph,
        role="repair_parts_rack",
        count=20,
        evidence="parts",
    )
    scene = fallback_scene_enrichment(
        graph,
        analysis_model=None,
        diagnostic="semantic model unavailable",
    )
    target = collect_repeated_role_targets(graph, blueprint, scene)[0]
    malformed = RepeatedEnrichmentWireBatch(
        targets=(
            RepeatedRoleWire(
                target_id=target.target_id,
                shared_design_language="Shared armored storage frames.",
                instances=(_instance(0), _instance(1)),
            ),
        )
    )

    enrichment = finalize_prompt_enrichment(graph, blueprint, scene, malformed)

    instances = enrichment.repeated_roles[0].instances
    assert len(instances) == 20
    assert len({instance.name for instance in instances}) == 20
    assert len({instance.construction_prompt for instance in instances}) == 20
    assert all(instance.source == "deterministic_fallback" for instance in instances)
    assert any("expected 20 descriptions" in item for item in enrichment.diagnostics)


def test_instance_wire_rejects_empty_or_unbounded_cues():
    with pytest.raises(ValidationError, match="at least 1 item"):
        InstancePromptWire(
            instance_index=0,
            name="Bay A",
            function="diagnostics",
            description="A diagnostic workcell.",
            geometry_cues=(),
            equipment_cues=("sensor rack",),
            material_cues=("steel",),
            operational_relationship="Feeds the release lane.",
        )
