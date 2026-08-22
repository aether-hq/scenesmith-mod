"""Graph-bound semantic prompt enrichment for complete, non-generic scenes.

The immutable requirement graph remains the source of truth.  This module adds
advisory operational and visual context in two bounded model passes:

1. Enrich the complete scene and every hard positive requirement.
2. After spatial compilation, author one unique brief for every repeated
   furniture role and opening in the accepted blueprint.

Both passes have deterministic fallbacks.  Enrichment may improve construction,
but it can never add, remove, or weaken a user-authored obligation.
"""

from __future__ import annotations

import json
import re

from typing import Any, Literal, Protocol

from agents import Agent, ModelSettings, RunConfig

from scenesmith.agent_utils.runtime.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.semantics.requirements.enrichment.fallbacks import (
    construction_prompt,
    fallback_requirement_prompt,
    fallback_role_enrichment,
    fallback_scene_enrichment,
    instance_artifact_ref,
    positive_hard_requirements,
)
from scenesmith.agent_utils.semantics.requirements.enrichment.guardrails import (
    format_hard_requirement_guardrails,
    hard_requirement_guardrails,
    requirement_guardrail,
)
from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_models import (
    InstancePrompt,
    RepeatedEnrichmentWireBatch,
    RepeatedRoleEnrichment,
    RepeatedRoleTarget,
    RepeatedRoleWire,
    RequirementPrompt,
    RequirementPromptWire,
    SceneEnrichmentDraft,
    SceneEnrichmentWire,
    SemanticPromptEnrichment,
    blueprint_content_hash,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirementGraph,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint


class _Runner(Protocol):
    @staticmethod
    async def run(*args: Any, **kwargs: Any) -> Any: ...


SCENE_ENRICHMENT_INSTRUCTIONS = """\
You are SceneSmith's semantic environment author. The immutable requirement graph
contains the user's exact obligations. Enrich it into a vivid, construction-ready
description of the complete real-world environment without changing any count,
scale, polarity, relationship, or source requirement.

Infer the domain, purpose, operational workflow, circulation, infrastructure,
logistics, safety systems, architectural hierarchy, and environmental storytelling
that make the requested place explain itself. Infer missing operational answers such
as how people, vehicles, materials, or products enter, move, receive service, and
leave. These additions are advisory context, never new user-authored requirements.
The hard_requirement_guardrails are authoritative over every advisory composition
note. A singular shared artifact must never become one artifact per repeated instance.

Return one requirement prompt for every hard positive requirement_id. Each prompt
must say what the artifact does, how it relates to the complete environment, and
what concrete geometry/equipment/material cues make it recognizable. Avoid generic
phrases such as "miscellaneous props" or "appropriate equipment". Do not use camera
or post-processing instructions as a substitute for environment design.

The enriched_prompt may be cinematic but must remain spatially and operationally
actionable. Keep it under 500 words. Keep all other free-text fields to one sentence,
each cue concrete, and inferred_elements to at most twelve. Output only the schema.
"""


REPEATED_ENRICHMENT_INSTRUCTIONS = """\
You are SceneSmith's repeated-instance art director. The input contains an immutable
scene graph, accepted blueprint, enriched master brief, and every blueprint target
that contains repeated furniture roles or openings.

Return exactly one target entry for every target_id and exactly instance_count
instances using zero-based indices. Give every instance a unique name, operational
function, construction description, geometry cues, equipment cues, material cues,
and relationship to the wider workflow. Preserve the shared role and design language
so the set remains coherent, but never copy a description or merely append a number.
Variation should come from function, configuration, attached infrastructure,
condition, workload, contents, and adjacency—not arbitrary color swaps. Do not alter
counts, dimensions, artifact IDs, or hard requirements. Use at most four geometry
cues, four equipment cues, and three material cues per instance. Keep every field to
one short sentence or phrase. Output only the schema.
"""


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "role"


def merge_scene_enrichment(
    graph: SceneRequirementGraph,
    wire: SceneEnrichmentWire,
    *,
    analysis_model: str | None,
) -> SceneEnrichmentDraft:
    diagnostics: list[str] = []
    expected = {
        requirement.requirement_id: requirement
        for requirement in positive_hard_requirements(graph)
    }
    authored: dict[str, RequirementPromptWire] = {}
    for prompt in wire.requirement_prompts:
        if prompt.requirement_id not in expected:
            diagnostics.append(
                f"model invented requirement prompt {prompt.requirement_id}"
            )
            continue
        if prompt.requirement_id in authored:
            diagnostics.append(
                f"model duplicated requirement prompt {prompt.requirement_id}"
            )
            continue
        authored[prompt.requirement_id] = prompt

    prompts: list[RequirementPrompt] = []
    for requirement_id, requirement in expected.items():
        prompt = authored.get(requirement_id)
        if prompt is None or not prompt.construction_prompt.strip():
            diagnostics.append(f"model omitted requirement prompt {requirement_id}")
            prompts.append(fallback_requirement_prompt(requirement))
            continue
        prompts.append(
            RequirementPrompt(
                requirement_id=requirement_id,
                subject=requirement.subject,
                operational_role=(
                    prompt.operational_role.strip()
                    or fallback_requirement_prompt(requirement).operational_role
                ),
                visual_identity=(
                    prompt.visual_identity.strip()
                    or fallback_requirement_prompt(requirement).visual_identity
                ),
                construction_prompt=(
                    f"{requirement_guardrail(requirement)} "
                    f"{prompt.construction_prompt.strip()}"
                ),
                source="model",
            )
        )

    fallback = fallback_scene_enrichment(
        graph,
        analysis_model=analysis_model,
        diagnostic="model field fallback",
    )
    enriched_prompt = wire.enriched_prompt.strip() or fallback.enriched_prompt
    if wire.enriched_prompt.strip():
        enriched_prompt += "\n\nNON-NEGOTIABLE SOURCE GUARDRAILS:\n"
        enriched_prompt += format_hard_requirement_guardrails(graph)
    return SceneEnrichmentDraft(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        source_prompt=graph.source_prompt,
        domain_context=wire.domain_context.strip() or fallback.domain_context,
        scene_purpose=wire.scene_purpose.strip() or fallback.scene_purpose,
        operational_logic=wire.operational_logic.strip() or fallback.operational_logic,
        spatial_logic=wire.spatial_logic.strip() or fallback.spatial_logic,
        visual_language=wire.visual_language.strip() or fallback.visual_language,
        enriched_prompt=enriched_prompt,
        inferred_elements=wire.inferred_elements or fallback.inferred_elements,
        requirement_prompts=tuple(prompts),
        analysis_model=analysis_model,
        analysis_status="partial" if diagnostics else "complete",
        diagnostics=tuple(diagnostics),
    )


async def analyze_scene_enrichment(
    graph: SceneRequirementGraph,
    *,
    model: str,
    run_config: RunConfig | None = None,
    model_settings: ModelSettings | None = None,
    runner: type[_Runner] = BoundedRunner,
) -> tuple[SceneEnrichmentDraft, Any]:
    analyst = Agent(
        name="Scene Semantic Environment Enricher",
        model=model,
        instructions=SCENE_ENRICHMENT_INSTRUCTIONS,
        output_type=SceneEnrichmentWire,
        model_settings=model_settings or ModelSettings(),
    )
    result = await runner.run(
        starting_agent=analyst,
        input=json.dumps(
            {
                "immutable_requirement_graph": graph.model_dump(mode="json"),
                "hard_requirement_guardrails": hard_requirement_guardrails(graph),
            },
            indent=2,
        ),
        max_turns=1,
        run_config=run_config,
        timeout_seconds=agent_run_timeout_seconds("scene_enrichment", max_turns=1),
    )
    wire = result.final_output_as(SceneEnrichmentWire)
    return merge_scene_enrichment(graph, wire, analysis_model=model), result


def collect_repeated_role_targets(
    graph: SceneRequirementGraph,
    blueprint: SceneBlueprint,
    scene: SceneEnrichmentDraft,
) -> tuple[RepeatedRoleTarget, ...]:
    requirement_prompts = {
        prompt.requirement_id: prompt.construction_prompt
        for prompt in scene.requirement_prompts
    }
    groups = {group.group_id: group for group in blueprint.furniture_groups}
    targets: list[RepeatedRoleTarget] = []
    claimed_roles: set[tuple[str, str]] = set()

    for constraint in blueprint.constraints:
        requirement_id = str(constraint.parameters.get("requirement_id") or "") or None
        role_key = str(constraint.parameters.get("role_key") or "") or None
        if role_key is not None:
            matching_groups = [
                groups[artifact_id]
                for artifact_id in constraint.target_ids
                if artifact_id in groups and role_key in groups[artifact_id].roles
            ]
            count = sum(group.roles[role_key] for group in matching_groups)
            if count >= 2:
                artifact_ids = tuple(group.group_id for group in matching_groups)
                target_id = f"repeat-{_slug(requirement_id or constraint.constraint_id)}-{_slug(role_key)}"
                targets.append(
                    RepeatedRoleTarget(
                        target_id=target_id,
                        requirement_id=requirement_id,
                        subject=role_key.replace("_", " "),
                        artifact_kind="furniture_role",
                        artifact_ids=artifact_ids,
                        role_key=role_key,
                        instance_count=count,
                        shared_prompt=requirement_prompts.get(
                            requirement_id or "",
                            f"Construct a coherent set of {role_key.replace('_', ' ')} instances.",
                        ),
                    )
                )
                claimed_roles.update(
                    (group.group_id, role_key) for group in matching_groups
                )
        elif constraint.kind == "semantic_opening" and len(constraint.target_ids) >= 2:
            subject = next(
                (
                    requirement.subject
                    for requirement in graph.requirements
                    if requirement.requirement_id == requirement_id
                ),
                "repeated opening",
            )
            targets.append(
                RepeatedRoleTarget(
                    target_id=f"repeat-{_slug(requirement_id or constraint.constraint_id)}-opening",
                    requirement_id=requirement_id,
                    subject=subject,
                    artifact_kind="opening",
                    artifact_ids=constraint.target_ids,
                    role_key=None,
                    instance_count=len(constraint.target_ids),
                    shared_prompt=requirement_prompts.get(
                        requirement_id or "",
                        f"Construct a coherent set of {subject} openings.",
                    ),
                )
            )

    for group in blueprint.furniture_groups:
        for role_key, count in group.roles.items():
            if count < 2 or (group.group_id, role_key) in claimed_roles:
                continue
            targets.append(
                RepeatedRoleTarget(
                    target_id=f"repeat-{_slug(group.group_id)}-{_slug(role_key)}",
                    requirement_id=None,
                    subject=role_key.replace("_", " "),
                    artifact_kind="furniture_role",
                    artifact_ids=(group.group_id,),
                    role_key=role_key,
                    instance_count=count,
                    shared_prompt=(
                        f"Construct {count} coordinated {role_key.replace('_', ' ')} "
                        f"instances for {group.name}."
                    ),
                )
            )
    return tuple(sorted(targets, key=lambda item: item.target_id))


async def analyze_repeated_instance_enrichment(
    graph: SceneRequirementGraph,
    blueprint: SceneBlueprint,
    scene: SceneEnrichmentDraft,
    *,
    model: str,
    run_config: RunConfig | None = None,
    model_settings: ModelSettings | None = None,
    runner: type[_Runner] = BoundedRunner,
) -> tuple[RepeatedEnrichmentWireBatch, Any | None]:
    targets = collect_repeated_role_targets(graph, blueprint, scene)
    if not targets:
        return RepeatedEnrichmentWireBatch(targets=()), None
    analyst = Agent(
        name="Scene Repeated Instance Enricher",
        model=model,
        instructions=REPEATED_ENRICHMENT_INSTRUCTIONS,
        output_type=RepeatedEnrichmentWireBatch,
        model_settings=model_settings or ModelSettings(),
    )
    result = await runner.run(
        starting_agent=analyst,
        input=json.dumps(
            {
                "immutable_requirement_graph": graph.model_dump(mode="json"),
                "accepted_blueprint": blueprint.model_dump(mode="json"),
                "enriched_scene": scene.model_dump(mode="json"),
                "repeated_targets": [
                    target.model_dump(mode="json") for target in targets
                ],
            },
            indent=2,
        ),
        max_turns=1,
        run_config=run_config,
        timeout_seconds=agent_run_timeout_seconds(
            "repeated_instance_enrichment", max_turns=1
        ),
    )
    return result.final_output_as(RepeatedEnrichmentWireBatch), result


def _merge_repeated_role(
    target: RepeatedRoleTarget,
    wire: RepeatedRoleWire,
) -> RepeatedRoleEnrichment:
    if len(wire.instances) != target.instance_count:
        raise ValueError(
            f"{target.target_id} expected {target.instance_count} descriptions, "
            f"received {len(wire.instances)}"
        )
    by_index = {item.instance_index: item for item in wire.instances}
    if set(by_index) != set(range(target.instance_count)):
        raise ValueError(f"{target.target_id} returned invalid instance indices")
    instances = tuple(
        InstancePrompt(
            instance_id=f"{target.target_id}-{index:03d}",
            instance_index=index,
            artifact_ref=instance_artifact_ref(target, index),
            name=by_index[index].name.strip(),
            function=by_index[index].function.strip(),
            description=by_index[index].description.strip(),
            geometry_cues=by_index[index].geometry_cues,
            equipment_cues=by_index[index].equipment_cues,
            material_cues=by_index[index].material_cues,
            operational_relationship=by_index[index].operational_relationship.strip(),
            construction_prompt=construction_prompt(target, by_index[index]),
            source="model",
        )
        for index in range(target.instance_count)
    )
    return RepeatedRoleEnrichment(
        target_id=target.target_id,
        requirement_id=target.requirement_id,
        subject=target.subject,
        artifact_kind=target.artifact_kind,
        artifact_ids=target.artifact_ids,
        role_key=target.role_key,
        shared_design_language=(
            wire.shared_design_language.strip() or target.shared_prompt
        ),
        instances=instances,
    )


def compose_complete_prompt(
    graph: SceneRequirementGraph,
    scene: SceneEnrichmentDraft,
    repeated_roles: tuple[RepeatedRoleEnrichment, ...],
) -> str:
    lines = [
        "ORIGINAL USER REQUEST — immutable source truth:",
        graph.source_prompt,
        "",
        "NON-NEGOTIABLE SOURCE GUARDRAILS — override every inference:",
        format_hard_requirement_guardrails(graph),
        "",
        "ENRICHED COMPLETE-SCENE BRIEF — inferred construction context:",
        scene.enriched_prompt,
        "",
        f"DOMAIN: {scene.domain_context}",
        f"PURPOSE: {scene.scene_purpose}",
        f"OPERATIONAL LOGIC: {scene.operational_logic}",
        f"SPATIAL LOGIC: {scene.spatial_logic}",
        f"VISUAL LANGUAGE: {scene.visual_language}",
        "",
        "REQUIREMENT CONSTRUCTION SUBPROMPTS:",
    ]
    lines.extend(
        f"- [{prompt.requirement_id}] {prompt.subject}: {prompt.construction_prompt}"
        for prompt in scene.requirement_prompts
    )
    if repeated_roles:
        lines.extend(("", "UNIQUE REPEATED-INSTANCE SUBPROMPTS:"))
        for role in repeated_roles:
            lines.append(
                f"- {role.subject} ({len(role.instances)} instances; shared: "
                f"{role.shared_design_language})"
            )
            lines.extend(
                f"  - [{instance.instance_id}] {instance.construction_prompt}"
                for instance in role.instances
            )
    return "\n".join(lines)


def validate_scene_enrichment(
    scene: SceneEnrichmentDraft,
    graph: SceneRequirementGraph,
) -> None:
    """Verify that advisory scene prose preserves every graph obligation."""

    if scene.graph_id != graph.graph_id or scene.graph_hash != graph.content_hash:
        raise ValueError("scene enrichment does not match requirement graph")
    if scene.source_prompt != graph.source_prompt:
        raise ValueError("scene enrichment changed the immutable source prompt")
    expected = {
        requirement.requirement_id for requirement in positive_hard_requirements(graph)
    }
    observed = {prompt.requirement_id for prompt in scene.requirement_prompts}
    if observed != expected:
        raise ValueError(
            "scene enrichment does not cover every hard positive requirement"
        )


def validate_prompt_enrichment(
    enrichment: SemanticPromptEnrichment,
    graph: SceneRequirementGraph,
    blueprint: SceneBlueprint,
) -> None:
    """Verify graph, blueprint, and repeated-instance integrity."""

    validate_scene_enrichment(enrichment.scene, graph)
    if (
        enrichment.graph_id != graph.graph_id
        or enrichment.graph_hash != graph.content_hash
    ):
        raise ValueError("prompt enrichment does not match requirement graph")
    if enrichment.source_prompt != graph.source_prompt:
        raise ValueError("prompt enrichment changed the immutable source prompt")
    if enrichment.blueprint_id != blueprint.blueprint_id:
        raise ValueError("prompt enrichment does not match blueprint ID")
    if enrichment.blueprint_hash != blueprint_content_hash(blueprint):
        raise ValueError("prompt enrichment does not match blueprint content")
    expected = collect_repeated_role_targets(graph, blueprint, enrichment.scene)
    expected_counts = {target.target_id: target.instance_count for target in expected}
    observed_counts = {
        role.target_id: len(role.instances) for role in enrichment.repeated_roles
    }
    if observed_counts != expected_counts:
        raise ValueError("prompt enrichment changed repeated blueprint instance counts")


def finalize_prompt_enrichment(
    graph: SceneRequirementGraph,
    blueprint: SceneBlueprint,
    scene: SceneEnrichmentDraft,
    repeated_wire: RepeatedEnrichmentWireBatch | None,
    *,
    repeated_diagnostic: str | None = None,
) -> SemanticPromptEnrichment:
    validate_scene_enrichment(scene, graph)
    targets = collect_repeated_role_targets(graph, blueprint, scene)
    diagnostics = list(scene.diagnostics)
    if repeated_diagnostic:
        diagnostics.append(repeated_diagnostic)
    authored: dict[str, RepeatedRoleWire] = {}
    if repeated_wire is not None:
        for wire in repeated_wire.targets:
            if wire.target_id in authored:
                diagnostics.append(f"model duplicated repeated target {wire.target_id}")
            else:
                authored[wire.target_id] = wire
    expected_ids = {target.target_id for target in targets}
    for unknown in sorted(set(authored) - expected_ids):
        diagnostics.append(f"model invented repeated target {unknown}")

    repeated_roles: list[RepeatedRoleEnrichment] = []
    for target in targets:
        wire = authored.get(target.target_id)
        if wire is None:
            diagnostics.append(f"model omitted repeated target {target.target_id}")
            repeated_roles.append(
                fallback_role_enrichment(target, domain_context=scene.domain_context)
            )
            continue
        try:
            repeated_roles.append(_merge_repeated_role(target, wire))
        except ValueError as exc:
            diagnostics.append(str(exc))
            repeated_roles.append(
                fallback_role_enrichment(target, domain_context=scene.domain_context)
            )

    repeated_tuple = tuple(repeated_roles)
    if scene.analysis_status == "deterministic_fallback" and all(
        all(instance.source == "deterministic_fallback" for instance in role.instances)
        for role in repeated_tuple
    ):
        status: Literal["complete", "partial", "deterministic_fallback"] = (
            "deterministic_fallback"
        )
    elif diagnostics:
        status = "partial"
    else:
        status = "complete"
    enrichment = SemanticPromptEnrichment(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        blueprint_id=blueprint.blueprint_id,
        blueprint_hash=blueprint_content_hash(blueprint),
        source_prompt=graph.source_prompt,
        scene=scene,
        repeated_roles=repeated_tuple,
        complete_prompt=compose_complete_prompt(graph, scene, repeated_tuple),
        analysis_model=scene.analysis_model,
        analysis_status=status,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )
    validate_prompt_enrichment(enrichment, graph, blueprint)
    return enrichment
