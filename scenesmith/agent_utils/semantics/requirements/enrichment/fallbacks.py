"""Publishable deterministic fallbacks for semantic prompt enrichment."""

from __future__ import annotations

from scenesmith.agent_utils.semantics.requirements.enrichment.guardrails import (
    format_hard_requirement_guardrails,
    requirement_guardrail,
)
from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_models import (
    InferredSceneElement,
    InstancePrompt,
    InstancePromptWire,
    RepeatedRoleEnrichment,
    RepeatedRoleTarget,
    RequirementPrompt,
    SceneEnrichmentDraft,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirement,
    SceneRequirementGraph,
)

_FALLBACK_FUNCTIONS = (
    "arrival and intake",
    "primary service",
    "specialist diagnostics",
    "repair and adjustment",
    "materials staging",
    "quality verification",
    "high-throughput operations",
    "emergency response",
    "training and overflow",
    "final checkout and release",
)
_FALLBACK_FEATURES = (
    "an overhead traveling gantry",
    "a wall-fed utility spine",
    "a recessed floor service trench",
    "an articulated side service boom",
    "a raised inspection platform",
    "a protected diagnostic console bank",
    "a compact parts-transfer rail",
    "an emergency isolation station",
    "a modular scaffolding frame",
    "a marked handoff threshold",
)
_FALLBACK_LOCATIONS = (
    "port-forward",
    "port-midship",
    "port-aft",
    "aft transfer",
    "starboard-aft",
    "starboard-midship",
    "starboard-forward",
    "launch-side",
    "station-side",
    "central logistics",
)


def positive_hard_requirements(
    graph: SceneRequirementGraph,
) -> tuple[SceneRequirement, ...]:
    return tuple(
        requirement
        for requirement in graph.requirements
        if requirement.strength == "hard" and requirement.polarity == "required"
    )


def fallback_requirement_prompt(requirement: SceneRequirement) -> RequirementPrompt:
    arrangement = (
        requirement.composition.arrangement
        if requirement.composition is not None
        else "Place it where its intended role remains legible and usable."
    )
    construction = (
        requirement.composition.procedural_geometry
        if requirement.composition is not None
        and requirement.composition.procedural_geometry
        else f"Construct recognizable {requirement.subject} geometry."
    )
    scale = ""
    if requirement.scale is not None:
        scale = f" Preserve its {requirement.scale.qualitative_label} scale."
    return RequirementPrompt(
        requirement_id=requirement.requirement_id,
        subject=requirement.subject,
        operational_role=(
            requirement.topology.role
            if requirement.topology is not None
            else f"required {requirement.kind}"
        ),
        visual_identity=construction,
        construction_prompt=(
            f"{requirement_guardrail(requirement)} Build "
            f"{requirement.subject} as {construction} {arrangement}.{scale}"
        ),
        source="deterministic_fallback",
    )


def fallback_scene_enrichment(
    graph: SceneRequirementGraph,
    *,
    analysis_model: str | None,
    diagnostic: str,
) -> SceneEnrichmentDraft:
    composition = graph.composition
    domain = (
        composition.scene_type if composition is not None else "requested environment"
    )
    purpose = (
        composition.composition_summary
        if composition is not None
        else "Fulfill every immutable source obligation in one coherent environment."
    )
    operational = (
        composition.circulation_summary
        if composition is not None
        else "Keep every requested artifact connected by usable circulation."
    )
    spatial = (
        composition.topology_summary
        if composition is not None
        else "Organize required artifacts according to their source-bound relations."
    )
    visual = (
        f"Prioritize {', '.join(composition.focal_hierarchy)}."
        if composition is not None and composition.focal_hierarchy
        else "Give every required artifact a concrete and readable visual identity."
    )
    enriched_prompt = (
        " ".join(
            (
                graph.source_prompt,
                f"Treat the scene as {domain}.",
                purpose,
                operational,
                spatial,
                visual,
            )
        )
        + "\n\nNON-NEGOTIABLE SOURCE GUARDRAILS:\n"
        + format_hard_requirement_guardrails(graph)
    )
    return SceneEnrichmentDraft(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        source_prompt=graph.source_prompt,
        domain_context=domain,
        scene_purpose=purpose,
        operational_logic=operational,
        spatial_logic=spatial,
        visual_language=visual,
        enriched_prompt=enriched_prompt,
        inferred_elements=(
            InferredSceneElement(
                category="operations",
                description=purpose,
                rationale="Deterministically projected from the scene composition.",
            ),
            InferredSceneElement(
                category="circulation",
                description=operational,
                rationale="Deterministically projected from the circulation summary.",
            ),
            InferredSceneElement(
                category="architecture",
                description=spatial,
                rationale="Deterministically projected from the topology summary.",
            ),
        ),
        requirement_prompts=tuple(
            fallback_requirement_prompt(requirement)
            for requirement in positive_hard_requirements(graph)
        ),
        analysis_model=analysis_model,
        analysis_status="deterministic_fallback",
        diagnostics=(diagnostic,),
    )


def instance_artifact_ref(target: RepeatedRoleTarget, index: int) -> str:
    if target.artifact_kind == "opening" and index < len(target.artifact_ids):
        return target.artifact_ids[index]
    return f"{target.artifact_ids[index % len(target.artifact_ids)]}:{target.role_key}:{index}"


def construction_prompt(
    target: RepeatedRoleTarget,
    wire: InstancePromptWire,
) -> str:
    cue_text = "; ".join(
        (*wire.geometry_cues, *wire.equipment_cues, *wire.material_cues)
    )
    return (
        f"{wire.name}: {wire.description} Its function is {wire.function}. "
        f"Build with {cue_text}. {wire.operational_relationship} "
        f"Preserve the shared role: {target.shared_prompt}"
    ).strip()


def fallback_role_enrichment(
    target: RepeatedRoleTarget,
    *,
    domain_context: str,
) -> RepeatedRoleEnrichment:
    instances: list[InstancePrompt] = []
    for index in range(target.instance_count):
        function = _FALLBACK_FUNCTIONS[index % len(_FALLBACK_FUNCTIONS)]
        feature = _FALLBACK_FEATURES[index % len(_FALLBACK_FEATURES)]
        cycle = index // len(_FALLBACK_FUNCTIONS)
        location = _FALLBACK_LOCATIONS[(index + cycle) % len(_FALLBACK_LOCATIONS)]
        workcell = f"{location} {function} workcell {cycle + 1}"
        name = f"{target.subject.title()} — {workcell.title()}"
        description = (
            f"A {target.subject} specialized as the {workcell} within "
            f"{domain_context}, distinguished by {feature}."
        )
        wire = InstancePromptWire(
            instance_index=index,
            name=name,
            function=function,
            description=description,
            geometry_cues=(feature, f"service orientation toward {location}"),
            equipment_cues=(f"equipment supporting {function}",),
            material_cues=("shared environmental material language",),
            operational_relationship=(
                f"Connect this instance to the scene's {function} workflow."
            ),
        )
        instances.append(
            InstancePrompt(
                instance_id=f"{target.target_id}-{index:03d}",
                instance_index=index,
                artifact_ref=instance_artifact_ref(target, index),
                name=name,
                function=function,
                description=description,
                geometry_cues=wire.geometry_cues,
                equipment_cues=wire.equipment_cues,
                material_cues=wire.material_cues,
                operational_relationship=wire.operational_relationship,
                construction_prompt=construction_prompt(target, wire),
                source="deterministic_fallback",
            )
        )
    return RepeatedRoleEnrichment(
        target_id=target.target_id,
        requirement_id=target.requirement_id,
        subject=target.subject,
        artifact_kind=target.artifact_kind,
        artifact_ids=target.artifact_ids,
        role_key=target.role_key,
        shared_design_language=target.shared_prompt,
        instances=tuple(instances),
    )
