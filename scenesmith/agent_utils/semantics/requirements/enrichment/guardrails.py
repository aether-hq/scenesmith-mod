"""Deterministic source-truth guardrails for advisory prompt enrichment."""

from __future__ import annotations

from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirement,
    SceneRequirementGraph,
)


def requirement_guardrail(requirement: SceneRequirement) -> str:
    quantity = requirement.quantity
    if quantity.mode == "exact":
        quantity_text = f"exactly {quantity.value} total"
    elif quantity.mode == "minimum":
        quantity_text = f"at least {quantity.value} total"
    else:
        quantity_text = f"the source quantity {quantity.label!r} of"
    scale_text = ""
    if requirement.scale is not None:
        dimensions = (
            requirement.scale.preferred_dimensions_m
            or requirement.scale.minimum_dimensions_m
        )
        if dimensions is not None:
            scale_text = " at " + " x ".join(f"{value:g}m" for value in dimensions)
    return (
        f"[{requirement.requirement_id}] Preserve {quantity_text} "
        f"{requirement.subject}{scale_text}; do not multiply, merge, or omit it."
    )


def hard_requirement_guardrails(
    graph: SceneRequirementGraph,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "requirement_id": requirement.requirement_id,
            "subject": requirement.subject,
            "quantity_mode": requirement.quantity.mode,
            "quantity_value": requirement.quantity.value,
            "quantity_label": requirement.quantity.label,
            "guardrail": requirement_guardrail(requirement),
        }
        for requirement in graph.requirements
        if requirement.strength == "hard" and requirement.polarity == "required"
    )


def format_hard_requirement_guardrails(graph: SceneRequirementGraph) -> str:
    return "\n".join(
        requirement_guardrail(requirement)
        for requirement in graph.requirements
        if requirement.strength == "hard" and requirement.polarity == "required"
    )
