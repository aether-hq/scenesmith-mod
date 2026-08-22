"""Deterministic expansion of compact semantic-model interpretations."""

from __future__ import annotations

import re

from scenesmith.agent_utils.semantics.requirements.requirement_graph.literal_parsing import (
    _source_supports_relation,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    CompositionPlan,
    FulfillmentStrategy,
    InterpretedRequirementKind,
    LiteralObligationCandidate,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementInterpretationWire,
    RequirementInterpretationWireBatch,
    RequirementMergeError,
    RequirementQuantity,
    RequirementRelation,
    RequirementScale,
    TopologyOpinion,
    VerificationPolicy,
    VerificationStage,
)


def _strategy_order(
    recommended: FulfillmentStrategy,
) -> tuple[FulfillmentStrategy, ...]:
    return (recommended,) + tuple(
        strategy
        for strategy in ("catalog", "composed", "procedural")
        if strategy != recommended
    )


def _wire_quantity(
    candidate: LiteralObligationCandidate,
    wire: RequirementInterpretationWire,
) -> RequirementQuantity:
    # Model-authored counts may clarify an explicit numeric quantity, but they
    # cannot turn vague prose or grammatical plurality into a new hard count.
    # A few source-bound linguistic forms have deterministic lower bounds;
    # density-specific providers remain responsible for stronger calibrated
    # contracts such as library book rows and room-kit populations.
    interpreted_minimum = None
    qualitative_minimums = {
        "multiple": 2,
        "several": 3,
        "many": 3,
        "hundreds": 3,
        "thousands": 3,
        "a bunch of": 3,
    }
    if wire.kind == "level" and re.search(
        r"\bmulti[- ]?level\b", candidate.evidence.text, flags=re.IGNORECASE
    ):
        interpreted_minimum = 2
    if wire.source_quantity_id:
        explicit = next(
            (
                quantity
                for quantity in candidate.explicit_quantities
                if quantity.quantity_id == wire.source_quantity_id
            ),
            None,
        )
        if explicit is not None:
            if explicit.mode == "qualitative":
                interpreted_minimum = qualitative_minimums.get(explicit.label)
            return explicit.as_requirement_quantity().model_copy(
                update={"interpreted_minimum": interpreted_minimum}
            )
    return RequirementQuantity(
        mode="qualitative",
        label="present",
        interpreted_minimum=interpreted_minimum,
    )


_SOURCE_OPENING_RE = re.compile(
    r"\b(?:door|window|opening|portal|aperture)s?\b",
    flags=re.IGNORECASE,
)


def _source_bound_kind(
    candidate: LiteralObligationCandidate,
    interpreted_kind: InterpretedRequirementKind,
) -> InterpretedRequirementKind:
    """Normalize an objective ontology mismatch without changing semantics."""

    if interpreted_kind == "connector" and _SOURCE_OPENING_RE.search(
        candidate.evidence.text
    ):
        return "opening"
    return interpreted_kind


_EXPLICIT_METRIC_DIMENSION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m|met(?:er|re)s?|ft|feet|foot)\b",
    flags=re.IGNORECASE,
)


def _wire_scale(
    candidate: LiteralObligationCandidate,
    wire: RequirementInterpretationWire,
) -> RequirementScale | None:
    if not any(
        (
            wire.scale_label,
            wire.scale_relative_to,
            wire.minimum_dimensions_m,
            wire.preferred_dimensions_m,
            wire.clearance_m is not None,
        )
    ):
        return None
    minimum_dimensions_m = wire.minimum_dimensions_m
    if not _EXPLICIT_METRIC_DIMENSION_RE.search(candidate.evidence.text):
        # Qualitative scale is a composition target, not authority to invent a
        # hard metric beyond the configured construction envelope. Explicit
        # dimensions in the source remain immutable minima.
        minimum_dimensions_m = None
    return RequirementScale(
        qualitative_label=wire.scale_label or "measured",
        relative_to=wire.scale_relative_to,
        minimum_dimensions_m=minimum_dimensions_m,
        preferred_dimensions_m=wire.preferred_dimensions_m,
        clearance_m=wire.clearance_m,
        rationale="Semantic scale envelope derived from the source-bound clause.",
    )


def _wire_verification(
    wire: RequirementInterpretationWire,
    quantity: RequirementQuantity,
    scale: RequirementScale | None,
) -> VerificationPolicy:
    if quantity.mode == "exact":
        quantity_criterion = (
            f"Artifact evidence proves exactly {quantity.value} instances"
        )
    elif quantity.mode == "minimum":
        quantity_criterion = (
            f"Artifact evidence proves at least {quantity.value} instances"
        )
    elif quantity.interpreted_minimum is not None:
        quantity_criterion = (
            "Artifact evidence proves at least "
            f"{quantity.interpreted_minimum} instances"
        )
    else:
        quantity_criterion = "Artifact evidence proves the requested subject is present"
    criteria = [quantity_criterion]
    if scale is not None:
        criteria.append("Artifact dimensions satisfy the recorded scale envelope")
    if wire.relations:
        criteria.append(
            "Recorded source relationships are satisfied by artifact evidence"
        )
    stage: VerificationStage = (
        "topology"
        if wire.kind
        in {"level", "repeated_zone", "opening", "connector", "spatial_constraint"}
        else "semantic"
    )
    return VerificationPolicy(
        stage=stage,
        method="deterministic_requirement_artifact_evidence",
        measurable_criteria=tuple(criteria),
    )


def expand_requirement_interpretations(
    candidates: tuple[LiteralObligationCandidate, ...],
    wire_batch: RequirementInterpretationWireBatch,
) -> RequirementInterpretationBatch:
    """Expand the bounded model wire format into the durable rich contract."""

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    source_text = " ".join(candidate.evidence.text for candidate in candidates)
    proposals: list[RequirementInterpretationProposal] = []
    for wire in wire_batch.requirements:
        candidate = candidates_by_id.get(wire.candidate_id)
        if candidate is None:
            raise RequirementMergeError(
                f"model returned unknown candidate_id {wire.candidate_id}"
            )
        wire = wire.model_copy(
            update={"kind": _source_bound_kind(candidate, wire.kind)}
        )
        quantity = _wire_quantity(candidate, wire)
        scale = _wire_scale(candidate, wire)
        rationale = "The semantic model classified this immutable source clause."
        proposals.append(
            RequirementInterpretationProposal(
                candidate_id=wire.candidate_id,
                subject=wire.subject,
                kind=wire.kind,
                source_quantity_id=wire.source_quantity_id,
                quantity=quantity,
                scale=scale,
                relations=tuple(
                    RequirementRelation(
                        predicate=relation.predicate,
                        target=relation.target,
                        rationale="Source-bound semantic relationship.",
                    )
                    for relation in wire.relations
                    if _source_supports_relation(relation, source_text)
                ),
                topology=TopologyOpinion(
                    role=f"{wire.kind}: {wire.subject}",
                    enclosure="scene composition",
                    adjacency=tuple(relation.target for relation in wire.relations),
                    circulation="Preserve stated connections, adjacency, and clearance.",
                    rationale=rationale,
                ),
                composition=CompositionPlan(
                    recommended_strategy=wire.recommended_strategy,
                    strategy_order=_strategy_order(wire.recommended_strategy),
                    procedural_geometry=wire.fallback_construction,
                    arrangement=wire.arrangement,
                    rationale=rationale,
                ),
                verification=_wire_verification(wire, quantity, scale),
                interpretation_rationale=rationale,
            )
        )
    return RequirementInterpretationBatch(
        schema_version=wire_batch.schema_version,
        composition=wire_batch.composition,
        requirements=tuple(proposals),
        analysis_summary=wire_batch.analysis_summary,
    )
