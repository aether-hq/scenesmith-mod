"""Immutable source-bound requirement graph models."""

from __future__ import annotations

import hashlib
import os

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

CURRENT_REQUIREMENT_SCHEMA_VERSION = 2


def semantic_model_name(configured_model: str | None) -> str:
    """Resolve the stronger model reserved for semantic ownership gates."""

    override = os.environ.get("SCENESMITH_SEMANTIC_MODEL", "").strip()
    return override or str(configured_model or "").strip()


InterpretedRequirementKind = Literal[
    "scene_type",
    "level",
    "repeated_zone",
    "hero_object",
    "opening",
    "connector",
    "object_group",
    "spatial_constraint",
    "style",
]
RequirementKind = Literal[
    "scene_type",
    "level",
    "repeated_zone",
    "hero_object",
    "opening",
    "connector",
    "object_group",
    "spatial_constraint",
    "style",
    "unclassified",
]
VerificationStage = Literal[
    "blueprint",
    "topology",
    "asset",
    "placement",
    "semantic",
    "physics",
    "render",
]
FulfillmentStrategy = Literal["catalog", "composed", "procedural"]
CandidateModality = Literal["required", "forbidden", "optional"]
EnforcementDisposition = Literal["blocking", "advisory", "unresolved_blocking"]


def _enforcement_disposition(
    candidate: LiteralObligationCandidate,
    kind: RequirementKind,
) -> tuple[EnforcementDisposition, str]:
    """Apply objective rollout policy after the LLM has classified semantics."""

    if candidate.modality == "optional":
        return "advisory", "The source clause is explicitly optional."
    if kind == "unclassified":
        return (
            "unresolved_blocking",
            "A required source clause cannot publish until it is classified.",
        )
    if kind == "style":
        return (
            "advisory",
            "The style obligation is retained, but its subjective verifier is not "
            "yet calibrated for blocking enforcement.",
        )
    return (
        "blocking",
        "The source clause is required or forbidden and has objective model-supplied "
        "verification criteria.",
    )


class RequirementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptEvidence(RequirementModel):
    text: str
    start: int
    end: int

    @model_validator(mode="after")
    def validate_span(self) -> "PromptEvidence":
        if self.start < 0 or self.end <= self.start:
            raise ValueError("prompt evidence span must be positive and non-empty")
        return self


class RequirementQuantity(RequirementModel):
    mode: Literal["exact", "minimum", "qualitative"]
    value: int | None = None
    label: str = ""
    unit: str = "instances"
    source_quantity_id: str | None = None
    interpreted_minimum: int | None = None

    @model_validator(mode="after")
    def validate_quantity(self) -> "RequirementQuantity":
        if self.mode in {"exact", "minimum"} and (
            self.value is None or self.value <= 0
        ):
            raise ValueError(f"{self.mode} quantity requires a positive value")
        if self.mode == "qualitative" and not self.label:
            raise ValueError("qualitative quantity requires a label")
        if self.interpreted_minimum is not None and self.interpreted_minimum <= 0:
            raise ValueError("interpreted minimum must be positive")
        return self


class ExplicitQuantity(RequirementModel):
    quantity_id: str
    evidence: PromptEvidence
    mode: Literal["exact", "minimum", "qualitative"]
    value: int | None = None
    label: str = ""

    @model_validator(mode="after")
    def validate_quantity(self) -> "ExplicitQuantity":
        if self.mode in {"exact", "minimum"} and (
            self.value is None or self.value <= 0
        ):
            raise ValueError("literal numeric quantity must be positive")
        if self.mode == "qualitative" and not self.label:
            raise ValueError("literal qualitative quantity requires its source label")
        return self

    def as_requirement_quantity(self) -> RequirementQuantity:
        return RequirementQuantity(
            mode=self.mode,
            value=self.value,
            label=self.label,
            source_quantity_id=self.quantity_id,
        )


class LiteralObligationCandidate(RequirementModel):
    """An immutable assertive source clause, before semantic classification."""

    candidate_id: str
    evidence: PromptEvidence
    modality: CandidateModality
    explicit_quantities: tuple[ExplicitQuantity, ...] = ()


class RequirementScale(RequirementModel):
    qualitative_label: str
    relative_to: str | None = None
    minimum_dimensions_m: tuple[float, float, float] | None = None
    preferred_dimensions_m: tuple[float, float, float] | None = None
    clearance_m: float | None = None
    rationale: str

    @model_validator(mode="after")
    def validate_dimensions(self) -> "RequirementScale":
        for dimensions in (self.minimum_dimensions_m, self.preferred_dimensions_m):
            if dimensions is not None and any(value <= 0 for value in dimensions):
                raise ValueError("scale dimensions must be positive")
        if self.clearance_m is not None and self.clearance_m < 0:
            raise ValueError("clearance cannot be negative")
        return self


class RequirementRelation(RequirementModel):
    predicate: str
    target: str
    rationale: str = ""


class TopologyOpinion(RequirementModel):
    role: str
    enclosure: str
    adjacency: tuple[str, ...] = ()
    circulation: str
    rationale: str


class CompositionPlan(RequirementModel):
    recommended_strategy: FulfillmentStrategy
    strategy_order: tuple[FulfillmentStrategy, ...] = (
        "catalog",
        "composed",
        "procedural",
    )
    reusable_parts: tuple[str, ...] = ()
    procedural_geometry: str = ""
    arrangement: str
    rationale: str

    @model_validator(mode="after")
    def validate_strategy_order(self) -> "CompositionPlan":
        if set(self.strategy_order) != {"catalog", "composed", "procedural"}:
            raise ValueError(
                "strategy_order must rank catalog, composed, and procedural exactly once"
            )
        return self


class VerificationPolicy(RequirementModel):
    stage: VerificationStage
    method: str
    measurable_criteria: tuple[str, ...]

    @model_validator(mode="after")
    def validate_criteria(self) -> "VerificationPolicy":
        if not self.measurable_criteria:
            raise ValueError("verification requires at least one measurable criterion")
        return self


class RequirementInterpretationProposal(RequirementModel):
    """Provider-neutral structured semantic opinion for one source candidate."""

    candidate_id: str
    subject: str
    kind: InterpretedRequirementKind
    source_quantity_id: str | None = None
    quantity: RequirementQuantity
    scale: RequirementScale | None = None
    relations: tuple[RequirementRelation, ...] = ()
    topology: TopologyOpinion
    composition: CompositionPlan
    verification: VerificationPolicy
    qualifiers: tuple[str, ...] = ()
    interpretation_rationale: str


class SceneCompositionOpinion(RequirementModel):
    scene_type: str
    overall_scale: str
    preferred_dimensions_m: tuple[float, float, float]
    composition_summary: str
    topology_summary: str
    circulation_summary: str
    density: str
    focal_hierarchy: tuple[str, ...]

    @model_validator(mode="after")
    def validate_dimensions(self) -> "SceneCompositionOpinion":
        if any(value <= 0 for value in self.preferred_dimensions_m):
            raise ValueError("preferred scene dimensions must be positive")
        return self


class RequirementInterpretationBatch(RequirementModel):
    schema_version: Literal[1] = 1
    composition: SceneCompositionOpinion
    requirements: tuple[RequirementInterpretationProposal, ...]
    analysis_summary: str


class RequirementRelationWire(RequirementModel):
    """Compact source relationship emitted by the semantic model."""

    predicate: str
    target: str


class RequirementInterpretationWire(RequirementModel):
    """Compact semantic opinion expanded into the durable requirement model."""

    candidate_id: str
    subject: str
    kind: InterpretedRequirementKind
    source_quantity_id: str | None
    interpreted_minimum: int | None
    scale_label: str
    scale_relative_to: str | None
    minimum_dimensions_m: tuple[float, float, float] | None
    preferred_dimensions_m: tuple[float, float, float] | None
    clearance_m: float | None
    relations: tuple[RequirementRelationWire, ...]
    recommended_strategy: FulfillmentStrategy
    fallback_construction: str
    arrangement: str


class RequirementInterpretationWireBatch(RequirementModel):
    schema_version: Literal[1] = 1
    composition: SceneCompositionOpinion
    requirements: tuple[RequirementInterpretationWire, ...]
    analysis_summary: str


class SceneRequirement(RequirementModel):
    requirement_id: str
    source_candidate_id: str
    kind: RequirementKind
    subject: str
    strength: Literal["hard", "soft"]
    polarity: CandidateModality
    source: Literal["user", "inferred", "system"] = "user"
    evidence: PromptEvidence
    quantity: RequirementQuantity
    scale: RequirementScale | None = None
    relations: tuple[RequirementRelation, ...] = ()
    topology: TopologyOpinion | None = None
    composition: CompositionPlan | None = None
    qualifiers: tuple[str, ...] = ()
    verification: VerificationPolicy
    interpretation_status: Literal["classified", "unclassified"]
    interpretation_rationale: str = ""
    enforcement: EnforcementDisposition
    enforcement_rationale: str


class RequirementGraphIssue(RequirementModel):
    code: Literal[
        "exact_quantity_conflict",
        "quantity_range_conflict",
        "polarity_conflict",
    ]
    severity: Literal["error"] = "error"
    canonical_subject: str
    requirement_ids: tuple[str, ...]
    message: str


class SceneRequirementGraph(RequirementModel):
    """Versioned semantic obligations extracted before construction."""

    schema_version: Literal[2] = CURRENT_REQUIREMENT_SCHEMA_VERSION
    graph_id: str
    source_prompt: str
    candidates: tuple[LiteralObligationCandidate, ...]
    requirements: tuple[SceneRequirement, ...]
    composition: SceneCompositionOpinion | None = None
    analysis_status: Literal["complete", "partial", "unavailable"]
    analysis_model: str | None = None
    merge_issues: tuple[str, ...] = ()
    validation_issues: tuple[RequirementGraphIssue, ...] = ()

    @model_validator(mode="after")
    def validate_provenance(self) -> "SceneRequirementGraph":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        requirement_ids = [item.requirement_id for item in self.requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirement IDs must be unique")

        candidates = {
            candidate.candidate_id: candidate for candidate in self.candidates
        }
        covered_candidates: set[str] = set()
        covered_quantities: set[str] = set()
        for candidate in self.candidates:
            evidence = candidate.evidence
            if self.source_prompt[evidence.start : evidence.end] != evidence.text:
                raise ValueError(
                    f"candidate {candidate.candidate_id} evidence does not match prompt"
                )
            for quantity in candidate.explicit_quantities:
                q_evidence = quantity.evidence
                if (
                    self.source_prompt[q_evidence.start : q_evidence.end]
                    != q_evidence.text
                ):
                    raise ValueError(
                        f"quantity {quantity.quantity_id} evidence does not match prompt"
                    )

        for requirement in self.requirements:
            candidate = candidates.get(requirement.source_candidate_id)
            if candidate is None:
                raise ValueError(
                    f"requirement {requirement.requirement_id} has unknown candidate"
                )
            if requirement.evidence != candidate.evidence:
                raise ValueError(
                    "requirement evidence must equal immutable candidate evidence"
                )
            expected_strength = "soft" if candidate.modality == "optional" else "hard"
            if requirement.strength != expected_strength:
                raise ValueError("model interpretation cannot weaken source modality")
            if requirement.polarity != candidate.modality:
                raise ValueError("model interpretation cannot change source polarity")
            expected_enforcement = _enforcement_disposition(
                candidate, requirement.kind
            )[0]
            if requirement.enforcement != expected_enforcement:
                raise ValueError(
                    "requirement enforcement does not match source-bound policy"
                )
            covered_candidates.add(candidate.candidate_id)
            if requirement.quantity.source_quantity_id:
                covered_quantities.add(requirement.quantity.source_quantity_id)

        if covered_candidates != set(candidate_ids):
            raise ValueError("every literal candidate must remain represented")
        expected_quantities = {
            quantity.quantity_id
            for candidate in self.candidates
            for quantity in candidate.explicit_quantities
        }
        if not expected_quantities <= covered_quantities:
            raise ValueError("every explicit source quantity must remain represented")
        return self

    @property
    def content_hash(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def is_valid(self) -> bool:
        return not self.validation_issues


class ShadowRequirementResult(RequirementModel):
    requirement_id: str
    status: Literal["satisfied", "missing", "ambiguous"]
    expected: str
    observed: str
    artifact_ids: tuple[str, ...] = ()


class SemanticShadowAudit(RequirementModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    mode: Literal["shadow"] = "shadow"
    results: tuple[ShadowRequirementResult, ...]

    @property
    def satisfied_count(self) -> int:
        return sum(result.status == "satisfied" for result in self.results)

    @property
    def missing_count(self) -> int:
        return sum(result.status == "missing" for result in self.results)

    @property
    def ambiguous_count(self) -> int:
        return sum(result.status == "ambiguous" for result in self.results)


class RequirementMergeError(ValueError):
    """A model interpretation attempted to alter or omit source truth."""


class RequirementGraphValidationError(ValueError):
    """The preserved user obligations contradict each other."""
