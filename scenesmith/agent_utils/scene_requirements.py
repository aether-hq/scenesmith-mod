"""Source-bound scene obligations with LLM semantic interpretation.

Deterministic code preserves literal assertive clauses, modality, and
unambiguous quantities without guessing what a noun means. A structured model
then interprets scene type, scale, composition, topology, fulfillment
strategies, and verification. The merger allows enrichment, but never deletion,
source-span changes, modality weakening, or explicit-quantity changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from agents import Agent, ModelSettings, RunConfig
from pydantic import BaseModel, ConfigDict, model_validator

from scenesmith.agent_utils.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.scene_blueprint import SceneBlueprint

if TYPE_CHECKING:
    from scenesmith.agent_utils.house import HouseLayout
    from scenesmith.agent_utils.room import RoomScene


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


class _Runner(Protocol):
    @staticmethod
    async def run(*args: Any, **kwargs: Any) -> Any: ...


_NUMBER_VALUES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_QUALITATIVE_QUANTIFIERS = {
    "all",
    "many",
    "multiple",
    "several",
    "hundreds",
    "thousands",
    "a bunch of",
}
_NUMBER_PATTERN = "|".join(sorted(_NUMBER_VALUES, key=len, reverse=True))
_QUALITATIVE_PATTERN = "|".join(
    re.escape(value)
    for value in sorted(_QUALITATIVE_QUANTIFIERS, key=len, reverse=True)
)
_QUANTITY_PATTERN = re.compile(
    rf"\b(?:(?P<minimum>at\s+least)\s+)?"
    rf"(?P<token>\d+|{_NUMBER_PATTERN}|{_QUALITATIVE_PATTERN}|an?|both|a\s+couple\s+of)\b",
    flags=re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(
    r",+|\b(?:and|but|while|with|featuring|containing|having|where|without)\b",
    flags=re.IGNORECASE,
)
_FORBIDDEN_MODALITY = re.compile(
    r"\b(?:no|not|without|avoid|avoiding|exclude|excluding|forbid|forbidden)\b",
    flags=re.IGNORECASE,
)
_OPTIONAL_MODALITY = re.compile(
    r"\b(?:could|might|may|optional|optionally|perhaps)\b",
    flags=re.IGNORECASE,
)
_DISCOURSE_ONLY = re.compile(
    r"^(?:(?:and|then|also)\s+)?(?:so\s+on|etc|etcetera)\.?$",
    flags=re.IGNORECASE,
)

_RELATION_STOP_WORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with"}
)


def _relation_word(token: str) -> str:
    """Return a small grammar-only stem for source relationship validation."""

    word = token.casefold()
    if word in {
        "floor",
        "floors",
        "level",
        "levels",
        "storey",
        "storeys",
        "story",
        "stories",
    }:
        return "vertical-level"
    if word.endswith("ies") and len(word) > 4:
        return f"{word[:-3]}y"
    if word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        if word.endswith(word[-1:] * 2):
            word = word[:-1]
        return word
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("es") and len(word) > 4:
        if word.endswith(("ches", "shes", "sses", "xes", "zes")):
            return word[:-2]
        return word[:-1]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _relation_words(text: str) -> frozenset[str]:
    return frozenset(
        _relation_word(token)
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if token.casefold() not in _RELATION_STOP_WORDS
    )


def _source_supports_relation(
    relation: RequirementRelationWire, source_text: str
) -> bool:
    """Reject model-invented relationships that have no literal source support."""

    source_words = _relation_words(source_text)
    predicate_words = _relation_words(relation.predicate)
    target_words = _relation_words(relation.target)
    return bool(predicate_words and predicate_words & source_words) and bool(
        target_words and target_words <= source_words
    )


def _stable_id(prefix: str, prompt: str, start: int, end: int, suffix: str = "") -> str:
    digest = hashlib.sha1(
        f"{prompt}:{start}:{end}:{suffix}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _trim_span(prompt: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and (prompt[start].isspace() or prompt[start] in ",;:"):
        start += 1
    while end > start and (prompt[end - 1].isspace() or prompt[end - 1] in ",;:"):
        end -= 1
    if start >= end:
        return None
    text = prompt[start:end]
    if not re.search(r"[A-Za-z0-9]", text) or _DISCOURSE_ONLY.fullmatch(text.strip()):
        return None
    return start, end


def _candidate_spans(prompt: str) -> tuple[tuple[int, int], ...]:
    """Split assertive prose using grammar only, never domain vocabulary."""

    spans: list[tuple[int, int]] = []
    sentence_start = 0
    sentence_regions: list[tuple[int, int]] = []
    for index, character in enumerate(prompt):
        if character not in ".;!?":
            continue
        if character == ".":
            before = prompt[max(0, index - 12) : index]
            next_character = prompt[index + 1] if index + 1 < len(prompt) else ""
            # Initialisms/abbreviations and decimal-like tokens are not sentence
            # boundaries. This is lexical punctuation handling, not semantics.
            if (
                index > 0 and prompt[index - 1].isalnum() and next_character.isalnum()
            ) or re.search(r"(?:\b[A-Za-z]\.)+[A-Za-z]$", before):
                continue
        sentence_regions.append((sentence_start, index))
        sentence_start = index + 1
    sentence_regions.append((sentence_start, len(prompt)))

    for region_start, region_end in sentence_regions:
        cursor = region_start
        for boundary in _CLAUSE_BOUNDARY.finditer(prompt, region_start, region_end):
            trimmed = _trim_span(prompt, cursor, boundary.start())
            if trimmed:
                spans.append(trimmed)
            # Retain grammar markers so the model sees accompaniment/negation.
            cursor = (
                boundary.start() if boundary.group(0).strip() != "," else boundary.end()
            )
        trimmed = _trim_span(prompt, cursor, region_end)
        if trimmed:
            spans.append(trimmed)
    return tuple(dict.fromkeys(spans))


def _quantity_from_match(prompt: str, match: re.Match[str]) -> ExplicitQuantity:
    token = re.sub(r"\s+", " ", match.group("token").casefold())
    if token == "both":
        mode: Literal["exact", "minimum", "qualitative"] = "exact"
        value, label = 2, ""
    elif token == "a couple of":
        mode, value, label = "exact", 2, ""
    elif token in _QUALITATIVE_QUANTIFIERS:
        mode, value, label = "qualitative", None, token
    else:
        mode = "minimum" if match.group("minimum") else "exact"
        value = int(token) if token.isdigit() else _NUMBER_VALUES.get(token, 1)
        label = ""
    start, end = match.span()
    return ExplicitQuantity(
        quantity_id=_stable_id("qty", prompt, start, end),
        evidence=PromptEvidence(text=prompt[start:end], start=start, end=end),
        mode=mode,
        value=value,
        label=label,
    )


def literal_candidates_from_prompt(
    prompt: str,
) -> tuple[LiteralObligationCandidate, ...]:
    """Preserve literal clauses and quantities without semantic classification."""

    candidates: list[LiteralObligationCandidate] = []
    for start, end in _candidate_spans(prompt):
        text = prompt[start:end]
        if _FORBIDDEN_MODALITY.search(text):
            modality: CandidateModality = "forbidden"
        elif _OPTIONAL_MODALITY.search(text):
            modality = "optional"
        else:
            modality = "required"
        quantities = tuple(
            _quantity_from_match(prompt, match)
            for match in _QUANTITY_PATTERN.finditer(prompt, start, end)
        )
        candidates.append(
            LiteralObligationCandidate(
                candidate_id=_stable_id("candidate", prompt, start, end),
                evidence=PromptEvidence(text=text, start=start, end=end),
                modality=modality,
                explicit_quantities=quantities,
            )
        )
    return tuple(candidates)


REQUIREMENT_ANALYST_INSTRUCTIONS = """\
You are the semantic obligation analyst for a general 3D world builder. Analyze
arbitrary domains; never assume the scene is a house or that unfamiliar nouns are
decorations. The input contains the full prompt and immutable source candidates.

Return an exhaustive but compact structured interpretation. Output only the
requested schema. Every free-text field must be a short phrase or one sentence of
at most 18 words. Use at most two relations, three adjacency entries, three reusable
parts, three qualifiers, and two measurable criteria per requirement. Do not restate
the prompt or add commentary outside schema fields.

Interpret the candidates as follows:
1. Emit one or more requirements for EVERY candidate_id. Split candidates that
   contain multiple obligations. Never omit, soften, or replace an explicit request.
2. Reference every explicit quantity by its exact quantity_id. Copy its mode/value/
   label exactly. You may add an interpreted_minimum for qualitative quantities.
3. Judge semantics rather than matching keywords: scene type, object/zone/opening/
   connector role, topology, relationships, and whether an item is a hero element.
   Doors, windows, portals, and apertures are openings. Reserve connector for
   stairs, ramps, ladders, and elevators that join different vertical levels.
   Emit a relationship only when both its predicate and target are explicitly stated
   in the source prompt. Do not add plausible domain relationships or inferred uses.
4. Give concise physical-size envelopes and clearances only when they affect
   fulfillment.
5. Give concise composition guidance: focal hierarchy, density, circulation,
   spatial organization, and how parts form the requested whole.
6. Recommend one fulfillment strategy: suitable catalog asset, composition from
   reusable parts, or procedural geometry. Name the concrete fallback construction
   briefly; deterministic code ranks the two remaining strategies.
7. Define measurable verification criteria. If the concept is unfamiliar, preserve
   it and describe what evidence would prove it; never silently reinterpret it as a
   generic room or generic prop.
"""


def requirement_analysis_input(
    prompt: str, candidates: tuple[LiteralObligationCandidate, ...]
) -> str:
    payload = {
        "source_prompt": prompt,
        "immutable_candidates": [candidate.model_dump() for candidate in candidates],
    }
    return json.dumps(payload, indent=2)


async def analyze_requirement_candidates(
    prompt: str,
    candidates: tuple[LiteralObligationCandidate, ...],
    *,
    model: str,
    run_config: RunConfig | None = None,
    model_settings: ModelSettings | None = None,
    runner: type[_Runner] = BoundedRunner,
) -> tuple[RequirementInterpretationBatch, Any]:
    """Ask the configured LLM for structured semantic and spatial judgment."""

    analyst = Agent(
        name="Scene Semantic Obligation Analyst",
        model=model,
        instructions=REQUIREMENT_ANALYST_INSTRUCTIONS,
        output_type=RequirementInterpretationWireBatch,
        model_settings=model_settings or ModelSettings(),
    )
    result = await runner.run(
        starting_agent=analyst,
        input=requirement_analysis_input(prompt, candidates),
        max_turns=1,
        run_config=run_config,
        timeout_seconds=agent_run_timeout_seconds("requirements", max_turns=1),
    )
    wire_batch = result.final_output_as(RequirementInterpretationWireBatch)
    return expand_requirement_interpretations(candidates, wire_batch), result


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


def _quantity_equal(proposal: RequirementQuantity, explicit: ExplicitQuantity) -> bool:
    return (
        proposal.mode == explicit.mode
        and proposal.value == explicit.value
        and proposal.label.casefold() == explicit.label.casefold()
    )


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


def _canonical_subject(subject: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", subject.casefold()))


def _detect_requirement_conflicts(
    requirements: list[SceneRequirement],
) -> tuple[RequirementGraphIssue, ...]:
    by_subject: dict[str, list[SceneRequirement]] = {}
    for requirement in requirements:
        if requirement.enforcement == "advisory" or requirement.kind == "unclassified":
            continue
        canonical = _canonical_subject(requirement.subject)
        if canonical:
            by_subject.setdefault(canonical, []).append(requirement)

    issues: list[RequirementGraphIssue] = []
    for canonical, grouped in sorted(by_subject.items()):
        required = [item for item in grouped if item.polarity == "required"]
        forbidden = [item for item in grouped if item.polarity == "forbidden"]
        if required and forbidden:
            issues.append(
                RequirementGraphIssue(
                    code="polarity_conflict",
                    canonical_subject=canonical,
                    requirement_ids=tuple(
                        item.requirement_id for item in required + forbidden
                    ),
                    message=(
                        f"{canonical!r} is both required and forbidden by explicit "
                        "source clauses"
                    ),
                )
            )

        exact = {
            int(item.quantity.value): item
            for item in required
            if item.quantity.mode == "exact" and item.quantity.value is not None
        }
        if len(exact) > 1:
            issues.append(
                RequirementGraphIssue(
                    code="exact_quantity_conflict",
                    canonical_subject=canonical,
                    requirement_ids=tuple(
                        item.requirement_id
                        for item in required
                        if item.quantity.mode == "exact"
                    ),
                    message=(
                        f"{canonical!r} has incompatible exact counts "
                        f"{sorted(exact)}"
                    ),
                )
            )

        minimums = [
            int(item.quantity.value)
            for item in required
            if item.quantity.mode == "minimum" and item.quantity.value is not None
        ]
        if len(exact) == 1 and minimums and max(minimums) > next(iter(exact)):
            issues.append(
                RequirementGraphIssue(
                    code="quantity_range_conflict",
                    canonical_subject=canonical,
                    requirement_ids=tuple(
                        item.requirement_id
                        for item in required
                        if item.quantity.mode in {"exact", "minimum"}
                    ),
                    message=(
                        f"{canonical!r} requires exactly {next(iter(exact))} but "
                        f"also at least {max(minimums)}"
                    ),
                )
            )
    return tuple(issues)


def assert_requirement_graph_consistent(graph: SceneRequirementGraph) -> None:
    """Raise a specific error before enforcing a contradictory graph."""

    if graph.validation_issues:
        messages = "; ".join(issue.message for issue in graph.validation_issues)
        raise RequirementGraphValidationError(messages)


def _unclassified_requirements(
    prompt: str,
    candidate: LiteralObligationCandidate,
    *,
    start_ordinal: int,
    reason: str,
) -> list[SceneRequirement]:
    quantities = candidate.explicit_quantities or (None,)
    output: list[SceneRequirement] = []
    enforcement, enforcement_rationale = _enforcement_disposition(
        candidate, "unclassified"
    )
    for offset, explicit in enumerate(quantities):
        quantity = (
            explicit.as_requirement_quantity()
            if explicit is not None
            else RequirementQuantity(mode="qualitative", label="explicit obligation")
        )
        output.append(
            SceneRequirement(
                requirement_id=_stable_id(
                    "req",
                    prompt,
                    candidate.evidence.start,
                    candidate.evidence.end,
                    f"unclassified:{start_ordinal + offset}",
                ),
                source_candidate_id=candidate.candidate_id,
                kind="unclassified",
                subject=candidate.evidence.text,
                strength="soft" if candidate.modality == "optional" else "hard",
                polarity=candidate.modality,
                evidence=candidate.evidence,
                quantity=quantity,
                verification=VerificationPolicy(
                    stage="semantic",
                    method="unclassified_source_obligation",
                    measurable_criteria=(
                        "A semantic interpretation must be produced before enforcement",
                    ),
                ),
                interpretation_status="unclassified",
                interpretation_rationale=reason,
                enforcement=enforcement,
                enforcement_rationale=enforcement_rationale,
            )
        )
    return output


def merge_requirement_interpretations(
    prompt: str,
    candidates: tuple[LiteralObligationCandidate, ...],
    batch: RequirementInterpretationBatch | None,
    *,
    analysis_model: str | None = None,
    analysis_error: str | None = None,
    allow_partial: bool = True,
) -> SceneRequirementGraph:
    """Merge model opinions while preserving every immutable source obligation."""

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    quantity_by_id = {
        quantity.quantity_id: (candidate, quantity)
        for candidate in candidates
        for quantity in candidate.explicit_quantities
    }
    proposals_by_candidate: dict[str, list[RequirementInterpretationProposal]] = {
        candidate.candidate_id: [] for candidate in candidates
    }
    if batch is not None:
        for proposal in batch.requirements:
            if proposal.candidate_id not in candidate_by_id:
                raise RequirementMergeError(
                    f"model returned unknown candidate_id {proposal.candidate_id}"
                )
            if proposal.source_quantity_id:
                owner_and_quantity = quantity_by_id.get(proposal.source_quantity_id)
                if owner_and_quantity is None:
                    raise RequirementMergeError(
                        f"model returned unknown quantity_id {proposal.source_quantity_id}"
                    )
                owner, explicit = owner_and_quantity
                if owner.candidate_id != proposal.candidate_id:
                    raise RequirementMergeError(
                        "model attached an explicit quantity to a different source clause"
                    )
                if not _quantity_equal(proposal.quantity, explicit):
                    raise RequirementMergeError(
                        f"model altered explicit quantity {explicit.quantity_id}"
                    )
            proposals_by_candidate[proposal.candidate_id].append(proposal)

    requirements: list[SceneRequirement] = []
    issues: list[str] = []
    claimed_quantities: set[str] = set()
    for candidate in candidates:
        proposals = proposals_by_candidate[candidate.candidate_id]
        if not proposals:
            issues.append(f"uninterpreted candidate {candidate.candidate_id}")
            requirements.extend(
                _unclassified_requirements(
                    prompt,
                    candidate,
                    start_ordinal=len(requirements),
                    reason=analysis_error or "model omitted this literal candidate",
                )
            )
            continue

        for proposal in proposals:
            enforcement, enforcement_rationale = _enforcement_disposition(
                candidate, proposal.kind
            )
            if proposal.source_quantity_id:
                claimed_quantities.add(proposal.source_quantity_id)
                explicit = quantity_by_id[proposal.source_quantity_id][1]
                quantity = explicit.as_requirement_quantity().model_copy(
                    update={
                        "interpreted_minimum": proposal.quantity.interpreted_minimum
                    }
                )
            else:
                quantity = proposal.quantity.model_copy(
                    update={"source_quantity_id": None}
                )
            ordinal = len(requirements)
            requirements.append(
                SceneRequirement(
                    requirement_id=_stable_id(
                        "req",
                        prompt,
                        candidate.evidence.start,
                        candidate.evidence.end,
                        f"{proposal.kind}:{proposal.subject}:{ordinal}",
                    ),
                    source_candidate_id=candidate.candidate_id,
                    kind=proposal.kind,
                    subject=proposal.subject,
                    strength="soft" if candidate.modality == "optional" else "hard",
                    polarity=candidate.modality,
                    evidence=candidate.evidence,
                    quantity=quantity,
                    scale=proposal.scale,
                    relations=proposal.relations,
                    topology=proposal.topology,
                    composition=proposal.composition,
                    qualifiers=proposal.qualifiers,
                    verification=proposal.verification,
                    interpretation_status="classified",
                    interpretation_rationale=proposal.interpretation_rationale,
                    enforcement=enforcement,
                    enforcement_rationale=enforcement_rationale,
                )
            )

        missing_quantities = [
            quantity
            for quantity in candidate.explicit_quantities
            if quantity.quantity_id not in claimed_quantities
        ]
        if missing_quantities:
            issues.extend(
                f"uninterpreted quantity {quantity.quantity_id}"
                for quantity in missing_quantities
            )
            for quantity in missing_quantities:
                placeholder_candidate = candidate.model_copy(
                    update={"explicit_quantities": (quantity,)}
                )
                requirements.extend(
                    _unclassified_requirements(
                        prompt,
                        placeholder_candidate,
                        start_ordinal=len(requirements),
                        reason="model omitted this explicit source quantity",
                    )
                )

    if issues and not allow_partial:
        raise RequirementMergeError("; ".join(issues))
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    if batch is None:
        status: Literal["complete", "partial", "unavailable"] = "unavailable"
    elif issues:
        status = "partial"
    else:
        status = "complete"
    return SceneRequirementGraph(
        graph_id=f"requirements-{prompt_digest}",
        source_prompt=prompt,
        candidates=candidates,
        requirements=tuple(requirements),
        composition=batch.composition if batch is not None else None,
        analysis_status=status,
        analysis_model=analysis_model,
        merge_issues=tuple(issues),
        validation_issues=_detect_requirement_conflicts(requirements),
    )


def requirement_graph_from_prompt(
    prompt: str,
    interpretations: RequirementInterpretationBatch | None = None,
    *,
    analysis_model: str | None = None,
    analysis_error: str | None = None,
    allow_partial: bool = True,
) -> SceneRequirementGraph:
    """Build a stable graph from literal candidates and optional LLM opinions."""

    candidates = literal_candidates_from_prompt(prompt)
    return merge_requirement_interpretations(
        prompt,
        candidates,
        interpretations,
        analysis_model=analysis_model,
        analysis_error=analysis_error,
        allow_partial=allow_partial,
    )


def _persist_model(model: RequirementModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump_json(indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def persist_requirement_graph(graph: SceneRequirementGraph, output_path: Path) -> None:
    _persist_model(graph, output_path)


def load_requirement_graph(path: Path) -> SceneRequirementGraph:
    return SceneRequirementGraph.model_validate_json(path.read_text(encoding="utf-8"))


def persist_shadow_audit(audit: SemanticShadowAudit, output_path: Path) -> None:
    _persist_model(audit, output_path)


def _quantity_text(quantity: RequirementQuantity) -> str:
    if quantity.mode == "qualitative":
        return quantity.label
    operator = "exactly" if quantity.mode == "exact" else "at least"
    return f"{operator} {quantity.value}"


def _object_matches(scene: "RoomScene", subject: str) -> tuple[str, ...]:
    tokens = [
        re.escape(token)
        for token in re.findall(r"[a-z0-9]+", subject.casefold())
        if len(token) > 1
    ]
    if not tokens:
        return ()
    pattern = re.compile(".*".join(tokens), flags=re.IGNORECASE)
    return tuple(
        str(object_id)
        for object_id, scene_object in scene.objects.items()
        if pattern.search(f"{object_id} {scene_object.name}")
    )


def _count_satisfies(requirement: SceneRequirement, observed: int) -> bool:
    if requirement.polarity == "forbidden":
        return observed == 0
    quantity = requirement.quantity
    if quantity.mode == "exact":
        return observed == quantity.value
    if quantity.mode == "minimum":
        return observed >= int(quantity.value or 0)
    minimum = quantity.interpreted_minimum
    return observed >= minimum if minimum is not None else observed > 0


def _audit_requirement(
    requirement: SceneRequirement,
    *,
    blueprint: SceneBlueprint,
    scene: "RoomScene",
    house_layout: "HouseLayout | None",
) -> ShadowRequirementResult:
    expected = f"{_quantity_text(requirement.quantity)} {requirement.subject}"
    if requirement.interpretation_status == "unclassified":
        return ShadowRequirementResult(
            requirement_id=requirement.requirement_id,
            status="ambiguous",
            expected=expected,
            observed="no valid semantic interpretation",
        )
    if requirement.kind == "scene_type":
        space_types = tuple(
            f"{space.room_type} {space.name}".casefold() for space in blueprint.spaces
        )
        matched = any(requirement.subject.casefold() in item for item in space_types)
        return ShadowRequirementResult(
            requirement_id=requirement.requirement_id,
            status="satisfied" if matched else "missing",
            expected=expected,
            observed=", ".join(space_types) or "no spaces",
        )
    if requirement.kind == "level":
        artifact_ids = tuple(level.level_id for level in blueprint.levels)
        return ShadowRequirementResult(
            requirement_id=requirement.requirement_id,
            status=(
                "satisfied"
                if _count_satisfies(requirement, len(artifact_ids))
                else "missing"
            ),
            expected=expected,
            observed=f"{len(artifact_ids)} levels",
            artifact_ids=artifact_ids,
        )
    if requirement.kind == "connector":
        artifact_ids = tuple(
            connector.connector_id for connector in blueprint.connectors
        )
        return ShadowRequirementResult(
            requirement_id=requirement.requirement_id,
            status=(
                "satisfied"
                if _count_satisfies(requirement, len(artifact_ids))
                else "missing"
            ),
            expected=expected,
            observed=f"{len(artifact_ids)} connectors (subtype not yet certified)",
            artifact_ids=artifact_ids,
        )
    if requirement.kind == "opening":
        artifact_ids = [opening.opening_id for opening in blueprint.openings]
        if house_layout is not None:
            artifact_ids.extend(str(door.id) for door in house_layout.doors)
            artifact_ids.extend(
                str(portal.portal_id) for portal in house_layout.portals
            )
        count_ok = _count_satisfies(requirement, len(artifact_ids))
        return ShadowRequirementResult(
            requirement_id=requirement.requirement_id,
            status="ambiguous" if count_ok else "missing",
            expected=expected,
            observed=f"{len(artifact_ids)} openings; semantic subtype unverified",
            artifact_ids=tuple(artifact_ids),
        )
    if requirement.kind == "style":
        style_tokens = " ".join(blueprint.design_tokens.style_keywords).casefold()
        matched = requirement.subject.casefold() in style_tokens
        return ShadowRequirementResult(
            requirement_id=requirement.requirement_id,
            status="satisfied" if matched else "ambiguous",
            expected=expected,
            observed=style_tokens or "no structured style tokens",
        )

    artifact_ids = _object_matches(scene, requirement.subject)
    count_ok = _count_satisfies(requirement, len(artifact_ids))
    geometry_checks_pending = bool(
        requirement.relations or requirement.scale or requirement.topology
    )
    if not count_ok:
        status: Literal["satisfied", "missing", "ambiguous"] = "missing"
    elif geometry_checks_pending:
        status = "ambiguous"
    else:
        status = "satisfied"
    return ShadowRequirementResult(
        requirement_id=requirement.requirement_id,
        status=status,
        expected=expected,
        observed=f"{len(artifact_ids)} matching objects",
        artifact_ids=artifact_ids,
    )


def audit_requirement_graph(
    graph: SceneRequirementGraph,
    *,
    blueprint: SceneBlueprint,
    scene: "RoomScene",
    house_layout: "HouseLayout | None" = None,
) -> SemanticShadowAudit:
    """Measure prompt satisfaction without changing the build verdict."""

    return SemanticShadowAudit(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        results=tuple(
            _audit_requirement(
                requirement,
                blueprint=blueprint,
                scene=scene,
                house_layout=house_layout,
            )
            for requirement in graph.requirements
        ),
    )
