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

from scenesmith.agent_utils.runtime.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.interpretation import (
    expand_requirement_interpretations,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.literal_parsing import (
    _stable_id,
    literal_candidates_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    ExplicitQuantity,
    LiteralObligationCandidate,
    RequirementGraphIssue,
    RequirementGraphValidationError,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementInterpretationWireBatch,
    RequirementMergeError,
    RequirementModel,
    RequirementQuantity,
    SceneRequirement,
    SceneRequirementGraph,
    SemanticShadowAudit,
    ShadowRequirementResult,
    VerificationPolicy,
    _enforcement_disposition,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.house import HouseLayout
    from scenesmith.agent_utils.scene.room import RoomScene


class _Runner(Protocol):
    @staticmethod
    async def run(*args: Any, **kwargs: Any) -> Any: ...


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


def _quantity_equal(proposal: RequirementQuantity, explicit: ExplicitQuantity) -> bool:
    return (
        proposal.mode == explicit.mode
        and proposal.value == explicit.value
        and proposal.label.casefold() == explicit.label.casefold()
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
