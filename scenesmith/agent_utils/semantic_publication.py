"""LLM semantic binding plus deterministic fail-closed publication certification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile

from pathlib import Path
from typing import Any, Literal, Protocol

from agents import Agent, ModelSettings, RunConfig
from pydantic import BaseModel, ConfigDict, Field

from scenesmith.agent_utils.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.requirement_blueprint_compiler import (
    SpatialRequirementCompilation,
)
from scenesmith.agent_utils.scene_blueprint import SceneBlueprint
from scenesmith.agent_utils.scene_requirements import (
    SceneRequirement,
    SceneRequirementGraph,
)


class PublicationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticArtifact(PublicationModel):
    artifact_id: str
    artifact_class: Literal[
        "scene",
        "level",
        "space",
        "opening",
        "connector",
        "constraint",
        "scene_object",
    ]
    name: str
    description: str = ""
    dimensions_m: tuple[float, float, float] | None = None
    position_m: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationVerification(PublicationModel):
    predicate: str
    target: str
    satisfied: bool
    evidence_artifact_ids: tuple[str, ...]
    measurement: str


class RequirementVerificationClaim(PublicationModel):
    requirement_id: str
    status: Literal["satisfied", "missing", "ambiguous"]
    artifact_ids: tuple[str, ...]
    observed_count: int
    relation_results: tuple[RelationVerification, ...] = ()
    semantic_rationale: str


class SemanticVerificationBatch(PublicationModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    claims: tuple[RequirementVerificationClaim, ...]
    audit_summary: str


class CertifiedRequirement(PublicationModel):
    requirement_id: str
    subject: str
    artifact_ids: tuple[str, ...]
    observed_count: int
    evidence_hash: str


class SemanticPublicationCertificate(PublicationModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    spatial_compilation_hash: str
    physics_verified: bool
    physics_evidence_refs: tuple[str, ...]
    requirements: tuple[CertifiedRequirement, ...]
    publishable: Literal[True] = True


class SemanticPublicationError(RuntimeError):
    """One or more immutable hard obligations lack valid final evidence."""

    def __init__(
        self,
        message: str,
        *,
        failures: tuple[str, ...] = (),
        certified_requirements: tuple[CertifiedRequirement, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failures = failures
        self.certified_requirements = certified_requirements


class _Runner(Protocol):
    @staticmethod
    async def run(*args: Any, **kwargs: Any) -> Any: ...


SEMANTIC_VERIFIER_INSTRUCTIONS = """
You are SceneSmith's final semantic evidence binder. The immutable requirement graph
is source truth. The artifact inventory contains only artifacts that really survived
construction; planned furniture groups are intentionally absent.

For every hard requirement, return exactly one claim using its unchanged ID.
Semantically bind only artifacts that genuinely implement the requested concept.
Never count a generic room, generic prop, wall, or decorative proxy as a specialized
object or operational zone merely because it is physically present. Never claim a
count larger than the number of distinct bound artifacts. Assess identity,
composition, topology, scale, focal role, and relationships using the whole prompt
and the LLM-authored requirement interpretation.

Required obligations are satisfied only when concrete surviving artifacts meet all
semantic and measurable criteria. Forbidden obligations are satisfied only when no
matching artifact exists. Return missing or ambiguous rather than guessing. For each
authored relationship, return a relation result with concrete evidence IDs and a
clear measurement or spatial observation.

Keep evidence bounded. For exact requirements, bind exactly the expected number of
artifacts. For minimum or qualitative requirements, bind only the smallest distinct
set that proves the requested minimum; do not enumerate surplus matching artifacts.
"""


_EXPECTED_FINAL_ARTIFACT_CLASSES = {
    "scene_type": frozenset({"scene", "space"}),
    "level": frozenset({"level"}),
    "repeated_zone": frozenset({"space", "scene_object"}),
    "hero_object": frozenset({"scene_object"}),
    "opening": frozenset({"opening"}),
    "connector": frozenset({"connector"}),
    "object_group": frozenset({"scene_object"}),
    "spatial_constraint": frozenset({"scene", "space", "scene_object"}),
    "style": frozenset({"scene", "space", "scene_object"}),
}

_SEMANTIC_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "decor",
        "for",
        "group",
        "of",
        "scale",
        "style",
        "the",
    }
)


def _semantic_word(token: str) -> str:
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
    if word.endswith("es") and len(word) > 4:
        if word.endswith(("ches", "shes", "sses", "xes", "zes")):
            return word[:-2]
        return word[:-1]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _semantic_words(text: str) -> frozenset[str]:
    return frozenset(
        _semantic_word(token)
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if token.casefold() not in _SEMANTIC_STOP_WORDS
    )


def _artifact_semantic_words(artifact: SemanticArtifact) -> frozenset[str]:
    metadata_text = " ".join(
        str(value)
        for key, value in artifact.metadata.items()
        if key
        in {
            "catalog_semantics",
            "dense_library_populated_case",
            "generated_from",
            "ontology_path",
            "opening_type",
            "role",
        }
    )
    return _semantic_words(
        f"{artifact.artifact_id} {artifact.name} {artifact.description} {metadata_text}"
    )


def _deterministic_claim(
    requirement: SceneRequirement,
    compilation: SpatialRequirementCompilation,
    artifacts: tuple[SemanticArtifact, ...],
) -> RequirementVerificationClaim | None:
    """Bind conservative final evidence without spending a model turn.

    Topology requirements reach this gate only after constructed-topology
    validation. Object requirements are bound only through surviving semantic
    names/descriptions/metadata; ambiguous qualified or relational objects remain
    model-verifier work.
    """

    expected_classes = _EXPECTED_FINAL_ARTIFACT_CLASSES.get(
        requirement.kind, frozenset()
    )
    candidates = [
        artifact
        for artifact in artifacts
        if not expected_classes or artifact.artifact_class in expected_classes
    ]
    binding = next(
        (
            item
            for item in compilation.bindings
            if item.requirement_id == requirement.requirement_id
        ),
        None,
    )
    topology_owned = binding is not None and binding.owner_stage == "topology"
    if requirement.relations and not topology_owned:
        return None
    if requirement.qualifiers:
        return None

    subject_words = _semantic_words(requirement.subject)
    binding_words = _semantic_words(binding.role_key or "") if binding else frozenset()
    bound_ids = frozenset(binding.artifact_ids) if binding else frozenset()

    def match_score(artifact: SemanticArtifact) -> int:
        if artifact.artifact_id in bound_ids:
            return 100
        direct_words = _semantic_words(f"{artifact.artifact_id} {artifact.name}")
        artifact_words = _artifact_semantic_words(artifact)
        if requirement.kind == "level":
            return 100 if artifact.artifact_class == "level" else 0
        if requirement.kind in {"scene_type", "opening", "connector"}:
            if subject_words and subject_words <= direct_words:
                return 80
            return 40 if subject_words & artifact_words else 0
        if binding_words and binding_words <= artifact_words:
            return 60
        if subject_words and subject_words <= direct_words:
            return 80
        return 20 if subject_words and subject_words <= artifact_words else 0

    matched = sorted(
        (artifact for artifact in candidates if match_score(artifact) > 0),
        key=match_score,
        reverse=True,
    )
    expected = _expected_count(requirement)
    if requirement.polarity == "forbidden":
        selected = matched
        status: Literal["satisfied", "missing", "ambiguous"] = (
            "satisfied" if not selected else "missing"
        )
    elif len(matched) < expected:
        return None
    else:
        selected = matched[:expected]
        status = "satisfied"

    artifact_ids = tuple(artifact.artifact_id for artifact in selected)
    relation_results = tuple(
        RelationVerification(
            predicate=relation.predicate,
            target=relation.target,
            satisfied=True,
            evidence_artifact_ids=artifact_ids,
            measurement=(
                "Constructed-topology validation preserved this source relationship."
            ),
        )
        for relation in requirement.relations
    )
    return RequirementVerificationClaim(
        requirement_id=requirement.requirement_id,
        status=status,
        artifact_ids=artifact_ids,
        observed_count=len(set(artifact_ids)),
        relation_results=relation_results,
        semantic_rationale=(
            "Surviving artifact semantics and deterministic topology evidence "
            "match the source-bound requirement."
        ),
    )


def _artifact_dimensions(obj: Any) -> tuple[float, float, float] | None:
    if obj.bbox_min is None or obj.bbox_max is None:
        return None
    dimensions = obj.bbox_max - obj.bbox_min
    return tuple(float(value) for value in dimensions)


def semantic_artifact_inventory(
    blueprint: SceneBlueprint,
    scene: Any,
    house_layout: Any | None = None,
) -> tuple[SemanticArtifact, ...]:
    """Build a bounded, provider-neutral final artifact inventory."""

    if house_layout is not None and house_layout.room_specs:
        minimum_x = min(float(room.position[0]) for room in house_layout.room_specs)
        minimum_y = min(float(room.position[1]) for room in house_layout.room_specs)
        maximum_x = max(
            float(room.position[0]) + float(room.length)
            for room in house_layout.room_specs
        )
        maximum_y = max(
            float(room.position[1]) + float(room.width)
            for room in house_layout.room_specs
        )
        minimum_z = min(float(level.elevation) for level in house_layout.levels)
        maximum_z = max(
            float(level.elevation) + float(level.nominal_height)
            for level in house_layout.levels
        )
        scene_dimensions = (
            maximum_x - minimum_x,
            maximum_y - minimum_y,
            maximum_z - minimum_z,
        )
    else:
        scene_dimensions = (
            max((space.dimensions_m[0] for space in blueprint.spaces), default=0.0),
            max((space.dimensions_m[1] for space in blueprint.spaces), default=0.0),
            max(
                (
                    level.elevation_m + level.clear_height_m
                    for level in blueprint.levels
                ),
                default=0.0,
            ),
        )
    artifacts: list[SemanticArtifact] = [
        SemanticArtifact(
            artifact_id=blueprint.blueprint_id,
            artifact_class="scene",
            name="compiled scene",
            description=blueprint.source_prompt,
            dimensions_m=scene_dimensions,
        )
    ]
    if house_layout is None:
        artifacts.extend(
            SemanticArtifact(
                artifact_id=level.level_id,
                artifact_class="level",
                name=level.name,
                dimensions_m=(0.0, 0.0, level.clear_height_m),
                position_m=(0.0, 0.0, level.elevation_m),
            )
            for level in blueprint.levels
        )
        artifacts.extend(
            SemanticArtifact(
                artifact_id=space.space_id,
                artifact_class="space",
                name=space.name,
                description=space.room_type,
                dimensions_m=(space.dimensions_m[0], space.dimensions_m[1], 0.0),
            )
            for space in blueprint.spaces
        )
    else:
        level_elevations = {
            level.level_id: float(level.elevation) for level in house_layout.levels
        }
        artifacts.extend(
            SemanticArtifact(
                artifact_id=level.level_id,
                artifact_class="level",
                name=level.level_id,
                dimensions_m=(0.0, 0.0, level.nominal_height),
                position_m=(0.0, 0.0, level.elevation),
            )
            for level in house_layout.levels
        )
        artifacts.extend(
            SemanticArtifact(
                artifact_id=room.room_id,
                artifact_class="space",
                name=room.room_type,
                description=room.prompt,
                dimensions_m=(room.length, room.width, house_layout.wall_height),
                position_m=(
                    room.position[0],
                    room.position[1],
                    float(
                        room.elevation
                        if room.elevation is not None
                        else level_elevations[room.level_id]
                    ),
                ),
            )
            for room in house_layout.room_specs
        )
        artifacts.extend(
            SemanticArtifact(
                artifact_id=connector.connector_id,
                artifact_class="connector",
                name=connector.connector_type.value,
                dimensions_m=(
                    connector.width,
                    0.0,
                    connector.clearance_height,
                ),
                metadata={
                    "start_space_id": connector.start.space_id,
                    "end_space_id": connector.end.space_id,
                },
            )
            for connector in house_layout.connectors
        )
    artifacts.extend(
        SemanticArtifact(
            artifact_id=opening.opening_id,
            artifact_class="opening",
            name=opening.opening_type,
            dimensions_m=(
                opening.width,
                scene.room_geometry.wall_thickness,
                opening.height,
            ),
            position_m=tuple(float(value) for value in opening.center_world),
            metadata={
                "opening_type": opening.opening_type,
                "wall_direction": opening.wall_direction,
                "sill_height": opening.sill_height,
            },
        )
        for opening in scene.room_geometry.openings
    )
    for object_id, obj in scene.objects.items():
        position = tuple(float(value) for value in obj.transform.translation())
        artifacts.append(
            SemanticArtifact(
                artifact_id=str(object_id),
                artifact_class="scene_object",
                name=obj.name,
                description=obj.description,
                dimensions_m=_artifact_dimensions(obj),
                position_m=position,
                metadata={
                    key: value
                    for key, value in obj.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                },
            )
        )
    return tuple(artifacts)


def semantic_verification_input(
    graph: SceneRequirementGraph,
    compilation: SpatialRequirementCompilation,
    artifacts: tuple[SemanticArtifact, ...],
    *,
    requirement_ids: frozenset[str] | None = None,
) -> str:
    hard_requirements = []
    for requirement in graph.requirements:
        if requirement.strength != "hard" or (
            requirement_ids is not None
            and requirement.requirement_id not in requirement_ids
        ):
            continue
        hard_requirements.append(
            {
                "requirement_id": requirement.requirement_id,
                "kind": requirement.kind,
                "subject": requirement.subject,
                "polarity": requirement.polarity,
                "enforcement": requirement.enforcement,
                "source_evidence": requirement.evidence.text,
                "quantity": requirement.quantity.model_dump(
                    mode="json", exclude_none=True
                ),
                "scale": (
                    requirement.scale.model_dump(mode="json", exclude_none=True)
                    if requirement.scale is not None
                    else None
                ),
                "relations": [
                    relation.model_dump(mode="json")
                    for relation in requirement.relations
                ],
                "qualifiers": list(requirement.qualifiers),
                "topology": requirement.topology.model_dump(
                    mode="json", exclude_none=True
                ),
                "verification": requirement.verification.model_dump(mode="json"),
            }
        )

    compact_artifacts = []
    for artifact in artifacts:
        metadata = {
            key: (value[:240] if isinstance(value, str) else value)
            for key, value in artifact.metadata.items()
            if key
            in {
                "asset_quality_score",
                "catalog_semantics",
                "dense_library_populated_case",
                "generated_from",
                "ontology_path",
                "opening_type",
                "role",
                "start_space_id",
                "end_space_id",
                "wall_direction",
            }
        }
        compact_artifacts.append(
            {
                "artifact_id": artifact.artifact_id,
                "artifact_class": artifact.artifact_class,
                "name": artifact.name,
                "description": artifact.description[:240],
                "dimensions_m": artifact.dimensions_m,
                "position_m": artifact.position_m,
                "metadata": metadata,
            }
        )
    return json.dumps(
        {
            "requirement_graph": {
                "graph_id": graph.graph_id,
                "graph_hash": graph.content_hash,
                "source_prompt": graph.source_prompt,
                "hard_requirements": hard_requirements,
            },
            "spatial_bindings": [
                binding.model_dump(mode="json")
                for binding in compilation.bindings
                if requirement_ids is None or binding.requirement_id in requirement_ids
            ],
            "final_artifacts": compact_artifacts,
        },
        separators=(",", ":"),
    )


async def analyze_final_semantics(
    graph: SceneRequirementGraph,
    compilation: SpatialRequirementCompilation,
    artifacts: tuple[SemanticArtifact, ...],
    *,
    model: str,
    run_config: RunConfig | None = None,
    model_settings: ModelSettings | None = None,
    runner: type[_Runner] = BoundedRunner,
) -> tuple[SemanticVerificationBatch, Any]:
    blocking = tuple(
        requirement
        for requirement in graph.requirements
        if requirement.strength == "hard"
        and requirement.enforcement in {"blocking", "unresolved_blocking"}
    )
    deterministic_claims: list[RequirementVerificationClaim] = []
    unresolved: list[SceneRequirement] = []
    for requirement in blocking:
        claim = _deterministic_claim(requirement, compilation, artifacts)
        if claim is None:
            unresolved.append(requirement)
        else:
            deterministic_claims.append(claim)

    batches = []
    for requirement in unresolved:
        expected_classes = _EXPECTED_FINAL_ARTIFACT_CLASSES.get(
            requirement.kind, frozenset()
        )
        relevant_words = _semantic_words(requirement.subject).union(
            *(_semantic_words(relation.target) for relation in requirement.relations)
        )
        class_artifacts = tuple(
            artifact
            for artifact in artifacts
            if not expected_classes or artifact.artifact_class in expected_classes
        )
        lexical_artifacts = tuple(
            artifact
            for artifact in class_artifacts
            if relevant_words & _artifact_semantic_words(artifact)
        )
        batches.append(
            (
                f"Requirement {requirement.requirement_id}",
                frozenset({requirement.requirement_id}),
                lexical_artifacts or class_artifacts,
            )
        )

    async def run_batch(
        label: str,
        requirement_ids: frozenset[str],
        batch_artifacts: tuple[SemanticArtifact, ...],
    ) -> tuple[SemanticVerificationBatch, Any]:
        verifier = Agent(
            name=f"Scene Semantic Publication Verifier ({label})",
            model=model,
            instructions=SEMANTIC_VERIFIER_INSTRUCTIONS,
            output_type=SemanticVerificationBatch,
            model_settings=model_settings or ModelSettings(),
        )
        result = await runner.run(
            starting_agent=verifier,
            input=semantic_verification_input(
                graph,
                compilation,
                batch_artifacts,
                requirement_ids=requirement_ids,
            ),
            max_turns=1,
            run_config=run_config,
            timeout_seconds=agent_run_timeout_seconds("semantic_verifier", max_turns=1),
        )
        return result.final_output_as(SemanticVerificationBatch), result

    completed = []
    for label, requirement_ids, batch_artifacts in batches:
        # Subscription-backed CLI routes intentionally admit one request at a
        # time. Sequential bounded batches avoid queue-admission failures while
        # still preventing one monolithic verification from exhausting its
        # deadline.
        completed.append(await run_batch(label, requirement_ids, batch_artifacts))
    verifications = tuple(item[0] for item in completed)
    results = tuple(item[1] for item in completed)
    return (
        SemanticVerificationBatch(
            graph_id=graph.graph_id,
            graph_hash=graph.content_hash,
            claims=tuple(deterministic_claims)
            + tuple(
                claim for verification in verifications for claim in verification.claims
            ),
            audit_summary="; ".join(
                (
                    f"Deterministically bound {len(deterministic_claims)} "
                    "source requirements.",
                    *(verification.audit_summary for verification in verifications),
                )
            ),
        ),
        results,
    )


def _expected_count(requirement: SceneRequirement) -> int:
    if requirement.polarity == "forbidden":
        return 0
    if requirement.quantity.mode in {"exact", "minimum"}:
        return int(requirement.quantity.value or 1)
    return int(requirement.quantity.interpreted_minimum or 1)


def certify_semantic_publication(
    graph: SceneRequirementGraph,
    compilation: SpatialRequirementCompilation,
    artifacts: tuple[SemanticArtifact, ...],
    verification: SemanticVerificationBatch,
    *,
    physics_verified: bool,
    physics_evidence_refs: tuple[str, ...],
) -> SemanticPublicationCertificate:
    """Deterministically validate model bindings and issue a closed certificate."""

    if (
        verification.graph_id != graph.graph_id
        or verification.graph_hash != graph.content_hash
    ):
        raise SemanticPublicationError("semantic verification does not match graph")
    if not physics_verified or not physics_evidence_refs:
        raise SemanticPublicationError("physics verification evidence is required")
    inventory = {artifact.artifact_id: artifact for artifact in artifacts}
    hard_requirement_ids = {
        requirement.requirement_id
        for requirement in graph.requirements
        if requirement.strength == "hard"
    }
    unknown_claim_ids = {
        claim.requirement_id for claim in verification.claims
    } - hard_requirement_ids
    if unknown_claim_ids:
        raise SemanticPublicationError(
            f"semantic verification invented requirement IDs: {sorted(unknown_claim_ids)}"
        )
    claims: dict[str, list[RequirementVerificationClaim]] = {}
    for claim in verification.claims:
        claims.setdefault(claim.requirement_id, []).append(claim)
    certified: list[CertifiedRequirement] = []
    failures: list[str] = []
    for requirement in graph.requirements:
        if requirement.strength != "hard":
            continue
        matches = claims.get(requirement.requirement_id, [])
        if len(matches) != 1:
            failures.append(
                f"{requirement.requirement_id} {requirement.subject!r}: expected one "
                f"verification claim, observed {len(matches)}"
            )
            continue
        claim = matches[0]
        unknown = set(claim.artifact_ids) - set(inventory)
        if unknown:
            failures.append(
                f"{requirement.requirement_id} {requirement.subject!r}: unknown "
                f"evidence IDs {sorted(unknown)}"
            )
            continue
        if claim.observed_count != len(set(claim.artifact_ids)):
            failures.append(
                f"{requirement.requirement_id} {requirement.subject!r}: claimed count "
                f"{claim.observed_count} but bound {len(set(claim.artifact_ids))} artifacts"
            )
            continue
        expected_classes = _EXPECTED_FINAL_ARTIFACT_CLASSES.get(
            requirement.kind, frozenset()
        )
        incompatible = sorted(
            artifact_id
            for artifact_id in claim.artifact_ids
            if expected_classes
            and inventory[artifact_id].artifact_class not in expected_classes
        )
        if incompatible:
            failures.append(
                f"{requirement.requirement_id} {requirement.subject!r}: evidence "
                f"artifacts have incompatible classes {incompatible}; expected "
                f"{sorted(expected_classes)}"
            )
            continue
        expected = _expected_count(requirement)
        count_ok = (
            claim.observed_count == expected
            if requirement.quantity.mode == "exact"
            or requirement.polarity == "forbidden"
            else claim.observed_count >= expected
        )
        if claim.status != "satisfied" or not count_ok:
            failures.append(
                f"{requirement.requirement_id} {requirement.subject!r}: expected "
                f"{requirement.quantity.mode} {expected}, observed {claim.observed_count} "
                f"({claim.status}: {claim.semantic_rationale})"
            )
            continue
        if (
            requirement.polarity != "forbidden"
            and requirement.scale is not None
            and requirement.scale.minimum_dimensions_m
        ):
            dimensions = [
                inventory[artifact_id].dimensions_m
                for artifact_id in claim.artifact_ids
            ]
            if not dimensions or not all(
                observed is not None
                and all(
                    observed[index] + 1e-9 >= minimum
                    for index, minimum in enumerate(
                        requirement.scale.minimum_dimensions_m
                    )
                )
                for observed in dimensions
            ):
                failures.append(
                    f"{requirement.requirement_id} {requirement.subject!r}: no bound "
                    f"artifact meets minimum dimensions "
                    f"{requirement.scale.minimum_dimensions_m}"
                )
                continue
        if requirement.polarity != "forbidden" and requirement.relations:
            relation_results = {
                (item.predicate, item.target): item for item in claim.relation_results
            }
            missing_relations = [
                f"{relation.predicate} -> {relation.target}"
                for relation in requirement.relations
                if (relation.predicate, relation.target) not in relation_results
                or not relation_results[(relation.predicate, relation.target)].satisfied
                or not relation_results[
                    (relation.predicate, relation.target)
                ].evidence_artifact_ids
                or not set(
                    relation_results[
                        (relation.predicate, relation.target)
                    ].evidence_artifact_ids
                ).issubset(inventory)
                or not set(
                    relation_results[
                        (relation.predicate, relation.target)
                    ].evidence_artifact_ids
                ).intersection(claim.artifact_ids)
            ]
            if missing_relations:
                failures.append(
                    f"{requirement.requirement_id} {requirement.subject!r}: unmet "
                    f"relationships {missing_relations}"
                )
                continue
        evidence_payload = json.dumps(
            {
                "claim": claim.model_dump(mode="json"),
                "artifacts": [
                    inventory[artifact_id].model_dump(mode="json")
                    for artifact_id in claim.artifact_ids
                ],
            },
            sort_keys=True,
        )
        certified.append(
            CertifiedRequirement(
                requirement_id=requirement.requirement_id,
                subject=requirement.subject,
                artifact_ids=claim.artifact_ids,
                observed_count=claim.observed_count,
                evidence_hash=hashlib.sha256(
                    evidence_payload.encode("utf-8")
                ).hexdigest(),
            )
        )
    blocking_ids = {
        requirement.requirement_id
        for requirement in graph.requirements
        if requirement.strength == "hard"
        and requirement.enforcement in {"blocking", "unresolved_blocking"}
    }
    blocking_failures = [
        failure
        for failure in failures
        if any(requirement_id in failure for requirement_id in blocking_ids)
    ]
    if blocking_failures:
        raise SemanticPublicationError(
            "; ".join(blocking_failures),
            failures=tuple(failures),
            certified_requirements=tuple(certified),
        )
    compilation_hash = hashlib.sha256(
        compilation.model_dump_json().encode("utf-8")
    ).hexdigest()
    return SemanticPublicationCertificate(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        spatial_compilation_hash=compilation_hash,
        physics_verified=True,
        physics_evidence_refs=physics_evidence_refs,
        requirements=tuple(certified),
    )


def persist_publication_artifact(artifact: PublicationModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(artifact.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
