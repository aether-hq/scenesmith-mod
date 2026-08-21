"""LLM semantic binding plus deterministic fail-closed publication certification."""

from __future__ import annotations

import hashlib
import json
import os
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
"""


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

    artifacts: list[SemanticArtifact] = [
        SemanticArtifact(
            artifact_id=blueprint.blueprint_id,
            artifact_class="scene",
            name="compiled scene",
            description=blueprint.source_prompt,
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
) -> str:
    return json.dumps(
        {
            "requirement_graph": graph.model_dump(mode="json"),
            "spatial_bindings": [
                binding.model_dump(mode="json") for binding in compilation.bindings
            ],
            "final_artifacts": [
                artifact.model_dump(mode="json") for artifact in artifacts
            ],
        },
        indent=2,
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
    verifier = Agent(
        name="Scene Semantic Publication Verifier",
        model=model,
        instructions=SEMANTIC_VERIFIER_INSTRUCTIONS,
        output_type=SemanticVerificationBatch,
        model_settings=model_settings or ModelSettings(),
    )
    result = await runner.run(
        starting_agent=verifier,
        input=semantic_verification_input(graph, compilation, artifacts),
        max_turns=1,
        run_config=run_config,
        timeout_seconds=agent_run_timeout_seconds("semantic_verifier", max_turns=1),
    )
    return result.final_output_as(SemanticVerificationBatch), result


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
            if not any(
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
            relation_results = {item.predicate: item for item in claim.relation_results}
            missing_relations = [
                relation.predicate
                for relation in requirement.relations
                if relation.predicate not in relation_results
                or not relation_results[relation.predicate].satisfied
                or not relation_results[relation.predicate].evidence_artifact_ids
                or not set(
                    relation_results[relation.predicate].evidence_artifact_ids
                ).issubset(inventory)
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
