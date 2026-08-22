"""Immutable wire models used by semantic publication and certification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
