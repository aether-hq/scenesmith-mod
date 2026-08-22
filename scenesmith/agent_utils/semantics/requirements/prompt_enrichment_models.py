"""Immutable models for graph-bound semantic prompt enrichment."""

from __future__ import annotations

import hashlib
import json

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint


class EnrichmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InferredSceneElement(EnrichmentModel):
    category: Literal[
        "architecture",
        "circulation",
        "infrastructure",
        "logistics",
        "operations",
        "safety",
        "story",
        "visual_language",
    ]
    description: str
    rationale: str


class RequirementPromptWire(EnrichmentModel):
    requirement_id: str
    operational_role: str
    visual_identity: str
    construction_prompt: str


class SceneEnrichmentWire(EnrichmentModel):
    domain_context: str
    scene_purpose: str
    operational_logic: str
    spatial_logic: str
    visual_language: str
    enriched_prompt: str
    inferred_elements: tuple[InferredSceneElement, ...] = Field(max_length=12)
    requirement_prompts: tuple[RequirementPromptWire, ...]


class RequirementPrompt(EnrichmentModel):
    requirement_id: str
    subject: str
    operational_role: str
    visual_identity: str
    construction_prompt: str
    source: Literal["model", "deterministic_fallback"]


class SceneEnrichmentDraft(EnrichmentModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    source_prompt: str
    domain_context: str
    scene_purpose: str
    operational_logic: str
    spatial_logic: str
    visual_language: str
    enriched_prompt: str
    inferred_elements: tuple[InferredSceneElement, ...]
    requirement_prompts: tuple[RequirementPrompt, ...]
    analysis_model: str | None = None
    analysis_status: Literal["complete", "partial", "deterministic_fallback"]
    diagnostics: tuple[str, ...] = ()

    @property
    def content_hash(self) -> str:
        return _content_hash(self.model_dump(mode="json"))


class RepeatedRoleTarget(EnrichmentModel):
    target_id: str
    requirement_id: str | None
    subject: str
    artifact_kind: Literal["furniture_role", "opening"]
    artifact_ids: tuple[str, ...]
    role_key: str | None
    instance_count: int
    shared_prompt: str

    @model_validator(mode="after")
    def validate_target(self) -> "RepeatedRoleTarget":
        if self.instance_count < 2:
            raise ValueError("a repeated-role target requires at least two instances")
        if not self.artifact_ids:
            raise ValueError("a repeated-role target requires blueprint artifacts")
        if self.artifact_kind == "furniture_role" and not self.role_key:
            raise ValueError("a repeated furniture target requires role_key")
        return self


class InstancePromptWire(EnrichmentModel):
    instance_index: int = Field(ge=0)
    name: str
    function: str
    description: str
    geometry_cues: tuple[str, ...] = Field(min_length=1, max_length=4)
    equipment_cues: tuple[str, ...] = Field(min_length=1, max_length=4)
    material_cues: tuple[str, ...] = Field(min_length=1, max_length=3)
    operational_relationship: str

    @model_validator(mode="after")
    def validate_substantive_fields(self) -> "InstancePromptWire":
        fields = (
            self.name,
            self.function,
            self.description,
            self.operational_relationship,
            *self.geometry_cues,
            *self.equipment_cues,
            *self.material_cues,
        )
        if any(not value.strip() for value in fields):
            raise ValueError("instance descriptions and cues must be non-empty")
        return self


class RepeatedRoleWire(EnrichmentModel):
    target_id: str
    shared_design_language: str
    instances: tuple[InstancePromptWire, ...] = Field(min_length=2)


class RepeatedEnrichmentWireBatch(EnrichmentModel):
    targets: tuple[RepeatedRoleWire, ...]


class InstancePrompt(EnrichmentModel):
    instance_id: str
    instance_index: int
    artifact_ref: str
    name: str
    function: str
    description: str
    geometry_cues: tuple[str, ...]
    equipment_cues: tuple[str, ...]
    material_cues: tuple[str, ...]
    operational_relationship: str
    construction_prompt: str
    source: Literal["model", "deterministic_fallback"]


class RepeatedRoleEnrichment(EnrichmentModel):
    target_id: str
    requirement_id: str | None
    subject: str
    artifact_kind: Literal["furniture_role", "opening"]
    artifact_ids: tuple[str, ...]
    role_key: str | None
    shared_design_language: str
    instances: tuple[InstancePrompt, ...]

    @model_validator(mode="after")
    def validate_instances(self) -> "RepeatedRoleEnrichment":
        expected = tuple(range(len(self.instances)))
        observed = tuple(item.instance_index for item in self.instances)
        if observed != expected:
            raise ValueError(
                f"instances for {self.target_id} must use contiguous zero-based indices"
            )
        names = [item.name.casefold().strip() for item in self.instances]
        prompts = [
            item.construction_prompt.casefold().strip() for item in self.instances
        ]
        if len(names) != len(set(names)):
            raise ValueError(f"instances for {self.target_id} require unique names")
        if len(prompts) != len(set(prompts)):
            raise ValueError(
                f"instances for {self.target_id} require unique construction prompts"
            )
        return self


class SemanticPromptEnrichment(EnrichmentModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    blueprint_id: str
    blueprint_hash: str
    source_prompt: str
    scene: SceneEnrichmentDraft
    repeated_roles: tuple[RepeatedRoleEnrichment, ...]
    complete_prompt: str
    analysis_model: str | None = None
    analysis_status: Literal["complete", "partial", "deterministic_fallback"]
    diagnostics: tuple[str, ...] = ()

    @property
    def content_hash(self) -> str:
        return _content_hash(self.model_dump(mode="json"))


def _content_hash(payload: Any) -> str:

    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def blueprint_content_hash(blueprint: SceneBlueprint) -> str:

    return _content_hash(blueprint.model_dump(mode="json", exclude={"repair_log"}))
