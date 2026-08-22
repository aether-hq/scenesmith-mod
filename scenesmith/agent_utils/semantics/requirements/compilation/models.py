"""Immutable spatial-compilation wire and durable models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    ConnectorEndpoint,
    LevelBlueprint,
    OpeningBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
)


class CompilerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequirementBlueprintBinding(CompilerModel):
    requirement_id: str
    owner_stage: Literal[
        "blueprint",
        "topology",
        "asset",
        "placement",
        "semantic",
        "render",
    ]
    artifact_ids: tuple[str, ...]
    role_key: str | None = None
    planned_instances: int
    rationale: str


class SpatialRequirementCompilation(CompilerModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    blueprint: SceneBlueprint
    bindings: tuple[RequirementBlueprintBinding, ...]
    compilation_summary: str


class NamedStringWire(CompilerModel):
    name: str
    value: str


class RoleCountWire(CompilerModel):
    role: str
    count: int


class IntermediateLandingWire(CompilerModel):
    space_id: str
    level_id: str
    position_m: tuple[float, float, float]


class ConnectorBlueprintWire(CompilerModel):
    connector_id: str
    kind: Literal[
        "stairs_straight",
        "stairs_l",
        "stairs_u",
        "stairs_spiral",
        "ramp",
        "ladder",
        "elevator",
    ]
    start: ConnectorEndpoint
    end: ConnectorEndpoint
    width_m: float
    intermediate_landings: tuple[IntermediateLandingWire, ...]


class FurnitureGroupBlueprintWire(CompilerModel):
    group_id: str
    name: str
    space_id: str
    roles: tuple[RoleCountWire, ...]
    focal_target: str | None
    density: Literal["sparse", "balanced", "layered"]


class BlueprintDesignTokensWire(CompilerModel):
    style_keywords: tuple[str, ...]
    palette: tuple[str, ...]
    material_roles: tuple[NamedStringWire, ...]
    lighting_mood: str
    focal_hierarchy: tuple[str, ...]


class RequirementBlueprintBindingWire(CompilerModel):
    requirement_id: str
    owner_stage: Literal[
        "blueprint",
        "topology",
        "asset",
        "placement",
        "semantic",
        "render",
    ]
    artifact_ids: tuple[str, ...]
    role_key: str | None


class SpatialRequirementCompilationWire(CompilerModel):
    """Bounded model output expanded deterministically into the durable contract."""

    blueprint_id: str
    levels: tuple[LevelBlueprint, ...]
    spaces: tuple[SpaceBlueprint, ...]
    openings: tuple[OpeningBlueprint, ...]
    connectors: tuple[ConnectorBlueprintWire, ...]
    furniture_groups: tuple[FurnitureGroupBlueprintWire, ...]
    design_tokens: BlueprintDesignTokensWire
    bindings: tuple[RequirementBlueprintBindingWire, ...]
    compilation_summary: str


class SpatialCompilationError(ValueError):
    """The spatial compilation omitted or weakened an immutable obligation."""


class TopologyRequirementEvidence(CompilerModel):
    requirement_id: str
    actual_artifact_ids: tuple[str, ...]
    observed_count: int
    diagnostic: str


class TopologyStageManifest(CompilerModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    evidence: tuple[TopologyRequirementEvidence, ...]
    passed: Literal[True] = True
