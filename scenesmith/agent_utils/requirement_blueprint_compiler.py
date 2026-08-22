"""LLM-authored spatial compilation of immutable semantic obligations."""

from __future__ import annotations

import json
import os
import tempfile

from pathlib import Path
from typing import Any, Literal, Protocol

from agents import Agent, AgentOutputSchema, ModelSettings, RunConfig
from pydantic import BaseModel, ConfigDict

from scenesmith.agent_utils.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.scene_blueprint import (
    BlueprintConstraint,
    BlueprintDesignTokens,
    ConnectorBlueprint,
    ConnectorEndpoint,
    FurnitureGroupBlueprint,
    LevelBlueprint,
    OpeningBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
)
from scenesmith.agent_utils.scene_requirements import (
    SceneRequirement,
    SceneRequirementGraph,
)
from scenesmith.agent_utils.semantic_strategies import SemanticCapabilityManifest


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


class _Runner(Protocol):
    @staticmethod
    async def run(*args: Any, **kwargs: Any) -> Any: ...


SPATIAL_COMPILER_INSTRUCTIONS = """
You are SceneSmith's spatial and topological compiler. The input requirement graph
is immutable source truth; you may design an implementation but may not omit,
rename, weaken, merge away, or reduce any blocking obligation.

Return one compact provider-neutral blueprint wire and a binding for every hard
requirement.
Use the LLM-authored metric scale, topology, relations, composition plan, and scene
composition opinion. Do not substitute a generic room for an unfamiliar concept.
Output only the requested schema. Keep compilation_summary to two short sentences,
every rationale and prompt to at most 18 words, and parameter prose to short evidence
phrases. Do not restate the graph or narrate arithmetic already represented by fields.

Rules:
1. Do not repeat graph_id, graph_hash, source_prompt, mode, constraints, or locked
   IDs. Deterministic expansion injects them from immutable input.
2. Every blocking required obligation must bind to concrete blueprint artifact IDs.
   Scene types bind the blueprint ID; levels bind level IDs; openings bind opening
   IDs; connectors bind connector IDs; spatial constraints bind constraint IDs;
   object and repeated-zone obligations bind furniture groups and a precise role_key
   when implemented inside one shared volume.
3. Emit one binding for every hard obligation. Deterministic expansion adds its hard
   BlueprintConstraint, count, verification criteria, metric envelope, and relations.
4. Exact counts stay exact. Minimum and LLM-interpreted qualitative minima may be
   exceeded only when the composition needs it. A furniture binding's role counts
   must sum to the immutable manifest's planned instances.
5. Forbidden obligations bind to owner_stage=semantic, artifact_ids=[], and
   and no role_key. They are absence checks, never construction requests.
6. Prefer one coherent primary volume for an interior set. Repeated operational
   zones inside it may be explicit furniture groups backed by hard topology
   constraints; do not invent disconnected rooms merely to satisfy a count.
7. Choose metric dimensions from the authored scale and composition needs. Scene
   dimensions are not computational budgets: never shrink a hangar, opening, or hero
   object to residential defaults. If the caller supplies an explicit backend
   capability limit, preserve hard minima so deterministic validation can report the
   capability mismatch instead of silently clipping it.
8. Produce IDs that are stable, readable, and globally unique. Deterministic
   expansion locks every artifact and constraint against later candidate discard.
"""


def _planned_instances(requirement: SceneRequirement) -> int:
    if requirement.polarity == "forbidden":
        return 0
    if requirement.quantity.mode in {"exact", "minimum"}:
        return int(requirement.quantity.value or 1)
    return int(requirement.quantity.interpreted_minimum or 1)


def _artifact_collections(blueprint: SceneBlueprint) -> dict[str, str]:
    artifacts = {blueprint.blueprint_id: "scene"}
    artifacts.update({item.level_id: "level" for item in blueprint.levels})
    artifacts.update({item.space_id: "space" for item in blueprint.spaces})
    artifacts.update({item.opening_id: "opening" for item in blueprint.openings})
    artifacts.update({item.connector_id: "connector" for item in blueprint.connectors})
    artifacts.update(
        {item.group_id: "furniture_group" for item in blueprint.furniture_groups}
    )
    artifacts.update(
        {item.constraint_id: "constraint" for item in blueprint.constraints}
    )
    return artifacts


_EXPECTED_ARTIFACT_CLASSES = {
    "scene_type": frozenset({"scene"}),
    "level": frozenset({"level"}),
    "repeated_zone": frozenset({"space", "furniture_group"}),
    "hero_object": frozenset({"furniture_group"}),
    "opening": frozenset({"opening"}),
    "connector": frozenset({"connector"}),
    "object_group": frozenset({"furniture_group"}),
    "spatial_constraint": frozenset({"constraint"}),
    "style": frozenset({"scene"}),
}


def _bound_role_count(
    blueprint: SceneBlueprint, binding: RequirementBlueprintBinding
) -> int | None:
    if binding.role_key is None:
        return None
    groups = {group.group_id: group for group in blueprint.furniture_groups}
    return sum(
        groups[artifact_id].roles.get(binding.role_key, 0)
        for artifact_id in binding.artifact_ids
        if artifact_id in groups
    )


def _validate_expected_mode_spaces(
    blueprint: SceneBlueprint,
    expected_mode: Literal["room", "house"] | None,
) -> None:
    if expected_mode != "room":
        return
    space_level_ids = [space.level_id for space in blueprint.spaces]
    level_ids = {level.level_id for level in blueprint.levels}
    if len(space_level_ids) != len(level_ids) or set(space_level_ids) != level_ids:
        raise SpatialCompilationError(
            "room-mode compilation must use one coherent space per level; repeated "
            "internal zones belong in bound furniture groups and constraints"
        )


def validate_spatial_compilation(
    compilation: SpatialRequirementCompilation,
    graph: SceneRequirementGraph,
    *,
    maximum_dimension_m: float | None = None,
    maximum_height_m: float | None = None,
    expected_mode: Literal["room", "house"] | None = None,
) -> None:
    """Fail closed on omissions, count changes, invalid bindings, or scale clipping."""

    if (
        compilation.graph_id != graph.graph_id
        or compilation.graph_hash != graph.content_hash
    ):
        raise SpatialCompilationError(
            "spatial compilation does not match requirement graph"
        )
    blueprint = compilation.blueprint
    if expected_mode is not None and blueprint.mode != expected_mode:
        raise SpatialCompilationError(
            f"spatial compiler changed mode from {expected_mode!r} to "
            f"{blueprint.mode!r}"
        )
    _validate_expected_mode_spaces(blueprint, expected_mode)
    if blueprint.source_prompt != graph.source_prompt:
        raise SpatialCompilationError("spatial compiler changed the source prompt")
    if maximum_dimension_m is not None and any(
        max(space.dimensions_m) > maximum_dimension_m for space in blueprint.spaces
    ):
        raise SpatialCompilationError(
            f"compiled plan exceeds configured {maximum_dimension_m:g}m floor dimension"
        )
    total_height = max(
        level.elevation_m + level.clear_height_m for level in blueprint.levels
    )
    if maximum_height_m is not None and total_height > maximum_height_m:
        raise SpatialCompilationError(
            f"compiled plan exceeds configured {maximum_height_m:g}m height"
        )

    artifacts = _artifact_collections(blueprint)
    hard_requirement_ids = {
        requirement.requirement_id
        for requirement in graph.requirements
        if requirement.strength == "hard"
    }
    unknown_binding_ids = {
        binding.requirement_id for binding in compilation.bindings
    } - hard_requirement_ids
    if unknown_binding_ids:
        raise SpatialCompilationError(
            f"spatial compiler invented requirement IDs: {sorted(unknown_binding_ids)}"
        )
    bindings_by_requirement: dict[str, list[RequirementBlueprintBinding]] = {}
    for binding in compilation.bindings:
        bindings_by_requirement.setdefault(binding.requirement_id, []).append(binding)
        unknown = set(binding.artifact_ids) - set(artifacts)
        if unknown:
            raise SpatialCompilationError(
                f"{binding.requirement_id} binds unknown artifacts: {sorted(unknown)}"
            )

    constraints_by_requirement: dict[str, list[Any]] = {}
    for constraint in blueprint.constraints:
        requirement_id = str(constraint.parameters.get("requirement_id") or "")
        if requirement_id:
            constraints_by_requirement.setdefault(requirement_id, []).append(constraint)

    for requirement in graph.requirements:
        if requirement.strength != "hard":
            continue
        bindings = bindings_by_requirement.get(requirement.requirement_id, [])
        if len(bindings) != 1:
            raise SpatialCompilationError(
                f"{requirement.requirement_id} ({requirement.subject!r}) has "
                f"{len(bindings)} blueprint bindings; expected exactly one"
            )
        binding = bindings[0]
        expected_count = _planned_instances(requirement)
        if binding.planned_instances != expected_count:
            raise SpatialCompilationError(
                f"{requirement.requirement_id} changed planned count from "
                f"{expected_count} to {binding.planned_instances}"
            )
        if requirement.polarity == "forbidden":
            if binding.owner_stage != "semantic" or binding.artifact_ids:
                raise SpatialCompilationError(
                    f"forbidden requirement {requirement.requirement_id} must be an "
                    "artifact-free semantic absence guard"
                )
            continue
        if requirement.enforcement == "blocking" and not binding.artifact_ids:
            raise SpatialCompilationError(
                f"blocking requirement {requirement.requirement_id} has no artifact binding"
            )
        expected_classes = _EXPECTED_ARTIFACT_CLASSES.get(requirement.kind, frozenset())
        observed_classes = {artifacts[item] for item in binding.artifact_ids}
        if expected_classes and not observed_classes.intersection(expected_classes):
            raise SpatialCompilationError(
                f"{requirement.requirement_id} ({requirement.kind}) binds "
                f"incompatible artifact classes {sorted(observed_classes)}"
            )
        role_count = _bound_role_count(blueprint, binding)
        if "furniture_group" in observed_classes:
            role_count_ok = (
                role_count == expected_count
                if requirement.quantity.mode == "exact"
                else role_count is not None and role_count >= expected_count
            )
            if binding.role_key is None or not role_count_ok:
                raise SpatialCompilationError(
                    f"{requirement.requirement_id} furniture role count is "
                    f"{role_count!r}; expected {expected_count} "
                    f"({requirement.quantity.mode})"
                )
        elif (
            requirement.quantity.mode == "exact"
            and len(binding.artifact_ids) != expected_count
        ):
            raise SpatialCompilationError(
                f"{requirement.requirement_id} binds {len(binding.artifact_ids)} artifacts; "
                f"expected exactly {expected_count}"
            )
        elif len(binding.artifact_ids) < expected_count:
            raise SpatialCompilationError(
                f"{requirement.requirement_id} binds {len(binding.artifact_ids)} "
                f"artifacts; expected at least {expected_count}"
            )
        owned_constraints = constraints_by_requirement.get(
            requirement.requirement_id, []
        )
        if requirement.enforcement == "blocking" and not owned_constraints:
            raise SpatialCompilationError(
                f"blocking requirement {requirement.requirement_id} has no hard "
                "blueprint constraint"
            )
        for constraint in owned_constraints:
            if constraint.strength != "hard" or constraint.source != "user":
                raise SpatialCompilationError(
                    f"{requirement.requirement_id} obligation constraint is not "
                    "hard user-authored intent"
                )
            if (
                int(constraint.parameters.get("planned_instances", -1))
                != expected_count
            ):
                raise SpatialCompilationError(
                    f"{requirement.requirement_id} obligation constraint changed its "
                    "planned instance count"
                )
            if not constraint.parameters.get("verification_criteria"):
                raise SpatialCompilationError(
                    f"{requirement.requirement_id} obligation constraint omitted "
                    "verification criteria"
                )
            if requirement.scale is not None and requirement.scale.minimum_dimensions_m:
                observed_dimensions = constraint.parameters.get("minimum_dimensions_m")
                if (
                    not isinstance(observed_dimensions, (list, tuple))
                    or len(observed_dimensions) != 3
                ):
                    raise SpatialCompilationError(
                        f"{requirement.requirement_id} obligation constraint omitted "
                        "minimum_dimensions_m"
                    )
                if any(
                    float(observed_dimensions[index]) + 1e-9 < required
                    for index, required in enumerate(
                        requirement.scale.minimum_dimensions_m
                    )
                ):
                    raise SpatialCompilationError(
                        f"{requirement.requirement_id} obligation constraint shrank "
                        "the LLM-authored minimum dimensions"
                    )
            if requirement.relations and not constraint.parameters.get("relationships"):
                raise SpatialCompilationError(
                    f"{requirement.requirement_id} obligation constraint omitted "
                    "required spatial relationships"
                )
        if not set(binding.artifact_ids) <= set(blueprint.locked_ids):
            raise SpatialCompilationError(
                f"{requirement.requirement_id} has unlocked bound artifacts"
            )


def spatial_compilation_input(
    graph: SceneRequirementGraph,
    manifest: SemanticCapabilityManifest,
    *,
    mode: Literal["room", "house"],
    maximum_dimension_m: float | None = None,
    maximum_height_m: float | None = None,
    maximum_opening_width_m: float | None = None,
    maximum_opening_height_m: float | None = None,
) -> str:
    return json.dumps(
        {
            "requirement_graph": graph.model_dump(mode="json"),
            "capability_manifest": manifest.model_dump(mode="json"),
            "construction_capabilities": {
                "mode": mode,
                "maximum_floor_dimension_m": maximum_dimension_m,
                "maximum_total_height_m": maximum_height_m,
                "maximum_opening_width_m": maximum_opening_width_m,
                "maximum_opening_height_m": maximum_opening_height_m,
            },
        },
        indent=2,
    )


def _unique_named_values(
    entries: tuple[NamedStringWire, ...],
    *,
    label: str,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in entries:
        if entry.name in values:
            raise SpatialCompilationError(f"duplicate {label} key {entry.name!r}")
        values[entry.name] = entry.value
    return values


def _role_counts(entries: tuple[RoleCountWire, ...]) -> dict[str, int]:
    roles: dict[str, int] = {}
    for entry in entries:
        if entry.role in roles:
            raise SpatialCompilationError(f"duplicate furniture role {entry.role!r}")
        roles[entry.role] = entry.count
    return roles


def _constraint_parameters(
    requirement: SceneRequirement,
    binding: RequirementBlueprintBinding,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "requirement_id": requirement.requirement_id,
        "planned_instances": _planned_instances(requirement),
        "verification_criteria": list(requirement.verification.measurable_criteria),
    }
    if binding.role_key is not None:
        parameters["role_key"] = binding.role_key
    if requirement.scale is not None:
        if requirement.scale.minimum_dimensions_m is not None:
            parameters["minimum_dimensions_m"] = list(
                requirement.scale.minimum_dimensions_m
            )
        if requirement.scale.preferred_dimensions_m is not None:
            parameters["preferred_dimensions_m"] = list(
                requirement.scale.preferred_dimensions_m
            )
        if requirement.scale.clearance_m is not None:
            parameters["clearance_m"] = requirement.scale.clearance_m
    if requirement.relations:
        parameters["relationships"] = [
            relation.model_dump(mode="json") for relation in requirement.relations
        ]
    return parameters


def _assert_hard_scene_scale_fits(
    graph: SceneRequirementGraph,
    *,
    maximum_dimension_m: float | None,
    maximum_height_m: float | None,
) -> None:
    for requirement in graph.requirements:
        if requirement.kind != "scene_type" or requirement.scale is None:
            continue
        minimum = requirement.scale.minimum_dimensions_m
        if minimum is None:
            continue
        if maximum_dimension_m is not None and max(minimum[0], minimum[2]) > float(
            maximum_dimension_m
        ):
            raise SpatialCompilationError(
                f"{requirement.requirement_id} user-authored floor minimum "
                f"{minimum} exceeds the {maximum_dimension_m:g}m compiler limit"
            )
        if maximum_height_m is not None and minimum[1] > float(maximum_height_m):
            raise SpatialCompilationError(
                f"{requirement.requirement_id} user-authored height minimum "
                f"{minimum[1]:g}m exceeds the {maximum_height_m:g}m compiler limit"
            )


def _project_levels_to_height(
    levels: tuple[LevelBlueprint, ...],
    maximum_height_m: float | None,
) -> tuple[LevelBlueprint, ...]:
    if maximum_height_m is None:
        return levels
    total_height = max(level.elevation_m + level.clear_height_m for level in levels)
    if total_height <= maximum_height_m:
        return levels
    per_level_height = float(maximum_height_m) / len(levels)
    if per_level_height < 2.2:
        raise SpatialCompilationError(
            f"{len(levels)} required levels cannot fit usable clear heights inside "
            f"the {maximum_height_m:g}m compiler limit"
        )
    ordered = sorted(levels, key=lambda level: level.elevation_m)
    projected_by_id: dict[str, LevelBlueprint] = {}
    elevation = 0.0
    for level in ordered:
        clear_height = min(level.clear_height_m, per_level_height)
        projected_by_id[level.level_id] = level.model_copy(
            update={
                "elevation_m": elevation,
                "clear_height_m": clear_height,
            }
        )
        elevation += clear_height
    return tuple(projected_by_id[level.level_id] for level in levels)


def _project_endpoint(
    endpoint: ConnectorEndpoint,
    *,
    level_elevations: dict[str, float],
    maximum_dimension_m: float | None,
) -> ConnectorEndpoint:
    x, y, _ = endpoint.position_m
    if maximum_dimension_m is not None:
        half_extent = float(maximum_dimension_m) / 2.0
        x = max(-half_extent, min(half_extent, x))
        y = max(-half_extent, min(half_extent, y))
    return endpoint.model_copy(
        update={
            "position_m": (
                x,
                y,
                level_elevations.get(endpoint.level_id, endpoint.position_m[2]),
            )
        }
    )


def expand_spatial_compilation(
    graph: SceneRequirementGraph,
    wire: SpatialRequirementCompilationWire,
    *,
    mode: Literal["room", "house"],
    maximum_dimension_m: float | None = None,
    maximum_height_m: float | None = None,
    maximum_opening_width_m: float | None = None,
    maximum_opening_height_m: float | None = None,
) -> SpatialRequirementCompilation:
    """Expand compact model topology into the immutable durable blueprint."""

    _assert_hard_scene_scale_fits(
        graph,
        maximum_dimension_m=maximum_dimension_m,
        maximum_height_m=maximum_height_m,
    )
    requirements = {item.requirement_id: item for item in graph.requirements}
    bindings: list[RequirementBlueprintBinding] = []
    for item in wire.bindings:
        requirement = requirements.get(item.requirement_id)
        artifact_ids = item.artifact_ids
        if requirement is not None and requirement.kind == "spatial_constraint":
            artifact_ids = (f"constraint-{requirement.requirement_id}",)
        bindings.append(
            RequirementBlueprintBinding(
                requirement_id=item.requirement_id,
                owner_stage=item.owner_stage,
                artifact_ids=artifact_ids,
                role_key=item.role_key,
                planned_instances=(
                    _planned_instances(requirement) if requirement is not None else 0
                ),
                rationale="Compact spatial compiler bound this source obligation.",
            )
        )

    binding_by_requirement = {binding.requirement_id: binding for binding in bindings}
    constraints = tuple(
        BlueprintConstraint(
            constraint_id=f"constraint-{requirement.requirement_id}",
            kind=f"semantic_{requirement.kind}",
            target_ids=binding_by_requirement[requirement.requirement_id].artifact_ids,
            parameters=_constraint_parameters(
                requirement,
                binding_by_requirement[requirement.requirement_id],
            ),
            strength="hard",
            source="user",
        )
        for requirement in graph.requirements
        if requirement.strength == "hard"
        and requirement.polarity != "forbidden"
        and requirement.requirement_id in binding_by_requirement
    )
    levels = _project_levels_to_height(wire.levels, maximum_height_m)
    level_elevations = {level.level_id: level.elevation_m for level in levels}
    spaces = tuple(
        item.model_copy(
            update={
                "dimensions_m": tuple(
                    (
                        min(float(maximum_dimension_m), dimension)
                        if maximum_dimension_m is not None
                        else dimension
                    )
                    for dimension in item.dimensions_m
                )
            }
        )
        for item in wire.spaces
    )
    openings = tuple(
        item.model_copy(
            update={
                "width_m": (
                    min(item.width_m, maximum_opening_width_m)
                    if maximum_opening_width_m is not None
                    else item.width_m
                ),
                "height_m": (
                    min(item.height_m, maximum_opening_height_m)
                    if maximum_opening_height_m is not None
                    else item.height_m
                ),
            }
        )
        for item in wire.openings
    )
    connectors = tuple(
        ConnectorBlueprint(
            connector_id=item.connector_id,
            kind=item.kind,
            start=_project_endpoint(
                item.start,
                level_elevations=level_elevations,
                maximum_dimension_m=maximum_dimension_m,
            ),
            end=_project_endpoint(
                item.end,
                level_elevations=level_elevations,
                maximum_dimension_m=maximum_dimension_m,
            ),
            width_m=item.width_m,
            parameters={
                "intermediate_landings": [
                    {
                        **landing.model_dump(mode="json"),
                        "position_m": list(
                            _project_endpoint(
                                ConnectorEndpoint(
                                    space_id=landing.space_id,
                                    level_id=landing.level_id,
                                    position_m=landing.position_m,
                                ),
                                level_elevations=level_elevations,
                                maximum_dimension_m=maximum_dimension_m,
                            ).position_m
                        ),
                    }
                    for landing in item.intermediate_landings
                ]
            },
        )
        for item in wire.connectors
    )
    furniture_groups = tuple(
        FurnitureGroupBlueprint(
            group_id=item.group_id,
            name=item.name,
            space_id=item.space_id,
            roles=_role_counts(item.roles),
            focal_target=item.focal_target,
            density=item.density,
        )
        for item in wire.furniture_groups
    )
    design_tokens = BlueprintDesignTokens(
        style_keywords=wire.design_tokens.style_keywords,
        palette=wire.design_tokens.palette,
        material_roles=_unique_named_values(
            wire.design_tokens.material_roles,
            label="material role",
        ),
        lighting_mood=wire.design_tokens.lighting_mood,
        focal_hierarchy=wire.design_tokens.focal_hierarchy,
    )
    locked_ids = tuple(
        dict.fromkeys(
            [
                wire.blueprint_id,
                *(item.level_id for item in levels),
                *(item.space_id for item in spaces),
                *(item.opening_id for item in wire.openings),
                *(item.connector_id for item in connectors),
                *(item.group_id for item in furniture_groups),
                *(item.constraint_id for item in constraints),
            ]
        )
    )
    blueprint = SceneBlueprint(
        blueprint_id=wire.blueprint_id,
        source_prompt=graph.source_prompt,
        mode=mode,
        levels=levels,
        spaces=spaces,
        openings=openings,
        connectors=connectors,
        furniture_groups=furniture_groups,
        design_tokens=design_tokens,
        constraints=constraints,
        locked_ids=locked_ids,
    )
    return SpatialRequirementCompilation(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        blueprint=blueprint,
        bindings=tuple(bindings),
        compilation_summary=wire.compilation_summary,
    )


def blueprint_with_obligation_brief(
    compilation: SpatialRequirementCompilation,
    graph: SceneRequirementGraph,
) -> SceneBlueprint:
    """Carry immutable obligations into every downstream room-agent prompt."""

    requirements = {item.requirement_id: item for item in graph.requirements}
    brief = {
        "semantic_obligations": [
            {
                "requirement_id": binding.requirement_id,
                "subject": requirements[binding.requirement_id].subject,
                "polarity": requirements[binding.requirement_id].polarity,
                "kind": requirements[binding.requirement_id].kind,
                "planned_instances": binding.planned_instances,
                "artifact_ids": binding.artifact_ids,
                "role_key": binding.role_key,
                "scale": (
                    requirements[binding.requirement_id].scale.model_dump(mode="json")
                    if requirements[binding.requirement_id].scale is not None
                    else None
                ),
                "relations": [
                    relation.model_dump(mode="json")
                    for relation in requirements[binding.requirement_id].relations
                ],
                "verification": requirements[
                    binding.requirement_id
                ].verification.model_dump(mode="json"),
            }
            for binding in compilation.bindings
        ]
    }
    suffix = (
        "\n\nIMMUTABLE SEMANTIC OBLIGATIONS — do not omit or reduce:\n"
        + json.dumps(brief, separators=(",", ":"), sort_keys=True)
    )
    return compilation.blueprint.model_copy(
        update={
            "spaces": tuple(
                space.model_copy(
                    update={"prompt": (space.prompt or graph.source_prompt) + suffix}
                )
                for space in compilation.blueprint.spaces
            )
        }
    )


def validate_constructed_topology(
    compilation: SpatialRequirementCompilation,
    graph: SceneRequirementGraph,
    house_layout: Any,
) -> TopologyStageManifest:
    """Verify topology-owned bindings against the constructed HouseLayout."""

    blueprint = compilation.blueprint
    bindings = {item.requirement_id: item for item in compilation.bindings}
    requirements = {item.requirement_id: item for item in graph.requirements}
    blueprint_artifact_classes = _artifact_collections(blueprint)
    blueprint_level_order = {
        level.level_id: index
        for index, level in enumerate(
            sorted(blueprint.levels, key=lambda item: item.elevation_m)
        )
    }
    actual_levels = {item.level_id: item for item in house_layout.levels}
    actual_spaces = {item.room_id: item for item in house_layout.room_specs}
    actual_connectors = {item.connector_id: item for item in house_layout.connectors}
    available_actual_space_ids = set(actual_spaces)
    unclaimed_actual_space_ids = set(actual_spaces)
    space_aliases: dict[str, str] = {}
    for blueprint_space in blueprint.spaces:
        if blueprint_space.space_id in available_actual_space_ids:
            constructed_space_id = blueprint_space.space_id
        elif blueprint.mode == "room" and len(actual_spaces) == 1:
            constructed_space_id = next(iter(actual_spaces))
        else:
            matching_space_ids = sorted(
                candidate_id
                for candidate_id in unclaimed_actual_space_ids
                if getattr(actual_spaces[candidate_id], "room_type", None)
                == blueprint_space.room_type
            )
            if not matching_space_ids:
                continue
            constructed_space_id = matching_space_ids[0]
        space_aliases[blueprint_space.space_id] = constructed_space_id
        unclaimed_actual_space_ids.discard(constructed_space_id)

    def actual_space_id(blueprint_space_id: str) -> str:
        return space_aliases.get(blueprint_space_id, blueprint_space_id)

    actual_openings: dict[str, list[tuple[str, Any]]] = {}
    for room_id, room_geometry in house_layout.room_geometries.items():
        for opening in room_geometry.openings:
            actual_openings.setdefault(opening.opening_id, []).append(
                (room_id, opening)
            )
    used_openings: set[str] = set()
    used_connectors: set[str] = set()
    evidence: list[TopologyRequirementEvidence] = []

    for requirement_id, binding in bindings.items():
        requirement = requirements[requirement_id]
        if requirement.polarity == "forbidden":
            continue
        bound_classes = {
            blueprint_artifact_classes[artifact_id]
            for artifact_id in binding.artifact_ids
        }
        actual_ids: tuple[str, ...] = ()
        semantic_observed_count: int | None = None
        if requirement.kind == "scene_type":
            if not actual_spaces:
                raise SpatialCompilationError(
                    f"{requirement_id} {requirement.subject!r}: constructed topology "
                    "contains no primary space"
                )
            actual_ids = tuple(sorted(actual_spaces))
        elif requirement.kind == "level":
            expected = _planned_instances(requirement)
            observed = len(actual_levels)
            count_ok = (
                observed == expected
                if requirement.quantity.mode == "exact"
                else observed >= expected
            )
            if not count_ok:
                raise SpatialCompilationError(
                    f"{requirement_id} {requirement.subject!r}: expected "
                    f"{requirement.quantity.mode} {expected} constructed levels, "
                    f"observed {observed}"
                )
            actual_ids = tuple(sorted(actual_levels))
        elif requirement.kind == "repeated_zone" and "space" in bound_classes:
            mapped_space_ids = tuple(
                actual_space_id(space_id) for space_id in binding.artifact_ids
            )
            missing = set(mapped_space_ids) - set(actual_spaces)
            if missing:
                raise SpatialCompilationError(
                    f"{requirement_id} {requirement.subject!r}: bound structural "
                    f"zones were not constructed: {sorted(missing)}"
                )
            actual_ids = mapped_space_ids
        elif requirement.kind == "connector":
            expected_connectors = [
                connector
                for connector in blueprint.connectors
                if connector.connector_id in binding.artifact_ids
            ]
            matched: list[str] = []
            for expected_connector in expected_connectors:
                expected_stops = [
                    (
                        actual_space_id(expected_connector.start.space_id),
                        expected_connector.start.level_id,
                    ),
                    *[
                        (
                            actual_space_id(str(landing.get("space_id", ""))),
                            str(landing.get("level_id", "")),
                        )
                        for landing in expected_connector.parameters.get(
                            "intermediate_landings", []
                        )
                        if isinstance(landing, dict)
                    ],
                    (
                        actual_space_id(expected_connector.end.space_id),
                        expected_connector.end.level_id,
                    ),
                ]
                expected_stops.sort(key=lambda stop: blueprint_level_order[stop[1]])
                for expected_endpoints in (
                    {start, end}
                    for start, end in zip(expected_stops, expected_stops[1:])
                ):
                    match = next(
                        (
                            connector
                            for connector in actual_connectors.values()
                            if connector.connector_id not in used_connectors
                            and getattr(
                                connector.connector_type,
                                "value",
                                connector.connector_type,
                            )
                            == expected_connector.kind
                            and connector.width + 1e-9 >= expected_connector.width_m
                            and {
                                (
                                    connector.start.space_id,
                                    connector.start.level_id,
                                ),
                                (
                                    connector.end.space_id,
                                    connector.end.level_id,
                                ),
                            }
                            == expected_endpoints
                        ),
                        None,
                    )
                    if match is None:
                        raise SpatialCompilationError(
                            f"{requirement_id} {requirement.subject!r}: missing "
                            f"constructed {expected_connector.kind} connector with "
                            f"width >= {expected_connector.width_m:g}m"
                        )
                    used_connectors.add(match.connector_id)
                    matched.append(match.connector_id)
            actual_ids = tuple(matched)
            semantic_observed_count = len(expected_connectors)
        elif requirement.kind == "opening":
            expected_openings = sorted(
                (
                    opening
                    for opening in blueprint.openings
                    if opening.opening_id in binding.artifact_ids
                ),
                key=lambda item: item.width_m * item.height_m,
                reverse=True,
            )
            matched = []
            kind_aliases = {
                "door": {"door"},
                "window": {"window"},
                "open_connection": {"open", "open_connection"},
            }
            for expected_opening in expected_openings:
                minimum_width = expected_opening.width_m
                minimum_height = expected_opening.height_m
                if (
                    requirement.scale is not None
                    and requirement.scale.minimum_dimensions_m
                ):
                    minimum_width = max(
                        minimum_width,
                        requirement.scale.minimum_dimensions_m[0],
                    )
                    minimum_height = max(
                        minimum_height,
                        requirement.scale.minimum_dimensions_m[2],
                    )
                match = next(
                    (
                        (opening_id, opening_records)
                        for opening_id, opening_records in sorted(
                            actual_openings.items(),
                            key=lambda item: max(
                                record[1].width * record[1].height for record in item[1]
                            ),
                            reverse=True,
                        )
                        if opening_id not in used_openings
                        and any(
                            getattr(
                                opening.opening_type,
                                "value",
                                opening.opening_type,
                            )
                            in kind_aliases[expected_opening.kind]
                            and opening.width + 1e-9 >= minimum_width
                            and opening.height + 1e-9 >= minimum_height
                            for _, opening in opening_records
                        )
                        and actual_space_id(expected_opening.host_space_id)
                        in {room_id for room_id, _ in opening_records}
                        and (
                            expected_opening.connects_to_space_id is None
                            or actual_space_id(expected_opening.connects_to_space_id)
                            in {room_id for room_id, _ in opening_records}
                        )
                    ),
                    None,
                )
                if match is None:
                    raise SpatialCompilationError(
                        f"{requirement_id} {requirement.subject!r}: missing "
                        f"constructed {expected_opening.kind} opening >= "
                        f"{minimum_width:g}m × {minimum_height:g}m"
                    )
                opening_id, _ = match
                used_openings.add(opening_id)
                matched.append(opening_id)
            actual_ids = tuple(matched)
        else:
            continue
        evidence.append(
            TopologyRequirementEvidence(
                requirement_id=requirement_id,
                actual_artifact_ids=actual_ids,
                observed_count=(
                    semantic_observed_count
                    if semantic_observed_count is not None
                    else len(actual_ids)
                ),
                diagnostic=(f"Constructed topology preserved {requirement.subject!r}."),
            )
        )
    return TopologyStageManifest(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        evidence=tuple(evidence),
    )


def persist_spatial_compilation(
    compilation: SpatialRequirementCompilation, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(compilation.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_spatial_compilation(path: Path) -> SpatialRequirementCompilation:
    return SpatialRequirementCompilation.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def spatial_compilation_output_schema() -> AgentOutputSchema:
    """Return the strict compact wire schema used by the spatial model."""

    return AgentOutputSchema(SpatialRequirementCompilationWire)


def persist_topology_manifest(
    manifest: TopologyStageManifest, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(manifest.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


async def compile_requirement_blueprint(
    graph: SceneRequirementGraph,
    manifest: SemanticCapabilityManifest,
    *,
    model: str,
    mode: Literal["room", "house"],
    maximum_dimension_m: float | None = None,
    maximum_height_m: float | None = None,
    maximum_opening_width_m: float | None = None,
    maximum_opening_height_m: float | None = None,
    run_config: RunConfig | None = None,
    model_settings: ModelSettings | None = None,
    runner: type[_Runner] = BoundedRunner,
) -> tuple[SpatialRequirementCompilation, Any]:
    compiler = Agent(
        name="Scene Spatial and Topological Compiler",
        model=model,
        instructions=SPATIAL_COMPILER_INSTRUCTIONS,
        output_type=spatial_compilation_output_schema(),
        model_settings=model_settings or ModelSettings(),
    )
    result = await runner.run(
        starting_agent=compiler,
        input=spatial_compilation_input(
            graph,
            manifest,
            mode=mode,
            maximum_dimension_m=maximum_dimension_m,
            maximum_height_m=maximum_height_m,
            maximum_opening_width_m=maximum_opening_width_m,
            maximum_opening_height_m=maximum_opening_height_m,
        ),
        max_turns=1,
        run_config=run_config,
        timeout_seconds=agent_run_timeout_seconds("spatial_compiler", max_turns=1),
    )
    wire = result.final_output_as(SpatialRequirementCompilationWire)
    compilation = expand_spatial_compilation(
        graph,
        wire,
        mode=mode,
        maximum_dimension_m=maximum_dimension_m,
        maximum_height_m=maximum_height_m,
        maximum_opening_width_m=maximum_opening_width_m,
        maximum_opening_height_m=maximum_opening_height_m,
    )
    validate_spatial_compilation(
        compilation,
        graph,
        maximum_dimension_m=maximum_dimension_m,
        maximum_height_m=maximum_height_m,
        expected_mode=mode,
    )
    return compilation, result
