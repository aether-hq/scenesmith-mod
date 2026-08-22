"""Model-input serialization and deterministic spatial-compilation expansion."""

from __future__ import annotations

import json

from typing import TYPE_CHECKING, Any, Literal

from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    NamedStringWire,
    RequirementBlueprintBinding,
    RoleCountWire,
    SpatialCompilationError,
    SpatialRequirementCompilation,
    SpatialRequirementCompilationWire,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirement,
    SceneRequirementGraph,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    BlueprintConstraint,
    BlueprintDesignTokens,
    ConnectorBlueprint,
    ConnectorEndpoint,
    FurnitureGroupBlueprint,
    LevelBlueprint,
    SceneBlueprint,
)
from scenesmith.agent_utils.semantics.requirements.semantic_strategies import (
    SemanticCapabilityManifest,
)

if TYPE_CHECKING:
    from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_models import (
        SceneEnrichmentDraft,
    )


def _planned_instances(requirement: SceneRequirement) -> int:
    if requirement.polarity == "forbidden":
        return 0
    if requirement.quantity.mode in {"exact", "minimum"}:
        return int(requirement.quantity.value or 1)
    return int(requirement.quantity.interpreted_minimum or 1)


def spatial_compilation_input(
    graph: SceneRequirementGraph,
    manifest: SemanticCapabilityManifest,
    *,
    mode: Literal["room", "house"],
    scene_enrichment: "SceneEnrichmentDraft | None" = None,
    maximum_dimension_m: float | None = None,
    maximum_height_m: float | None = None,
    maximum_opening_width_m: float | None = None,
    maximum_opening_height_m: float | None = None,
) -> str:
    return json.dumps(
        {
            "requirement_graph": graph.model_dump(mode="json"),
            "capability_manifest": manifest.model_dump(mode="json"),
            "semantic_scene_enrichment": (
                scene_enrichment.model_dump(mode="json")
                if scene_enrichment is not None
                else None
            ),
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
