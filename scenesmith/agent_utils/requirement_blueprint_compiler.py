"""LLM-authored spatial compilation of immutable semantic obligations."""

from __future__ import annotations

import json
import os
import tempfile

from pathlib import Path
from typing import Any, Literal, Protocol

from agents import Agent, ModelSettings, RunConfig
from pydantic import BaseModel, ConfigDict

from scenesmith.agent_utils.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.scene_blueprint import SceneBlueprint
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


class SpatialCompilationError(ValueError):
    """The spatial compilation omitted or weakened an immutable obligation."""


class _Runner(Protocol):
    @staticmethod
    async def run(*args: Any, **kwargs: Any) -> Any: ...


SPATIAL_COMPILER_INSTRUCTIONS = """
You are SceneSmith's spatial and topological compiler. The input requirement graph
is immutable source truth; you may design an implementation but may not omit,
rename, weaken, merge away, or reduce any blocking obligation.

Return one provider-neutral SceneBlueprint and a binding for every hard requirement.
Use the LLM-authored metric scale, topology, relations, composition plan, and scene
composition opinion. Do not substitute a generic room for an unfamiliar concept.

Rules:
1. Copy graph_id, graph_hash, and source_prompt exactly.
2. Every blocking required obligation must bind to concrete blueprint artifact IDs.
   Scene types bind the blueprint ID; levels bind level IDs; openings bind opening
   IDs; connectors bind connector IDs; spatial constraints bind constraint IDs;
   object and repeated-zone obligations bind furniture groups and a precise role_key
   when implemented inside one shared volume.
3. Add a hard BlueprintConstraint for every blocking required obligation. Its
   parameters must contain requirement_id, planned_instances, verification criteria,
   and any metric dimensions/clearance or spatial relationships from the graph.
4. Exact counts stay exact. Minimum and LLM-interpreted qualitative minima may be
   exceeded only when the composition needs it. A furniture binding's planned count
   must equal the sum of its bound group role counts.
5. Forbidden obligations bind to owner_stage=semantic, artifact_ids=[], and
   planned_instances=0. They are absence checks, never construction requests.
6. Prefer one coherent primary volume for an interior set. Repeated operational
   zones inside it may be explicit furniture groups backed by hard topology
   constraints; do not invent disconnected rooms merely to satisfy a count.
7. Keep every planned dimension within the supplied compiler limits. If the limits
   cannot meet a hard minimum, do not shrink the requirement; return the hard minimum
   so deterministic validation can reject the capability mismatch specifically.
8. Produce IDs that are stable, readable, globally unique, and include all bound IDs
   in SceneBlueprint.locked_ids so later candidate selection cannot discard them.
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


def validate_spatial_compilation(
    compilation: SpatialRequirementCompilation,
    graph: SceneRequirementGraph,
    *,
    maximum_dimension_m: float,
    maximum_height_m: float,
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
    if expected_mode == "room" and len(blueprint.spaces) != 1:
        raise SpatialCompilationError(
            "room-mode compilation must use one coherent primary space; repeated "
            "internal zones belong in bound furniture groups and constraints"
        )
    if blueprint.source_prompt != graph.source_prompt:
        raise SpatialCompilationError("spatial compiler changed the source prompt")
    if any(max(space.dimensions_m) > maximum_dimension_m for space in blueprint.spaces):
        raise SpatialCompilationError(
            f"compiled plan exceeds configured {maximum_dimension_m:g}m floor dimension"
        )
    total_height = max(
        level.elevation_m + level.clear_height_m for level in blueprint.levels
    )
    if total_height > maximum_height_m:
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
            if binding.role_key is None or role_count != expected_count:
                raise SpatialCompilationError(
                    f"{requirement.requirement_id} furniture role count is "
                    f"{role_count!r}; expected {expected_count}"
                )
        elif (
            requirement.quantity.mode == "exact"
            and len(binding.artifact_ids) != expected_count
        ):
            raise SpatialCompilationError(
                f"{requirement.requirement_id} binds {len(binding.artifact_ids)} artifacts; "
                f"expected exactly {expected_count}"
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
    maximum_dimension_m: float,
    maximum_height_m: float,
) -> str:
    return json.dumps(
        {
            "requirement_graph": graph.model_dump(mode="json"),
            "capability_manifest": manifest.model_dump(mode="json"),
            "compiler_limits": {
                "mode": mode,
                "maximum_floor_dimension_m": maximum_dimension_m,
                "maximum_total_height_m": maximum_height_m,
            },
        },
        indent=2,
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


async def compile_requirement_blueprint(
    graph: SceneRequirementGraph,
    manifest: SemanticCapabilityManifest,
    *,
    model: str,
    mode: Literal["room", "house"],
    maximum_dimension_m: float,
    maximum_height_m: float,
    run_config: RunConfig | None = None,
    model_settings: ModelSettings | None = None,
    runner: type[_Runner] = BoundedRunner,
) -> tuple[SpatialRequirementCompilation, Any]:
    compiler = Agent(
        name="Scene Spatial and Topological Compiler",
        model=model,
        instructions=SPATIAL_COMPILER_INSTRUCTIONS,
        output_type=SpatialRequirementCompilation,
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
        ),
        max_turns=1,
        run_config=run_config,
        timeout_seconds=agent_run_timeout_seconds("spatial_compiler", max_turns=1),
    )
    compilation = result.final_output_as(SpatialRequirementCompilation)
    validate_spatial_compilation(
        compilation,
        graph,
        maximum_dimension_m=maximum_dimension_m,
        maximum_height_m=maximum_height_m,
        expected_mode=mode,
    )
    return compilation, result
