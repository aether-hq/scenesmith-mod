"""LLM-authored spatial compilation of immutable semantic obligations."""

from __future__ import annotations

import os
import tempfile

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from agents import Agent, AgentOutputSchema, ModelSettings, RunConfig

from scenesmith.agent_utils.runtime.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.semantics.requirements.compilation.expansion import (
    expand_spatial_compilation,
    spatial_compilation_input,
)
from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    RequirementBlueprintBinding,
    SpatialCompilationError,
    SpatialRequirementCompilation,
    SpatialRequirementCompilationWire,
    TopologyRequirementEvidence,
    TopologyStageManifest,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirement,
    SceneRequirementGraph,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint
from scenesmith.agent_utils.semantics.requirements.semantic_strategies import (
    SemanticCapabilityManifest,
)

if TYPE_CHECKING:
    from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_models import (
        SceneEnrichmentDraft,
    )


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
9. The optional semantic scene enrichment is advisory operational and visual design
   context. Use it to make the topology and artifact plan domain-coherent, but never
   let it override or invent immutable graph requirements.
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
    scene_enrichment: "SceneEnrichmentDraft | None" = None,
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
            scene_enrichment=scene_enrichment,
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
