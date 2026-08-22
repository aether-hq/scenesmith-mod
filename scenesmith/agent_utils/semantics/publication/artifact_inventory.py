"""Provider-neutral inventory serialization for final semantic verification."""

from __future__ import annotations

import json

from typing import Any

from scenesmith.agent_utils.semantics.publication.publication_models import (
    SemanticArtifact,
)
from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    SpatialRequirementCompilation,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirementGraph,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint


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
