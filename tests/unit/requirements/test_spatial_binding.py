"""Tests for generic semantic fulfillment capability preflight."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scenesmith.agent_utils.scene.house_parts.openings import OpeningType
from scenesmith.agent_utils.semantics.requirements.compilation.expansion import (
    blueprint_with_obligation_brief,
    expand_spatial_compilation,
)
from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    BlueprintDesignTokensWire,
    FurnitureGroupBlueprintWire,
    RequirementBlueprintBinding,
    RequirementBlueprintBindingWire,
    RoleCountWire,
    SpatialCompilationError,
    SpatialRequirementCompilation,
    SpatialRequirementCompilationWire,
)
from scenesmith.agent_utils.semantics.requirements.requirement_blueprint_compiler import (
    load_spatial_compilation,
    persist_spatial_compilation,
    validate_constructed_topology,
    validate_spatial_compilation,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.literal_parsing import (
    literal_candidates_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    CompositionPlan,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementQuantity,
    RequirementScale,
    SceneCompositionOpinion,
    TopologyOpinion,
    VerificationPolicy,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    BlueprintConstraint,
    ConnectorBlueprint,
    ConnectorEndpoint,
    FurnitureGroupBlueprint,
    LevelBlueprint,
    OpeningBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
)
from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    merge_requirement_interpretations,
)
from scenesmith.agent_utils.semantics.requirements.semantic_strategies import (
    SemanticCapabilityProfile,
)

FIXED_TIME = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _clock():
    return FIXED_TIME


def _graph(
    *,
    kind="hero_object",
    strategy_order=("catalog", "composed", "procedural"),
    prompt="Exactly 10 velorian assemblies dominate the chamber.",
    scale=None,
    relations=(),
):
    candidates = literal_candidates_from_prompt(prompt)
    assert len(candidates) == 1
    candidate = candidates[0]
    quantity = candidate.explicit_quantities[0]
    proposal = RequirementInterpretationProposal(
        candidate_id=candidate.candidate_id,
        subject="velorian assembly",
        kind=kind,
        source_quantity_id=quantity.quantity_id,
        quantity=RequirementQuantity(
            mode=quantity.mode,
            value=quantity.value,
            source_quantity_id=quantity.quantity_id,
        ),
        scale=scale,
        relations=relations,
        topology=TopologyOpinion(
            role="dominant repeated artifact",
            enclosure="inside the primary chamber",
            circulation="preserve circulation between instances",
            rationale="Judged from the complete prompt.",
        ),
        composition=CompositionPlan(
            recommended_strategy=strategy_order[0],
            strategy_order=strategy_order,
            reusable_parts=("frame", "surface", "functional core"),
            procedural_geometry="Generate a measured envelope and visible parts.",
            arrangement="Build ten legible instances around a shared work volume.",
            rationale="The structured semantic model selected this strategy order.",
        ),
        verification=VerificationPolicy(
            stage="semantic",
            method="count_and_measure",
            measurable_criteria=("Exactly ten distinct instances are present",),
        ),
        interpretation_rationale="The LLM classified an arbitrary domain concept.",
    )
    return merge_requirement_interpretations(
        prompt,
        candidates,
        RequirementInterpretationBatch(
            composition=SceneCompositionOpinion(
                scene_type="arbitrary test chamber",
                overall_scale="large enough for ten dominant assemblies",
                preferred_dimensions_m=(30.0, 20.0, 10.0),
                composition_summary="Ten repeated artifacts define the room.",
                topology_summary="One chamber with repeated perimeter positions.",
                circulation_summary="A central circulation loop remains clear.",
                density="dense but traversable",
                focal_hierarchy=("velorian assemblies",),
            ),
            requirements=(proposal,),
            analysis_summary="The arbitrary concept was interpreted without a noun table.",
        ),
        analysis_model="fixture-semantic-model",
    )


def _profile(**updates):
    values = {
        "catalog_available": True,
        "generated_geometry_available": True,
        "reusable_composition_available": True,
        "structural_compiler_available": True,
    }
    values.update(updates)
    return SemanticCapabilityProfile(**values)


def _spatial_compilation(graph, *, role_count=10, include_constraint=True):
    requirement = graph.requirements[0]
    expected_count = int(
        requirement.quantity.value or requirement.quantity.interpreted_minimum or 1
    )
    level = LevelBlueprint(
        level_id="level-primary",
        name="Primary",
        elevation_m=0.0,
        clear_height_m=8.0,
    )
    space = SpaceBlueprint(
        space_id="space-primary",
        name="Primary chamber",
        room_type="specialized chamber",
        level_id=level.level_id,
        dimensions_m=(20.0, 16.0),
        prompt=graph.source_prompt,
    )
    group = FurnitureGroupBlueprint(
        group_id="group-velorian",
        name="Velorian assemblies",
        space_id=space.space_id,
        roles={"velorian assembly": role_count},
        density="layered",
    )
    constraint = BlueprintConstraint(
        constraint_id="constraint-velorian",
        kind="semantic_obligation",
        target_ids=(group.group_id,),
        parameters={
            "requirement_id": requirement.requirement_id,
            "planned_instances": expected_count,
            "verification_criteria": ["exactly ten instances"],
            **(
                {"minimum_dimensions_m": list(requirement.scale.minimum_dimensions_m)}
                if requirement.scale is not None
                and requirement.scale.minimum_dimensions_m is not None
                else {}
            ),
            **(
                {
                    "relationships": [
                        relation.model_dump(mode="json")
                        for relation in requirement.relations
                    ]
                }
                if requirement.relations
                else {}
            ),
        },
        strength="hard",
        source="user",
    )
    blueprint = SceneBlueprint(
        blueprint_id="scene-velorian",
        source_prompt=graph.source_prompt,
        levels=(level,),
        spaces=(space,),
        furniture_groups=(group,),
        constraints=((constraint,) if include_constraint else ()),
        locked_ids=(group.group_id,),
    )
    return SpatialRequirementCompilation(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        blueprint=blueprint,
        bindings=(
            RequirementBlueprintBinding(
                requirement_id=requirement.requirement_id,
                owner_stage="asset",
                artifact_ids=(group.group_id,),
                role_key="velorian assembly",
                planned_instances=expected_count,
                rationale="The LLM compiled the exact repeated artifact count.",
            ),
        ),
        compilation_summary="A source-bound arbitrary-domain plan.",
    )


def test_spatial_compiler_binding_preserves_exact_count_and_locked_artifact():
    graph = _graph()
    compilation = _spatial_compilation(graph)

    validate_spatial_compilation(
        compilation,
        graph,
        maximum_dimension_m=24.0,
        maximum_height_m=12.0,
    )


def test_spatial_compiler_rejects_count_loss_and_missing_hard_constraint():
    graph = _graph()
    with pytest.raises(SpatialCompilationError, match="expected 10"):
        validate_spatial_compilation(
            _spatial_compilation(graph, role_count=9),
            graph,
            maximum_dimension_m=24.0,
            maximum_height_m=12.0,
        )
    with pytest.raises(SpatialCompilationError, match="no hard blueprint constraint"):
        validate_spatial_compilation(
            _spatial_compilation(graph, include_constraint=False),
            graph,
            maximum_dimension_m=24.0,
            maximum_height_m=12.0,
        )


def test_spatial_compilation_persists_and_obligation_brief_reaches_room_prompt(
    tmp_path,
):
    graph = _graph()
    compilation = _spatial_compilation(graph)
    path = tmp_path / "semantic_spatial_compilation.json"

    persist_spatial_compilation(compilation, path)
    blueprint = blueprint_with_obligation_brief(compilation, graph)

    assert load_spatial_compilation(path) == compilation
    assert graph.requirements[0].requirement_id in blueprint.spaces[0].prompt
    assert '"planned_instances":10' in blueprint.spaces[0].prompt


def _opening_compilation(graph):
    base = _spatial_compilation(graph)
    openings = tuple(
        OpeningBlueprint(
            opening_id=f"opening-{index}",
            kind="open_connection",
            host_space_id=base.blueprint.spaces[0].space_id,
            width_m=2.0,
            height_m=3.0,
        )
        for index in range(10)
    )
    blueprint = base.blueprint.model_copy(
        update={
            "openings": openings,
            "locked_ids": tuple(item.opening_id for item in openings),
        }
    )
    binding = base.bindings[0].model_copy(
        update={
            "owner_stage": "topology",
            "artifact_ids": tuple(item.opening_id for item in openings),
            "role_key": None,
        }
    )
    return base.model_copy(update={"blueprint": blueprint, "bindings": (binding,)})


def test_topology_gate_rejects_nine_of_ten_constructed_openings():
    graph = _graph(kind="opening")
    compilation = _opening_compilation(graph)

    def layout(opening_count, opening_type=OpeningType.OPEN):
        openings = [
            SimpleNamespace(
                opening_id=f"actual-opening-{index}",
                opening_type=opening_type,
                width=2.0,
                height=3.0,
            )
            for index in range(opening_count)
        ]
        return SimpleNamespace(
            levels=[],
            room_specs=[SimpleNamespace(room_id="space-primary")],
            connectors=[],
            room_geometries={"space-primary": SimpleNamespace(openings=openings)},
        )

    with pytest.raises(SpatialCompilationError, match="missing constructed"):
        validate_constructed_topology(compilation, graph, layout(9))
    with pytest.raises(SpatialCompilationError, match="missing constructed"):
        validate_constructed_topology(
            compilation,
            graph,
            layout(10, OpeningType.DOOR),
        )
    manifest = validate_constructed_topology(compilation, graph, layout(10))
    assert manifest.passed
    assert manifest.evidence[0].observed_count == 10


def test_topology_gate_rejects_ordinary_opening_for_llm_authored_massive_scale():
    graph = _graph(
        kind="opening",
        scale=RequirementScale(
            qualitative_label="massive",
            minimum_dimensions_m=(6.0, 0.2, 5.0),
            rationale="The LLM judged the opening must admit the dominant vehicle.",
        ),
    )
    compilation = _opening_compilation(graph)

    def layout(width, height):
        openings = [
            SimpleNamespace(
                opening_id=f"actual-opening-{index}",
                opening_type=OpeningType.OPEN,
                width=width,
                height=height,
            )
            for index in range(10)
        ]
        return SimpleNamespace(
            levels=[],
            room_specs=[SimpleNamespace(room_id="space-primary")],
            connectors=[],
            room_geometries={"space-primary": SimpleNamespace(openings=openings)},
        )

    with pytest.raises(
        SpatialCompilationError,
        match=r"missing constructed open_connection opening >= 6m × 5m",
    ):
        validate_constructed_topology(compilation, graph, layout(1.0, 2.1))
    assert validate_constructed_topology(
        compilation,
        graph,
        layout(6.0, 5.0),
    ).passed


def test_topology_gate_maps_room_mode_blueprint_spaces_to_constructed_room():
    graph = _graph(
        kind="connector",
        prompt="Exactly 1 spiral staircase connects the three library levels.",
    )
    requirement = graph.requirements[0]
    levels = (
        LevelBlueprint(level_id="ground", name="Ground", elevation_m=0.0),
        LevelBlueprint(level_id="mezzanine", name="Mezzanine", elevation_m=4.0),
        LevelBlueprint(level_id="upper", name="Upper", elevation_m=8.0),
    )
    spaces = (
        SpaceBlueprint(
            space_id="semantic-ground",
            name="Ground library",
            room_type="library",
            level_id="ground",
            dimensions_m=(16.0, 14.0),
        ),
        SpaceBlueprint(
            space_id="semantic-mezzanine",
            name="Mezzanine library",
            room_type="library",
            level_id="mezzanine",
            dimensions_m=(16.0, 14.0),
        ),
        SpaceBlueprint(
            space_id="semantic-upper",
            name="Upper library",
            room_type="library",
            level_id="upper",
            dimensions_m=(16.0, 14.0),
        ),
    )
    connector = ConnectorBlueprint(
        connector_id="semantic-spiral",
        kind="stairs_spiral",
        start=ConnectorEndpoint(
            space_id=spaces[0].space_id,
            level_id=levels[0].level_id,
            position_m=(3.0, 3.0, 0.0),
        ),
        end=ConnectorEndpoint(
            space_id=spaces[2].space_id,
            level_id=levels[2].level_id,
            position_m=(3.0, 3.0, 8.0),
        ),
        width_m=3.2,
        parameters={
            "intermediate_landings": [
                {
                    "space_id": spaces[1].space_id,
                    "level_id": levels[1].level_id,
                    "position_m": [3.0, 3.0, 4.0],
                }
            ]
        },
    )
    constraint = BlueprintConstraint(
        constraint_id="constraint-spiral",
        kind="semantic_connector",
        target_ids=(connector.connector_id,),
        parameters={
            "requirement_id": requirement.requirement_id,
            "planned_instances": 1,
            "verification_criteria": ["One usable spiral connector"],
        },
        strength="hard",
        source="user",
    )
    blueprint = SceneBlueprint(
        blueprint_id="semantic-library",
        source_prompt=graph.source_prompt,
        mode="room",
        levels=levels,
        spaces=spaces,
        connectors=(connector,),
        constraints=(constraint,),
        locked_ids=(connector.connector_id, constraint.constraint_id),
    )
    compilation = SpatialRequirementCompilation(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        blueprint=blueprint,
        bindings=(
            RequirementBlueprintBinding(
                requirement_id=requirement.requirement_id,
                owner_stage="topology",
                artifact_ids=(connector.connector_id,),
                role_key=None,
                planned_instances=1,
                rationale="The spiral staircase binds the two semantic levels.",
            ),
        ),
        compilation_summary="A three-level room-mode library.",
    )
    actual_connectors = (
        SimpleNamespace(
            connector_id=connector.connector_id,
            connector_type="stairs_spiral",
            width=3.2,
            start=SimpleNamespace(space_id="library", level_id="ground"),
            end=SimpleNamespace(space_id="library", level_id="mezzanine"),
        ),
        SimpleNamespace(
            connector_id=f"{connector.connector_id}-segment-2",
            connector_type="stairs_spiral",
            width=3.2,
            start=SimpleNamespace(space_id="library", level_id="mezzanine"),
            end=SimpleNamespace(space_id="library", level_id="upper"),
        ),
    )
    layout = SimpleNamespace(
        levels=[SimpleNamespace(level_id=level.level_id) for level in levels],
        room_specs=[SimpleNamespace(room_id="library", room_type="library")],
        connectors=list(actual_connectors),
        room_geometries={"library": SimpleNamespace(openings=[])},
    )

    manifest = validate_constructed_topology(compilation, graph, layout)

    assert manifest.passed
    assert manifest.evidence[0].actual_artifact_ids == tuple(
        item.connector_id for item in actual_connectors
    )
    assert manifest.evidence[0].observed_count == 1


def test_compact_spatial_wire_expands_constraints_counts_and_locks():
    graph = _graph()
    requirement = graph.requirements[0]
    wire = SpatialRequirementCompilationWire(
        blueprint_id="assembly-hall",
        levels=(LevelBlueprint(level_id="ground", name="Ground"),),
        spaces=(
            SpaceBlueprint(
                space_id="main-hall",
                name="Main hall",
                room_type="assembly_hall",
                level_id="ground",
                dimensions_m=(30, 30),
            ),
        ),
        openings=(
            OpeningBlueprint(
                opening_id="oversized-window",
                kind="window",
                host_space_id="main-hall",
                width_m=8.0,
                height_m=9.0,
            ),
        ),
        connectors=(),
        furniture_groups=(
            FurnitureGroupBlueprintWire(
                group_id="assemblies",
                name="Velorian assemblies",
                space_id="main-hall",
                roles=(RoleCountWire(role="assembly", count=10),),
                focal_target=None,
                density="balanced",
            ),
        ),
        design_tokens=BlueprintDesignTokensWire(
            style_keywords=(),
            palette=(),
            material_roles=(),
            lighting_mood="neutral",
            focal_hierarchy=("assemblies",),
        ),
        bindings=(
            RequirementBlueprintBindingWire(
                requirement_id=requirement.requirement_id,
                owner_stage="placement",
                artifact_ids=("assemblies",),
                role_key="assembly",
            ),
        ),
        compilation_summary="Bound the exact assembly requirement.",
    )

    unbounded = expand_spatial_compilation(graph, wire, mode="room")
    assert unbounded.blueprint.spaces[0].dimensions_m == (30, 30)
    assert unbounded.blueprint.openings[0].width_m == 8
    assert unbounded.blueprint.openings[0].height_m == 9
    validate_spatial_compilation(unbounded, graph, expected_mode="room")

    compilation = expand_spatial_compilation(
        graph,
        wire,
        mode="room",
        maximum_dimension_m=20,
        maximum_height_m=12,
        maximum_opening_width_m=4,
        maximum_opening_height_m=4,
    )

    assert compilation.bindings[0].planned_instances == 10
    assert compilation.blueprint.spaces[0].dimensions_m == (20, 20)
    assert compilation.blueprint.openings[0].width_m == 4
    assert compilation.blueprint.openings[0].height_m == 4
    assert compilation.blueprint.constraints[0].parameters["planned_instances"] == 10
    assert set(compilation.bindings[0].artifact_ids) <= set(
        compilation.blueprint.locked_ids
    )
    validate_spatial_compilation(
        compilation,
        graph,
        maximum_dimension_m=20,
        maximum_height_m=12,
        expected_mode="room",
    )
