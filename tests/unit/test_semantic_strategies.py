"""Tests for generic semantic fulfillment capability preflight."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scenesmith.agent_utils.scene_requirements import (
    CompositionPlan,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementQuantity,
    RequirementRelation,
    RequirementScale,
    SceneCompositionOpinion,
    TopologyOpinion,
    VerificationPolicy,
    literal_candidates_from_prompt,
    merge_requirement_interpretations,
    requirement_graph_from_prompt,
)
from scenesmith.agent_utils.requirement_blueprint_compiler import (
    RequirementBlueprintBinding,
    SpatialCompilationError,
    SpatialRequirementCompilation,
    blueprint_with_obligation_brief,
    load_spatial_compilation,
    persist_spatial_compilation,
    validate_constructed_topology,
    validate_spatial_compilation,
)
from scenesmith.agent_utils.scene_blueprint import (
    BlueprintConstraint,
    FurnitureGroupBlueprint,
    LevelBlueprint,
    OpeningBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
)
from scenesmith.agent_utils.house import OpeningType
from scenesmith.agent_utils.semantic_ledger import initialize_semantic_ledger
from scenesmith.agent_utils.semantic_publication import (
    RelationVerification,
    RequirementVerificationClaim,
    SemanticArtifact,
    SemanticPublicationError,
    SemanticVerificationBatch,
    certify_semantic_publication,
)
from scenesmith.agent_utils.semantic_strategies import (
    CapabilityPreflightError,
    SemanticCapabilityProfile,
    apply_capability_manifest_to_ledger,
    assert_capability_preflight_passed,
    capability_preflight,
    capability_profile_from_config,
    initialize_strategy_journal,
    load_capability_manifest,
    load_strategy_journal,
    persist_capability_manifest,
    persist_strategy_journal,
    record_strategy_attempt,
    StrategyAttemptError,
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


def test_llm_order_selects_first_executable_strategy_and_preserves_exact_count():
    graph = _graph(strategy_order=("procedural", "catalog", "composed"))
    manifest = capability_preflight(graph, _profile())

    assert manifest.preflight_passed
    assert manifest.plans[0].selected_strategy == "procedural"
    assert manifest.plans[0].planned_instances == 10
    assert [item.strategy for item in manifest.plans[0].ordered_strategies] == [
        "procedural",
        "catalog",
        "composed",
    ]


def test_missing_catalog_falls_through_to_model_ranked_composition():
    manifest = capability_preflight(
        _graph(),
        _profile(catalog_available=False),
    )

    plan = manifest.plans[0]
    assert manifest.preflight_passed
    assert plan.ordered_strategies[0].status == "unavailable"
    assert plan.selected_strategy == "composed"
    assert plan.selected_provider == "reusable_assembly_compiler"


def test_structural_semantics_use_topology_compiler_without_room_kit_or_catalog():
    graph = _graph(kind="repeated_zone")
    manifest = capability_preflight(
        graph,
        _profile(
            catalog_available=False,
            generated_geometry_available=False,
            reusable_composition_available=False,
        ),
    )

    plan = manifest.plans[0]
    assert manifest.preflight_passed
    assert plan.selected_strategy == "composed"
    assert plan.selected_provider == "spatial_topology_compiler"
    assert plan.planned_instances == 10


def test_no_viable_provider_fails_with_requirement_specific_diagnostic():
    graph = _graph()
    manifest = capability_preflight(
        graph,
        _profile(
            catalog_available=False,
            generated_geometry_available=False,
            reusable_composition_available=False,
            structural_compiler_available=False,
        ),
    )

    assert not manifest.preflight_passed
    assert manifest.blocking_failures == (graph.requirements[0].requirement_id,)
    with pytest.raises(CapabilityPreflightError, match="velorian assembly"):
        assert_capability_preflight_passed(manifest)


def test_unclassified_hard_requirement_never_enters_construction():
    graph = requirement_graph_from_prompt("A qelthic engine dominates the vault.")
    manifest = capability_preflight(graph, _profile())

    assert not manifest.preflight_passed
    assert all(plan.selected_strategy is None for plan in manifest.plans)


def test_forbidden_requirement_uses_absence_guard_not_construction_provider():
    graph = _graph(prompt="No 10 velorian assemblies may enter the chamber.")
    manifest = capability_preflight(
        graph,
        _profile(
            catalog_available=False,
            generated_geometry_available=False,
            reusable_composition_available=False,
            structural_compiler_available=False,
        ),
    )

    assert manifest.preflight_passed
    assert manifest.plans[0].selected_strategy == "verification_guard"
    assert manifest.plans[0].selected_provider == "semantic_absence_verifier"


def test_manifest_advances_or_fails_append_only_ledger_idempotently():
    graph = _graph()
    ledger = initialize_semantic_ledger(graph, clock=_clock)
    manifest = capability_preflight(graph, _profile())
    advanced = apply_capability_manifest_to_ledger(
        ledger,
        graph,
        manifest,
        manifest_ref="artifact:capability-manifest",
        clock=_clock,
    )
    retry = apply_capability_manifest_to_ledger(
        advanced,
        graph,
        manifest,
        manifest_ref="artifact:capability-manifest",
        clock=_clock,
    )

    assert retry == advanced
    assert advanced.entries[0].current_status == "strategy_assigned"
    assert advanced.revision == 2


def test_profile_is_derived_from_provider_configuration_not_scene_vocabulary():
    profile = capability_profile_from_config(
        {
            "experiment": {"semantic_obligations": {}},
            "furniture_agent": {"asset_manager": {"general_asset_source": "hssd"}},
            "wall_agent": {"asset_manager": {"general_asset_source": "polyhaven"}},
        }
    )

    assert profile.catalog_available
    assert not profile.generated_geometry_available
    assert profile.reusable_composition_available
    assert profile.structural_compiler_available


def test_manifest_persistence_round_trip(tmp_path):
    manifest = capability_preflight(_graph(), _profile())
    path = tmp_path / "semantic_capability_manifest.json"

    persist_capability_manifest(manifest, path)

    assert load_capability_manifest(path) == manifest


def test_strategy_attempts_are_provider_checked_evidenced_and_idempotent(tmp_path):
    manifest = capability_preflight(_graph(), _profile())
    journal = initialize_strategy_journal(manifest)
    plan = manifest.plans[0]
    recorded = record_strategy_attempt(
        journal,
        manifest,
        attempt_key="construct:velorian:1",
        requirement_id=plan.requirement_id,
        strategy=plan.selected_strategy,
        provider_id=plan.selected_provider,
        stage="construction",
        outcome="succeeded",
        evidence_refs=("artifact:velorian-0",),
        clock=_clock,
    )
    retry = record_strategy_attempt(
        recorded,
        manifest,
        attempt_key="construct:velorian:1",
        requirement_id=plan.requirement_id,
        strategy=plan.selected_strategy,
        provider_id=plan.selected_provider,
        stage="construction",
        outcome="succeeded",
        evidence_refs=("artifact:velorian-0",),
        clock=_clock,
    )
    path = tmp_path / "semantic_strategy_journal.json"
    persist_strategy_journal(retry, path)

    assert retry == recorded
    assert load_strategy_journal(path) == recorded
    with pytest.raises(StrategyAttemptError, match="not available"):
        record_strategy_attempt(
            journal,
            manifest,
            attempt_key="bad-provider",
            requirement_id=plan.requirement_id,
            strategy="catalog",
            provider_id="missing-provider",
            stage="construction",
            outcome="failed",
            diagnostic="catalog did not contain the required artifact",
            clock=_clock,
        )


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


def _verification(graph, artifact_ids, *, observed_count=None, status="satisfied"):
    return SemanticVerificationBatch(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        claims=(
            RequirementVerificationClaim(
                requirement_id=graph.requirements[0].requirement_id,
                status=status,
                artifact_ids=tuple(artifact_ids),
                observed_count=(
                    len(artifact_ids) if observed_count is None else observed_count
                ),
                semantic_rationale="The final artifacts implement the arbitrary concept.",
            ),
        ),
        audit_summary="Source-bound final evidence audit.",
    )


def test_publication_certificate_requires_ten_distinct_surviving_artifacts():
    graph = _graph()
    compilation = _spatial_compilation(graph)
    artifacts = tuple(
        SemanticArtifact(
            artifact_id=f"velorian-{index}",
            artifact_class="scene_object",
            name=f"Velorian assembly {index}",
            dimensions_m=(2.0, 1.0, 1.5),
        )
        for index in range(10)
    )

    certificate = certify_semantic_publication(
        graph,
        compilation,
        artifacts,
        _verification(graph, [item.artifact_id for item in artifacts]),
        physics_verified=True,
        physics_evidence_refs=("physics:final-scene-zero-violations",),
    )

    assert certificate.publishable
    assert certificate.requirements[0].observed_count == 10


def test_publication_rejects_nine_artifacts_or_invented_count():
    graph = _graph()
    compilation = _spatial_compilation(graph)
    artifacts = tuple(
        SemanticArtifact(
            artifact_id=f"velorian-{index}",
            artifact_class="scene_object",
            name=f"Velorian assembly {index}",
        )
        for index in range(9)
    )

    with pytest.raises(SemanticPublicationError, match="expected exact 10"):
        certify_semantic_publication(
            graph,
            compilation,
            artifacts,
            _verification(graph, [item.artifact_id for item in artifacts]),
            physics_verified=True,
            physics_evidence_refs=("physics:clean",),
        )
    with pytest.raises(SemanticPublicationError, match="claimed count 10"):
        certify_semantic_publication(
            graph,
            compilation,
            artifacts,
            _verification(
                graph,
                [item.artifact_id for item in artifacts],
                observed_count=10,
            ),
            physics_verified=True,
            physics_evidence_refs=("physics:clean",),
        )


def test_publication_requires_separate_physics_evidence():
    graph = _graph()
    with pytest.raises(SemanticPublicationError, match="physics verification"):
        certify_semantic_publication(
            graph,
            _spatial_compilation(graph),
            (),
            _verification(graph, (), status="missing"),
            physics_verified=False,
            physics_evidence_refs=(),
        )


def test_publication_rejects_undersized_hero_and_unproven_relationship():
    graph = _graph(
        scale=RequirementScale(
            qualitative_label="huge",
            minimum_dimensions_m=(4.0, 2.0, 1.5),
            rationale="The LLM judged this minimum from scene context.",
        ),
        relations=(
            RequirementRelation(
                predicate="centered_in",
                target="primary chamber",
                rationale="The source makes it the central focal group.",
            ),
        ),
    )
    compilation = _spatial_compilation(graph)
    artifacts = tuple(
        SemanticArtifact(
            artifact_id=f"velorian-{index}",
            artifact_class="scene_object",
            name=f"Velorian assembly {index}",
            dimensions_m=(2.0, 1.0, 1.0),
        )
        for index in range(10)
    )
    with pytest.raises(SemanticPublicationError, match="minimum dimensions"):
        certify_semantic_publication(
            graph,
            compilation,
            artifacts,
            _verification(graph, [item.artifact_id for item in artifacts]),
            physics_verified=True,
            physics_evidence_refs=("physics:clean",),
        )
    full_sized = tuple(
        artifact.model_copy(update={"dimensions_m": (4.0, 2.0, 1.5)})
        for artifact in artifacts
    )
    with pytest.raises(SemanticPublicationError, match="unmet relationships"):
        certify_semantic_publication(
            graph,
            compilation,
            full_sized,
            _verification(graph, [item.artifact_id for item in full_sized]),
            physics_verified=True,
            physics_evidence_refs=("physics:clean",),
        )


def test_publication_rejects_generic_room_substituted_for_missing_hero():
    graph = _graph(
        prompt="Exactly 1 velorian fighter dominates the chamber.",
    )
    compilation = _spatial_compilation(graph, role_count=1)
    generic_room = SemanticArtifact(
        artifact_id="generic-room",
        artifact_class="space",
        name="generic room",
        dimensions_m=(30.0, 20.0, 10.0),
    )

    with pytest.raises(SemanticPublicationError, match="incompatible classes"):
        certify_semantic_publication(
            graph,
            compilation,
            (generic_room,),
            _verification(graph, (generic_room.artifact_id,)),
            physics_verified=True,
            physics_evidence_refs=("physics:clean",),
        )


def test_publication_requires_every_repeated_artifact_to_meet_scale_minimum():
    graph = _graph(
        scale=RequirementScale(
            qualitative_label="fighter-sized",
            minimum_dimensions_m=(3.0, 2.0, 2.0),
            rationale="Every repeated cell must contain the serviced artifact.",
        )
    )
    compilation = _spatial_compilation(graph)
    artifacts = tuple(
        SemanticArtifact(
            artifact_id=f"bay-{index}",
            artifact_class="scene_object",
            name=f"Bay {index}",
            dimensions_m=((3.0, 2.0, 2.0) if index < 9 else (1.0, 1.0, 1.0)),
        )
        for index in range(10)
    )

    with pytest.raises(SemanticPublicationError, match="minimum dimensions"):
        certify_semantic_publication(
            graph,
            compilation,
            artifacts,
            _verification(graph, tuple(item.artifact_id for item in artifacts)),
            physics_verified=True,
            physics_evidence_refs=("physics:clean",),
        )


def test_publication_rejects_relationship_claim_for_wrong_target():
    graph = _graph(
        prompt="Exactly 1 velorian fighter dominates the chamber.",
        relations=(
            RequirementRelation(
                predicate="centered_in",
                target="primary chamber",
                rationale="The fighter is the central focal object.",
            ),
        ),
    )
    compilation = _spatial_compilation(graph, role_count=1)
    fighter = SemanticArtifact(
        artifact_id="fighter",
        artifact_class="scene_object",
        name="Velorian fighter",
        dimensions_m=(8.0, 5.0, 2.5),
    )
    verification = _verification(graph, (fighter.artifact_id,)).model_copy(
        update={
            "claims": (
                _verification(graph, (fighter.artifact_id,))
                .claims[0]
                .model_copy(
                    update={
                        "relation_results": (
                            RelationVerification(
                                predicate="centered_in",
                                target="service corridor",
                                satisfied=True,
                                evidence_artifact_ids=(fighter.artifact_id,),
                                measurement="Centered in the wrong target.",
                            ),
                        )
                    }
                ),
            )
        }
    )

    with pytest.raises(SemanticPublicationError, match="primary chamber"):
        certify_semantic_publication(
            graph,
            compilation,
            (fighter,),
            verification,
            physics_verified=True,
            physics_evidence_refs=("physics:clean",),
        )
