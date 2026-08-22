"""Tests for generic semantic fulfillment capability preflight."""

import asyncio

from datetime import UTC, datetime

import pytest

from scenesmith.agent_utils.semantics.publication.artifact_inventory import (
    semantic_verification_input,
)
from scenesmith.agent_utils.semantics.publication.publication_models import (
    RelationVerification,
    RequirementVerificationClaim,
    SemanticArtifact,
    SemanticPublicationError,
    SemanticVerificationBatch,
)
from scenesmith.agent_utils.semantics.publication.semantic_publication import (
    analyze_final_semantics,
    certify_semantic_publication,
)
from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    RequirementBlueprintBinding,
    SpatialRequirementCompilation,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.literal_parsing import (
    literal_candidates_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    CompositionPlan,
    RequirementInterpretationBatch,
    RequirementInterpretationProposal,
    RequirementQuantity,
    RequirementRelation,
    RequirementScale,
    SceneCompositionOpinion,
    TopologyOpinion,
    VerificationPolicy,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    BlueprintConstraint,
    FurnitureGroupBlueprint,
    LevelBlueprint,
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


def test_semantic_verifier_input_is_compact_and_preserves_binding_evidence():
    graph = _graph()
    compilation = _spatial_compilation(graph)
    artifacts = (
        SemanticArtifact(
            artifact_id="unrelated-statue",
            artifact_class="scene_object",
            name="Classical statue",
            description="A figure holding a miniature velorian assembly.",
        ),
    ) + tuple(
        SemanticArtifact(
            artifact_id=f"velorian-{index}",
            artifact_class="scene_object",
            name="Velorian assembly",
            description="purpose-built semantic object " * 20,
            metadata={
                "catalog_semantics": "Velorian ontology " * 30,
                "irrelevant_debug_blob": "x" * 2000,
            },
        )
        for index in range(100)
    )

    payload = semantic_verification_input(graph, compilation, artifacts)

    assert graph.graph_id in payload
    assert graph.requirements[0].requirement_id in payload
    assert "velorian-99" in payload
    assert "irrelevant_debug_blob" not in payload
    assert "candidate_sources" not in payload
    assert len(payload) < 100_000


def test_final_semantics_binds_unambiguous_surviving_objects_without_model_turn():
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

    class RunnerThatMustNotRun:
        @staticmethod
        async def run(*_args, **_kwargs):
            raise AssertionError("unambiguous evidence should not spend a model turn")

    verification, results = asyncio.run(
        analyze_final_semantics(
            graph,
            compilation,
            artifacts,
            model="fixture-model",
            runner=RunnerThatMustNotRun,
        )
    )

    assert results == ()
    assert verification.claims[0].status == "satisfied"
    assert verification.claims[0].observed_count == 10
    assert "unrelated-statue" not in verification.claims[0].artifact_ids


def test_final_semantics_aggregates_model_batch_audit_summaries():
    graph = _graph()
    compilation = _spatial_compilation(graph)
    batch = _verification(graph, (), observed_count=0, status="missing")

    class Result:
        @staticmethod
        def final_output_as(_output_type):
            return batch

    class Runner:
        @staticmethod
        async def run(*_args, **_kwargs):
            return Result()

    verification, results = asyncio.run(
        analyze_final_semantics(
            graph,
            compilation,
            (),
            model="fixture-model",
            runner=Runner,
        )
    )

    assert len(results) == 1
    assert verification.claims == batch.claims
    assert verification.audit_summary == (
        "Deterministically bound 0 source requirements.; "
        "Source-bound final evidence audit."
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
