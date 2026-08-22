import asyncio
import json
import logging

from pathlib import Path

from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.runtime.base_stateful_agent import log_agent_usage
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.agent_utils.semantics.environment.semantic_group_materializer import (
    load_and_materialize_locked_semantic_groups,
)
from scenesmith.agent_utils.semantics.publication.artifact_inventory import (
    semantic_artifact_inventory,
)
from scenesmith.agent_utils.semantics.publication.publication_models import (
    SemanticPublicationError,
)
from scenesmith.agent_utils.semantics.publication.semantic_publication import (
    analyze_final_semantics,
    certify_semantic_publication,
    persist_publication_artifact,
)
from scenesmith.agent_utils.semantics.requirements.requirement_blueprint_compiler import (
    load_spatial_compilation,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    semantic_model_name,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint
from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    audit_requirement_graph,
    load_requirement_graph,
    persist_shadow_audit,
)
from scenesmith.agent_utils.semantics.requirements.semantic_ledger import (
    load_or_initialize_semantic_ledger,
    persist_semantic_ledger,
    persist_semantic_ledger_summary,
    start_certified_retry_ledger,
    transition_requirement,
)
from scenesmith.agent_utils.semantics.requirements.semantic_strategies import (
    load_capability_manifest,
    load_strategy_journal,
    persist_strategy_journal,
    record_strategy_attempt,
)
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)

console_logger = logging.getLogger(__name__)

# Pipeline stages in execution order (derived from AgentType enum).
PIPELINE_STAGES = [agent.value for agent in AgentType]

# Stage dependencies for resume from checkpoint.
# Maps start_stage to the checkpoint it needs from the previous stage.
STAGE_CHECKPOINTS = {
    "floor_plan": None,
    "furniture": None,
    "wall_mounted": "scene_after_furniture",
    "ceiling_mounted": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
}

# Maps start_stage to the asset directories it needs from previous stages.
STAGE_ASSET_DIRS = {
    "floor_plan": [],
    "furniture": [],
    "wall_mounted": ["furniture"],
    "ceiling_mounted": ["furniture", "wall_mounted"],
    "manipuland": ["furniture", "wall_mounted", "ceiling_mounted"],
}

from scenesmith.experiments.indoor.runtime_support import (
    _require_semantic_publication_inputs,
)


def certify_room_publication(
    *,
    scene: RoomScene,
    room_dir: Path,
    house_layout: HouseLayout | None,
    cfg_dict: dict,
    manipuland_agent: StatefulManipulandAgent,
    final_physics_verified: bool,
    final_physics_evidence_refs: tuple[str, ...],
) -> None:
    """Verify and certify semantic obligations before scene publication."""

    requirement_graph_path = room_dir.parent / "scene_requirement_graph.json"
    scene_blueprint_path = room_dir.parent / "scene_blueprint.json"
    spatial_compilation_path = room_dir.parent / "semantic_spatial_compilation.json"
    capability_manifest_path = room_dir.parent / "semantic_capability_manifest.json"
    strategy_journal_path = room_dir.parent / "semantic_strategy_journal.json"
    if (
        requirement_graph_path.is_file()
        and scene_blueprint_path.is_file()
        and spatial_compilation_path.is_file()
        and capability_manifest_path.is_file()
        and strategy_journal_path.is_file()
    ):
        requirement_graph = load_requirement_graph(requirement_graph_path)
        scene_blueprint = SceneBlueprint.model_validate_json(
            scene_blueprint_path.read_text(encoding="utf-8")
        )
        spatial_compilation = load_spatial_compilation(spatial_compilation_path)
        artifacts = semantic_artifact_inventory(
            scene_blueprint,
            scene,
            house_layout=house_layout,
        )
        configured_model = semantic_model_name(
            str(cfg_dict["floor_plan_agent"].get("openai", {}).get("model") or "")
        )
        if not configured_model:
            raise RuntimeError(
                "Semantic publication verifier model is not configured; "
                "the scene cannot publish."
            )
        verification, verification_results = asyncio.run(
            analyze_final_semantics(
                requirement_graph,
                spatial_compilation,
                artifacts,
                model=configured_model,
                run_config=manipuland_agent._create_run_config(),
                model_settings=manipuland_agent._get_model_settings(
                    settings_key="designer"
                ),
            )
        )
        for batch_index, verification_result in enumerate(
            verification_results, start=1
        ):
            log_agent_usage(
                result=verification_result,
                agent_name=f"SEMANTIC PUBLICATION VERIFIER {batch_index}",
            )
        repair_requirement_ids = frozenset(
            claim.requirement_id
            for claim in verification.claims
            if claim.status in {"missing", "ambiguous"}
        )
        repaired_object_ids = load_and_materialize_locked_semantic_groups(
            scene,
            room_dir.parent,
            requirement_ids=repair_requirement_ids,
        )
        if repaired_object_ids:
            console_logger.warning(
                "Semantic verifier requested repair for %d obligations; "
                "materialized %d locked blueprint objects and rechecking physics",
                len(repair_requirement_ids),
                len(repaired_object_ids),
            )
            repair_collisions = compute_scene_collisions(scene)
            if repair_collisions:
                descriptions = "; ".join(
                    collision.to_description() for collision in repair_collisions[:10]
                )
                raise SemanticPublicationError(
                    "Automatic semantic repair produced physical conflicts: "
                    + descriptions,
                    failures=(descriptions,),
                )
            final_physics_verified = True
            final_physics_evidence_refs = tuple(final_physics_evidence_refs) + (
                "physics:automatic-semantic-repair-zero-collisions",
            )
            artifacts = semantic_artifact_inventory(
                scene_blueprint,
                scene,
                house_layout=house_layout,
            )
            verification, repair_verification_results = asyncio.run(
                analyze_final_semantics(
                    requirement_graph,
                    spatial_compilation,
                    artifacts,
                    model=configured_model,
                    run_config=manipuland_agent._create_run_config(),
                    model_settings=manipuland_agent._get_model_settings(
                        settings_key="designer"
                    ),
                )
            )
            for batch_index, verification_result in enumerate(
                repair_verification_results, start=1
            ):
                log_agent_usage(
                    result=verification_result,
                    agent_name=(f"SEMANTIC REPAIR VERIFIER {batch_index}"),
                )
            (room_dir / "semantic_repair.json").write_text(
                json.dumps(
                    {
                        "repair_requirement_ids": sorted(repair_requirement_ids),
                        "repaired_object_ids": [
                            str(object_id) for object_id in repaired_object_ids
                        ],
                        "repair_passes": 1,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            console_logger.info(
                "Automatic semantic repair completed; publication verification "
                "was rerun against the repaired final artifacts"
            )
        verification_path = room_dir / "semantic_verification.json"
        persist_publication_artifact(verification, verification_path)
        semantic_ledger = load_or_initialize_semantic_ledger(
            room_dir.parent / "semantic_obligation_ledger.json",
            requirement_graph,
        )
        capability_manifest = load_capability_manifest(capability_manifest_path)
        strategy_journal = load_strategy_journal(strategy_journal_path)
        try:
            certificate = certify_semantic_publication(
                requirement_graph,
                spatial_compilation,
                artifacts,
                verification,
                physics_verified=final_physics_verified,
                physics_evidence_refs=final_physics_evidence_refs,
            )
        except SemanticPublicationError as exc:
            failure_path = room_dir / "semantic_publication_failure.json"
            failure_path.write_text(
                json.dumps(
                    {
                        "graph_id": requirement_graph.graph_id,
                        "error": str(exc),
                        "failures": list(exc.failures),
                        "physics_verified": final_physics_verified,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            for requirement in requirement_graph.requirements:
                if requirement.strength != "hard" or requirement.enforcement not in {
                    "blocking",
                    "unresolved_blocking",
                }:
                    continue
                matching_failures = tuple(
                    failure
                    for failure in exc.failures
                    if requirement.requirement_id in failure
                )
                if exc.failures and not matching_failures:
                    continue
                diagnostic = (
                    "; ".join(matching_failures) if matching_failures else str(exc)
                )
                current_entry = next(
                    item
                    for item in semantic_ledger.entries
                    if item.requirement_id == requirement.requirement_id
                )
                if current_entry.current_status in {"fulfilled", "failed"}:
                    continue
                plan = next(
                    item
                    for item in capability_manifest.plans
                    if item.requirement_id == requirement.requirement_id
                )
                if (
                    plan.selected_strategy is not None
                    and plan.selected_provider is not None
                ):
                    strategy_journal = record_strategy_attempt(
                        strategy_journal,
                        capability_manifest,
                        attempt_key=f"final-failed:{requirement.requirement_id}",
                        requirement_id=requirement.requirement_id,
                        strategy=plan.selected_strategy,
                        provider_id=plan.selected_provider,
                        stage="construction_to_semantic",
                        outcome="failed",
                        diagnostic=diagnostic,
                    )
                semantic_ledger = transition_requirement(
                    semantic_ledger,
                    requirement.requirement_id,
                    "failed",
                    event_key=f"semantic-final:failed:{requirement.requirement_id}",
                    actor="semantic_publication_gate",
                    stage="semantic",
                    evidence_refs=(str(verification_path), str(failure_path)),
                    failure_reason=diagnostic,
                )
            persist_strategy_journal(strategy_journal, strategy_journal_path)
            persist_semantic_ledger(
                semantic_ledger,
                room_dir.parent / "semantic_obligation_ledger.json",
            )
            persist_semantic_ledger_summary(
                semantic_ledger,
                room_dir / "semantic_obligation_summary.json",
            )
            raise
        certificate_path = room_dir / "semantic_publication_certificate.json"
        persist_publication_artifact(certificate, certificate_path)

        retry_ledger = start_certified_retry_ledger(
            semantic_ledger,
            requirement_graph,
        )
        if retry_ledger is not semantic_ledger:
            console_logger.warning(
                "The resumed source run contained terminal semantic failures; "
                "starting a fresh run-local ledger attempt for the new verified "
                "publication certificate"
            )
            semantic_ledger = retry_ledger

        certified_by_id = {
            item.requirement_id: item for item in certificate.requirements
        }
        claims_by_id = {item.requirement_id: item for item in verification.claims}
        for requirement in requirement_graph.requirements:
            if requirement.strength != "hard":
                continue
            plan = next(
                item
                for item in capability_manifest.plans
                if item.requirement_id == requirement.requirement_id
            )
            current_entry = next(
                item
                for item in semantic_ledger.entries
                if item.requirement_id == requirement.requirement_id
            )
            if current_entry.current_status == "fulfilled":
                continue
            certified_requirement = certified_by_id.get(requirement.requirement_id)
            if certified_requirement is None:
                claim = claims_by_id.get(requirement.requirement_id)
                failure_reason = (
                    f"Advisory semantic verification did not pass: "
                    f"{claim.semantic_rationale if claim is not None else 'missing claim'}"
                )
                semantic_ledger = transition_requirement(
                    semantic_ledger,
                    requirement.requirement_id,
                    "failed",
                    event_key=f"semantic-final:failed:{requirement.requirement_id}",
                    actor="semantic_publication_gate",
                    stage="semantic",
                    evidence_refs=(str(verification_path),),
                    failure_reason=failure_reason,
                )
                continue
            evidence_refs = tuple(
                f"artifact:{artifact_id}"
                for artifact_id in certified_requirement.artifact_ids
            )
            if (
                plan.selected_strategy is not None
                and plan.selected_provider is not None
            ):
                strategy_journal = record_strategy_attempt(
                    strategy_journal,
                    capability_manifest,
                    attempt_key=f"final:{requirement.requirement_id}",
                    requirement_id=requirement.requirement_id,
                    strategy=plan.selected_strategy,
                    provider_id=plan.selected_provider,
                    stage="construction_to_semantic",
                    outcome="succeeded",
                    evidence_refs=evidence_refs or (str(verification_path),),
                )
            if current_entry.current_status == "extracted":
                semantic_ledger = transition_requirement(
                    semantic_ledger,
                    requirement.requirement_id,
                    "planned",
                    event_key=f"semantic-final:planned:{requirement.requirement_id}",
                    actor="semantic_publication_gate",
                    stage="planning",
                )
                current_entry = next(
                    item
                    for item in semantic_ledger.entries
                    if item.requirement_id == requirement.requirement_id
                )
            if current_entry.current_status == "planned":
                semantic_ledger = transition_requirement(
                    semantic_ledger,
                    requirement.requirement_id,
                    "strategy_assigned",
                    event_key=(
                        "semantic-final:strategy-assigned:"
                        f"{requirement.requirement_id}"
                    ),
                    actor="semantic_publication_gate",
                    stage="capability",
                )
                current_entry = next(
                    item
                    for item in semantic_ledger.entries
                    if item.requirement_id == requirement.requirement_id
                )
            if current_entry.current_status == "strategy_assigned":
                semantic_ledger = transition_requirement(
                    semantic_ledger,
                    requirement.requirement_id,
                    "constructed",
                    event_key=(
                        f"semantic-final:constructed:{requirement.requirement_id}"
                    ),
                    actor="semantic_publication_gate",
                    stage="construction",
                    evidence_refs=evidence_refs or (str(verification_path),),
                )
                current_entry = next(
                    item
                    for item in semantic_ledger.entries
                    if item.requirement_id == requirement.requirement_id
                )
            if current_entry.current_status != "verified":
                semantic_ledger = transition_requirement(
                    semantic_ledger,
                    requirement.requirement_id,
                    "verified",
                    event_key=f"semantic-final:verified:{requirement.requirement_id}",
                    actor="semantic_publication_gate",
                    stage="semantic",
                    evidence_refs=(str(verification_path),),
                )
            semantic_ledger = transition_requirement(
                semantic_ledger,
                requirement.requirement_id,
                "fulfilled",
                event_key=f"semantic-final:fulfilled:{requirement.requirement_id}",
                actor="semantic_publication_gate",
                stage="publication",
                evidence_refs=(str(certificate_path),),
            )
        persist_strategy_journal(strategy_journal, strategy_journal_path)
        persist_semantic_ledger(
            semantic_ledger,
            room_dir.parent / "semantic_obligation_ledger.json",
        )
        ledger_summary = persist_semantic_ledger_summary(
            semantic_ledger,
            room_dir / "semantic_obligation_summary.json",
        )
        if not ledger_summary.publishable or not ledger_summary.closed:
            raise RuntimeError(
                "Semantic publication certificate was created but the immutable "
                "obligation ledger did not close publishably."
            )
        console_logger.info(
            "Semantic publication gate passed: %d requirements certified, "
            "physics_verified=%s, ledger_revision=%d",
            len(certificate.requirements),
            certificate.physics_verified,
            ledger_summary.revision,
        )
    elif requirement_graph_path.is_file() and scene_blueprint_path.is_file():
        # Compatibility observation for legacy checkpoints created before the
        # enforced spatial compilation/certificate contract existed.
        try:
            requirement_graph = load_requirement_graph(requirement_graph_path)
            scene_blueprint = SceneBlueprint.model_validate_json(
                scene_blueprint_path.read_text(encoding="utf-8")
            )
            shadow_audit = audit_requirement_graph(
                requirement_graph,
                blueprint=scene_blueprint,
                scene=scene,
                house_layout=house_layout,
            )
            persist_shadow_audit(shadow_audit, room_dir / "semantic_shadow_audit.json")
            missing_subjects = [
                requirement.subject
                for requirement, result in zip(
                    requirement_graph.requirements,
                    shadow_audit.results,
                    strict=True,
                )
                if result.status == "missing"
            ]
            console_logger.warning(
                "Semantic shadow audit: satisfied=%d missing=%d ambiguous=%d; "
                "missing obligations=%s",
                shadow_audit.satisfied_count,
                shadow_audit.missing_count,
                shadow_audit.ambiguous_count,
                missing_subjects,
            )
        except Exception as exc:
            console_logger.warning(
                "Legacy semantic shadow audit could not be completed: %s",
                exc,
            )
    if requirement_graph_path.is_file() and not spatial_compilation_path.is_file():
        try:
            requirement_graph = load_requirement_graph(requirement_graph_path)
            semantic_ledger = load_or_initialize_semantic_ledger(
                room_dir.parent / "semantic_obligation_ledger.json",
                requirement_graph,
            )
            ledger_summary = persist_semantic_ledger_summary(
                semantic_ledger,
                room_dir / "semantic_obligation_summary.json",
            )
            console_logger.info(
                "Semantic ledger summary: revision=%d closed=%s publishable=%s "
                "blocking_failed=%d blocking_unresolved=%d",
                ledger_summary.revision,
                ledger_summary.closed,
                ledger_summary.publishable,
                len(ledger_summary.blocking_failed),
                len(ledger_summary.blocking_unresolved),
            )
        except Exception as exc:
            console_logger.warning(
                "Semantic ledger summary could not be completed; publication gate "
                "is not active yet: %s",
                exc,
            )

    _require_semantic_publication_inputs(
        {
            "scene_requirement_graph.json": requirement_graph_path,
            "scene_blueprint.json": scene_blueprint_path,
            "semantic_spatial_compilation.json": spatial_compilation_path,
            "semantic_capability_manifest.json": capability_manifest_path,
            "semantic_strategy_journal.json": strategy_journal_path,
            "semantic_publication_certificate.json": (
                room_dir / "semantic_publication_certificate.json"
            ),
        }
    )
