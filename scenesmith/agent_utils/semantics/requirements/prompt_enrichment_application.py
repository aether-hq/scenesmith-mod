"""Validation, blueprint application, and persistence for prompt enrichment."""

from __future__ import annotations

import os
import tempfile

from pathlib import Path

from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_models import (
    EnrichmentModel,
    SceneEnrichmentDraft,
    SemanticPromptEnrichment,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    SceneRequirementGraph,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    BlueprintConstraint,
    SceneBlueprint,
)
from scenesmith.agent_utils.semantics.requirements.semantic_prompt_enrichment import (
    validate_prompt_enrichment,
)


def apply_prompt_enrichment(
    blueprint: SceneBlueprint,
    enrichment: SemanticPromptEnrichment,
    graph: SceneRequirementGraph,
) -> SceneBlueprint:
    """Inject complete and per-instance prompts without changing hard semantics."""

    validate_prompt_enrichment(enrichment, graph, blueprint)
    constraints = list(blueprint.constraints)
    existing_ids = {constraint.constraint_id for constraint in constraints}
    for role in enrichment.repeated_roles:
        payload = [instance.model_dump(mode="json") for instance in role.instances]
        matched = False
        updated: list[BlueprintConstraint] = []
        for constraint in constraints:
            requirement_id = str(constraint.parameters.get("requirement_id") or "")
            role_key = str(constraint.parameters.get("role_key") or "") or None
            requirement_matches = (
                role.requirement_id is not None
                and requirement_id == role.requirement_id
            )
            role_matches = role.role_key is None or role_key == role.role_key
            artifacts_match = bool(set(constraint.target_ids) & set(role.artifact_ids))
            if requirement_matches and role_matches and artifacts_match:
                parameters = dict(constraint.parameters)
                parameters.update(
                    {
                        "instance_prompts": payload,
                        "prompt_enrichment_hash": enrichment.content_hash,
                    }
                )
                updated.append(constraint.model_copy(update={"parameters": parameters}))
                matched = True
            else:
                updated.append(constraint)
        constraints = updated
        if not matched:
            constraint_id = f"enrichment-{role.target_id}"
            suffix = 1
            while constraint_id in existing_ids:
                suffix += 1
                constraint_id = f"enrichment-{role.target_id}-{suffix}"
            existing_ids.add(constraint_id)
            constraints.append(
                BlueprintConstraint(
                    constraint_id=constraint_id,
                    kind="semantic_instance_enrichment",
                    target_ids=role.artifact_ids,
                    parameters={
                        "requirement_id": role.requirement_id,
                        "role_key": role.role_key,
                        "instance_prompts": payload,
                        "prompt_enrichment_hash": enrichment.content_hash,
                    },
                    strength="soft",
                    source="inferred",
                )
            )

    requirement_prompts = {
        prompt.requirement_id: prompt.construction_prompt
        for prompt in enrichment.scene.requirement_prompts
    }
    updated_constraints = []
    for constraint in constraints:
        requirement_id = str(constraint.parameters.get("requirement_id") or "")
        requirement_prompt = requirement_prompts.get(requirement_id)
        if requirement_prompt is None:
            updated_constraints.append(constraint)
            continue
        parameters = dict(constraint.parameters)
        parameters.update(
            {
                "requirement_prompt": requirement_prompt,
                "prompt_enrichment_hash": enrichment.content_hash,
            }
        )
        updated_constraints.append(
            constraint.model_copy(update={"parameters": parameters})
        )

    suffix = (
        "\n\nSEMANTIC PROMPT ENRICHMENT — inferred context is advisory; immutable "
        "requirements still win:\n" + enrichment.complete_prompt
    )
    return blueprint.model_copy(
        update={
            "spaces": tuple(
                space.model_copy(
                    update={
                        "prompt": (space.prompt or blueprint.source_prompt) + suffix
                    }
                )
                for space in blueprint.spaces
            ),
            "constraints": tuple(updated_constraints),
        }
    )


def _persist_model(model: EnrichmentModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def persist_scene_enrichment(scene: SceneEnrichmentDraft, output_path: Path) -> None:
    _persist_model(scene, output_path)


def load_scene_enrichment(path: Path) -> SceneEnrichmentDraft:
    return SceneEnrichmentDraft.model_validate_json(path.read_text(encoding="utf-8"))


def persist_prompt_enrichment(
    enrichment: SemanticPromptEnrichment, output_path: Path
) -> None:
    _persist_model(enrichment, output_path)


def load_prompt_enrichment(path: Path) -> SemanticPromptEnrichment:
    return SemanticPromptEnrichment.model_validate_json(
        path.read_text(encoding="utf-8")
    )
