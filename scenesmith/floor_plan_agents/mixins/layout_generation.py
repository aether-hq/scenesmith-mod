"""Stateful floor plan agent using planner/designer/critic workflow.

This module implements the floor plan agent trio for designing house layouts
with rooms, doors, windows, and materials, then generates the geometry.
"""

import json
import logging
import shutil

from pathlib import Path

from scenesmith.agent_utils.design.design_system import (
    apply_style_bible,
    compile_style_bible,
    load_design_system_from_env,
    persist_design_contract,
)
from scenesmith.agent_utils.runtime.base_stateful_agent import log_agent_usage
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.scene_candidates import (
    create_candidate_tournament,
    persist_candidate_tournament,
)
from scenesmith.agent_utils.semantics.requirements.blueprint_io import (
    persist_scene_blueprint,
)
from scenesmith.agent_utils.semantics.requirements.compilation.expansion import (
    blueprint_with_obligation_brief,
)
from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    SpatialCompilationError,
)
from scenesmith.agent_utils.semantics.requirements.prompt_enrichment_application import (
    apply_prompt_enrichment,
    persist_prompt_enrichment,
    persist_scene_enrichment,
)
from scenesmith.agent_utils.semantics.requirements.requirement_blueprint_compiler import (
    compile_requirement_blueprint,
    persist_spatial_compilation,
    persist_topology_manifest,
    validate_constructed_topology,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.literal_parsing import (
    literal_candidates_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    semantic_model_name,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    BlueprintDesignTokens,
    blueprint_from_prompt,
    floor_plan_submission_from_blueprint,
)
from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    analyze_requirement_candidates,
    merge_requirement_interpretations,
    persist_requirement_graph,
    requirement_graph_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.semantic_ledger import (
    load_or_initialize_semantic_ledger,
    persist_semantic_ledger,
    persist_semantic_ledger_summary,
    transition_requirement,
)
from scenesmith.agent_utils.semantics.requirements.semantic_prompt_enrichment import (
    analyze_repeated_instance_enrichment,
    analyze_scene_enrichment,
    fallback_scene_enrichment,
    finalize_prompt_enrichment,
)
from scenesmith.agent_utils.semantics.requirements.semantic_strategies import (
    SemanticCapabilityProfile,
    apply_capability_manifest_to_ledger,
    assert_capability_preflight_passed,
    capability_preflight,
    initialize_strategy_journal,
    load_strategy_journal,
    persist_capability_manifest,
    persist_strategy_journal,
    record_strategy_attempt,
)
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.tools.geometry_cache import GeometryCache
from scenesmith.floor_plan_agents.tools.submission.structural_submission import (
    structural_submission_from_blueprint,
    synthesize_structural_layout,
)
from scenesmith.prompts.registry import FloorPlanAgentPrompts

console_logger = logging.getLogger(__name__)


class FloorPlanLayoutGenerationMixin:
    """Stateful floor plan agent using planner/designer/critic workflow.

    This agent designs house layouts through an iterative process of:
    1. Designer proposes rooms, doors, windows using layout tools.
    2. Critic evaluates the design with VLM-based visual critique.
    3. Iteration continues until the design meets quality criteria.

    The layout is stored in a HouseLayout object that tracks:
    - Room specifications with adjacency constraints
    - Door and window placements on walls
    - Material assignments for floors and walls

    After design completion, geometry is generated for each room:
    - Floor meshes as GLTF
    - Wall meshes with door/window openings as GLTF
    - Full SDF/URDF assembly for Drake simulation
    """

    # Floor plan agent doesn't place objects, so no placement style tool.
    _is_placement_agent: bool = False

    async def generate_house_layout(self, prompt: str, output_dir: Path) -> HouseLayout:
        """Generate a house layout with floor plan geometry.

        This is the main entry point for floor plan generation. It runs the agent trio
        to design the layout, then generates geometry for all rooms.

        Args:
            prompt: Description of the house/room to design.
            output_dir: Directory to save generated geometry files.

        Returns:
            HouseLayout with designed layout and generated RoomGeometry.
        """
        self._reset_workflow_budget()

        # Initialize state (wall_height has sensible default, agent can override).
        # Set house_dir early so materials resolver can use it.
        house_dir = output_dir.parent if output_dir else self.logger.output_dir
        self.layout = HouseLayout(house_dir=house_dir, house_prompt=prompt)
        design_system = load_design_system_from_env()
        style_bible = compile_style_bible(design_system) if design_system else None
        styled_prompt = (
            apply_style_bible(prompt, style_bible) if style_bible else prompt
        )
        self.layout.house_prompt = styled_prompt
        if design_system and style_bible:
            persist_design_contract(design_system, style_bible, self.logger.output_dir)

        literal_candidates = literal_candidates_from_prompt(styled_prompt)
        configured_model = getattr(getattr(self.cfg, "openai", None), "model", None)
        semantic_model = semantic_model_name(configured_model)
        try:
            if not semantic_model:
                raise RuntimeError("semantic obligation model is not configured")
            interpretations, analysis_result = await analyze_requirement_candidates(
                styled_prompt,
                literal_candidates,
                model=semantic_model,
                run_config=self._create_run_config(),
                model_settings=self._get_model_settings(settings_key="designer"),
            )
            log_agent_usage(
                result=analysis_result,
                agent_name="SEMANTIC OBLIGATION ANALYST",
            )
            self.requirement_graph = merge_requirement_interpretations(
                styled_prompt,
                literal_candidates,
                interpretations,
                analysis_model=semantic_model,
            )
        except Exception as exc:
            # Preserve every source clause as unclassified if interpretation fails.
            # Unclassified hard obligations are unresolved-blocking, so capability
            # preflight and publication both fail closed with source evidence.
            console_logger.exception(
                "Semantic obligation analysis failed; preserving literal candidates"
            )
            self.requirement_graph = requirement_graph_from_prompt(
                styled_prompt,
                analysis_model=semantic_model or None,
                analysis_error=f"{type(exc).__name__}: {exc}",
            )
        persist_requirement_graph(
            self.requirement_graph,
            self.logger.output_dir / "scene_requirement_graph.json",
        )
        try:
            if not semantic_model:
                raise RuntimeError("semantic enrichment model is not configured")
            scene_enrichment, enrichment_result = await analyze_scene_enrichment(
                self.requirement_graph,
                model=semantic_model,
                run_config=self._create_run_config(),
                model_settings=self._get_model_settings(settings_key="designer"),
            )
            log_agent_usage(
                result=enrichment_result,
                agent_name="SEMANTIC ENVIRONMENT ENRICHER",
            )
        except Exception as exc:
            console_logger.exception(
                "Semantic scene enrichment failed; using deterministic context"
            )
            scene_enrichment = fallback_scene_enrichment(
                self.requirement_graph,
                analysis_model=semantic_model or None,
                diagnostic=f"{type(exc).__name__}: {exc}",
            )
        persist_scene_enrichment(
            scene_enrichment,
            self.logger.output_dir / "semantic_scene_enrichment.json",
        )
        self.semantic_ledger = load_or_initialize_semantic_ledger(
            self.logger.output_dir / "semantic_obligation_ledger.json",
            self.requirement_graph,
        )
        configured_capabilities = getattr(self.cfg, "semantic_capabilities", None)
        capability_manifest = None
        if configured_capabilities is not None:
            capability_manifest_path = (
                self.logger.output_dir / "semantic_capability_manifest.json"
            )
            capability_profile = SemanticCapabilityProfile.model_validate(
                dict(configured_capabilities)
            )
            capability_manifest = capability_preflight(
                self.requirement_graph,
                capability_profile,
            )
            persist_capability_manifest(
                capability_manifest,
                capability_manifest_path,
            )
            persist_strategy_journal(
                initialize_strategy_journal(capability_manifest),
                self.logger.output_dir / "semantic_strategy_journal.json",
            )
            self.semantic_ledger = apply_capability_manifest_to_ledger(
                self.semantic_ledger,
                self.requirement_graph,
                capability_manifest,
                manifest_ref=str(capability_manifest_path),
            )
            persist_semantic_ledger(
                self.semantic_ledger,
                self.logger.output_dir / "semantic_obligation_ledger.json",
            )
            assert_capability_preflight_passed(capability_manifest)
        persist_semantic_ledger_summary(
            self.semantic_ledger,
            self.logger.output_dir / "semantic_obligation_summary.json",
        )
        console_logger.info(
            "Semantic requirement graph %s captured %d literal candidates "
            "and %d obligations (analysis=%s, hash=%s)",
            self.requirement_graph.graph_id,
            len(self.requirement_graph.candidates),
            len(self.requirement_graph.requirements),
            self.requirement_graph.analysis_status,
            self.requirement_graph.content_hash,
        )

        if capability_manifest is not None:
            spatial_compilation, spatial_result = await compile_requirement_blueprint(
                self.requirement_graph,
                capability_manifest,
                model=semantic_model,
                mode=self.mode,
                scene_enrichment=scene_enrichment,
                maximum_dimension_m=None,
                maximum_height_m=None,
                maximum_opening_width_m=None,
                maximum_opening_height_m=None,
                run_config=self._create_run_config(),
                model_settings=self._get_model_settings(settings_key="designer"),
            )
            log_agent_usage(
                result=spatial_result,
                agent_name="SPATIAL AND TOPOLOGICAL COMPILER",
            )
            persist_spatial_compilation(
                spatial_compilation,
                self.logger.output_dir / "semantic_spatial_compilation.json",
            )
            self.blueprint = blueprint_with_obligation_brief(
                spatial_compilation,
                self.requirement_graph,
            )
        else:
            self.blueprint = blueprint_from_prompt(
                styled_prompt,
                mode=self.mode,
                default_dimensions_m=(
                    7.0,
                    7.0,
                ),
                maximum_dimension_m=None,
            )
        if style_bible is not None:
            self.blueprint = self.blueprint.model_copy(
                update={
                    "design_tokens": BlueprintDesignTokens(
                        style_keywords=style_bible.asset_search_tags,
                        palette=tuple(style_bible.palette_roles.values()),
                        material_roles=style_bible.material_roles,
                        lighting_mood=style_bible.ceiling_direction,
                        focal_hierarchy=tuple(
                            design_system.set_dressing.focal_hierarchy
                        ),
                    )
                }
            )
        if capability_manifest is None:
            self.candidate_tournament = create_candidate_tournament(
                self.blueprint,
                prompt=prompt,
                candidate_count=6,
            )
            persist_candidate_tournament(
                self.candidate_tournament,
                self.logger.output_dir / "scene_candidates.json",
            )
            self.blueprint = self.candidate_tournament.winner.blueprint
        repeated_wire = None
        repeated_diagnostic = None
        try:
            if not semantic_model:
                raise RuntimeError("semantic enrichment model is not configured")
            repeated_wire, repeated_result = await analyze_repeated_instance_enrichment(
                self.requirement_graph,
                self.blueprint,
                scene_enrichment,
                model=semantic_model,
                run_config=self._create_run_config(),
                model_settings=self._get_model_settings(settings_key="designer"),
            )
            if repeated_result is not None:
                log_agent_usage(
                    result=repeated_result,
                    agent_name="REPEATED INSTANCE ENRICHER",
                )
        except Exception as exc:
            console_logger.exception(
                "Repeated-instance enrichment failed; using deterministic variants"
            )
            repeated_diagnostic = f"{type(exc).__name__}: {exc}"
        prompt_enrichment = finalize_prompt_enrichment(
            self.requirement_graph,
            self.blueprint,
            scene_enrichment,
            repeated_wire,
            repeated_diagnostic=repeated_diagnostic,
        )
        persist_prompt_enrichment(
            prompt_enrichment,
            self.logger.output_dir / "semantic_prompt_enrichment.json",
        )
        self.blueprint = apply_prompt_enrichment(
            self.blueprint,
            prompt_enrichment,
            self.requirement_graph,
        )
        console_logger.info(
            "Semantic prompt enrichment %s authored %d requirement briefs and "
            "%d unique repeated-instance briefs (status=%s)",
            prompt_enrichment.content_hash,
            len(prompt_enrichment.scene.requirement_prompts),
            sum(len(role.instances) for role in prompt_enrichment.repeated_roles),
            prompt_enrichment.analysis_status,
        )
        persist_scene_blueprint(
            self.blueprint, self.logger.output_dir / "scene_blueprint.json"
        )
        self.house_prompt = self.blueprint.to_prompt_brief()
        if self.candidate_tournament is not None:
            console_logger.info(
                "Selected proxy candidate %s (score %.2f) as SceneBlueprint %s with "
                "%d spaces and %d connectors",
                self.candidate_tournament.winner_id,
                self.candidate_tournament.winner.scores.total,
                self.blueprint.blueprint_id,
                len(self.blueprint.spaces),
                len(self.blueprint.connectors),
            )
        else:
            console_logger.info(
                "Accepted requirement-bound SceneBlueprint %s with %d spaces, "
                "%d connectors, and %d hard bindings",
                self.blueprint.blueprint_id,
                len(self.blueprint.spaces),
                len(self.blueprint.connectors),
                len(spatial_compilation.bindings),
            )

        # Initialize geometry cache for reusing unchanged room geometry.
        cache_dir = house_dir / ".geometry_cache"
        self._geometry_cache = GeometryCache(cache_dir=cache_dir)

        # Create agents.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)

        if self.cfg.max_critique_rounds <= 0:
            blueprint_submission = floor_plan_submission_from_blueprint(self.blueprint)
            blueprint_submission["structural"] = structural_submission_from_blueprint(
                self.blueprint,
                blueprint_submission["room_specs"],
                max_total_height=self._construction_wall_height_max(),
            )
            if (
                capability_manifest is not None
                and len(designer_tools) == 1
                and getattr(designer_tools[0], "name", "") == "submit_floor_plan"
            ):
                console_logger.info(
                    "Using authoritative requirement-bound floor-plan authoring"
                )
                result = await designer_tools[0].on_invoke_tool(
                    None, json.dumps(blueprint_submission)
                )
            else:
                # There is no critic to coordinate, so a planner would only add a
                # second serial LLM launch before requesting the same design.
                result = await self._request_initial_design_impl()
        else:
            critic_tools = self._create_critic_tools()
            self.critic = self._create_critic_agent(tools=critic_tools)

            planner_tools = self._create_planner_tools()
            self.planner = self._create_planner_agent(tools=planner_tools)

            runner_instruction = self.prompt_registry.get_prompt(
                prompt_enum=FloorPlanAgentPrompts.PLANNER_RUNNER_INSTRUCTION,
            )
            result = await self._run_planner_with_partial_recovery(
                runner_instruction=runner_instruction,
                agent_name="PLANNER (FLOOR PLAN)",
                state_hash=self.layout.content_hash,
            )
        if capability_manifest is not None:
            self._apply_locked_blueprint_topology()
        if not self._write_resumable_layout_checkpoint():
            raise RuntimeError(
                "Floor-plan stage did not produce a structurally valid checkpoint; "
                "the incomplete room was not exported. Resume the build to retry "
                "the layout stage."
            )
        if capability_manifest is not None:
            topology_manifest_path = (
                self.logger.output_dir / "semantic_topology_manifest.json"
            )
            strategy_journal_path = (
                self.logger.output_dir / "semantic_strategy_journal.json"
            )
            strategy_journal = load_strategy_journal(strategy_journal_path)
            try:
                topology_manifest = validate_constructed_topology(
                    spatial_compilation,
                    self.requirement_graph,
                    self.layout,
                )
            except SpatialCompilationError as exc:
                diagnostic = str(exc)
                failed_plans = [
                    plan
                    for plan in capability_manifest.plans
                    if plan.requirement_id in diagnostic
                ]
                for plan in failed_plans:
                    if (
                        plan.selected_strategy is not None
                        and plan.selected_provider is not None
                    ):
                        strategy_journal = record_strategy_attempt(
                            strategy_journal,
                            capability_manifest,
                            attempt_key=f"topology-failed:{plan.requirement_id}",
                            requirement_id=plan.requirement_id,
                            strategy=plan.selected_strategy,
                            provider_id=plan.selected_provider,
                            stage="topology",
                            outcome="failed",
                            diagnostic=diagnostic,
                        )
                    self.semantic_ledger = transition_requirement(
                        self.semantic_ledger,
                        plan.requirement_id,
                        "failed",
                        event_key=f"topology:failed:{plan.requirement_id}",
                        actor="spatial_topology_gate",
                        stage="topology",
                        evidence_refs=(str(topology_manifest_path),),
                        failure_reason=diagnostic,
                    )
                persist_strategy_journal(strategy_journal, strategy_journal_path)
                persist_semantic_ledger(
                    self.semantic_ledger,
                    self.logger.output_dir / "semantic_obligation_ledger.json",
                )
                persist_semantic_ledger_summary(
                    self.semantic_ledger,
                    self.logger.output_dir / "semantic_obligation_summary.json",
                )
                raise
            persist_topology_manifest(topology_manifest, topology_manifest_path)
            for evidence in topology_manifest.evidence:
                plan = next(
                    item
                    for item in capability_manifest.plans
                    if item.requirement_id == evidence.requirement_id
                )
                evidence_refs = tuple(
                    f"topology:{artifact_id}"
                    for artifact_id in evidence.actual_artifact_ids
                ) or (str(topology_manifest_path),)
                if (
                    plan.selected_strategy is not None
                    and plan.selected_provider is not None
                ):
                    strategy_journal = record_strategy_attempt(
                        strategy_journal,
                        capability_manifest,
                        attempt_key=f"topology:{evidence.requirement_id}",
                        requirement_id=evidence.requirement_id,
                        strategy=plan.selected_strategy,
                        provider_id=plan.selected_provider,
                        stage="topology",
                        outcome="succeeded",
                        evidence_refs=evidence_refs,
                    )
                self.semantic_ledger = transition_requirement(
                    self.semantic_ledger,
                    evidence.requirement_id,
                    "constructed",
                    event_key=f"topology:constructed:{evidence.requirement_id}",
                    actor="spatial_topology_gate",
                    stage="topology",
                    evidence_refs=evidence_refs,
                )
            persist_strategy_journal(strategy_journal, strategy_journal_path)
            persist_semantic_ledger(
                self.semantic_ledger,
                self.logger.output_dir / "semantic_obligation_ledger.json",
            )
            persist_semantic_ledger_summary(
                self.semantic_ledger,
                self.logger.output_dir / "semantic_obligation_summary.json",
            )

        # Final critique.
        # Check if scene changed since last checkpoint to avoid redundant critique.
        current_scene_hash = self.layout.content_hash()

        if (
            self.cfg.max_critique_rounds <= 0
            or self._workflow_limit_reached
            or self._critique_calls >= int(self.cfg.max_critique_rounds)
        ):
            console_logger.info("Final critique skipped: critique budget unavailable")
            vision_tools = self._get_vision_tools()
            self.final_render_dir = vision_tools.last_render_dir
        elif (
            self.checkpoint_scene_hash is not None
            and current_scene_hash == self.checkpoint_scene_hash
        ):
            console_logger.info(
                "Scene unchanged since last critique, skipping final critique"
            )
        else:
            console_logger.info(
                "Scene changed since last critique, computing final critique"
            )
            # Pass update_checkpoint=False to preserve N-1 checkpoint for reset check.
            await self._request_critique_bounded(update_checkpoint=False)

        # Validate final scene against thresholds and potentially reset.
        await self._finalize_scene_and_scores()

        # Generate geometry for all rooms.
        console_logger.info("Generating geometry for all rooms")
        self._generate_all_room_geometries(output_dir=output_dir)

        # Log cache statistics.
        if self._geometry_cache is not None:
            self._geometry_cache.log_stats()

        # Save final layout.
        layout_path = self.logger.output_dir / "house_layout.json"
        with open(layout_path, "w") as f:
            json.dump(self.layout.to_dict(), f, indent=2)
        console_logger.info(f"House layout saved to: {layout_path}")

        # Export floor plan to .dmd.yaml and .blend.
        self._export_floor_plan(output_dir=output_dir)

        # Clean up geometry cache (files already copied to output).
        if self._geometry_cache is not None:
            shutil.rmtree(self._geometry_cache.cache_dir, ignore_errors=True)
            self._geometry_cache = None

        return self.layout

    async def revise_house_layout(
        self,
        *,
        existing_layout: HouseLayout,
        feedback: str,
        output_dir: Path,
        locks: tuple[str, ...] = (),
    ) -> HouseLayout:
        """Branch a durable layout checkpoint and apply architectural feedback.

        Explicit multi-level circulation has a deterministic implementation so
        common mezzanine/stair revisions do not depend on a model restating a
        fragile structural schema. Other architectural feedback reuses the
        provider-neutral designer against the restored checkpoint.
        """

        self._reset_workflow_budget()
        house_dir = output_dir.parent if output_dir else self.logger.output_dir
        self.layout = existing_layout
        self.layout.house_dir = house_dir
        original_prompt = self.layout.house_prompt
        revision_instruction = (
            f"Original scene: {original_prompt}\n"
            f"Revision request: {feedback}\n"
            f"Preserve locks: {', '.join(locks) if locks else 'all unaffected properties'}."
        )
        self.layout.house_prompt = revision_instruction
        self.house_prompt = revision_instruction
        for room_spec in self.layout.room_specs:
            room_spec.prompt = f"{room_spec.prompt}\nRevision request: {feedback}"

        cache_dir = house_dir / ".geometry_cache"
        self._geometry_cache = GeometryCache(cache_dir=cache_dir)
        room_specs = [
            {
                "type": spec.room_id,
                "width": spec.length,
                "depth": spec.width,
                "prompt": spec.prompt,
            }
            for spec in self.layout.room_specs
        ]
        existing_storey_height = (
            self.layout.levels[0].nominal_height
            if self.layout.levels
            else self.layout.wall_height
        )
        structural = synthesize_structural_layout(
            feedback,
            room_specs,
            min(existing_storey_height, self._construction_wall_height_max()),
            max_total_height=self._construction_wall_height_max(),
            level_count_hint=len(self.layout.levels),
        )

        if structural is not None:
            total_height = max(
                level["elevation"] + level["nominal_height"]
                for level in structural["levels"]
            )
            self.layout.wall_height = total_height
            tools = FloorPlanTools(
                layout=self.layout,
                mode=self.mode,
                materials_config=self._create_materials_config(),
                wall_height_min=self.cfg.wall_height.min,
                wall_height_max=self._construction_wall_height_max(),
                room_dim_min=self.cfg.min_floor_plan_dim_m,
                room_dim_max=self._construction_room_dim_max(),
            )
            result = tools._set_structural_layout_impl(structural)
            if not result.success:
                raise RuntimeError(
                    f"Deterministic architectural revision failed: {result.message}"
                )
            console_logger.info(
                "Applied deterministic checkpoint revision: %s", result.message
            )
        else:
            designer_tools = self._create_designer_tools()
            self.designer = self._create_designer_agent(tools=designer_tools)
            await self._request_design_change_impl(revision_instruction)

        if not self._write_resumable_layout_checkpoint():
            raise RuntimeError(
                "Architectural revision did not produce a structurally valid "
                "checkpoint; the prior version remains active."
            )
        self._generate_all_room_geometries(output_dir=output_dir)
        layout_path = self.logger.output_dir / "house_layout.json"
        with open(layout_path, "w") as file:
            json.dump(self.layout.to_dict(), file, indent=2)
        self._export_floor_plan(output_dir=output_dir)
        if self._geometry_cache is not None:
            shutil.rmtree(self._geometry_cache.cache_dir, ignore_errors=True)
            self._geometry_cache = None
        return self.layout
