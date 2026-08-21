"""Stateful floor plan agent using planner/designer/critic workflow.

This module implements the floor plan agent trio for designing house layouts
with rooms, doors, windows, and materials, then generates the geometry.
"""

import copy
import json
import logging
import math
import shutil

from pathlib import Path
from typing import Any

import lxml.etree as ET
import numpy as np
import trimesh
import yaml

from agents import Agent, FunctionTool
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.action_logger import log_scene_action
from scenesmith.agent_utils.agent_runtime import (
    BoundedRunner as Runner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    log_agent_usage,
)
from scenesmith.agent_utils.blender import BlenderServer
from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.clearance_zones import compute_openings_data
from scenesmith.agent_utils.design_system import (
    apply_style_bible,
    compile_style_bible,
    load_design_system_from_env,
    persist_design_contract,
)
from scenesmith.agent_utils.house import (
    HouseLayout,
    Opening,
    OpeningType,
    PlacedRoom,
    RoomGeometry,
    RoomSpec,
    Wall,
    WallDirection,
    compute_wall_normals,
    legacy_openings_to_boundary_portals,
)
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.rendering import save_directive_as_blend
from scenesmith.agent_utils.room import AgentType, ObjectType, SceneObject, UniqueID
from scenesmith.agent_utils.scene_blueprint import (
    BlueprintDesignTokens,
    SceneBlueprint,
    blueprint_from_prompt,
    floor_plan_submission_from_blueprint,
    persist_scene_blueprint,
)
from scenesmith.agent_utils.scene_requirements import (
    analyze_requirement_candidates,
    literal_candidates_from_prompt,
    merge_requirement_interpretations,
    persist_requirement_graph,
    requirement_graph_from_prompt,
)
from scenesmith.agent_utils.scene_candidates import (
    CandidateTournament,
    create_candidate_tournament,
    persist_candidate_tournament,
)
from scenesmith.agent_utils.scoring import (
    FloorPlanCritiqueWithScores,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
)
from scenesmith.agent_utils.semantic_ledger import (
    load_or_initialize_semantic_ledger,
    persist_semantic_ledger,
    persist_semantic_ledger_summary,
)
from scenesmith.agent_utils.semantic_strategies import (
    SemanticCapabilityProfile,
    apply_capability_manifest_to_ledger,
    assert_capability_preflight_passed,
    capability_preflight,
    persist_capability_manifest,
)
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.floor_plan_agents.base_floor_plan_agent import BaseFloorPlanAgent
from scenesmith.floor_plan_agents.tools.floor_plan_submission import (
    normalize_floor_plan_submission,
    synthesize_structural_layout,
)
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.tools.geometry_cache import (
    GeometryCache,
    floor_cache_key,
    wall_cache_key,
    window_cache_key,
)
from scenesmith.floor_plan_agents.tools.vision_tools import FloorPlanVisionTools
from scenesmith.floor_plan_agents.tools.wall_geometry import (
    WallDimensions,
    WallOpening,
    WallSpec,
    create_wall_gltf as create_wall_gltf_with_openings,
)
from scenesmith.floor_plan_agents.tools.window_geometry import create_window_mesh
from scenesmith.prompts.registry import FloorPlanAgentPrompts
from scenesmith.utils.gltf_generation import create_floor_gltf, get_zup_to_yup_matrix
from scenesmith.utils.logging import BaseLogger
from scenesmith.utils.material import Material

console_logger = logging.getLogger(__name__)


class StatefulFloorPlanAgent(BaseStatefulAgent, BaseFloorPlanAgent):
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

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.FLOOR_PLAN

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        render_allocation: RenderAllocation | None = None,
    ):
        """Initialize the floor plan agent.

        Args:
            cfg: Hydra configuration for the agent.
            logger: Logger for output and debugging.
            render_allocation: Provider-owned Blender render slot.
        """
        BaseFloorPlanAgent.__init__(self, cfg=cfg, logger=logger)
        BaseStatefulAgent.__init__(self, cfg=cfg, logger=logger)

        # Start BlenderServer for rendering.
        console_logger.info("Starting BlenderServer for floor plan rendering")
        self.blender_server = BlenderServer(
            port_range=tuple(cfg.rendering.blender_server_port_range),
            render_allocation=render_allocation,
            log_file=logger.output_dir / "scene.log",
            render_provider=str(cfg.rendering.get("provider", "auto")),
        )
        self.blender_server.start()
        self.blender_server.wait_until_ready()

        # Vision tools for floor plan rendering (lazy initialized).
        self._vision_tools: FloorPlanVisionTools | None = None

        # Geometry cache for reusing unchanged room geometry across iterations.
        self._geometry_cache: GeometryCache | None = None

        # Prompt and layout state.
        self.house_prompt: str = ""
        self.layout: HouseLayout = HouseLayout()
        self.blueprint: SceneBlueprint | None = None
        self.candidate_tournament: CandidateTournament | None = None

        # Create persistent agent sessions.
        self.designer_session, self.critic_session = self._create_sessions()

    def _get_vision_tools(self) -> FloorPlanVisionTools:
        """Get or create the shared vision tools instance."""
        if self._vision_tools is None:
            output_dir = self.logger.output_dir / "floor_plans"
            self._vision_tools = FloorPlanVisionTools(
                layout=self.layout,
                output_dir=output_dir,
                blender_server=self.blender_server,
                wall_thickness=self.cfg.wall_thickness,
                floor_thickness=self.cfg.floor_thickness,
                render_size=self.cfg.rendering.render_size,
                generate_geometries_callback=lambda: self._generate_all_room_geometries(
                    output_dir=output_dir
                ),
            )
        return self._vision_tools

    def cleanup(self) -> None:
        """Cleanup resources held by the agent."""
        # Stop BlenderServer (matches other agents' pattern).
        if self.blender_server is not None and self.blender_server.is_running():
            console_logger.info("Stopping BlenderServer")
            self.blender_server.stop()

        # Call parent cleanup.
        BaseFloorPlanAgent.cleanup(self)

    def _create_designer_tools(self) -> list[FunctionTool]:
        """Create tools for the designer agent.

        Returns:
            List of function tools for floor plan design.
        """
        floor_plan_tools = FloorPlanTools(
            layout=self.layout,
            mode=self.mode,
            materials_config=self._create_materials_config(),
            min_opening_separation=self.cfg.room_placement.min_opening_separation,
            placement_timeout_seconds=self.cfg.room_placement.timeout_seconds,
            placement_scoring_weights=self._create_scoring_weights(),
            placement_exterior_wall_clearance_m=self.cfg.room_placement.exterior_wall_clearance_m,
            door_window_config=self._create_door_window_config(),
            wall_height_min=self.cfg.wall_height.min,
            wall_height_max=self.cfg.wall_height.max,
            room_dim_min=self.cfg.min_floor_plan_dim_m,
            room_dim_max=self.cfg.max_floor_plan_dim_m,
            checkpoint_callback=self._write_resumable_layout_checkpoint,
        )

        if self.cfg.max_critique_rounds <= 0:
            # Production fast path: one model request produces design intent and
            # local code executes the ordered primitives. Critique-enabled runs
            # retain the granular toolbox for iterative editing.
            return [floor_plan_tools.submit_floor_plan_tool]

        vision_tools = self._get_vision_tools()
        workflow_tools = WorkflowTools()

        return (
            list(floor_plan_tools.tools.values())
            + list(vision_tools.tools.values())
            + list(workflow_tools.tools.values())
        )

    def _create_critic_tools(self) -> list[FunctionTool]:
        """Create tools for the critic agent.

        Critic needs:
        - observe_scene, render_ascii (vision_tools) - for visual context
        - validate (floor_plan_tools) - for layout/connectivity status

        Returns:
            List of function tools for floor plan critique.
        """
        vision_tools = self._get_vision_tools()

        # Add validate tool from floor_plan_tools (read-only).
        floor_plan_tools = FloorPlanTools(
            layout=self.layout,
            mode=self.mode,
            materials_config=self._create_materials_config(),
            min_opening_separation=self.cfg.room_placement.min_opening_separation,
            placement_timeout_seconds=self.cfg.room_placement.timeout_seconds,
            placement_scoring_weights=self._create_scoring_weights(),
            placement_exterior_wall_clearance_m=self.cfg.room_placement.exterior_wall_clearance_m,
            door_window_config=self._create_door_window_config(),
            wall_height_min=self.cfg.wall_height.min,
            wall_height_max=self.cfg.wall_height.max,
            room_dim_min=self.cfg.min_floor_plan_dim_m,
            room_dim_max=self.cfg.max_floor_plan_dim_m,
        )

        return list(vision_tools.tools.values()) + [floor_plan_tools.tools["validate"]]

    def _create_designer_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the designer agent.

        Args:
            tools: Tools to provide to the designer.

        Returns:
            Configured designer agent.
        """
        one_shot = (
            len(tools) == 1 and getattr(tools[0], "name", "") == "submit_floor_plan"
        )
        agent = super()._create_designer_agent(
            tools=tools,
            prompt_enum=(
                FloorPlanAgentPrompts.DESIGNER_ONE_SHOT_AGENT
                if one_shot
                else FloorPlanAgentPrompts.DESIGNER_AGENT
            ),
            mode=self.mode,
            house_prompt=self.house_prompt,
        )
        if one_shot:
            # The tool result is already the validated stage result. Asking the
            # model for a prose summary caused a redundant 32-second request and
            # allowed it to claim completion before validation.
            agent.tool_use_behavior = "stop_on_first_tool"
        return agent

    def _create_critic_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the critic agent.

        Args:
            tools: Tools to provide to the critic.

        Returns:
            Configured critic agent.
        """
        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=FloorPlanAgentPrompts.CRITIC_AGENT,
            output_type=FloorPlanCritiqueWithScores,
            mode=self.mode,
            house_prompt=self.house_prompt,
        )

    def _create_planner_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create the planner agent.

        Args:
            tools: Tools to provide to the planner.

        Returns:
            Configured planner agent.
        """
        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=FloorPlanAgentPrompts.PLANNER_AGENT,
            mode=self.mode,
            house_prompt=self.house_prompt,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=self.cfg.reset_single_category_threshold,
            reset_total_sum_threshold=self.cfg.reset_total_sum_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _get_final_scores_directory(self) -> Path:
        """Get directory for final scores.

        Returns:
            Path to final scores directory.
        """
        return self.logger.output_dir / "final_floor_plan"

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Prompt enum for critic instruction.
        """
        return FloorPlanAgentPrompts.CRITIC_RUNNER_INSTRUCTION

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Prompt enum for design change instruction.
        """
        return FloorPlanAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Prompt enum for initial design instruction.
        """
        return FloorPlanAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dictionary of kwargs for initial design prompt.
        """
        return {}

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile.

        Args:
            mode: Placement noise mode.
        """
        # Floor plan doesn't use placement noise.

    async def _request_critique_impl(self, update_checkpoint: bool = True) -> str:
        """Implementation for critique request.

        Runs critic which calls observe_scene, render_ascii, and validate tools.
        Images persist in session via ToolOutputImage.

        Args:
            update_checkpoint: Whether to shift checkpoints. Set to False for
                final critique calls to preserve N-1 checkpoint for reset check.

        Returns:
            Critique text with scores.
        """
        console_logger.info("Tool called: request_critique")

        # Critiques can involve multiple long-running vision/model turns. Save a
        # portable, geometry-complete layout before entering that boundary so a
        # malformed model response or interruption can resume at furniture.
        self._write_resumable_layout_checkpoint()

        # Get critique instruction.
        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FloorPlanAgentPrompts.CRITIC_RUNNER_INSTRUCTION,
        )

        # Run critic.
        # Critic will call observe_scene, render_ascii, and validate tools.
        result = await Runner.run(
            starting_agent=self.critic,
            input=critique_instruction,
            session=self.critic_session,
            max_turns=self.cfg.agents.critic_agent.max_turns,
            run_config=self._create_run_config(),
            timeout_seconds=agent_run_timeout_seconds(
                "critic",
                max_turns=self.cfg.agents.critic_agent.max_turns,
            ),
        )
        log_agent_usage(result=result, agent_name="CRITIC (FLOOR PLAN)")
        vision_tools = self._get_vision_tools()

        # Parse structured output.
        response = result.final_output_as(FloorPlanCritiqueWithScores)

        # Log critique.
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="FLOOR PLAN CRITIQUE SCORES")

        # Save scores to render directory.
        scores_dict = scores_to_dict(response)
        render_dir = vision_tools.last_render_dir

        # Always track the final render directory (separate from checkpoint logic).
        # This is needed because final critique uses update_checkpoint=False, but we
        # still need to know the actual last render dir for copying to final output.
        self.final_render_dir = render_dir

        scores_path = render_dir / "scores.yaml"
        with open(scores_path, "w") as f:
            yaml.dump(scores_dict, f, default_flow_style=False, sort_keys=False)
        console_logger.info(f"Scores saved to: {scores_path}")

        # Shift checkpoints only during iteration critiques, not final critique.
        # This preserves N-1 checkpoint for reset check in _finalize_scene_and_scores.
        if update_checkpoint:
            # Update checkpoint state (shift current to previous before saving new).
            self.previous_scene_checkpoint = self.scene_checkpoint
            self.previous_checkpoint_scores = self.checkpoint_scores
            self.previous_checkpoint_render_dir = self.checkpoint_render_dir

            # Save new checkpoint (current scene state).
            self.scene_checkpoint = copy.deepcopy(self.layout.to_dict())
            self.checkpoint_scores = response
            self.checkpoint_render_dir = (
                render_dir if render_dir and render_dir.exists() else None
            )

            # Reuse render cache hash for checkpoint change detection.
            self.checkpoint_scene_hash = self.layout.content_hash()

        # Compute score deltas BEFORE updating previous_scores.
        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )

        # Always update previous_scores for delta formatting in planner.
        self.previous_scores = response

        return response.critique + score_change_msg

    def _write_resumable_layout_checkpoint(self) -> bool:
        """Atomically persist the latest structurally valid floor-plan state."""
        if (
            not self.layout.room_specs
            or not self.layout.placement_valid
            or not self.layout.connectivity_valid
        ):
            console_logger.info(
                "Skipping floor-plan checkpoint because the layout is not yet valid"
            )
            return False
        try:
            self.layout.validate_structure()
            self._generate_all_room_geometries(
                output_dir=self.logger.output_dir / "floor_plans"
            )
            checkpoint_path = self.logger.output_dir / "house_layout.json"
            pending_path = checkpoint_path.with_suffix(".json.pending")
            with pending_path.open("w") as file:
                json.dump(
                    self.layout.to_dict(scene_dir=self.logger.output_dir),
                    file,
                    indent=2,
                )
            pending_path.replace(checkpoint_path)
            console_logger.info(
                "Saved resumable floor-plan checkpoint to %s", checkpoint_path
            )
            return True
        except Exception as exc:
            console_logger.warning(
                "Could not save resumable floor-plan checkpoint: %s", exc
            )
            return False

    @log_scene_action
    def _perform_checkpoint_reset(self, checkpoint_state_dict: dict) -> None:
        """Restore layout and scores to previous checkpoint (N-1).

        Override of base class method to restore HouseLayout instead of RoomScene.

        Args:
            checkpoint_state_dict: Checkpoint state dictionary to restore from.
                During normal operation, this is self.previous_scene_checkpoint.
                During replay, this is the logged checkpoint state.
        """
        # Restore layout from checkpoint (N-1 iteration).
        self.layout = HouseLayout.from_dict(
            data=checkpoint_state_dict, house_dir=self.layout.house_dir
        )

        # Force SDF regeneration since files on disk are not versioned.
        # Without this, room_geometries from checkpoint would be used,
        # but SDF files on disk have door positions from later iterations.
        self.layout.room_geometries.clear()

        # Update vision tools with restored layout (preserve render counter).
        if self._vision_tools is not None:
            self._vision_tools.update_layout(self.layout)
            self._vision_tools.clear_cache()

        # Recreate designer/critic tools (they reference self.layout directly).
        self._recreate_tools_with_layout()

        # Reset score tracking to previous checkpoint state.
        if self.previous_checkpoint_scores is not None:
            self.checkpoint_scores = copy.deepcopy(self.previous_checkpoint_scores)
            self.previous_scores = copy.deepcopy(self.previous_checkpoint_scores)

        # Invalidate current checkpoint since we went back.
        if self.previous_scene_checkpoint is not None:
            self.scene_checkpoint = self.previous_scene_checkpoint
            self.checkpoint_render_dir = self.previous_checkpoint_render_dir

    def _recreate_tools_with_layout(self) -> None:
        """Recreate tools after layout restoration to ensure they reference current layout."""
        # Designer tools reference self.layout, need to recreate them.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)

        # Critic tools also reference self.layout.
        critic_tools = self._create_critic_tools()
        self.critic = self._create_critic_agent(tools=critic_tools)

    async def _finalize_scene_and_scores(self) -> None:
        """Validate final scene against thresholds and save scores.

        Override of base class to use FloorPlanVisionTools instead of rendering_manager.
        The base class assumes `self.rendering_manager` and `self.scene` exist, but
        floor plan agent uses `FloorPlanVisionTools` and `self.layout` instead.
        """
        # Check if final scores warrant resetting to previous checkpoint.
        # Use previous_scores (actual final critique) vs checkpoint_scores (last checkpoint).
        # Note: Final critique uses update_checkpoint=False, so previous_scores holds the
        # actual final scores while checkpoint_scores holds the last iteration's scores.
        if self.previous_scores is not None and self.checkpoint_scores is not None:
            should_reset, reason = self._should_reset_to_checkpoint(
                current_scores=self.previous_scores,
                previous_scores=self.checkpoint_scores,
            )
            console_logger.info(
                f"Reset check result: should_reset={should_reset}, reason={reason}"
            )

            if should_reset:
                console_logger.info(
                    f"Final scene scores are degraded ({reason}). "
                    f"Resetting to checkpoint (N-1)."
                )

                # Restore layout to checkpoint (N-1) directly. Don't use
                # _perform_checkpoint_reset() here since that's designed for mid-loop
                # resets and modifies checkpoint tracking variables.
                self.layout = HouseLayout.from_dict(
                    data=self.scene_checkpoint, house_dir=self.layout.house_dir
                )

                # Force SDF regeneration since files on disk are not versioned.
                # Without this, room_geometries from checkpoint would be used,
                # but SDF files on disk have dimensions from later iterations.
                self.layout.room_geometries.clear()

                self._vision_tools = None
                self._recreate_tools_with_layout()

                # Render the reset state using vision tools.
                console_logger.info("Rendering final scene after reset")
                vision_tools = self._get_vision_tools()
                vision_tools.clear_cache()  # Force new render.
                vision_tools._observe_scene_impl()
                render_dir = vision_tools.last_render_dir
                self.checkpoint_render_dir = render_dir
                self.final_render_dir = render_dir  # Update so correct dir is copied.

                # Save scores to the new render directory.
                # Use checkpoint_scores (N-1) since we reset to that state.
                if self.checkpoint_scores is not None:
                    scores_dict = scores_to_dict(self.checkpoint_scores)
                    scores_path = render_dir / "scores.yaml"
                    with open(scores_path, "w") as f:
                        yaml.dump(
                            scores_dict,
                            f,
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    console_logger.info(f"Scores saved to: {scores_path}")

                console_logger.info(f"Final scene restored to checkpoint state.")

        # Copy final scores and renders to final_floor_plan/ directory.
        # Use final_render_dir (tracks actual last render) instead of checkpoint_render_dir
        # (which may be stale when final critique uses update_checkpoint=False).
        render_dir_to_copy = self.final_render_dir or self.checkpoint_render_dir
        if render_dir_to_copy is not None:
            final_scene_dir = self._get_final_scores_directory()
            final_scene_dir.mkdir(parents=True, exist_ok=True)

            # Copy scores.
            scores_source = render_dir_to_copy / "scores.yaml"
            if scores_source.exists():
                scores_dest = final_scene_dir / "scores.yaml"
                shutil.copy(scores_source, scores_dest)
                console_logger.info(f"Saved final scores to {scores_dest}")
            else:
                console_logger.warning(
                    f"Scores file not found at {scores_source}, cannot copy"
                )

            # Copy render images.
            render_images = list(render_dir_to_copy.glob("*.png"))
            if render_images:
                for img_path in render_images:
                    img_dest = final_scene_dir / img_path.name
                    shutil.copy(img_path, img_dest)
                console_logger.info(
                    f"Copied {len(render_images)} render images to {final_scene_dir}"
                )
            else:
                console_logger.warning(
                    f"No render images found in {render_dir_to_copy}"
                )

    async def _request_initial_design_impl(self) -> str:
        """Implementation for initial design request.

        Returns:
            Designer's report of initial design.
        """
        console_logger.info("Tool called: request_initial_design")

        # Get instruction.
        one_shot = (
            len(self.designer.tools) == 1
            and getattr(self.designer.tools[0], "name", "") == "submit_floor_plan"
        )
        instruction = self.prompt_registry.get_prompt(
            prompt_enum=(
                FloorPlanAgentPrompts.DESIGNER_ONE_SHOT_INSTRUCTION
                if one_shot
                else FloorPlanAgentPrompts.DESIGNER_INITIAL_INSTRUCTION
            ),
        )

        # Run designer.
        result = await Runner.run(
            starting_agent=self.designer,
            input=instruction,
            session=self.designer_session,
            max_turns=self.cfg.agents.designer_agent.max_turns,
            run_config=self._create_run_config(),
            timeout_seconds=agent_run_timeout_seconds(
                "designer",
                max_turns=self.cfg.agents.designer_agent.max_turns,
            ),
        )
        log_agent_usage(result=result, agent_name="DESIGNER (INITIAL FLOOR PLAN)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (INITIAL)"
            )

        return result.final_output

    async def _request_design_change_impl(self, instruction: str) -> str:
        """Implementation for design change request.

        Args:
            instruction: Changes to make based on critique.

        Returns:
            Designer's report of changes.
        """
        console_logger.info("Tool called: request_design_change")

        # Get instruction.
        full_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FloorPlanAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION,
            instruction=instruction,
        )

        # Run designer.
        result = await Runner.run(
            starting_agent=self.designer,
            input=full_instruction,
            session=self.designer_session,
            max_turns=self.cfg.agents.designer_agent.max_turns,
            run_config=self._create_run_config(),
            timeout_seconds=agent_run_timeout_seconds(
                "designer",
                max_turns=self.cfg.agents.designer_agent.max_turns,
            ),
        )
        log_agent_usage(result=result, agent_name="DESIGNER (CHANGE FLOOR PLAN)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (CHANGE)"
            )

        return result.final_output

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
        try:
            if not configured_model:
                raise RuntimeError("semantic obligation model is not configured")
            interpretations, analysis_result = await analyze_requirement_candidates(
                styled_prompt,
                literal_candidates,
                model=str(configured_model),
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
                analysis_model=str(configured_model),
            )
        except Exception as exc:
            # Step 2 is intentionally shadow-only. Preserve every source clause as
            # unclassified if the semantic route fails; enforcement will treat an
            # unclassified hard obligation as non-publishable when that gate lands.
            console_logger.exception(
                "Semantic obligation analysis failed; preserving literal candidates"
            )
            self.requirement_graph = requirement_graph_from_prompt(
                styled_prompt,
                analysis_model=(str(configured_model) if configured_model else None),
                analysis_error=f"{type(exc).__name__}: {exc}",
            )
        persist_requirement_graph(
            self.requirement_graph,
            self.logger.output_dir / "scene_requirement_graph.json",
        )
        self.semantic_ledger = load_or_initialize_semantic_ledger(
            self.logger.output_dir / "semantic_obligation_ledger.json",
            self.requirement_graph,
        )
        configured_capabilities = getattr(self.cfg, "semantic_capabilities", None)
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
            "Semantic requirement shadow graph %s captured %d literal candidates "
            "and %d obligations (analysis=%s, hash=%s)",
            self.requirement_graph.graph_id,
            len(self.requirement_graph.candidates),
            len(self.requirement_graph.requirements),
            self.requirement_graph.analysis_status,
            self.requirement_graph.content_hash,
        )

        self.blueprint = blueprint_from_prompt(
            styled_prompt,
            mode=self.mode,
            default_dimensions_m=(
                min(7.0, float(self.cfg.max_floor_plan_dim_m)),
                min(7.0, float(self.cfg.max_floor_plan_dim_m)),
            ),
            maximum_dimension_m=float(self.cfg.max_floor_plan_dim_m),
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
        persist_scene_blueprint(
            self.blueprint, self.logger.output_dir / "scene_blueprint.json"
        )
        self.house_prompt = self.blueprint.to_prompt_brief()
        console_logger.info(
            "Selected proxy candidate %s (score %.2f) as SceneBlueprint %s with "
            "%d spaces and %d connectors",
            self.candidate_tournament.winner_id,
            self.candidate_tournament.winner.scores.total,
            self.blueprint.blueprint_id,
            len(self.blueprint.spaces),
            len(self.blueprint.connectors),
        )

        # Initialize geometry cache for reusing unchanged room geometry.
        cache_dir = house_dir / ".geometry_cache"
        self._geometry_cache = GeometryCache(cache_dir=cache_dir)

        # Create agents.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)

        if self.cfg.max_critique_rounds <= 0:
            blueprint_submission = floor_plan_submission_from_blueprint(self.blueprint)
            deterministic_intent = normalize_floor_plan_submission(
                blueprint_submission,
                prompt=styled_prompt,
                mode=self.mode,
                room_dim_min=self.cfg.min_floor_plan_dim_m,
                room_dim_max=self.cfg.max_floor_plan_dim_m,
                wall_height_min=self.cfg.wall_height.min,
                wall_height_max=self.cfg.wall_height.max,
            )
            if (
                deterministic_intent.structural is not None
                and len(designer_tools) == 1
                and getattr(designer_tools[0], "name", "") == "submit_floor_plan"
            ):
                # Explicit storeys, mezzanines, platforms, and stairs already
                # have a checked local authoring path. Avoid spending a model
                # turn asking it to restate that structural schema.
                console_logger.info(
                    "Using deterministic multi-level floor-plan authoring"
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
        if not self._write_resumable_layout_checkpoint():
            raise RuntimeError(
                "Floor-plan stage did not produce a structurally valid checkpoint; "
                "the incomplete room was not exported. Resume the build to retry "
                "the layout stage."
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
            min(existing_storey_height, self.cfg.wall_height.max),
            max_total_height=self.cfg.wall_height.max,
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
                wall_height_max=self.cfg.wall_height.max,
                room_dim_min=self.cfg.min_floor_plan_dim_m,
                room_dim_max=self.cfg.max_floor_plan_dim_m,
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

    def _generate_all_room_geometries(self, output_dir: Path) -> None:
        """Generate geometry for rooms missing from the layout cache.

        Only regenerates geometry for rooms that are not in room_geometries.
        This allows per-room invalidation to work correctly - when a room's
        geometry is invalidated, only that room is regenerated on next render.

        Args:
            output_dir: Directory to save GLTF files.
        """
        replacement_space_ids = {
            mesh.space_id
            for mesh in self.layout.structural_meshes
            if mesh.replaces_room_shell
        }
        for room_spec in self.layout.room_specs:
            if room_spec.room_id in replacement_space_ids:
                # compile_structural_meshes below creates the room-compatible
                # SDF and RoomGeometry. Do not surround it with a legacy box.
                self.layout.invalidate_room_geometry(room_spec.room_id)
                continue
            # Skip rooms that already have geometry (not invalidated).
            if room_spec.room_id in self.layout.room_geometries:
                console_logger.debug(
                    f"Skipping geometry for {room_spec.room_id} (cached)"
                )
                continue

            console_logger.info(f"Generating geometry for {room_spec.room_id}")
            room_geometry = self._generate_room_geometry(
                room_spec=room_spec, output_dir=output_dir
            )
            self.layout.set_room_geometry(room_spec.room_id, room_geometry)

        # Structural assets are authored independently of the legacy room loop,
        # but must exist before ``to_drake_directive`` is called.  Compile them
        # after rooms so room-local surface sidecars can be attached to the
        # generated RoomGeometry and consumed by later placement agents.
        structural_dir = output_dir / "structural"
        if self.layout.connectors:
            self.layout.compile_connectors(structural_dir / "connectors")
        if self.layout.structural_meshes:
            self.layout.compile_structural_meshes(structural_dir / "meshes")
        if self.layout.platforms:
            self.layout.compile_platforms(structural_dir / "platforms")
        if self.layout.heightfields:
            self.layout.compile_heightfields(structural_dir / "heightfields")

        blocked_connectors = self.layout.geometrically_blocked_connectors()
        if blocked_connectors:
            raise ValueError(
                "Structural connectors fail support/headroom/width clearance: "
                + ", ".join(sorted(blocked_connectors))
                + ". Add or resize independent floor/ceiling openings, widen "
                "the connector, or revise its route."
            )

    def _export_floor_plan(self, output_dir: Path) -> None:
        """Export floor plan to .blend and .dmd.yaml files.

        Creates the final_floor_plan directory with:
        - floor_plan.blend: Blender file with PBR materials (matches renders)
        - floor_plan.dmd.yaml: Drake directive for simulation (references SDF files)

        The blend file is created from the DMD directive, which references the
        same SDF and GLTF files used in simulation and preview renders.

        Args:
            output_dir: Base directory for floor plan outputs.
        """
        # Create final floor plan directory.
        final_dir = output_dir / "final_floor_plan"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Export Drake directive first (used by both simulation and blend export).
        # Use house_dir as base for package://scene/ URIs (not final_dir subdirectory).
        directive_path = final_dir / "floor_plan.dmd.yaml"
        house_dir = self.layout.house_dir
        directive_content = self.layout.to_drake_directive(base_dir=house_dir)
        with open(directive_path, "w") as f:
            f.write(directive_content)
        console_logger.info(f"Floor plan directive saved to: {directive_path}")

        # Convert DMD to .blend for external use.
        # Pass house_dir as scene_dir for package://scene/ resolution.
        blend_path = final_dir / "floor_plan.blend"
        save_directive_as_blend(
            directive_path=directive_path,
            output_path=blend_path,
            scene_dir=house_dir,
        )
        console_logger.info(f"Floor plan .blend saved to: {blend_path}")

    def _generate_floor_geometry(
        self,
        placed_room: PlacedRoom,
        room_id: str,
        floor_material: Material,
        floor_thickness: float,
        floors_dir: Path,
        link_element: ET.Element,
    ) -> Path:
        """Generate floor GLTF and add to SDF.

        Uses geometry cache to reuse unchanged floors across iterations.

        Args:
            placed_room: The placed room with dimensions.
            room_id: Room identifier for path generation.
            floor_material: Floor material with PBR textures.
            floor_thickness: Floor thickness in meters.
            floors_dir: Directory to save floor GLTF.
            link_element: SDF link element to add floor to.

        Returns:
            Path to the generated floor GLTF file.
        """
        cache_key = floor_cache_key(
            width=placed_room.width,
            depth=placed_room.depth,
            thickness=floor_thickness,
            material=floor_material,
        )

        def create_fn(output_path: Path) -> None:
            create_floor_gltf(
                width=placed_room.width,
                depth=placed_room.depth,
                thickness=floor_thickness,
                material=floor_material,
                output_path=output_path,
                texture_scale=0.5,
                center_x=0.0,
                center_y=0.0,
                center_z=-floor_thickness / 2,
            )

        assert self._geometry_cache is not None
        floor_gltf_path = self._geometry_cache.get_or_create_floor(
            cache_key=cache_key, output_dir=floors_dir, create_fn=create_fn
        )

        # Add floor to SDF.
        floor_gltf_rel = f"../floor_plans/{room_id}/floors/floor.gltf"
        self._add_gltf_floor_visual(link_element, floor_gltf_rel)
        self._add_floor_collision(
            link_element, length=placed_room.width, width=placed_room.depth
        )

        return floor_gltf_path

    def _generate_window_frame(
        self,
        opening: Opening,
        wall_thickness: float,
        is_horizontal: bool,
        effective_wall_length: float,
        local_x: float,
        local_y: float,
        room_id: str,
        room_output_dir: Path,
        link_element: ET.Element,
    ) -> None:
        """Generate window frame mesh and add to SDF.

        Uses geometry cache to reuse unchanged windows across iterations.

        Args:
            opening: Window opening specification.
            wall_thickness: Wall thickness in meters.
            is_horizontal: Whether this is a horizontal (N/S) wall.
            effective_wall_length: Wall length (with corner extension if applicable).
            local_x: Wall X position in room coordinates.
            local_y: Wall Y position in room coordinates.
            room_id: Room identifier for path generation.
            room_output_dir: Room output directory.
            link_element: SDF link element to add window to.
        """
        cache_key = window_cache_key(
            width=opening.width,
            height=opening.height,
            depth=wall_thickness,
            is_horizontal=is_horizontal,
            shape=opening.shape.value,
        )

        def create_fn(output_path: Path) -> None:
            # Create window frame mesh (in Z-up coords, facing +Y).
            window_scene = create_window_mesh(
                width=opening.width,
                height=opening.height,
                depth=wall_thickness,
                shape=opening.shape.value,
            )

            # Z-up to Y-up transform for GLTF export.
            zup_to_yup = get_zup_to_yup_matrix()

            # Create new scene for export with transformed meshes.
            export_scene = trimesh.Scene()
            for part_name, part_mesh in window_scene.geometry.items():
                mesh_copy = part_mesh.copy()

                # For E/W walls (vertical), rotate 90° around Z in Z-up coords
                # BEFORE transforming to Y-up. This aligns window with wall.
                if not is_horizontal:
                    rotation = trimesh.transformations.rotation_matrix(
                        np.pi / 2, [0, 0, 1]  # Z is up in Z-up coordinates.
                    )
                    mesh_copy.apply_transform(rotation)

                # Transform from Z-up to Y-up for GLTF export.
                mesh_copy.apply_transform(zup_to_yup)
                export_scene.add_geometry(mesh_copy, geom_name=part_name)

            # Export window frame as GLTF.
            export_scene.export(str(output_path), file_type="gltf")

        # Create windows subdirectory.
        windows_dir = room_output_dir / "windows"
        window_subdir = windows_dir / opening.opening_id

        assert self._geometry_cache is not None
        self._geometry_cache.get_or_create_window(
            cache_key=cache_key, output_dir=window_subdir, create_fn=create_fn
        )

        # Calculate window position in local room coordinates.
        # opening.position_along_wall is LEFT EDGE, need CENTER.
        opening_center_along_wall = opening.position_along_wall + opening.width / 2
        # Window center height = sill_height + height/2.
        window_z = opening.sill_height + opening.height / 2

        # Convert position along wall to local room coords.
        if is_horizontal:
            # N/S walls: window moves along X axis.
            window_x = opening_center_along_wall - effective_wall_length / 2
            window_y = local_y
        else:
            # E/W walls: window moves along Y axis.
            window_x = local_x
            window_y = opening_center_along_wall - effective_wall_length / 2

        # Add window visual to SDF.
        window_gltf_rel = (
            f"../floor_plans/{room_id}/windows/{opening.opening_id}/window.gltf"
        )
        self._add_window_frame_visual(
            link_element=link_element,
            window_name=opening.opening_id,
            gltf_relative_path=window_gltf_rel,
            pose_x=window_x,
            pose_y=window_y,
            pose_z=window_z,
        )
        console_logger.debug(
            f"Added window frame {opening.opening_id} at "
            f"({window_x:.2f}, {window_y:.2f}, {window_z:.2f})"
        )

    def _generate_exterior_wall(
        self,
        wall: Wall,
        dimensions: WallDimensions,
        wall_openings: list[WallOpening],
        placed_room: PlacedRoom,
        offset: float,
        local_x: float,
        local_y: float,
        is_horizontal: bool,
        room_id: str,
        walls_dir: Path,
        link_element: ET.Element,
    ) -> None:
        """Generate exterior wall layer for a wall.

        Creates the outer layer of a dual-wall system for exterior walls.

        Args:
            wall: Wall specification with direction and exterior flag.
            dimensions: Wall dimensions (width, height, thickness).
            wall_openings: List of openings in the wall.
            placed_room: The placed room with dimensions.
            offset: Half wall thickness for positioning.
            local_x: Inner wall X position.
            local_y: Inner wall Y position.
            is_horizontal: Whether this is a horizontal (N/S) wall.
            room_id: Room identifier for path generation.
            walls_dir: Directory to save wall GLTFs.
            link_element: SDF link element to add wall to.
        """
        # Get exterior material.
        exterior_material = self.layout.exterior_material
        if exterior_material is None:
            exterior_material = Material.from_path(Path("materials/Plaster001_1K-JPG"))

        # Compute outer wall position (opposite offset from inner wall).
        if wall.direction == WallDirection.NORTH:
            outer_local_y = placed_room.depth / 2 + offset
            outer_local_x = local_x
        elif wall.direction == WallDirection.SOUTH:
            outer_local_y = -placed_room.depth / 2 - offset
            outer_local_x = local_x
        elif wall.direction == WallDirection.EAST:
            outer_local_x = placed_room.width / 2 + offset
            outer_local_y = local_y
        else:  # WEST
            outer_local_x = -placed_room.width / 2 - offset
            outer_local_y = local_y

        # Generate outer wall GLTF with exterior material and caching.
        outer_wall_name = f"{wall.direction.value}_wall_exterior"
        outer_wall_subdir = walls_dir / outer_wall_name
        openings_dicts = [o.to_dict() for o in wall_openings] if wall_openings else None
        cache_key = wall_cache_key(
            width=dimensions.width,
            height=dimensions.height,
            thickness=dimensions.thickness,
            material=exterior_material,
            openings=openings_dicts,
        )

        def create_exterior_wall_fn(output_path: Path) -> None:
            create_wall_gltf_with_openings(
                dimensions=dimensions,
                openings=wall_openings if wall_openings else None,
                output_path=output_path,
                uv_scale=0.5,
                material=exterior_material,
            )

        assert self._geometry_cache is not None
        self._geometry_cache.get_or_create_wall(
            cache_key=cache_key,
            output_dir=outer_wall_subdir,
            create_fn=create_exterior_wall_fn,
        )

        # Add outer wall visual with pose.
        outer_wall_gltf_rel = (
            f"../floor_plans/{room_id}/walls/{outer_wall_name}/wall.gltf"
        )
        self._add_gltf_wall_visual_with_pose(
            link_element=link_element,
            wall_name=outer_wall_name,
            gltf_relative_path=outer_wall_gltf_rel,
            pose_x=outer_local_x,
            pose_y=outer_local_y,
            pose_z=0.0,
            is_horizontal=is_horizontal,
        )

        console_logger.debug(
            f"Added exterior wall {outer_wall_name} at "
            f"({outer_local_x:.2f}, {outer_local_y:.2f})"
        )

    def _generate_single_wall(
        self,
        wall: Wall,
        placed_room: PlacedRoom,
        wall_height: float,
        wall_thickness: float,
        wall_material: Path,
        room_id: str,
        room_output_dir: Path,
        walls_dir: Path,
        link_element: ET.Element,
    ) -> WallSpec | None:
        """Generate geometry for a single wall.

        Args:
            wall: Wall specification from PlacedRoom.
            placed_room: The placed room with dimensions.
            wall_height: Wall height in meters.
            wall_thickness: Wall thickness in meters.
            wall_material: Path to wall material folder.
            room_id: Room identifier for path generation.
            room_output_dir: Room output directory.
            walls_dir: Directory to save wall GLTFs.
            link_element: SDF link element to add wall to.

        Returns:
            WallSpec for the wall, or None if wall was skipped.
        """
        # Determine wall length for this direction.
        if wall.direction in (WallDirection.NORTH, WallDirection.SOUTH):
            wall_length_dim = placed_room.width
        else:
            wall_length_dim = placed_room.depth

        # Skip walls that are entirely covered by an OPEN opening.
        if any(
            opening.opening_type == OpeningType.OPEN
            and opening.width >= wall_length_dim - 0.001  # 1mm tolerance.
            for opening in wall.openings
        ):
            console_logger.debug(
                f"Skipping wall {wall.wall_id} - fully covered by OPEN opening"
            )
            return None

        # Determine wall orientation and local position.
        offset = wall_thickness / 2
        is_horizontal = wall.direction in (WallDirection.NORTH, WallDirection.SOUTH)
        if is_horizontal:
            if wall.direction == WallDirection.NORTH:
                local_y = placed_room.depth / 2 - offset
            else:
                local_y = -placed_room.depth / 2 + offset
            local_x = 0.0
        else:
            if wall.direction == WallDirection.EAST:
                local_x = placed_room.width / 2 - offset
            else:
                local_x = -placed_room.width / 2 + offset
            local_y = 0.0

        wall_name = f"{wall.direction.value}_wall"

        # Convert openings to WallOpening format.
        wall_openings = []
        for opening in wall.openings:
            effective_height = (
                wall_height
                if opening.opening_type == OpeningType.OPEN
                else opening.height
            )
            wall_openings.append(
                WallOpening(
                    position_along_wall=opening.position_along_wall,
                    width=opening.width,
                    height=effective_height,
                    sill_height=opening.sill_height,
                    opening_type=opening.opening_type,
                    shape=opening.shape,
                )
            )

        # Determine if wall needs corner extension.
        length_override = None
        if wall.is_exterior and is_horizontal:
            length_override = wall_length_dim + wall_thickness
            console_logger.debug(
                f"Corner extension: {wall.wall_id} extended from "
                f"{wall_length_dim:.3f}m to {length_override:.3f}m"
            )

        # Create wall dimensions.
        effective_wall_length = length_override if length_override else wall_length_dim
        dimensions = WallDimensions(
            width=effective_wall_length,
            height=wall_height,
            thickness=wall_thickness,
        )

        # Generate wall GLTF with caching.
        wall_subdir = walls_dir / wall_name
        openings_dicts = [o.to_dict() for o in wall_openings] if wall_openings else None
        cache_key = wall_cache_key(
            width=dimensions.width,
            height=dimensions.height,
            thickness=dimensions.thickness,
            material=wall_material,
            openings=openings_dicts,
        )

        def create_wall_fn(output_path: Path) -> None:
            create_wall_gltf_with_openings(
                dimensions=dimensions,
                openings=wall_openings if wall_openings else None,
                output_path=output_path,
                uv_scale=0.5,
                material=wall_material,
            )

        assert self._geometry_cache is not None
        self._geometry_cache.get_or_create_wall(
            cache_key=cache_key,
            output_dir=wall_subdir,
            create_fn=create_wall_fn,
        )

        # Create WallSpec for SDF collision and SceneObject creation.
        if is_horizontal:
            bbox_width = placed_room.width
            bbox_depth = wall_thickness
        else:
            bbox_width = wall_thickness
            bbox_depth = placed_room.depth

        spec = WallSpec(
            name=wall_name,
            center_x=local_x,
            center_y=local_y,
            bbox_width=bbox_width,
            bbox_depth=bbox_depth,
            thickness=wall_thickness,
        )

        # Add wall visual to SDF.
        wall_gltf_rel = f"../floor_plans/{room_id}/walls/{wall_name}/wall.gltf"
        self._add_gltf_wall_visual_with_pose(
            link_element=link_element,
            wall_name=wall_name,
            gltf_relative_path=wall_gltf_rel,
            pose_x=local_x,
            pose_y=local_y,
            pose_z=0.0,
            is_horizontal=is_horizontal,
        )

        # Add wall collision with cutouts for doors/open connections.
        self._add_wall_collision_with_openings(
            link_element=link_element,
            wall_spec=spec,
            wall_height=wall_height,
            openings=wall_openings,
            wall_length=wall_length_dim,
            is_horizontal=is_horizontal,
        )

        # Generate window frames for WINDOW openings.
        for opening in wall.openings:
            if opening.opening_type == OpeningType.WINDOW:
                self._generate_window_frame(
                    opening=opening,
                    wall_thickness=wall_thickness,
                    is_horizontal=is_horizontal,
                    effective_wall_length=effective_wall_length,
                    local_x=local_x,
                    local_y=local_y,
                    room_id=room_id,
                    room_output_dir=room_output_dir,
                    link_element=link_element,
                )

        # Generate exterior wall layer if this is an exterior wall.
        if wall.is_exterior:
            self._generate_exterior_wall(
                wall=wall,
                dimensions=dimensions,
                wall_openings=wall_openings,
                placed_room=placed_room,
                offset=offset,
                local_x=local_x,
                local_y=local_y,
                is_horizontal=is_horizontal,
                room_id=room_id,
                walls_dir=walls_dir,
                link_element=link_element,
            )

        return spec

    def _generate_room_geometry(
        self, room_spec: RoomSpec, output_dir: Path
    ) -> RoomGeometry:
        """Generate geometry for a single room.

        Uses walls from placed_rooms which include door/window openings.
        Room geometry is generated in local coordinates (centered at origin),
        then positioned by Drake directive.

        Args:
            room_spec: Room specification.
            output_dir: Directory to save GLTF files.

        Returns:
            RoomGeometry with walls, floor, and SDF.
        """
        wall_height = self.layout.wall_height
        wall_thickness = self.cfg.wall_thickness
        floor_thickness = self.cfg.floor_thickness

        # Find the PlacedRoom for this spec.
        placed_room = None
        for pr in self.layout.placed_rooms:
            if pr.room_id == room_spec.room_id:
                placed_room = pr
                break

        if not placed_room:
            raise ValueError(
                f"No placed room found for room_id '{room_spec.room_id}'. "
                f"Ensure placement algorithm ran successfully."
            )

        from scenesmith.agent_utils.structural_geometry import ElevationProfile

        if (
            room_spec.footprint is not None
            or room_spec.floor_footprint is not None
            or room_spec.ceiling_footprint is not None
            or room_spec.floor_profile != ElevationProfile()
            or room_spec.ceiling_profile is not None
            or not room_spec.has_overhead_cover
            or abs(room_spec.yaw) > 1e-9
            or len(self.layout.levels) > 1
            or bool(self.layout.connectors)
            or any(
                portal.source_space_id == room_spec.room_id
                for portal in self.layout.portals
            )
            or any(
                heightfield.space_id == room_spec.room_id and heightfield.replaces_floor
                for heightfield in self.layout.heightfields
            )
        ):
            return self._generate_polygon_room_geometry(
                room_spec=room_spec,
                placed_room=placed_room,
                output_dir=output_dir,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                floor_thickness=floor_thickness,
            )

        # Get materials for this room.
        room_materials = self.layout.room_materials.get(room_spec.room_id)
        wall_material = Material.from_path("materials/Plaster001_1K-JPG")  # Default.
        floor_material = Material.from_path("materials/Wood094_1K-JPG")  # Default.

        if room_materials:
            if room_materials.wall_material:
                wall_material = room_materials.wall_material
            if room_materials.floor_material:
                floor_material = room_materials.floor_material

        # Create SDF structure.
        root_item = ET.Element("sdf", version="1.7", nsmap={"drake": "drake.mit.edu"})
        model_item = ET.SubElement(root_item, "model", name="room_geometry")
        link_item = ET.SubElement(model_item, "link", name="room_geometry_body_link")

        # Create subdirectories for GLTFs.
        room_output_dir = output_dir / room_spec.room_id
        walls_dir = room_output_dir / "walls"
        floors_dir = room_output_dir / "floors"
        walls_dir.mkdir(parents=True, exist_ok=True)
        floors_dir.mkdir(parents=True, exist_ok=True)

        # Generate floor geometry.
        floor_gltf_path = self._generate_floor_geometry(
            placed_room=placed_room,
            room_id=room_spec.room_id,
            floor_material=floor_material,
            floor_thickness=floor_thickness,
            floors_dir=floors_dir,
            link_element=link_item,
        )

        # Generate walls and collect wall specs.
        wall_specs_for_objects: list[WallSpec] = []
        for wall in placed_room.walls:
            spec = self._generate_single_wall(
                wall=wall,
                placed_room=placed_room,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                wall_material=wall_material,
                room_id=room_spec.room_id,
                room_output_dir=room_output_dir,
                walls_dir=walls_dir,
                link_element=link_item,
            )
            if spec is not None:
                wall_specs_for_objects.append(spec)

        # Save room geometry SDF (includes floor and walls).
        sdf_output_dir = self.logger.output_dir / "room_geometry"
        room_geometry_path = self.logger.log_sdf(
            name=f"room_geometry_{room_spec.room_id}",
            sdf_tree=ET.ElementTree(root_item),
            output_dir=sdf_output_dir,
        )

        # Create wall objects.
        walls = self._create_wall_objects(
            wall_specs=wall_specs_for_objects, wall_height=wall_height
        )
        wall_normals = compute_wall_normals(walls=walls)

        # Create floor object. Floor is part of room geometry SDF, not standalone.
        floor_object = SceneObject(
            object_id=UniqueID(f"floor_{room_spec.room_id}"),
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Floor surface",
            transform=RigidTransform(),
            geometry_path=floor_gltf_path,
            sdf_path=None,
            bbox_min=np.array(
                [-placed_room.width / 2, -placed_room.depth / 2, -floor_thickness]
            ),
            bbox_max=np.array([placed_room.width / 2, placed_room.depth / 2, 0.0]),
            immutable=True,
        )

        # Compute openings data for physics validation and label rendering.
        openings = compute_openings_data(
            placed_room=placed_room,
            wall_height=wall_height,
            door_clearance_distance=self.cfg.clearance_zones.door_clearance_distance,
            window_clearance_distance=self.cfg.clearance_zones.window_clearance_distance,
        )

        return RoomGeometry(
            sdf_tree=ET.ElementTree(root_item),
            sdf_path=room_geometry_path,
            walls=walls,
            floor=floor_object,
            wall_normals=wall_normals,
            # RoomGeometry: length=X-dim, width=Y-dim (matches RoomSpec convention).
            # PlacedRoom: width=X-dim, depth=Y-dim.
            width=placed_room.depth,
            length=placed_room.width,
            wall_height=wall_height,
            has_overhead_cover=room_spec.has_overhead_cover,
            wall_thickness=wall_thickness,
            openings=openings,
        )

    def _generate_polygon_room_geometry(
        self,
        *,
        room_spec: RoomSpec,
        placed_room: PlacedRoom,
        output_dir: Path,
        wall_height: float,
        wall_thickness: float,
        floor_thickness: float,
    ) -> RoomGeometry:
        """Generate an arbitrary polygon room without cardinal-wall assumptions."""

        from scenesmith.agent_utils.structural_compiler import (
            compile_polygon_space,
            triangle_group_mesh,
            write_compiled_structure,
        )
        from scenesmith.agent_utils.structural_geometry import Footprint2D, SurfaceRole

        source_footprint = room_spec.footprint or Footprint2D.rectangle(
            room_spec.length, room_spec.width
        )
        min_x, min_y, max_x, max_y = source_footprint.bounds
        footprint_width = max_x - min_x
        footprint_depth = max_y - min_y
        if not math.isclose(
            footprint_width, room_spec.length, rel_tol=0.0, abs_tol=1e-6
        ) or not math.isclose(
            footprint_depth, room_spec.width, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"Room '{room_spec.room_id}' footprint bounds are "
                f"{footprint_width:g} × {footprint_depth:g}, but length/width "
                f"are {room_spec.length:g} × {room_spec.width:g}"
            )

        footprint = source_footprint.centered_on_bounds()
        floor_footprint = (
            room_spec.floor_footprint or source_footprint
        ).centered_on_bounds()
        ceiling_footprint = (
            room_spec.ceiling_footprint or source_footprint
        ).centered_on_bounds()
        authored_portals = [
            portal
            for portal in self.layout.portals
            if portal.source_space_id == room_spec.room_id
            and portal.boundary_loop_index is not None
        ]
        authored_ids = {portal.portal_id for portal in authored_portals}
        legacy_portals = legacy_openings_to_boundary_portals(
            room_spec, placed_room, wall_height
        )
        compiled = compile_polygon_space(
            structure_id=f"room_geometry_{room_spec.room_id}",
            footprint=footprint,
            floor_footprint=floor_footprint,
            ceiling_footprint=ceiling_footprint,
            include_floor=not any(
                heightfield.space_id == room_spec.room_id and heightfield.replaces_floor
                for heightfield in self.layout.heightfields
            ),
            include_ceiling=room_spec.has_overhead_cover,
            floor_profile=room_spec.floor_profile,
            ceiling_profile=room_spec.ceiling_profile,
            wall_height=wall_height,
            floor_thickness=floor_thickness,
            ceiling_thickness=floor_thickness,
            portals=[
                *authored_portals,
                *(
                    portal
                    for portal in legacy_portals
                    if portal.portal_id not in authored_ids
                ),
            ],
        )
        room_materials = self.layout.room_materials.get(room_spec.room_id)
        floor_material = (
            room_materials.floor_material
            if room_materials is not None and room_materials.floor_material is not None
            else Material.from_path("materials/Wood094_1K-JPG")
        )
        floor_overlays = (
            {
                "floor_finish": (
                    triangle_group_mesh(
                        compiled.visual_mesh,
                        compiled.triangle_groups["floor_top"],
                        translation=(0.0, 0.0, 0.0015),
                    ),
                    floor_material,
                    floor_material.texture_scale or 0.5,
                )
            }
            if compiled.triangle_groups.get("floor_top")
            else None
        )
        paths = write_compiled_structure(
            compiled,
            output_dir / room_spec.room_id / "structural",
            model_name="room_geometry",
            link_name="room_geometry_body_link",
            visual_overlays=floor_overlays,
        )

        wall_objects: list[SceneObject] = []
        wall_normals: dict[str, np.ndarray] = {}
        for patch in compiled.surfaces:
            if SurfaceRole.BOUNDARY not in patch.surface.roles:
                continue
            start, end = patch.boundary[0], patch.boundary[1]
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            yaw = math.atan2(end[1] - start[1], end[0] - start[0])
            z_values = [point[2] for point in patch.boundary]
            z_min, z_max = min(z_values), max(z_values)
            name = patch.surface.surface_id
            wall_objects.append(
                SceneObject(
                    object_id=UniqueID(name),
                    object_type=ObjectType.WALL,
                    name=name,
                    description="Arbitrary structural boundary wall",
                    transform=RigidTransform(
                        RollPitchYaw(0.0, 0.0, yaw),
                        [
                            (start[0] + end[0]) / 2.0,
                            (start[1] + end[1]) / 2.0,
                            (z_min + z_max) / 2.0,
                        ],
                    ),
                    bbox_min=np.array(
                        [-length / 2.0, -wall_thickness / 2.0, -(z_max - z_min) / 2.0]
                    ),
                    bbox_max=np.array(
                        [length / 2.0, wall_thickness / 2.0, (z_max - z_min) / 2.0]
                    ),
                    metadata={"structural_surface_id": name},
                    immutable=True,
                )
            )
            wall_normals[name] = np.asarray(patch.normal[:2])

        floor_indices = compiled.triangle_groups["floor_top"]
        floor_vertices = [
            compiled.visual_mesh.vertices[vertex_index]
            for triangle_index in floor_indices
            for vertex_index in compiled.visual_mesh.triangles[triangle_index]
        ]
        floor_object = SceneObject(
            object_id=UniqueID(f"floor_{room_spec.room_id}"),
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Polygonal floor support surface",
            transform=RigidTransform(),
            geometry_path=paths.mesh_path,
            bbox_min=np.array(
                [
                    min(point[0] for point in floor_vertices),
                    min(point[1] for point in floor_vertices),
                    min(point[2] for point in floor_vertices) - floor_thickness,
                ]
            ),
            bbox_max=np.array(
                [
                    max(point[0] for point in floor_vertices),
                    max(point[1] for point in floor_vertices),
                    max(point[2] for point in floor_vertices),
                ]
            ),
            metadata={
                "structural_surface_id": f"room_geometry_{room_spec.room_id}_floor"
            },
            immutable=True,
        )

        return RoomGeometry(
            sdf_tree=ET.parse(paths.sdf_path),
            sdf_path=paths.sdf_path,
            walls=wall_objects,
            floor=floor_object,
            wall_normals=wall_normals,
            width=footprint_depth,
            length=footprint_width,
            wall_height=wall_height,
            has_overhead_cover=room_spec.has_overhead_cover,
            wall_thickness=wall_thickness,
            openings=compute_openings_data(
                placed_room=placed_room,
                wall_height=wall_height,
                door_clearance_distance=self.cfg.clearance_zones.door_clearance_distance,
                window_clearance_distance=(
                    self.cfg.clearance_zones.window_clearance_distance
                ),
            ),
            footprint=footprint,
            floor_footprint=floor_footprint,
            ceiling_footprint=ceiling_footprint,
            floor_profile=room_spec.floor_profile,
            ceiling_profile=room_spec.ceiling_profile,
            structural_surfaces=[patch.surface for patch in compiled.surfaces],
            structural_surface_path=paths.surfaces_path,
        )

    @staticmethod
    def _get_wall_specifications(
        length: float, width: float, wall_thickness: float = 0.05
    ) -> list[WallSpec]:
        """Get wall specifications for a rectangular room.

        Args:
            length: Room length in the x-direction (meters).
            width: Room width in the y-direction (meters).
            wall_thickness: Thickness of walls in meters.

        Returns:
            List of WallSpec objects defining all four walls.
        """
        half_length = length / 2.0
        half_width = width / 2.0

        return [
            WallSpec(
                name="left_wall",
                center_x=-half_length,
                center_y=0.0,
                bbox_width=wall_thickness,
                bbox_depth=width,
                thickness=wall_thickness,
            ),
            WallSpec(
                name="right_wall",
                center_x=half_length,
                center_y=0.0,
                bbox_width=wall_thickness,
                bbox_depth=width,
                thickness=wall_thickness,
            ),
            WallSpec(
                name="back_wall",
                center_x=0.0,
                center_y=-half_width,
                bbox_width=length,
                bbox_depth=wall_thickness,
                thickness=wall_thickness,
            ),
            WallSpec(
                name="front_wall",
                center_x=0.0,
                center_y=half_width,
                bbox_width=length,
                bbox_depth=wall_thickness,
                thickness=wall_thickness,
            ),
        ]

    @staticmethod
    def _add_gltf_wall_visual_with_pose(
        link_element: ET.Element,
        wall_name: str,
        gltf_relative_path: str,
        pose_x: float,
        pose_y: float,
        pose_z: float,
        is_horizontal: bool,
    ) -> None:
        """Add GLTF wall visual to SDF link element with pose.

        Args:
            link_element: SDF link element to add visual to.
            wall_name: Name of the wall.
            gltf_relative_path: Relative path to GLTF file.
            pose_x: X position of wall center.
            pose_y: Y position of wall center.
            pose_z: Z position of wall center.
            is_horizontal: True for north/south walls, False for east/west walls.
        """
        visual = ET.SubElement(link_element, "visual", name=f"{wall_name}_visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = gltf_relative_path

        # Add pose element.
        # Wall meshes are created centered at origin along X axis.
        # For north/south walls (horizontal), no rotation needed.
        # For east/west walls (vertical), rotate 90° around Z.
        pose = ET.SubElement(visual, "pose")
        if is_horizontal:
            pose.text = f"{pose_x} {pose_y} {pose_z} 0 0 0"
        else:
            # Rotate 90° around Z axis for vertical walls.
            pose.text = f"{pose_x} {pose_y} {pose_z} 0 0 1.5708"

    @staticmethod
    def _add_window_frame_visual(
        link_element: ET.Element,
        window_name: str,
        gltf_relative_path: str,
        pose_x: float,
        pose_y: float,
        pose_z: float,
    ) -> None:
        """Add GLTF window frame visual to SDF link element with pose.

        Args:
            link_element: SDF link element to add visual to.
            window_name: Name of the window.
            gltf_relative_path: Relative path to GLTF file.
            pose_x: X position of window center in local room coords.
            pose_y: Y position of window center in local room coords.
            pose_z: Z position of window center in local room coords.
        """
        visual = ET.SubElement(link_element, "visual", name=f"{window_name}_visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = gltf_relative_path

        # Window mesh rotation is baked in during GLTF export.
        pose = ET.SubElement(visual, "pose")
        pose.text = f"{pose_x} {pose_y} {pose_z} 0 0 0"

    def _create_wall_objects(
        self, wall_specs: list[WallSpec], wall_height: float
    ) -> list[SceneObject]:
        """Create wall objects from specifications.

        Args:
            wall_specs: Wall specifications defining wall geometry.
            wall_height: Wall height in the z-direction (meters).

        Returns:
            List of wall SceneObjects with proper transforms and bounding boxes.
        """
        walls = []

        for spec in wall_specs:
            # Create transform at wall center.
            transform = RigidTransform(
                p=[spec.center_x, spec.center_y, wall_height / 2.0]
            )

            # Bounding box in object frame (centered at origin).
            bbox_min = np.array(
                [-spec.bbox_width / 2.0, -spec.bbox_depth / 2.0, -wall_height / 2.0]
            )
            bbox_max = np.array(
                [spec.bbox_width / 2.0, spec.bbox_depth / 2.0, wall_height / 2.0]
            )

            wall_obj = SceneObject(
                object_id=UniqueID(spec.name),
                object_type=ObjectType.WALL,
                name=spec.name,
                description=f"Room {spec.name}",
                transform=transform,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                immutable=True,
            )
            walls.append(wall_obj)

        return walls

    @staticmethod
    def _add_gltf_floor_visual(
        link_element: ET.Element, gltf_relative_path: str
    ) -> None:
        """Add GLTF floor visual to SDF link element.

        Args:
            link_element: SDF link element to add visual to.
            gltf_relative_path: Relative path to GLTF file.
        """
        visual = ET.SubElement(link_element, "visual", name="floor_visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = gltf_relative_path

    @staticmethod
    def _add_floor_collision(
        link_element: ET.Element, length: float, width: float
    ) -> None:
        """Add floor collision geometry to SDF link element.

        Args:
            link_element: SDF link element to add collision to.
            length: Floor length in meters.
            width: Floor width in meters.
        """
        collision = ET.SubElement(link_element, "collision", name="floor_collision")
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = f"{length} {width} 0.1"
        pose = ET.SubElement(collision, "pose")
        pose.text = "0 0 -0.05 0 0 0"

    @staticmethod
    def _add_wall_collision(
        link_element: ET.Element, wall_spec: WallSpec, wall_height: float
    ) -> None:
        """Add wall collision geometry to SDF link element.

        Args:
            link_element: SDF link element to add collision to.
            wall_spec: Wall specification.
            wall_height: Wall height in meters.
        """
        collision = ET.SubElement(
            link_element, "collision", name=f"{wall_spec.name}_collision"
        )
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = f"{wall_spec.bbox_width} {wall_spec.bbox_depth} {wall_height}"
        pose = ET.SubElement(collision, "pose")
        pose.text = (
            f"{wall_spec.center_x} {wall_spec.center_y} {wall_height / 2.0} 0 0 0"
        )

    @staticmethod
    def _add_wall_collision_with_openings(
        link_element: ET.Element,
        wall_spec: WallSpec,
        wall_height: float,
        openings: list[WallOpening],
        wall_length: float,
        is_horizontal: bool,
    ) -> None:
        """Add wall collision geometry with cutouts for doors and open connections.

        Creates multiple collision boxes that avoid door/open sections, allowing
        passage through doors while maintaining solid collision for windows.

        Args:
            link_element: SDF link element to add collision to.
            wall_spec: Wall specification.
            wall_height: Wall height in meters.
            openings: List of openings in this wall.
            wall_length: Full length of the wall (width or depth of room).
            is_horizontal: True for NORTH/SOUTH walls, False for EAST/WEST.
        """
        # Filter to only passable openings (doors and open connections, not windows).
        passable_openings = [
            o for o in openings if o.opening_type != OpeningType.WINDOW
        ]

        if not passable_openings:
            # No doors/open connections - use single solid box.
            StatefulFloorPlanAgent._add_wall_collision(
                link_element, wall_spec, wall_height
            )
            return

        # Sort openings by position along wall.
        sorted_openings = sorted(passable_openings, key=lambda o: o.position_along_wall)

        # Compute solid segments between openings.
        # Each segment is (start_pos, end_pos) along the wall.
        segments: list[tuple[float, float]] = []
        current_pos = 0.0

        for opening in sorted_openings:
            if opening.position_along_wall > current_pos:
                # Solid segment before this opening.
                segments.append((current_pos, opening.position_along_wall))
            current_pos = opening.position_along_wall + opening.width

        # Final segment after last opening.
        if current_pos < wall_length:
            segments.append((current_pos, wall_length))

        # Create collision box for each solid segment.
        for i, (start, end) in enumerate(segments):
            segment_length = end - start
            if segment_length <= 0.001:
                # Skip very small segments (floating point artifacts).
                continue

            # Segment center relative to wall center.
            segment_center_along_wall = start + segment_length / 2 - wall_length / 2

            collision = ET.SubElement(
                link_element, "collision", name=f"{wall_spec.name}_collision_{i}"
            )
            geometry = ET.SubElement(collision, "geometry")
            box = ET.SubElement(geometry, "box")
            size = ET.SubElement(box, "size")

            if is_horizontal:
                # NORTH/SOUTH wall: segments run along X axis.
                size.text = f"{segment_length} {wall_spec.bbox_depth} {wall_height}"
                pose_x = wall_spec.center_x + segment_center_along_wall
                pose_y = wall_spec.center_y
            else:
                # EAST/WEST wall: segments run along Y axis.
                size.text = f"{wall_spec.bbox_width} {segment_length} {wall_height}"
                pose_x = wall_spec.center_x
                pose_y = wall_spec.center_y + segment_center_along_wall

            pose = ET.SubElement(collision, "pose")
            pose.text = f"{pose_x} {pose_y} {wall_height / 2.0} 0 0 0"
