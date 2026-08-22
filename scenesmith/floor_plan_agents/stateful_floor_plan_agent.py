"""Stateful floor plan agent using planner/designer/critic workflow.

This module implements the floor plan agent trio for designing house layouts
with rooms, doors, windows, and materials, then generates the geometry.
"""

import logging

from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool
from omegaconf import DictConfig

from scenesmith.agent_utils.blender import BlenderServer
from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.design.design_system import load_design_system_from_env
from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.runtime.base_stateful_agent import (
    BaseStatefulAgent,
    log_agent_usage,
)
from scenesmith.agent_utils.runtime.scoring import FloorPlanCritiqueWithScores
from scenesmith.agent_utils.runtime.workflow_tools import WorkflowTools
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.agent_utils.scene.scene_candidates import CandidateTournament
from scenesmith.agent_utils.semantics.requirements.requirement_blueprint_compiler import (
    compile_requirement_blueprint,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint
from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    analyze_requirement_candidates,
)
from scenesmith.floor_plan_agents.base_floor_plan_agent import BaseFloorPlanAgent
from scenesmith.floor_plan_agents.mixins import (
    layout_generation as layout_generation_mixin,
)
from scenesmith.floor_plan_agents.mixins.checkpoint_workflow import (
    FloorPlanCheckpointWorkflowMixin,
)
from scenesmith.floor_plan_agents.mixins.geometry_export import (
    FloorPlanGeometryExportMixin,
)
from scenesmith.floor_plan_agents.mixins.geometry_orchestration import (
    FloorPlanGeometryOrchestrationMixin,
)
from scenesmith.floor_plan_agents.mixins.layout_generation import (
    FloorPlanLayoutGenerationMixin,
)
from scenesmith.floor_plan_agents.mixins.wall_geometry import FloorPlanWallGeometryMixin
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.tools.geometry_cache import GeometryCache
from scenesmith.floor_plan_agents.tools.vision_tools import FloorPlanVisionTools
from scenesmith.prompts.registry import FloorPlanAgentPrompts
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


class StatefulFloorPlanAgent(
    FloorPlanCheckpointWorkflowMixin,
    FloorPlanLayoutGenerationMixin,
    FloorPlanGeometryOrchestrationMixin,
    FloorPlanWallGeometryMixin,
    FloorPlanGeometryExportMixin,
    BaseStatefulAgent,
    BaseFloorPlanAgent,
):
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

    async def generate_house_layout(
        self,
        prompt: str,
        output_dir: Path,
    ) -> HouseLayout:
        """Preserve established module-level dependency injection hooks."""

        layout_generation_mixin.load_design_system_from_env = (
            load_design_system_from_env
        )
        layout_generation_mixin.analyze_requirement_candidates = (
            analyze_requirement_candidates
        )
        layout_generation_mixin.log_agent_usage = log_agent_usage
        layout_generation_mixin.compile_requirement_blueprint = (
            compile_requirement_blueprint
        )
        return await FloorPlanLayoutGenerationMixin.generate_house_layout(
            self,
            prompt,
            output_dir,
        )

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

    def _construction_room_dim_max(self) -> float:
        """Return a validation envelope that can never shrink accepted intent."""

        configured = float(self.cfg.max_floor_plan_dim_m)
        if self.blueprint is None:
            return configured
        return max(
            configured,
            *(max(space.dimensions_m) for space in self.blueprint.spaces),
        )

    def _construction_wall_height_max(self) -> float:
        """Return a shell envelope that contains the accepted semantic levels."""

        configured = float(self.cfg.wall_height.max)
        if self.blueprint is None:
            return configured
        lowest = min(level.elevation_m for level in self.blueprint.levels)
        semantic_height = (
            max(
                level.elevation_m + level.clear_height_m
                for level in self.blueprint.levels
            )
            - lowest
        )
        opening_height = max(
            (
                opening.sill_height_m + opening.height_m
                for opening in self.blueprint.openings
            ),
            default=0.0,
        )
        return max(configured, semantic_height, opening_height)

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
            wall_height_max=self._construction_wall_height_max(),
            room_dim_min=self.cfg.min_floor_plan_dim_m,
            room_dim_max=self._construction_room_dim_max(),
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
            wall_height_max=self._construction_wall_height_max(),
            room_dim_min=self.cfg.min_floor_plan_dim_m,
            room_dim_max=self._construction_room_dim_max(),
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
