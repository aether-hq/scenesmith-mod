"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import logging
import math

from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool
from omegaconf import DictConfig

from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.rendering.rendering_manager import RenderingManager
from scenesmith.agent_utils.runtime.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.runtime.scoring import ManipulandCritiqueWithScores
from scenesmith.agent_utils.runtime.workflow_tools import WorkflowTools
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    AgentType,
    ObjectType,
    SupportSurface,
    UniqueID,
)
from scenesmith.agent_utils.scene.scene_analyzer import (
    FurnitureSelection,
    SceneAnalyzer,
)
from scenesmith.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.manipuland_agents.tools.vision_tools import ManipulandVisionTools
from scenesmith.prompts.registry import ManipulandAgentPrompts
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)

from scenesmith.manipuland_agents.mixins import dense_assets as dense_assets_mixin
from scenesmith.manipuland_agents.mixins.dense_assets import DenseBookRowAssetMixin
from scenesmith.manipuland_agents.mixins.dense_placement import (
    DenseBookRowPlacementMixin,
)
from scenesmith.manipuland_agents.mixins.template_recovery import (
    ManipulandTemplateRecoveryMixin,
)
from scenesmith.manipuland_agents.mixins.workflow import ManipulandAgentWorkflowMixin


class StatefulManipulandAgent(
    DenseBookRowAssetMixin,
    DenseBookRowPlacementMixin,
    ManipulandTemplateRecoveryMixin,
    ManipulandAgentWorkflowMixin,
    BaseStatefulAgent,
    BaseManipulandAgent,
):
    """Manipuland placement with planner/designer/critic agents per furniture.

    Workflow:
    1. Initial analysis: Identify which furniture to populate
    2. Per-furniture loop: Create fresh agents for each furniture surface
    3. Per-furniture workflow: Planner coordinates designer/critic
    4. Agent-driven termination: Planner decides when surface is complete
    """

    @classmethod
    def _physically_invalid_dense_book_row_ids(
        cls,
        scene: RoomScene,
        cfg: DictConfig,
        *,
        row_ids: set[UniqueID] | None = None,
    ) -> set[UniqueID]:
        """Preserve the module-level collision hook used by integrations/tests."""

        dense_assets_mixin.compute_scene_collisions = compute_scene_collisions
        return DenseBookRowAssetMixin._physically_invalid_dense_book_row_ids.__func__(
            cls,
            scene,
            cfg,
            row_ids=row_ids,
        )

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.MANIPULAND

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        num_workers: int = 1,
        render_allocation: RenderAllocation | None = None,
    ):
        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
        )
        # Initialize manipuland-specific base class.
        BaseManipulandAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
            articulated_server_host=articulated_server_host,
            articulated_server_port=articulated_server_port,
            materials_server_host=materials_server_host,
            materials_server_port=materials_server_port,
            num_workers=num_workers,
            render_allocation=render_allocation,
        )

        # Initialize pending images for image injection during critique.
        self.pending_images: list[dict[str, Any]] = []

        # Current furniture selection context (set per-furniture in workflow).
        self.current_furniture_selection: FurnitureSelection | None = None

        # Context image for manipuland designer initialization (per-furniture).
        self.manipuland_context_image_path: Path | None = None

    def _render_furniture_for_context(self) -> Path:
        """Render furniture with clean angled front view for context image input.

        Uses furniture_selection mode with empty annotate_object_types to get
        a clean render without any labels, bounding boxes, or coordinate overlays.
        For articulated furniture, opens joints to show interior surfaces.
        Includes context furniture (e.g., chairs around a table) for spatial reference.

        Uses adaptive camera elevation based on furniture type:
        - Tables (1 surface): High elevation (60°) - looking down at surface
        - Shelves (multiple surfaces): Low elevation (30°) - see all levels from front

        Camera is positioned to view the furniture's front face (+Y in local frame),
        accounting for the furniture's world rotation.

        Special case for floor: Renders top-down view of entire room with all
        furniture visible, similar to observe_scene. This provides spatial context
        for floor item placement (rugs, floor lamps, etc.).

        Returns:
            Path to directory containing rendered images.
        """
        furniture = self.scene.get_object(self.current_furniture_id)

        # Special case: Floor needs top-down view of entire room with all furniture.
        # This provides spatial context for floor item placement.
        if furniture.object_type == ObjectType.FLOOR:
            # Include all furniture objects for room context.
            all_furniture_ids = [
                obj.object_id
                for obj in self.scene.objects.values()
                if obj.object_type == ObjectType.FURNITURE
            ]
            return self.rendering_manager.render_scene(
                scene=self.scene,
                blender_server=self.blender_server,
                include_objects=[self.current_furniture_id] + all_furniture_ids,
                exclude_room_geometry=False,  # Include floor/walls for context
                rendering_mode="furniture_selection",  # Disables grid/frame
                annotate_object_types=[],  # Disables all labels/bboxes
                render_name=f"context_input_{self.current_furniture_id}",
                # Top-down view for floor context.
                include_vertical_views=True,  # Include top view
                override_side_view_count=0,  # No side views, just top
            )

        # Get context furniture IDs from current selection.
        context_ids = (
            self.current_furniture_selection.context_furniture_ids
            if self.current_furniture_selection
            else []
        )

        # Include current furniture + validated context furniture (same pattern as
        # observe_scene).
        valid_context_ids = [
            ctx_id for ctx_id in context_ids if ctx_id in self.scene.objects
        ]
        include_objects = [self.current_furniture_id] + valid_context_ids

        # Check if furniture is articulated (has doors/drawers).
        is_articulated = furniture.metadata.get("is_articulated", False)

        # Determine elevation based on furniture type (number of support surfaces).
        # Tables with 1 surface benefit from high angle looking down at surface.
        # Shelves with multiple surfaces need low angle to see all levels.
        num_surfaces = (
            len(furniture.support_surfaces) if furniture.support_surfaces else 1
        )
        if num_surfaces == 1:
            elevation = 60.0  # High angle - looking down at table surface
        else:
            elevation = 30.0  # Low angle - see all shelf levels from front

        # Calculate camera azimuth to view the furniture's front face.
        # Furniture "front" is +Y in local frame. We need to find where that
        # points in world frame and position the camera there.
        # For a Z-rotation (yaw) of θ, the camera should be at azimuth = 90° + θ.
        rotation_matrix = furniture.transform.rotation().matrix()
        # Extract yaw (Z rotation) from rotation matrix: atan2(R[1,0], R[0,0]).
        yaw_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # Camera azimuth: 90° (front at +Y) + furniture yaw rotation.
        front_azimuth = 90.0 + math.degrees(yaw_rad)

        return self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=include_objects,
            exclude_room_geometry=True,  # Furniture only, no floor/walls
            rendering_mode="furniture_selection",  # Disables grid/frame
            annotate_object_types=[],  # Disables all labels/bboxes
            articulated_open=is_articulated,  # Open joints to show interior surfaces
            context_furniture_ids=valid_context_ids,  # For proper visibility in render
            render_name=f"context_input_{self.current_furniture_id}",
            # Render single angled view from furniture's front face.
            include_vertical_views=False,  # No pure top/bottom views
            override_side_view_count=1,  # Single angled view
            side_view_start_azimuth_degrees=front_azimuth,  # Front of furniture
            side_view_elevation_degrees=elevation,  # Adaptive elevation
        )

    def _get_furniture_dimensions(self, furniture) -> str:
        """Compute human-readable furniture dimensions from bbox.

        Args:
            furniture: SceneObject with bbox_min and bbox_max.

        Returns:
            Human-readable dimensions string.
        """
        if furniture.bbox_min is None or furniture.bbox_max is None:
            return "dimensions unknown"

        dims = furniture.bbox_max - furniture.bbox_min
        width, depth, height = dims[0], dims[1], dims[2]
        return f"{width:.2f}m wide × {depth:.2f}m deep × {height:.2f}m tall"

    def _generate_manipuland_context_image(self) -> Path | None:
        """Generate context image for manipuland placement.

        Renders the furniture and uses image editing API to add suggested objects.
        This provides visual guidance for the manipuland designer agent.

        Returns:
            Path to generated context image, or None if generation fails or disabled.
        """
        if not self.cfg.context_image_generation.enabled:
            return None

        render_dir = self._render_furniture_for_context()

        selection = self.current_furniture_selection
        furniture = self.scene.get_object(selection.furniture_id)

        # Select correct image based on furniture type.
        # Floor uses top-down view; other furniture uses angled front view.
        if furniture.object_type == ObjectType.FLOOR:
            render = render_dir / "0_top.png"
        else:
            render = render_dir / "0_side.png"

        try:
            return self.asset_manager.image_generator.generate_manipuland_context_image(
                reference_image_path=render,
                furniture_description=furniture.description,
                furniture_dimensions=self._get_furniture_dimensions(furniture),
                suggested_items=selection.suggested_items,
                prompt_constraints=selection.prompt_constraints,
                style_notes=selection.style_notes,
                output_path=render_dir / "context_edited.png",
            )
        except Exception as e:
            console_logger.warning(f"Context image generation failed: {e}")
            return None

    def _get_context_image_path(self) -> Path | None:
        """Get the AI-generated context image for initial design.

        Returns:
            Path to manipuland context image if available, None otherwise.
        """
        return self.manipuland_context_image_path

    def _create_designer_tools(
        self,
        current_furniture_id: UniqueID,
        support_surfaces: dict[str, SupportSurface],
    ) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Args:
            current_furniture_id: ID of furniture being populated.
            support_surfaces: Dictionary mapping surface_id to SupportSurface.

        Returns:
            List of tools for the designer agent.
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
        )
        self.manipuland_tools = ManipulandTools(
            scene=self.scene,
            asset_manager=self.asset_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            support_surfaces=support_surfaces,
        )
        workflow_tools = WorkflowTools()

        tools = [
            *vision_tools.tools.values(),
            *self.manipuland_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]
        # These two reads are deterministic and are injected into the initial
        # instruction below. Removing them saves a model turn and prevents CLI
        # models from confusing SceneSmith function tools with Claude Code's own
        # disabled filesystem tools.
        core_tools = {
            "generate_manipuland_assets",
            "place_manipuland_on_surface",
            "move_manipuland",
            "remove_manipuland",
            "check_physics",
        }
        filtered = [tool for tool in tools if tool.name in core_tools]
        console_logger.info(
            "Manipuland designer tool surface reduced from %d to %d: %s",
            len(tools),
            len(filtered),
            sorted(tool.name for tool in filtered),
        )
        return filtered

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Inject deterministic local state before the first designer turn."""
        scene_state = self.manipuland_tools._get_current_scene_state_impl()
        available_assets = self.manipuland_tools._list_available_assets_impl()
        enriched = (
            f"{instruction}\n\n"
            "<PRELOADED_CURRENT_SCENE_STATE>\n"
            f"{scene_state}\n"
            "</PRELOADED_CURRENT_SCENE_STATE>\n\n"
            "<PRELOADED_AVAILABLE_ASSETS>\n"
            f"{available_assets}\n"
            "</PRELOADED_AVAILABLE_ASSETS>\n\n"
            "The two read-only results above were computed locally and are current. "
            "Do not ask for them again. Your first response must call one of the "
            "available SceneSmith mutation tools."
        )
        return super()._build_initial_design_input(enriched)

    def _create_designer_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create designer agent with furniture-specific context.

        Args:
            tools: Tools to provide to the designer.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = ManipulandAgentPrompts[designer_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            has_reference_image=self.manipuland_context_image_path is not None,
        )

    def _create_critic_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create critic agent with furniture-specific context.

        Args:
            tools: Tools to provide to the critic.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured critic agent with structured output.
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = ManipulandAgentPrompts[critic_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=ManipulandCritiqueWithScores,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create planner agent with furniture-specific context.

        Args:
            tools: Tools to provide to the planner.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = ManipulandAgentPrompts[planner_config.prompt]
        single_threshold = self.cfg.reset_single_category_threshold
        total_threshold = self.cfg.reset_total_sum_threshold

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=single_threshold,
            reset_total_sum_threshold=total_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _create_tools_for_furniture(
        self, furniture_id: UniqueID
    ) -> tuple[list[FunctionTool], list[FunctionTool], list[FunctionTool]]:
        """Create tools for planner, designer, and critic.

        Args:
            furniture_id: ID of current furniture.

        Returns:
            Tuple of (planner_tools, designer_tools, critic_tools).
        """
        # Get all support surfaces for this furniture.
        furniture = self.scene.get_object(furniture_id)
        if not furniture or not furniture.support_surfaces:
            raise ValueError(f"Furniture {furniture_id} has no support surfaces")

        # Build dict mapping surface_id strings to SupportSurface objects.
        support_surfaces = {
            str(surface.surface_id): surface for surface in furniture.support_surfaces
        }

        # Create designer tools using base class helper method.
        # This ensures consistency with furniture agent architecture and includes
        # WorkflowTools for task management.
        designer_tools = self._create_designer_tools(
            current_furniture_id=furniture_id, support_surfaces=support_surfaces
        )

        # Planner gets all designer tools (same access).
        planner_tools = designer_tools

        # Create critic tools using helper method.
        critic_tools = self._create_critic_tools(furniture_id=furniture_id)

        return planner_tools, designer_tools, critic_tools

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Manipuland-specific initial design instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dict with has_reference_image flag.
        """
        return {
            "has_reference_image": self.manipuland_context_image_path is not None,
        }

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Manipuland-specific design change instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Manipuland-specific critic instruction prompt.
        """
        return ManipulandAgentPrompts.MANIPULAND_CRITIC_RUNNER_INSTRUCTION

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for manipuland tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        self.manipuland_tools.set_noise_profile(mode)

    def _create_critic_tools(self, furniture_id: UniqueID) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Args:
            furniture_id: ID of furniture being critiqued (for context rendering).

        Returns:
            List of tools for the critic (read-only scene validation tools).
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        # Create vision tools for critic (read-only operations).
        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
        )

        # Critic gets read-only tools (observe only).
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            self.manipuland_tools.tools["get_current_scene_state"],
        ]

    def _setup_furniture_context(self, furniture_selection: FurnitureSelection) -> None:
        """Set up per-furniture rendering and analysis context.

        Args:
            furniture_selection: Selection data for this furniture including
                suggested items, prompt constraints, and style notes.
        """
        # Clear pending images from previous furniture iteration.
        # This prevents image leakage if session callback somehow doesn't trigger.
        self.pending_images = []

        furniture_id = furniture_selection.furniture_id

        # Create per-furniture rendering manager with subdirectory.
        self.rendering_manager = RenderingManager(
            cfg=self.cfg.rendering,
            logger=self.logger,
            subdirectory=f"manipulands_{furniture_id}",
        )

        # Update scene_analyzer to use per-furniture rendering manager.
        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.vlm_service,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )

        # Store current furniture selection for agent creation.
        self.current_furniture_id = furniture_id
        self.current_furniture_selection = furniture_selection

    def _initialize_checkpoint_state(self) -> None:
        """Reset checkpoint state for new furniture iteration.

        Called at the start of each furniture iteration to clear checkpoint
        state from the previous furniture piece. The attributes themselves
        were initialized in __init__().
        """
        # Reset checkpoint state to None for new furniture iteration.
        self.previous_scene_checkpoint = None
        self.scene_checkpoint = None
        self.previous_checkpoint_scores = None
        self.checkpoint_scores = None
        self.previous_scores = None
        self.previous_checkpoint_render_dir = None
        self.checkpoint_render_dir = None
        # Keep placement_style as-is (it persists across furniture iterations).

    def _setup_furniture_agents(
        self, furniture_id: UniqueID, furniture_description: str
    ) -> None:
        """Create agents and sessions for this furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
            furniture_description: Human-readable furniture description.
        """
        # Create fresh tools and agents for this furniture.
        # First create designer/critic tools.
        (
            _,  # planner_tools created later after agents exist
            designer_tools,
            critic_tools,
        ) = self._create_tools_for_furniture(furniture_id)

        # Create sessions using base class helper.
        # Sessions are stored as instance variables for planner tool closures.
        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=f"{furniture_id}_"
        )

        # Create agents using base class helpers with override methods.
        self.designer = self._create_designer_agent(
            tools=designer_tools, furniture_description=furniture_description
        )

        self.critic = self._create_critic_agent(
            tools=critic_tools, furniture_description=furniture_description
        )

        # Now create planner tools (can reference self.designer/critic/sessions).
        planner_tools = self._create_planner_tools()

        # Create planner agent using base class helper with override method.
        self.planner = self._create_planner_agent(
            tools=planner_tools, furniture_description=furniture_description
        )

    async def _run_furniture_workflow(self, furniture_id: UniqueID) -> None:
        """Execute the multi-agent workflow for a furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
        """
        self._reset_workflow_budget()

        # With critique disabled the planner has no decision to make: its only
        # useful action is request_initial_design(). Calling another model just
        # to dispatch that action doubles latency and creates an extra failure
        # boundary. Run the designer directly and retain a deterministic local
        # fallback when the provider misses its deadline.
        if self.cfg.max_critique_rounds <= 0:
            starting_hash = self.scene.content_hash()
            await self._request_initial_design_bounded()
            if self.scene.content_hash() == starting_hash:
                placed = self._place_cached_assets_deterministically()
                if placed:
                    console_logger.warning(
                        "Designer made no mutation; deterministic fallback placed %d "
                        "cached manipuland(s)",
                        placed,
                    )
                else:
                    console_logger.warning(
                        "Designer made no mutation and no cached manipuland could be "
                        "placed deterministically"
                    )
            await self._finalize_scene_and_scores()
            console_logger.info(
                "Completed direct manipuland placement for furniture %s", furniture_id
            )
            return

        # Get runner instruction for planner to start workflow.
        planner_runner_prompt = (
            ManipulandAgentPrompts.MANIPULAND_PLANNER_RUNNER_INSTRUCTION
        )
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=planner_runner_prompt,
        )

        result = await self._run_planner_with_partial_recovery(
            runner_instruction=runner_instruction,
            agent_name="PLANNER (MANIPULAND)",
            state_hash=self.scene.content_hash,
        )

        # Compute final critique and scores for completed furniture.
        # Check if scene changed since last checkpoint to avoid redundant critique.
        current_scene_hash = self.scene.content_hash()

        if (
            self.cfg.max_critique_rounds <= 0
            or self._workflow_limit_reached
            or self._critique_calls >= int(self.cfg.max_critique_rounds)
        ):
            console_logger.info("Final critique skipped: critique budget unavailable")
            self.final_render_dir = self.rendering_manager.last_render_dir
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

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()

        console_logger.info(
            f"Completed manipuland placement for furniture {furniture_id}"
        )
