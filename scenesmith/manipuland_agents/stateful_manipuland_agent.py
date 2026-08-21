"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import json
import logging
import math
import re

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from agents import Agent, FunctionTool, custom_span
from agents.exceptions import ModelBehaviorError
from omegaconf import DictConfig
from pydrake.math import RigidTransform

from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.physical_feasibility import (
    apply_per_furniture_postprocessing,
)
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.rendering_manager import RenderingManager
from scenesmith.agent_utils.room import (
    AgentType,
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
    extract_and_propagate_support_surfaces,
)
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection, SceneAnalyzer
from scenesmith.agent_utils.scoring import (
    ManipulandCritiqueWithScores,
)
from scenesmith.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenesmith.manipuland_agents.tools.vision_tools import ManipulandVisionTools
from scenesmith.prompts.registry import ManipulandAgentPrompts
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


class StatefulManipulandAgent(BaseStatefulAgent, BaseManipulandAgent):
    """Manipuland placement with planner/designer/critic agents per furniture.

    Workflow:
    1. Initial analysis: Identify which furniture to populate
    2. Per-furniture loop: Create fresh agents for each furniture surface
    3. Per-furniture workflow: Planner coordinates designer/critic
    4. Agent-driven termination: Planner decides when surface is complete
    """

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

    @staticmethod
    def _semantic_tokens(value: str) -> set[str]:
        """Return stable content tokens for cheap cached-asset matching."""
        stop_words = {
            "a",
            "an",
            "and",
            "for",
            "of",
            "on",
            "small",
            "the",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
            if token not in stop_words
        }

    def _place_cached_assets_deterministically(self) -> int:
        """Place locally cached semantic matches on the current support surface.

        This is deliberately a fallback, not another generative path. It is
        network-free, uses the same validated placement primitive as the agent,
        and keeps a useful checkpoint when any LLM provider is slow or malformed.
        """
        selection = self.current_furniture_selection
        if selection is None or not self.manipuland_tools.support_surfaces:
            return 0

        suggestion_text = str(selection.suggested_items or "")
        suggestion_tokens = self._semantic_tokens(suggestion_text)
        assets = [
            asset
            for asset in self.asset_manager.list_available_assets()
            if asset.object_type == ObjectType.MANIPULAND
        ]

        def relevance(asset: SceneObject) -> tuple[int, float, str]:
            label = f"{asset.name} {asset.description}"
            label_tokens = self._semantic_tokens(label)
            normalized_name = asset.name.lower().replace("_", " ")
            exact = int(normalized_name in suggestion_text.lower())
            overlap = len(suggestion_tokens & label_tokens)
            quality = float(asset.metadata.get("asset_quality_score", 0.0))
            return (exact * 100 + overlap, quality, str(asset.object_id))

        ranked = sorted(assets, key=relevance, reverse=True)
        matched = [asset for asset in ranked if relevance(asset)[0] > 0][:3]
        if not matched:
            return 0

        surfaces = sorted(
            self.manipuland_tools.support_surfaces.values(),
            key=lambda surface: (surface.area, str(surface.surface_id)),
            reverse=True,
        )
        fractions_by_count = {
            1: [0.0],
            2: [-0.22, 0.22],
            3: [-0.30, 0.0, 0.30],
        }
        fractions = fractions_by_count[len(matched)]
        placed = 0

        for index, asset in enumerate(matched):
            for surface in surfaces:
                minimum = surface.bounding_box_min
                maximum = surface.bounding_box_max
                center_x = float((minimum[0] + maximum[0]) / 2.0)
                center_y = float((minimum[1] + maximum[1]) / 2.0)
                span_x = float(maximum[0] - minimum[0])
                span_y = float(maximum[1] - minimum[1])
                offset = fractions[index]
                if span_x >= span_y:
                    primary = (center_x + offset * span_x, center_y)
                else:
                    primary = (center_x, center_y + offset * span_y)
                # The first pose provides semantic spacing. Center and modest
                # cross-axis offsets make the fallback resilient to non-rectangular
                # support meshes while all bounds remain validated by the tool.
                candidates = [
                    primary,
                    (center_x, center_y),
                    (center_x + 0.12 * span_x, center_y - 0.12 * span_y),
                    (center_x - 0.12 * span_x, center_y + 0.12 * span_y),
                ]
                for position_x, position_y in candidates:
                    raw_result = (
                        self.manipuland_tools._place_manipuland_on_surface_impl(
                            asset_id=str(asset.object_id),
                            surface_id=str(surface.surface_id),
                            position_x=position_x,
                            position_z=position_y,
                            rotation_degrees=0.0,
                            _action_metadata={
                                "furniture_id": str(self.current_furniture_id),
                                "surface_id": str(surface.surface_id),
                                "placement_method": "deterministic_llm_fallback",
                            },
                        )
                    )
                    try:
                        success = bool(json.loads(raw_result).get("success"))
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        success = False
                    if success:
                        placed += 1
                        break
                if success:
                    break

        return placed

    @staticmethod
    def _is_intrinsic_catalog_book_row_asset(asset: SceneObject) -> bool:
        """Require catalog evidence for the visible multi-book row artifact."""

        if asset.object_type != ObjectType.MANIPULAND:
            return False
        catalog_id = str((asset.metadata or {}).get("catalog_id") or "").casefold()
        return catalog_id.endswith("book_encyclopedia_set_01")

    @staticmethod
    def _requests_dense_book_rows(
        furniture: SceneObject,
        selection: FurnitureSelection | None,
    ) -> bool:
        if selection is None:
            return False
        furniture_text = f"{furniture.name} {furniture.description}".casefold()
        suggestion_text = str(selection.suggested_items or "").casefold()
        return (
            any(term in furniture_text for term in ("shelf", "bookcase"))
            and "dense rows" in suggestion_text
            and "book" in suggestion_text
        )

    def _ensure_dense_book_row_asset(self) -> SceneObject | None:
        """Reuse or acquire the exact authored encyclopedia-set catalog mesh."""

        available = self.asset_manager.list_available_assets()
        row_asset = next(
            (
                asset
                for asset in available
                if self._is_intrinsic_catalog_book_row_asset(asset)
            ),
            None,
        )
        if row_asset is not None:
            return row_asset

        selection = self.current_furniture_selection
        result = self.asset_manager.generate_assets(
            AssetGenerationRequest(
                object_descriptions=[
                    "upright encyclopedia book set row with tightly packed visible "
                    "leather-bound volumes and spines facing forward"
                ],
                short_names=["encyclopedia_book_row"],
                object_type=ObjectType.MANIPULAND,
                desired_dimensions=[[0.45, 0.12, 0.22]],
                style_context=str(getattr(selection, "style_notes", "") or ""),
            )
        )
        return next(
            (
                asset
                for asset in result.successful_assets
                if self._is_intrinsic_catalog_book_row_asset(asset)
            ),
            None,
        )

    @staticmethod
    def _internal_bookcase_surfaces(
        furniture: SceneObject,
    ) -> list[SupportSurface]:
        """Return authored support planes inside an upright bookcase shell."""

        if furniture.bbox_min is None or furniture.bbox_max is None:
            return []
        object_height = float(furniture.bbox_max[2] - furniture.bbox_min[2])
        if object_height <= 0.0:
            return []
        object_elevation = float(furniture.transform.translation()[2])
        bottom = object_elevation + float(furniture.bbox_min[2])
        top = object_elevation + float(furniture.bbox_max[2])
        edge_margin = max(0.10, 0.075 * object_height)
        return sorted(
            [
                surface
                for surface in furniture.support_surfaces
                if bottom + edge_margin
                < float(surface.transform.translation()[2])
                < top - edge_margin
            ],
            key=lambda surface: (
                float(surface.transform.translation()[2]),
                str(surface.surface_id),
            ),
        )

    @staticmethod
    def _dense_book_row_owner_by_surface(
        scene: RoomScene,
    ) -> dict[UniqueID, UniqueID]:
        return {
            surface.surface_id: obj.object_id
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            for surface in obj.support_surfaces
        }

    @staticmethod
    def _dense_bookcase_story_levels(
        bookcases: list[SceneObject],
        *,
        elevation_tolerance_m: float = 0.05,
    ) -> dict[UniqueID, float]:
        """Cluster bounded post-simulation drift into authored story levels."""

        positioned: list[tuple[float, SceneObject]] = []
        for bookcase in bookcases:
            try:
                positioned.append(
                    (float(bookcase.transform.translation()[2]), bookcase)
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
        clusters: list[list[tuple[float, SceneObject]]] = []
        for elevation, bookcase in sorted(
            positioned,
            key=lambda item: (item[0], str(item[1].object_id)),
        ):
            if (
                clusters
                and abs(elevation - clusters[-1][0][0]) <= elevation_tolerance_m
            ):
                clusters[-1].append((elevation, bookcase))
            else:
                clusters.append([(elevation, bookcase)])

        levels: dict[UniqueID, float] = {}
        for cluster in clusters:
            representative = round(
                sum(elevation for elevation, _ in cluster) / len(cluster),
                3,
            )
            levels.update(
                {bookcase.object_id: representative for _, bookcase in cluster}
            )
        return levels

    @staticmethod
    def _bind_dense_book_row_to_owner_surface(
        row: SceneObject,
        owner: SceneObject,
        surface: SupportSurface,
    ) -> None:
        """Persist the authored tier pose in its owning furniture frame."""

        row.metadata["dense_library_book_row"] = True
        row.metadata["dense_library_owner_bound"] = str(owner.object_id)
        row.metadata["dense_library_owner_surface_local_transform"] = (
            (owner.transform.inverse() @ surface.transform).GetAsMatrix4().tolist()
        )

    def _normalize_intrinsic_dense_book_rows(self) -> tuple[int, int]:
        """Bind exact catalog rows placed by any workflow to their bookcase tier."""

        normalized = str(getattr(self.scene, "text_description", "")).casefold()
        explicit_dense_library = (
            "library" in normalized
            and "large" in normalized
            and "thousand" in normalized
            and bool(re.search(r"\bmulti[ -]?level\b", normalized))
        )
        if not explicit_dense_library:
            return 0, 0

        owner_by_surface = self._dense_book_row_owner_by_surface(self.scene)
        surface_by_id = {
            surface.surface_id: surface
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            for surface in obj.support_surfaces
        }
        bound = 0
        discarded = 0
        for row in list(self.scene.objects.values()):
            if (
                not self._is_intrinsic_catalog_book_row_asset(row)
                or row.placement_info is None
            ):
                continue
            surface_id = row.placement_info.parent_surface_id
            owner_id = owner_by_surface.get(surface_id)
            owner = self.scene.get_object(owner_id) if owner_id is not None else None
            surface = surface_by_id.get(surface_id)
            if (
                owner is None
                or surface is None
                or owner.object_type != ObjectType.FURNITURE
                or not any(
                    term in f"{owner.name} {owner.description}".casefold()
                    for term in ("shelf", "bookcase")
                )
            ):
                continue
            effective_surface_transform = (
                self._dense_book_row_effective_surface_transform(
                    row,
                    owner,
                    surface,
                )
            )
            if not self._dense_book_row_is_contained(
                row,
                surface,
                surface_transform=effective_surface_transform,
            ):
                self.scene.remove_object(row.object_id)
                discarded += 1
                continue
            was_bound = bool((row.metadata or {}).get("dense_library_book_row"))
            self._bind_dense_book_row_to_owner_surface(row, owner, surface)
            if not was_bound:
                bound += 1
        return bound, discarded

    @staticmethod
    def _dense_book_row_effective_surface_transform(
        row: SceneObject,
        owner: SceneObject,
        surface: SupportSurface,
    ) -> RigidTransform:
        """Resolve a cached authored tier through its owner's current pose."""

        metadata = row.metadata or {}
        if metadata.get("dense_library_owner_bound") != str(owner.object_id):
            return surface.transform
        local_transform = metadata.get("dense_library_owner_surface_local_transform")
        if local_transform is None:
            return surface.transform
        try:
            matrix = np.asarray(local_transform, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                return surface.transform
            return owner.transform @ RigidTransform(matrix)
        except (RuntimeError, TypeError, ValueError):
            return surface.transform

    @staticmethod
    def _dense_book_row_is_contained(
        row: SceneObject,
        surface: SupportSurface,
        *,
        edge_clearance_m: float = 0.002,
        surface_transform: RigidTransform | None = None,
    ) -> bool:
        """Require the row's actual footprint to remain inside its authored tier."""

        if row.bbox_min is None or row.bbox_max is None:
            return False
        row_height = float(row.bbox_max[2] - row.bbox_min[2])
        surface_clearance = float(
            surface.bounding_box_max[2] - surface.bounding_box_min[2]
        )
        if row_height > surface_clearance + 1e-6:
            return False

        effective_surface_transform = surface_transform or surface.transform
        relative = effective_surface_transform.inverse() @ row.transform
        minimum = surface.bounding_box_min
        maximum = surface.bounding_box_max
        for x in (float(row.bbox_min[0]), float(row.bbox_max[0])):
            for y in (float(row.bbox_min[1]), float(row.bbox_max[1])):
                point = relative @ np.array([x, y, 0.0])
                point_2d = point[:2]
                if not (
                    minimum[0] + edge_clearance_m
                    <= point_2d[0]
                    <= maximum[0] - edge_clearance_m
                    and minimum[1] + edge_clearance_m
                    <= point_2d[1]
                    <= maximum[1] - edge_clearance_m
                ):
                    return False
                if not surface.contains_point_2d(point_2d):
                    return False
        return True

    @classmethod
    def _physically_invalid_dense_book_row_ids(
        cls,
        scene: RoomScene,
        cfg: DictConfig,
        *,
        row_ids: set[UniqueID] | None = None,
    ) -> set[UniqueID]:
        """Return tagged rows with collision beyond allowed owner support contact."""

        if row_ids is None:
            row_ids = {
                obj.object_id
                for obj in scene.objects.values()
                if obj.object_type == ObjectType.MANIPULAND
                and bool((obj.metadata or {}).get("dense_library_book_row"))
            }
        if not row_ids:
            return set()

        owner_by_surface = cls._dense_book_row_owner_by_surface(scene)
        surface_by_id = {
            surface.surface_id: surface
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            for surface in obj.support_surfaces
        }
        invalid: set[UniqueID] = set()
        for row_id in row_ids:
            row = scene.get_object(row_id)
            parent_surface_id = (
                row.placement_info.parent_surface_id
                if row is not None and row.placement_info is not None
                else None
            )
            surface = surface_by_id.get(parent_surface_id)
            owner_id = owner_by_surface.get(parent_surface_id)
            owner = scene.get_object(owner_id) if owner_id is not None else None
            effective_surface_transform = (
                cls._dense_book_row_effective_surface_transform(row, owner, surface)
                if row is not None and owner is not None and surface is not None
                else None
            )
            if (
                row is None
                or owner is None
                or surface is None
                or not cls._dense_book_row_is_contained(
                    row,
                    surface,
                    surface_transform=effective_surface_transform,
                )
            ):
                invalid.add(row_id)

        physics_cfg = cfg.physics_validation
        collisions = compute_scene_collisions(
            scene=scene,
            penetration_threshold=physics_cfg.object_penetration_threshold_m,
            floor_penetration_tolerance=physics_cfg.floor_penetration_tolerance_m,
            current_furniture_id=None,
            manipuland_furniture_tolerance_m=0.0,
        )
        for collision in collisions:
            pair = {
                UniqueID(collision.object_a_id),
                UniqueID(collision.object_b_id),
            }
            colliding_rows = pair & row_ids
            for row_id in colliding_rows:
                row = scene.get_object(row_id)
                owner_id = (
                    owner_by_surface.get(row.placement_info.parent_surface_id)
                    if row is not None and row.placement_info is not None
                    else None
                )
                other_ids = pair - {row_id}
                is_owner_bound_contact = owner_id is not None and owner_id in other_ids
                if not is_owner_bound_contact:
                    invalid.add(row_id)
        return invalid

    def _dense_book_row_pose_is_collision_free(
        self,
        furniture_id: UniqueID,
        row_id: UniqueID,
    ) -> bool:
        """Validate a tentative row against its owner and room structure."""

        furniture = self.scene.get_object(furniture_id)
        row = self.scene.get_object(row_id)
        if furniture is None or row is None:
            return False
        surface = next(
            (
                candidate
                for candidate in furniture.support_surfaces
                if row.placement_info is not None
                and candidate.surface_id == row.placement_info.parent_surface_id
            ),
            None,
        )
        if surface is None or not self._dense_book_row_is_contained(row, surface):
            return False
        invalid = self._physically_invalid_dense_book_row_ids(
            self.scene,
            self.cfg,
            row_ids={row_id},
        )
        if invalid:
            console_logger.warning(
                "Rejected dense book-row pose for %s on %s due to containment or "
                "non-owner collision",
                row_id,
                furniture_id,
            )
        return not invalid

    @staticmethod
    def _dense_book_rows_on_furniture(
        scene: RoomScene,
        furniture: SceneObject,
    ) -> list[SceneObject]:
        surface_ids = {surface.surface_id for surface in furniture.support_surfaces}
        return [
            obj
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.MANIPULAND
            and bool((obj.metadata or {}).get("dense_library_book_row"))
            and obj.placement_info is not None
            and obj.placement_info.parent_surface_id in surface_ids
        ]

    def _place_dense_book_rows_deterministically(
        self,
        furniture: SceneObject,
    ) -> int:
        """Place one proven multi-book artifact on every compatible internal tier."""

        if not self._requests_dense_book_rows(
            furniture, self.current_furniture_selection
        ):
            return 0
        row_asset = self._ensure_dense_book_row_asset()
        if (
            row_asset is None
            or row_asset.bbox_min is None
            or row_asset.bbox_max is None
        ):
            return 0

        row_height = float(row_asset.bbox_max[2] - row_asset.bbox_min[2])
        existing_surface_ids = {
            obj.placement_info.parent_surface_id
            for obj in getattr(self.scene, "objects", {}).values()
            if obj.object_type == ObjectType.MANIPULAND
            and obj.placement_info is not None
            and bool((obj.metadata or {}).get("dense_library_book_row"))
        }
        previous_noise_profile = getattr(
            self.manipuland_tools, "active_noise_profile", None
        )
        if previous_noise_profile is not None:
            self.manipuland_tools.active_noise_profile = SimpleNamespace(
                position_xy_std_meters=0.0,
                rotation_yaw_std_degrees=0.0,
            )
        placed = 0
        try:
            for surface in self._internal_bookcase_surfaces(furniture):
                if surface.surface_id in existing_surface_ids:
                    continue
                clearance = float(
                    surface.bounding_box_max[2] - surface.bounding_box_min[2]
                )
                if row_height > clearance + 1e-6:
                    continue
                minimum = surface.bounding_box_min
                maximum = surface.bounding_box_max
                center_x = float((minimum[0] + maximum[0]) / 2.0)
                center_y = float((minimum[1] + maximum[1]) / 2.0)
                span_x = float(maximum[0] - minimum[0])
                positions = tuple(
                    (center_x + fraction * span_x, center_y)
                    for fraction in (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3)
                )
                candidates = (
                    (position_x, position_y, rotation_degrees)
                    for rotation_degrees in (0.0, 180.0)
                    for position_x, position_y in positions
                )
                for position_x, position_y, rotation_degrees in candidates:
                    raw_result = (
                        self.manipuland_tools._place_manipuland_on_surface_impl(
                            asset_id=str(row_asset.object_id),
                            surface_id=str(surface.surface_id),
                            position_x=position_x,
                            position_z=position_y,
                            rotation_degrees=rotation_degrees,
                            _action_metadata={
                                "furniture_id": str(self.current_furniture_id),
                                "surface_id": str(surface.surface_id),
                                "placement_method": "deterministic_dense_book_row",
                            },
                        )
                    )
                    try:
                        result = json.loads(raw_result)
                    except (json.JSONDecodeError, TypeError):
                        result = {}
                    if not result.get("success"):
                        continue
                    object_id = result.get("object_id")
                    placed_object = (
                        self.scene.get_object(UniqueID(object_id))
                        if object_id
                        else None
                    )
                    if placed_object is not None:
                        if not self._dense_book_row_pose_is_collision_free(
                            furniture.object_id,
                            placed_object.object_id,
                        ):
                            self.scene.remove_object(placed_object.object_id)
                            continue
                        self._bind_dense_book_row_to_owner_surface(
                            placed_object,
                            furniture,
                            surface,
                        )
                    placed += 1
                    break
        finally:
            if previous_noise_profile is not None:
                self.manipuland_tools.active_noise_profile = previous_noise_profile
        return placed

    @staticmethod
    def _validate_dense_library_book_rows(
        scene: RoomScene,
        *,
        invalid_row_ids: set[UniqueID] | None = None,
    ) -> int:
        """Require surviving intrinsic book rows on every authored library story."""

        normalized = str(getattr(scene, "text_description", "")).casefold()
        explicit_dense_library = (
            "library" in normalized
            and "large" in normalized
            and "thousand" in normalized
            and bool(re.search(r"\bmulti[ -]?level\b", normalized))
        )
        if not explicit_dense_library:
            return 0

        bookcases = [
            obj
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            and any(
                term in f"{obj.name} {obj.description}".casefold()
                for term in ("shelf", "bookcase")
            )
        ]
        bookcase_levels = StatefulManipulandAgent._dense_bookcase_story_levels(
            bookcases
        )
        support_levels: dict[UniqueID, float] = {}
        levels = set(bookcase_levels.values())
        for bookcase in bookcases:
            level = bookcase_levels.get(bookcase.object_id)
            if level is None:
                continue
            support_levels.update(
                {surface.surface_id: level for surface in bookcase.support_surfaces}
            )
        if len(levels) < 2:
            return 0

        invalid_row_ids = invalid_row_ids or set()
        tagged_invalid = sorted(
            str(obj.object_id)
            for obj in scene.objects.values()
            if obj.object_id in invalid_row_ids
            and obj.object_type == ObjectType.MANIPULAND
            and bool((obj.metadata or {}).get("dense_library_book_row"))
        )
        if tagged_invalid:
            raise ModelBehaviorError(
                "Dense library contains physically invalid tagged book rows: "
                f"{', '.join(tagged_invalid)}. The detail stage cannot publish "
                "this checkpoint."
            )

        counts = {level: 0 for level in levels}
        owner_by_surface = StatefulManipulandAgent._dense_book_row_owner_by_surface(
            scene
        )
        row_counts_by_owner = {bookcase.object_id: 0 for bookcase in bookcases}
        for obj in scene.objects.values():
            if (
                obj.object_type != ObjectType.MANIPULAND
                or not bool((obj.metadata or {}).get("dense_library_book_row"))
                or obj.placement_info is None
            ):
                continue
            level = support_levels.get(obj.placement_info.parent_surface_id)
            if level is not None:
                counts[level] += 1
                owner_id = owner_by_surface.get(obj.placement_info.parent_surface_id)
                if owner_id in row_counts_by_owner:
                    row_counts_by_owner[owner_id] += 1

        required_per_level = 12
        deficits = [
            (level, counts[level])
            for level in sorted(levels)
            if counts[level] < required_per_level
        ]
        if deficits:
            details = "; ".join(
                f"{level:.3f}m placed {placed}, required {required_per_level}"
                for level, placed in deficits
            )
            raise ModelBehaviorError(
                "Dense library book-row coverage deficits: "
                f"{details}. The detail stage cannot publish this checkpoint."
            )

        grouped_run_deficits: list[tuple[float, int]] = []
        for level in sorted(levels):
            grouped_bookcases = [
                bookcase
                for bookcase in bookcases
                if bookcase_levels.get(bookcase.object_id) == level
                and (bookcase.metadata or {}).get("dense_library_grouped_run")
                is not None
            ]
            if not grouped_bookcases:
                continue
            populated = sum(
                row_counts_by_owner.get(bookcase.object_id, 0) >= 3
                for bookcase in grouped_bookcases
            )
            if populated < 3:
                grouped_run_deficits.append((level, populated))
        if grouped_run_deficits:
            details = "; ".join(
                f"populated bookcase wall run at {level:.3f}m has {populated}, "
                "required 3"
                for level, populated in grouped_run_deficits
            )
            raise ModelBehaviorError(
                "Dense library grouped-run population deficits: "
                f"{details}. The detail stage cannot publish this checkpoint."
            )
        return sum(counts.values())

    @staticmethod
    def _furniture_template_key(furniture: SceneObject) -> tuple[str, float] | None:
        """Identify interchangeable instances without an LLM call."""
        if furniture.geometry_path is None:
            return None
        return (
            str(furniture.geometry_path.resolve()),
            round(furniture.scale_factor, 6),
        )

    def _clone_manipulands_between_identical_furniture(
        self,
        source_id: UniqueID,
        target_id: UniqueID,
        *,
        dense_book_rows_only: bool = False,
        excluded_object_ids: set[UniqueID] | None = None,
    ) -> int:
        """Copy a composed surface arrangement into an identical asset frame.

        Surface-relative rigid transforms are transferred exactly, so repeated
        beds/tables/shelves do not require another planner/designer tool loop.
        """
        source = self.scene.get_object(source_id)
        target = self.scene.get_object(target_id)
        if source is None or target is None:
            return 0
        if len(source.support_surfaces) != len(target.support_surfaces):
            return 0

        surface_pairs = {
            source_surface.surface_id: target_surface
            for source_surface, target_surface in zip(
                source.support_surfaces, target.support_surfaces, strict=True
            )
        }
        excluded_object_ids = excluded_object_ids or set()
        originals = [
            obj
            for obj in list(self.scene.objects.values())
            if obj.object_type == ObjectType.MANIPULAND
            and obj.placement_info is not None
            and obj.placement_info.parent_surface_id in surface_pairs
            and obj.object_id not in excluded_object_ids
            and (
                not dense_book_rows_only
                or bool((obj.metadata or {}).get("dense_library_book_row"))
            )
        ]
        for original in originals:
            source_surface = next(
                surface
                for surface in source.support_surfaces
                if surface.surface_id == original.placement_info.parent_surface_id
            )
            target_surface = surface_pairs[source_surface.surface_id]
            relative_transform = source_surface.transform.inverse() @ original.transform
            target_transform = target_surface.transform @ relative_transform
            position_2d, rotation_2d = target_surface.from_world_pose(target_transform)
            clone = SceneObject(
                object_id=self.scene.generate_unique_id(original.name),
                object_type=original.object_type,
                name=original.name,
                description=original.description,
                transform=target_transform,
                geometry_path=original.geometry_path,
                sdf_path=original.sdf_path,
                image_path=original.image_path,
                support_surfaces=[],
                placement_info=PlacementInfo(
                    parent_surface_id=target_surface.surface_id,
                    position_2d=position_2d,
                    rotation_2d=rotation_2d,
                    placement_method="template_transfer",
                ),
                metadata=original.metadata.copy(),
                bbox_min=(
                    original.bbox_min.copy() if original.bbox_min is not None else None
                ),
                bbox_max=(
                    original.bbox_max.copy() if original.bbox_max is not None else None
                ),
                immutable=original.immutable,
                scale_factor=original.scale_factor,
            )
            if bool(clone.metadata.get("dense_library_book_row")):
                self._bind_dense_book_row_to_owner_surface(
                    clone,
                    target,
                    target_surface,
                )
            self.scene.add_object(clone)
        return len(originals)

    def _recover_dense_library_book_row_deficits(self) -> int:
        """Fill story deficits from proven rows on identical local bookcases."""

        normalized = str(getattr(self.scene, "text_description", "")).casefold()
        explicit_dense_library = (
            "library" in normalized
            and "large" in normalized
            and "thousand" in normalized
            and bool(re.search(r"\bmulti[ -]?level\b", normalized))
        )
        if not explicit_dense_library:
            return 0

        bookcases = [
            obj
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            and any(
                term in f"{obj.name} {obj.description}".casefold()
                for term in ("shelf", "bookcase")
            )
        ]
        bookcase_levels = self._dense_bookcase_story_levels(bookcases)
        bookcases_by_level: dict[float, list[SceneObject]] = {}
        for bookcase in bookcases:
            level = bookcase_levels.get(bookcase.object_id)
            if level is None:
                continue
            bookcases_by_level.setdefault(level, []).append(bookcase)
        if len(bookcases_by_level) < 2:
            return 0

        invalid_row_ids = self._physically_invalid_dense_book_row_ids(
            self.scene,
            self.cfg,
        )
        required_per_level = 12
        recovered = 0
        for level in sorted(bookcases_by_level):
            level_bookcases = sorted(
                bookcases_by_level[level], key=lambda obj: str(obj.object_id)
            )
            rows_by_bookcase = {
                bookcase.object_id: [
                    row
                    for row in self._dense_book_rows_on_furniture(self.scene, bookcase)
                    if row.object_id not in invalid_row_ids
                ]
                for bookcase in level_bookcases
            }
            level_count = sum(len(rows) for rows in rows_by_bookcase.values())
            grouped_bookcase_ids = {
                bookcase.object_id
                for bookcase in level_bookcases
                if (bookcase.metadata or {}).get("dense_library_grouped_run")
                is not None
            }

            def populated_grouped_cases() -> int:
                return sum(
                    len(rows_by_bookcase[bookcase_id]) >= 3
                    for bookcase_id in grouped_bookcase_ids
                )

            if level_count >= required_per_level and (
                not grouped_bookcase_ids or populated_grouped_cases() >= 3
            ):
                continue

            targets = [
                bookcase
                for bookcase in level_bookcases
                if not rows_by_bookcase[bookcase.object_id]
            ]
            targets.sort(
                key=lambda bookcase: (
                    (bookcase.metadata or {}).get("dense_library_grouped_run") is None,
                    str(bookcase.object_id),
                )
            )
            for target in targets:
                target_key = self._furniture_template_key(target)
                if target_key is None:
                    continue
                compatible_sources = [
                    source
                    for source in level_bookcases
                    if source.object_id != target.object_id
                    and self._furniture_template_key(source) == target_key
                    and rows_by_bookcase[source.object_id]
                ]
                if not compatible_sources:
                    continue
                source = max(
                    compatible_sources,
                    key=lambda obj: (
                        len(rows_by_bookcase[obj.object_id]),
                        str(obj.object_id),
                    ),
                )
                before_ids = set(self.scene.objects)
                self._clone_manipulands_between_identical_furniture(
                    source.object_id,
                    target.object_id,
                    dense_book_rows_only=True,
                    excluded_object_ids=invalid_row_ids,
                )
                new_row_ids = {
                    object_id
                    for object_id in set(self.scene.objects) - before_ids
                    if bool(
                        (self.scene.objects[object_id].metadata or {}).get(
                            "dense_library_book_row"
                        )
                    )
                }
                if not new_row_ids:
                    continue
                newly_invalid = self._physically_invalid_dense_book_row_ids(
                    self.scene,
                    self.cfg,
                ).intersection(new_row_ids)
                for object_id in newly_invalid:
                    self.scene.remove_object(object_id)
                surviving = new_row_ids - newly_invalid
                rows_by_bookcase[target.object_id] = [
                    self.scene.objects[object_id] for object_id in surviving
                ]
                recovered += len(surviving)
                level_count += len(surviving)
                if level_count >= required_per_level and (
                    not grouped_bookcase_ids or populated_grouped_cases() >= 3
                ):
                    break

        return recovered

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving per-furniture manipuland placement state.

        Returns:
            Path to scene_states/manipuland_furniture_{id} directory.
        """
        return (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{self.current_furniture_id}"
        )

    async def add_manipulands(self, scene: RoomScene) -> None:
        """Add manipulands to furniture surfaces in the scene.

        This method implements a two-phase workflow:
        1. VLM-based furniture analysis to identify which pieces need manipulands
        2. Per-furniture multi-agent workflow (planner/designer/critic) to
           populate selected furniture with appropriate small objects

        The scene is mutated in place to add manipuland objects. Fresh agent
        contexts are created for each furniture piece to bound token usage.

        Side effects:
        - Scene objects are added (manipulands placed on furniture)
        - Support surfaces are extracted and assigned to furniture
        - Render cache is cleared before processing
        - Per-furniture subdirectories created under logger output directory
        - Checkpoint state saved after each critique iteration
        - Final scores copied to furniture_<id>/final_scene/ directories

        Requirements:
        - Furniture must have geometry_path (non-None)
        - Furniture must have valid bounding boxes (bbox_min, bbox_max)
        - Scene must have text_description for agent context

        Args:
            scene: RoomScene with furniture already placed. Furniture objects must
                have geometry and bounding boxes to be considered for manipuland
                placement.

        Raises:
            Exception: If support surface extraction fails (indicates invalid
                furniture geometry). Agent execution errors are logged but do
                not halt processing of remaining furniture.
        """
        console_logger.info("Starting manipuland placement")
        self.scene = scene

        # Clear render cache to ensure fresh renders for manipulands.
        # This prevents cache key collisions when object IDs are reused.
        self.rendering_manager.clear_cache()

        # Phase 1: Initial analysis - identify which furniture to populate.
        furniture_data = await self._analyze_furniture_for_placement(scene)

        if not furniture_data:
            console_logger.info("No furniture identified for manipuland placement")
            return

        console_logger.info(
            f"Identified {len(furniture_data)} furniture pieces to populate"
        )

        # Phase 1b: Select context furniture for each selection.
        if self.cfg.context_furniture.enabled:
            # Get path to furniture_selection images (already rendered).
            furniture_selection_dir = (
                self.rendering_manager._base_output_dir
                / "scene_renders"
                / "furniture_selection"
            )
            images_dir = (
                furniture_selection_dir if furniture_selection_dir.exists() else None
            )

            context_map = self.scene_analyzer.select_context_furniture(
                scene=scene,
                furniture_selections=furniture_data,
                furniture_selection_images_dir=images_dir,
            )

            # Attach context to each selection.
            for selection in furniture_data:
                selection.context_furniture_ids = context_map.get(
                    selection.furniture_id, []
                )

        # Phase 2: Per-furniture loop. Identical asset instances share one
        # composed template in their canonical surface frame.
        populated_templates: dict[tuple[str, float], UniqueID] = {}
        for furniture_selection in furniture_data:
            furniture_id = furniture_selection.furniture_id
            # Create custom span for this furniture's manipuland placement.
            with custom_span(
                name=f"manipulands_{furniture_id}",
                data={"furniture_id": str(furniture_id)},
            ):
                console_logger.info(f"Populating furniture: {furniture_id}")
                if furniture_selection.suggested_items:
                    console_logger.info(
                        f"Suggested items: {furniture_selection.suggested_items}"
                    )
                    console_logger.info(
                        f"Prompt constraints: {furniture_selection.prompt_constraints}"
                    )
                    console_logger.info(
                        f"Style notes: {furniture_selection.style_notes}"
                    )

                # Extract support surface for this furniture.
                furniture = scene.get_object(furniture_id)
                if not furniture:
                    console_logger.warning(
                        f"Furniture {furniture_id} not found, skipping"
                    )
                    continue

                # Extract all support surfaces using HSM algorithm.
                hsm_config = SupportSurfaceExtractionConfig.from_config(
                    cfg=self.cfg.support_surface_extraction
                )
                surfaces = extract_and_propagate_support_surfaces(
                    scene=self.scene, furniture_object=furniture, config=hsm_config
                )

                console_logger.info(
                    f"Extracted {len(surfaces)} support surface(s) for {furniture_id}"
                )

                # Skip furniture with no support surfaces (e.g., plants, unsuitable geometry).
                if not surfaces:
                    console_logger.warning(
                        f"No support surfaces found for {furniture_id}, skipping manipuland placement"
                    )
                    continue

                template_key = self._furniture_template_key(furniture)
                template_source = (
                    populated_templates.get(template_key)
                    if template_key is not None
                    else None
                )
                if template_source is not None:
                    clone_count = self._clone_manipulands_between_identical_furniture(
                        template_source, furniture_id
                    )
                    if clone_count:
                        console_logger.info(
                            "Transferred %d manipuland(s) from identical furniture "
                            "%s to %s without another LLM workflow",
                            clone_count,
                            template_source,
                            furniture_id,
                        )
                        continue

                try:
                    # Set up per-furniture context.
                    self._setup_furniture_context(furniture_selection)

                    # Generate context image for manipuland placement (if enabled).
                    self.manipuland_context_image_path = (
                        self._generate_manipuland_context_image()
                    )

                    # Initialize checkpoint state.
                    self._initialize_checkpoint_state()

                    # Get furniture description for agent prompts.
                    furniture_obj = scene.get_object(furniture_id)
                    furniture_description = (
                        furniture_obj.description if furniture_obj else "furniture"
                    )

                    # Create agents and sessions.
                    self._setup_furniture_agents(
                        furniture_id=furniture_id,
                        furniture_description=furniture_description,
                    )

                    book_rows_placed = self._place_dense_book_rows_deterministically(
                        furniture
                    )
                    if book_rows_placed:
                        console_logger.info(
                            "Deterministically populated %d internal bookcase tiers "
                            "for %s",
                            book_rows_placed,
                            furniture_id,
                        )

                    dense_rows = self._dense_book_rows_on_furniture(
                        self.scene,
                        furniture,
                    )
                    if len(dense_rows) >= 4:
                        console_logger.info(
                            "Skipping manipuland LLM workflow for %s: %d clean "
                            "deterministic book rows already satisfy this bookcase",
                            furniture_id,
                            len(dense_rows),
                        )
                    else:
                        # Run multi-agent workflow.
                        await self._run_furniture_workflow(furniture_id)

                    # Per-furniture post-processing (after manipulands placed).
                    if self.cfg.per_furniture_postprocessing.enabled:
                        sim_cfg = self.cfg.per_furniture_postprocessing.simulation
                        sim_html_path = None
                        if sim_cfg.save_html:
                            sim_html_path = (
                                self.scene.scene_dir
                                / "simulation"
                                / "per_furniture"
                                / f"{furniture_id}_simulation.html"
                            )
                        self.scene = apply_per_furniture_postprocessing(
                            full_scene=self.scene,
                            furniture_id=furniture_id,
                            config=self.cfg.per_furniture_postprocessing,
                            simulation_html_path=sim_html_path,
                        )

                    if template_key is not None:
                        populated_templates[template_key] = furniture_id

                except Exception as e:
                    console_logger.error(
                        f"Error populating furniture {furniture_id}: {e}", exc_info=True
                    )
                    # Continue to next furniture piece.
                    continue

        normalized_rows, discarded_rows = self._normalize_intrinsic_dense_book_rows()
        if normalized_rows or discarded_rows:
            console_logger.info(
                "Normalized exact dense book rows before final dynamics: "
                "%d owner-bound, %d uncontained discarded",
                normalized_rows,
                discarded_rows,
            )

        recovered_dense_book_rows = self._recover_dense_library_book_row_deficits()
        if recovered_dense_book_rows:
            console_logger.info(
                "Recovered %d dense book rows across additional compatible "
                "same-story bookcases",
                recovered_dense_book_rows,
            )
        invalid_dense_book_rows = self._physically_invalid_dense_book_row_ids(
            self.scene,
            self.cfg,
        )
        dense_book_rows = self._validate_dense_library_book_rows(
            self.scene,
            invalid_row_ids=invalid_dense_book_rows,
        )
        if dense_book_rows:
            console_logger.info(
                "Dense library book-row completion passed with %d surviving rows",
                dense_book_rows,
            )
        console_logger.info("Manipuland placement complete")

    async def _analyze_furniture_for_placement(
        self, scene: RoomScene
    ) -> list[FurnitureSelection]:
        """Analyze which furniture should have manipulands.

        Delegates to SceneAnalyzer for VLM-based furniture selection.

        Args:
            scene: RoomScene with furniture.

        Returns:
            List of FurnitureSelection objects with assignment context.
        """
        return self.scene_analyzer.analyze_furniture_for_manipulands(
            scene=scene,
            prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
        )
