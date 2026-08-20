"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import json
import logging
import re

from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool
from agents.exceptions import ModelBehaviorError
from omegaconf import DictConfig

from scenesmith.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.reachability import (
    compute_reachability,
    format_reachability_for_critic,
)
from scenesmith.agent_utils.room import AgentType, ObjectType, RoomScene
from scenesmith.agent_utils.room_kits import (
    RoomKitSelection,
    persist_room_kit,
    select_room_kit,
)
from scenesmith.agent_utils.scoring import FurnitureCritiqueWithScores
from scenesmith.agent_utils.workflow_tools import WorkflowTools
from scenesmith.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
from scenesmith.furniture_agents.tools.scene_tools import SceneTools
from scenesmith.furniture_agents.tools.vision_tools import VisionTools
from scenesmith.prompts.registry import FurnitureAgentPrompts
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


def _validate_room_kit_completion(
    scene: RoomScene, room_kit: RoomKitSelection | None
) -> int:
    """Reject a matched semantic room kit that did not place required furniture."""

    furniture_count = sum(
        obj.object_type == ObjectType.FURNITURE for obj in scene.objects.values()
    )
    if room_kit is None:
        return furniture_count

    required_minimum = sum(
        slot.minimum_count for slot in room_kit.slots if slot.required
    )
    if furniture_count < required_minimum:
        raise ModelBehaviorError(
            f"Semantic room kit {room_kit.kit_id} placed {furniture_count} "
            f"furniture objects; required minimum is {required_minimum}. "
            "The furniture stage cannot publish this checkpoint."
        )
    console_logger.info(
        "Semantic room kit %s completion gate passed: %d furniture objects "
        "(minimum %d)",
        room_kit.kit_id,
        furniture_count,
        required_minimum,
    )
    return furniture_count


class StatefulFurnitureAgent(BaseStatefulAgent, BaseFurnitureAgent):
    """Natural conversation between persistent agents with proper image injection."""

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.FURNITURE

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
        # Initialize furniture-specific base class.
        BaseFurnitureAgent.__init__(
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

        # Create persistent agent sessions using base class method.
        self.designer_session, self.critic_session = self._create_sessions()

        # Context image for designer initialization (furniture-specific).
        self.context_image_path: Path | None = None
        self.room_kit_brief = (
            "No semantic room kit matched; use the scene requirements."
        )

    def _create_designer_agent(self, tools: list[FunctionTool]) -> Agent:
        """Create designer agent with tools.

        Args:
            tools: Tools to provide to the designer

        Returns:
            Configured designer agent
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = FurnitureAgentPrompts[designer_config.prompt]
        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            has_reference_image=self.context_image_path is not None,
        )

    def _create_critic_tools(self) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Returns:
            List of tools for the critic (read-only scene validation tools)
        """
        vision_tools = VisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )
        scene_tools = SceneTools(scene=self.scene, cfg=self.cfg)

        # Return vision tools + read-only scene tools.
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            scene_tools.tools["get_current_scene_state"],
            scene_tools.tools["check_facing_tool"],
        ]

    def _create_critic_agent(
        self, scene: RoomScene, tools: list[FunctionTool]
    ) -> Agent:
        """Create critic agent with scene context.

        Args:
            scene: RoomScene to provide context for the critic
            tools: Tools to provide to the critic

        Returns:
            Configured critic agent with structured output
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = FurnitureAgentPrompts[critic_config.prompt]
        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=FurnitureCritiqueWithScores,
            scene_description=scene.text_description,
        )

    def _create_planner_agent(
        self, scene: RoomScene, tools: list[FunctionTool]
    ) -> Agent:
        """Create planner agent with scene-specific context.

        Args:
            scene: RoomScene to provide context for the planner
            tools: Tools to provide to the planner

        Returns:
            Configured planner agent
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = FurnitureAgentPrompts[planner_config.prompt]
        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            scene_prompt=scene.text_description,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=self.cfg.reset_single_category_threshold,
            reset_total_sum_threshold=self.cfg.reset_total_sum_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _create_designer_tools(self) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Returns:
            List of tools for the designer agent.
        """
        vision_tools = VisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )
        self.furniture_tools = FurnitureTools(
            scene=self.scene, asset_manager=self.asset_manager, cfg=self.cfg
        )
        scene_tools = SceneTools(scene=self.scene, cfg=self.cfg)
        workflow_tools = WorkflowTools()

        return [
            *vision_tools.tools.values(),
            *self.furniture_tools.tools.values(),
            *scene_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]

    def _render_empty_room(self) -> Path:
        """Render top-down view of empty room showing doors/windows.

        Uses furniture_selection mode which disables coordinate grid/frame.
        Pass annotate_object_types=[] to disable all labels and bounding boxes.
        Result: clean room geometry with doors/windows visible but unlabeled.

        Returns:
            Path to directory containing rendered image.
        """
        return self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=[],  # Empty room only
            render_name="empty_room_context",
            rendering_mode="furniture_selection",  # Disables grid/frame
            annotate_object_types=[],  # Disables all labels/bboxes
        )

    def _generate_and_save_context_image(self, scene: RoomScene) -> Path:
        """Generate and save context image for design guidance.

        Renders an empty room showing doors/windows, then uses image editing
        to add suggested furniture placement.

        Args:
            scene: RoomScene to generate context image for.

        Returns:
            Path to saved context image.
        """
        console_logger.info("Generating context image for scene...")

        # Render empty room showing doors/windows.
        room_render_dir = self._render_empty_room()
        # Get the top-down image from the render directory.
        room_render = room_render_dir / "0_top.png"

        # Generate context image using the render as reference.
        # Save alongside the input render for easy association.
        output_path = room_render_dir / "context_edited.png"
        image_path = (
            self.asset_manager.image_generator.generate_furniture_context_image(
                reference_image_path=room_render,
                scene_description=scene.text_description,
                width_m=scene.room_geometry.width,
                length_m=scene.room_geometry.length,
                output_path=output_path,
            )
        )

        console_logger.info(f"Context image saved to: {image_path}")
        return image_path

    @staticmethod
    def _semantic_tokens(value: str) -> set[str]:
        """Normalize a short role or asset label for deterministic matching."""

        stop_words = {"a", "an", "and", "for", "of", "the", "with"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
            if token not in stop_words
        }

    @classmethod
    def _slot_relevance(cls, asset: Any, slot: Any) -> tuple[int, float, str]:
        """Rank one cached asset against a semantic room-kit slot."""

        role_names = (slot.role, *getattr(slot, "aliases", ()))
        normalized_roles = {
            " ".join(sorted(cls._semantic_tokens(role))) for role in role_names
        }
        asset_name = " ".join(sorted(cls._semantic_tokens(str(asset.name))))
        asset_tokens = cls._semantic_tokens(
            f"{asset.name} {getattr(asset, 'description', '')}"
        )
        role_tokens = set().union(*(cls._semantic_tokens(role) for role in role_names))
        exact = int(asset_name in normalized_roles)
        overlap = len(asset_tokens & role_tokens)
        quality = float(
            (getattr(asset, "metadata", None) or {}).get("asset_quality_score", 0.0)
        )
        return (exact * 100 + overlap, quality, str(asset.object_id))

    def _deterministic_room_positions(
        self, *, wall: bool
    ) -> list[tuple[float, float, float]]:
        """Return conservative unique SE(2) poses inside the room envelope."""

        half_x = max(0.5, float(self.scene.room_geometry.length) / 2.0 - 0.65)
        half_y = max(0.5, float(self.scene.room_geometry.width) / 2.0 - 0.65)
        if wall:
            return [
                (-0.55 * half_x, 0.88 * half_y, 180.0),
                (0.55 * half_x, 0.88 * half_y, 180.0),
                (-0.55 * half_x, -0.88 * half_y, 0.0),
                (0.55 * half_x, -0.88 * half_y, 0.0),
                (-0.88 * half_x, 0.45 * half_y, -90.0),
                (-0.88 * half_x, -0.45 * half_y, -90.0),
                (0.88 * half_x, 0.45 * half_y, 90.0),
                (0.88 * half_x, -0.45 * half_y, 90.0),
            ]
        return [
            (0.0, -0.18 * half_y, 0.0),
            (-0.42 * half_x, -0.18 * half_y, -90.0),
            (0.42 * half_x, -0.18 * half_y, 90.0),
            (-0.42 * half_x, 0.38 * half_y, -135.0),
            (0.42 * half_x, 0.38 * half_y, 135.0),
            (0.0, 0.58 * half_y, 180.0),
            (-0.68 * half_x, -0.58 * half_y, -45.0),
            (0.68 * half_x, -0.58 * half_y, 45.0),
            (-0.68 * half_x, 0.68 * half_y, -135.0),
            (0.68 * half_x, 0.68 * half_y, 135.0),
            (0.0, -0.72 * half_y, 0.0),
            (-0.72 * half_x, 0.0, -90.0),
            (0.72 * half_x, 0.0, 90.0),
        ]

    def _place_room_kit_minimums_deterministically(
        self, room_kit: RoomKitSelection
    ) -> int:
        """Recover required kit roles from acquired assets without another model call.

        Every attempt goes through ``FurnitureTools`` so structural support,
        enclosure, contextual, and collision validation remain authoritative.
        """

        assets = [
            asset
            for asset in self.asset_manager.list_available_assets()
            if asset.object_type == ObjectType.FURNITURE
        ]
        if not assets:
            return 0

        self.furniture_tools.set_noise_profile(PlacementNoiseMode.PERFECT)
        attempted_positions: set[tuple[float, float]] = set()
        placed = 0

        for slot in room_kit.slots:
            if not slot.required:
                continue
            existing = sum(
                obj.object_type == ObjectType.FURNITURE
                and self._slot_relevance(obj, slot)[0] >= 100
                for obj in self.scene.objects.values()
            )
            missing = max(0, int(slot.minimum_count) - existing)
            if missing == 0:
                continue

            ranked = sorted(
                assets,
                key=lambda asset: self._slot_relevance(asset, slot),
                reverse=True,
            )
            if not ranked or self._slot_relevance(ranked[0], slot)[0] <= 0:
                console_logger.warning(
                    "No cached furniture asset matched required room-kit role %s",
                    slot.role,
                )
                continue
            asset = ranked[0]
            positions = self._deterministic_room_positions(
                wall=getattr(slot, "placement_class", "floor") == "wall"
            )

            for _ in range(missing):
                success = False
                for x, y, yaw in positions:
                    position_key = (round(x, 4), round(y, 4))
                    if position_key in attempted_positions:
                        continue
                    attempted_positions.add(position_key)
                    raw_result = self.furniture_tools._add_furniture_to_scene_impl(
                        asset_id=str(asset.object_id),
                        x=x,
                        y=y,
                        z=0.0,
                        roll=0.0,
                        pitch=0.0,
                        yaw=yaw,
                    )
                    try:
                        success = bool(json.loads(raw_result).get("success"))
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        success = False
                    if success:
                        placed += 1
                        break
                if not success:
                    console_logger.warning(
                        "Deterministic recovery exhausted valid poses for room-kit "
                        "role %s after placing %d of %d missing instances",
                        slot.role,
                        placed,
                        missing,
                    )
                    break

        return placed

    async def add_furniture(self, scene: RoomScene) -> None:
        """Add furniture to a scene.

        Args:
            scene: RoomScene to add furniture to (mutated in place)
        """
        self._reset_workflow_budget()

        # Store everything as instance variables for closure access.
        self.scene = scene

        room_area_m2 = float(scene.room_geometry.width) * float(
            scene.room_geometry.length
        )
        room_kit = select_room_kit(scene.text_description, room_area_m2=room_area_m2)
        if room_kit is not None:
            self.room_kit_brief = room_kit.to_prompt_brief()
            persist_room_kit(room_kit, scene.scene_dir / "room_kit.json")
            console_logger.info(
                "Selected semantic room kit %s with counts %s",
                room_kit.kit_id,
                room_kit.slot_counts,
            )
        else:
            self.room_kit_brief = (
                "No semantic room kit matched; infer a compact functional grouping "
                "from the scene requirements."
            )

        # Generate context image if configured. If generation fails, continue without it.
        if self.cfg.context_image_generation.enabled:
            try:
                self.context_image_path = self._generate_and_save_context_image(scene)
            except Exception as e:
                console_logger.warning(
                    f"Context image generation failed, continuing without it: {e}"
                )
                self.context_image_path = None

        # Create designer, critic, and planner with tools once for this scene.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)
        critic_tools = self._create_critic_tools()
        self.critic = self._create_critic_agent(scene=scene, tools=critic_tools)
        planner_tools = self._create_planner_tools()
        self.planner = self._create_planner_agent(scene=scene, tools=planner_tools)

        # Get runner instruction from prompt registry.
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_PLANNER_RUNNER_INSTRUCTION,
        )

        # Run the furniture placement workflow.
        result = await self._run_planner_with_partial_recovery(
            runner_instruction=runner_instruction,
            agent_name="PLANNER (FURNITURE)",
            state_hash=self.scene.content_hash,
        )

        if room_kit is not None:
            recovered = self._place_room_kit_minimums_deterministically(room_kit)
            if recovered:
                console_logger.warning(
                    "Deterministic room-kit recovery placed %d required furniture "
                    "objects from cached assets",
                    recovered,
                )

        # Compute final critique and scores for completed scene.
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
        _validate_room_kit_completion(self.scene, room_kit)

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving final furniture placement state.

        Returns:
            Path to scene_states/furniture directory.
        """
        return self.logger.output_dir / "scene_states" / "furniture"

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Furniture-specific critic instruction prompt.
        """
        return FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Furniture-specific initial design instruction prompt.
        """
        return FurnitureAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dict with scene description and reference image flag.
        """
        return {
            "scene_description": self.scene.text_description,
            "has_reference_image": self.context_image_path is not None,
            "room_kit_brief": self.room_kit_brief,
        }

    def _get_context_image_path(self) -> Path | None:
        """Get the AI-generated context image for initial design.

        Returns:
            Path to context image if available, None otherwise.
        """
        return self.context_image_path

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Furniture-specific design change instruction prompt.
        """
        return FurnitureAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION_STATEFUL

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for furniture tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        self.furniture_tools.set_noise_profile(mode)

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        """Get extra kwargs for critic prompt (reachability context).

        Computes room reachability and formats it for critic context injection.
        This allows the critic to score reachability based on computed metrics.

        Returns:
            Dict with reachability_context and robot_width for prompt template.
        """
        robot_width = self.cfg.reachability.robot_width
        result = compute_reachability(scene=self.scene, robot_width=robot_width)
        reachability_context = format_reachability_for_critic(result)

        return {
            "reachability_context": reachability_context,
            "robot_width": robot_width,
        }
