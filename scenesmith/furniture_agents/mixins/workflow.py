"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import logging

from pathlib import Path
from typing import Any

from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.design.room_kits import persist_room_kit, select_room_kit
from scenesmith.agent_utils.physics.reachability import (
    compute_reachability,
    format_reachability_for_critic,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.prompts.registry import FurnitureAgentPrompts

console_logger = logging.getLogger(__name__)

from scenesmith.furniture_agents.room_kit.planning import (
    _normalize_dense_library_bookcases,
)
from scenesmith.furniture_agents.room_kit.validation import (
    _validate_furniture_collision_free,
    _validate_room_kit_completion,
)


class FurnitureAgentWorkflowMixin:
    """Furniture workflow execution and prompt-specific overrides."""

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

        _validate_furniture_collision_free(
            self.scene,
            self.cfg.physics_validation,
        )

        if room_kit is not None:
            prepruned, recovered = self._preprune_and_recover_room_kit(room_kit)
            if prepruned:
                console_logger.info(
                    "Pre-pruned %d surplus room-kit furniture objects before "
                    "deterministic recovery",
                    prepruned,
                )
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

        support_elevations = self.furniture_tools._major_support_elevations()
        if room_kit is not None:
            pruned = _normalize_dense_library_bookcases(
                self.scene,
                room_kit,
                support_elevations,
                remove_object=self.furniture_tools._remove_furniture_impl,
            )
            if pruned:
                console_logger.info(
                    "Pruned %d surplus room-kit furniture objects from explicit "
                    "dense library",
                    pruned,
                )
            _validate_furniture_collision_free(
                self.scene,
                self.cfg.physics_validation,
            )

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()
        _validate_room_kit_completion(
            self.scene,
            room_kit,
            support_elevations=support_elevations,
            enforce_exact_level_counts=True,
        )

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
