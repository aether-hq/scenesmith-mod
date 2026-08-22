"""Base class for stateful agents using planner/designer/critic workflow.

This module provides the shared framework for all design agents (floor plan,
furniture, wall, manipuland), extracting the common multi-agent architecture
while allowing domain-specific customization through abstract methods and
subclass-defined tools.
"""

import copy
import logging
import shutil

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from agents import FunctionTool, RunResult, function_tool
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from omegaconf import DictConfig

from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.core.checkpoint_state import (
    initialize_checkpoint_attributes,
)
from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.llm.contracts.errors import LLMHarnessError
from scenesmith.agent_utils.runtime.agent_runtime import (
    AgentWorkflowTimeout,
    BoundedRunner as Runner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.runtime.scoring import (
    CritiqueWithScores,
    compute_total_score,
    log_agent_response,
)
from scenesmith.agent_utils.runtime.stateful.agent_setup_mixin import (
    StatefulAgentSetupMixin,
)
from scenesmith.agent_utils.runtime.stateful.request_mixin import (
    StatefulAgentRequestMixin,
)
from scenesmith.agent_utils.runtime.stateful.usage import log_agent_usage
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.prompts import prompt_registry
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


class BaseStatefulAgent(StatefulAgentSetupMixin, StatefulAgentRequestMixin, ABC):
    """Base class for stateful agents with planner/designer/critic workflow.

    This class provides the shared framework for multi-agent design workflows,
    including:
    - Session management (SQLiteSession for persistent conversation history)
    - Checkpoint state initialization and rollback functionality
    - Agent creation patterns (planner, designer, critic)
    - Shared configuration and logging infrastructure

    Domain-specific behavior is implemented through abstract methods and
    subclass-defined tools/prompts, keeping the framework general while
    allowing specialization.

    Required attributes (initialized by subclasses):
    - self.scene: Scene object with restore_from_state_dict() method
    - self.rendering_manager: RenderingManager with clear_cache() method
    - self.previous_scene_checkpoint: Previous scene state dict
    - self.scene_checkpoint: Current scene state dict
    - self.previous_checkpoint_scores: Previous scores
    - self.checkpoint_scores: Current scores
    - self.previous_scores: Scores from last iteration
    - self.previous_checkpoint_render_dir: Previous render directory
    - self.checkpoint_render_dir: Current render directory
    - self.cfg: Config with reset thresholds
    """

    # Whether this agent places objects (includes placement style tool).
    # Override to False in floor plan agent which doesn't place objects.
    _is_placement_agent: bool = True

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent for collision filtering.

        Each agent type can only modify certain object types:
        - FURNITURE: Floor-standing furniture
        - MANIPULAND: Objects placed on furniture surfaces
        - WALL_MOUNTED: Objects mounted on walls
        - CEILING_MOUNTED: Objects mounted on ceilings

        Returns:
            AgentType for this agent.
        """

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
    ):
        """Initialize base placement agent with shared infrastructure.

        Args:
            cfg: Hydra configuration object.
            logger: Logger for experiment tracking.
            geometry_server_host: Host for geometry generation server.
            geometry_server_port: Port for geometry generation server.
            hssd_server_host: Host for HSSD retrieval server.
            hssd_server_port: Port for HSSD retrieval server.
        """
        self.cfg = cfg
        self.logger = logger
        self.geometry_server_host = geometry_server_host
        self.geometry_server_port = geometry_server_port
        self.hssd_server_host = hssd_server_host
        self.hssd_server_port = hssd_server_port

        # Use global prompt registry (same pattern as domain base classes).
        self.prompt_registry = prompt_registry

        # Initialize checkpoint state (N-1 and N pattern for rollback).
        initialize_checkpoint_attributes(target=self)

        # These limits are enforced in code. Prompt instructions are advisory and
        # previously allowed planners to repeat the same expensive sub-workflow
        # until the outer 30-minute deadline fired.
        self._reset_workflow_budget()

    def _reset_workflow_budget(self) -> None:
        """Reset enforced counters for one independently composed stage/item."""
        self._initial_design_calls = 0
        self._critique_calls = 0
        self._design_change_calls = 0
        self._workflow_limit_reached = False

    def _workflow_stop_message(self, phase: str, reason: Exception | str) -> str:
        """Record a recoverable limit and tell the planner to finish immediately."""
        self._workflow_limit_reached = True
        message = (
            f"{phase} stopped at its safety budget ({reason}). "
            "Keep the current scene, do not call more design tools, and finish now."
        )
        console_logger.warning(message)
        return message

    async def _request_initial_design_bounded(self) -> str:
        """Run at most one initial design and preserve mutations at a safety limit."""
        if self._workflow_limit_reached:
            return self._workflow_stop_message(
                "Initial design", "the stage safety budget is already exhausted"
            )
        if self._initial_design_calls >= 1:
            return (
                "Initial design already ran. Do not request it again; finish the "
                "workflow or use the remaining critique budget."
            )
        self._initial_design_calls += 1
        try:
            return await self._request_initial_design_impl()
        except (
            AgentWorkflowTimeout,
            MaxTurnsExceeded,
            ModelBehaviorError,
            LLMHarnessError,
        ) as exc:
            return self._workflow_stop_message("Initial design", exc)

    async def _request_critique_bounded(self, update_checkpoint: bool = True) -> str:
        """Run a critic only while the configured critique budget remains."""
        max_calls = max(0, int(self.cfg.max_critique_rounds))
        if self._workflow_limit_reached:
            return self._workflow_stop_message(
                "Critique", "the stage safety budget is already exhausted"
            )
        if self._critique_calls >= max_calls:
            return (
                f"Critique budget exhausted ({max_calls}). Keep the current scene "
                "and finish now."
            )
        self._critique_calls += 1
        try:
            return await self._request_critique_impl(
                update_checkpoint=update_checkpoint
            )
        except (
            AgentWorkflowTimeout,
            MaxTurnsExceeded,
            ModelBehaviorError,
            LLMHarnessError,
        ) as exc:
            return self._workflow_stop_message("Critique", exc)

    async def _request_design_change_bounded(self, instruction: str) -> str:
        """Run no more revisions than configured critique rounds."""
        max_calls = max(0, int(self.cfg.max_critique_rounds))
        if self._workflow_limit_reached:
            return self._workflow_stop_message(
                "Design revision", "the stage safety budget is already exhausted"
            )
        if self._design_change_calls >= max_calls:
            return (
                f"Design revision budget exhausted ({max_calls}). Keep the current "
                "scene and finish now."
            )
        self._design_change_calls += 1
        try:
            return await self._request_design_change_impl(instruction)
        except (
            AgentWorkflowTimeout,
            MaxTurnsExceeded,
            ModelBehaviorError,
            LLMHarnessError,
        ) as exc:
            return self._workflow_stop_message("Design revision", exc)

    async def _run_planner_with_partial_recovery(
        self,
        *,
        runner_instruction: str,
        agent_name: str,
        state_hash: Callable[[], int],
    ) -> RunResult | None:
        """Run a planner with a hard stage deadline and retain useful work."""
        starting_hash = state_hash()
        try:
            result: RunResult = await Runner.run(
                starting_agent=self.planner,
                input=runner_instruction,
                max_turns=self.cfg.agents.planner_agent.max_turns,
                run_config=self._create_run_config(),
                timeout_seconds=agent_run_timeout_seconds(
                    "planner",
                    max_turns=self.cfg.agents.planner_agent.max_turns,
                ),
            )
        except (
            AgentWorkflowTimeout,
            MaxTurnsExceeded,
            ModelBehaviorError,
            LLMHarnessError,
        ) as exc:
            self._workflow_limit_reached = True
            if state_hash() == starting_hash:
                raise
            console_logger.warning(
                "%s stopped at its safety budget (%s); preserving the changed "
                "scene for stage validation",
                agent_name,
                exc,
            )
            return None

        log_agent_usage(result=result, agent_name=agent_name)
        if result.final_output:
            log_agent_response(response=result.final_output, agent_name=agent_name)
        return result

    def _should_reset_to_checkpoint(
        self,
        current_scores: CritiqueWithScores,
        previous_scores: CritiqueWithScores | None,
    ) -> tuple[bool, str]:
        """Check if current scores warrant resetting to previous checkpoint.

        Uses same threshold logic as planner agent instructions.

        Args:
            current_scores: Scores for the current scene state.
            previous_scores: Scores from the previous checkpoint (N-1).

        Returns:
            (should_reset, reason) tuple where reason explains which threshold
            was exceeded.
        """
        if previous_scores is None:
            return False, ""

        # Check single category drops.
        current_scores_list = current_scores.get_scores()
        previous_scores_list = previous_scores.get_scores()
        for current_score, previous_score in zip(
            current_scores_list, previous_scores_list
        ):
            drop = previous_score.grade - current_score.grade

            if drop >= self.cfg.reset_single_category_threshold:
                return True, f"{current_score.name} dropped {drop} points"

        # Check total sum drop.
        current_sum = compute_total_score(current_scores)
        previous_sum = compute_total_score(previous_scores)
        total_drop = previous_sum - current_sum

        if total_drop >= self.cfg.reset_total_sum_threshold:
            return True, f"total score dropped {total_drop} points"

        return False, ""

    @log_scene_action
    def _perform_checkpoint_reset(self, checkpoint_state_dict: dict) -> None:
        """Restore scene and scores to previous checkpoint (N-1).

        This is the core reset operation shared by both the planner tool
        and the final scene validation logic.

        Args:
            checkpoint_state_dict: Checkpoint state dictionary to restore from.
                During normal operation, this is self.previous_scene_checkpoint.
                During replay, this is the logged checkpoint state.
        """
        # Restore scene from checkpoint (N-1 iteration).
        self.scene.restore_from_state_dict(checkpoint_state_dict)

        # Clear render cache to force new renders after reset.
        self.rendering_manager.clear_cache()

        # Reset score tracking to previous checkpoint state.
        # Note: During replay, these may be None which is okay.
        if self.previous_checkpoint_scores is not None:
            self.checkpoint_scores = copy.deepcopy(self.previous_checkpoint_scores)
            self.previous_scores = copy.deepcopy(self.previous_checkpoint_scores)

        # Invalidate current checkpoint since we went back.
        # Note: During replay, these may be None which is okay.
        if self.previous_scene_checkpoint is not None:
            self.scene_checkpoint = self.previous_scene_checkpoint
            self.checkpoint_render_dir = self.previous_checkpoint_render_dir

    @abstractmethod
    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving final scene scores.

        Returns:
            Path to the directory where final scores should be saved.
        """

    async def _finalize_scene_and_scores(self) -> None:
        """Validate final scene against thresholds and save scores.

        This method checks if the final scene's scores are degraded compared
        to the previous checkpoint. If so, it resets to the better checkpoint.
        Finally, it copies the scores to the final_scene directory for easy access.

        The final directory path is determined by the subclass implementation
        of _get_final_scores_directory().
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

            console_logger.debug(
                f"Reset check result: should_reset={should_reset}, reason={reason}"
            )

            if should_reset:
                console_logger.info(
                    f"Final scene scores are degraded ({reason}). "
                    f"Resetting to checkpoint (N-1)."
                )

                # Restore scene to checkpoint (N-1) directly. Don't use
                # _perform_checkpoint_reset() here since that's designed for mid-loop
                # resets and modifies checkpoint tracking variables.
                self.scene.restore_from_state_dict(self.scene_checkpoint)
                self.rendering_manager.clear_cache()

                scores_parts = [
                    f"{score.name}={score.grade}"
                    for score in self.checkpoint_scores.get_scores()
                ]
                console_logger.info(
                    f"Final scene restored to checkpoint state. "
                    f"Checkpoint scores: {', '.join(scores_parts)}"
                )

                # Update final_render_dir to point to restored checkpoint's render.
                self.final_render_dir = self.checkpoint_render_dir

        # Copy final scores and renders to per-stage directory.
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

    def _create_reset_checkpoint_tool(self) -> FunctionTool:
        """Create tool for resetting scene to previous checkpoint.

        Returns:
            FunctionTool that allows agents to reset to previous checkpoint.
        """

        @function_tool
        async def reset_scene_to_checkpoint(reason: str) -> str:
            """Reset scene to previous iteration state when changes made it worse.

            Use this when the designer's changes resulted in significant score
            degradation.

            Args:
                reason: Explanation of why you're resetting.

            Returns:
                Confirmation with checkpoint details and scores.
            """
            console_logger.info("Tool called: reset_scene_to_checkpoint")

            if (
                self.previous_scene_checkpoint is None
                or self.previous_checkpoint_scores is None
            ):
                console_logger.warning("No previous checkpoint available to reset to.")
                return (
                    "ERROR: No previous checkpoint available to reset to. "
                    "You must call request_critique() at least twice to create "
                    "enough checkpoints for reset functionality."
                )

            self._perform_checkpoint_reset(
                checkpoint_state_dict=self.previous_scene_checkpoint
            )

            # Log reset event.
            console_logger.info(f"Scene reset to checkpoint. Reason: {reason}")

            # Return confirmation with checkpoint scores.
            # Build scores string dynamically using get_scores() for agent-agnostic output.
            scores_parts = [
                f"{score.name}={score.grade}"
                for score in self.checkpoint_scores.get_scores()
            ]
            scores_str = ", ".join(scores_parts)

            return (
                f"Scene reset to state from 2 iterations ago.\n"
                f"Checkpoint scores: {scores_str}\n"
                f"Reset reason: {reason}\n"
                "Continue with design improvements from this restored state."
            )

        return reset_scene_to_checkpoint

    def _create_placement_style_tool(self) -> FunctionTool:
        """Create tool for selecting placement style (natural vs perfect).

        Returns:
            FunctionTool that allows agents to select placement style.
        """

        @function_tool
        def select_placement_style(style: str) -> str:
            """Select placement style based on scene prompt analysis.

            MUST be called FIRST before any placement operations.

            Analyzes the scene description to determine whether to use:
            - "natural": Realistic, lived-in scenes with slight imperfections
            - "perfect": Precise, exhibition-quality placement with no variation

            Args:
                style: Either "natural" or "perfect"

            Returns:
                Confirmation of selected style and readiness for placement.
            """
            style_lower = style.lower()
            if style_lower == "natural":
                mode = PlacementNoiseMode.NATURAL
            elif style_lower == "perfect":
                mode = PlacementNoiseMode.PERFECT
            else:
                console_logger.warning(
                    f"Invalid placement style '{style}', defaulting to 'natural'"
                )
                mode = PlacementNoiseMode.NATURAL
                style_lower = "natural"

            # Set noise profile on domain-specific tools.
            self._set_placement_noise_profile(mode)
            self.placement_style = style_lower

            return (
                f"Placement style set to '{style_lower}'. "
                f"Ready for placement with {style_lower} variation."
            )

        return select_placement_style

    def _create_planner_tools(self) -> list[FunctionTool]:
        """Create planner tools for the design workflow.

        Returns tools that the planner uses to coordinate designer and critic:
        - select_placement_style: Set natural vs perfect placement (placement agents only)
        - request_initial_design: Request initial design from designer
        - request_critique: Request evaluation from critic
        - request_design_change: Request design modifications based on feedback
        - reset_scene_to_checkpoint: Reset to last checkpoint state

        Returns:
            List of function tools for planner agent.
        """

        @function_tool
        async def request_initial_design() -> str:
            """Request the designer to create the initial design.

            The designer will analyze the context and create an appropriate
            initial layout or arrangement.

            Returns:
                Designer's report of what was created and why.
            """
            return await self._request_initial_design_bounded()

        @function_tool
        async def request_critique() -> str:
            """Request the critic to evaluate the current design.

            The critic will examine the current state and provide feedback
            on what works well and what needs improvement.

            Returns:
                Critic's detailed evaluation with specific improvement suggestions.
            """
            return await self._request_critique_bounded()

        @function_tool
        async def request_design_change(instruction: str) -> str:
            """Request the designer to address specific issues.

            Based on the critic's feedback, provide clear instructions about
            what to change. The designer will modify the design to address
            the issues while maintaining what works well.

            Args:
                instruction: Specific changes to make based on critique feedback.

            Returns:
                Designer's report of what was changed.
            """
            return await self._request_design_change_bounded(instruction)

        tools: list[FunctionTool] = [request_initial_design]

        # Only add critique-related tools if critique rounds are enabled.
        # This prevents the planner from accidentally calling critique tools
        # when max_critique_rounds is 0.
        if self.cfg.max_critique_rounds > 0:
            reset_scene_to_checkpoint = self._create_reset_checkpoint_tool()
            tools.extend(
                [request_critique, request_design_change, reset_scene_to_checkpoint]
            )

        # Add placement style tool for placement agents (not floor plan).
        if self._is_placement_agent:
            placement_style_tool = self._create_placement_style_tool()
            tools.insert(0, placement_style_tool)

        return tools
