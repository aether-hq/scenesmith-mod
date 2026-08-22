"""Base class for stateful agents using planner/designer/critic workflow.

This module provides the shared framework for all design agents (floor plan,
furniture, wall, manipuland), extracting the common multi-agent architecture
while allowing domain-specific customization through abstract methods and
subclass-defined tools.
"""

import copy
import logging

from abc import abstractmethod
from pathlib import Path
from typing import Any

import yaml

from agents.exceptions import ModelBehaviorError

from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.physics.physics_tools import check_physics_violations
from scenesmith.agent_utils.runtime.agent_runtime import (
    BoundedRunner as Runner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.runtime.scoring import (
    CritiqueWithScores,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
)
from scenesmith.agent_utils.runtime.stateful.usage import log_agent_usage
from scenesmith.utils.openai import encode_image_to_base64

console_logger = logging.getLogger(__name__)


class StatefulAgentRequestMixin:
    """Critique, design-change, and initial-design request implementations."""

    @abstractmethod
    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Prompt enum for domain-specific critic instruction.
        """

    @abstractmethod
    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for domain-specific tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """

    def _get_extra_critique_kwargs(self) -> dict[str, Any]:
        """Get extra keyword arguments for critic prompt template.

        Override in subclasses to inject domain-specific context into critic prompts.
        For example, furniture agent overrides this to add reachability context.

        Returns:
            Dictionary of extra kwargs to pass to prompt rendering.
        """
        return {}

    async def _request_critique_impl(self, update_checkpoint: bool = True) -> str:
        """Implementation for critique request.

        Runs critic agent (which calls observe_scene to render and get images),
        saves scores, and manages checkpoint state.

        Args:
            update_checkpoint: Whether to shift checkpoints. Set to False for
                final critique calls to preserve N-1 checkpoint for reset check.

        Returns:
            Critique text with optional score deltas for planner.
        """
        console_logger.info("Tool called: request_critique")

        # Get current furniture ID for manipuland agents.
        current_furniture_id = getattr(self, "current_furniture_id", None)

        # Get physics violations using the same logic as the check_physics tool.
        # This ensures the critic sees exactly the same information as the designer.
        physics_context = check_physics_violations(
            scene=self.scene,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            agent_type=self.agent_type,
        )

        # Critic evaluates with physics context. It will call observe_scene to
        # render and get visual context (images persist in session via ToolOutputImage).
        prompt_enum = self._get_critique_prompt_enum()

        # Get any agent-specific extra kwargs for the prompt (e.g., reachability).
        extra_kwargs = self._get_extra_critique_kwargs()

        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            physics_context=physics_context,
            placement_style=self.placement_style,
            **extra_kwargs,
        )
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
        log_agent_usage(result=result, agent_name="CRITIC")

        # Parse structured output.
        response = result.final_output_as(CritiqueWithScores)

        # Log critique text and scores to console.
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="CRITIQUE SCORES")

        # Save scores to YAML next to scene renders (from observe_scene call).
        images_dir = self.rendering_manager.last_render_dir
        if images_dir:
            scores_dict = scores_to_dict(response)
            scores_path = images_dir / "scores.yaml"
            with open(scores_path, "w") as f:
                yaml.dump(
                    data=scores_dict,
                    stream=f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            console_logger.info(f"Scores saved to: {scores_path}")
        else:
            console_logger.error(
                "No render directory available - scores not saved to file"
            )

        # Compute score deltas and format for planner if we have previous scores.
        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )

        # Shift checkpoints only during iteration critiques, not final critique.
        # This preserves N-1 checkpoint for reset check in _finalize_scene_and_scores.
        if update_checkpoint:
            # Shift current checkpoint to previous before saving new one.
            # This maintains N-1 and N checkpoints for rollback functionality.
            self.previous_scene_checkpoint = self.scene_checkpoint
            self.previous_checkpoint_scores = self.checkpoint_scores
            self.previous_checkpoint_render_dir = self.checkpoint_render_dir

            # Save new checkpoint (current scene state).
            self.scene_checkpoint = copy.deepcopy(self.scene.to_state_dict())
            self.checkpoint_scores = response
            self.checkpoint_render_dir = images_dir

            # Reuse render cache hash for checkpoint change detection.
            self.checkpoint_scene_hash = self.scene.content_hash()

        # Always update previous_scores for delta formatting in planner.
        self.previous_scores = response

        # Always track the final render directory (separate from checkpoint logic).
        # This is needed because final critique uses update_checkpoint=False, but we
        # still need to know the actual last render dir for copying to final output.
        self.final_render_dir = images_dir

        # Return natural language critique with score deltas for planner.
        return response.critique + score_change_msg

    @abstractmethod
    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Prompt enum for domain-specific design change instruction.
        """

    async def _request_design_change_impl(self, instruction: str) -> str:
        """Implementation for design change request.

        Args:
            instruction: Specific changes to make based on critique feedback.

        Returns:
            Designer's report of what was changed.
        """
        console_logger.info("Tool called: request_design_change")

        # Get instruction from prompt registry with domain-specific enum.
        prompt_enum = self._get_design_change_prompt_enum()
        full_instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum,
            instruction=instruction,
        )

        # Designer run with critique-based instruction.
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
        log_agent_usage(result=result, agent_name="DESIGNER (CHANGE)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (CHANGE)"
            )

        return result.final_output

    @abstractmethod
    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Prompt enum for domain-specific initial design instruction.
        """

    @abstractmethod
    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dictionary of kwargs to pass to get_prompt() for initial design.
        """

    def _get_context_image_path(self) -> Path | None:
        """Get optional context image path for initial design.

        Subclasses can override to provide an AI-generated reference image
        that will be included in the initial design user message.

        Returns:
            Path to context image, or None if not available.
        """
        return None

    def _build_initial_design_input(self, instruction: str) -> str | list[dict]:
        """Build the input for initial design request.

        If a context image is available, constructs a multimodal message
        with both text instruction and the reference image.

        Args:
            instruction: Text instruction for the designer.

        Returns:
            Either plain text or a list with a multimodal user message.
        """
        context_image_path = self._get_context_image_path()
        if context_image_path and context_image_path.exists():
            # Build multimodal input with text + image.
            console_logger.info(
                f"Including context image in initial design: {context_image_path}"
            )
            image_base64 = encode_image_to_base64(context_image_path)
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_base64}",
                        },
                    ],
                }
            ]
        # No context image - use plain text.
        return instruction

    async def _request_initial_design_impl(self) -> str:
        """Implementation for initial design request.

        Returns:
            Designer's report of initial design.
        """
        console_logger.info("Tool called: request_initial_design")

        # Get instruction from prompt registry with domain-specific enum and kwargs.
        prompt_enum = self._get_initial_design_prompt_enum()
        prompt_kwargs = self._get_initial_design_prompt_kwargs()
        instruction = self.prompt_registry.get_prompt(
            prompt_enum=prompt_enum, **prompt_kwargs
        )

        # Build input (may include context image if enabled).
        input_message = self._build_initial_design_input(instruction)

        # Designer runs with initial design instruction.
        starting_hash = self.scene.content_hash()
        result = await Runner.run(
            starting_agent=self.designer,
            input=input_message,
            session=self.designer_session,
            max_turns=self.cfg.agents.designer_agent.max_turns,
            run_config=self._create_run_config(),
            timeout_seconds=agent_run_timeout_seconds(
                "designer",
                max_turns=self.cfg.agents.designer_agent.max_turns,
            ),
        )
        if self.scene.content_hash() == starting_hash:
            raise ModelBehaviorError(
                "Initial designer returned without making a scene mutation"
            )
        log_agent_usage(result=result, agent_name="DESIGNER (INITIAL)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (INITIAL)"
            )

        return result.final_output
