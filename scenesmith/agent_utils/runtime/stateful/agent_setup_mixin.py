"""Base class for stateful agents using planner/designer/critic workflow.

This module provides the shared framework for all design agents (floor plan,
furniture, wall, manipuland), extracting the common multi-agent architecture
while allowing domain-specific customization through abstract methods and
subclass-defined tools.
"""

import logging

from typing import Any

from agents import Agent, FunctionTool, ModelSettings, RunConfig, SQLiteSession
from agents.memory.session import Session
from openai import Timeout
from openai.types.shared import Reasoning

from scenesmith.agent_utils.llm.chat_completions_image_filter import (
    ChatCompletionsToolImageFilter,
    CompositeCallModelInputFilter,
)
from scenesmith.agent_utils.llm.intra_turn_image_filter import IntraTurnImageFilter
from scenesmith.agent_utils.llm.llm_harness import (
    LLMHarnessConfig,
    agents_model_provider,
    detect_capabilities,
)
from scenesmith.agent_utils.runtime.scoring import CritiqueWithScores
from scenesmith.agent_utils.runtime.turn_trimming_session import TurnTrimmingSession

console_logger = logging.getLogger(__name__)


class StatefulAgentSetupMixin:
    """Agent construction, model settings, sessions, and run configuration."""

    def _get_model_settings(
        self,
        settings_key: str | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ModelSettings | None:
        """Create ModelSettings with timeout, reasoning effort, verbosity, and tool.

        Args:
            settings_key: Key in cfg.openai.reasoning_effort and cfg.openai.verbosity
                for this agent (e.g., "designer", "critic", "planner"). If None,
                no reasoning effort or verbosity is set.
            tool_choice: Tool name to force as first call (e.g., "observe_scene").
                Resets after first tool call by default to prevent infinite loops.
            parallel_tool_calls: Whether to allow parallel tool calls. Set to False
                for planner agents to prevent race conditions on shared sessions.

        Returns:
            ModelSettings with timeout, reasoning, verbosity, and tool_choice if
            configured, None otherwise.
        """
        kwargs: dict = {}
        extra_args: dict = {}
        harness = LLMHarnessConfig.from_env(default_model=self.cfg.openai.model)

        # Every provider receives the same bounded request deadline. Historical
        # YAML values of 1800 seconds must never override the harness policy.
        if hasattr(self.cfg, "api_timeout"):
            timeout_cfg = self.cfg.api_timeout
            request_timeout = harness.timeout_seconds + (
                5 if harness.uses_cli_bridge else 0
            )
            timeout = Timeout(
                connect=min(float(timeout_cfg.connect), request_timeout),
                read=request_timeout,
                write=min(float(timeout_cfg.write), request_timeout),
                pool=min(float(timeout_cfg.pool), request_timeout),
            )
            extra_args["timeout"] = timeout

        # Add service_tier if configured (non-null/non-empty).
        service_tier = getattr(self.cfg.openai, "service_tier", None)
        if service_tier:
            extra_args["service_tier"] = service_tier

        if extra_args:
            kwargs["extra_args"] = extra_args

        # Add reasoning effort and verbosity if key is provided.
        capabilities = detect_capabilities(harness)
        if settings_key and capabilities.native_reasoning_controls:
            reasoning_cfg = self.cfg.openai.reasoning_effort
            effort = getattr(reasoning_cfg, settings_key)
            kwargs["reasoning"] = Reasoning(effort=effort)

            verbosity_cfg = self.cfg.openai.verbosity
            verbosity = getattr(verbosity_cfg, settings_key)
            kwargs["verbosity"] = verbosity

        # Add tool_choice to force specific tool call first.
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        # Add parallel_tool_calls setting if specified.
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        return ModelSettings(**kwargs) if kwargs else None

    def _create_designer_agent(
        self, tools: list[FunctionTool], prompt_enum: Any, **prompt_kwargs: Any
    ) -> Agent:
        """Create designer agent with tools and domain-specific prompt.

        This method provides the shared pattern for designer agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the designer.
            prompt_enum: Prompt enum from domain-specific registry.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        return Agent(
            name=designer_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
            ),
            # A designer exists to mutate the scene. Requiring a first tool call
            # prevents provider-specific prose refusals from being accepted as a
            # successful design; the SDK resets tool choice after that call.
            model_settings=self._get_model_settings(
                settings_key="designer", tool_choice="required"
            ),
        )

    def _create_critic_agent(
        self,
        tools: list[FunctionTool],
        prompt_enum: Any,
        output_type: type[CritiqueWithScores],
        **prompt_kwargs: Any,
    ) -> Agent:
        """Create critic agent with structured output.

        This method provides the shared pattern for critic agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the critic.
            prompt_enum: Prompt enum from domain-specific registry.
            output_type: CritiqueWithScores subclass for structured output.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured critic agent with domain-specific CritiqueWithScores type.
        """
        critic_config = self.cfg.agents.critic_agent
        return Agent(
            name=critic_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
            ),
            output_type=output_type,
            # Force observe_scene tool call first to ensure visual context.
            model_settings=self._get_model_settings(
                settings_key="critic", tool_choice="observe_scene"
            ),
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], prompt_enum: Any, **prompt_kwargs: Any
    ) -> Agent:
        """Create planner agent for workflow coordination.

        This method provides the shared pattern for planner agent creation,
        allowing subclasses to specify the prompt enum and context.

        Args:
            tools: Tools to provide to the planner.
            prompt_enum: Prompt enum from domain-specific registry.
            **prompt_kwargs: Additional kwargs for prompt template rendering.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        return Agent(
            name=planner_config.name,
            model=self.cfg.openai.model,
            tools=tools,
            instructions=self.prompt_registry.get_prompt(
                prompt_enum=prompt_enum,
                **prompt_kwargs,
            ),
            # Disable parallel tool calls to prevent race conditions on shared
            # sessions (designer_session, critic_session). When the model returns
            # multiple tool calls in one response, they would otherwise run
            # concurrently and cause SQLite locking issues.
            model_settings=self._get_model_settings(
                settings_key="planner", parallel_tool_calls=False
            ),
        )

    def _create_sessions(self, session_prefix: str = "") -> tuple[Session, Session]:
        """Create designer and critic sessions for persistent conversation history.

        Sessions are optionally wrapped with TurnTrimmingSession for memory
        management if session_memory is enabled in config.

        Args:
            session_prefix: Optional prefix for session IDs (e.g., furniture ID).

        Returns:
            Tuple of (designer_session, critic_session).
        """
        designer_id = f"{session_prefix}designer" if session_prefix else "designer"
        critic_id = f"{session_prefix}critic" if session_prefix else "critic"

        designer_sqlite = SQLiteSession(
            session_id=designer_id,
            db_path=self.logger.output_dir / f"{designer_id}.db",
        )
        critic_sqlite = SQLiteSession(
            session_id=critic_id,
            db_path=self.logger.output_dir / f"{critic_id}.db",
        )

        # Wrap with memory management if configured.
        memory_cfg = self.cfg.session_memory
        if memory_cfg and memory_cfg.enabled:
            console_logger.info(
                f"Enabling turn-trimming session (keep_last_n_turns="
                f"{memory_cfg.keep_last_n_turns}, summarization="
                f"{memory_cfg.enable_summarization})"
            )
            designer_session: Session = TurnTrimmingSession(
                wrapped_session=designer_sqlite, cfg=self.cfg
            )
            critic_session: Session = TurnTrimmingSession(
                wrapped_session=critic_sqlite, cfg=self.cfg
            )
        else:
            designer_session = designer_sqlite
            critic_session = critic_sqlite

        return designer_session, critic_session

    def _create_run_config(self) -> RunConfig:
        """Create RunConfig with model input filters.

        Intra-turn stripping reduces token usage when agents call observe_scene
        multiple times within a turn. The Chat Completions image filter keeps
        image-returning tools usable if the SDK is configured away from the
        default Responses API.

        Returns:
            RunConfig with call_model_input_filter set.
        """
        input_filters = []
        intra_cfg = self.cfg.session_memory.intra_turn_observation_stripping
        if intra_cfg.enabled:
            input_filters.append(IntraTurnImageFilter(cfg=self.cfg))

        input_filters.append(ChatCompletionsToolImageFilter())

        harness = LLMHarnessConfig.from_env(default_model=self.cfg.openai.model)
        kwargs: dict[str, Any] = {
            "call_model_input_filter": CompositeCallModelInputFilter(input_filters)
        }
        provider = agents_model_provider(harness)
        # RunConfig's default provider is a sentinel-backed SDK provider. Passing
        # explicit None replaces it and fails later at get_model().
        if provider is not None:
            kwargs["model_provider"] = provider
        return RunConfig(**kwargs)
