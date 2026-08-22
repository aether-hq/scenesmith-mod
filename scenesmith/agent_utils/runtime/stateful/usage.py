"""Token-usage reporting shared by stateful-agent workflow components."""

import logging

from agents import RunResult

console_logger = logging.getLogger(__name__)


def log_agent_usage(result: RunResult, agent_name: str) -> None:
    """Log aggregate and final-context token usage for one agent run."""

    usage = result.context_wrapper.usage
    cached = (
        usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    )
    reasoning = (
        usage.output_tokens_details.reasoning_tokens
        if usage.output_tokens_details
        else 0
    )
    final_context = (
        usage.request_usage_entries[-1].input_tokens
        if usage.request_usage_entries
        else usage.input_tokens
    )
    console_logger.info(
        "[%s] Token usage: input=%s, output=%s, reasoning=%s, cached=%s, "
        "total=%s, requests=%s, final_context_length=%s",
        agent_name,
        f"{usage.input_tokens:,}",
        f"{usage.output_tokens:,}",
        f"{reasoning:,}",
        f"{cached:,}",
        f"{usage.total_tokens:,}",
        usage.requests,
        f"{final_context:,}",
    )
