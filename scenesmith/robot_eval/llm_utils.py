"""Lightweight LLM utilities for robot_eval module.

Provides simple async structured LLM calls without Agent SDK overhead.
Uses OpenAI's structured output API for guaranteed schema compliance.
"""

import asyncio
import json

from typing import TypeVar

from pydantic import BaseModel

from scenesmith.agent_utils.vlm_service import VLMService

T = TypeVar("T", bound=BaseModel)

# Lazy-initialized provider-neutral harness facade.
_service: VLMService | None = None


def _get_service() -> VLMService:
    """Get the common SceneSmith LLM harness."""
    global _service
    if _service is None:
        _service = VLMService()
    return _service


async def structured_llm_call(
    model: str, system_prompt: str, user_input: str, output_type: type[T]
) -> T:
    """Make an async LLM call with structured Pydantic output.

    Uses OpenAI's structured output API which constrains the LLM to only
    produce valid schema-compliant output. Simpler than Agent SDK for
    single-turn LLM calls without tools.

    Args:
        model: Model name (e.g., "gpt-5.2").
        system_prompt: System instructions for the LLM.
        user_input: User message/query.
        output_type: Pydantic model class for structured output.

    Returns:
        Parsed Pydantic model instance.

    Raises:
        OpenAI API errors if the call fails.
    """
    schema = json.dumps(output_type.model_json_schema(), separators=(",", ":"))
    content = await asyncio.to_thread(
        _get_service().create_completion,
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{system_prompt}\nReturn JSON matching this schema: {schema}",
            },
            {"role": "user", "content": user_input},
        ],
        reasoning_effort="low",
        verbosity="low",
        response_format={"type": "json_object"},
    )
    return output_type.model_validate_json(content)
