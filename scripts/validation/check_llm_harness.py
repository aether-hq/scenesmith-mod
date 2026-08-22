#!/usr/bin/env python3
"""Live conformance probe for a configured SceneSmith multimodal LLM provider."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time

from agents import Agent, RunConfig, function_tool
from PIL import Image, ImageDraw, ImageFont

from scenesmith.agent_utils.llm.llm_harness import (
    LLMHarness,
    LLMHarnessConfig,
    LLMRequest,
    agents_model_provider,
    install_agents_runtime,
)
from scenesmith.agent_utils.runtime.agent_runtime import (
    BoundedRunner,
    agent_run_timeout_seconds,
)


def _probe_image() -> str:
    image = Image.new("RGB", (768, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 760, 248), outline="red", width=12)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 112)
    draw.text((115, 58), "SCENE42", fill="black", font=font, stroke_width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def run_probe() -> dict:
    started = time.monotonic()
    config = LLMHarnessConfig.from_env()
    model, capabilities = install_agents_runtime(config)
    harness = LLMHarness(config)

    vision_started = time.monotonic()
    vision_raw = await asyncio.to_thread(
        harness.complete,
        LLMRequest(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Read the image. Return JSON with exactly one key named "
                        "code whose value is the large alphanumeric text in it."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What code is visible?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_probe_image()}"
                            },
                        },
                    ],
                },
            ],
            reasoning_effort="low",
            verbosity="low",
            response_format={"type": "json_object"},
            vision_detail="high",
        ),
    )
    vision = json.loads(vision_raw)
    if str(vision.get("code", "")).upper() != "SCENE42":
        raise RuntimeError(f"Vision conformance failed: {vision}")

    tool_calls: list[str] = []

    @function_tool
    def record_probe(value: str) -> str:
        """Record the exact conformance token supplied by the model."""
        tool_calls.append(value)
        return "recorded"

    tool_started = time.monotonic()
    agent = Agent(
        name="SceneSmith harness conformance",
        model=model,
        instructions=(
            "Call record_probe exactly once with value SCENE42, then finish. "
            "Do not answer without using the tool."
        ),
        tools=[record_probe],
    )
    provider = agents_model_provider(config)
    run_config = RunConfig(model_provider=provider) if provider else RunConfig()
    await BoundedRunner.run(
        starting_agent=agent,
        input="Run the tool conformance probe now.",
        max_turns=3,
        run_config=run_config,
        timeout_seconds=agent_run_timeout_seconds(max_turns=3),
    )
    if tool_calls != ["SCENE42"]:
        raise RuntimeError(f"Tool conformance failed: calls={tool_calls}")

    return {
        "ok": True,
        "provider": config.provider,
        "model": config.model,
        "capabilities": {
            "vision": capabilities.vision,
            "tools": capabilities.tools,
            "structured_output": capabilities.structured_output,
        },
        "latency_seconds": {
            "vision_and_json": round(tool_started - vision_started, 3),
            "tool_loop": round(time.monotonic() - tool_started, 3),
            "total": round(time.monotonic() - started, 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine JSON")
    args = parser.parse_args()
    result = asyncio.run(run_probe())
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
