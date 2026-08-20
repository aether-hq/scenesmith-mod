"""Backward-compatible facade over SceneSmith's provider-neutral LLM harness."""

from __future__ import annotations

from typing import Any

from scenesmith.agent_utils.llm_harness import LLMHarness, LLMRequest


class VLMService:
    """Direct text/multimodal completion service.

    Callers provide one normalized request shape. Provider selection, capability
    checks, timeouts, retries, JSON normalization, and telemetry are owned by
    :class:`LLMHarness`; no SceneSmith feature code branches on a vendor.
    """

    def __init__(
        self,
        service_tier: str | None = None,
        *,
        harness: LLMHarness | None = None,
    ) -> None:
        self.harness = harness or LLMHarness(service_tier=service_tier)
        self.provider = self.harness.config.provider

    def create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        reasoning_effort: str,
        verbosity: str,
        response_format: dict[str, str] | None = None,
        vision_detail: str = "auto",
    ) -> str:
        """Execute a normalized, bounded LLM request.

        ``model`` remains in the public signature for compatibility with existing
        SceneSmith callers. The harness's configured model is authoritative so a
        resumed job cannot silently switch providers or models mid-stage.
        """
        return self.harness.complete(
            LLMRequest(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                response_format=response_format,
                vision_detail=vision_detail,
            )
        )
