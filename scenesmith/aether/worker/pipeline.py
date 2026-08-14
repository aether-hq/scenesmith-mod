"""In-process contextual completion over a live native RoomScene."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .author_client import AetherCompletionClient
from .completion_loop import run_completion_loop
from .runtime import NativeOperationFactory, SwitchingCompletionRuntime


def run_native_contextual_completion(
    *,
    scene: Any,
    stage_input: dict[str, Any],
    cfg_dict: dict[str, Any],
    logger: Any,
    house_layout: Any,
    ceiling_height: float,
    render_gpu_id: int | None,
    room_dir: Path,
) -> dict[str, Any]:
    """Measure, author, execute, and persist every bounded completion round."""
    required = {
        "AETHER_API_URL": os.environ.get("AETHER_API_URL"),
        "AETHER_PROJECT_ID": os.environ.get("AETHER_PROJECT_ID"),
        "AETHER_BEARER_TOKEN": os.environ.get("AETHER_BEARER_TOKEN"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise RuntimeError(
            "contextual completion requires attributed Aether inference: "
            + ", ".join(missing)
        )
    from ..physical_evidence import PhysicalEvidenceProvider, room_geometry_digest
    from ..scene_census import build_scene_census

    baseline = room_geometry_digest(scene)
    evidence = PhysicalEvidenceProvider(
        scene, stage_input, baseline_geometry_sha256=baseline
    )
    initial_census = build_scene_census(
        stage_input,
        scene.to_state_dict(),
        evidence(),
        round_index=0,
        scene_root=room_dir,
    )
    client = AetherCompletionClient(
        base_url=str(required["AETHER_API_URL"]),
        project_id=str(required["AETHER_PROJECT_ID"]),
        bearer_token=str(required["AETHER_BEARER_TOKEN"]),
        workspace_id=os.environ.get("AETHER_WORKSPACE_ID"),
        poll_interval_seconds=float(
            os.environ.get("AETHER_COMPLETION_POLL_SECONDS", "1")
        ),
        timeout_seconds=float(
            os.environ.get("AETHER_COMPLETION_TIMEOUT_SECONDS", "300")
        ),
    )
    operation_factory = NativeOperationFactory(
        scene=scene,
        stage_input=stage_input,
        cfg_dict=cfg_dict,
        logger=logger,
        house_layout=house_layout,
        ceiling_height=ceiling_height,
        render_gpu_id=render_gpu_id,
    )
    runtime = SwitchingCompletionRuntime(
        scene=scene,
        operation_factory=operation_factory,
        evidence_provider=evidence,
        style_context=stage_input["room_prompt"],
    )
    try:
        return run_completion_loop(
            stage_input,
            initial_census,
            author_patch=client.author_patch,
            runtime=runtime,
            artifact_root=room_dir / "contextual_completion",
            scene_root=room_dir,
        )
    finally:
        runtime.close()
