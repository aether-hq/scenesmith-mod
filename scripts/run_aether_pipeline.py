#!/usr/bin/env python3
"""Run the mandatory native SceneSmith stages for one accepted Aether room.

This worker deliberately stops with a non-zero status after native realization
until the finished-scene postprocessor has produced and qualified its own receipt.
It never turns a five-stage core scene into a success-shaped finished environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CORE_STAGES = ("floor-plan", "furniture", "wall-mounted", "ceiling-mounted", "manipuland")
_CHECKPOINTS = {
    "floor-plan": "scene_000/house_layout.json",
    "furniture": "scene_000/room_*/scene_states/scene_after_furniture/scene_state.json",
    "wall-mounted": "scene_000/room_*/scene_states/scene_after_wall_objects/scene_state.json",
    "ceiling-mounted": "scene_000/room_*/scene_states/scene_after_ceiling_objects/scene_state.json",
    "manipuland": "scene_000/combined_house/house_state.json",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-input", required=True, type=Path)
    parser.add_argument("--pipeline-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-name", default="cpu_full_objaverse")
    return parser.parse_args()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"native stage evidence {pattern!r} resolved to {len(matches)} files in {root}"
        )
    return matches[0]


def _write_failure(output: Path, *, code: str, message: str) -> None:
    payload = {
        "contractVersion": 1,
        "state": "failed",
        "code": code,
        "message": message,
    }
    (output / "pipeline-failure.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    arguments = _arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    stage_input = json.loads(arguments.stage_input.read_text())
    manifest = json.loads(arguments.pipeline_manifest.read_text())
    if stage_input.get("realization_engine") != "scenesmith":
        raise RuntimeError("stage input did not select SceneSmith")
    if stage_input.get("pipeline_profile") != "full":
        raise RuntimeError("stage input did not select the full profile")
    if stage_input.get("people_allowed") is not False:
        raise RuntimeError("environment realization cannot include people")
    declared = tuple(manifest.get("required_stage_order", ()))
    if not all(stage in declared for stage in CORE_STAGES):
        raise RuntimeError("pipeline manifest omits a native SceneSmith stage")

    native_output = arguments.output / "native"
    command = [
        sys.executable,
        "main.py",
        "--config-name",
        arguments.config_name,
        f"hydra.run.dir={native_output}",
        f"+name=aether-{stage_input['job_id']}",
        f"experiment.pipeline.aether_stage_input={arguments.stage_input}",
        "experiment.pipeline.start_stage=floor_plan",
        "experiment.pipeline.stop_stage=manipuland",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["SCENESMITH_MODEL"] = environment.get(
        "SCENESMITH_MODEL", "openai/gpt-5-mini"
    )
    started = time.monotonic()
    with (arguments.output / "native-stdout.log").open("w") as stdout:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        _write_failure(
            arguments.output,
            code="scenesmith-native-stage-failed",
            message=f"native SceneSmith exited {completed.returncode}; inspect native-stdout.log",
        )
        raise SystemExit(completed.returncode)

    receipts = []
    previous_digest = hashlib.sha256(arguments.stage_input.read_bytes()).hexdigest()
    for stage in CORE_STAGES:
        evidence = _find_one(native_output, _CHECKPOINTS[stage])
        output_digest = _digest(evidence)
        receipts.append(
            {
                "name": stage,
                "state": "succeeded",
                "input_sha256": previous_digest,
                "output_sha256": output_digest,
                "implementation": "scenesmith-native",
                "seconds": 0.0,
                "findings": [f"evidence={evidence.relative_to(arguments.output)}"],
            }
        )
        previous_digest = output_digest
    core_receipt = {
        "contractVersion": 1,
        "jobId": stage_input["job_id"],
        "state": "succeeded",
        "executionBackend": "CPU",
        "cudaVisibleDevices": environment["CUDA_VISIBLE_DEVICES"],
        "seconds": round(time.monotonic() - started, 3),
        "stages": receipts,
    }
    (arguments.output / "scenesmith-core-receipt.json").write_text(
        json.dumps(core_receipt, indent=2) + "\n"
    )

    # Post stages are intentionally a separate hard gate.  This file proves the
    # native core completed, but cannot satisfy Genesis's finished receipt contract.
    _write_failure(
        arguments.output,
        code="finished-postprocessor-unavailable",
        message=(
            "all five native SceneSmith stages completed, but contextual completion, "
            "PBR/lighting, inspection/repair, and browser export have no qualified worker"
        ),
    )
    raise SystemExit(78)


if __name__ == "__main__":
    main()
