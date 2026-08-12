#!/usr/bin/env python3
"""Build a Genesis-compatible census from a real SceneSmith checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenesmith.aether import build_scene_census


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-input", type=Path, required=True)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    census = build_scene_census(
        json.loads(args.stage_input.read_text()),
        json.loads(args.scene_state.read_text()),
        json.loads(args.validation_evidence.read_text()),
        round_index=args.round,
        scene_root=args.scene_state.parent,
    )
    args.output.write_text(json.dumps(census, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
