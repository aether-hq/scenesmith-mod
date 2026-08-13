"""The Aether worker may prove the native core without claiming finished success."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts" / "run_aether_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_aether_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AetherCpuWorkerTests(unittest.TestCase):
    def test_core_stage_contract_is_complete_and_ordered(self) -> None:
        worker = _module()
        self.assertEqual(
            worker.CORE_STAGES,
            (
                "floor-plan",
                "furniture",
                "wall-mounted",
                "ceiling-mounted",
                "manipuland",
            ),
        )
        self.assertEqual(tuple(worker._CHECKPOINTS), worker.CORE_STAGES)

    def test_failure_receipt_is_not_success_shaped(self) -> None:
        worker = _module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            worker._write_failure(
                output, code="postprocess-missing", message="not qualified"
            )
            payload = json.loads((output / "pipeline-failure.json").read_text())
        self.assertEqual(
            payload,
            {
                "contractVersion": 1,
                "state": "failed",
                "code": "postprocess-missing",
                "message": "not qualified",
            },
        )

    def test_capability_manifest_remains_unqualified_until_post_stages_exist(self) -> None:
        path = Path(__file__).parents[2] / "scripts" / "aether-pipeline-capabilities.json"
        payload = json.loads(path.read_text())
        self.assertEqual(payload["state"], "development")
        self.assertEqual(payload["qualified_execution_backends"], [])
        self.assertEqual(payload["blocking_stage"], "contextual-completion")


if __name__ == "__main__":
    unittest.main()
