"""Tests for exhaustive, machine-readable sample rebuild inventory."""

import json
import tempfile
import unittest

from pathlib import Path

from examples.rebuild_all_samples import apply_render_report, rebuild_all_samples


class TestRebuildAllSamples(unittest.TestCase):
    def test_inventory_covers_runnable_and_model_gated_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "sample-build-manifest.json"

            manifest = rebuild_all_samples(
                gallery_output=root / "gallery",
                prison_output=root / "prison",
                manifest_path=manifest_path,
            )

            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            records = {record["id"]: record for record in manifest["samples"]}
            self.assertIn("heldout_branching_network_v1", records)
            self.assertIn("heldout_dragon_scale_cavern_v1", records)
            self.assertIn("original_scenesmith_bar", records)
            self.assertIn("original_aether_bar", records)
            self.assertIn("prison_escape_long_way_out", records)
            gated = [
                record
                for record in records.values()
                if record["support_status"] == "model_gated"
            ]
            self.assertEqual(len(gated), 3)
            self.assertTrue(
                all(record["render"]["status"] == "not_run" for record in gated)
            )
            self.assertTrue(all(record["diagnostics"] for record in gated))
            runnable = [
                record
                for record in records.values()
                if record["support_status"] == "runnable"
            ]
            self.assertEqual(len(runnable), 4)
            self.assertTrue(
                all(record["build"]["status"] == "compiled" for record in runnable)
            )
            self.assertTrue(
                all(record["render"]["status"] == "pending" for record in runnable)
            )
            baseline = records["original_scenesmith_bar"]
            self.assertEqual(baseline["support_status"], "reference_only")
            self.assertFalse(baseline["build"]["rebuilt_from_recipe"])
            self.assertEqual(manifest["summary"]["runnable"], 4)
            self.assertEqual(manifest["summary"]["reference_only"], 1)
            self.assertEqual(manifest["summary"]["model_gated"], 3)

            render_report = {
                "schema_version": 1,
                "provider": "test/webgl2",
                "renders": [
                    {
                        "id": record["id"],
                        "status": "passed",
                        "screenshot": str(root / f"{record['id']}.png"),
                    }
                    for record in records.values()
                    if record["render"]["status"] == "pending"
                ],
            }
            report_path = root / "render-report.json"
            report_path.write_text(json.dumps(render_report))

            finalized = apply_render_report(manifest_path, report_path)

            self.assertEqual(finalized["summary"]["rendered"], 5)
            self.assertEqual(finalized["summary"]["render_failed"], 0)
            self.assertTrue(
                all(
                    record["render"]["status"] == "passed"
                    for record in finalized["samples"]
                    if record["support_status"] != "model_gated"
                )
            )


if __name__ == "__main__":
    unittest.main()
