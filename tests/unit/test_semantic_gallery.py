"""Regression tests for the auto-discovered semantic scene gallery."""

import json
import subprocess
import sys
import tempfile
import unittest

from pathlib import Path

from examples.semantic_gallery.generate_gallery import (
    _camera_hint,
    discover_trial_paths,
    generate_gallery,
)
from examples.semantic_gallery.serve_gallery import manifest_asset_paths
from scenesmith.agent_utils.semantic_environments import SemanticEnvironmentSpec

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRIAL_DIRECTORY = (
    REPOSITORY_ROOT / "docs" / "geometry-extension" / "llm-trials" / "results"
)
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "examples" / "semantic_gallery"


class TestSemanticGallery(unittest.TestCase):
    def test_discovery_includes_every_heldout_trial_but_not_summary(self) -> None:
        discovered = discover_trial_paths(TRIAL_DIRECTORY)

        self.assertEqual(
            discovered, tuple(sorted(TRIAL_DIRECTORY.glob("heldout_*.json")))
        )
        self.assertGreaterEqual(len(discovered), 2)
        self.assertNotIn(TRIAL_DIRECTORY / "summary.json", discovered)

    def test_gallery_compiles_every_discovered_scene_and_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "generated"
            manifest = generate_gallery(output, trial_directory=TRIAL_DIRECTORY)

            records = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in discover_trial_paths(TRIAL_DIRECTORY)
            ]
            expected_ids = {
                record["trial_id"] for record in records if record["result"] == "PASS"
            }
            unavailable_ids = {
                record["trial_id"] for record in records if record["result"] != "PASS"
            }
            self.assertEqual(
                {scene["id"] for scene in manifest["scenes"]}, expected_ids
            )
            self.assertEqual(manifest["scene_count"], len(expected_ids))
            self.assertEqual(
                {record["id"] for record in manifest["unavailable"]},
                unavailable_ids,
            )
            self.assertEqual(
                json.loads((output / "manifest.json").read_text(encoding="utf-8")),
                manifest,
            )
            self.assertEqual(
                generate_gallery(output, trial_directory=TRIAL_DIRECTORY), manifest
            )

            for scene in manifest["scenes"]:
                with self.subTest(scene=scene["id"]):
                    self.assertTrue((output / scene["shell"]["mesh_path"]).is_file())
                    self.assertEqual(len(scene["shell"]["artifact_hash"]), 64)
                    self.assertGreater(scene["shell"]["triangles"], 0)
                    self.assertEqual(len(scene["bounds"]["minimum"]), 3)
                    self.assertEqual(len(scene["bounds"]["maximum"]), 3)
                    for detail in scene["details"]:
                        self.assertTrue((output / detail["mesh_path"]).is_file())
                        self.assertEqual(len(detail["artifact_hash"]), 64)
                        self.assertGreater(detail["triangles"], 0)

            dragon = next(
                scene
                for scene in manifest["scenes"]
                if scene["id"] == "heldout_dragon_scale_cavern_v1"
            )
            self.assertEqual(len(dragon["details"]), 2)
            self.assertEqual(dragon["metrics"]["detail_instances"], 25)

    def test_server_manifest_preflight_resolves_all_gallery_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "generated"
            manifest = generate_gallery(output, trial_directory=TRIAL_DIRECTORY)

            paths = manifest_asset_paths(output, manifest)

            expected_count = sum(
                1 + len(scene["details"]) for scene in manifest["scenes"]
            )
            self.assertEqual(len(paths), expected_count)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(all(output.resolve() in path.parents for path in paths))

    def test_viewer_loads_scene_list_from_manifest_without_trial_ids(self) -> None:
        viewer = (EXAMPLE_DIRECTORY / "viewer.html").read_text(encoding="utf-8")

        self.assertIn("generated/manifest.json", viewer)
        self.assertIn("manifest.scenes", viewer)
        self.assertIn("PointerLockControls", viewer)
        self.assertIn("scene.shell.mesh_path", viewer)
        self.assertIn("scene.details", viewer)
        self.assertIn("replaceChildren", viewer)
        self.assertNotIn("button.innerHTML", viewer)
        self.assertNotIn("heldout_branching_network_v1", viewer)
        self.assertNotIn("heldout_dragon_scale_cavern_v1", viewer)

    def test_documented_scripts_are_directly_executable(self) -> None:
        for script in ("generate_gallery.py", "serve_gallery.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(EXAMPLE_DIRECTORY / script),
                        "--help",
                    ],
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    check=False,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_camera_hint_applies_owning_region_transform(self) -> None:
        environment = SemanticEnvironmentSpec.from_dict(
            {
                "schema_version": 1,
                "regions": [
                    {
                        "id": "translated_region",
                        "kind": "subterranean",
                        "bounds": {"minimum": [-5, -5, -5], "maximum": [5, 5, 5]},
                        "transform": {
                            "translation": [10, 20, 30],
                            "rotation_rpy": [0, 0, 1.5707963267948966],
                        },
                    }
                ],
                "chambers": [
                    {
                        "id": "entry_chamber",
                        "region_id": "translated_region",
                        "center": [2, 0, 0],
                        "size": [4, 4, 4],
                    }
                ],
            }
        )

        hint = _camera_hint(environment, [5, 15, 25], [15, 25, 35])

        for actual, expected in zip(hint["position"], [10, 22, 30], strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertGreater(hint["target"][1], hint["position"][1])

    def test_server_rejects_manifest_assets_outside_generated_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "generated"
            output.mkdir()
            outside = Path(temporary_directory) / "outside.obj"
            outside.touch()
            manifest = {
                "scenes": [
                    {
                        "shell": {"mesh_path": "../outside.obj"},
                        "details": [],
                    }
                ]
            }

            with self.assertRaisesRegex(ValueError, "escapes generated root"):
                manifest_asset_paths(output, manifest)


if __name__ == "__main__":
    unittest.main()
