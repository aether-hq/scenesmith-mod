"""Regression tests for the auto-discovered semantic scene gallery."""

import json
import math
import subprocess
import sys
import tempfile
import unittest

from pathlib import Path

from examples.semantic_gallery.generate_gallery import (
    _camera_hint,
    discover_control_paths,
    discover_trial_paths,
    generate_gallery,
    rebuild_gallery,
)
from examples.semantic_gallery.serve_gallery import manifest_asset_paths
from scenesmith.agent_utils.semantic_environments import SemanticEnvironmentSpec

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRIAL_DIRECTORY = (
    REPOSITORY_ROOT / "docs" / "geometry-extension" / "llm-trials" / "results"
)
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "examples" / "semantic_gallery"
CONTROL_DIRECTORY = EXAMPLE_DIRECTORY / "sources"


class TestSemanticGallery(unittest.TestCase):
    def test_bar_controls_include_full_fidelity_visual_and_semantic_packet(
        self,
    ) -> None:
        discovered = discover_control_paths(CONTROL_DIRECTORY)

        self.assertEqual(
            discovered,
            (
                CONTROL_DIRECTORY / "original_aether_bar.json",
                CONTROL_DIRECTORY / "original_scenesmith_bar.json",
            ),
        )
        source = json.loads(discovered[0].read_text(encoding="utf-8"))
        self.assertEqual(
            source["source"]["packet_sha256"],
            "7a6405a637b8ac9438c1f244125dca5f23ec512b720a2a47771ec9c8d179fc0f",
        )
        self.assertEqual(source["shell"]["room_id"], "public-bar")
        self.assertEqual(source["shell"]["dimensions_m"], [15.45, 4.8, 10.61])
        self.assertEqual(len(source["shell"]["openings"]), 3)
        self.assertEqual(len(source["placements"]), 104)
        self.assertEqual(len(source["cameras"]), 4)

        visual = json.loads(discovered[1].read_text(encoding="utf-8"))
        self.assertEqual(visual["id"], "original_scenesmith_bar")
        self.assertEqual(visual["source"]["kind"], "pinned_gltf_visual_baseline")
        self.assertEqual(
            visual["source"]["artifact_sha256"],
            "90dad0948e638aa07c400ae2ea6d34cceb3ba259d59ea1656043691435d02f1d",
        )
        self.assertEqual(visual["expected"]["mesh_instances"], 280)
        self.assertEqual(visual["expected"]["triangles"], 187086)
        self.assertGreaterEqual(visual["expected"]["materials"], 50)
        self.assertGreaterEqual(visual["expected"]["textures"], 50)

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
            expected_ids.add("original_aether_bar")
            expected_ids.add("original_scenesmith_bar")
            unavailable_ids = {
                record["trial_id"] for record in records if record["result"] != "PASS"
            }
            self.assertEqual(
                {scene["id"] for scene in manifest["scenes"]}, expected_ids
            )
            self.assertEqual(manifest["scene_count"], len(expected_ids))
            control_start = manifest["trial_count"]
            self.assertEqual(
                manifest["scenes"][control_start]["id"], "original_scenesmith_bar"
            )
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
                    if scene["representation"] == "full_fidelity_gltf":
                        asset = scene["scene_asset"]
                        self.assertEqual(asset["format"], "glb")
                        self.assertTrue((output / asset["path"]).is_file())
                        self.assertEqual(len(asset["sha256"]), 64)
                    else:
                        self.assertTrue(
                            (output / scene["shell"]["mesh_path"]).is_file()
                        )
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

            bar = next(
                scene
                for scene in manifest["scenes"]
                if scene["id"] == "original_aether_bar"
            )
            self.assertEqual(bar["source_kind"], "accepted_aether_room_packet")
            self.assertEqual(
                bar["compiler"],
                "scenesmith.agent_utils.structural_compiler.compile_polygon_space",
            )
            self.assertEqual(bar["representation"], "semantic_proxy_diagnostic")
            self.assertEqual(bar["shell"]["triangles"], 44)
            self.assertEqual(
                sum(item["instance_count"] for item in bar["details"]), 104
            )
            self.assertEqual(
                {item["material_key"] for item in bar["details"]},
                {"dressing", "fixtures", "furniture", "practical_light"},
            )
            self.assertEqual(
                bar["summary_metrics"],
                [
                    {"label": "rooms", "value": 1},
                    {"label": "portals", "value": 3},
                    {"label": "placements", "value": 104},
                    {"label": "cameras", "value": 4},
                    {"label": "reference meshes", "value": 149},
                    {"label": "reference tris", "value": 59560},
                ],
            )

            visual = next(
                scene
                for scene in manifest["scenes"]
                if scene["id"] == "original_scenesmith_bar"
            )
            self.assertEqual(visual["representation"], "full_fidelity_gltf")
            self.assertEqual(visual["scene_asset"]["expected_mesh_instances"], 280)
            self.assertEqual(visual["scene_asset"]["expected_triangles"], 187086)
            self.assertEqual(
                visual["scene_asset"]["sha256"],
                "90dad0948e638aa07c400ae2ea6d34cceb3ba259d59ea1656043691435d02f1d",
            )

    def test_clean_rebuild_publishes_source_and_provider_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "generated"
            output.mkdir()
            (output / "stale.obj").write_text("must disappear")

            manifest = rebuild_gallery(
                output,
                trial_directory=TRIAL_DIRECTORY,
                control_directory=CONTROL_DIRECTORY,
            )

            self.assertFalse((output / "stale.obj").exists())
            self.assertEqual(
                json.loads((output / "manifest.json").read_text()),
                manifest,
            )
            for scene in manifest["scenes"]:
                with self.subTest(scene=scene["id"]):
                    build = scene["build"]
                    self.assertTrue((REPOSITORY_ROOT / build["source_path"]).is_file())
                    self.assertEqual(len(build["source_sha256"]), 64)
                    self.assertTrue(build["provider"])
                    self.assertTrue(build["compiler_version"])
            visual = next(
                scene
                for scene in manifest["scenes"]
                if scene["id"] == "original_scenesmith_bar"
            )
            self.assertFalse(visual["build"]["rebuilt_from_recipe"])
            self.assertEqual(visual["build"]["status"], "reference_only")
            self.assertTrue(
                all(
                    scene["build"]["rebuilt_from_recipe"]
                    for scene in manifest["scenes"]
                    if scene["id"] != "original_scenesmith_bar"
                )
            )

    def test_server_manifest_preflight_resolves_all_gallery_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "generated"
            manifest = generate_gallery(output, trial_directory=TRIAL_DIRECTORY)

            paths = manifest_asset_paths(output, manifest)

            expected_count = sum(
                (1 if scene.get("scene_asset") else 1 + len(scene["details"]))
                for scene in manifest["scenes"]
            )
            self.assertEqual(len(paths), expected_count)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(all(output.resolve() in path.parents for path in paths))

    def test_viewer_loads_scene_list_from_manifest_without_trial_ids(self) -> None:
        viewer = (EXAMPLE_DIRECTORY / "viewer.html").read_text(encoding="utf-8")

        self.assertIn("generated/manifest.json", viewer)
        self.assertIn("manifest.scenes", viewer)
        self.assertIn("PointerLockControls", viewer)
        self.assertIn("GLTFLoader", viewer)
        self.assertIn("scene.scene_asset", viewer)
        self.assertIn("scene.shell.mesh_path", viewer)
        self.assertIn("scene.details", viewer)
        self.assertIn("scene.summary_metrics", viewer)
        self.assertIn("detail.material_key", viewer)
        self.assertIn("headLight.position.copy(camera.position)", viewer)
        self.assertIn("configureAtmosphere(scene)", viewer)
        self.assertIn("replaceChildren", viewer)
        self.assertNotIn("button.innerHTML", viewer)
        self.assertNotIn("heldout_branching_network_v1", viewer)
        self.assertNotIn("heldout_dragon_scale_cavern_v1", viewer)
        self.assertNotIn("original_aether_bar", viewer)

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

    def test_server_can_override_checked_in_control_directory(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE_DIRECTORY / "serve_gallery.py"),
                "--help",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--controls-dir", result.stdout)

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

        self.assertGreater(math.dist(hint["position"], [10, 22, 30]), 0.5)
        self.assertLess(math.dist(hint["position"], [10, 22, 30]), 2.1)
        self.assertGreater(math.dist(hint["target"], hint["position"]), 1.0)

    def test_passage_camera_starts_at_eye_height_above_authored_floor(self) -> None:
        record = json.loads(
            (TRIAL_DIRECTORY / "heldout_branching_network_v1.json").read_text(
                encoding="utf-8"
            )
        )
        environment = SemanticEnvironmentSpec.from_dict(record["semantic_environment"])

        hint = _camera_hint(environment, [-40, -50, -30], [40, 50, 20])

        first_floor_span = [16.0, -14.0, -11.0]
        span_length = math.dist([0.0, 0.0, 0.0], first_floor_span)
        tangent = [value / span_length for value in first_floor_span]
        across = [-tangent[1], tangent[0], 0.0]
        passage_vertical = [
            tangent[1] * across[2] - tangent[2] * across[1],
            tangent[2] * across[0] - tangent[0] * across[2],
            tangent[0] * across[1] - tangent[1] * across[0],
        ]
        floor_at_camera = [value * (1.25 / span_length) for value in first_floor_span]
        clearance = sum(
            (hint["position"][axis] - floor_at_camera[axis]) * passage_vertical[axis]
            for axis in range(3)
        )
        self.assertGreater(clearance, 1.0)
        self.assertGreater(math.dist(hint["position"], hint["target"]), 5.0)

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
