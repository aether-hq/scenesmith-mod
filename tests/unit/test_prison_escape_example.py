"""Regression test for the prison escape geometry showcase."""

import json
import tempfile
import unittest

from pathlib import Path

from examples.prison_escape.generate_scene import generate_scene


class TestPrisonEscapeExample(unittest.TestCase):
    def test_generates_long_lit_clear_escape_tunnel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            manifest = generate_scene(output_dir)

            self.assertGreater(manifest["tunnel"]["centerline_length_m"], 65)
            self.assertLessEqual(manifest["tunnel"]["end_elevation_m"], -5)
            self.assertGreaterEqual(manifest["lighting"]["fixture_count"], 8)
            self.assertEqual(manifest["verification"]["blocked_connectors"], [])
            self.assertIn("escape_outlet", manifest["verification"]["walk_reachable"])
            self.assertEqual(manifest["wall_breach"]["width_m"], 3.6)
            self.assertEqual(
                manifest["tunnel"]["semantic_source_id"], "long_way_out"
            )
            self.assertEqual(len(manifest["tunnel"]["environment_hash"]), 64)

            for relative_path in (
                "prison_escape.dmd.yaml",
                "structural_layout.json",
                "manifest.json",
                "preview.svg",
                "structures/rooms/prison_block/room_geometry_prison_block.sdf",
                "structures/meshes/escape_tunnel_shell/escape_tunnel_shell.sdf",
                "details/lights/ceiling_lights.sdf",
            ):
                self.assertTrue((output_dir / relative_path).is_file(), relative_path)

            exported = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(exported, manifest)
            layout = json.loads((output_dir / "structural_layout.json").read_text())
            self.assertIsNotNone(layout["semantic_environment"])
            self.assertEqual(layout["structural_meshes"], [])

    def test_web_viewer_references_generated_architecture(self) -> None:
        example_dir = Path(__file__).parents[2] / "examples" / "prison_escape"
        viewer = (example_dir / "viewer.html").read_text(encoding="utf-8")

        self.assertIn("PointerLockControls", viewer)
        self.assertIn("room_geometry_prison_block.obj", viewer)
        self.assertIn("escape_tunnel_shell.obj", viewer)
        self.assertIn("ceiling_lights.obj", viewer)
        self.assertIn("generated/manifest.json", viewer)
        self.assertIn("mode === 'walk' ? 'fly' : 'walk'", viewer)


if __name__ == "__main__":
    unittest.main()
