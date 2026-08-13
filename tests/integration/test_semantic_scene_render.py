"""Supported-host image-render gate for generated semantic architecture."""

import os
import tempfile
import unittest

from pathlib import Path

from examples.prison_escape.generate_scene import generate_scene

try:
    import bpy
    import cv2

    from mathutils import Vector
except ImportError:  # pragma: no cover - unsupported local environments
    bpy = cv2 = Vector = None


def _look_at(camera, target) -> None:
    camera.rotation_euler = (
        (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )


@unittest.skipUnless(
    os.environ.get("SCENESMITH_RUN_SEMANTIC_RENDER_TESTS") == "1",
    "semantic render gate is enabled only on the supported CI host",
)
@unittest.skipIf(bpy is None, "Blender Python is unavailable on this host")
class TestSemanticSceneRender(unittest.TestCase):
    def test_generated_tunnel_shell_produces_a_nonblank_architecture_render(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = generate_scene(root)
            mesh_path = root / manifest["tunnel"]["mesh_path"]
            render_path = root / "semantic_tunnel_gate.png"

            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
            shell = bpy.context.selected_objects[0]
            material = bpy.data.materials.new("semantic_rock")
            material.diffuse_color = (0.28, 0.16, 0.08, 1.0)
            shell.data.materials.append(material)

            camera_data = bpy.data.cameras.new("semantic_gate_camera")
            camera = bpy.data.objects.new("semantic_gate_camera", camera_data)
            bpy.context.scene.collection.objects.link(camera)
            camera.location = (12.0, 0.2, 1.2)
            camera_data.lens = 22.0
            camera_data.clip_end = 200.0
            _look_at(camera, (35.0, 2.5, -0.5))
            bpy.context.scene.camera = camera

            for index, position in enumerate(
                ((13.0, 0.0, 2.2), (27.0, 1.5, 1.0), (43.0, 4.0, 0.0))
            ):
                light_data = bpy.data.lights.new(f"gate_light_{index}", type="POINT")
                light_data.energy = 900.0
                light_data.color = (1.0, 0.65, 0.32)
                light_data.shadow_soft_size = 2.0
                light = bpy.data.objects.new(f"gate_light_{index}", light_data)
                bpy.context.scene.collection.objects.link(light)
                light.location = position

            scene = bpy.context.scene
            scene.render.engine = "BLENDER_EEVEE_NEXT"
            scene.render.resolution_x = 480
            scene.render.resolution_y = 270
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = str(render_path)
            if scene.world is None:
                scene.world = bpy.data.worlds.new("semantic_gate_world")
            scene.world.color = (0.005, 0.005, 0.005)
            bpy.ops.render.render(write_still=True)

            image = cv2.imread(str(render_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, (270, 480))
            self.assertGreater(float(image.std()), 3.0)
            self.assertGreater(float((image > 8).mean()), 0.01)


if __name__ == "__main__":
    unittest.main()
