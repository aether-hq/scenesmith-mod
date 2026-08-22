import hashlib
import math
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from mathutils import Matrix

from scenesmith.agent_utils.blender import BlenderRenderer, RenderParams


class TestRenderParams(unittest.TestCase):
    """Test cases for RenderParams dataclass."""

    def test_render_params_creation(self):
        """Test creating RenderParams with required fields."""
        params = RenderParams(
            scene=Path("/tmp/test.gltf"),
            scene_sha256="abc123",
            image_type="color",
            width=640,
            height=480,
            near=0.1,
            far=100.0,
            focal_x=320.0,
            focal_y=320.0,
            fov_x=1.047,
            fov_y=0.785,
            center_x=320.0,
            center_y=240.0,
        )

        self.assertEqual(params.scene, Path("/tmp/test.gltf"))
        self.assertEqual(params.scene_sha256, "abc123")
        self.assertEqual(params.image_type, "color")
        self.assertEqual(params.width, 640)
        self.assertEqual(params.height, 480)
        self.assertEqual(params.near, 0.1)
        self.assertEqual(params.far, 100.0)
        self.assertEqual(params.focal_x, 320.0)
        self.assertEqual(params.focal_y, 320.0)
        self.assertEqual(params.fov_x, 1.047)
        self.assertEqual(params.fov_y, 0.785)
        self.assertEqual(params.center_x, 320.0)
        self.assertEqual(params.center_y, 240.0)
        self.assertIsNone(params.min_depth)
        self.assertIsNone(params.max_depth)


class TestBlenderRenderer(unittest.TestCase):
    """Test cases for BlenderRenderer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.blend_file = self.temp_dir / "test.blend"
        self.settings_file = self.temp_dir / "settings.py"

    def test_blender_renderer_init(self):
        """Test BlenderRenderer initialization."""
        renderer = BlenderRenderer()
        self.assertIsNone(renderer._blend_file)
        self.assertIsNone(renderer._bpy_settings_file)
        self.assertIsNone(renderer._client_objects)

    def test_blender_renderer_init_with_files(self):
        """Test BlenderRenderer initialization with blend and settings files."""
        renderer = BlenderRenderer(
            blend_file=self.blend_file,
            bpy_settings_file=self.settings_file,
        )
        self.assertEqual(renderer._blend_file, self.blend_file)
        self.assertEqual(renderer._bpy_settings_file, self.settings_file)

    @patch("scenesmith.agent_utils.blender.renderer.bpy")
    def test_save_blend_file_imports_supplemental_structural_visuals(self, mock_bpy):
        visual_path = self.temp_dir / "renaissance_gallery.glb"
        visual_path.touch()
        imported_root = MagicMock()
        imported_root.parent = None
        imported_root.matrix_world = Matrix.Identity(4)
        imported_root.type = "EMPTY"
        imported_mesh = MagicMock()
        imported_mesh.parent = imported_root
        imported_mesh.type = "MESH"
        mock_bpy.context.selected_objects = [imported_root, imported_mesh]
        renderer = BlenderRenderer()
        params = RenderParams(
            scene=Path("/tmp/test.gltf"),
            scene_sha256="abc123",
            image_type="color",
            width=4,
            height=4,
            near=0.1,
            far=100.0,
            focal_x=2.0,
            focal_y=2.0,
            fov_x=1.0,
            fov_y=1.0,
            center_x=2.0,
            center_y=2.0,
        )

        with (
            patch.object(renderer, "_setup_scene"),
            patch.object(renderer, "_import_and_organize_gltf"),
        ):
            renderer.save_blend_file(
                params,
                self.blend_file,
                additional_visuals=[
                    {
                        "path": str(visual_path),
                        "translation": [10.0, 20.0, 3.0],
                        "yaw_radians": math.pi / 2,
                        "role": "structural_detail",
                        "source_id": "renaissance_gallery",
                    }
                ],
            )

        mock_bpy.ops.import_scene.gltf.assert_called_once_with(
            filepath=str(visual_path)
        )
        self.assertEqual(
            tuple(imported_root.matrix_world.translation), (10.0, 20.0, 3.0)
        )
        imported_mesh.__setitem__.assert_any_call("aether_role", "structural_detail")
        imported_mesh.__setitem__.assert_any_call(
            "aether_source_id", "renaissance_gallery"
        )
        mock_bpy.ops.wm.save_as_mainfile.assert_called_once_with(
            filepath=str(self.blend_file)
        )

    @patch("scenesmith.agent_utils.blender.surfaces.scene_setup_mixin.bpy")
    @patch("scenesmith.agent_utils.blender.renderer.bpy")
    def test_reset_scene(self, mock_renderer_bpy, mock_setup_bpy):
        """Test that reset_scene resets Blender scene and removes default objects."""
        # Mock the data objects in scene_setup_mixin (where reset_scene is defined).
        mock_setup_bpy.data.objects = [Mock(), Mock()]

        renderer = BlenderRenderer()
        renderer.reset_scene()

        # Should call factory settings read and delete objects.
        mock_setup_bpy.ops.wm.read_factory_settings.assert_called_once()
        mock_setup_bpy.ops.object.delete.assert_called_once()

        # Should select each object.
        for obj in mock_setup_bpy.data.objects:
            obj.select_set.assert_called_with(True)

    @patch("scenesmith.agent_utils.blender.renderer.bpy")
    def test_add_default_light_source(self, mock_bpy):
        """Test that add_default_light_source adds a point light."""
        mock_light = MagicMock()
        mock_light_object = MagicMock()
        mock_bpy.data.lights.new.return_value = mock_light
        mock_bpy.data.objects.new.return_value = mock_light_object

        renderer = BlenderRenderer()
        renderer.add_default_light_source()

        # Should create light and light object.
        mock_bpy.data.lights.new.assert_called_once_with(
            name="DefaultLight", type="POINT"
        )
        mock_bpy.data.objects.new.assert_called_once_with(
            name="DefaultLight", object_data=mock_light
        )
        self.assertEqual(mock_light.energy, 1000)
        self.assertEqual(mock_light_object.location, (4.0, 1.0, 6.0))

    @patch("scenesmith.agent_utils.blender.surfaces.scene_setup_mixin.bpy")
    @patch("scenesmith.agent_utils.blender.render_settings.bpy")
    @patch("scenesmith.agent_utils.blender.geometry.camera_utils.bpy")
    @patch("scenesmith.agent_utils.blender.renderer.bpy")
    def test_render_image_creates_output_file(
        self, mock_renderer_bpy, mock_camera_bpy, mock_settings_bpy, mock_setup_bpy
    ):
        """Test that render_image creates rendered output file."""
        # Create mock scene file with correct checksum.
        test_scene_content = b"test gltf content"
        test_scene_path = Path("/tmp/test.gltf")

        # Mock the file reading and bpy imports.
        with patch.object(Path, "read_bytes", return_value=test_scene_content):
            # Calculate correct checksum.
            correct_sha256 = hashlib.sha256(test_scene_content).hexdigest()

            renderer = BlenderRenderer()
            params = RenderParams(
                scene=test_scene_path,
                scene_sha256=correct_sha256,
                image_type="color",
                width=640,
                height=480,
                near=0.1,
                far=100.0,
                focal_x=320.0,
                focal_y=320.0,
                fov_x=1.047,
                fov_y=0.785,
                center_x=320.0,
                center_y=240.0,
            )
            output_path = Path("/tmp/output.png")

            # Mock scene and camera objects for all modules.
            mock_scene = Mock()
            mock_scene.render = Mock()
            mock_scene.render.resolution_x = None
            mock_scene.render.resolution_y = None
            mock_scene.render.filepath = None

            # Mock world nodes for setup_regular_world().
            mock_world = Mock()
            mock_world.use_nodes = False
            mock_scene.world = mock_world
            mock_bg_node = Mock()
            mock_bg_input = Mock()
            mock_bg_input.default_value = None
            mock_bg_node.inputs = {0: mock_bg_input}
            mock_node_tree = Mock()
            mock_node_tree.nodes.get = Mock(return_value=mock_bg_node)
            mock_world.node_tree = mock_node_tree

            mock_camera_data = Mock()
            mock_camera_object = Mock()

            # Mock collections for GLTF import (_import_and_organize_gltf).
            mock_collection = Mock()
            mock_collection.objects = (
                []
            )  # Must be iterable for disable_backface_culling.
            mock_setup_bpy.data.collections.new.return_value = mock_collection
            mock_setup_bpy.context.selected_objects = []

            # Set up bpy mocks for all modules.
            for mock_bpy in [
                mock_renderer_bpy,
                mock_camera_bpy,
                mock_settings_bpy,
                mock_setup_bpy,
            ]:
                mock_bpy.context.scene = mock_scene
                mock_bpy.data.cameras.new.return_value = mock_camera_data
                mock_bpy.data.objects.new.return_value = mock_camera_object
                mock_bpy.data.worlds.new = Mock(return_value=mock_world)

            renderer.render_image(params, output_path)

            # Should call import of gltf scene in scene_setup_mixin.
            mock_setup_bpy.ops.import_scene.gltf.assert_called_once_with(
                filepath=str(test_scene_path)
            )

            # Should set render parameters.
            self.assertEqual(mock_scene.render.resolution_x, 640)
            self.assertEqual(mock_scene.render.resolution_y, 480)
            self.assertEqual(mock_scene.render.filepath, str(output_path))

            # Should call render.
            mock_renderer_bpy.ops.render.render.assert_called_once_with(
                write_still=True
            )

            # Should set up collections and rotation in scene_setup_mixin.
            mock_setup_bpy.data.collections.new.assert_called()
            mock_setup_bpy.ops.transform.rotate.assert_called_once()
