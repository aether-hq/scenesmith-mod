import io
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from mathutils import Vector

from scenesmith.agent_utils.blender import (
    BlenderRenderApp,
    BlenderRenderer,
    RenderParams,
)
from scenesmith.agent_utils.blender.geometry.scene_utils import get_floor_bounds
from scenesmith.agent_utils.blender.overlays.coordinate_frame import (
    create_coordinate_frame,
)
from scenesmith.agent_utils.blender.overlays.image_annotations import (
    annotate_image_with_coordinates,
)


class TestMetricRendering(unittest.TestCase):
    """Test cases for metric rendering functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_render_endpoint_uses_standard_rendering(self):
        """Test that /render endpoint uses standard rendering only."""
        app = BlenderRenderApp(temp_dir=self.temp_dir)

        # Mock the parse_params to return standard image type.
        mock_params = RenderParams(
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

        with app.test_request_context("/render", method="POST"):
            with patch.object(app, "_parse_params", return_value=mock_params):
                with patch.object(app, "_render") as mock_standard_render:
                    mock_buffer = io.BytesIO(b"fake_png_data")
                    mock_standard_render.return_value = mock_buffer

                    app._render_endpoint()

                    # Standard endpoint should only call standard rendering.
                    mock_standard_render.assert_called_once_with(mock_params)

    @patch("scenesmith.agent_utils.blender.overlays.coordinate_frame.bpy")
    def test_metric_overlays_add_coordinate_frame_and_grid(self, mock_bpy):
        """Test that metric overlays add coordinate frame and grid markers."""
        bbox_center = Vector((0, 0, 0))
        max_dim = 10.0

        # Mock Blender primitive operations and object creation.
        mock_objects = []

        def create_mock_object(name_prefix):
            mock_obj = Mock()
            mock_obj.name = f"{name_prefix}_{len(mock_objects)}"
            mock_obj.data = Mock()
            mock_obj.data.materials = Mock()
            mock_obj.data.materials.append = Mock()
            mock_obj.rotation_mode = "QUATERNION"
            mock_obj.rotation_quaternion = Mock()
            mock_objects.append(mock_obj)
            return mock_obj

        # Mock primitive creation operations.
        def mock_cylinder_add(**_kwargs):
            obj = create_mock_object("cylinder")
            mock_bpy.context.active_object = obj

        def mock_cone_add(**_kwargs):
            obj = create_mock_object("cone")
            mock_bpy.context.active_object = obj

        def mock_text_add(**_kwargs):
            obj = create_mock_object("text")
            mock_bpy.context.active_object = obj

        mock_bpy.ops.mesh.primitive_cylinder_add = mock_cylinder_add
        mock_bpy.ops.mesh.primitive_cone_add = mock_cone_add
        mock_bpy.ops.object.text_add = mock_text_add

        # Mock camera and scene.
        mock_camera = Mock()
        mock_camera.location = Vector((0, 0, 10))
        mock_scene = Mock()
        mock_scene.camera = mock_camera
        mock_bpy.context.scene = mock_scene

        # Mock materials with proper node structure.
        mock_material = Mock()
        mock_material.use_nodes = True
        mock_material.node_tree = Mock()
        mock_bsdf = Mock()
        mock_bsdf.inputs = Mock()
        mock_bsdf.inputs.__getitem__ = Mock(return_value=Mock())
        mock_material.node_tree.nodes = Mock()
        mock_material.node_tree.nodes.get = Mock(return_value=mock_bsdf)
        mock_bpy.data.materials.new = Mock(return_value=mock_material)
        mock_bpy.context.collection.objects.link = Mock()

        # Call the extracted function directly.
        create_coordinate_frame(
            position=bbox_center,
            max_dim=max_dim,
        )

        # Should create coordinate frame objects (3 axes, each with shaft + tip = 6
        # objects minimum).
        self.assertGreaterEqual(len(mock_objects), 6)

        # Objects are automatically added to the active collection by bpy.ops commands.
        # Verify that materials were created for the coordinate frames (one per axis).
        self.assertGreaterEqual(mock_bpy.data.materials.new.call_count, 3)

    @patch("scenesmith.agent_utils.blender.geometry.camera_utils.world_to_camera_view")
    def test_strategic_marker_placement_generates_exactly_nine_markers(
        self, mock_world_to_camera
    ):
        """Test that strategic marker placement generates exactly 9 markers."""
        renderer = BlenderRenderer()

        # Ensure _surface_corners is None to use floor bounds mode.
        renderer._surface_corners = None

        # Mock client objects for floor bounds.
        mock_mesh = Mock()
        mock_mesh.type = "MESH"
        mock_mesh.bound_box = [
            (-3, -2, 0),
            (3, -2, 0),
            (-3, 2, 0),
            (3, 2, 0),
            (-3, -2, 2),
            (3, -2, 2),
            (-3, 2, 2),
            (3, 2, 2),
        ]
        mock_mesh.matrix_world = Mock()
        mock_mesh.matrix_world.__matmul__ = lambda self, corner: Vector(corner)

        mock_objects = Mock()
        mock_objects.objects = [mock_mesh]
        renderer._client_objects = mock_objects

        # Mock world_to_camera_view to return normalized coords.
        mock_world_to_camera.return_value = Vector((0.5, 0.5, 1))

        # Mock scene with render resolution.
        mock_scene = Mock()
        mock_scene.render.resolution_x = 1920
        mock_scene.render.resolution_y = 1080
        mock_camera = Mock()
        visual_marks = renderer._get_visual_marks(mock_scene, mock_camera)

        # Should generate exactly 9 strategic positions.
        self.assertEqual(len(visual_marks), 9)

        # Check that we have the expected coordinate positions.
        expected_positions = [
            (-3.0, -2.0),  # bottom-left
            (3.0, -2.0),  # bottom-right
            (-3.0, 2.0),  # top-left
            (3.0, 2.0),  # top-right
            (0.0, -2.0),  # bottom-center
            (0.0, 2.0),  # top-center
            (-3.0, 0.0),  # left-center
            (3.0, 0.0),  # right-center
            (0.0, 0.0),  # center
        ]

        actual_positions = set(visual_marks.keys())
        expected_positions_set = set(expected_positions)
        self.assertEqual(actual_positions, expected_positions_set)

    @patch("scenesmith.agent_utils.blender.geometry.camera_utils.world_to_camera_view")
    def test_half_meter_precision_rounding(self, mock_world_to_camera):
        """Test that strategic markers use floor bounds directly without rounding."""
        renderer = BlenderRenderer()

        # Ensure _surface_corners is None to use floor bounds mode.
        renderer._surface_corners = None

        # Mock client objects with floor bounds.
        mock_mesh = Mock()
        mock_mesh.type = "MESH"
        mock_mesh.bound_box = [
            (-3.7, -2.3, 0),
            (3.7, -2.3, 0),
            (-3.7, 2.3, 0),
            (3.7, 2.3, 0),
            (-3.7, -2.3, 2),
            (3.7, -2.3, 2),
            (-3.7, 2.3, 2),
            (3.7, 2.3, 2),
        ]
        mock_mesh.matrix_world = Mock()
        mock_mesh.matrix_world.__matmul__ = lambda self, corner: Vector(corner)

        mock_objects = Mock()
        mock_objects.objects = [mock_mesh]
        renderer._client_objects = mock_objects

        # Mock world_to_camera_view to return normalized coords.
        mock_world_to_camera.return_value = Vector((0.5, 0.5, 1))

        # Mock scene with render resolution.
        mock_scene = Mock()
        mock_scene.render.resolution_x = 1920
        mock_scene.render.resolution_y = 1080
        mock_camera = Mock()
        visual_marks = renderer._get_visual_marks(mock_scene, mock_camera)

        # Should generate 9 markers.
        self.assertEqual(len(visual_marks), 9)

        # Coordinates should be derived from floor bounds (not rounded).
        # With floor bounds [-3.7, -2.3, 0, 3.7, 2.3], we expect:
        # center_x = 0.0, center_y = 0.0
        expected_x_values = {-3.7, 0.0, 3.7}
        expected_y_values = {-2.3, 0.0, 2.3}

        actual_x_values = {x for x, _ in visual_marks.keys()}
        actual_y_values = {y for _, y in visual_marks.keys()}

        # Check that x and y values match expectations (with floating point tolerance).
        for expected_x in expected_x_values:
            self.assertTrue(
                any(
                    abs(actual_x - expected_x) < 0.0001 for actual_x in actual_x_values
                ),
                f"Expected x value {expected_x} not found in {actual_x_values}",
            )

        for expected_y in expected_y_values:
            self.assertTrue(
                any(
                    abs(actual_y - expected_y) < 0.0001 for actual_y in actual_y_values
                ),
                f"Expected y value {expected_y} not found in {actual_y_values}",
            )

    def test_floor_bounds_detection_with_mesh_objects(self):
        """Test floor bounds detection from mesh objects."""
        # Mock mesh objects with bounding boxes.
        mock_mesh1 = Mock()
        mock_mesh1.type = "MESH"
        mock_mesh1.bound_box = [
            (-2, -1, 0),
            (2, -1, 0),
            (-2, 1, 0),
            (2, 1, 0),
            (-2, -1, 2),
            (2, -1, 2),
            (-2, 1, 2),
            (2, 1, 2),
        ]
        mock_mesh1.matrix_world = Mock()
        mock_mesh1.matrix_world.__matmul__ = lambda self, corner: Vector(corner)
        mock_mesh1.users_collection = []

        mock_mesh2 = Mock()
        mock_mesh2.type = "MESH"
        mock_mesh2.bound_box = [
            (-1, -3, 1),
            (1, -3, 1),
            (-1, 3, 1),
            (1, 3, 1),
            (-1, -3, 3),
            (1, -3, 3),
            (-1, 3, 3),
            (1, 3, 3),
        ]
        mock_mesh2.matrix_world = Mock()
        mock_mesh2.matrix_world.__matmul__ = lambda self, corner: Vector(corner)
        mock_mesh2.users_collection = []

        # Mock client objects.
        mock_objects = Mock()
        mock_objects.objects = [mock_mesh1, mock_mesh2]

        floor_bounds = get_floor_bounds(mock_objects)

        # Should find the lowest Z (floor level) and compute 2D bounds.
        self.assertEqual(len(floor_bounds), 5)
        min_x, min_y, floor_z, max_x, max_y = floor_bounds

        # Floor should be at Z=0 (lowest geometry).
        self.assertEqual(floor_z, 0)

        # 2D bounds should encompass all floor-level geometry.
        self.assertEqual(min_x, -2)  # Most negative X from mesh1
        self.assertEqual(max_x, 2)  # Most positive X from mesh1
        self.assertEqual(min_y, -1)  # Most negative Y from mesh1
        self.assertEqual(max_y, 1)  # Most positive Y from mesh1

    def test_floor_bounds_fallback_when_no_objects(self):
        """Test floor bounds raises ValueError when no mesh objects exist."""
        # No client objects should raise ValueError (fail-fast).
        with self.assertRaises(ValueError) as context:
            get_floor_bounds(None)

        self.assertIn("No client objects available", str(context.exception))

    @patch("scenesmith.agent_utils.blender.overlays.coordinate_frame.bpy")
    def test_camera_distance_fallback_in_test_environment(self, mock_bpy):
        """Test camera distance calculation fallback for test environments."""
        # Mock camera with invalid location (empty).
        mock_camera = Mock()
        mock_camera.location = []  # Empty location causes ValueError
        mock_scene = Mock()
        mock_scene.camera = mock_camera
        mock_bpy.context.scene = mock_scene

        bbox_center = Vector((0, 0, 0))
        max_dim = 10.0

        # Should not raise exception and use fallback distance.
        try:
            create_coordinate_frame(
                position=bbox_center,
                max_dim=max_dim,
            )
            # If we get here, fallback worked correctly
            self.assertTrue(True)
        except (ValueError, AttributeError):
            self.fail("Should use fallback camera distance in test environment")

    def test_coordinate_formatting_removes_unnecessary_decimals(self):
        """Test that coordinate formatting produces clean text without trailing zeros."""
        # Test the formatting logic used in coordinate display.
        test_cases = [
            (5.0, "5"),  # Should remove .0
            (5.5, "5.5"),  # Should keep .5
            (10.0, "10"),  # Should remove .0
            (3.25, "3.25"),  # Should keep .25
            (-2.0, "-2"),  # Should remove .0 for negatives
            (-1.5, "-1.5"),  # Should keep .5 for negatives
        ]

        for input_val, expected_str in test_cases:
            formatted_str = f"{input_val:g}"
            self.assertEqual(
                formatted_str,
                expected_str,
                f"Expected {input_val} to format as '{expected_str}', got "
                f"'{formatted_str}'",
            )

    def test_coordinate_annotation_method_exists_and_callable(self):
        """Test that coordinate annotation function exists and is callable."""
        # Test that the extracted function exists and can be called.
        self.assertTrue(callable(annotate_image_with_coordinates))

        # Test basic functionality without complex font mocking.
        with patch("PIL.Image.open") as mock_open:
            mock_pil_image = Mock()
            mock_pil_image.mode = "RGB"
            mock_pil_image.size = (100, 100)
            mock_open.return_value = mock_pil_image

            # Should not raise an exception when given empty marks.
            try:
                annotate_image_with_coordinates(
                    image_path=Path("/tmp/test.png"), marks={}
                )
                # If we get here, the function handled empty marks correctly.
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Function should handle empty visual marks gracefully: {e}")


if __name__ == "__main__":
    unittest.main()
