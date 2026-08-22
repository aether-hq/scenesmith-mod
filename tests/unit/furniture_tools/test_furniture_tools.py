import json
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.physics.validation.models import CollisionPair
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
from scenesmith.furniture_agents.tools.response_dataclasses import FurnitureErrorType


class BaseAgentToolsTest(unittest.TestCase):
    """Base class for agent tools tests with common setup/teardown."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_scene(self, objects=None, description="Test scene"):
        """Create a standard mock scene for testing."""
        mock_scene = Mock(spec=RoomScene)
        mock_scene.objects = objects or {}
        mock_scene.text_description = description
        mock_scene.action_log_path = None
        return mock_scene

    def create_mock_scene_object(
        self,
        name: str,
        object_type: ObjectType,
        position: list[float] = None,
        rotation: list[float] = None,
        sdf_path: str = None,
        geometry_path: str = None,
    ):
        """Create a minimal mock scene object with transform data.

        Args:
            name: Object name (e.g., "Table")
            object_type: ObjectType enum value
            position: [x, y, z] coordinates (defaults to [0, 0, 0])
            rotation: [roll, pitch, yaw] angles in radians (defaults to [0, 0, 0])
            sdf_path: Optional SDF file path
            geometry_path: Optional geometry file path

        Returns:
            Mock object with real Drake transform for testing.
        """
        position = position or [0.0, 0.0, 0.0]
        rotation = rotation or [0.0, 0.0, 0.0]

        mock = Mock()
        mock.name = name
        mock.object_type = object_type
        mock.description = f"Test {name.lower()}"
        mock.sdf_path = Path(sdf_path) if sdf_path else None
        mock.geometry_path = Path(geometry_path) if geometry_path else None

        # Use real Drake RigidTransform for compatibility with SimplifiedFurnitureInfo.
        rpy = RollPitchYaw(rotation[0], rotation[1], rotation[2])
        mock.transform = RigidTransform(rpy, position)

        # Add bounding box fields.
        mock.bbox_min = None
        mock.bbox_max = None

        # Add immutable field (defaults to False for regular furniture).
        mock.immutable = False

        return mock


class TestFurnitureTools(BaseAgentToolsTest):
    """Test FurnitureTools class contracts."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.mock_scene = self.create_mock_scene()
        self.mock_asset_manager = Mock()

        # Load base and specific configurations from actual config files.
        base_config_path = (
            Path(__file__).parents[3]
            / "configurations/furniture_agent/base_furniture_agent.yaml"
        )
        specific_config_path = (
            Path(__file__).parents[3]
            / "configurations/furniture_agent/stateful_furniture_agent.yaml"
        )
        base_config = OmegaConf.load(base_config_path)
        specific_config = OmegaConf.load(specific_config_path)

        # First merge base with specific config.
        merged_config = OmegaConf.merge(base_config, specific_config)

        # Define test overrides for fast testing.
        test_overrides = {
            "openai": {
                "model": "gpt-4o-mini",  # Cheaper model for testing
                "reasoning_effort": {
                    "planner": "low",  # Faster for tests
                    "designer": "low",
                    "critic": "low",
                },
                "verbosity": {
                    "planner": "low",
                    "designer": "low",
                    "critic": "low",
                },
            },
            "loop_detection": {
                "enabled": False,  # Disable loop detection for unit tests
            },
        }
        # Merge configurations (base config provides all other values).
        self.test_config = OmegaConf.merge(merged_config, test_overrides)

        self.furniture_tools = FurnitureTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=self.test_config,
        )

    def test_furniture_tools_initialization(self):
        """Test FurnitureTools initializes properly."""
        self.assertIsNotNone(self.furniture_tools)
        self.assertEqual(self.furniture_tools.scene, self.mock_scene)
        self.assertEqual(self.furniture_tools.asset_manager, self.mock_asset_manager)

    def test_tools_dictionary_exposed(self):
        """Test that tools dictionary is properly exposed."""
        self.assertTrue(hasattr(self.furniture_tools, "tools"))
        self.assertIsInstance(self.furniture_tools.tools, dict)

        # Should have furniture-related tools.
        tools_keys = list(self.furniture_tools.tools.keys())
        self.assertGreater(len(tools_keys), 0, "Should have at least one tool")

    def test_tools_are_callable(self):
        """Test that FurnitureTools follow standard tool interface."""
        for tool_name, tool_func in self.furniture_tools.tools.items():
            self.assertTrue(
                hasattr(tool_func, "on_invoke_tool"),
                f"FurnitureTools tool {tool_name} should have on_invoke_tool method",
            )

    def test_scene_modification(self):
        """Test that FurnitureTools provide scene modification capabilities."""
        self.assertTrue(
            len(self.furniture_tools.tools) > 0,
            "FurnitureTools should expose furniture manipulation tools",
        )

    def test_multiple_furniture_placements_unique_ids(self):
        """Test that multiple placements of the same asset create unique object IDs."""

        # Mock floor plan for bounds checking.
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0
        self.mock_scene.room_geometry.width = 10.0

        # Mock a scene object to return from the asset registry.
        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_chair_gen")
        mock_asset.name = "Chair"
        mock_asset.description = "Test chair"
        mock_asset.object_type = ObjectType.FURNITURE
        mock_asset.geometry_path = Path("/test/chair.obj")
        mock_asset.sdf_path = Path("/test/chair.sdf")
        mock_asset.image_path = None
        mock_asset.support_surfaces = []
        mock_asset.metadata = {}
        mock_asset.transform = RigidTransform()

        # Mock asset manager to return this asset.
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset
        self.mock_asset_manager.list_available_assets.return_value = [mock_asset]

        # Mock scene.add_object to capture what gets added.
        added_objects = []

        def capture_add_object(obj):
            added_objects.append(obj)

        self.mock_scene.add_object = capture_add_object

        # Mock generate_unique_id to return unique IDs for each call.
        self.mock_scene.generate_unique_id.side_effect = [
            UniqueID("chair"),
            UniqueID("chair_2"),
        ]

        # First placement.
        result1_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=0, y=0, z=0
        )

        # Second placement.
        result2_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=2, y=0, z=0
        )

        # Parse results.
        result1 = json.loads(result1_str)
        result2 = json.loads(result2_str)

        # Both should succeed.
        self.assertTrue(result1["success"], f"First placement failed: {result1}")
        self.assertTrue(result2["success"], f"Second placement failed: {result2}")

        # Object IDs should be different.
        obj_id1 = result1["object_id"]
        obj_id2 = result2["object_id"]
        self.assertNotEqual(
            obj_id1, obj_id2, "Multiple placements should create unique object IDs"
        )

        # Should have added two different objects to scene.
        self.assertEqual(len(added_objects), 2)
        self.assertNotEqual(
            str(added_objects[0].object_id), str(added_objects[1].object_id)
        )

    def _configure_candidate_collision_test(self) -> SceneObject:
        asset = SceneObject(
            object_id=UniqueID("renaissance_bookcase_asset"),
            object_type=ObjectType.FURNITURE,
            name="Renaissance bookcase",
            description="A full-height library bookcase",
            transform=RigidTransform(),
            geometry_path=Path("/test/bookcase.obj"),
            sdf_path=Path("/test/bookcase.sdf"),
            bbox_min=np.array([-0.5, -0.2, 0.0]),
            bbox_max=np.array([0.5, 0.2, 2.0]),
        )
        self.mock_asset_manager.get_asset_by_id.return_value = asset
        self.mock_asset_manager.list_available_assets.return_value = [asset]
        self.mock_scene.objects = {}
        self.mock_scene.generate_unique_id.return_value = UniqueID(
            "renaissance_bookcase"
        )
        self.mock_scene.add_object.side_effect = (
            lambda item: self.mock_scene.objects.__setitem__(item.object_id, item)
        )
        self.mock_scene.remove_object.side_effect = self.mock_scene.objects.pop
        self.mock_scene.get_object.side_effect = self.mock_scene.objects.get
        self.mock_scene.move_object.side_effect = (
            lambda object_id, new_transform: setattr(
                self.mock_scene.objects[object_id], "transform", new_transform
            )
        )
        self.mock_scene.get_objects_by_type.side_effect = lambda object_type: [
            item
            for item in self.mock_scene.objects.values()
            if item.object_type == object_type
        ]
        self.furniture_tools._check_floor_bounds = Mock(return_value=(True, ""))
        self.furniture_tools._surface_aligned_pose = Mock(return_value=None)
        self.furniture_tools._validate_spatial_envelope = Mock(return_value=(True, ""))
        self.furniture_tools._validate_contextual_zones = Mock(return_value=(True, ""))
        return asset

    @patch("scenesmith.furniture_agents.tools.furniture_tools.compute_scene_collisions")
    def test_add_rolls_back_candidate_local_structural_collision(
        self, compute_collisions
    ):
        asset = self._configure_candidate_collision_test()
        compute_collisions.return_value = [
            CollisionPair(
                object_a_name="floor",
                object_a_id="room_geometry",
                object_b_name="Renaissance bookcase",
                object_b_id="renaissance_bookcase",
                penetration_depth=4.9957,
            )
        ]

        result = json.loads(
            self.furniture_tools._add_furniture_to_scene_impl(
                asset_id=str(asset.object_id), x=2.002, y=1.0, z=0.0
            )
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.INVALID_POSITION.value
        )
        self.assertIn("room_geometry", result["message"])
        self.assertEqual(self.mock_scene.objects, {})

    @patch("scenesmith.furniture_agents.tools.furniture_tools.compute_scene_collisions")
    def test_add_accepts_floor_contact_filtered_by_physics_tolerance(
        self, compute_collisions
    ):
        asset = self._configure_candidate_collision_test()
        compute_collisions.return_value = []

        result = json.loads(
            self.furniture_tools._add_furniture_to_scene_impl(
                asset_id=str(asset.object_id), x=2.002, y=1.0, z=0.0
            )
        )

        self.assertTrue(result["success"], result)
        self.assertIn(UniqueID("renaissance_bookcase"), self.mock_scene.objects)
        compute_collisions.assert_called_once()
        self.assertEqual(
            compute_collisions.call_args.kwargs["floor_penetration_tolerance"],
            self.test_config.physics_validation.floor_penetration_tolerance_m,
        )

    @patch("scenesmith.furniture_agents.tools.furniture_tools.compute_scene_collisions")
    def test_move_rolls_back_candidate_local_structural_collision(
        self, compute_collisions
    ):
        asset = self._configure_candidate_collision_test()
        asset.object_id = UniqueID("renaissance_bookcase")
        self.mock_scene.objects[asset.object_id] = asset
        compute_collisions.return_value = [
            CollisionPair(
                object_a_name="wall",
                object_a_id="room_geometry",
                object_b_name=asset.name,
                object_b_id=str(asset.object_id),
                penetration_depth=0.25,
            )
        ]

        result = json.loads(
            self.furniture_tools._move_furniture_impl(
                object_id=str(asset.object_id),
                x=3.0,
                y=1.0,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
            )
        )

        self.assertFalse(result["success"])
        self.assertIn("room_geometry", result["message"])
        np.testing.assert_allclose(asset.transform.translation(), [0.0, 0.0, 0.0])

    def test_asset_id_not_found_error_handling(self):
        """Test error handling when asset_id doesn't exist in registry."""
        # Mock asset manager to return None (asset not found).
        self.mock_asset_manager.get_asset_by_id.return_value = None
        self.mock_asset_manager.list_available_assets.return_value = []

        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id="nonexistent_asset_id", x=0, y=0, z=0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertIn("not found in registry", result["message"])

    def test_immutable_objects_cannot_be_moved_or_removed(self):
        """Test that immutable objects (walls) reject move/remove operations."""
        # Create immutable object.
        immutable_obj = SceneObject(
            object_id=UniqueID("wall"),
            object_type=ObjectType.WALL,
            name="Test Wall",
            description="An immutable wall",
            transform=RigidTransform(),
            immutable=True,
        )

        self.mock_scene.objects = {immutable_obj.object_id: immutable_obj}
        self.mock_scene.get_object.return_value = immutable_obj

        # Test: Cannot move immutable objects.
        move_result = json.loads(
            self.furniture_tools._move_furniture_impl(
                object_id=str(immutable_obj.object_id),
                x=1,
                y=1,
                z=0,
                roll=0,
                pitch=0,
                yaw=0,
            )
        )
        self.assertFalse(move_result["success"])
        self.assertEqual(
            move_result["error_type"], FurnitureErrorType.IMMUTABLE_OBJECT.value
        )

        # Test: Cannot remove immutable objects.
        remove_result = json.loads(
            self.furniture_tools._remove_furniture_impl(
                object_id=str(immutable_obj.object_id)
            )
        )
        self.assertFalse(remove_result["success"])
        self.assertEqual(
            remove_result["error_type"], FurnitureErrorType.IMMUTABLE_OBJECT.value
        )

    def test_add_furniture_out_of_bounds_x_positive(self):
        """Test placement fails when X coordinate exceeds positive floor boundary."""
        # Mock floor plan with 10m x 8m dimensions.
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        # Mock asset.
        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_chair")
        mock_asset.name = "Chair"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at X=6.0 (exceeds max X=5.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=6.0, y=0.0, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        self.assertIn("out of floor plan bounds", result["message"])
        self.assertIn("X=[-5.000, 5.000]", result["message"])

    def test_add_furniture_out_of_bounds_x_negative(self):
        """Test placement fails when X coordinate exceeds negative floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_table")
        mock_asset.name = "Table"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at X=-5.5 (exceeds min X=-5.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=-5.5, y=0.0, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )

    def test_add_furniture_out_of_bounds_y_positive(self):
        """Test placement fails when Y coordinate exceeds positive floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_sofa")
        mock_asset.name = "Sofa"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at Y=4.5 (exceeds max Y=4.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=0.0, y=4.5, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        self.assertIn("Y=[-4.000, 4.000]", result["message"])

    def test_add_furniture_out_of_bounds_y_negative(self):
        """Test placement fails when Y coordinate exceeds negative floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_lamp")
        mock_asset.name = "Lamp"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at Y=-4.1 (exceeds min Y=-4.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=0.0, y=-4.1, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )

    def test_add_furniture_at_boundary(self):
        """Test placement succeeds when exactly at floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_chair")
        mock_asset.name = "Chair"
        mock_asset.description = "Test chair"
        mock_asset.object_type = ObjectType.FURNITURE
        mock_asset.geometry_path = Path("/test/chair.obj")
        mock_asset.sdf_path = Path("/test/chair.sdf")
        mock_asset.image_path = None
        mock_asset.support_surfaces = []
        mock_asset.metadata = {}
        mock_asset.transform = RigidTransform()

        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset
        self.mock_scene.generate_unique_id.return_value = UniqueID("chair")
        self.mock_scene.add_object = Mock()

        # Placement exactly at boundary (X=5.0, Y=4.0) should succeed.
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=5.0, y=4.0, z=0.0
        )

        result = json.loads(result_str)
        self.assertTrue(result["success"], f"Boundary placement failed: {result}")

    def test_move_furniture_out_of_bounds(self):
        """Test moving furniture to out-of-bounds position fails."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        # Create movable furniture object.
        furniture_obj = SceneObject(
            object_id=UniqueID("chair"),
            object_type=ObjectType.FURNITURE,
            name="Test Chair",
            description="A movable chair",
            transform=RigidTransform(),
            immutable=False,
        )

        self.mock_scene.get_object.return_value = furniture_obj

        # Attempt to move to out-of-bounds position (X=7.0).
        result_str = self.furniture_tools._move_furniture_impl(
            object_id=str(furniture_obj.object_id),
            x=7.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        self.assertIn("out of floor plan bounds", result["message"])

    def test_bounds_check_before_noise_application(self):
        """Test that bounds are checked before placement noise is applied."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_table")
        mock_asset.name = "Table"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement with requested position just outside bounds.
        # If bounds check happens before noise, should fail immediately.
        # If after noise, might succeed depending on noise direction.
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=5.01, y=0.0, z=0.0
        )

        result = json.loads(result_str)
        # Should fail because bounds check happens before noise.
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        # Error message should show the exact requested coordinates, not noisy ones.
        self.assertIn("(5.010,", result["message"])

    def _envelope_object(
        self,
        *,
        bbox_min=(-0.5, -0.5, 0.0),
        bbox_max=(0.5, 0.5, 1.0),
        position=(0.0, 0.0, 0.0),
    ):
        return SceneObject(
            object_id=UniqueID("envelope_object"),
            object_type=ObjectType.FURNITURE,
            name="Envelope object",
            description="Furniture used to test room containment",
            transform=RigidTransform(p=position),
            bbox_min=np.asarray(bbox_min, dtype=float),
            bbox_max=np.asarray(bbox_max, dtype=float),
        )

    def _legacy_envelope_room(self, *, covered=True):
        room = Mock()
        room.length = 6.0
        room.width = 6.0
        room.wall_height = 3.2
        room.wall_thickness = 0.05
        room.has_overhead_cover = covered
        self.mock_scene.room_geometry = room
        self.furniture_tools._structural_surface_index = False

    def test_indoor_furniture_is_rejected_when_it_exceeds_ceiling(self):
        self._legacy_envelope_room(covered=True)
        valid, message = self.furniture_tools._validate_spatial_envelope(
            self._envelope_object(bbox_max=(0.5, 0.5, 4.33))
        )

        self.assertFalse(valid)
        self.assertIn("overhead", message)
        self.assertIn("4.330m", message)
        self.assertIn("3.200m", message)

    def test_open_air_space_has_no_vertical_furniture_limit(self):
        self._legacy_envelope_room(covered=False)
        valid, message = self.furniture_tools._validate_spatial_envelope(
            self._envelope_object(bbox_max=(0.5, 0.5, 4.33))
        )

        self.assertTrue(valid, message)

    def test_full_furniture_footprint_must_fit_inside_room(self):
        self._legacy_envelope_room(covered=False)
        valid, message = self.furniture_tools._validate_spatial_envelope(
            self._envelope_object(position=(2.8, 0.0, 0.0))
        )

        self.assertFalse(valid)
        self.assertIn("footprint", message)
