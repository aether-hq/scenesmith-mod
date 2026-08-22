import asyncio
import json
import math
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType
from scenesmith.furniture_agents.tools.response_dataclasses import (
    Position3D,
    Rotation3D,
    SceneObjectInfo,
    SceneStateResult,
)
from scenesmith.furniture_agents.tools.scene_tools import SceneTools


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


class TestSceneTools(BaseAgentToolsTest):
    """Test SceneTools class contracts."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.mock_scene = self.create_mock_scene()

        # Load base configuration from actual config file.
        config_path = (
            Path(__file__).parents[3]
            / "configurations/furniture_agent/base_furniture_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)

        self.scene_tools = SceneTools(scene=self.mock_scene, cfg=self.cfg)

    def test_scene_tools_initialization(self):
        """Test SceneTools initializes properly."""
        self.assertIsNotNone(self.scene_tools)
        self.assertEqual(self.scene_tools.scene, self.mock_scene)

    def test_get_current_scene_state_returns_json(self):
        """Unit test: SceneTools extracts object data and returns JSON."""
        # Create mock scene objects using helper function.
        table_mock = self.create_mock_scene_object(
            name="Table",
            object_type=ObjectType.FURNITURE,
            position=[1.0, 2.0, 3.0],
            rotation=[0.0, 0.0, math.radians(90.0)],
            sdf_path="/path/to/table.sdf",
            geometry_path="/path/to/table.obj",
        )
        table_mock.description = "A wooden table"  # Override generated description

        chair_mock = self.create_mock_scene_object(
            name="Chair",
            object_type=ObjectType.FURNITURE,
            position=[4.0, 5.0, 6.0],
            rotation=[0.0, 0.0, 0.0],
        )
        chair_mock.description = "A comfortable chair"  # Override generated description

        objects = {
            "table_1": table_mock,
            "chair_1": chair_mock,
        }
        self.mock_scene.objects = objects

        # Test the tool function directly.
        result = self.scene_tools._get_current_scene_impl()

        # Verify JSON-like result is returned.
        self.assertIsInstance(result, str)
        # The result should contain furniture information.
        self.assertIn("furniture", result.lower())

    def test_scene_state_result_serialization(self):
        """SceneStateResult DTOs serialize to valid JSON."""
        objects = [
            SceneObjectInfo(
                object_id="table_1",
                description="A wooden dining table",
                position=Position3D(x=1.0, y=2.0, z=3.0),
                rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=90.0),
                object_type="FURNITURE",
                dimensions=None,
                world_bounds=None,
                immutable=False,
            ),
            SceneObjectInfo(
                object_id="chair_1",
                description="A wooden dining chair",
                position=Position3D(x=4.0, y=5.0, z=6.0),
                rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=0.0),
                object_type="FURNITURE",
                dimensions=None,
                world_bounds=None,
                immutable=False,
            ),
        ]

        result = SceneStateResult(success=True, furniture_count=2, objects=objects)

        # Test JSON serialization.
        json_str = result.to_json()
        self.assertIsInstance(json_str, str)

        # Verify JSON content structure and values.
        parsed = json.loads(json_str)
        self.assertEqual(parsed["success"], True)
        self.assertEqual(parsed["furniture_count"], 2)
        self.assertEqual(len(parsed["objects"]), 2)

        # Check first object details.
        table_obj = parsed["objects"][0]
        self.assertEqual(table_obj["object_id"], "table_1")
        self.assertEqual(table_obj["object_type"], "FURNITURE")
        self.assertEqual(table_obj["position"]["x"], 1.0)
        self.assertEqual(table_obj["rotation"]["yaw"], 90.0)
        self.assertEqual(table_obj["immutable"], False)

        # Check second object details.
        chair_obj = parsed["objects"][1]
        self.assertEqual(chair_obj["object_id"], "chair_1")
        self.assertEqual(chair_obj["object_type"], "FURNITURE")
        self.assertIsNone(chair_obj["dimensions"])

        # Verify the JSON contains expected keywords.
        self.assertIn("furniture", json_str.lower())

    def test_tools_dictionary_exposed(self):
        """Test that tools dictionary is properly exposed."""
        # Verify tools dictionary exists and contains expected tools.
        self.assertTrue(hasattr(self.scene_tools, "tools"))
        self.assertIsInstance(self.scene_tools.tools, dict)

        # Should have at least the get_current_scene_state.
        self.assertIn("get_current_scene_state", self.scene_tools.tools)

    def test_tools_are_callable(self):
        """Test that SceneTools follow standard tool interface."""
        for tool_name, tool_func in self.scene_tools.tools.items():
            # FunctionTool objects are not directly callable, but they have an invoke
            # method.
            self.assertTrue(
                hasattr(tool_func, "on_invoke_tool"),
                f"SceneTools tool {tool_name} should have on_invoke_tool method",
            )

    def test_tool_invocation(self):
        """Test that SceneTools can be invoked and return appropriate types."""
        get_scene_tool = self.scene_tools.tools["get_current_scene_state"]

        # Provide minimal context and input for FunctionTool.
        mock_ctx = Mock()
        result = asyncio.run(get_scene_tool.on_invoke_tool(mock_ctx, {}))

        self.assertIsInstance(result, str)
        json.loads(result)  # Will raise if invalid JSON.

    def test_error_handling(self):
        """Test that SceneTools handle errors gracefully."""
        get_scene_tool = self.scene_tools.tools["get_current_scene_state"]

        try:
            mock_ctx = Mock()
            result = asyncio.run(get_scene_tool.on_invoke_tool(mock_ctx, {}))
            # Should return some form of response.
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Tool raised unhandled exception: {e}")
