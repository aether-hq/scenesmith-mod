import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)


class TestSceneUniqueIDGeneration(unittest.TestCase):
    """Test cases for RoomScene.generate_unique_id() sequential numbering."""

    def setUp(self):
        """Set up test scene."""
        self.test_data_dir = Path(__file__).parents[2] / "test_data"
        # Create minimal RoomGeometry for testing ID generation.
        room_geometry = RoomGeometry(
            sdf_tree=ET.Element("sdf"),
            sdf_path=None,
        )
        self.scene = RoomScene(
            room_geometry=room_geometry, scene_dir=self.test_data_dir
        )

    def test_first_object_gets_zero_suffix(self):
        """Test that first object of a type gets _0 suffix."""
        object_id = self.scene.generate_unique_id("chair")
        self.assertEqual(str(object_id), "chair_0")

    def test_sequential_numbering(self):
        """Test that subsequent objects get sequential suffixes."""
        # Add first chair (suffix _0).
        chair1_id = self.scene.generate_unique_id("chair")
        chair1 = SceneObject(
            object_id=chair1_id,
            object_type=ObjectType.FURNITURE,
            name="chair",
            description="First chair",
            transform=RigidTransform(),
            sdf_path=None,
        )
        self.scene.add_object(chair1)

        # Add second chair (suffix _1).
        chair2_id = self.scene.generate_unique_id("chair")
        self.assertEqual(str(chair2_id), "chair_1")

    def test_base36_encoding(self):
        """Test that base-36 encoding works (0-9, a-z)."""
        # Add 11 chairs (chair_0 through chair_a).
        for i in range(11):
            chair_id = self.scene.generate_unique_id("chair")
            chair = SceneObject(
                object_id=chair_id,
                object_type=ObjectType.FURNITURE,
                name="chair",
                description=f"Chair {i+1}",
                transform=RigidTransform(),
                sdf_path=None,
            )
            self.scene.add_object(chair)

        # 12th chair should use 'b' (base-36 for 11).
        chair12_id = self.scene.generate_unique_id("chair")
        self.assertEqual(str(chair12_id), "chair_b")

    def test_different_types_independent(self):
        """Test that different object types have independent numbering."""
        chair_id = self.scene.generate_unique_id("chair")
        table_id = self.scene.generate_unique_id("table")

        # Both should start with suffix _0 (first of their type).
        self.assertEqual(str(chair_id), "chair_0")
        self.assertEqual(str(table_id), "table_0")


class TestRoomGeometry(unittest.TestCase):
    """Test cases for RoomGeometry serialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parents[2] / "test_data"
        self.sdf_path = self.test_data_dir / "simple_room_geometry.sdf"

    def test_to_dict_minimal(self):
        """Test RoomGeometry serialization with minimal fields."""
        sdf_tree = ET.parse(self.sdf_path)
        room_geometry = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=self.sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        room_geometry_dict = room_geometry.to_dict()

        # Check basic fields.
        self.assertEqual(room_geometry_dict["width"], 5.0)
        self.assertEqual(room_geometry_dict["length"], 6.0)
        self.assertIsNone(room_geometry_dict["floor"])
        self.assertEqual(room_geometry_dict["sdf_path"], str(self.sdf_path))

    def test_to_dict_with_scene_dir(self):
        """Test RoomGeometry serialization with path relativization."""
        sdf_tree = ET.parse(self.sdf_path)

        # Create paths within a scene directory.
        scene_dir = Path("/tmp/scene")
        sdf_path = scene_dir / "room_geometry.sdf"

        room_geometry = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        room_geometry_dict = room_geometry.to_dict(scene_dir=scene_dir)

        # Paths should be relative.
        self.assertEqual(room_geometry_dict["sdf_path"], "room_geometry.sdf")

    def test_to_dict_with_floor(self):
        """Test RoomGeometry serialization with floor object."""
        sdf_tree = ET.parse(self.sdf_path)

        floor_obj = SceneObject(
            object_id=UniqueID.generate(),
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Floor object",
            transform=RigidTransform(),
        )

        room_geometry = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=self.sdf_path,
            walls=[],
            floor=floor_obj,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        room_geometry_dict = room_geometry.to_dict()

        # Floor should be serialized.
        self.assertIsNotNone(room_geometry_dict["floor"])
        self.assertEqual(room_geometry_dict["floor"]["name"], "Floor")
        self.assertEqual(
            room_geometry_dict["floor"]["object_type"], ObjectType.FLOOR.value
        )

    def test_from_dict_minimal(self):
        """Test RoomGeometry deserialization with minimal fields."""
        room_geometry_dict = {
            "sdf_path": str(self.sdf_path),
            "width": 5.0,
            "length": 6.0,
            "floor": None,
        }

        room_geometry = RoomGeometry.from_dict(room_geometry_dict)

        # Check fields.
        self.assertEqual(room_geometry.width, 5.0)
        self.assertEqual(room_geometry.length, 6.0)
        self.assertEqual(room_geometry.sdf_path, self.sdf_path)
        self.assertIsNone(room_geometry.floor)
        self.assertEqual(room_geometry.walls, [])
        self.assertEqual(room_geometry.wall_normals, {})

        # sdf_tree should be re-parsed from file.
        self.assertIsNotNone(room_geometry.sdf_tree)

    def test_from_dict_with_scene_dir(self):
        """Test RoomGeometry deserialization with path resolution."""
        # Copy test file to a temp location for this test.

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test file.
            shutil.copy(self.sdf_path, test_sdf)

            room_geometry_dict = {
                "sdf_path": "room_geometry.sdf",
                "width": 5.0,
                "length": 6.0,
                "floor": None,
            }

            room_geometry = RoomGeometry.from_dict(
                room_geometry_dict, scene_dir=scene_dir
            )

            # Paths should be resolved relative to scene_dir.
            self.assertEqual(room_geometry.sdf_path, test_sdf)
            self.assertIsNotNone(room_geometry.sdf_tree)

    def test_serialization_roundtrip_minimal(self):
        """Test RoomGeometry serialization roundtrip with minimal fields."""
        sdf_tree = ET.parse(self.sdf_path)
        original = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=self.sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        # Serialize and deserialize.
        room_geometry_dict = original.to_dict()
        restored = RoomGeometry.from_dict(room_geometry_dict)

        # Check equality.
        self.assertEqual(restored.width, original.width)
        self.assertEqual(restored.length, original.length)
        self.assertEqual(restored.sdf_path, original.sdf_path)
        self.assertIsNone(restored.floor)
        self.assertIsNotNone(restored.sdf_tree)

    def test_serialization_roundtrip_with_floor(self):
        """Test RoomGeometry serialization roundtrip with floor object."""

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test files.
            shutil.copy(self.sdf_path, test_sdf)

            sdf_tree = ET.parse(test_sdf)
            floor_obj = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.FLOOR,
                name="Floor",
                description="Floor object",
                transform=RigidTransform(p=np.array([1.0, 2.0, 0.0])),
            )

            original = RoomGeometry(
                sdf_tree=sdf_tree,
                sdf_path=test_sdf,
                walls=[],
                floor=floor_obj,
                wall_normals={},
                width=5.0,
                length=6.0,
            )

            # Serialize and deserialize with scene_dir.
            room_geometry_dict = original.to_dict(scene_dir=scene_dir)
            restored = RoomGeometry.from_dict(room_geometry_dict, scene_dir=scene_dir)

            # Check all fields.
            self.assertEqual(restored.width, original.width)
            self.assertEqual(restored.length, original.length)
            self.assertEqual(restored.sdf_path, original.sdf_path)
            self.assertIsNotNone(restored.floor)
            self.assertEqual(restored.floor.name, original.floor.name)
            self.assertEqual(restored.floor.object_type, original.floor.object_type)
            np.testing.assert_array_almost_equal(
                restored.floor.transform.translation(),
                original.floor.transform.translation(),
            )

    def test_wall_normals_serialization_roundtrip(self):
        """Test wall_normals survives serialization roundtrip."""

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test file.
            shutil.copy(self.sdf_path, test_sdf)

            sdf_tree = ET.parse(test_sdf)

            # Create floor plan with wall_normals.
            wall_normals = {
                "left_wall": np.array([1.0, 0.0]),
                "right_wall": np.array([-1.0, 0.0]),
                "back_wall": np.array([0.0, 1.0]),
                "front_wall": np.array([0.0, -1.0]),
            }

            original = RoomGeometry(
                sdf_tree=sdf_tree,
                sdf_path=test_sdf,
                walls=[],
                floor=None,
                wall_normals=wall_normals,
                width=5.0,
                length=6.0,
            )

            # Serialize and deserialize.
            room_geometry_dict = original.to_dict(scene_dir=scene_dir)
            restored = RoomGeometry.from_dict(room_geometry_dict, scene_dir=scene_dir)

            # Verify wall_normals were preserved.
            self.assertEqual(len(restored.wall_normals), 4)
            for wall_name, expected_normal in wall_normals.items():
                self.assertIn(wall_name, restored.wall_normals)
                np.testing.assert_array_almost_equal(
                    restored.wall_normals[wall_name], expected_normal
                )

    def test_from_dict_missing_sdf_raises_error(self):
        """Test RoomGeometry.from_dict() raises ValueError on missing SDF file."""
        floor_plan_dict = {
            "sdf_path": "nonexistent.sdf",
            "width": 5.0,
            "length": 6.0,
            "floor": None,
            "wall_normals": {},
        }

        # Should raise ValueError, not just warn.
        with self.assertRaises(ValueError) as context:
            RoomGeometry.from_dict(floor_plan_dict)

        self.assertIn("SDF file not found", str(context.exception))


class TestSceneRoomGeometryIntegration(unittest.TestCase):
    """Integration tests for RoomScene serialization with floor plan."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parents[2] / "test_data"
        self.sdf_path = self.test_data_dir / "simple_room_geometry.sdf"

    def test_scene_serialization_with_floor_plan_and_walls(self):
        """
        Integration test: RoomScene serialization includes floor plan, and walls
        are correctly populated from scene.objects after restoration.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test files.
            shutil.copy(self.sdf_path, test_sdf)

            # Create floor plan with floor object and wall_normals.
            sdf_tree = ET.parse(test_sdf)
            floor_obj = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.FLOOR,
                name="Floor",
                description="Test floor",
                transform=RigidTransform(),
            )

            wall_normals = {
                "left_wall": np.array([1.0, 0.0]),
                "right_wall": np.array([-1.0, 0.0]),
            }

            room_geometry = RoomGeometry(
                sdf_tree=sdf_tree,
                sdf_path=test_sdf,
                walls=[],  # Will be populated after Scene restoration
                floor=floor_obj,
                wall_normals=wall_normals,
                width=5.0,
                length=6.0,
            )

            # Create scene with floor plan and wall objects.
            scene = RoomScene(
                room_geometry=room_geometry,
                scene_dir=scene_dir,
                text_description="Test scene with floor plan",
            )

            # Add wall objects to scene.
            wall1 = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.WALL,
                name="Wall1",
                description="Test wall",
                transform=RigidTransform(),
            )
            wall2 = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.WALL,
                name="Wall2",
                description="Another test wall",
                transform=RigidTransform(),
            )
            scene.add_object(wall1)
            scene.add_object(wall2)

            # Manually populate room_geometry.walls (normally done by furniture agent).
            room_geometry.walls = [wall1, wall2]

            # Serialize scene.
            state_dict = scene.to_state_dict()

            # Verify room_geometry is in state_dict.
            self.assertIn("room_geometry", state_dict)
            self.assertIsNotNone(state_dict["room_geometry"])
            self.assertEqual(state_dict["room_geometry"]["width"], 5.0)
            self.assertEqual(state_dict["room_geometry"]["length"], 6.0)
            self.assertIsNotNone(state_dict["room_geometry"]["floor"])
            self.assertEqual(len(state_dict["room_geometry"]["wall_normals"]), 2)

            # Create new scene and restore.
            restored_scene = RoomScene(
                room_geometry=None,
                scene_dir=scene_dir,
                text_description="",
            )
            restored_scene.restore_from_state_dict(state_dict)

            # Verify floor plan was restored.
            self.assertIsNotNone(restored_scene.room_geometry)
            self.assertEqual(restored_scene.room_geometry.width, 5.0)
            self.assertEqual(restored_scene.room_geometry.length, 6.0)

            # Verify floor object was restored.
            self.assertIsNotNone(restored_scene.room_geometry.floor)
            self.assertEqual(restored_scene.room_geometry.floor.name, "Floor")

            # Verify wall_normals were restored.
            self.assertEqual(len(restored_scene.room_geometry.wall_normals), 2)
            np.testing.assert_array_almost_equal(
                restored_scene.room_geometry.wall_normals["left_wall"],
                np.array([1.0, 0.0]),
            )
            np.testing.assert_array_almost_equal(
                restored_scene.room_geometry.wall_normals["right_wall"],
                np.array([-1.0, 0.0]),
            )

            # CRITICAL: Verify walls were populated from scene.objects.
            self.assertEqual(len(restored_scene.room_geometry.walls), 2)
            wall_names = {w.name for w in restored_scene.room_geometry.walls}
            self.assertEqual(wall_names, {"Wall1", "Wall2"})

            # Verify walls in room_geometry.walls are the same objects as in scene.objects.
            for wall in restored_scene.room_geometry.walls:
                self.assertIn(wall.object_id, restored_scene.objects)
                self.assertIs(wall, restored_scene.objects[wall.object_id])


if __name__ == "__main__":
    unittest.main()
