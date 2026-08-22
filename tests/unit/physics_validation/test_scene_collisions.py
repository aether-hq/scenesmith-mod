import tempfile
import unittest

from pathlib import Path

import lxml.etree as ET
import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.physics.validation.models import CollisionPair
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.agent_utils.structure.compiler.polygon_spaces import (
    compile_polygon_space,
)
from scenesmith.agent_utils.structure.compiler.surfaces import compile_platform
from scenesmith.agent_utils.structure.compiler.writing import write_compiled_structure
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    Footprint2D,
    PlatformSpec,
)


class TestCollisionPair(unittest.TestCase):
    """Test CollisionPair dataclass."""

    def test_to_description_with_meaningful_penetration(self):
        """Test description formatting for meaningful penetration depth."""
        collision = CollisionPair(
            object_a_name="dining chair",
            object_a_id="dining_chair_a3f2e8b1",
            object_b_name="dining table",
            object_b_id="dining_table_5c9d7e2f",
            penetration_depth=0.05,  # 5cm penetration
        )

        expected = (
            "dining_chair_a3f2e8b1 collides with "
            "dining_table_5c9d7e2f (5.0cm penetration)"
        )
        self.assertEqual(collision.to_description(), expected)

    def test_to_description_with_minimal_penetration(self):
        """Test description formatting for sub-millimeter penetration."""
        collision = CollisionPair(
            object_a_name="chair",
            object_a_id="chair_12345678",
            object_b_name="table",
            object_b_id="table_87654321",
            penetration_depth=0.0001,  # 0.01cm penetration
        )

        expected = "chair_12345678 collides with table_87654321 (touching)"
        self.assertEqual(collision.to_description(), expected)


class TestComputeSceneCollisions(unittest.TestCase):
    """Test compute_scene_collisions function with real physics."""

    def setUp(self):
        """Set up test fixtures with real scene data."""
        # Create base scene with floor plan.
        test_data_dir = Path(__file__).parents[2] / "test_data"
        self.floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        self.box_sdf_path = test_data_dir / "simple_box.sdf"
        self.sphere_sdf_path = test_data_dir / "simple_sphere.sdf"

        # Create room geometry.
        test_data_dir = Path(__file__).parents[2] / "test_data"
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

    def test_no_collisions_with_separated_objects(self):
        """Test that separated objects don't report collisions."""
        # Add two boxes clearly separated.
        box1 = SceneObject(
            object_id=UniqueID("box1"),
            object_type=ObjectType.FURNITURE,
            name="Box 1",
            description="Test box 1",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),  # At origin
            sdf_path=self.box_sdf_path,
        )
        box2 = SceneObject(
            object_id=UniqueID("box2"),
            object_type=ObjectType.FURNITURE,
            name="Box 2",
            description="Test box 2",
            transform=RigidTransform(np.array([3.0, 0.0, 0.5])),  # 3m away
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box1)
        self.scene.add_object(box2)

        # Test actual physics.
        collisions = compute_scene_collisions(self.scene)

        # Filter out floor plan internal collisions (walls, etc.) and focus on furniture.
        furniture_collisions = [
            c
            for c in collisions
            if not (
                c.object_a_name.startswith("floor plan")
                and c.object_b_name.startswith("floor plan")
            )
        ]
        self.assertEqual(
            len(furniture_collisions),
            0,
            "Should not detect collisions between separated furniture objects",
        )

    def test_furniture_inside_compiled_polygon_room_is_not_in_shell_collision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(
                compile_polygon_space(
                    structure_id="compiled_room",
                    footprint=Footprint2D.rectangle(4, 3),
                    wall_height=3.0,
                ),
                temporary_directory,
                model_name="room_geometry",
                link_name="room_geometry_body_link",
            )
            room_geometry = RoomGeometry(
                sdf_tree=ET.parse(paths.sdf_path),
                sdf_path=paths.sdf_path,
                width=3.0,
                length=4.0,
                wall_height=3.0,
            )
            scene = RoomScene(
                room_geometry=room_geometry,
                scene_dir=Path(temporary_directory),
            )
            scene.add_object(
                SceneObject(
                    object_id=UniqueID("center_box"),
                    object_type=ObjectType.FURNITURE,
                    name="Center box",
                    description="Box standing on the compiled room floor",
                    transform=RigidTransform(np.array([2.0, 1.5, 0.5])),
                    sdf_path=self.box_sdf_path,
                )
            )

            collisions = compute_scene_collisions(scene)

        self.assertFalse(
            any("center_box" in {c.object_a_id, c.object_b_id} for c in collisions)
        )

    def test_platform_support_contact_uses_floor_penetration_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(
                compile_platform(
                    PlatformSpec(
                        platform_id="upper_gallery",
                        space_id="library",
                        footprint=Footprint2D.rectangle(4, 4),
                        elevation=2.0,
                    )
                ),
                temporary_directory,
            )
            room_geometry = RoomGeometry(
                sdf_tree=ET.parse(self.floor_plan_path),
                sdf_path=self.floor_plan_path,
                additional_structural_surface_paths=[paths.surfaces_path],
            )
            scene = RoomScene(
                room_geometry=room_geometry,
                scene_dir=Path(temporary_directory),
            )
            scene.add_object(
                SceneObject(
                    object_id=UniqueID("exact_platform_box"),
                    object_type=ObjectType.FURNITURE,
                    name="Platform box",
                    description="Box standing exactly on the upper gallery",
                    transform=RigidTransform(np.array([2.0, 2.0, 2.25])),
                    sdf_path=self.box_sdf_path,
                )
            )
            scene.add_object(
                SceneObject(
                    object_id=UniqueID("settled_platform_box"),
                    object_type=ObjectType.FURNITURE,
                    name="Settled platform box",
                    description="Box settled 3cm into the upper gallery support",
                    transform=RigidTransform(np.array([1.0, 1.0, 2.22])),
                    sdf_path=self.box_sdf_path,
                )
            )

            collisions = compute_scene_collisions(
                scene, floor_penetration_tolerance=0.05
            )

        self.assertFalse(
            any(
                {"exact_platform_box", "settled_platform_box"}
                & {c.object_a_id, c.object_b_id}
                for c in collisions
            )
        )

    def test_furniture_to_furniture_multiple_collisions(self):
        """Test multiple furniture collisions with deduplication.

        Creates 3 overlapping boxes to test:
        1. Multiple collisions are detected
        2. Deduplication works (no A-B and B-A duplicates)
        """
        # Create 3 boxes in a row with overlap.
        for i in range(3):
            box = SceneObject(
                object_id=UniqueID(f"box{i}"),
                object_type=ObjectType.FURNITURE,
                name=f"Box {i}",
                description=f"Test box {i}",
                transform=RigidTransform(
                    np.array([i * 0.3, 0.0, 0.5])
                ),  # 0.2m overlap each
                sdf_path=self.box_sdf_path,
            )
            self.scene.add_object(box)

        collisions = compute_scene_collisions(self.scene)

        # Filter for furniture-to-furniture collisions.
        furniture_collisions = [
            c
            for c in collisions
            if "Box" in c.object_a_name and "Box" in c.object_b_name
        ]

        # Should detect exactly 2 collision pairs: (0,1) and (1,2).
        # Box 0 and Box 2 don't overlap.
        self.assertEqual(
            len(furniture_collisions),
            2,
            f"Expected 2 furniture collision pairs, got {len(furniture_collisions)}",
        )

        # Verify no duplicates (each pair should appear only once).
        collision_pairs = set()
        for c in furniture_collisions:
            pair = tuple(sorted([c.object_a_name, c.object_b_name]))
            self.assertNotIn(pair, collision_pairs, "Duplicate collision pair detected")
            collision_pairs.add(pair)

    def test_exact_touching_objects_no_collision(self):
        """Test that exactly touching objects don't report collision.

        Objects with faces exactly touching (0 penetration) should not
        be reported as colliding.
        """
        # Place two 0.5m boxes exactly 0.5m apart (touching faces).
        box1 = SceneObject(
            object_id=UniqueID("box1"),
            object_type=ObjectType.FURNITURE,
            name="Box 1",
            description="Test box 1",
            transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
            sdf_path=self.box_sdf_path,
        )
        box2 = SceneObject(
            object_id=UniqueID("box2"),
            object_type=ObjectType.FURNITURE,
            name="Box 2",
            description="Test box 2",
            transform=RigidTransform(np.array([0.5, 0.0, 0.5])),  # Exactly touching
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box1)
        self.scene.add_object(box2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for box-to-box collisions.
        box_collisions = [
            c
            for c in collisions
            if "Box" in c.object_a_name and "Box" in c.object_b_name
        ]

        # Should not detect collision for exactly touching objects.
        self.assertEqual(
            len(box_collisions),
            0,
            "Should not report collision for exactly touching objects",
        )

    def test_floor_penetration_tolerance_ignored(self):
        """Test that floor collision tolerance works correctly.

        Objects slightly penetrating the floor (< 5cm) should not be reported.
        Objects deeply penetrating the floor (> 5cm) should be reported.
        """
        # Test object slightly penetrating floor (2cm).
        box_slight = SceneObject(
            object_id=UniqueID("box_slight"),
            object_type=ObjectType.FURNITURE,
            name="Box Slight",
            description="Slightly penetrating box",
            transform=RigidTransform(np.array([0.0, 0.0, 0.23])),  # 2cm into floor
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box_slight)

        collisions = compute_scene_collisions(
            self.scene, floor_penetration_tolerance=0.05  # 5cm tolerance
        )

        # Should not report slight floor penetration.
        floor_collisions_slight = [
            c
            for c in collisions
            if (
                "floor" in c.object_a_name.lower() or "floor" in c.object_b_name.lower()
            )
            and "Slight" in (c.object_a_name + c.object_b_name)
        ]
        self.assertEqual(
            len(floor_collisions_slight),
            0,
            "Should not report floor collision with penetration < tolerance",
        )

        # Now test deep penetration.
        box_deep = SceneObject(
            object_id=UniqueID("box_deep"),
            object_type=ObjectType.FURNITURE,
            name="Box Deep",
            description="Deeply penetrating box",
            transform=RigidTransform(np.array([2.0, 0.0, 0.15])),  # 10cm into floor
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(box_deep)

        collisions = compute_scene_collisions(
            self.scene, floor_penetration_tolerance=0.05  # 5cm tolerance
        )

        # Should report deep floor penetration.
        floor_collisions_deep = [
            c
            for c in collisions
            if (
                "floor" in c.object_a_name.lower() or "floor" in c.object_b_name.lower()
            )
            and "Deep" in (c.object_a_name + c.object_b_name)
        ]
        self.assertGreater(
            len(floor_collisions_deep),
            0,
            "Should report floor collision with penetration > tolerance",
        )

    def test_ceiling_to_ceiling_collision_detected(self):
        """Test collision detection works for CEILING_MOUNTED objects.

        This is a regression test for a bug where ceiling objects were always
        welded even during collision checking, which caused Drake's broadphase
        to miss ceiling-to-ceiling collisions.
        """
        # Add two overlapping ceiling-mounted objects (e.g., track lights).
        ceiling1 = SceneObject(
            object_id=UniqueID("track_light_1"),
            object_type=ObjectType.CEILING_MOUNTED,
            name="Track Light 1",
            description="First ceiling track light",
            transform=RigidTransform(np.array([0.0, 0.0, 3.0])),  # At ceiling height
            sdf_path=self.box_sdf_path,
        )
        ceiling2 = SceneObject(
            object_id=UniqueID("track_light_2"),
            object_type=ObjectType.CEILING_MOUNTED,
            name="Track Light 2",
            description="Second ceiling track light",
            transform=RigidTransform(np.array([0.3, 0.0, 3.0])),  # 0.2m overlap
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(ceiling1)
        self.scene.add_object(ceiling2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for ceiling-to-ceiling collisions.
        ceiling_collisions = [
            c
            for c in collisions
            if "Track Light" in c.object_a_name and "Track Light" in c.object_b_name
        ]

        # Should detect the collision between the two ceiling objects.
        self.assertGreater(
            len(ceiling_collisions),
            0,
            "Should detect collision between overlapping CEILING_MOUNTED objects",
        )

        # Verify penetration depth is approximately 0.2m.
        if ceiling_collisions:
            penetration = ceiling_collisions[0].penetration_depth
            self.assertAlmostEqual(penetration, 0.2, places=1)

    def test_wall_mounted_to_wall_mounted_collision_detected(self):
        """Test collision detection works for WALL_MOUNTED objects.

        This is a regression test for a bug where wall-mounted objects were
        always welded even during collision checking.
        """
        # Add two overlapping wall-mounted objects.
        wall1 = SceneObject(
            object_id=UniqueID("painting_1"),
            object_type=ObjectType.WALL_MOUNTED,
            name="Painting 1",
            description="First wall painting",
            transform=RigidTransform(np.array([0.0, 2.0, 1.5])),  # On wall
            sdf_path=self.box_sdf_path,
        )
        wall2 = SceneObject(
            object_id=UniqueID("painting_2"),
            object_type=ObjectType.WALL_MOUNTED,
            name="Painting 2",
            description="Second wall painting",
            transform=RigidTransform(np.array([0.3, 2.0, 1.5])),  # 0.2m overlap
            sdf_path=self.box_sdf_path,
        )
        self.scene.add_object(wall1)
        self.scene.add_object(wall2)

        collisions = compute_scene_collisions(self.scene)

        # Filter for wall-to-wall object collisions.
        wall_object_collisions = [
            c
            for c in collisions
            if "Painting" in c.object_a_name and "Painting" in c.object_b_name
        ]

        # Should detect the collision between the two wall-mounted objects.
        self.assertGreater(
            len(wall_object_collisions),
            0,
            "Should detect collision between overlapping WALL_MOUNTED objects",
        )

    def test_collision_detection_by_object_type(self):
        """Test collision detection works for both FURNITURE and MANIPULAND object types."""
        # Test both object types to ensure collision detection works regardless of
        # welding behavior.
        for object_type in [ObjectType.FURNITURE, ObjectType.MANIPULAND]:
            with self.subTest(object_type=object_type):
                # Clear scene between subtests
                test_data_dir = Path(__file__).parents[2] / "test_data"
                self.scene = RoomScene(
                    room_geometry=self.scene.room_geometry, scene_dir=test_data_dir
                )

                # Add two overlapping objects of the specified type
                box1 = SceneObject(
                    object_id=UniqueID("box1"),
                    object_type=object_type,
                    name=f"{object_type.value.title()} 1",
                    description=f"Test {object_type.value} 1",
                    transform=RigidTransform(np.array([0.0, 0.0, 0.5])),
                    sdf_path=self.box_sdf_path,
                )
                box2 = SceneObject(
                    object_id=UniqueID("box2"),
                    object_type=object_type,
                    name=f"{object_type.value.title()} 2",
                    description=f"Test {object_type.value} 2",
                    transform=RigidTransform(np.array([0.3, 0.0, 0.5])),  # 0.2m overlap
                    sdf_path=self.box_sdf_path,
                )
                self.scene.add_object(box1)
                self.scene.add_object(box2)

                collisions = compute_scene_collisions(self.scene)

                # Should detect collision between the two objects.
                object_collisions = [
                    c
                    for c in collisions
                    if object_type.value.title() in c.object_a_name
                    and object_type.value.title() in c.object_b_name
                ]
                self.assertGreater(
                    len(object_collisions),
                    0,
                    f"Should detect collision between overlapping {object_type.value} "
                    "objects",
                )

                # Verify penetration depth is approximately 0.2m.
                if object_collisions:
                    penetration = object_collisions[0].penetration_depth
                    self.assertAlmostEqual(penetration, 0.2, places=1)
