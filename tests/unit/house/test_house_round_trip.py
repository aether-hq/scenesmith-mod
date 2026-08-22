"""Tests for house.py dataclass serialization."""

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import (
    ConnectionType,
    Door,
    Opening,
    OpeningType,
    PlacedRoom,
    RoomMaterials,
    Wall,
    WallDirection,
    Window,
    WindowShape,
)
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import (
    RoomSpec,
    legacy_openings_to_boundary_portals,
)
from scenesmith.agent_utils.semantics.environment.models.chambers import (
    Bounds3D,
    CavernChamberSpec,
    EnvironmentRegionSpec,
)
from scenesmith.agent_utils.semantics.environment.models.common import (
    DetailSurfaceRole,
    EnvironmentKind,
    FormationType,
)
from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.semantics.environment.models.features import DetailFieldSpec
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    Footprint2D,
    LevelSpec,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    PortalSpec,
    PortalType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)
from scenesmith.utils.geometry.material import Material


class TestRoundTrip(unittest.TestCase):
    """Test round-trip serialization for house dataclasses."""

    def test_opening_round_trip(self) -> None:
        """Opening survives to_dict/from_dict."""
        original = Opening(
            opening_id="opening_1",
            opening_type=OpeningType.WINDOW,
            position_along_wall=0.6,
            width=1.5,
            height=1.2,
            sill_height=0.9,
            shape=WindowShape.ARCHED,
        )
        restored = Opening.from_dict(original.to_dict())
        assert restored.opening_id == original.opening_id
        assert restored.opening_type == original.opening_type
        assert restored.position_along_wall == original.position_along_wall
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.sill_height == original.sill_height
        assert restored.shape == WindowShape.ARCHED

    def test_door_round_trip(self) -> None:
        """Door survives to_dict/from_dict."""
        original = Door(
            id="door_1",
            boundary_label="living_room|kitchen",
            position_segment=0.5,
            position_exact=2.5,
            door_type="interior",
            room_a="living_room",
            room_b="kitchen",
            width=0.9,
            height=2.1,
        )
        restored = Door.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.boundary_label == original.boundary_label
        assert restored.position_segment == original.position_segment
        assert restored.position_exact == original.position_exact
        assert restored.door_type == original.door_type
        assert restored.room_a == original.room_a
        assert restored.room_b == original.room_b
        assert restored.width == original.width
        assert restored.height == original.height

    def test_window_round_trip(self) -> None:
        """Window survives to_dict/from_dict."""
        original = Window(
            id="window_1",
            boundary_label="living_room|exterior",
            position_along_wall=0.6,
            room_id="living_room",
            wall_direction=WallDirection.NORTH,
            width=1.5,
            height=1.2,
            sill_height=0.9,
            shape=WindowShape.ARCHED,
        )
        restored = Window.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.boundary_label == original.boundary_label
        assert restored.position_along_wall == original.position_along_wall
        assert restored.room_id == original.room_id
        assert restored.wall_direction == original.wall_direction
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.sill_height == original.sill_height
        assert restored.shape == WindowShape.ARCHED

    def test_room_geometry_hash_tracks_structural_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sidecar = Path(temporary_directory) / "platform.surfaces.json"
            sidecar.write_text('{"surfaces": []}', encoding="utf-8")
            geometry = RoomGeometry(
                sdf_tree=ET.ElementTree(ET.Element("sdf")),
                sdf_path=Path(temporary_directory) / "room.sdf",
                additional_structural_surface_paths=[sidecar],
            )
            first_hash = geometry.content_hash()

            sidecar.write_text('{"surfaces": [{"id": "changed"}]}', encoding="utf-8")

            self.assertNotEqual(first_hash, geometry.content_hash())

    def test_room_ids_are_path_and_model_safe(self) -> None:
        with self.assertRaisesRegex(GeometryValidationError, "invalid_identifier"):
            RoomSpec("../unsafe")
        with self.assertRaisesRegex(GeometryValidationError, "invalid_identifier"):
            RoomSpec(42)  # type: ignore[arg-type]

    def test_layout_rejects_cross_category_identifier_collision(self) -> None:
        layout = HouseLayout(
            levels=[LevelSpec("ground")],
            room_specs=[RoomSpec("shared")],
            portals=[
                PortalSpec(
                    "shared",
                    PortalType.DOOR,
                    "shared",
                )
            ],
        )

        with self.assertRaisesRegex(GeometryValidationError, "duplicate_scene_id"):
            layout.validate_structure()

    def test_layout_rejects_room_collision_with_derived_detail_instance(self) -> None:
        region = EnvironmentRegionSpec(
            "detail_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 20)),
        )
        chamber = CavernChamberSpec(
            "detail_chamber", "detail_region", (0, 0, 0), (10, 10, 10)
        )
        environment = SemanticEnvironmentSpec(
            regions=(region,),
            chambers=(chamber,),
            detail_fields=(
                DetailFieldSpec(
                    "ceiling_teeth",
                    "detail_region",
                    "detail_chamber",
                    FormationType.STALACTITE,
                    DetailSurfaceRole.OVERHEAD,
                    1,
                    (1, 1, 1),
                    (1, 1, 2),
                    7,
                ),
            ),
        )
        layout = HouseLayout(
            levels=[LevelSpec("ground")],
            room_specs=[RoomSpec("ceiling_teeth_0000")],
            semantic_environment=environment,
        )

        with self.assertRaisesRegex(GeometryValidationError, "duplicate_scene_id"):
            layout.validate_structure()

    def test_legacy_cardinal_opening_maps_to_boundary_portal(self) -> None:
        spec = RoomSpec("hall", length=6, width=4)
        placed = PlacedRoom(
            "hall",
            (0, 0),
            6,
            4,
            walls=[
                Wall(
                    wall_id="hall_north",
                    room_id="hall",
                    direction=WallDirection.NORTH,
                    start_point=(0, 4),
                    end_point=(6, 4),
                    length=6,
                    openings=[
                        Opening(
                            opening_id="window",
                            opening_type=OpeningType.WINDOW,
                            position_along_wall=1,
                            width=1,
                            height=1.2,
                            sill_height=0.8,
                        )
                    ],
                )
            ],
        )

        portals = legacy_openings_to_boundary_portals(spec, placed, 2.5)

        self.assertEqual(len(portals), 1)
        self.assertEqual(portals[0].boundary_edge_index, 2)
        self.assertAlmostEqual(portals[0].position_along or 0, 4.5)
        self.assertEqual(portals[0].sill_height, 0.8)

    def test_room_materials_round_trip(self) -> None:
        """RoomMaterials survives to_dict/from_dict."""
        original = RoomMaterials(
            wall_material=Material.from_path(Path("materials/plaster")),
            floor_material=Material.from_path(Path("materials/wood")),
        )
        restored = RoomMaterials.from_dict(original.to_dict())
        assert restored.wall_material == original.wall_material
        assert restored.floor_material == original.floor_material

    def test_wall_round_trip(self) -> None:
        """Wall survives to_dict/from_dict."""
        original = Wall(
            wall_id="living_room_north",
            room_id="living_room",
            direction=WallDirection.NORTH,
            start_point=(0.0, 6.0),
            end_point=(5.0, 6.0),
            length=5.0,
            is_exterior=True,
            faces_rooms=["kitchen"],
            openings=[
                Opening(
                    opening_id="opening_1",
                    opening_type=OpeningType.WINDOW,
                    position_along_wall=0.6,
                    width=1.5,
                    height=1.2,
                    sill_height=0.9,
                ),
            ],
        )
        restored = Wall.from_dict(original.to_dict())
        assert restored.wall_id == original.wall_id
        assert restored.room_id == original.room_id
        assert restored.direction == original.direction
        assert restored.start_point == original.start_point
        assert restored.end_point == original.end_point
        assert restored.length == original.length
        assert restored.is_exterior == original.is_exterior
        assert restored.faces_rooms == original.faces_rooms
        assert len(restored.openings) == 1
        assert restored.openings[0].opening_id == original.openings[0].opening_id

    def test_placed_room_round_trip(self) -> None:
        """PlacedRoom survives to_dict/from_dict."""
        original = PlacedRoom(
            room_id="living_room",
            position=(1.0, 2.0),
            width=5.0,
            depth=6.0,
            walls=[
                Wall(
                    wall_id="living_room_north",
                    room_id="living_room",
                    direction=WallDirection.NORTH,
                    start_point=(0.0, 6.0),
                    end_point=(5.0, 6.0),
                    length=5.0,
                    is_exterior=True,
                    faces_rooms=[],
                    openings=[],
                ),
            ],
        )
        restored = PlacedRoom.from_dict(original.to_dict())
        assert restored.room_id == original.room_id
        assert restored.position == original.position
        assert restored.width == original.width
        assert restored.depth == original.depth
        assert len(restored.walls) == 1
        assert restored.walls[0].wall_id == original.walls[0].wall_id

    def test_room_spec_round_trip(self) -> None:
        """RoomSpec survives to_dict/from_dict."""
        boundary = Footprint2D.rectangle(6, 5)
        original = RoomSpec(
            room_id="living_room",
            room_type="living_room",
            prompt="A cozy living room",
            position=(1.0, 2.0),
            width=5.0,
            length=6.0,
            connections={
                "kitchen": ConnectionType.DOOR,
                "dining_room": ConnectionType.OPEN,
            },
            footprint=boundary,
            floor_footprint=Footprint2D(
                outer=boundary.outer,
                holes=(((2, 2), (2, 3), (4, 3), (4, 2)),),
            ),
            ceiling_footprint=boundary,
            has_overhead_cover=False,
        )
        restored = RoomSpec.from_dict(original.to_dict())
        assert restored.room_id == original.room_id
        assert restored.room_type == original.room_type
        assert restored.prompt == original.prompt
        assert restored.position == original.position
        assert restored.width == original.width
        assert restored.length == original.length
        assert restored.connections == original.connections
        assert restored.floor_footprint == original.floor_footprint
        assert restored.ceiling_footprint == original.ceiling_footprint
        assert restored.has_overhead_cover is False

    def test_house_layout_round_trip(self) -> None:
        """HouseLayout with nested objects survives to_dict/from_dict."""
        original = HouseLayout(
            wall_height=2.8,
            room_specs=[
                RoomSpec(
                    room_id="living_room",
                    room_type="living_room",
                    prompt="A cozy living room",
                    position=(0.0, 0.0),
                    width=5.0,
                    length=6.0,
                    connections={"kitchen": ConnectionType.DOOR},
                ),
            ],
            doors=[
                Door(
                    id="door_1",
                    boundary_label="living_room|kitchen",
                    position_segment=0.5,
                    position_exact=2.5,
                    door_type="interior",
                    room_a="living_room",
                    room_b="kitchen",
                    width=0.9,
                    height=2.1,
                ),
            ],
            windows=[
                Window(
                    id="window_1",
                    boundary_label="living_room|exterior",
                    position_along_wall=0.6,
                    room_id="living_room",
                    wall_direction=WallDirection.NORTH,
                    width=1.5,
                    height=1.2,
                    sill_height=0.9,
                ),
            ],
            room_materials={
                "living_room": RoomMaterials(
                    wall_material=Material.from_path(Path("materials/plaster")),
                    floor_material=Material.from_path(Path("materials/wood")),
                ),
            },
            exterior_material=Material.from_path(Path("materials/brick")),
            placed_rooms=[
                PlacedRoom(
                    room_id="living_room",
                    position=(0.0, 0.0),
                    width=5.0,
                    depth=6.0,
                    walls=[
                        Wall(
                            wall_id="living_room_north",
                            room_id="living_room",
                            direction=WallDirection.NORTH,
                            start_point=(0.0, 6.0),
                            end_point=(5.0, 6.0),
                            length=5.0,
                            is_exterior=True,
                            faces_rooms=[],
                            openings=[],
                        ),
                    ],
                ),
            ],
            placement_valid=True,
            connectivity_valid=True,
            boundary_labels={
                "living_room|kitchen": ("living_room", "kitchen"),
            },
        )

        restored = HouseLayout.from_dict(original.to_dict())

        assert restored.wall_height == original.wall_height
        assert restored.placement_valid == original.placement_valid
        assert restored.connectivity_valid == original.connectivity_valid
        assert restored.exterior_material == original.exterior_material
        assert len(restored.room_specs) == 1
        assert restored.room_specs[0].room_id == original.room_specs[0].room_id
        assert len(restored.doors) == 1
        assert restored.doors[0].id == original.doors[0].id
        assert len(restored.windows) == 1
        assert restored.windows[0].id == original.windows[0].id
        assert "living_room" in restored.room_materials
        assert len(restored.placed_rooms) == 1
        assert restored.placed_rooms[0].room_id == original.placed_rooms[0].room_id
        assert restored.boundary_labels == original.boundary_labels
