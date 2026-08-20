"""Tests for house.py dataclass serialization."""

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

from scenesmith.agent_utils.house import (
    ConnectionType,
    Door,
    HouseLayout,
    Opening,
    OpeningType,
    PlacedRoom,
    RoomGeometry,
    RoomMaterials,
    RoomSpec,
    Wall,
    WallDirection,
    Window,
    WindowShape,
    legacy_openings_to_boundary_portals,
)
from scenesmith.agent_utils.semantic_environments import (
    Bounds3D,
    CavernChamberSpec,
    DetailFieldSpec,
    DetailSurfaceRole,
    EnvironmentKind,
    EnvironmentRegionSpec,
    FormationType,
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structural_compiler import TriangleMesh
from scenesmith.agent_utils.structural_geometry import (
    SCHEMA_VERSION,
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    GeometryValidationError,
    HeightfieldSpec,
    LevelSpec,
    PlatformSpec,
    PortalSpec,
    PortalType,
    StructuralMeshSpec,
    SurfaceRole,
    UnsupportedGeometryError,
)
from scenesmith.utils.material import Material


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


class TestV2StructuralLayout(unittest.TestCase):
    """Tests for v2 migration, levels, footprints, and room transforms."""

    def test_v1_dictionary_migrates_to_ground_level(self) -> None:
        legacy = {
            "wall_height": 2.5,
            "rooms": [
                {
                    "id": "main",
                    "type": "room",
                    "width": 4.0,
                    "length": 5.0,
                }
            ],
            "placed_rooms": [
                {
                    "room_id": "main",
                    "position": [0.0, 0.0],
                    "width": 5.0,
                    "depth": 4.0,
                    "walls": [],
                }
            ],
        }

        migrated = HouseLayout.from_dict(legacy)

        assert migrated.schema_version == SCHEMA_VERSION
        assert [level.level_id for level in migrated.levels] == ["ground"]
        assert migrated.room_specs[0].level_id == "ground"
        assert migrated.placed_rooms[0].level_id == "ground"
        assert migrated.get_room_elevation("main") == 0.0

    def test_multilevel_layout_round_trip_preserves_structure(self) -> None:
        footprint = Footprint2D(outer=((0, 0), (5, 0), (5, 4), (2, 4), (2, 2), (0, 2)))
        connector = ConnectorSpec(
            connector_id="stairs",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (4, 0, 3)),
            parameters={"riser_count": 18},
        )
        layout = HouseLayout(
            levels=[LevelSpec("ground", 0.0), LevelSpec("upper_level", 3.0)],
            room_specs=[
                RoomSpec("lower"),
                RoomSpec("upper", level_id="upper_level", footprint=footprint),
            ],
            placed_rooms=[
                PlacedRoom("lower", (0, 0), 5, 4),
                PlacedRoom(
                    "upper",
                    (1, 2),
                    5,
                    4,
                    level_id="upper_level",
                    yaw=0.25,
                    footprint=footprint,
                ),
            ],
            connectors=[connector],
        )
        layout.validate_structure()

        restored = HouseLayout.from_dict(layout.to_dict())

        assert restored.levels == layout.levels
        assert restored.connectors == layout.connectors
        assert restored.placed_rooms[1].footprint == footprint
        assert restored.placed_rooms[1].yaw == 0.25
        assert restored.get_room_elevation("upper") == 3.0

    def test_drake_directive_preserves_room_z_and_yaw(self) -> None:
        layout = HouseLayout(
            levels=[LevelSpec("upper", 3.25)],
            room_specs=[RoomSpec("gallery", level_id="upper")],
            placed_rooms=[
                PlacedRoom(
                    "gallery",
                    (2.0, 4.0),
                    6.0,
                    4.0,
                    level_id="upper",
                    yaw=math.pi / 6,
                )
            ],
            room_geometries={
                "gallery": RoomGeometry(
                    sdf_tree=ET.ElementTree(ET.Element("sdf")),
                    sdf_path=Path("gallery.sdf"),
                )
            },
        )

        directive = layout.to_drake_directive()

        assert "translation: [5.0, 6.0, 3.25]" in directive
        assert "angle_deg: 29.999" in directive
        assert "axis: [0, 0, 1]" in directive

    def test_yaw_only_room_is_routed_through_surface_compiler(self) -> None:
        layout = HouseLayout(
            room_specs=[RoomSpec("rotated", length=5, width=4, yaw=math.pi / 4)],
            placed_rooms=[PlacedRoom("rotated", (2, 3), 5, 4, yaw=math.pi / 4)],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = layout.compile_polygon_rooms(Path(temporary_directory))

        self.assertIn("rotated", paths)
        self.assertIsNotNone(layout.room_geometries["rotated"].structural_surface_path)

    def test_house_compiles_and_exports_connector_models(self) -> None:
        connector = ConnectorSpec(
            connector_id="stairs",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (4, 0, 3)),
            parameters={"riser_count": 18},
        )
        layout = HouseLayout(
            levels=[LevelSpec("ground", 0), LevelSpec("upper_level", 3)],
            room_specs=[RoomSpec("lower"), RoomSpec("upper", level_id="upper_level")],
            placed_rooms=[
                PlacedRoom("lower", (0, 0), 4, 4),
                PlacedRoom("upper", (0, 0), 4, 4, level_id="upper_level"),
            ],
            room_geometries={
                "lower": RoomGeometry(
                    sdf_tree=ET.ElementTree(ET.Element("sdf")),
                    sdf_path=Path("lower.sdf"),
                ),
                "upper": RoomGeometry(
                    sdf_tree=ET.ElementTree(ET.Element("sdf")),
                    sdf_path=Path("upper.sdf"),
                ),
            },
            connectors=[connector],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = layout.compile_connectors(output_dir / "connectors")
            directive = layout.to_drake_directive(base_dir=output_dir)

            assert paths["stairs"].exists()
            assert "name: structure_stairs" in directive
            assert "child: structure_stairs::structure_link" in directive
            self.assertIn(
                f"package://scene/{paths['stairs'].relative_to(output_dir)}", directive
            )
            state = layout.to_dict(scene_dir=output_dir)
            self.assertEqual(
                state["connector_geometry_paths"]["stairs"],
                str(paths["stairs"].relative_to(output_dir)),
            )

    def test_export_refuses_uncompiled_connector(self) -> None:
        connector = ConnectorSpec(
            connector_id="ramp",
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (12, 0, 1)),
        )
        layout = HouseLayout(
            levels=[LevelSpec("ground", 0), LevelSpec("upper_level", 1)],
            room_specs=[RoomSpec("lower"), RoomSpec("upper", level_id="upper_level")],
            placed_rooms=[PlacedRoom("lower", (0, 0), 4, 4)],
            room_geometries={
                "lower": RoomGeometry(
                    sdf_tree=ET.ElementTree(ET.Element("sdf")),
                    sdf_path=Path("lower.sdf"),
                )
            },
            connectors=[connector],
        )
        with self.assertRaisesRegex(ValueError, "compile_connectors"):
            layout.to_drake_directive()

    def test_embedded_natural_passage_uses_room_mesh_without_duplicate_model(
        self,
    ) -> None:
        connector = ConnectorSpec(
            connector_id="cave_tunnel",
            connector_type=ConnectorType.NATURAL_PASSAGE,
            start=ConnectorEndpoint("lower", "ground", (1, 1, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (3, 1, 3)),
            parameters={"geometry_embedded": True, "waypoints": [(2, 1, 1.5)]},
        )
        layout = HouseLayout(
            levels=[LevelSpec("ground", 0), LevelSpec("upper_level", 3)],
            room_specs=[RoomSpec("lower"), RoomSpec("upper", level_id="upper_level")],
            placed_rooms=[
                PlacedRoom("lower", (0, 0), 4, 4),
                PlacedRoom("upper", (0, 0), 4, 4, level_id="upper_level"),
            ],
            room_geometries={
                room_id: RoomGeometry(
                    sdf_tree=ET.ElementTree(ET.Element("sdf")),
                    sdf_path=Path(f"{room_id}.sdf"),
                )
                for room_id in ("lower", "upper")
            },
            connectors=[connector],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = layout.compile_connectors(Path(temporary_directory) / "connectors")
            directive = layout.to_drake_directive()

        assert paths == {}
        assert "structure_cave_tunnel" not in directive
        assert layout.build_topology().reachable("lower", capabilities={"walk"}) == {
            "lower",
            "upper",
        }

    def test_nonembedded_natural_passage_fails_explicitly(self) -> None:
        connector = ConnectorSpec(
            connector_id="unmodeled_tunnel",
            connector_type=ConnectorType.NATURAL_PASSAGE,
            start=ConnectorEndpoint("lower", "ground", (1, 1, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (3, 1, 3)),
        )
        layout = HouseLayout(
            levels=[LevelSpec("ground", 0), LevelSpec("upper_level", 3)],
            room_specs=[RoomSpec("lower"), RoomSpec("upper", level_id="upper_level")],
            connectors=[connector],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(UnsupportedGeometryError, "natural_passage"):
                layout.compile_connectors(Path(temporary_directory))

    def test_house_compiles_polygon_room_with_legacy_link_contract(self) -> None:
        footprint = Footprint2D(outer=((0, 0), (5, 0), (5, 2), (2, 2), (2, 4), (0, 4)))
        layout = HouseLayout(
            room_specs=[RoomSpec("gallery", length=5, width=4, footprint=footprint)],
            placed_rooms=[PlacedRoom("gallery", (10, 20), 5, 4, footprint=footprint)],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = layout.compile_polygon_rooms(output_dir / "rooms")
            directive = layout.to_drake_directive(base_dir=output_dir)
            geometry = layout.room_geometries["gallery"]
            sdf_root = ET.parse(paths["gallery"]).getroot()

            assert sdf_root.find(".//link[@name='room_geometry_body_link']") is not None
            assert "translation: [12.5, 22.0, 0.0]" in directive
            self.assertIn(
                f"package://scene/{paths['gallery'].relative_to(output_dir)}",
                directive,
            )
            assert geometry.footprint is not None
            assert geometry.footprint.bounds == (-2.5, -2.0, 2.5, 2.0)
            assert geometry.structural_surface_path is not None
            assert geometry.structural_surface_path.exists()
            assert any(
                SurfaceRole.TRAVERSABLE in surface.roles
                for surface in geometry.structural_surfaces
            )

    def test_house_compiles_rectangular_room_with_sloped_floor(self) -> None:
        layout = HouseLayout(
            room_specs=[
                RoomSpec(
                    "ramp_room",
                    length=6,
                    width=4,
                    floor_profile=ElevationProfile(
                        profile_type=ElevationProfileType.SLOPED,
                        gradient=(0.1, 0.0),
                    ),
                )
            ],
            placed_rooms=[PlacedRoom("ramp_room", (0, 0), 6, 4)],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = layout.compile_polygon_rooms(output_dir / "rooms")
            geometry = layout.room_geometries["ramp_room"]

            assert paths["ramp_room"].exists()
            assert geometry.footprint is not None
            assert geometry.footprint.bounds == (-3.0, -2.0, 3.0, 2.0)
            assert geometry.floor_profile.gradient == (0.1, 0.0)

    def test_polygon_room_bounds_must_match_legacy_dimensions(self) -> None:
        layout = HouseLayout(
            room_specs=[
                RoomSpec(
                    "bad",
                    length=6,
                    width=4,
                    footprint=Footprint2D.rectangle(5, 4),
                )
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "footprint bounds"):
                layout.compile_polygon_rooms(Path(temporary_directory))

    def test_house_compiles_and_exports_freeform_cavern(self) -> None:
        mesh = TriangleMesh(
            vertices=((0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 3)),
            triangles=((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            source_path = output_dir / "source" / "cavern.obj"
            source_path.parent.mkdir()
            source_path.write_text(mesh.to_obj(), encoding="utf-8")
            layout = HouseLayout(
                room_specs=[RoomSpec("cavern")],
                placed_rooms=[PlacedRoom("cavern", (0, 0), 5, 5)],
                room_geometries={
                    "cavern": RoomGeometry(
                        sdf_tree=ET.ElementTree(ET.Element("sdf")),
                        sdf_path=Path("cavern.sdf"),
                    )
                },
                structural_meshes=[
                    StructuralMeshSpec(
                        mesh_id="cavern_shell",
                        space_id="cavern",
                        mesh_path=str(source_path),
                        unit_scale=1.0,
                        require_watertight=True,
                    )
                ],
            )

            paths = layout.compile_structural_meshes(output_dir / "compiled")
            directive = layout.to_drake_directive(base_dir=output_dir)
            state = layout.to_dict(scene_dir=output_dir)
            # RoomScene deserialization imports Drake; structural mesh state is
            # independently round-trippable in this dependency-light suite.
            state["room_geometries"] = {}
            restored = HouseLayout.from_dict(state, house_dir=output_dir)

            assert paths["cavern_shell"].exists()
            assert "name: structure_cavern_shell" in directive
            assert "child: structure_cavern_shell::structure_link" in directive
            assert "parent: room_cavern_frame" in directive
            self.assertIn(
                f"package://scene/{paths['cavern_shell'].relative_to(output_dir)}",
                directive,
            )
            assert state["structural_meshes"][0]["mesh_path"] == ("source/cavern.obj")
            assert layout.room_geometries[
                "cavern"
            ].additional_structural_surface_paths == [
                paths["cavern_shell"].with_suffix(".surfaces.json")
            ]
            assert restored.structural_meshes[0].mesh_path == str(source_path)

    def test_freeform_cavern_can_replace_flat_room_shell(self) -> None:
        mesh = TriangleMesh(
            vertices=((-2, -2, 0), (2, -2, 0), (0, 2, 0), (0, 0, 3)),
            # Inward-facing winding: this is an interior cavern shell.
            triangles=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            source_path = output_dir / "cavern.obj"
            source_path.write_text(mesh.to_obj(), encoding="utf-8")
            layout = HouseLayout(
                room_specs=[RoomSpec("cavern", length=4, width=4)],
                placed_rooms=[PlacedRoom("cavern", (-2, -2), 4, 4)],
                structural_meshes=[
                    StructuralMeshSpec(
                        mesh_id="cavern_shell",
                        space_id="cavern",
                        mesh_path=str(source_path),
                        unit_scale=1.0,
                        require_watertight=True,
                        normal_orientation="interior",
                        replaces_room_shell=True,
                    )
                ],
            )

            paths = layout.compile_structural_meshes(output_dir / "compiled")
            directive = layout.to_drake_directive(base_dir=output_dir)
            geometry = layout.room_geometries["cavern"]
            sdf_root = ET.parse(paths["cavern_shell"]).getroot()

            assert sdf_root.find(".//link[@name='room_geometry_body_link']") is not None
            assert geometry.sdf_path == paths["cavern_shell"]
            assert geometry.structural_surface_path == paths[
                "cavern_shell"
            ].with_suffix(".surfaces.json")
            assert geometry.additional_structural_surface_paths == []
            assert "name: room_geometry_cavern" in directive
            assert "child: room_geometry_cavern::room_geometry_body_link" in directive
            assert "name: structure_cavern_shell" not in directive

    def test_duplicate_freeform_room_shells_are_rejected(self) -> None:
        meshes = [
            StructuralMeshSpec(
                mesh_id=f"shell_{index}",
                space_id="cavern",
                mesh_path=f"shell_{index}.obj",
                unit_scale=1.0,
                replaces_room_shell=True,
            )
            for index in range(2)
        ]
        layout = HouseLayout(room_specs=[RoomSpec("cavern")], structural_meshes=meshes)

        with self.assertRaisesRegex(GeometryValidationError, "one structural mesh"):
            layout.validate_structure()

    def test_house_compiles_mezzanine_in_room_frame(self) -> None:
        layout = HouseLayout(
            room_specs=[RoomSpec("loft", level_id="upper")],
            levels=[LevelSpec("upper", elevation=3.0)],
            placed_rooms=[PlacedRoom("loft", (10, 20), 8, 6, level_id="upper")],
            room_geometries={
                "loft": RoomGeometry(
                    sdf_tree=ET.ElementTree(ET.Element("sdf")),
                    sdf_path=Path("loft.sdf"),
                )
            },
            platforms=[
                PlatformSpec(
                    platform_id="mezzanine",
                    space_id="loft",
                    footprint=Footprint2D.rectangle(4, 2),
                    elevation=2.5,
                    open_edge_indices=(2,),
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = layout.compile_platforms(output_dir / "platforms")
            directive = layout.to_drake_directive(base_dir=output_dir)
            state = layout.to_dict(scene_dir=output_dir)
            state["room_geometries"] = {}
            restored = HouseLayout.from_dict(state, house_dir=output_dir)

            assert paths["mezzanine"].exists()
            assert "parent: room_loft_frame" in directive
            assert "child: structure_mezzanine::structure_link" in directive
            assert layout.room_geometries[
                "loft"
            ].additional_structural_surface_paths == [
                paths["mezzanine"].with_suffix(".surfaces.json")
            ]
            assert restored.platforms == layout.platforms

    def test_house_compiles_heightfield_in_room_frame(self) -> None:
        layout = HouseLayout(
            room_specs=[RoomSpec("cavern")],
            placed_rooms=[PlacedRoom("cavern", (0, 0), 4, 4)],
            room_geometries={
                "cavern": RoomGeometry(
                    sdf_tree=ET.ElementTree(ET.Element("sdf")),
                    sdf_path=Path("cavern.sdf"),
                )
            },
            heightfields=[
                HeightfieldSpec(
                    heightfield_id="rough_floor",
                    space_id="cavern",
                    heights=((0, 0.1), (0.2, 0.3)),
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = layout.compile_heightfields(output_dir / "heightfields")
            directive = layout.to_drake_directive(base_dir=output_dir)
            state = layout.to_dict(scene_dir=output_dir)
            state["room_geometries"] = {}
            restored = HouseLayout.from_dict(state, house_dir=output_dir)

            assert paths["rough_floor"].exists()
            assert "parent: room_cavern_frame" in directive
            assert "child: structure_rough_floor::structure_link" in directive
            assert layout.room_geometries[
                "cavern"
            ].additional_structural_surface_paths == [
                paths["rough_floor"].with_suffix(".surfaces.json")
            ]
            assert restored.heightfields == layout.heightfields

    def test_heightfield_can_replace_room_floor_without_overlap(self) -> None:
        layout = HouseLayout(
            room_specs=[RoomSpec("terrain", length=4, width=4)],
            placed_rooms=[PlacedRoom("terrain", (0, 0), 4, 4)],
            heightfields=[
                HeightfieldSpec(
                    heightfield_id="terrain_floor",
                    space_id="terrain",
                    heights=((0, 0.1), (0.2, 0.3)),
                    cell_size=(4, 4),
                    origin=(-2, -2, 0),
                    replaces_floor=True,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            layout.compile_polygon_rooms(output_dir / "room")
            primary = layout.room_geometries["terrain"]
            self.assertFalse(
                any(
                    SurfaceRole.SUPPORT in surface.roles
                    for surface in primary.structural_surfaces
                )
            )

            paths = layout.compile_heightfields(output_dir / "heightfield")

            self.assertTrue(paths["terrain_floor"].exists())
            self.assertEqual(len(primary.additional_structural_surface_paths), 1)


if __name__ == "__main__":
    unittest.main()
