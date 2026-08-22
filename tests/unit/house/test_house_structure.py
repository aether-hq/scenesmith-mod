"""Tests for house.py dataclass serialization."""

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

from pygltflib import GLTF2

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom, RoomMaterials
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.structure.geometry_models.common import SCHEMA_VERSION
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    LevelSpec,
    SurfaceRole,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    UnsupportedGeometryError,
)
from scenesmith.utils.geometry.material import Material


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

    def test_polygon_room_exports_selected_floor_material_as_visual_finish(
        self,
    ) -> None:
        floor_material = Material.from_path(
            Path(__file__).parents[3] / "data/materials/WoodFloor014"
        )
        footprint = Footprint2D.rectangle(6, 5)
        layout = HouseLayout(
            room_specs=[RoomSpec("library", length=6, width=5, footprint=footprint)],
            placed_rooms=[PlacedRoom("library", (0, 0), 6, 5, footprint=footprint)],
            room_materials={"library": RoomMaterials(floor_material=floor_material)},
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = layout.compile_polygon_rooms(Path(temporary_directory))
            sdf_path = paths["library"]
            visual_uris = [
                node.text
                for node in ET.parse(sdf_path).findall(".//visual/geometry/mesh/uri")
            ]

            assert len(visual_uris) == 2
            assert visual_uris[0].endswith(".obj")
            assert visual_uris[1].endswith(".floor_finish.glb")
            finish_path = sdf_path.parent / visual_uris[1]
            gltf = GLTF2().load(str(finish_path))
            assert gltf.meshes[0].primitives[0].material == 0
            assert len(gltf.images) == 3
            assert all(image.uri.startswith("data:image/") for image in gltf.images)
            assert all(
                node.find("geometry/mesh") is None
                for node in ET.parse(sdf_path).findall(".//collision")
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
