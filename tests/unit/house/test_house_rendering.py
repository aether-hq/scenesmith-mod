"""Tests for house.py dataclass serialization."""

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform
from pygltflib import GLTF2

from scenesmith.agent_utils.scene.house import HouseLayout, HouseScene
from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom, RoomMaterials
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.agent_utils.structure.compiler.models import TriangleMesh
from scenesmith.agent_utils.structure.compiler.surfaces import compile_platform
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    Footprint2D,
    HeightfieldSpec,
    LevelSpec,
    PlatformSpec,
    SurfaceRole,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)
from scenesmith.utils.geometry.material import Material


class TestV2StructuralLayout(unittest.TestCase):
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

    def test_house_platform_inherits_room_floor_material_and_uvs(self) -> None:
        material = Material.from_path(
            Path(__file__).parents[3] / "data/materials/WoodFloor014"
        )
        platform = PlatformSpec(
            platform_id="renaissance_gallery",
            space_id="library",
            footprint=Footprint2D(
                outer=((0, 0), (8, 0), (8, 8), (0, 8)),
                holes=(((2, 2), (2, 6), (6, 6), (6, 2)),),
            ),
            elevation=4.0,
            guarded_hole_indices=(0,),
        )
        layout = HouseLayout(
            room_specs=[RoomSpec("library")],
            room_materials={
                "library": RoomMaterials(
                    wall_material=Material.from_path("materials/plaster"),
                    floor_material=material,
                )
            },
            platforms=[platform],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = layout.compile_platforms(Path(temporary_directory))
            sdf_path = paths[platform.platform_id]
            visual_uri = ET.parse(sdf_path).findtext(".//visual/geometry/mesh/uri")
            assert visual_uri is not None
            assert visual_uri.endswith(".glb")
            visual_path = sdf_path.parent / visual_uri
            assert visual_path.is_file()
            gltf = GLTF2().load(str(visual_path))
            primitive = gltf.meshes[0].primitives[0]
            assert primitive.attributes.TEXCOORD_0 is not None
            assert primitive.material == 0
            assert len(gltf.materials) == 1
            assert len(gltf.images) == 3
            assert all(image.uri.startswith("data:image/") for image in gltf.images)
            assert gltf.accessors[primitive.indices].count == (
                len(compile_platform(platform).visual_mesh.triangles) * 3
            )

    def test_house_blender_platform_visuals_keep_room_frame_transform(self) -> None:
        platform = PlatformSpec(
            platform_id="renaissance_gallery",
            space_id="library",
            footprint=Footprint2D.rectangle(8, 6),
            elevation=4.0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory)
            visual_path = scene_dir / "renaissance_gallery.glb"
            sdf_path = visual_path.with_suffix(".sdf")
            visual_path.touch()
            sdf_path.write_text(
                "<sdf><model><link><visual><geometry><mesh>"
                "<uri>renaissance_gallery.glb</uri>"
                "</mesh></geometry></visual></link></model></sdf>",
                encoding="utf-8",
            )
            layout = HouseLayout(
                room_specs=[RoomSpec("library", level_id="upper")],
                levels=[LevelSpec("upper", elevation=3.0)],
                placed_rooms=[
                    PlacedRoom(
                        "library",
                        (10, 20),
                        8,
                        6,
                        level_id="upper",
                        yaw=math.pi / 2,
                    )
                ],
                platforms=[platform],
                platform_geometry_paths={platform.platform_id: sdf_path},
                house_dir=scene_dir,
            )

            visuals = HouseScene(layout=layout)._platform_blender_visuals()

            self.assertEqual(
                visuals,
                [
                    {
                        "path": str(visual_path),
                        "translation": [14.0, 23.0, 3.0],
                        "yaw_radians": math.pi / 2,
                        "role": "structural_detail",
                        "source_id": platform.platform_id,
                    }
                ],
            )

    def test_house_blender_room_floor_finish_keeps_room_frame_transform(self) -> None:
        footprint = Footprint2D.rectangle(6, 5)
        layout = HouseLayout(
            room_specs=[
                RoomSpec(
                    "library",
                    length=6,
                    width=5,
                    footprint=footprint,
                    level_id="upper",
                )
            ],
            levels=[LevelSpec("upper", elevation=3.0)],
            placed_rooms=[
                PlacedRoom(
                    "library",
                    (10, 20),
                    6,
                    5,
                    footprint=footprint,
                    level_id="upper",
                    yaw=math.pi / 2,
                )
            ],
            room_materials={
                "library": RoomMaterials(
                    floor_material=Material.from_path(
                        Path(__file__).parents[3] / "data/materials/WoodFloor014"
                    )
                )
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = layout.compile_polygon_rooms(Path(temporary_directory))
            sdf_path = paths["library"]
            # ElementTree does not support XPath contains(), so recover the
            # exact authored finish URI from the visual name.
            finish_uri = next(
                visual.findtext("geometry/mesh/uri")
                for visual in ET.parse(sdf_path).findall(".//visual")
                if "floor_finish" in visual.get("name", "")
            )

            visuals = HouseScene(layout=layout)._room_floor_blender_visuals()

            self.assertEqual(
                visuals,
                [
                    {
                        "path": str(sdf_path.parent / finish_uri),
                        "translation": [13.0, 22.5, 3.0],
                        "yaw_radians": math.pi / 2,
                        "role": "structural_detail",
                        "source_id": "library_floor_finish",
                    }
                ],
            )

    def test_house_blender_populates_validated_renaissance_bookcase_run(self) -> None:
        prompt = "large multi-level Renaissance library with thousands of books"
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_dir = Path(temporary_directory)
            layout = HouseLayout(
                room_specs=[RoomSpec("library", length=13, width=13)],
                placed_rooms=[PlacedRoom("library", (0, 0), 13, 13)],
                house_prompt=prompt,
                house_dir=scene_dir,
            )
            room = RoomScene(
                room_geometry=object(),
                scene_dir=scene_dir / "room_library",
                room_id="library",
                text_description=prompt,
            )
            for index, x in enumerate((-2.1, -1.05, 0.0, 1.05, 2.1)):
                owner_id = UniqueID(f"bookcase_{index}")
                room.add_object(
                    SceneObject(
                        object_id=owner_id,
                        object_type=ObjectType.FURNITURE,
                        name="renaissance_bookcase",
                        description="full-height Renaissance library bookcase",
                        transform=RigidTransform(p=[x, 5.2, 0.0]),
                        bbox_min=np.array([-0.48, -0.18, 0.0]),
                        bbox_max=np.array([0.48, 0.18, 2.0]),
                        metadata={
                            "dense_library_populated_case": 0.0,
                            **({"dense_library_grouped_run": 0.0} if index < 3 else {}),
                        },
                    )
                )
                for row_index in range(3):
                    room.add_object(
                        SceneObject(
                            object_id=UniqueID(f"book_row_{index}_{row_index}"),
                            object_type=ObjectType.MANIPULAND,
                            name="encyclopedia_book_row",
                            description="visible encyclopedia set",
                            transform=RigidTransform(),
                            metadata={
                                "dense_library_book_row": True,
                                "dense_library_owner_bound": str(owner_id),
                            },
                        )
                    )
            room.add_object(
                SceneObject(
                    object_id=UniqueID("isolated_bookcase"),
                    object_type=ObjectType.FURNITURE,
                    name="renaissance_bookcase",
                    description="isolated full-height bookcase",
                    transform=RigidTransform(p=[-5.0, -5.0, 0.0]),
                    bbox_min=np.array([-0.48, -0.18, 0.0]),
                    bbox_max=np.array([0.48, 0.18, 2.0]),
                )
            )

            visuals = HouseScene(
                layout=layout,
                rooms={"library": room},
            )._renaissance_bookcase_blender_visuals(scene_dir / "bookcase_dressing")

            self.assertEqual(len(visuals), 1)
            self.assertEqual(visuals[0]["populated_bookcases"], 5)
            self.assertEqual(visuals[0]["shelf_tiers_per_bookcase"], 6)
            self.assertGreaterEqual(visuals[0]["visible_book_spines"], 300)
            gltf = GLTF2().load(visuals[0]["path"])
            node_names = {node.name for node in gltf.nodes}
            self.assertIn("renaissance_bookcase_walnut", node_names)
            self.assertIn("renaissance_bookcase_antique_gold", node_names)
            self.assertGreaterEqual(len(gltf.meshes), 8)
            self.assertGreaterEqual(len(gltf.materials), 8)

    def test_textured_landing_retains_obj_collision_mesh(self) -> None:
        platform = PlatformSpec(
            platform_id="stair_landing",
            space_id="library",
            footprint=Footprint2D.rectangle(2, 2),
            elevation=4.0,
        )
        layout = HouseLayout(
            room_specs=[RoomSpec("library")],
            room_materials={
                "library": RoomMaterials(
                    floor_material=Material.from_path(
                        Path(__file__).parents[3] / "data/materials/WoodFloor014"
                    )
                )
            },
            platforms=[platform],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            sdf_path = layout.compile_platforms(Path(temporary_directory))[
                platform.platform_id
            ]
            sdf = ET.parse(sdf_path)
            assert sdf.findtext(".//visual/geometry/mesh/uri") == ("stair_landing.glb")
            assert sdf.findtext(".//collision/geometry/mesh/uri") == (
                "stair_landing.collision.obj"
            )
            assert (sdf_path.parent / "stair_landing.collision.obj").is_file()

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
