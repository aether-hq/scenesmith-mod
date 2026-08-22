"""Cross-feature structural scenarios from the geometry capability matrix."""

import tempfile
import unittest

from pathlib import Path

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.structure.compiler.models import TriangleMesh
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    LevelSpec,
    PlatformSpec,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
)
from scenesmith.agent_utils.structure.structural_surfaces import (
    StructuralSurfaceIndex,
    load_surface_patches,
)


class TestCrossFeatureScenarios(unittest.TestCase):
    def test_embedded_natural_passage_uses_sloped_shell_for_clearance(self) -> None:
        passage = ConnectorSpec(
            connector_id="rising_tunnel",
            connector_type=ConnectorType.NATURAL_PASSAGE,
            start=ConnectorEndpoint("lower", "ground", (0, 1, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (5, 1, 3)),
            parameters={"geometry_embedded": True},
        )
        layout = HouseLayout(
            wall_height=2.5,
            levels=[LevelSpec("ground", 0), LevelSpec("upper_level", 3)],
            room_specs=[
                RoomSpec(
                    "lower",
                    length=5,
                    width=2,
                    footprint=Footprint2D.rectangle(5, 2),
                    floor_profile=ElevationProfile(
                        ElevationProfileType.SLOPED,
                        base_elevation=1.5,
                        gradient=(0.6, 0),
                    ),
                ),
                RoomSpec("upper", level_id="upper_level"),
            ],
            placed_rooms=[PlacedRoom("lower", (0, 0), 5, 2)],
            connectors=[passage],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            layout.compile_polygon_rooms(output_dir / "rooms")
            self.assertEqual(layout.compile_connectors(output_dir / "connectors"), {})

            self.assertEqual(layout.geometrically_blocked_connectors(), frozenset())
            self.assertEqual(
                layout.build_topology().reachable(
                    "lower",
                    blocked_edges=layout.geometrically_blocked_connectors(),
                ),
                frozenset({"lower", "upper"}),
            )

    def test_concave_multilevel_house_with_stairs_exports_connected_scene(self) -> None:
        lower_footprint = Footprint2D(
            outer=((0, 0), (6, 0), (6, 2), (3, 2), (3, 5), (0, 5))
        )
        stairs = ConnectorSpec(
            connector_id="main_stairs",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (4, 0, 3)),
            parameters={"riser_count": 18},
        )
        layout = HouseLayout(
            levels=[LevelSpec("ground", 0), LevelSpec("upper_level", 3)],
            room_specs=[
                RoomSpec("lower", length=6, width=5, footprint=lower_footprint),
                RoomSpec(
                    "upper",
                    length=6,
                    width=5,
                    level_id="upper_level",
                    footprint=Footprint2D.rectangle(6, 5),
                ),
            ],
            placed_rooms=[
                PlacedRoom("lower", (0, 0), 6, 5, footprint=lower_footprint),
                PlacedRoom(
                    "upper",
                    (0, 0),
                    6,
                    5,
                    level_id="upper_level",
                    footprint=Footprint2D.rectangle(6, 5),
                ),
            ],
            connectors=[stairs],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            layout.compile_polygon_rooms(output_dir / "rooms")
            layout.compile_connectors(output_dir / "connectors")
            directive = layout.to_drake_directive(base_dir=output_dir)

            self.assertEqual(
                layout.build_topology().reachable("lower"),
                frozenset({"lower", "upper"}),
            )
            self.assertIn("translation: [3.0, 2.5, 3.0]", directive)
            self.assertIn("structure_main_stairs", directive)

    def test_atrium_hole_remains_empty_but_authored_bridge_supports_objects(
        self,
    ) -> None:
        footprint = Footprint2D(
            outer=((0, 0), (8, 0), (8, 8), (0, 8)),
            holes=(((3, 3), (3, 5), (5, 5), (5, 3)),),
        )
        layout = HouseLayout(
            room_specs=[RoomSpec("atrium", length=8, width=8, footprint=footprint)],
            placed_rooms=[PlacedRoom("atrium", (0, 0), 8, 8, footprint=footprint)],
            platforms=[
                PlatformSpec(
                    platform_id="bridge",
                    space_id="atrium",
                    footprint=Footprint2D.rectangle(4, 1).centered_on_bounds(),
                    elevation=0.2,
                    open_edge_indices=(0, 2),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            layout.compile_polygon_rooms(output_dir / "room")
            self.assertIsNone(
                StructuralSurfaceIndex(
                    load_surface_patches(
                        layout.room_geometries["atrium"].structural_surface_path
                    )
                ).support_pose(0, 0)
            )
            layout.compile_platforms(output_dir / "platform")
            geometry = layout.room_geometries["atrium"]
            index = StructuralSurfaceIndex(
                patch
                for path in (
                    geometry.structural_surface_path,
                    *geometry.additional_structural_surface_paths,
                )
                for patch in load_surface_patches(path)
            )
            pose = index.support_pose(0, 0)

            assert pose is not None
            self.assertAlmostEqual(pose.position[2], 0.2)
            self.assertEqual(pose.surface_id, "bridge_top")

    def test_cavern_shell_with_built_platform_uses_one_room_surface_index(self) -> None:
        mesh = TriangleMesh(
            vertices=((-3, -3, 0), (3, -3, 0), (0, 3, 0), (0, 0, 4)),
            triangles=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            source_path = output_dir / "cavern.obj"
            source_path.write_text(mesh.to_obj(), encoding="utf-8")
            layout = HouseLayout(
                room_specs=[RoomSpec("cavern", length=6, width=6)],
                placed_rooms=[PlacedRoom("cavern", (-3, -3), 6, 6)],
                structural_meshes=[
                    StructuralMeshSpec(
                        mesh_id="cavern_shell",
                        space_id="cavern",
                        mesh_path=str(source_path),
                        unit_scale=1,
                        normal_orientation="interior",
                        require_watertight=True,
                        replaces_room_shell=True,
                    )
                ],
                platforms=[
                    PlatformSpec(
                        platform_id="altar",
                        space_id="cavern",
                        footprint=Footprint2D.rectangle(2, 2).centered_on_bounds(),
                        elevation=1,
                    )
                ],
            )

            layout.compile_structural_meshes(output_dir / "meshes")
            layout.compile_platforms(output_dir / "platforms")
            geometry = layout.room_geometries["cavern"]
            index = StructuralSurfaceIndex(
                patch
                for path in (
                    geometry.structural_surface_path,
                    *geometry.additional_structural_surface_paths,
                )
                for patch in load_surface_patches(path)
            )
            pose = index.support_pose(0, 0)

            assert pose is not None
            self.assertEqual(pose.surface_id, "altar_top")
            self.assertAlmostEqual(pose.position[2], 1)
            self.assertEqual(len(geometry.additional_structural_surface_paths), 1)

    def test_stair_clearance_veto_requires_independent_slab_openings(self) -> None:
        boundary = Footprint2D.rectangle(6, 4)
        opening = Footprint2D(
            outer=boundary.outer,
            holes=(((0.5, 1.5), (0.5, 2.5), (5.5, 2.5), (5.5, 1.5)),),
        )
        stairs = ConnectorSpec(
            connector_id="stairs",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint("lower", "ground", (1, 2, 0)),
            end=ConnectorEndpoint("upper", "upper_level", (5, 2, 3)),
            width=1,
            parameters={"riser_count": 18},
        )

        def make_layout(*, cut_openings: bool) -> HouseLayout:
            return HouseLayout(
                levels=[LevelSpec("ground", 0), LevelSpec("upper_level", 3)],
                room_specs=[
                    RoomSpec(
                        "lower",
                        length=6,
                        width=4,
                        footprint=boundary,
                        ceiling_footprint=opening if cut_openings else boundary,
                    ),
                    RoomSpec(
                        "upper",
                        length=6,
                        width=4,
                        level_id="upper_level",
                        footprint=boundary,
                        floor_footprint=opening if cut_openings else boundary,
                    ),
                ],
                placed_rooms=[
                    PlacedRoom("lower", (0, 0), 6, 4, footprint=boundary),
                    PlacedRoom(
                        "upper",
                        (0, 0),
                        6,
                        4,
                        level_id="upper_level",
                        footprint=boundary,
                    ),
                ],
                connectors=[stairs],
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            blocked_layout = make_layout(cut_openings=False)
            blocked_layout.compile_polygon_rooms(output_dir / "blocked_rooms")
            blocked_layout.compile_connectors(output_dir / "blocked_connectors")
            self.assertEqual(
                blocked_layout.geometrically_blocked_connectors(),
                frozenset({"stairs"}),
            )
            self.assertEqual(
                len(
                    blocked_layout.build_topology().topology_geometry_mismatches(
                        {"stairs"}
                    )
                ),
                1,
            )

            clear_layout = make_layout(cut_openings=True)
            clear_layout.compile_polygon_rooms(output_dir / "clear_rooms")
            clear_layout.compile_connectors(output_dir / "clear_connectors")
            self.assertEqual(
                clear_layout.geometrically_blocked_connectors(), frozenset()
            )


if __name__ == "__main__":
    unittest.main()
