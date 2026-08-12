"""Tests for deterministic parametric structural compilers."""

import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

from scenesmith.agent_utils.structural_compiler import (
    TriangleMesh,
    compile_connector,
    compile_heightfield,
    compile_ladder,
    compile_multisegment_ramp,
    compile_multisegment_stairs,
    compile_platform,
    compile_polygon_space,
    compile_spiral_stairs,
    compile_straight_ramp,
    compile_straight_stairs,
    compile_structural_mesh,
    write_compiled_structure,
)
from scenesmith.agent_utils.structural_geometry import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    GeometryValidationError,
    HeightfieldSpec,
    MeshSurfaceAnnotation,
    PlatformSpec,
    PortalSpec,
    PortalType,
    StructuralMeshSpec,
    SurfaceRole,
    Transform3D,
    UnsafeConnectorError,
    UnsupportedGeometryError,
)


class TestTriangleMesh(unittest.TestCase):
    def test_rejects_degenerate_triangle(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero area"):
            TriangleMesh(
                vertices=((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                triangles=((0, 1, 2),),
            )

    def test_obj_export_uses_one_based_indices(self) -> None:
        mesh = TriangleMesh(
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            triangles=((0, 1, 2),),
        )
        obj = mesh.to_obj(object_name="triangle")
        self.assertIn("o triangle\n", obj)
        self.assertIn("f 1 2 3\n", obj)


class TestStraightStairs(unittest.TestCase):
    @staticmethod
    def connector(*, descending: bool = False) -> ConnectorSpec:
        lower = ConnectorEndpoint("lower", "ground", (0.0, 0.0, 0.0))
        upper = ConnectorEndpoint("upper", "upper", (4.0, 0.0, 3.0))
        return ConnectorSpec(
            connector_id="main_stairs",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=upper if descending else lower,
            end=lower if descending else upper,
            width=1.2,
            parameters={"riser_count": 18},
        )

    def test_compiles_step_boxes_and_tread_surfaces(self) -> None:
        compiled = compile_straight_stairs(self.connector())

        self.assertEqual(len(compiled.visual_mesh.vertices), 18 * 8)
        self.assertEqual(len(compiled.visual_mesh.triangles), 18 * 12)
        self.assertEqual(len(compiled.surfaces), 18)
        self.assertEqual(len(compiled.collision_primitives), 18)
        self.assertEqual(compiled.visual_mesh.bounds[0], (0.0, -0.6, 0.0))
        self.assertEqual(compiled.visual_mesh.bounds[1], (4.0, 0.6, 3.0))
        self.assertAlmostEqual(compiled.surfaces[0].boundary[0][2], 3.0 / 18)
        self.assertAlmostEqual(compiled.surfaces[-1].boundary[0][2], 3.0)
        self.assertIn(SurfaceRole.TRAVERSABLE, compiled.surfaces[0].surface.roles)
        self.assertTrue(
            all(
                primitive.dimensions[0] >= 0.22
                for primitive in compiled.collision_primitives
            )
        )

    def test_descending_semantics_produce_same_physical_bounds(self) -> None:
        ascending = compile_straight_stairs(self.connector())
        descending = compile_straight_stairs(self.connector(descending=True))
        self.assertEqual(ascending.visual_mesh.bounds, descending.visual_mesh.bounds)

    def test_non_stair_input_is_rejected(self) -> None:
        connector = ConnectorSpec(
            connector_id="ramp",
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint("a", "a", (0, 0, 0)),
            end=ConnectorEndpoint("b", "b", (12, 0, 1)),
        )
        with self.assertRaises(UnsupportedGeometryError):
            compile_straight_stairs(connector)


class TestStraightRamp(unittest.TestCase):
    def test_compiles_sloped_support_surface(self) -> None:
        connector = ConnectorSpec(
            connector_id="accessible_ramp",
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint("lower", "ground", (1.0, 2.0, 0.0)),
            end=ConnectorEndpoint("upper", "upper", (13.0, 2.0, 1.0)),
            width=1.5,
        )

        compiled = compile_straight_ramp(connector)

        self.assertEqual(len(compiled.visual_mesh.vertices), 8)
        self.assertEqual(len(compiled.visual_mesh.triangles), 12)
        self.assertEqual(compiled.visual_mesh.bounds[0], (1.0, 1.25, -0.15))
        self.assertEqual(compiled.visual_mesh.bounds[1], (13.0, 2.75, 1.0))
        normal = compiled.surfaces[0].normal
        self.assertGreater(normal[2], 0.99)
        self.assertLess(normal[0], 0.0)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in normal)), 1.0)

    def test_dispatcher_rejects_unimplemented_connector(self) -> None:
        connector = ConnectorSpec(
            connector_id="elevator",
            connector_type=ConnectorType.ELEVATOR,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (0, 0, 3)),
            required_capabilities=frozenset({"walk"}),
        )
        with self.assertRaisesRegex(UnsupportedGeometryError, "elevator"):
            compile_connector(connector)

    def test_switchback_ramp_compiles_runs_and_horizontal_landing(self) -> None:
        connector = ConnectorSpec(
            connector_id="switchback",
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (0, 2, 2)),
            width=1.5,
            parameters={"waypoints": [(12, 0, 1), (12, 2, 1)]},
        )

        compiled = compile_multisegment_ramp(connector)

        self.assertEqual(len(compiled.surfaces), 3)
        self.assertEqual(len(compiled.collision_primitives), 0)
        self.assertEqual(
            sum(
                patch.surface.metadata.get("segment_type") == "landing"
                for patch in compiled.surfaces
            ),
            1,
        )
        self.assertTrue(
            all(patch.surface.source_id == "switchback" for patch in compiled.surfaces)
        )


class TestMultisegmentStairs(unittest.TestCase):
    def test_l_stair_has_two_flights_and_turn_landing(self) -> None:
        connector = ConnectorSpec(
            connector_id="l_stairs",
            connector_type=ConnectorType.STAIRS_L,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (3, 3, 3)),
            width=1.2,
            parameters={
                "waypoints": [(3, 0, 1.5)],
                "riser_counts": [9, 9],
                "landing_length": 1.2,
            },
        )

        compiled = compile_multisegment_stairs(connector)

        self.assertEqual(len(compiled.collision_primitives), 19)
        self.assertEqual(len(compiled.surfaces), 19)
        self.assertTrue(
            all(patch.surface.source_id == "l_stairs" for patch in compiled.surfaces)
        )
        self.assertTrue(
            any(
                patch.surface.metadata.get("segment_type") == "landing"
                for patch in compiled.surfaces
            )
        )

    def test_u_stair_has_horizontal_landing_and_opposed_flights(self) -> None:
        connector = ConnectorSpec(
            connector_id="u_stairs",
            connector_type=ConnectorType.STAIRS_U,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (0, 1.2, 3)),
            width=1.0,
            parameters={
                "waypoints": [(3, 0, 1.5), (3, 1.2, 1.5)],
                "riser_counts": [9, 9],
            },
        )

        compiled = compile_connector(connector)

        self.assertEqual(len(compiled.collision_primitives), 19)
        self.assertEqual(len(compiled.surfaces), 19)
        landing = [
            patch
            for patch in compiled.surfaces
            if patch.surface.metadata.get("segment_type") == "landing"
        ]
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0].normal, (0.0, 0.0, 1.0))


class TestSpiralStairsAndLadders(unittest.TestCase):
    def test_spiral_stair_compiles_segmented_treads(self) -> None:
        connector = ConnectorSpec(
            connector_id="tower_spiral",
            connector_type=ConnectorType.STAIRS_SPIRAL,
            start=ConnectorEndpoint("lower", "ground", (2, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (2, 0, 3)),
            width=1.0,
            parameters={
                "center": [0, 0],
                "turns": 1,
                "direction": "ccw",
                "riser_count": 18,
            },
        )

        compiled = compile_spiral_stairs(connector)

        self.assertEqual(len(compiled.surfaces), 36)
        self.assertEqual(len(compiled.visual_mesh.vertices), 18 * 8)
        self.assertEqual(len(compiled.collision_primitives), 0)
        self.assertEqual(
            sum(
                SurfaceRole.OVERHEAD in patch.surface.roles
                for patch in compiled.surfaces
            ),
            18,
        )
        self.assertTrue(
            all(
                SurfaceRole.TRAVERSABLE in patch.surface.roles
                for patch in compiled.surfaces
                if SurfaceRole.SUPPORT in patch.surface.roles
            )
        )
        self.assertEqual(compile_connector(connector), compiled)

    def test_spiral_endpoint_mismatch_is_rejected(self) -> None:
        connector = ConnectorSpec(
            connector_id="bad_spiral",
            connector_type=ConnectorType.STAIRS_SPIRAL,
            start=ConnectorEndpoint("lower", "ground", (2, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (0, 2, 3)),
            parameters={"center": [0, 0], "turns": 1, "riser_count": 18},
        )
        with self.assertRaisesRegex(UnsafeConnectorError, "end XY"):
            compile_spiral_stairs(connector)

    def test_vertical_ladder_compiles_climb_rungs(self) -> None:
        connector = ConnectorSpec.from_dict(
            {
                "id": "shaft_ladder",
                "type": "ladder",
                "start": {
                    "space_id": "pit",
                    "level_id": "ground",
                    "position": [0, 0, 0],
                },
                "end": {
                    "space_id": "ledge",
                    "level_id": "upper",
                    "position": [0, 0, 3],
                },
                "width": 0.6,
                "parameters": {"rung_count": 11, "yaw_degrees": 30},
            }
        )

        compiled = compile_ladder(connector)

        self.assertEqual(connector.required_capabilities, frozenset({"climb"}))
        self.assertEqual(len(compiled.surfaces), 11)
        self.assertEqual(len(compiled.visual_mesh.vertices), (2 + 11) * 8)
        self.assertEqual(compile_connector(connector), compiled)

    def test_invalid_u_stair_direction_is_rejected(self) -> None:
        connector = ConnectorSpec(
            connector_id="bad_u",
            connector_type=ConnectorType.STAIRS_U,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (6, 1, 3)),
            parameters={
                "waypoints": [(3, 0, 1.5), (3, 1, 1.5)],
                "riser_counts": [9, 9],
            },
        )

        with self.assertRaisesRegex(UnsafeConnectorError, "opposite"):
            compile_multisegment_stairs(connector)


class TestPolygonSpace(unittest.TestCase):
    @staticmethod
    def _xy_area(compiled, group_name: str) -> float:
        area = 0.0
        for triangle_index in compiled.triangle_groups[group_name]:
            indices = compiled.visual_mesh.triangles[triangle_index]
            a, b, c = (compiled.visual_mesh.vertices[index] for index in indices)
            area += (
                abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
            )
        return area

    def test_concave_room_floor_matches_footprint_area(self) -> None:
        footprint = Footprint2D(outer=((0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)))
        compiled = compile_polygon_space(
            structure_id="l_gallery", footprint=footprint, wall_height=3.0
        )

        self.assertAlmostEqual(self._xy_area(compiled, "floor_top"), 7.0)
        self.assertEqual(len(compiled.surfaces), 3 + 6)
        self.assertEqual(compiled.visual_mesh.bounds[0], (0.0, 0.0, -0.1))
        self.assertEqual(compiled.visual_mesh.bounds[1], (4.0, 4.0, 3.1))
        self.assertTrue(
            all(
                compiled.visual_mesh.triangle_normal(index)[2] > 0.99
                for index in compiled.triangle_groups["floor_top"]
            )
        )

    def test_courtyard_hole_remains_empty(self) -> None:
        footprint = Footprint2D(
            outer=((0, 0), (6, 0), (6, 6), (0, 6)),
            holes=(((2, 2), (2, 4), (4, 4), (4, 2)),),
        )
        compiled = compile_polygon_space(structure_id="courtyard", footprint=footprint)

        self.assertAlmostEqual(self._xy_area(compiled, "floor_top"), 32.0)
        for triangle_index in compiled.triangle_groups["floor_top"]:
            triangle = compiled.visual_mesh.triangles[triangle_index]
            vertices = [compiled.visual_mesh.vertices[index] for index in triangle]
            centroid = (
                sum(vertex[0] for vertex in vertices) / 3.0,
                sum(vertex[1] for vertex in vertices) / 3.0,
            )
            self.assertFalse(2.0 < centroid[0] < 4.0 and 2.0 < centroid[1] < 4.0)
        wall_surfaces = [
            patch
            for patch in compiled.surfaces
            if SurfaceRole.BOUNDARY in patch.surface.roles
        ]
        self.assertEqual(len(wall_surfaces), 8)
        self.assertEqual(wall_surfaces[-1].surface.metadata["loop_kind"], "hole_0")

    def test_floor_and_ceiling_openings_are_independent(self) -> None:
        boundary = Footprint2D.rectangle(6, 4)
        floor = Footprint2D(
            outer=boundary.outer,
            holes=(((2, 1), (2, 3), (4, 3), (4, 1)),),
        )
        compiled = compile_polygon_space(
            structure_id="stair_landing",
            footprint=boundary,
            floor_footprint=floor,
            ceiling_footprint=boundary,
        )

        floor_surface, ceiling_surface, floor_underside = compiled.surfaces[:3]
        self.assertEqual(len(floor_surface.surface.metadata["holes"]), 1)
        self.assertEqual(ceiling_surface.surface.metadata["holes"], [])
        self.assertEqual(len(floor_underside.surface.metadata["holes"]), 1)
        floor_area = self._xy_area(compiled, "floor_top")
        ceiling_area = self._xy_area(compiled, "ceiling_bottom")
        self.assertAlmostEqual(floor_area, 20.0)
        self.assertAlmostEqual(ceiling_area, 24.0)

    def test_slab_outer_loop_must_match_room_boundary(self) -> None:
        with self.assertRaisesRegex(GeometryValidationError, "outer loop"):
            compile_polygon_space(
                structure_id="bad_slab",
                footprint=Footprint2D.rectangle(6, 4),
                floor_footprint=Footprint2D.rectangle(5, 4),
            )

    def test_heightfield_room_can_omit_default_floor_slab(self) -> None:
        compiled = compile_polygon_space(
            structure_id="terrain_room",
            footprint=Footprint2D.rectangle(5, 4),
            include_floor=False,
        )

        self.assertNotIn("floor_top", compiled.triangle_groups)
        self.assertFalse(
            any(
                SurfaceRole.SUPPORT in patch.surface.roles
                for patch in compiled.surfaces
            )
        )
        self.assertTrue(
            any(
                patch.surface.surface_id == "terrain_room_ceiling"
                for patch in compiled.surfaces
            )
        )

    def test_sloped_floor_and_ceiling_have_analytic_normals(self) -> None:
        compiled = compile_polygon_space(
            structure_id="sloped",
            footprint=Footprint2D.rectangle(4.0, 3.0),
            floor_profile=ElevationProfile(
                profile_type=ElevationProfileType.SLOPED,
                gradient=(0.1, 0.0),
            ),
            ceiling_profile=ElevationProfile(
                profile_type=ElevationProfileType.SLOPED,
                base_elevation=3.0,
                gradient=(0.05, 0.0),
            ),
        )

        floor_surface = compiled.surfaces[0]
        ceiling_surface = compiled.surfaces[1]
        self.assertLess(floor_surface.normal[0], 0.0)
        self.assertGreater(floor_surface.normal[2], 0.99)
        self.assertGreater(ceiling_surface.normal[0], 0.0)
        self.assertLess(ceiling_surface.normal[2], -0.99)
        self.assertAlmostEqual(floor_surface.boundary[1][2], 0.4)
        self.assertAlmostEqual(ceiling_surface.boundary[1][2], 3.2)

    def test_ceiling_below_floor_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ceiling must be above floor"):
            compile_polygon_space(
                structure_id="invalid",
                footprint=Footprint2D.rectangle(2.0, 2.0),
                floor_profile=ElevationProfile(base_elevation=2.0),
                ceiling_profile=ElevationProfile(base_elevation=1.0),
            )

    def test_boundary_portal_cuts_visual_and_collision_wall(self) -> None:
        portal = PortalSpec(
            portal_id="door",
            portal_type=PortalType.DOOR,
            source_space_id="room",
            width=1.0,
            height=2.1,
            boundary_loop_index=0,
            boundary_edge_index=0,
            position_along=2.0,
        )
        compiled = compile_polygon_space(
            structure_id="room",
            footprint=Footprint2D.rectangle(4, 3),
            wall_height=3.0,
            portals=[portal],
        )

        for triangle_index in compiled.triangle_groups["walls"]:
            vertices = [
                compiled.visual_mesh.vertices[index]
                for index in compiled.visual_mesh.triangles[triangle_index]
            ]
            centroid = tuple(
                sum(vertex[axis] for vertex in vertices) / 3.0 for axis in range(3)
            )
            inside_door = (
                abs(centroid[1]) < 1e-9
                and 1.5 < centroid[0] < 2.5
                and 0.0 < centroid[2] < 2.1
            )
            self.assertFalse(inside_door)
        door_panels = [
            patch
            for patch in compiled.surfaces
            if patch.surface.metadata.get("adjacent_portal_id") == "door"
        ]
        self.assertEqual(len(door_panels), 1)

    def test_portal_outside_boundary_is_rejected(self) -> None:
        portal = PortalSpec(
            portal_id="bad",
            portal_type=PortalType.DOOR,
            source_space_id="room",
            width=2.0,
            boundary_loop_index=0,
            boundary_edge_index=0,
            position_along=0.5,
        )
        with self.assertRaisesRegex(GeometryValidationError, "exceeds edge"):
            compile_polygon_space(
                structure_id="room",
                footprint=Footprint2D.rectangle(4, 3),
                portals=[portal],
            )


class TestPlatforms(unittest.TestCase):
    def test_mezzanine_has_support_and_open_edge_semantics(self) -> None:
        compiled = compile_platform(
            PlatformSpec(
                platform_id="mezzanine",
                space_id="loft",
                footprint=Footprint2D.rectangle(4, 2),
                elevation=3.0,
                thickness=0.2,
                open_edge_indices=(2,),
            )
        )

        self.assertEqual(compiled.visual_mesh.bounds, ((0, 0, 2.8), (4, 2, 3.0)))
        self.assertIn(SurfaceRole.SUPPORT, compiled.surfaces[0].surface.roles)
        self.assertIn(SurfaceRole.TRAVERSABLE, compiled.surfaces[0].surface.roles)
        open_edges = [
            patch
            for patch in compiled.surfaces
            if SurfaceRole.OPEN_EDGE in patch.surface.roles
        ]
        self.assertEqual(len(open_edges), 1)
        self.assertEqual(open_edges[0].surface.metadata["edge_index"], 2)

    def test_nontraversable_display_platform_is_support_only(self) -> None:
        compiled = compile_platform(
            PlatformSpec(
                platform_id="plinth",
                space_id="gallery",
                footprint=Footprint2D.rectangle(1, 1),
                elevation=0.5,
                traversable=False,
            )
        )

        self.assertEqual(
            compiled.surfaces[0].surface.roles, frozenset({SurfaceRole.SUPPORT})
        )


class TestHeightfields(unittest.TestCase):
    def test_grid_compiles_to_queryable_sloped_triangles(self) -> None:
        compiled = compile_heightfield(
            HeightfieldSpec(
                heightfield_id="terrain",
                space_id="cavern",
                heights=((0, 0.1, 0.2), (0.2, 0.3, 0.4), (0.4, 0.5, 0.6)),
                cell_size=(1, 1),
                origin=(-1, -1, 2),
            )
        )

        self.assertEqual(len(compiled.visual_mesh.vertices), 9)
        self.assertEqual(len(compiled.visual_mesh.triangles), 8)
        self.assertEqual(len(compiled.surfaces), 8)
        self.assertEqual(compiled.visual_mesh.bounds, ((-1, -1, 2), (1, 1, 2.6)))
        self.assertEqual(len(compiled.triangle_groups["traversable"]), 8)
        self.assertTrue(
            all(
                SurfaceRole.SUPPORT in patch.surface.roles
                for patch in compiled.surfaces
            )
        )

    def test_heightfield_budget_is_enforced_before_compilation(self) -> None:
        heightfield = HeightfieldSpec(
            heightfield_id="terrain",
            space_id="cavern",
            heights=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        )
        with self.assertRaisesRegex(GeometryValidationError, "budget"):
            compile_heightfield(heightfield, max_triangles=7)


class TestCompiledStructureExport(unittest.TestCase):
    def test_stairs_write_obj_sdf_and_surface_sidecar(self) -> None:
        connector = TestStraightStairs.connector()
        compiled = compile_straight_stairs(connector)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, temporary_directory)

            self.assertTrue(paths.mesh_path.exists())
            self.assertTrue(paths.sdf_path.exists())
            self.assertTrue(paths.surfaces_path.exists())
            self.assertIn("f 1 3 2", paths.mesh_path.read_text(encoding="utf-8"))

            sdf = ET.parse(paths.sdf_path)
            collisions = sdf.findall(".//collision")
            self.assertEqual(len(collisions), 18)
            self.assertTrue(
                all(node.find("geometry/box") is not None for node in collisions)
            )
            self.assertEqual(
                sdf.findtext(".//visual/geometry/mesh/uri"), paths.mesh_path.name
            )

            sidecar = json.loads(paths.surfaces_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["structure_id"], "main_stairs")
            self.assertEqual(len(sidecar["surfaces"]), 18)
            self.assertEqual(sidecar["bounds"], [[0.0, -0.6, 0.0], [4.0, 0.6, 3.0]])

    def test_ramp_uses_closed_mesh_collision(self) -> None:
        connector = ConnectorSpec(
            connector_id="ramp",
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint("a", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("b", "upper", (12, 0, 1)),
        )
        compiled = compile_straight_ramp(connector)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, Path(temporary_directory))
            sdf = ET.parse(paths.sdf_path)
            collision_uri = sdf.findtext(".//collision/geometry/mesh/uri")
            self.assertEqual(collision_uri, "ramp.obj")

    def test_writer_can_preserve_room_sdf_link_contract(self) -> None:
        compiled = compile_polygon_space(
            structure_id="polygon_room",
            footprint=Footprint2D.rectangle(4, 3),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(
                compiled,
                temporary_directory,
                model_name="room_geometry",
                link_name="room_geometry_body_link",
            )
            root = ET.parse(paths.sdf_path).getroot()

            self.assertEqual(root.find(".//model").attrib["name"], "room_geometry")
            self.assertIsNotNone(root.find(".//link[@name='room_geometry_body_link']"))


class TestFreeformStructuralMesh(unittest.TestCase):
    @staticmethod
    def chamber_mesh(*, duplicate_face: bool = False) -> TriangleMesh:
        triangles = (
            (0, 1, 2),
            (0, 2, 3),  # floor, +Z
            (4, 6, 5),
            (4, 7, 6),  # ceiling, -Z
            (0, 5, 1),
            (0, 4, 5),
            (1, 6, 2),
            (1, 5, 6),
            (2, 7, 3),
            (2, 6, 7),
            (3, 4, 0),
            (3, 7, 4),
        )
        if duplicate_face:
            triangles += ((0, 1, 2),)
        return TriangleMesh(
            vertices=(
                (0, 0, 0),
                (200, 0, 0),
                (200, 200, 0),
                (0, 200, 0),
                (0, 0, 200),
                (200, 0, 200),
                (200, 200, 200),
                (0, 200, 200),
            ),
            triangles=triangles,
        )

    def test_cavern_import_applies_units_transform_and_auto_classifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory) / "chamber.obj"
            mesh_path.write_text(self.chamber_mesh().to_obj(), encoding="utf-8")
            compiled = compile_structural_mesh(
                StructuralMeshSpec(
                    mesh_id="cavern",
                    space_id="chamber",
                    mesh_path=str(mesh_path),
                    unit_scale=0.01,
                    transform=Transform3D(translation=(10, 20, -2)),
                    require_watertight=True,
                    normal_orientation="interior",
                )
            )

            self.assertEqual(compiled.visual_mesh.bounds, ((10, 20, -2), (12, 22, 0)))
            support = [
                surface
                for surface in compiled.surfaces
                if SurfaceRole.SUPPORT in surface.surface.roles
            ]
            overhead = [
                surface
                for surface in compiled.surfaces
                if SurfaceRole.OVERHEAD in surface.surface.roles
            ]
            self.assertEqual(len(support), 2)
            self.assertEqual(len(overhead), 2)
            self.assertTrue(
                all(
                    surface.surface.metadata["space_id"] == "chamber"
                    for surface in support
                )
            )

    def test_authored_annotation_overrides_normal_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory) / "chamber.obj"
            mesh_path.write_text(self.chamber_mesh().to_obj(), encoding="utf-8")
            compiled = compile_structural_mesh(
                StructuralMeshSpec(
                    mesh_id="cavern",
                    space_id="chamber",
                    mesh_path=str(mesh_path),
                    unit_scale=1.0,
                    annotations=(
                        MeshSurfaceAnnotation(
                            annotation_id="hazard",
                            triangle_indices=(0,),
                            roles=frozenset({SurfaceRole.NON_INTERACTIVE}),
                        ),
                    ),
                )
            )

            self.assertEqual(
                compiled.surfaces[0].surface.roles,
                frozenset({SurfaceRole.NON_INTERACTIVE}),
            )
            self.assertFalse(compiled.surfaces[0].surface.metadata["auto_classified"])

    def test_duplicate_faces_require_explicit_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory) / "duplicate.obj"
            mesh_path.write_text(
                self.chamber_mesh(duplicate_face=True).to_obj(), encoding="utf-8"
            )
            spec = StructuralMeshSpec(
                mesh_id="cavern",
                space_id="chamber",
                mesh_path=str(mesh_path),
                unit_scale=1.0,
            )

            with self.assertRaisesRegex(GeometryValidationError, "duplicate"):
                compile_structural_mesh(spec)
            repaired = compile_structural_mesh(spec, repair=True)
            self.assertEqual(len(repaired.visual_mesh.triangles), 12)


if __name__ == "__main__":
    unittest.main()
