"""Dependency-light tests for the v2 structural geometry model."""

import math
import unittest

from scenesmith.agent_utils.structural_geometry import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    GeometryValidationError,
    HeightfieldSpec,
    InvalidFootprintError,
    InvalidTransformError,
    LevelSpec,
    MeshSurfaceAnnotation,
    PlatformSpec,
    PortalSpec,
    PortalType,
    StructuralMeshSpec,
    StructuralSurface,
    SurfaceRole,
    Transform3D,
    UnknownConnectorEndpointError,
    UnknownLevelError,
    UnsafeConnectorError,
    validate_structural_references,
)


class TestFootprint2D(unittest.TestCase):
    def test_rectangle_has_expected_area_bounds_and_membership(self) -> None:
        footprint = Footprint2D.rectangle(5.0, 4.0)

        self.assertEqual(footprint.area, 20.0)
        self.assertEqual(footprint.bounds, (0.0, 0.0, 5.0, 4.0))
        self.assertTrue(footprint.contains((2.5, 2.0)))
        self.assertTrue(footprint.contains((0.0, 2.0)))
        self.assertFalse(footprint.contains((5.1, 2.0)))

    def test_circle_tessellation_respects_chord_error(self) -> None:
        radius = 5.0
        tolerance = 0.01
        footprint = Footprint2D.circle(radius, tolerance, center=(2.0, -1.0))
        half_angle = math.pi / len(footprint.outer)
        chord_error = radius * (1.0 - math.cos(half_angle))

        self.assertLessEqual(chord_error, tolerance + 1e-12)
        self.assertTrue(footprint.contains((2.0, -1.0)))
        self.assertFalse(footprint.contains((7.1, -1.0)))
        self.assertEqual(
            Footprint2D.circle(radius, tolerance, center=(2.0, -1.0)), footprint
        )

    def test_concave_footprint_preserves_void(self) -> None:
        footprint = Footprint2D(outer=((0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)))

        self.assertEqual(footprint.area, 7.0)
        self.assertTrue(footprint.contains((0.5, 3.5)))
        self.assertFalse(footprint.contains((2.0, 2.0)))

    def test_hole_excludes_support_area_and_round_trips(self) -> None:
        footprint = Footprint2D(
            outer=((0, 0), (6, 0), (6, 6), (0, 6)),
            holes=(((2, 2), (2, 4), (4, 4), (4, 2)),),
        )

        self.assertEqual(footprint.area, 32.0)
        self.assertTrue(footprint.contains((1.0, 1.0)))
        self.assertFalse(footprint.contains((3.0, 3.0)))
        self.assertEqual(Footprint2D.from_dict(footprint.to_dict()), footprint)
        self.assertGreater(footprint.area, 0)
        self.assertGreater(
            sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(
                    footprint.outer, footprint.outer[1:] + footprint.outer[:1]
                )
            ),
            0,
        )

    def test_repeated_closing_vertex_is_normalized(self) -> None:
        footprint = Footprint2D(outer=((0, 0), (2, 0), (2, 2), (0, 2), (0, 0)))
        self.assertEqual(len(footprint.outer), 4)

    def test_self_intersecting_footprint_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidFootprintError, "self-intersects"):
            Footprint2D(outer=((0, 0), (2, 2), (0, 2), (2, 0)))

    def test_zero_length_edge_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidFootprintError, "zero length"):
            Footprint2D(outer=((0, 0), (2, 0), (2, 0), (0, 2)))

    def test_hole_outside_outer_loop_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidFootprintError, "strictly inside"):
            Footprint2D(
                outer=((0, 0), (4, 0), (4, 4), (0, 4)),
                holes=(((5, 5), (6, 5), (6, 6), (5, 6)),),
            )

    def test_overlapping_holes_are_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidFootprintError, "overlap"):
            Footprint2D(
                outer=((0, 0), (10, 0), (10, 10), (0, 10)),
                holes=(
                    ((1, 1), (1, 5), (5, 5), (5, 1)),
                    ((4, 4), (4, 8), (8, 8), (8, 4)),
                ),
            )

    def test_nonfinite_coordinate_is_rejected(self) -> None:
        with self.assertRaises(InvalidTransformError):
            Footprint2D(outer=((0, 0), (math.nan, 0), (0, 2)))


class TestProfilesAndSurfaces(unittest.TestCase):
    def test_sloped_profile_evaluates_height(self) -> None:
        profile = ElevationProfile(
            profile_type=ElevationProfileType.SLOPED,
            base_elevation=2.0,
            gradient=(0.1, -0.05),
        )
        self.assertAlmostEqual(profile.height_at((4.0, 2.0)), 2.3)
        self.assertEqual(ElevationProfile.from_dict(profile.to_dict()), profile)

    def test_planar_profile_rejects_nonzero_gradient(self) -> None:
        with self.assertRaisesRegex(GeometryValidationError, "use 'sloped'"):
            ElevationProfile(gradient=(0.1, 0.0))

    def test_nonanalytic_profile_query_is_explicitly_unsupported(self) -> None:
        profile = ElevationProfile(
            profile_type=ElevationProfileType.HEIGHTFIELD,
            parameters={"asset": "terrain.npy"},
        )
        with self.assertRaisesRegex(GeometryValidationError, "not implemented"):
            profile.height_at((0.0, 0.0))

    def test_surface_round_trip(self) -> None:
        surface = StructuralSurface(
            surface_id="mezzanine_top",
            roles=frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
            source_id="mezzanine",
            transform=Transform3D(translation=(0.0, 0.0, 3.0)),
            geometry_ref="meshes/mezzanine.glb",
            metadata={"material": "wood"},
        )
        self.assertEqual(StructuralSurface.from_dict(surface.to_dict()), surface)

    def test_boundary_portal_round_trip(self) -> None:
        portal = PortalSpec(
            portal_id="arched_door",
            portal_type=PortalType.ARCH,
            source_space_id="gallery",
            target_space_id="hall",
            width=1.5,
            height=2.4,
            boundary_loop_index=0,
            boundary_edge_index=3,
            position_along=2.0,
            sill_height=0.1,
        )
        self.assertEqual(PortalSpec.from_dict(portal.to_dict()), portal)

    def test_structural_mesh_requires_explicit_units_and_round_trips(self) -> None:
        annotation = MeshSurfaceAnnotation(
            annotation_id="floor",
            triangle_indices=(0, 1),
            roles=frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
        )
        mesh = StructuralMeshSpec(
            mesh_id="cavern",
            space_id="chamber",
            mesh_path="chamber.obj",
            unit_scale=0.01,
            transform=Transform3D(translation=(1, 2, 3)),
            annotations=(annotation,),
            replaces_room_shell=True,
        )

        self.assertEqual(StructuralMeshSpec.from_dict(mesh.to_dict()), mesh)
        self.assertTrue(mesh.to_dict()["replaces_room_shell"])
        with self.assertRaisesRegex(GeometryValidationError, "explicit unit_scale"):
            StructuralMeshSpec.from_dict(
                {
                    "id": "cavern",
                    "space_id": "chamber",
                    "mesh_path": "chamber.obj",
                }
            )

    def test_platform_round_trip_and_open_edge_validation(self) -> None:
        platform = PlatformSpec(
            platform_id="bridge",
            space_id="atrium",
            footprint=Footprint2D.rectangle(6, 1.5),
            elevation=3.0,
            open_edge_indices=(1, 3),
        )
        self.assertEqual(PlatformSpec.from_dict(platform.to_dict()), platform)
        with self.assertRaisesRegex(GeometryValidationError, "open_edge_indices"):
            PlatformSpec(
                platform_id="bad",
                space_id="atrium",
                footprint=Footprint2D.rectangle(1, 1),
                elevation=1,
                open_edge_indices=(4,),
            )

    def test_heightfield_round_trip_and_rectangular_validation(self) -> None:
        heightfield = HeightfieldSpec(
            heightfield_id="cave_floor",
            space_id="cavern",
            heights=((0, 0.1, 0.2), (0.2, 0.3, 0.4)),
            cell_size=(0.5, 0.75),
            origin=(1, 2, -1),
            replaces_floor=True,
        )
        self.assertEqual(heightfield.shape, (2, 3))
        self.assertEqual(HeightfieldSpec.from_dict(heightfield.to_dict()), heightfield)
        self.assertTrue(heightfield.to_dict()["replaces_floor"])
        with self.assertRaisesRegex(GeometryValidationError, "equal length"):
            HeightfieldSpec(
                heightfield_id="bad",
                space_id="cavern",
                heights=((0, 1), (0, 1, 2)),
            )

    def test_heightfield_queries_match_compiler_triangle_split(self) -> None:
        heightfield = HeightfieldSpec(
            heightfield_id="warped",
            space_id="cavern",
            heights=((0, 1), (2, 4)),
            cell_size=(2, 1),
            origin=(10, 20, 3),
        )

        self.assertAlmostEqual(heightfield.height_at(11.5, 20.25), 4.5)
        self.assertAlmostEqual(heightfield.height_at(10.5, 20.75), 5.0)
        normal = heightfield.normal_at(11.5, 20.25)
        self.assertAlmostEqual(sum(value * value for value in normal), 1.0)
        self.assertGreater(normal[2], 0.0)
        with self.assertRaisesRegex(GeometryValidationError, "outside"):
            heightfield.height_at(9, 20)

    def test_transform_rejects_infinite_rotation(self) -> None:
        with self.assertRaises(InvalidTransformError):
            Transform3D(rotation_rpy=(0.0, 0.0, math.inf))


class TestConnectorsAndReferences(unittest.TestCase):
    @staticmethod
    def _stairs() -> ConnectorSpec:
        return ConnectorSpec(
            connector_id="stairs_1",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint("lower", "ground", (0.0, 0.0, 0.0)),
            end=ConnectorEndpoint("upper", "upper", (4.0, 0.0, 3.0)),
            width=1.1,
            parameters={"riser_count": 18},
        )

    def test_level_and_connector_round_trip(self) -> None:
        level = LevelSpec("upper", elevation=3.0, nominal_height=2.8)
        stairs = self._stairs()

        self.assertEqual(LevelSpec.from_dict(level.to_dict()), level)
        self.assertEqual(ConnectorSpec.from_dict(stairs.to_dict()), stairs)
        stairs.validate_straight_access()

    def test_straight_stair_rejects_unsafe_risers(self) -> None:
        stairs = ConnectorSpec(
            connector_id="unsafe_stairs",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint("lower", "ground", (0.0, 0.0, 0.0)),
            end=ConnectorEndpoint("upper", "upper", (2.0, 0.0, 3.0)),
            parameters={"riser_count": 8},
        )
        with self.assertRaisesRegex(UnsafeConnectorError, "riser height"):
            stairs.validate_straight_access()

    def test_ramp_rejects_excessive_slope(self) -> None:
        ramp = ConnectorSpec(
            connector_id="steep_ramp",
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint("lower", "ground", (0.0, 0.0, 0.0)),
            end=ConnectorEndpoint("upper", "upper", (3.0, 0.0, 1.0)),
        )
        with self.assertRaisesRegex(UnsafeConnectorError, "ramp slope"):
            ramp.validate_straight_access()

    def test_connector_defaults_are_capability_appropriate(self) -> None:
        endpoints = (
            ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            ConnectorEndpoint("upper", "upper", (0, 0, 3)),
        )

        ladder = ConnectorSpec("ladder", ConnectorType.LADDER, *endpoints)
        shaft = ConnectorSpec("shaft", ConnectorType.SHAFT, *endpoints)
        passage = ConnectorSpec(
            "passage",
            ConnectorType.NATURAL_PASSAGE,
            *endpoints,
            parameters={"geometry_embedded": True},
        )

        self.assertEqual(ladder.required_capabilities, frozenset({"climb"}))
        self.assertEqual(shaft.required_capabilities, frozenset({"climb"}))
        self.assertEqual(passage.required_capabilities, frozenset({"walk"}))
        self.assertEqual(ConnectorSpec.from_dict(shaft.to_dict()), shaft)

    def test_embedded_geometry_contract_rejects_ambiguous_parameters(self) -> None:
        start = ConnectorEndpoint("lower", "ground", (0, 0, 0))
        end = ConnectorEndpoint("upper", "upper", (0, 0, 3))

        with self.assertRaisesRegex(UnsafeConnectorError, "must be a boolean"):
            ConnectorSpec(
                "passage",
                ConnectorType.NATURAL_PASSAGE,
                start,
                end,
                parameters={"geometry_embedded": "yes"},
            )
        with self.assertRaisesRegex(UnsafeConnectorError, "only valid"):
            ConnectorSpec(
                "stairs",
                ConnectorType.STAIRS_STRAIGHT,
                start,
                end,
                parameters={"geometry_embedded": True, "riser_count": 18},
            )
        with self.assertRaisesRegex(InvalidTransformError, r"waypoints\[0\]"):
            ConnectorSpec(
                "passage",
                ConnectorType.NATURAL_PASSAGE,
                start,
                end,
                parameters={"geometry_embedded": True, "waypoints": [(1, 2)]},
            )

    def test_valid_reference_graph(self) -> None:
        validate_structural_references(
            levels=(LevelSpec("ground", 0.0), LevelSpec("upper", 3.0)),
            space_level_ids={"lower": "ground", "upper": "upper"},
            connectors=(self._stairs(),),
            portals=(
                PortalSpec(
                    portal_id="outside_door",
                    portal_type=PortalType.DOOR,
                    source_space_id="lower",
                ),
            ),
        )

    def test_unknown_space_reference_is_rejected(self) -> None:
        stairs = self._stairs()
        with self.assertRaises(UnknownConnectorEndpointError):
            validate_structural_references(
                levels=(LevelSpec("ground"), LevelSpec("upper", 3.0)),
                space_level_ids={"lower": "ground"},
                connectors=(stairs,),
            )

    def test_unknown_level_reference_is_rejected(self) -> None:
        with self.assertRaises(UnknownLevelError):
            validate_structural_references(
                levels=(LevelSpec("ground"),),
                space_level_ids={"upper": "missing"},
            )

    def test_endpoint_level_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            GeometryValidationError, "connector_level_mismatch"
        ):
            validate_structural_references(
                levels=(LevelSpec("ground"), LevelSpec("upper", 3.0)),
                space_level_ids={"lower": "upper", "upper": "upper"},
                connectors=(self._stairs(),),
            )


if __name__ == "__main__":
    unittest.main()
