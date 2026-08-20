"""Tests for surface-native structural placement queries."""

import math
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scenesmith.ceiling_agents.tools.ceiling_tools import CeilingTools
from scenesmith.agent_utils.structural_compiler import (
    CompiledStructure,
    compile_polygon_space,
    write_compiled_structure,
)
from scenesmith.agent_utils.structural_geometry import (
    ElevationProfile,
    ElevationProfileType,
    Footprint2D,
    SurfaceRole,
    Transform3D,
)
from scenesmith.agent_utils.structural_surfaces import (
    StructuralSurfaceIndex,
    load_surface_patches,
    transform_surface_patches,
)


class TestStructuralSurfaceIndex(unittest.TestCase):
    def test_sloped_surface_returns_correct_height_normal_and_frame(self) -> None:
        compiled = compile_polygon_space(
            structure_id="slope",
            footprint=Footprint2D.rectangle(4, 3).centered_on_bounds(),
            floor_profile=ElevationProfile(
                profile_type=ElevationProfileType.SLOPED,
                gradient=(0.1, 0.0),
            ),
        )
        index = StructuralSurfaceIndex(compiled.surfaces)

        pose = index.support_pose(1.0, 0.0, reference_z=2.0, yaw=math.pi / 2)

        assert pose is not None
        self.assertAlmostEqual(pose.position[2], 0.1)
        self.assertLess(pose.normal[0], 0.0)
        self.assertAlmostEqual(sum(value * value for value in pose.normal), 1.0)
        self.assertAlmostEqual(
            sum(a * b for a, b in zip(pose.tangent_x, pose.normal)), 0.0
        )
        self.assertGreater(pose.clearance_to_edge, 0.0)

    def test_hole_has_no_support(self) -> None:
        compiled = compile_polygon_space(
            structure_id="courtyard",
            footprint=Footprint2D(
                outer=((0, 0), (6, 0), (6, 6), (0, 6)),
                holes=(((2, 2), (2, 4), (4, 4), (4, 2)),),
            ),
        )
        index = StructuralSurfaceIndex(compiled.surfaces)

        self.assertIsNone(index.support_pose(3, 3))
        self.assertIsNotNone(index.support_pose(1, 1))

    def test_reference_height_selects_correct_stacked_surface(self) -> None:
        lower = compile_polygon_space(
            structure_id="lower",
            footprint=Footprint2D.rectangle(4, 4),
        )
        upper = compile_polygon_space(
            structure_id="upper",
            footprint=Footprint2D.rectangle(4, 4),
            floor_profile=ElevationProfile(base_elevation=3.0),
        )
        index = StructuralSurfaceIndex((*lower.surfaces, *upper.surfaces))

        lower_pose = index.support_pose(1, 1, reference_z=2.5)
        upper_pose = index.support_pose(1, 1, reference_z=4.0)

        assert lower_pose is not None and upper_pose is not None
        self.assertEqual(lower_pose.position[2], 0.0)
        self.assertEqual(upper_pose.position[2], 3.0)

    def test_sidecar_round_trip_preserves_queryable_patches(self) -> None:
        compiled = compile_polygon_space(
            structure_id="room",
            footprint=Footprint2D.rectangle(3, 2),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, Path(temporary_directory))
            restored = load_surface_patches(paths.surfaces_path)
            index = StructuralSurfaceIndex(restored)

            self.assertEqual(len(restored), len(compiled.surfaces))
            self.assertEqual(len(index.by_role(SurfaceRole.BOUNDARY)), 4)
            self.assertIsNotNone(index.support_pose(1, 1))

    def test_sloped_overhead_returns_lowest_mount_pose(self) -> None:
        compiled = compile_polygon_space(
            structure_id="vault_bay",
            footprint=Footprint2D.rectangle(4, 3).centered_on_bounds(),
            ceiling_profile=ElevationProfile(
                profile_type=ElevationProfileType.SLOPED,
                base_elevation=3.0,
                gradient=(0.1, 0.0),
            ),
        )
        index = StructuralSurfaceIndex(compiled.surfaces)

        pose = index.overhead_pose(1.0, 0.0, reference_z=0.0)

        assert pose is not None
        self.assertAlmostEqual(pose.position[2], 3.1)
        self.assertLess(pose.normal[2], -0.99)
        self.assertAlmostEqual(
            sum(a * b for a, b in zip(pose.tangent_x, pose.normal)), 0.0
        )

    def test_ceiling_tool_excludes_the_floor_underside_from_mount_candidates(self) -> None:
        compiled = compile_polygon_space(
            structure_id="fixture_room",
            footprint=Footprint2D.rectangle(4, 3).centered_on_bounds(),
            wall_height=3.2,
        )
        index = StructuralSurfaceIndex(compiled.surfaces)
        floor = SimpleNamespace(
            compute_world_bounds=lambda: (
                np.array([-2.0, -1.5, -0.1]),
                np.array([2.0, 1.5, 0.0]),
            )
        )
        tools = CeilingTools.__new__(CeilingTools)
        tools.scene = SimpleNamespace(
            room_geometry=SimpleNamespace(floor=floor)
        )
        tools.ceiling_height = 3.2
        tools._get_structural_surface_index = lambda: index

        transform = tools._ceiling_transform(0.0, 0.0, 0.0)

        assert transform is not None
        self.assertAlmostEqual(transform.translation()[2], 3.2)

    def test_agent_clearance_detects_low_overhead_and_narrow_edge(self) -> None:
        compiled = compile_polygon_space(
            structure_id="low_tunnel",
            footprint=Footprint2D.rectangle(2, 4).centered_on_bounds(),
            wall_height=1.7,
        )
        index = StructuralSurfaceIndex(compiled.surfaces)

        tall = index.clearance_at(0, 0, agent_height=1.8, agent_radius=0.3)
        wide = index.clearance_at(0.85, 0, agent_height=1.6, agent_radius=0.3)
        compact = index.clearance_at(0, 0, agent_height=1.6, agent_radius=0.3)

        self.assertFalse(tall.fits)
        self.assertIn("insufficient_headroom", tall.reasons)
        self.assertFalse(wide.fits)
        self.assertIn("insufficient_edge_clearance", wide.reasons)
        self.assertTrue(compact.fits)
        self.assertAlmostEqual(compact.vertical_clearance or 0.0, 1.7)

    def test_patch_transform_preserves_query_in_parent_frame(self) -> None:
        compiled = compile_polygon_space(
            structure_id="rotated",
            footprint=Footprint2D.rectangle(4, 2).centered_on_bounds(),
        )
        patches = transform_surface_patches(
            compiled.surfaces,
            Transform3D(translation=(10, 20, 3), rotation_rpy=(0, 0, math.pi / 2)),
        )
        index = StructuralSurfaceIndex(patches)

        support = index.support_pose(10, 21, reference_z=4)
        overhead = index.overhead_pose(10, 21, reference_z=3)

        assert support is not None and overhead is not None
        self.assertAlmostEqual(support.position[2], 3)
        self.assertAlmostEqual(overhead.position[2], 5.5)


if __name__ == "__main__":
    unittest.main()
