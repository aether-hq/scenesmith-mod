"""Geometry proofs for the semantic natural-environment compiler."""

import ast
import json
import tempfile
import unittest

from pathlib import Path

import numpy as np
import trimesh

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.semantic_environment_compiler import (
    SemanticCompileOptions,
    compile_semantic_environment,
    semantic_mesh_annotations,
)
from scenesmith.agent_utils.semantic_environment_details import (
    compile_environment_details,
)
from scenesmith.agent_utils.semantic_environments import (
    Bounds3D,
    CavernChamberSpec,
    CavernShape,
    DetailCollisionPolicy,
    DetailFieldSpec,
    DetailSurfaceRole,
    EnvironmentKind,
    EnvironmentOpeningSpec,
    EnvironmentRegionSpec,
    FormationType,
    HeroFeatureSpec,
    HeroFeatureType,
    OpeningTarget,
    PassageCrossSectionSpec,
    PassageJunctionSpec,
    PassageNetworkSpec,
    PassageProfile,
    PassageSegmentSpec,
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structural_geometry import SurfaceRole
from scenesmith.agent_utils.structural_surfaces import load_surface_patches


def _cross_sections(width: float = 4.0, height: float = 4.0):
    return (
        PassageCrossSectionSpec(0.0, width, height),
        PassageCrossSectionSpec(1.0, width * 1.25, height * 1.2),
    )


def _environment(*, translated=(0.0, 0.0, 0.0), reordered=False):
    tx, ty, tz = translated

    def point(x, y, z=0.0):
        return (x + tx, y + ty, z + tz)

    junctions = (
        PassageJunctionSpec("root", point(0, 0)),
        PassageJunctionSpec("fork", point(6, 0)),
        PassageJunctionSpec("north", point(12, 5)),
        PassageJunctionSpec("south", point(12, -5)),
    )
    segments = (
        PassageSegmentSpec(
            "trunk", "root", "fork", (point(0, 0), point(6, 0)), _cross_sections()
        ),
        PassageSegmentSpec(
            "north_arm",
            "fork",
            "north",
            (point(6, 0), point(12, 5)),
            _cross_sections(3.5, 3.0),
        ),
        PassageSegmentSpec(
            "south_arm",
            "fork",
            "south",
            (point(6, 0), point(12, -5)),
            _cross_sections(3.0, 3.5),
        ),
    )
    if reordered:
        junctions = tuple(reversed(junctions))
        segments = tuple(reversed(segments))
    region = EnvironmentRegionSpec(
        "subsurface",
        EnvironmentKind.SUBTERRANEAN,
        Bounds3D(point(-10, -15, -8), point(25, 15, 15)),
    )
    return SemanticEnvironmentSpec(
        regions=(region,),
        passage_networks=(
            PassageNetworkSpec("branching_routes", "subsurface", junctions, segments),
        ),
    )


def _mesh_component_count(mesh: trimesh.Trimesh) -> int:
    """Count vertex-connected components without an optional graph backend."""

    parents = list(range(len(mesh.vertices)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    used: set[int] = set()
    for first, second, third in mesh.faces:
        first, second, third = int(first), int(second), int(third)
        used.update((first, second, third))
        union(first, second)
        union(first, third)
    return len({find(index) for index in used})


class TestSemanticEnvironmentCompiler(unittest.TestCase):
    options = SemanticCompileOptions(voxel_size=1.0, max_cells=100_000)

    def test_cav_001_variable_profile_passage_has_surface_roles(self) -> None:
        environment = _environment()
        compiled = compile_semantic_environment(environment, options=self.options)
        roles = {role for patch in compiled.surfaces for role in patch.surface.roles}

        self.assertGreater(len(compiled.visual_mesh.triangles), 100)
        self.assertIn(SurfaceRole.TRAVERSABLE, roles)
        self.assertIn(SurfaceRole.OVERHEAD, roles)
        self.assertIn(SurfaceRole.BOUNDARY, roles)
        self.assertEqual(compiled.visual_mesh, compiled.collision_mesh)

    def test_cav_002_y_branch_is_one_watertight_physical_void(self) -> None:
        compiled = compile_semantic_environment(_environment(), options=self.options)
        mesh = trimesh.Trimesh(
            vertices=compiled.visual_mesh.vertices,
            faces=compiled.visual_mesh.triangles,
            process=False,
        )

        self.assertTrue(mesh.is_watertight)
        self.assertEqual(_mesh_component_count(mesh), 1)
        self.assertLess(mesh.volume, 0.0)  # Winding faces the navigable void.

    def test_cav_003_chamber_and_passage_compile_as_one_joined_shell(self) -> None:
        base = _environment()
        chamber = CavernChamberSpec(
            "great_hall", "subsurface", (14, 5, 3), (14, 12, 10)
        )
        environment = SemanticEnvironmentSpec(
            regions=base.regions,
            chambers=(chamber,),
            passage_networks=base.passage_networks,
        )
        compiled = compile_semantic_environment(environment, options=self.options)
        mesh = trimesh.Trimesh(
            vertices=compiled.visual_mesh.vertices,
            faces=compiled.visual_mesh.triangles,
            process=False,
        )

        self.assertEqual(_mesh_component_count(mesh), 1)
        source_ids = {
            patch.surface.metadata["semantic_source_id"] for patch in compiled.surfaces
        }
        self.assertIn("great_hall", source_ids)
        self.assertIn("north_arm", source_ids)

    def test_cav_004_all_typed_passage_profiles_compile(self) -> None:
        for profile in PassageProfile:
            with self.subTest(profile=profile.value):
                data = _environment().to_dict()
                for segment in data["passage_networks"][0]["segments"]:
                    segment["profile"] = profile.value
                compiled = compile_semantic_environment(
                    SemanticEnvironmentSpec.from_dict(data), options=self.options
                )
                self.assertGreater(len(compiled.visual_mesh.triangles), 100)

    def test_cav_005_superellipsoid_chamber_compiles(self) -> None:
        environment = SemanticEnvironmentSpec(
            regions=_environment().regions,
            chambers=(
                CavernChamberSpec(
                    "angular_chamber",
                    "subsurface",
                    (5, 0, 2),
                    (16, 12, 10),
                    shape="superellipsoid",
                    orientation_rpy=(0.1, 0.2, 0.3),
                ),
            ),
        )

        compiled = compile_semantic_environment(environment, options=self.options)
        roles = {role for patch in compiled.surfaces for role in patch.surface.roles}
        self.assertIn(SurfaceRole.TRAVERSABLE, roles)
        self.assertIn(SurfaceRole.OVERHEAD, roles)

    def test_cav_006_sky_opening_is_a_real_non_watertight_aperture(self) -> None:
        base = _environment()
        chamber = CavernChamberSpec(
            "sky_chamber", "subsurface", (5, 0, 3), (16, 12, 10)
        )
        environment = SemanticEnvironmentSpec(
            regions=base.regions,
            chambers=(chamber,),
            openings=(
                EnvironmentOpeningSpec(
                    "oculus",
                    "subsurface",
                    "sky_chamber",
                    OpeningTarget.SKY,
                    (5, 0, 7.5),
                    (0, 0, 1),
                    (4, 4),
                    8,
                    weather_exposed=True,
                ),
            ),
        )

        compiled = compile_semantic_environment(environment, options=self.options)
        mesh = trimesh.Trimesh(
            vertices=compiled.visual_mesh.vertices,
            faces=compiled.visual_mesh.triangles,
            process=False,
        )
        opening_patches = [
            patch
            for patch in compiled.surfaces
            if patch.surface.metadata["semantic_source_id"] == "oculus"
        ]

        self.assertFalse(mesh.is_watertight)
        self.assertTrue(opening_patches)
        self.assertTrue(
            all(patch.surface.metadata["sky_exposed"] for patch in opening_patches)
        )
        self.assertTrue(
            all(patch.surface.metadata["weather_exposed"] for patch in opening_patches)
        )

    def test_cav_010_colossal_cavern_is_compact_and_fully_compilable(self) -> None:
        """A dragon-scale cavern remains a small semantic declaration."""

        environment = SemanticEnvironmentSpec(
            regions=(
                EnvironmentRegionSpec(
                    "mountain_interior",
                    EnvironmentKind.SUBTERRANEAN,
                    Bounds3D((-125, -90, -35), (125, 90, 110)),
                ),
            ),
            chambers=(
                CavernChamberSpec(
                    "colossal_chamber",
                    "mountain_interior",
                    (0, 0, 20),
                    (180, 120, 70),
                    shape=CavernShape.SUPERELLIPSOID,
                ),
            ),
            passage_networks=(
                PassageNetworkSpec(
                    "approach_routes",
                    "mountain_interior",
                    (
                        PassageJunctionSpec("entrance", (-115, 0, 0)),
                        PassageJunctionSpec(
                            "hall_entry", (-75, 0, 0), chamber_id="colossal_chamber"
                        ),
                    ),
                    (
                        PassageSegmentSpec(
                            "approach_tunnel",
                            "entrance",
                            "hall_entry",
                            ((-115, 0, 0), (-75, 0, 0)),
                            _cross_sections(12, 14),
                        ),
                    ),
                ),
            ),
            openings=(
                EnvironmentOpeningSpec(
                    "sky_break",
                    "mountain_interior",
                    "colossal_chamber",
                    OpeningTarget.SKY,
                    (0, 0, 54),
                    (0, 0, 1),
                    (24, 18),
                    30,
                    weather_exposed=True,
                ),
            ),
            detail_fields=(
                DetailFieldSpec(
                    "ceiling_formations",
                    "mountain_interior",
                    "colossal_chamber",
                    FormationType.STALACTITE,
                    DetailSurfaceRole.OVERHEAD,
                    60,
                    (0.8, 0.8, 2.0),
                    (4.0, 4.0, 18.0),
                    8675309,
                    protect_passage_network_ids=("approach_routes",),
                    route_clearance=6,
                    collision_policy=DetailCollisionPolicy.COARSE,
                ),
            ),
            hero_features=(
                HeroFeatureSpec(
                    "central_rock_spire",
                    "mountain_interior",
                    "colossal_chamber",
                    HeroFeatureType.ROCK_SPIRE,
                    (30, 10, -5),
                    (18, 14, 32),
                ),
            ),
        )

        shell = compile_semantic_environment(
            environment,
            options=SemanticCompileOptions(voxel_size=5.0, max_cells=200_000),
        )
        details = compile_environment_details(environment)
        mesh = trimesh.Trimesh(
            vertices=shell.visual_mesh.vertices,
            faces=shell.visual_mesh.triangles,
            process=False,
        )
        source_ids = {
            patch.surface.metadata["semantic_source_id"] for patch in shell.surfaces
        }

        self.assertGreater(mesh.extents[0], 170)
        self.assertGreater(mesh.extents[1], 110)
        self.assertFalse(mesh.is_watertight)
        self.assertEqual(_mesh_component_count(mesh), 1)
        self.assertIn("colossal_chamber", source_ids)
        self.assertIn("approach_tunnel", source_ids)
        self.assertIn("sky_break", source_ids)
        self.assertEqual(len(details.instances), 61)
        self.assertEqual(
            {structure.structure_id for structure in details.structures},
            {"ceiling_formations", "central_rock_spire"},
        )
        self.assertLess(
            len(json.dumps(environment.to_dict(), separators=(",", ":"))), 5000
        )

    def test_gen_001_declaration_order_does_not_change_mesh(self) -> None:
        first = compile_semantic_environment(_environment(), options=self.options)
        second = compile_semantic_environment(
            _environment(reordered=True), options=self.options
        )

        self.assertEqual(first.visual_mesh, second.visual_mesh)
        self.assertEqual(first.triangle_groups, second.triangle_groups)

    def test_gen_002_renaming_does_not_change_geometry(self) -> None:
        original = _environment()
        renamed_data = original.to_dict()
        renamed_data["passage_networks"][0]["id"] = "routes_renamed"
        renamed_data["passage_networks"][0]["segments"][0]["id"] = "edge_renamed"
        renamed = SemanticEnvironmentSpec.from_dict(renamed_data)

        first = compile_semantic_environment(original, options=self.options)
        second = compile_semantic_environment(renamed, options=self.options)
        self.assertEqual(first.visual_mesh, second.visual_mesh)

    def test_gen_003_translation_is_metamorphic(self) -> None:
        delta = (30.0, -20.0, 7.0)
        first = compile_semantic_environment(_environment(), options=self.options)
        shifted = compile_semantic_environment(
            _environment(translated=delta), options=self.options
        )

        self.assertEqual(first.visual_mesh.triangles, shifted.visual_mesh.triangles)
        for before, after in zip(
            first.visual_mesh.vertices, shifted.visual_mesh.vertices
        ):
            for axis in range(3):
                self.assertAlmostEqual(after[axis] - before[axis], delta[axis])

    def test_gen_004_annotations_cover_every_triangle_once(self) -> None:
        compiled = compile_semantic_environment(_environment(), options=self.options)
        annotations = semantic_mesh_annotations(compiled)
        indices = [index for item in annotations for index in item.triangle_indices]

        self.assertEqual(
            sorted(indices), list(range(len(compiled.visual_mesh.triangles)))
        )
        self.assertEqual(len(indices), len(set(indices)))

    def test_gen_006_collinear_path_subdivision_preserves_geometry(self) -> None:
        def straight(path):
            region = EnvironmentRegionSpec(
                "subsurface",
                EnvironmentKind.SUBTERRANEAN,
                Bounds3D((-5, -10, -5), (25, 10, 10)),
            )
            return SemanticEnvironmentSpec(
                regions=(region,),
                passage_networks=(
                    PassageNetworkSpec(
                        "routes",
                        "subsurface",
                        (
                            PassageJunctionSpec("a", (0, 0, 0)),
                            PassageJunctionSpec("b", (20, 0, 0)),
                        ),
                        (
                            PassageSegmentSpec(
                                "edge", "a", "b", path, _cross_sections(4, 4)
                            ),
                        ),
                    ),
                ),
            )

        first = compile_semantic_environment(
            straight(((0, 0, 0), (20, 0, 0))), options=self.options
        )
        subdivided = compile_semantic_environment(
            straight(((0, 0, 0), (5, 0, 0), (12, 0, 0), (20, 0, 0))),
            options=self.options,
        )

        self.assertEqual(first.visual_mesh.triangles, subdivided.visual_mesh.triangles)
        for before, after in zip(
            first.visual_mesh.vertices, subdivided.visual_mesh.vertices
        ):
            for axis in range(3):
                self.assertAlmostEqual(before[axis], after[axis], places=12)

    def test_open_boundary_removes_endpoint_cap(self) -> None:
        data = _environment().to_dict()
        data["passage_networks"][0]["junctions"][0]["open_boundary"] = True
        compiled = compile_semantic_environment(
            SemanticEnvironmentSpec.from_dict(data), options=self.options
        )
        mesh = trimesh.Trimesh(
            vertices=compiled.visual_mesh.vertices,
            faces=compiled.visual_mesh.triangles,
            process=False,
        )

        self.assertFalse(mesh.is_watertight)
        self.assertTrue(
            any(count == 1 for count in np.bincount(mesh.edges_unique_inverse))
        )

    def test_gen_005_production_source_has_no_scenario_special_cases(self) -> None:
        root = Path(__file__).parents[2] / "scenesmith" / "agent_utils"
        banned = {"prison", "escape_tunnel", "dragon_lair", "great_hall"}
        for name in (
            "semantic_environments.py",
            "semantic_environment_compiler.py",
        ):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            strings = {
                node.value.lower()
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            for token in banned:
                self.assertFalse(
                    any(token in value for value in strings),
                    f"{token!r} leaked into {name}",
                )

    def test_house_layout_compiles_serializes_and_exports_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout = HouseLayout(semantic_environment=_environment())

            paths = layout.compile_semantic_environment(
                root / "environment", voxel_size=1.0
            )
            directive = layout.to_drake_directive(base_dir=root)
            restored = HouseLayout.from_dict(layout.to_dict(scene_dir=root), root)
            sidecar = json.loads(paths.surfaces_path.read_text(encoding="utf-8"))
            restored_patches = load_surface_patches(paths.surfaces_path)

            self.assertTrue(paths.mesh_path.is_file())
            self.assertTrue(paths.sdf_path.is_file())
            self.assertTrue(paths.surfaces_path.is_file())
            self.assertIn("structure_semantic_environment", directive)
            self.assertEqual(restored.semantic_environment, layout.semantic_environment)
            self.assertEqual(
                restored.semantic_environment_geometry_path,
                layout.semantic_environment_geometry_path,
            )
            self.assertEqual(sidecar["surface_encoding"], "triangle_mesh_v1")
            self.assertNotIn("surfaces", sidecar)
            self.assertEqual(len(restored_patches), sidecar["visual_triangles"])


if __name__ == "__main__":
    unittest.main()
