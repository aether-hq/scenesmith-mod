"""Geometry proofs for the semantic natural-environment compiler."""

import json
import math
import unittest

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from scenesmith.agent_utils.semantics.environment.models.chambers import (
    Bounds3D,
    CavernChamberSpec,
    EnvironmentRegionSpec,
)
from scenesmith.agent_utils.semantics.environment.models.common import (
    CavernShape,
    EnvironmentKind,
    OpeningShape,
    OpeningTarget,
    PassageFloorMode,
    PassageProfile,
)
from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.semantics.environment.models.features import (
    EnvironmentOpeningSpec,
)
from scenesmith.agent_utils.semantics.environment.models.passages import (
    PassageCrossSectionSpec,
    PassageJunctionSpec,
    PassageNetworkSpec,
    PassageSegmentSpec,
)
from scenesmith.agent_utils.semantics.environment.semantic_environment_compiler import (
    SEMANTIC_COMPILER_CAPABILITIES,
    SemanticCompileOptions,
    _build_primitives,
    _union_value,
    compile_semantic_environment,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import SurfaceRole
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)


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


def _nearest_ray_hit(mesh, origin: np.ndarray, direction: np.ndarray) -> float:
    """Return the nearest forward triangle intersection without optional rtree."""

    nearest = math.inf
    for triangle in mesh.triangles:
        first, second, third = (
            np.asarray(mesh.vertices[index], dtype=float) for index in triangle
        )
        edge_a, edge_b = second - first, third - first
        cross = np.cross(direction, edge_b)
        determinant = np.dot(edge_a, cross)
        if abs(determinant) <= 1e-9:
            continue
        inverse = 1.0 / determinant
        offset = origin - first
        u = inverse * np.dot(offset, cross)
        if u < 0.0 or u > 1.0:
            continue
        q = np.cross(offset, edge_a)
        v = inverse * np.dot(direction, q)
        if v < 0.0 or u + v > 1.0:
            continue
        distance = inverse * np.dot(edge_b, q)
        if distance > 1e-6:
            nearest = min(nearest, distance)
    return nearest


class TestSemanticEnvironmentCompiler(unittest.TestCase):
    options = SemanticCompileOptions(voxel_size=1.0, max_cells=100_000)

    def test_compiler_capability_registry_is_exhaustive_and_fail_closed(self) -> None:
        self.assertEqual(
            frozenset(SEMANTIC_COMPILER_CAPABILITIES["chamber_shapes"]),
            frozenset(CavernShape),
        )
        self.assertEqual(
            frozenset(SEMANTIC_COMPILER_CAPABILITIES["passage_profiles"]),
            frozenset(PassageProfile),
        )
        self.assertEqual(
            frozenset(SEMANTIC_COMPILER_CAPABILITIES["opening_shapes"]),
            frozenset(OpeningShape),
        )
        self.assertEqual(
            frozenset(SEMANTIC_COMPILER_CAPABILITIES["opening_targets"]),
            frozenset(OpeningTarget),
        )
        self.assertEqual(
            frozenset(SEMANTIC_COMPILER_CAPABILITIES["passage_floor_modes"]),
            frozenset(PassageFloorMode),
        )
        self.assertFalse(
            SEMANTIC_COMPILER_CAPABILITIES["passage_floor_modes"][
                PassageFloorMode.STEPS
            ]
        )

    def test_unsupported_step_floor_fails_during_semantic_validation(self) -> None:
        data = _environment().to_dict()
        data["passage_networks"][0]["segments"][0]["floor_mode"] = "steps"

        with self.assertRaisesRegex(
            GeometryValidationError, "unsupported_passage_floor_mode"
        ):
            SemanticEnvironmentSpec.from_dict(data)

    def test_closed_grid_aligned_chamber_is_watertight_at_multiple_resolutions(
        self,
    ) -> None:
        region = EnvironmentRegionSpec(
            "closed_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 20)),
        )
        environment = SemanticEnvironmentSpec(
            regions=(region,),
            chambers=(
                CavernChamberSpec(
                    "closed_chamber", "closed_region", (0, 0, 0), (20, 20, 20)
                ),
            ),
        )

        for voxel_size in (0.5, 1.0, 2.0):
            with self.subTest(voxel_size=voxel_size):
                compiled = compile_semantic_environment(
                    environment,
                    options=SemanticCompileOptions(
                        voxel_size=voxel_size, max_cells=250_000
                    ),
                )
                mesh = trimesh.Trimesh(
                    vertices=compiled.visual_mesh.vertices,
                    faces=compiled.visual_mesh.triangles,
                    process=False,
                )
                self.assertTrue(mesh.is_watertight)

    def test_declared_opening_must_create_a_physical_aperture(self) -> None:
        region = EnvironmentRegionSpec(
            "sealed_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 20)),
        )
        chamber = CavernChamberSpec(
            "sealed_chamber", "sealed_region", (0, 0, 0), (20, 20, 20)
        )
        environment = SemanticEnvironmentSpec(
            regions=(region,),
            chambers=(chamber,),
            openings=(
                EnvironmentOpeningSpec(
                    "sealed_opening",
                    "sealed_region",
                    "sealed_chamber",
                    OpeningTarget.SKY,
                    (0, 0, 0),
                    (0, 0, 1),
                    (2, 2),
                    1,
                ),
            ),
        )

        with self.assertRaisesRegex(
            GeometryValidationError, "declared_opening_missing"
        ):
            compile_semantic_environment(environment, options=self.options)

    def test_declared_opening_does_not_hide_an_unrelated_mesh_crack(self) -> None:
        region = EnvironmentRegionSpec(
            "cracked_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 30)),
        )
        environment = SemanticEnvironmentSpec(
            regions=(region,),
            chambers=(
                CavernChamberSpec(
                    "cracked_chamber", "cracked_region", (0, 0, 0), (20, 20, 20)
                ),
            ),
            openings=(
                EnvironmentOpeningSpec(
                    "declared_opening",
                    "cracked_region",
                    "cracked_chamber",
                    OpeningTarget.SKY,
                    (0, 0, 9.5),
                    (0, 0, 1),
                    (4, 4),
                    8,
                ),
            ),
        )
        valid = compile_semantic_environment(environment, options=self.options)
        removed_index = min(
            range(len(valid.surfaces)),
            key=lambda index: abs(valid.surfaces[index].boundary[0][2] + 9.0),
        )
        cracked = replace(
            valid.visual_mesh,
            triangles=tuple(
                triangle
                for index, triangle in enumerate(valid.visual_mesh.triangles)
                if index != removed_index
            ),
        )

        with patch(
            "scenesmith.agent_utils.semantics.environment.semantic_environment_compiler._extract_mesh",
            return_value=cracked,
        ):
            with self.assertRaisesRegex(
                GeometryValidationError, "undeclared_boundary_edge"
            ):
                compile_semantic_environment(environment, options=self.options)

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

    def test_coarse_extraction_vertices_remain_on_nonlinear_passage_boundary(
        self,
    ) -> None:
        """A coarse mesh may be faceted, but must not cut through authored free space."""

        record = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "docs/geometry-extension/llm-trials/results/heldout_branching_network_v1.json"
            ).read_text(encoding="utf-8")
        )
        environment = SemanticEnvironmentSpec.from_dict(record["semantic_environment"])
        compiled = compile_semantic_environment(
            environment,
            options=SemanticCompileOptions(
                voxel_size=1.4,
                max_cells=500_000,
                max_triangles=500_000,
            ),
        )
        primitives = _build_primitives(environment)
        maximum_boundary_error = max(
            abs(_union_value(primitives, vertex))
            for vertex in compiled.visual_mesh.vertices
        )
        centerline_start = np.array([0.0, 0.0, 0.0])
        centerline_end = np.array([16.0, -14.0, -11.0])
        tangent = centerline_end / np.linalg.norm(centerline_end)
        across = np.cross([0.0, 0.0, 1.0], tangent)
        across /= np.linalg.norm(across)
        vertical = np.cross(tangent, across)
        camera = centerline_start + tangent * 1.25 + vertical * 1.6
        target = centerline_start + tangent * 10.0 + vertical * 1.6
        direction = target - camera
        direction /= np.linalg.norm(direction)
        first_hit = _nearest_ray_hit(compiled.visual_mesh, camera, direction)

        self.assertLess(maximum_boundary_error, 1e-6)
        self.assertTrue(math.isinf(first_hit) or first_hit > 7.5, first_hit)

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

    def test_all_opening_shapes_and_targets_compile(self) -> None:
        for shape in OpeningShape:
            for target in OpeningTarget:
                with self.subTest(shape=shape.value, target=target.value):
                    region = EnvironmentRegionSpec(
                        "opening_region",
                        EnvironmentKind.SUBTERRANEAN,
                        Bounds3D((-20, -20, -20), (20, 20, 30)),
                    )
                    chamber = CavernChamberSpec(
                        "opening_chamber",
                        "opening_region",
                        (0, 0, 0),
                        (20, 20, 20),
                    )
                    environment = SemanticEnvironmentSpec(
                        regions=(region,),
                        chambers=(chamber,),
                        openings=(
                            EnvironmentOpeningSpec(
                                "opening",
                                "opening_region",
                                "opening_chamber",
                                target,
                                (0, 0, 9.5),
                                (0, 0, 1),
                                (4, 4),
                                8,
                                shape=shape,
                            ),
                        ),
                    )
                    compiled = compile_semantic_environment(
                        environment, options=self.options
                    )
                    self.assertIn(
                        "opening",
                        {
                            item.surface.metadata["semantic_source_id"]
                            for item in compiled.surfaces
                        },
                    )

    def test_unsupported_chamber_shapes_fail_during_semantic_validation(self) -> None:
        for shape in (CavernShape.LOFT, CavernShape.VAULTED, CavernShape.MESH):
            with self.subTest(shape=shape.value):
                region = EnvironmentRegionSpec(
                    "unsupported_region",
                    EnvironmentKind.SUBTERRANEAN,
                    Bounds3D((-20, -20, -20), (20, 20, 20)),
                )
                with self.assertRaisesRegex(
                    GeometryValidationError, "unsupported_chamber_shape"
                ):
                    SemanticEnvironmentSpec(
                        regions=(region,),
                        chambers=(
                            CavernChamberSpec(
                                "unsupported_chamber",
                                "unsupported_region",
                                (0, 0, 0),
                                (10, 10, 10),
                                shape=shape,
                            ),
                        ),
                    )

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
