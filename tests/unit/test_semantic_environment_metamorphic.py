"""Generated/metamorphic semantic-geometry tests with independent predicates."""

import hashlib
import math
import os
import subprocess
import sys
import textwrap
import unittest

from dataclasses import replace
from pathlib import Path

from scenesmith.agent_utils.semantic_environment_compiler import (
    SemanticCompileOptions,
    compile_semantic_environment,
)
from scenesmith.agent_utils.semantic_environments import (
    Bounds3D,
    CavernChamberSpec,
    EnvironmentKind,
    EnvironmentRegionSpec,
    PassageCrossSectionSpec,
    PassageJunctionSpec,
    PassageNetworkSpec,
    PassageSegmentSpec,
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structural_compiler import audit_triangle_mesh


def _rotate_z(point, yaw):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        cosine * point[0] - sine * point[1],
        sine * point[0] + cosine * point[1],
        point[2],
    )


def _star_environment(degree: int, *, scale: float = 1.0) -> SemanticEnvironmentSpec:
    radius = 12.0 * scale
    root = PassageJunctionSpec("root", (0, 0, 0))
    leaves = tuple(
        PassageJunctionSpec(
            f"leaf_{index}",
            (
                radius * math.cos(math.tau * index / degree),
                radius * math.sin(math.tau * index / degree),
                ((index % 3) - 1) * scale,
            ),
        )
        for index in range(degree)
    )
    sections = (
        PassageCrossSectionSpec(0, 3 * scale, 3 * scale),
        PassageCrossSectionSpec(1, 4 * scale, 4 * scale),
    )
    segments = tuple(
        PassageSegmentSpec(
            f"edge_{index}",
            "root",
            leaf.junction_id,
            (root.position, leaf.position),
            sections,
        )
        for index, leaf in enumerate(leaves)
    )
    return SemanticEnvironmentSpec(
        regions=(
            EnvironmentRegionSpec(
                "generated_region",
                EnvironmentKind.SUBTERRANEAN,
                Bounds3D(
                    (-20 * scale, -20 * scale, -10 * scale),
                    (20 * scale, 20 * scale, 10 * scale),
                ),
            ),
        ),
        passage_networks=(
            PassageNetworkSpec(
                "generated_network", "generated_region", (root, *leaves), segments
            ),
        ),
    )


class TestGeneratedSemanticGeometry(unittest.TestCase):
    def test_generated_chamber_and_graph_families_one_through_fifty_validate(
        self,
    ) -> None:
        for size in range(1, 51):
            with self.subTest(family="graph", size=size):
                graph = _star_environment(size)
                network = graph.passage_networks[0]
                self.assertEqual(network.degree("root"), size)
                self.assertEqual(len(network.reachable("root")), size + 1)

            with self.subTest(family="chambers", size=size):
                chambers = tuple(
                    CavernChamberSpec(
                        f"chamber_{index}",
                        "generated_region",
                        (4.0 * index, 0.0, 0.0),
                        (2.0, 2.0, 2.0),
                    )
                    for index in range(size)
                )
                environment = SemanticEnvironmentSpec(
                    regions=(
                        EnvironmentRegionSpec(
                            "generated_region",
                            EnvironmentKind.SUBTERRANEAN,
                            Bounds3D(
                                (-2.0, -2.0, -2.0),
                                (4.0 * (size - 1) + 2.0, 2.0, 2.0),
                            ),
                        ),
                    ),
                    chambers=chambers,
                )
                self.assertEqual(len(environment.chambers), size)

    def test_generated_graph_degrees_one_through_five_compile_closed(self) -> None:
        for degree in range(1, 6):
            with self.subTest(degree=degree):
                environment = _star_environment(degree)
                compiled = compile_semantic_environment(
                    environment,
                    options=SemanticCompileOptions(voxel_size=1.5, max_cells=250_000),
                )
                audit = audit_triangle_mesh(compiled.visual_mesh)

                self.assertEqual(environment.passage_networks[0].degree("root"), degree)
                self.assertTrue(audit.is_closed)
                self.assertTrue(audit.is_winding_consistent)

    def test_rotation_is_metamorphic(self) -> None:
        environment = _star_environment(3)
        yaw = math.pi / 2
        region = environment.regions[0]
        rotated_region = replace(
            region,
            transform=replace(region.transform, rotation_rpy=(0, 0, yaw)),
        )
        rotated_environment = replace(environment, regions=(rotated_region,))

        first = compile_semantic_environment(
            environment, options=SemanticCompileOptions(voxel_size=1.5)
        )
        second = compile_semantic_environment(
            rotated_environment, options=SemanticCompileOptions(voxel_size=1.5)
        )

        first_bounds = first.visual_mesh.bounds
        second_bounds = second.visual_mesh.bounds
        rotated_corners = tuple(
            _rotate_z((x, y, z), yaw)
            for x in (first_bounds[0][0], first_bounds[1][0])
            for y in (first_bounds[0][1], first_bounds[1][1])
            for z in (first_bounds[0][2], first_bounds[1][2])
        )
        expected_bounds = tuple(
            tuple(
                reducer(point[axis] for point in rotated_corners) for axis in range(3)
            )
            for reducer in (min, max)
        )
        for expected, actual in zip(expected_bounds, second_bounds):
            for axis in range(3):
                self.assertAlmostEqual(actual[axis], expected[axis], delta=1.5)
        self.assertEqual(
            {patch.surface.metadata["semantic_source_id"] for patch in first.surfaces},
            {patch.surface.metadata["semantic_source_id"] for patch in second.surfaces},
        )

    def test_uniform_scale_preserves_normalized_bounds_and_topology(self) -> None:
        first = compile_semantic_environment(
            _star_environment(4, scale=1),
            options=SemanticCompileOptions(voxel_size=1.5),
        )
        scaled = compile_semantic_environment(
            _star_environment(4, scale=2),
            options=SemanticCompileOptions(voxel_size=3.0),
        )

        self.assertEqual(first.visual_mesh.triangles, scaled.visual_mesh.triangles)
        for before, after in zip(
            first.visual_mesh.vertices, scaled.visual_mesh.vertices
        ):
            for axis in range(3):
                self.assertAlmostEqual(after[axis], before[axis] * 2, places=9)

    def test_resolution_changes_detail_not_semantic_source_coverage(self) -> None:
        environment = _star_environment(5)
        source_ids = {
            segment.segment_id for segment in environment.passage_networks[0].segments
        }
        triangle_counts = []
        for voxel_size in (2.0, 1.5, 1.0):
            compiled = compile_semantic_environment(
                environment,
                options=SemanticCompileOptions(
                    voxel_size=voxel_size, max_cells=500_000
                ),
            )
            triangle_counts.append(len(compiled.visual_mesh.triangles))
            compiled_sources = {
                patch.surface.metadata["semantic_source_id"]
                for patch in compiled.surfaces
            }
            self.assertEqual(compiled_sources, source_ids)
        self.assertEqual(triangle_counts, sorted(triangle_counts))

    def test_compilation_is_deterministic_across_hash_seeds(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        script = textwrap.dedent(
            """
            import hashlib
            from tests.unit.test_semantic_environment_metamorphic import _star_environment
            from scenesmith.agent_utils.semantic_environment_compiler import (
                SemanticCompileOptions,
                compile_semantic_environment,
            )

            compiled = compile_semantic_environment(
                _star_environment(3),
                options=SemanticCompileOptions(voxel_size=2.0),
            )
            payload = (
                compiled.visual_mesh.to_obj() + compiled.collision_mesh.to_obj()
            ).encode("utf-8")
            print(hashlib.sha256(payload).hexdigest())
            """
        )
        digests = []
        for hash_seed in ("1", "971"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = hash_seed
            output = subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=repository,
                env=environment,
                text=True,
            )
            digest = output.splitlines()[-1].strip()
            self.assertEqual(len(digest), hashlib.sha256().digest_size * 2)
            digests.append(digest)
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
