"""Determinism and safety tests for semantic geological detail fields."""

import json
import math
import tempfile
import unittest

from pathlib import Path

import trimesh

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.semantic_environment_details import (
    FORMATION_MESH_FAMILY,
    HERO_FORMATION_TYPE,
    compile_environment_details,
    sample_detail_field,
)
from scenesmith.agent_utils.semantic_environments import (
    FORMATION_SURFACE_ROLES,
    Bounds3D,
    CavernChamberSpec,
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
    PassageSegmentSpec,
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structural_compiler import write_compiled_structure
from scenesmith.agent_utils.structural_geometry import GeometryValidationError


def _distance_to_segment(point, start, end):
    span = tuple(end[axis] - start[axis] for axis in range(3))
    length_squared = sum(value * value for value in span)
    amount = min(
        1.0,
        max(
            0.0,
            sum((point[axis] - start[axis]) * span[axis] for axis in range(3))
            / length_squared,
        ),
    )
    closest = tuple(start[axis] + span[axis] * amount for axis in range(3))
    return math.dist(point, closest)


def _instance_envelope(instance):
    half_height = instance.size[2] / 2
    center = tuple(
        instance.anchor[axis] + instance.axis[axis] * half_height for axis in range(3)
    )
    radius = math.hypot(max(instance.size[0], instance.size[1]) / 2, half_height)
    return center, radius


def _environment(seed=419, collision_policy=DetailCollisionPolicy.COARSE):
    region = EnvironmentRegionSpec(
        "underpeak",
        EnvironmentKind.SUBTERRANEAN,
        Bounds3D((-60, -50, -30), (80, 50, 60)),
    )
    chamber = CavernChamberSpec("large_chamber", "underpeak", (0, 0, 10), (100, 70, 50))
    route = PassageNetworkSpec(
        "approach",
        "underpeak",
        (
            PassageJunctionSpec("entry", (-45, 0, 0)),
            PassageJunctionSpec("center", (0, 0, 0), chamber_id="large_chamber"),
        ),
        (
            PassageSegmentSpec(
                "entry_run",
                "entry",
                "center",
                ((-45, 0, 0), (0, 0, 0)),
                (
                    PassageCrossSectionSpec(0, 8, 8),
                    PassageCrossSectionSpec(1, 12, 12),
                ),
            ),
        ),
    )
    field = DetailFieldSpec(
        "ceiling_teeth",
        "underpeak",
        "large_chamber",
        FormationType.STALACTITE,
        DetailSurfaceRole.OVERHEAD,
        30,
        (0.5, 0.5, 1.0),
        (2.5, 2.5, 8.0),
        seed,
        protect_passage_network_ids=("approach",),
        route_clearance=3.0,
        collision_policy=collision_policy,
    )
    return SemanticEnvironmentSpec(
        regions=(region,),
        chambers=(chamber,),
        passage_networks=(route,),
        openings=(
            EnvironmentOpeningSpec(
                "oculus",
                "underpeak",
                "large_chamber",
                OpeningTarget.SKY,
                (0, 0, 34),
                (0, 0, 1),
                (12, 10),
                15,
            ),
        ),
        detail_fields=(field,),
        hero_features=(
            HeroFeatureSpec(
                "perch",
                "underpeak",
                "large_chamber",
                HeroFeatureType.ROCK_SPIRE,
                (18, 10, -5),
                (8, 6, 14),
            ),
        ),
    )


class TestSemanticEnvironmentDetails(unittest.TestCase):
    def test_detail_dispatch_registries_are_exhaustive_and_fail_closed(self) -> None:
        self.assertEqual(frozenset(FORMATION_MESH_FAMILY), frozenset(FormationType))
        self.assertEqual(frozenset(HERO_FORMATION_TYPE), frozenset(HeroFeatureType))

    def test_every_supported_formation_role_and_collision_policy_compiles(self) -> None:
        base = _environment().to_dict()
        base["openings"] = []
        base["hero_features"] = []
        for formation_type in FormationType:
            for role in FORMATION_SURFACE_ROLES[formation_type]:
                for collision_policy in DetailCollisionPolicy:
                    with self.subTest(
                        formation_type=formation_type.value,
                        role=role.value,
                        collision_policy=collision_policy.value,
                    ):
                        data = json.loads(json.dumps(base))
                        field = data["detail_fields"][0]
                        field.update(
                            {
                                "formation_type": formation_type.value,
                                "surface_role": role.value,
                                "collision_policy": collision_policy.value,
                                "count": 1,
                                "min_size": [1.0, 1.0, 2.0],
                                "max_size": [1.0, 1.0, 2.0],
                            }
                        )
                        compiled = compile_environment_details(
                            SemanticEnvironmentSpec.from_dict(data)
                        ).structures[0]
                        mesh = compiled.visual_mesh

                        self.assertGreater(len(mesh.triangles), 0)
                        edge_uses = {}
                        for triangle in mesh.triangles:
                            self.assertEqual(len(set(triangle)), 3)
                            for first, second in (
                                (triangle[0], triangle[1]),
                                (triangle[1], triangle[2]),
                                (triangle[2], triangle[0]),
                            ):
                                edge = tuple(sorted((first, second)))
                                edge_uses[edge] = edge_uses.get(edge, 0) + 1
                        self.assertTrue(edge_uses)
                        self.assertTrue(all(count == 2 for count in edge_uses.values()))
                        audited = trimesh.Trimesh(
                            vertices=mesh.vertices,
                            faces=mesh.triangles,
                            process=False,
                        )
                        self.assertTrue(audited.is_winding_consistent)
                        self.assertGreater(audited.volume, 0.0)
                        expected_role = {
                            DetailSurfaceRole.OVERHEAD: "overhead",
                            DetailSurfaceRole.SUPPORT: "support",
                            DetailSurfaceRole.BOUNDARY: "boundary",
                        }[role]
                        self.assertTrue(
                            all(
                                expected_role
                                in {
                                    surface_role.value
                                    for surface_role in patch.surface.roles
                                }
                                for patch in compiled.surfaces
                            )
                        )

    def test_unsupported_formation_surface_combinations_fail_early(self) -> None:
        for formation_type in FormationType:
            for role in DetailSurfaceRole:
                if role in FORMATION_SURFACE_ROLES[formation_type]:
                    continue
                with self.subTest(formation_type=formation_type.value, role=role.value):
                    with self.assertRaisesRegex(
                        GeometryValidationError, "unsupported_formation_surface"
                    ):
                        DetailFieldSpec(
                            "invalid_field",
                            "underpeak",
                            "large_chamber",
                            formation_type,
                            role,
                            1,
                            (1, 1, 1),
                            (2, 2, 2),
                            1,
                        )

    def test_every_hero_type_compiles(self) -> None:
        base = _environment().to_dict()
        base["detail_fields"] = []
        base["openings"] = []
        for hero_type in HeroFeatureType:
            for collision_policy in DetailCollisionPolicy:
                with self.subTest(
                    hero_type=hero_type.value,
                    collision_policy=collision_policy.value,
                ):
                    data = json.loads(json.dumps(base))
                    data["hero_features"][0]["feature_type"] = hero_type.value
                    data["hero_features"][0][
                        "collision_policy"
                    ] = collision_policy.value
                    compiled = compile_environment_details(
                        SemanticEnvironmentSpec.from_dict(data)
                    ).structures[0]

                    self.assertGreater(len(compiled.visual_mesh.triangles), 0)

    def test_coarse_collision_is_distinct_and_smaller_than_full_geometry(self) -> None:
        coarse = compile_environment_details(
            _environment(collision_policy=DetailCollisionPolicy.COARSE)
        ).structures[0]
        full = compile_environment_details(
            _environment(collision_policy=DetailCollisionPolicy.FULL)
        ).structures[0]

        self.assertLess(
            len(coarse.collision_mesh.triangles), len(coarse.visual_mesh.triangles)
        )
        self.assertEqual(full.collision_mesh, full.visual_mesh)

    def test_detail_fields_avoid_each_other_across_independent_seeds(self) -> None:
        data = _environment().to_dict()
        data["openings"] = []
        data["hero_features"] = []
        first = data["detail_fields"][0]
        first.update(
            {
                "id": "field_a",
                "formation_type": "stalagmite",
                "surface_role": "support",
                "count": 15,
                "min_size": [2, 2, 3],
                "max_size": [3, 3, 5],
                "seed": 73,
            }
        )
        second = json.loads(json.dumps(first))
        second.update({"id": "field_b", "seed": 74})
        data["detail_fields"] = [first, second]

        compiled = compile_environment_details(SemanticEnvironmentSpec.from_dict(data))
        envelopes = [_instance_envelope(instance) for instance in compiled.instances]

        for index, (center, radius) in enumerate(envelopes):
            for other_center, other_radius in envelopes[index + 1 :]:
                self.assertGreaterEqual(
                    math.dist(center, other_center), radius + other_radius
                )

    def test_variable_passage_swept_envelope_not_only_centerline_is_protected(
        self,
    ) -> None:
        data = _environment().to_dict()
        data["openings"] = []
        data["hero_features"] = []
        data["detail_fields"][0].update(
            {
                "formation_type": "stalagmite",
                "surface_role": "support",
                "count": 20,
                "min_size": [1, 1, 2],
                "max_size": [2, 2, 4],
                "route_clearance": 1,
            }
        )
        environment = SemanticEnvironmentSpec.from_dict(data)
        instances, _ = sample_detail_field(environment.detail_fields[0], environment)

        for instance in instances:
            center, radius = _instance_envelope(instance)
            # The passage expands to 12 m wide at the chamber junction.  The
            # exclusion therefore includes its 6 m half-width plus authoring
            # clearance and the complete detail envelope.
            self.assertGreaterEqual(
                _distance_to_segment(center, (-45, 0, 0), (0, 0, 0)),
                6.0 + environment.detail_fields[0].route_clearance + radius,
            )

    def test_form_001_same_seed_is_byte_deterministic(self) -> None:
        environment = _environment()

        first = compile_environment_details(environment)
        second = compile_environment_details(environment)

        self.assertEqual(first, second)
        self.assertEqual(
            first.structures[0].visual_mesh, second.structures[0].visual_mesh
        )

    def test_form_002_different_seed_changes_distribution_not_count(self) -> None:
        first = compile_environment_details(_environment(seed=419))
        second = compile_environment_details(_environment(seed=420))

        self.assertEqual(len(first.instances), len(second.instances))
        self.assertNotEqual(
            [item.anchor for item in first.instances],
            [item.anchor for item in second.instances],
        )

    def test_form_003_protected_route_opening_and_hero_masks_hold(self) -> None:
        environment = _environment()
        field = environment.detail_fields[0]
        instances, dropped = sample_detail_field(field, environment)

        self.assertGreater(dropped, 0)
        for instance in instances:
            center, radius = _instance_envelope(instance)
            self.assertGreaterEqual(
                _distance_to_segment(center, (-45, 0, 0), (0, 0, 0)),
                field.route_clearance + radius,
            )
            self.assertGreater(
                math.dist(center, environment.openings[0].center),
                max(environment.openings[0].size) / 2 + radius,
            )
            self.assertGreater(
                math.dist(center, environment.hero_features[0].anchor),
                max(environment.hero_features[0].size) / 2 + radius,
            )

    def test_form_003_masks_hold_across_50_seeds(self) -> None:
        for seed in range(50):
            data = _environment(seed=seed).to_dict()
            field_data = data["detail_fields"][0]
            field_data.update(
                {
                    "formation_type": FormationType.STALAGMITE.value,
                    "surface_role": DetailSurfaceRole.SUPPORT.value,
                    "count": 8,
                    "min_size": [1.0, 1.0, 10.0],
                    "max_size": [3.0, 3.0, 24.0],
                }
            )
            environment = SemanticEnvironmentSpec.from_dict(data)
            field = environment.detail_fields[0]
            instances, _ = sample_detail_field(field, environment)

            for instance in instances:
                center, radius = _instance_envelope(instance)
                self.assertGreaterEqual(
                    _distance_to_segment(center, (-45, 0, 0), (0, 0, 0)),
                    field.route_clearance + radius,
                    f"route mask failed at seed {seed}",
                )
                self.assertGreaterEqual(
                    math.dist(center, environment.openings[0].center),
                    max(environment.openings[0].size) / 2 + radius,
                    f"opening mask failed at seed {seed}",
                )
                self.assertGreaterEqual(
                    math.dist(center, environment.hero_features[0].anchor),
                    max(environment.hero_features[0].size) / 2 + radius,
                    f"hero mask failed at seed {seed}",
                )

    def test_formations_point_inward_from_their_sampled_surface(self) -> None:
        environment = _environment()
        instances, _ = sample_detail_field(environment.detail_fields[0], environment)
        chamber_center = environment.chambers[0].center

        for instance in instances:
            toward_center = tuple(
                chamber_center[axis] - instance.anchor[axis] for axis in range(3)
            )
            self.assertGreater(
                sum(instance.axis[axis] * toward_center[axis] for axis in range(3)),
                0.0,
            )

    def test_form_004_visual_only_detail_has_no_collision_export(self) -> None:
        compiled = compile_environment_details(
            _environment(collision_policy=DetailCollisionPolicy.VISUAL_ONLY)
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(
                compiled.structures[0], Path(temporary_directory)
            )
            sdf = paths.sdf_path.read_text(encoding="utf-8")

        self.assertFalse(compiled.structures[0].collision_enabled)
        self.assertNotIn("<collision", sdf)

    def test_form_005_hero_is_explicit_colliding_structure(self) -> None:
        compiled = compile_environment_details(_environment())
        hero = next(
            structure
            for structure in compiled.structures
            if structure.structure_id == "perch"
        )

        self.assertTrue(hero.collision_enabled)
        self.assertGreater(len(hero.visual_mesh.triangles), 0)

    def test_adv_no_legal_samples_fails_with_field_id(self) -> None:
        environment = _environment()
        data = environment.to_dict()
        field = data["detail_fields"][0]
        field["count"] = 1
        field["route_clearance"] = 1000
        impossible = SemanticEnvironmentSpec.from_dict(data)

        with self.assertRaisesRegex(
            GeometryValidationError,
            r"ceiling_teeth.*mask causes: passage=256",
        ):
            sample_detail_field(impossible.detail_fields[0], impossible)

    def test_house_layout_exports_detail_and_hero_directives(self) -> None:
        layout = HouseLayout(semantic_environment=_environment())
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout.compile_semantic_environment(root / "shell", voxel_size=4.0)
            paths = layout.compile_semantic_environment_details(root / "details")
            directive = layout.to_drake_directive(base_dir=root)
            restored = HouseLayout.from_dict(layout.to_dict(scene_dir=root), root)

        self.assertEqual(set(paths), {"ceiling_teeth", "perch"})
        self.assertIn("environment_detail_ceiling_teeth", directive)
        self.assertIn("environment_detail_perch", directive)
        self.assertEqual(
            restored.semantic_detail_geometry_paths,
            layout.semantic_detail_geometry_paths,
        )

    def test_house_layout_rejects_stale_semantic_detail_artifacts(self) -> None:
        layout = HouseLayout(semantic_environment=_environment(seed=419))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout.compile_semantic_environment(root / "shell", voxel_size=4.0)
            layout.compile_semantic_environment_details(root / "details")
            layout.semantic_environment = _environment(seed=420)

            with self.assertRaisesRegex(ValueError, "stale"):
                layout.to_drake_directive(base_dir=root)


if __name__ == "__main__":
    unittest.main()
