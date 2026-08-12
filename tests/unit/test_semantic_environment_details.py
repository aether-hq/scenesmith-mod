"""Determinism and safety tests for semantic geological detail fields."""

import math
import tempfile
import unittest

from pathlib import Path

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.semantic_environment_details import (
    compile_environment_details,
    sample_detail_field,
)
from scenesmith.agent_utils.semantic_environments import (
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
                    "count": 12,
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

        with self.assertRaisesRegex(GeometryValidationError, "ceiling_teeth"):
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


if __name__ == "__main__":
    unittest.main()
