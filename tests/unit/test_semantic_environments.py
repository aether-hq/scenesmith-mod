"""Semantic-model and genericity tests for natural 3D environments."""

import json
import unittest

from scenesmith.agent_utils.semantic_environments import (
    Bounds3D,
    CavernChamberSpec,
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
    PassageFloorMode,
    PassageJunctionSpec,
    PassageNetworkSpec,
    PassageSegmentSpec,
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structural_geometry import GeometryValidationError


def _region() -> EnvironmentRegionSpec:
    return EnvironmentRegionSpec(
        region_id="underground",
        kind=EnvironmentKind.SUBTERRANEAN,
        bounds=Bounds3D((-20, -20, -10), (30, 20, 20)),
        detail_seed=17,
    )


def _segment(
    segment_id: str,
    start: str,
    end: str,
    path: tuple[tuple[float, float, float], ...],
) -> PassageSegmentSpec:
    return PassageSegmentSpec(
        segment_id=segment_id,
        start_junction_id=start,
        end_junction_id=end,
        path=path,
        cross_sections=(
            PassageCrossSectionSpec(0.0, 4.0, 3.0),
            PassageCrossSectionSpec(1.0, 6.0, 5.0),
        ),
    )


class TestSemanticEnvironmentModel(unittest.TestCase):
    def test_llm_json_types_are_strict_instead_of_coercive(self) -> None:
        base = SemanticEnvironmentSpec(
            regions=(_region(),),
            passage_networks=(
                PassageNetworkSpec(
                    "strict_route",
                    "underground",
                    (
                        PassageJunctionSpec("strict_a", (0, 0, 0)),
                        PassageJunctionSpec("strict_b", (4, 0, 0)),
                    ),
                    (
                        _segment(
                            "strict_edge",
                            "strict_a",
                            "strict_b",
                            ((0, 0, 0), (4, 0, 0)),
                        ),
                    ),
                ),
            ),
        ).to_dict()
        base["passage_networks"][0]["junctions"][0]["open_boundary"] = "false"
        with self.assertRaisesRegex(GeometryValidationError, "invalid_boolean"):
            SemanticEnvironmentSpec.from_dict(base)

        base = SemanticEnvironmentSpec(
            regions=(_region(),),
            chambers=(
                CavernChamberSpec("room", "underground", (0, 0, 4), (10, 10, 8)),
            ),
        ).to_dict()
        base["detail_fields"] = [
            {
                "id": "bad_count",
                "region_id": "underground",
                "target_chamber_id": "room",
                "formation_type": "stalactite",
                "surface_role": "overhead",
                "count": 1.5,
                "min_size": [1, 1, 1],
                "max_size": [2, 2, 2],
                "seed": 4,
            }
        ]
        with self.assertRaisesRegex(GeometryValidationError, "invalid_integer"):
            SemanticEnvironmentSpec.from_dict(base)

        base = SemanticEnvironmentSpec(
            regions=(_region(),),
            chambers=(
                CavernChamberSpec(
                    "transform_room", "underground", (0, 0, 4), (10, 10, 8)
                ),
            ),
        ).to_dict()
        base["regions"][0]["transform"]["translation"][0] = "0"
        with self.assertRaisesRegex(GeometryValidationError, "invalid_number"):
            SemanticEnvironmentSpec.from_dict(base)

        base = SemanticEnvironmentSpec(
            regions=(_region(),),
            chambers=(
                CavernChamberSpec(
                    "numeric_room", "underground", (0, 0, 4), (10, 10, 8)
                ),
            ),
        ).to_dict()
        base["chambers"][0]["size"][0] = "10"
        with self.assertRaisesRegex(GeometryValidationError, "invalid_number"):
            SemanticEnvironmentSpec.from_dict(base)

        base = SemanticEnvironmentSpec(
            regions=(_region(),),
            chambers=(
                CavernChamberSpec("schema_room", "underground", (0, 0, 4), (10, 10, 8)),
            ),
        ).to_dict()
        base["schema_version"] = "1"
        with self.assertRaisesRegex(GeometryValidationError, "invalid_integer"):
            SemanticEnvironmentSpec.from_dict(base)

    def test_detail_and_scene_work_budgets_fail_before_sampling(self) -> None:
        region = EnvironmentRegionSpec(
            "budget_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-100, -100, -100), (100, 100, 100)),
        )
        chamber = CavernChamberSpec(
            "budget_chamber", "budget_region", (0, 0, 0), (100, 100, 100)
        )
        with self.assertRaisesRegex(
            GeometryValidationError, "semantic_budget_exceeded"
        ):
            SemanticEnvironmentSpec(
                regions=(region,),
                chambers=(chamber,),
                detail_fields=(
                    DetailFieldSpec(
                        "too_many",
                        "budget_region",
                        "budget_chamber",
                        FormationType.STALACTITE,
                        DetailSurfaceRole.OVERHEAD,
                        10_001,
                        (1, 1, 1),
                        (2, 2, 2),
                        7,
                    ),
                ),
            )

    def test_complete_hero_envelope_must_fit_inside_target_chamber(self) -> None:
        region = EnvironmentRegionSpec(
            "hero_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 20)),
        )
        chamber = CavernChamberSpec(
            "hero_chamber", "hero_region", (0, 0, 0), (10, 10, 10)
        )
        with self.assertRaisesRegex(GeometryValidationError, "hero_outside_chamber"):
            SemanticEnvironmentSpec(
                regions=(region,),
                chambers=(chamber,),
                hero_features=(
                    HeroFeatureSpec(
                        "oversized_hero",
                        "hero_region",
                        "hero_chamber",
                        HeroFeatureType.ROCK_SPIRE,
                        (0, 0, -4.5),
                        (12, 12, 12),
                    ),
                ),
            )

    def test_unsupported_floor_mode_is_rejected_by_the_semantic_contract(self) -> None:
        segment = PassageSegmentSpec(
            "steps",
            "a",
            "b",
            ((0, 0, 0), (4, 0, 1)),
            (
                PassageCrossSectionSpec(0, 2, 3),
                PassageCrossSectionSpec(1, 2, 3),
            ),
            floor_mode=PassageFloorMode.STEPS,
        )
        with self.assertRaisesRegex(
            GeometryValidationError, "unsupported_passage_floor_mode"
        ):
            SemanticEnvironmentSpec(
                regions=(_region(),),
                passage_networks=(
                    PassageNetworkSpec(
                        "route",
                        "underground",
                        (
                            PassageJunctionSpec("a", (0, 0, 0)),
                            PassageJunctionSpec("b", (4, 0, 1)),
                        ),
                        (segment,),
                    ),
                ),
            )

    def test_identifiers_must_be_safe_for_files_and_model_names(self) -> None:
        for unsafe_id in ("../escape", "nested/path", "has space", "line\nbreak"):
            with self.subTest(unsafe_id=unsafe_id):
                with self.assertRaisesRegex(
                    GeometryValidationError, "invalid_identifier"
                ):
                    EnvironmentRegionSpec(
                        unsafe_id,
                        EnvironmentKind.SUBTERRANEAN,
                        Bounds3D((-1, -1, -1), (1, 1, 1)),
                    )
        with self.assertRaisesRegex(GeometryValidationError, "invalid_identifier"):
            EnvironmentRegionSpec(
                42,  # type: ignore[arg-type]
                EnvironmentKind.SUBTERRANEAN,
                Bounds3D((-1, -1, -1), (1, 1, 1)),
            )

    def test_semantic_identifiers_are_globally_unique(self) -> None:
        region = EnvironmentRegionSpec(
            "region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 20)),
        )
        chamber = CavernChamberSpec("chamber", "region", (0, 0, 0), (10, 10, 10))
        detail = DetailFieldSpec(
            "shared",
            "region",
            "chamber",
            FormationType.STALACTITE,
            DetailSurfaceRole.OVERHEAD,
            1,
            (1, 1, 1),
            (1, 1, 2),
            1,
        )
        hero = HeroFeatureSpec(
            "shared",
            "region",
            "chamber",
            HeroFeatureType.ROCK_SPIRE,
            (0, 0, -4),
            (1, 1, 2),
        )

        with self.assertRaisesRegex(GeometryValidationError, "duplicate_semantic_id"):
            SemanticEnvironmentSpec(
                regions=(region,),
                chambers=(chamber,),
                detail_fields=(detail,),
                hero_features=(hero,),
            )

    def test_authored_identifiers_cannot_collide_with_derived_instance_ids(
        self,
    ) -> None:
        region = EnvironmentRegionSpec(
            "derived_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 20)),
        )
        chamber = CavernChamberSpec(
            "derived_chamber", "derived_region", (0, 0, 0), (10, 10, 10)
        )
        detail = DetailFieldSpec(
            "ceiling_teeth",
            "derived_region",
            "derived_chamber",
            FormationType.STALACTITE,
            DetailSurfaceRole.OVERHEAD,
            1,
            (1, 1, 1),
            (1, 1, 2),
            1,
        )

        with self.assertRaisesRegex(GeometryValidationError, "duplicate_semantic_id"):
            SemanticEnvironmentSpec(
                regions=(region,),
                chambers=(chamber,),
                detail_fields=(detail,),
                hero_features=(
                    HeroFeatureSpec(
                        "ceiling_teeth_0000",
                        "derived_region",
                        "derived_chamber",
                        HeroFeatureType.ROCK_SPIRE,
                        (0, 0, -4),
                        (1, 1, 2),
                    ),
                ),
            )

    def test_junction_opening_references_must_exist_in_the_same_region(self) -> None:
        region = EnvironmentRegionSpec(
            "opening_region",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-20, -20, -20), (20, 20, 20)),
        )
        chamber = CavernChamberSpec(
            "opening_chamber", "opening_region", (0, 0, 0), (10, 10, 10)
        )
        network = PassageNetworkSpec(
            "opening_route",
            "opening_region",
            (
                PassageJunctionSpec(
                    "opening_start", (0, 0, 0), opening_id="missing_opening"
                ),
                PassageJunctionSpec("opening_end", (4, 0, 0)),
            ),
            (
                _segment(
                    "opening_segment",
                    "opening_start",
                    "opening_end",
                    ((0, 0, 0), (4, 0, 0)),
                ),
            ),
        )

        with self.assertRaisesRegex(
            GeometryValidationError, "unknown_environment_opening"
        ):
            SemanticEnvironmentSpec(
                regions=(region,), chambers=(chamber,), passage_networks=(network,)
            )

    def test_round_trip_is_canonical_and_order_invariant(self) -> None:
        junctions = (
            PassageJunctionSpec("entrance", (0, 0, 0), open_boundary=True),
            PassageJunctionSpec("fork", (8, 0, -1)),
            PassageJunctionSpec("left", (15, 5, -1)),
        )
        main = _segment("main", "entrance", "fork", ((0, 0, 0), (8, 0, -1)))
        left = _segment("left_route", "fork", "left", ((8, 0, -1), (15, 5, -1)))
        first = SemanticEnvironmentSpec(
            regions=(_region(),),
            passage_networks=(
                PassageNetworkSpec("routes", "underground", junctions, (main, left)),
            ),
        )
        reordered = SemanticEnvironmentSpec(
            regions=(_region(),),
            passage_networks=(
                PassageNetworkSpec(
                    "routes",
                    "underground",
                    tuple(reversed(junctions)),
                    (left, main),
                ),
            ),
        )

        self.assertEqual(first.to_dict(), reordered.to_dict())
        self.assertEqual(first.content_hash(), reordered.content_hash())
        self.assertEqual(SemanticEnvironmentSpec.from_dict(first.to_dict()), first)
        json.dumps(first.to_dict())

    def test_branch_cycle_and_dead_end_survive_semantic_topology(self) -> None:
        junctions = tuple(
            PassageJunctionSpec(name, point)
            for name, point in (
                ("a", (0, 0, 0)),
                ("b", (5, 0, 0)),
                ("c", (5, 5, 0)),
                ("d", (10, 0, 0)),
            )
        )
        network = PassageNetworkSpec(
            "network",
            "underground",
            junctions,
            (
                _segment("ab", "a", "b", ((0, 0, 0), (5, 0, 0))),
                _segment("bc", "b", "c", ((5, 0, 0), (5, 5, 0))),
                _segment("ca", "c", "a", ((5, 5, 0), (0, 0, 0))),
                _segment("bd", "b", "d", ((5, 0, 0), (10, 0, 0))),
            ),
        )

        self.assertEqual(network.reachable("a"), frozenset({"a", "b", "c", "d"}))
        self.assertEqual(network.degree("b"), 3)
        self.assertEqual(network.dead_ends, frozenset({"d"}))
        self.assertEqual(network.cycle_rank, 1)

    def test_bound_segment_derives_existing_connector_contract(self) -> None:
        network = PassageNetworkSpec(
            "route",
            "underground",
            (
                PassageJunctionSpec(
                    "entry", (0, 0, 0), space_id="room", level_id="ground"
                ),
                PassageJunctionSpec(
                    "exit", (8, 2, -1), space_id="cavern", level_id="lower"
                ),
            ),
            (
                _segment(
                    "approach", "entry", "exit", ((0, 0, 0), (4, 0, 0), (8, 2, -1))
                ),
            ),
        )

        connector = network.to_connector_spec("approach", connector_id="room_to_cavern")

        self.assertEqual(connector.start.space_id, "room")
        self.assertEqual(connector.end.space_id, "cavern")
        self.assertEqual(connector.parameters["waypoints"], [[4.0, 0.0, 0.0]])
        self.assertTrue(connector.parameters["geometry_embedded"])
        self.assertEqual(connector.width, 4.0)
        self.assertEqual(connector.clearance_height, 3.0)

    def test_chamber_only_subterranean_region_is_valid(self) -> None:
        chamber = CavernChamberSpec(
            chamber_id="hall",
            region_id="underground",
            center=(2, 0, 4),
            size=(20, 12, 8),
            semantic_tags=frozenset({"arena", "lair"}),
        )
        environment = SemanticEnvironmentSpec(regions=(_region(),), chambers=(chamber,))

        self.assertEqual(environment.chambers[0].size, (20.0, 12.0, 8.0))
        self.assertEqual(
            SemanticEnvironmentSpec.from_dict(environment.to_dict()), environment
        )

    def test_openings_details_and_heroes_round_trip_as_compact_semantics(self) -> None:
        chamber = CavernChamberSpec("hall", "underground", (0, 0, 5), (24, 18, 12))
        environment = SemanticEnvironmentSpec(
            regions=(_region(),),
            chambers=(chamber,),
            openings=(
                EnvironmentOpeningSpec(
                    "oculus",
                    "underground",
                    "hall",
                    OpeningTarget.SKY,
                    (0, 0, 10.5),
                    (0, 0, 1),
                    (5, 4),
                    8,
                    weather_exposed=True,
                ),
            ),
            detail_fields=(
                DetailFieldSpec(
                    "ceiling_teeth",
                    "underground",
                    "hall",
                    FormationType.STALACTITE,
                    DetailSurfaceRole.OVERHEAD,
                    24,
                    (0.3, 0.3, 0.8),
                    (1.2, 1.2, 4.0),
                    419,
                ),
            ),
            hero_features=(
                HeroFeatureSpec(
                    "perch",
                    "underground",
                    "hall",
                    HeroFeatureType.ROCK_SPIRE,
                    (2, 1, 1),
                    (3, 3, 4),
                    semantic_tags=frozenset({"landmark", "perch"}),
                ),
            ),
        )

        restored = SemanticEnvironmentSpec.from_dict(environment.to_dict())

        self.assertEqual(restored, environment)
        self.assertTrue(restored.openings[0].sky_exposed)
        self.assertEqual(restored.detail_fields[0].seed, 419)
        self.assertLess(len(json.dumps(environment.to_dict())), 4000)

    def test_rejects_unknown_region_and_mismatched_path_endpoints(self) -> None:
        with self.assertRaisesRegex(GeometryValidationError, "unknown region"):
            SemanticEnvironmentSpec(
                regions=(_region(),),
                chambers=(
                    CavernChamberSpec("orphan", "missing", (0, 0, 0), (2, 2, 2)),
                ),
            )

        junctions = (
            PassageJunctionSpec("a", (0, 0, 0)),
            PassageJunctionSpec("b", (5, 0, 0)),
        )
        with self.assertRaisesRegex(GeometryValidationError, "start junction"):
            PassageNetworkSpec(
                "bad",
                "underground",
                junctions,
                (_segment("ab", "a", "b", ((1, 0, 0), (5, 0, 0))),),
            )

    def test_rejects_invalid_cross_section_order_and_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(GeometryValidationError, "strictly increasing"):
            PassageSegmentSpec(
                "bad",
                "a",
                "b",
                ((0, 0, 0), (2, 0, 0)),
                (
                    PassageCrossSectionSpec(0.5, 2, 2),
                    PassageCrossSectionSpec(0.5, 3, 3),
                ),
            )

        with self.assertRaisesRegex(GeometryValidationError, "duplicate junction"):
            PassageNetworkSpec(
                "bad",
                "underground",
                (
                    PassageJunctionSpec("a", (0, 0, 0)),
                    PassageJunctionSpec("a", (1, 0, 0)),
                ),
                (),
            )

    def test_unknown_fields_fail_instead_of_being_silently_dropped(self) -> None:
        data = SemanticEnvironmentSpec(
            regions=(_region(),),
            chambers=(CavernChamberSpec("hall", "underground", (0, 0, 0), (4, 4, 4)),),
        ).to_dict()
        data["chambers"][0]["ad_hoc_magic"] = True

        with self.assertRaisesRegex(GeometryValidationError, "ad_hoc_magic"):
            SemanticEnvironmentSpec.from_dict(data)

    def test_disjoint_chamber_binding_and_unknown_layout_space_are_rejected(
        self,
    ) -> None:
        network = PassageNetworkSpec(
            "route",
            "underground",
            (
                PassageJunctionSpec("entry", (0, 0, 0)),
                PassageJunctionSpec(
                    "hall",
                    (10, 0, 0),
                    chamber_id="small_chamber",
                    space_id="missing_space",
                    level_id="ground",
                ),
            ),
            (_segment("edge", "entry", "hall", ((0, 0, 0), (10, 0, 0))),),
        )
        with self.assertRaisesRegex(GeometryValidationError, "does not overlap"):
            SemanticEnvironmentSpec(
                regions=(_region(),),
                chambers=(
                    CavernChamberSpec(
                        "small_chamber", "underground", (15, 0, 0), (2, 2, 2)
                    ),
                ),
                passage_networks=(network,),
            )

        bound_environment = SemanticEnvironmentSpec(
            regions=(_region(),),
            passage_networks=(
                PassageNetworkSpec(
                    "bound",
                    "underground",
                    (
                        PassageJunctionSpec(
                            "a",
                            (0, 0, 0),
                            space_id="missing_space",
                            level_id="ground",
                        ),
                        PassageJunctionSpec("b", (5, 0, 0)),
                    ),
                    (_segment("edge", "a", "b", ((0, 0, 0), (5, 0, 0))),),
                ),
            ),
        )
        with self.assertRaisesRegex(GeometryValidationError, "missing_space"):
            bound_environment.validate_layout_bindings(
                space_level_ids={"known": "ground"}, level_ids=["ground"]
            )


if __name__ == "__main__":
    unittest.main()
