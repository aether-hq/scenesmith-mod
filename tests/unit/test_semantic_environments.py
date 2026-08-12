"""Semantic-model and genericity tests for natural 3D environments."""

import json
import unittest

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
