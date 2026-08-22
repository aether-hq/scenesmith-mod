"""End-to-end proofs for the single derived semantic scene contract."""

import tempfile
import unittest

from pathlib import Path

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.semantics.environment.models.chambers import (
    Bounds3D,
    CavernChamberSpec,
    EnvironmentRegionSpec,
)
from scenesmith.agent_utils.semantics.environment.models.common import (
    DetailCollisionPolicy,
    EnvironmentKind,
    HeroFeatureType,
    OpeningTarget,
)
from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.semantics.environment.models.features import (
    EnvironmentOpeningSpec,
    HeroFeatureSpec,
)
from scenesmith.agent_utils.semantics.environment.models.passages import (
    PassageCrossSectionSpec,
    PassageJunctionSpec,
    PassageNetworkSpec,
    PassageSegmentSpec,
)
from scenesmith.agent_utils.semantics.publication.scene_contract import (
    derive_scene_contract,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    LevelSpec,
    SurfaceRole,
)
from scenesmith.agent_utils.structure.structural_topology import EXTERIOR_NODE


def _layout(collision_policy: DetailCollisionPolicy) -> HouseLayout:
    region = EnvironmentRegionSpec(
        "contract_region",
        EnvironmentKind.SUBTERRANEAN,
        Bounds3D((-30, -20, -20), (30, 20, 30)),
    )
    chamber = CavernChamberSpec(
        "contract_chamber", "contract_region", (5, 0, 4), (20, 16, 12)
    )
    environment = SemanticEnvironmentSpec(
        regions=(region,),
        chambers=(chamber,),
        passage_networks=(
            PassageNetworkSpec(
                "contract_routes",
                "contract_region",
                (
                    PassageJunctionSpec(
                        "contract_entry",
                        (-12, 0, 0),
                        space_id="entry_room",
                        level_id="ground",
                        open_boundary=True,
                    ),
                    PassageJunctionSpec(
                        "contract_destination",
                        (0, 0, 0),
                        chamber_id="contract_chamber",
                        space_id="destination_room",
                        level_id="ground",
                    ),
                ),
                (
                    PassageSegmentSpec(
                        "contract_passage",
                        "contract_entry",
                        "contract_destination",
                        ((-12, 0, 0), (0, 0, 0)),
                        (
                            PassageCrossSectionSpec(0, 4, 4),
                            PassageCrossSectionSpec(1, 5, 5),
                        ),
                    ),
                ),
            ),
        ),
        openings=(
            EnvironmentOpeningSpec(
                "contract_skylight",
                "contract_region",
                "contract_chamber",
                OpeningTarget.SKY,
                (5, 0, 9.5),
                (0, 0, 1),
                (4, 4),
                8,
                passable=True,
            ),
        ),
        hero_features=(
            HeroFeatureSpec(
                "route_boulder",
                "contract_region",
                "contract_chamber",
                HeroFeatureType.BOULDER,
                (0, 0, 0),
                (2, 2, 3),
                collision_policy=collision_policy,
            ),
        ),
    )
    return HouseLayout(
        levels=[LevelSpec("ground", 0, 6)],
        room_specs=[RoomSpec("entry_room"), RoomSpec("destination_room")],
        semantic_environment=environment,
    )


class TestDerivedSceneContract(unittest.TestCase):
    def test_one_semantic_source_derives_topology_surfaces_and_runtime_products(
        self,
    ) -> None:
        layout = _layout(DetailCollisionPolicy.COARSE)
        with tempfile.TemporaryDirectory() as temporary_directory:
            contract = layout.derive_scene_contract(
                Path(temporary_directory), voxel_size=1.0
            )

            edge_ids = {edge.edge_id for edge in contract.topology.edges}
            self.assertIn("contract_passage", edge_ids)
            self.assertIn("contract_skylight", edge_ids)
            self.assertIn(
                EXTERIOR_NODE,
                contract.topology.reachable("entry_room", capabilities={"walk"}),
            )
            self.assertIn("contract_passage", contract.blocked_edge_ids)
            self.assertNotIn(
                "destination_room",
                contract.topology.reachable(
                    "entry_room",
                    capabilities={"walk"},
                    blocked_edges=contract.blocked_edge_ids,
                ),
            )
            self.assertIn("route_boulder", contract.collision_source_ids)
            boulder_queries = tuple(
                query
                for query in contract.surface_index.queries
                if query.patch.surface.source_id == "route_boulder"
            )
            self.assertTrue(boulder_queries)
            self.assertTrue(
                all(
                    query.patch.surface.geometry_ref.startswith("collision_triangle:")
                    for query in boulder_queries
                )
            )
            self.assertTrue(contract.surface_index.by_role(SurfaceRole.BOUNDARY))
            self.assertEqual(
                {product.source_kind for product in contract.products},
                {"semantic_shell", "semantic_detail"},
            )
            for product in contract.products:
                product.artifact.verify()

    def test_visual_only_detail_is_not_part_of_collision_query_contract(self) -> None:
        layout = _layout(DetailCollisionPolicy.VISUAL_ONLY)
        with tempfile.TemporaryDirectory() as temporary_directory:
            contract = derive_scene_contract(
                layout, Path(temporary_directory), voxel_size=1.0
            )

            self.assertNotIn("route_boulder", contract.collision_source_ids)
            self.assertNotIn("contract_passage", contract.blocked_edge_ids)

    def test_semantic_topology_is_not_published_before_physical_compilation(
        self,
    ) -> None:
        layout = _layout(DetailCollisionPolicy.FULL)

        edge_ids = {edge.edge_id for edge in layout.build_topology().edges}

        self.assertNotIn("contract_passage", edge_ids)
        self.assertNotIn("contract_skylight", edge_ids)


if __name__ == "__main__":
    unittest.main()
