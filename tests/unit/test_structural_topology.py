"""Tests for capability-aware structural topology."""

import unittest

from scenesmith.agent_utils.structural_geometry import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
    PortalSpec,
    PortalType,
)
from scenesmith.agent_utils.structural_topology import EXTERIOR_NODE, StructuralTopology


class TestStructuralTopology(unittest.TestCase):
    def test_stacked_spaces_remain_disconnected_without_connector(self) -> None:
        topology = StructuralTopology.build(space_ids={"lower", "upper"})

        self.assertEqual(topology.reachable("lower"), frozenset({"lower"}))
        self.assertEqual(len(topology.connected_components()), 2)

    def test_stairs_connect_levels_for_walkers(self) -> None:
        stairs = ConnectorSpec(
            connector_id="stairs",
            connector_type=ConnectorType.STAIRS_STRAIGHT,
            start=ConnectorEndpoint("lower", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("upper", "upper", (4, 0, 3)),
            parameters={"riser_count": 18},
        )
        topology = StructuralTopology.build(
            space_ids={"lower", "upper"}, connectors=[stairs]
        )

        self.assertEqual(topology.reachable("lower"), frozenset({"lower", "upper"}))
        self.assertEqual(
            topology.reachable("lower", blocked_edges={"stairs"}),
            frozenset({"lower"}),
        )
        self.assertEqual(
            topology.topology_geometry_mismatches({"stairs"}), (topology.edges[0],)
        )

    def test_ladder_requires_climb_capability(self) -> None:
        ladder = ConnectorSpec(
            connector_id="ladder",
            connector_type=ConnectorType.LADDER,
            start=ConnectorEndpoint("pit", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("ledge", "upper", (0, 0, 3)),
            required_capabilities=frozenset({"climb"}),
        )
        topology = StructuralTopology.build(
            space_ids={"pit", "ledge"}, connectors=[ladder]
        )

        self.assertEqual(topology.reachable("pit"), frozenset({"pit"}))
        self.assertEqual(
            topology.reachable("pit", capabilities={"walk", "climb"}),
            frozenset({"pit", "ledge"}),
        )

    def test_window_is_view_connection_not_walk_connection(self) -> None:
        window = PortalSpec(
            portal_id="window",
            portal_type=PortalType.WINDOW,
            source_space_id="room",
        )
        topology = StructuralTopology.build(space_ids={"room"}, portals=[window])

        self.assertNotIn(EXTERIOR_NODE, topology.reachable("room"))
        self.assertIn(EXTERIOR_NODE, topology.reachable("room", capabilities={"view"}))


if __name__ == "__main__":
    unittest.main()
