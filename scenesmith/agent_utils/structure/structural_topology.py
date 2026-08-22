"""Capability-aware connectivity for multilevel and freeform structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorSpec,
    PortalSpec,
    PortalType,
)

EXTERIOR_NODE = "__exterior__"


@dataclass(frozen=True)
class TopologyEdge:
    """Undirected semantic connection with an agent capability policy."""

    edge_id: str
    source: str
    target: str
    kind: str
    required_capabilities: frozenset[str]

    def is_available_to(self, capabilities: Iterable[str]) -> bool:
        return self.required_capabilities.issubset(frozenset(capabilities))


@dataclass(frozen=True)
class StructuralTopology:
    """Small deterministic graph used before navigation meshes exist."""

    nodes: frozenset[str]
    edges: tuple[TopologyEdge, ...]

    @classmethod
    def build(
        cls,
        *,
        space_ids: Iterable[str],
        portals: Sequence[PortalSpec] = (),
        connectors: Sequence[ConnectorSpec] = (),
    ) -> "StructuralTopology":
        nodes = set(space_ids)
        edges: list[TopologyEdge] = []
        for portal in portals:
            target = portal.target_space_id or EXTERIOR_NODE
            nodes.add(target)
            capabilities = (
                frozenset({"view"})
                if portal.portal_type == PortalType.WINDOW
                else frozenset({"walk"})
            )
            edges.append(
                TopologyEdge(
                    edge_id=portal.portal_id,
                    source=portal.source_space_id,
                    target=target,
                    kind=f"portal:{portal.portal_type.value}",
                    required_capabilities=capabilities,
                )
            )
        for connector in connectors:
            edges.append(
                TopologyEdge(
                    edge_id=connector.connector_id,
                    source=connector.start.space_id,
                    target=connector.end.space_id,
                    kind=f"connector:{connector.connector_type.value}",
                    required_capabilities=connector.required_capabilities,
                )
            )
        return cls(nodes=frozenset(nodes), edges=tuple(edges))

    def reachable(
        self,
        start: str,
        *,
        capabilities: Iterable[str] = ("walk",),
        blocked_edges: Iterable[str] = (),
    ) -> frozenset[str]:
        """Return nodes reachable by an agent under semantic/geometric policy."""

        if start not in self.nodes:
            raise KeyError(f"unknown topology node '{start}'")
        blocked = frozenset(blocked_edges)
        available = frozenset(capabilities)
        visited = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge.edge_id in blocked or not edge.is_available_to(available):
                    continue
                neighbor = None
                if edge.source == current:
                    neighbor = edge.target
                elif edge.target == current:
                    neighbor = edge.source
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return frozenset(visited)

    def connected_components(
        self,
        *,
        capabilities: Iterable[str] = ("walk",),
        blocked_edges: Iterable[str] = (),
    ) -> tuple[frozenset[str], ...]:
        """Return stable connected components for a capability class."""

        remaining = set(self.nodes)
        components: list[frozenset[str]] = []
        while remaining:
            start = min(remaining)
            component = self.reachable(
                start,
                capabilities=capabilities,
                blocked_edges=blocked_edges,
            )
            components.append(component)
            remaining.difference_update(component)
        return tuple(components)

    def topology_geometry_mismatches(
        self, geometrically_blocked_edges: Iterable[str]
    ) -> tuple[TopologyEdge, ...]:
        """Return semantic connections that clearance/collision marked blocked."""

        blocked = frozenset(geometrically_blocked_edges)
        return tuple(edge for edge in self.edges if edge.edge_id in blocked)
