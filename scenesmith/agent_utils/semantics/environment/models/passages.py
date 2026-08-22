"""Passage cross-sections, junctions, segments, and networks."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any, Mapping

from scenesmith.agent_utils.semantics.environment.models.common import (
    _REFERENCE_TOLERANCE,
    PassageFloorMode,
    PassageProfile,
    _finite,
    _identifier,
    _point,
    _reject_unknown_fields,
    _strict_bool,
    _strict_int,
    _unique_ids,
)
from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)


@dataclass(frozen=True)
class PassageCrossSectionSpec:
    """Width and height at one normalized station along a passage path."""

    station: float
    width: float
    height: float

    def __post_init__(self) -> None:
        station = _finite(self.station, "station")
        width = _finite(self.width, "width")
        height = _finite(self.height, "height")
        if not 0.0 <= station <= 1.0:
            raise GeometryValidationError(
                "invalid_cross_section", "station must be inside [0, 1]"
            )
        if width <= 0.0 or height <= 0.0:
            raise GeometryValidationError(
                "invalid_cross_section", "width and height must be positive"
            )
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def to_dict(self) -> dict[str, float]:
        return {"station": self.station, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageCrossSectionSpec":
        _reject_unknown_fields(data, {"station", "width", "height"})
        return cls(data["station"], data["width"], data["height"])


@dataclass(frozen=True)
class PassageJunctionSpec:
    """A graph node at a chamber, opening, bound space, or free 3D position."""

    junction_id: str
    position: Point3
    chamber_id: str | None = None
    opening_id: str | None = None
    space_id: str | None = None
    level_id: str | None = None
    open_boundary: bool = False
    semantic_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        junction_id = _identifier(self.junction_id, "junction_id")
        object.__setattr__(self, "junction_id", junction_id)
        object.__setattr__(
            self, "position", _point(self.position, "position", entity_id=junction_id)
        )
        for attribute in ("chamber_id", "opening_id", "space_id", "level_id"):
            value = getattr(self, attribute)
            if value is not None:
                object.__setattr__(self, attribute, _identifier(value, attribute))
        if (self.space_id is None) != (self.level_id is None):
            raise GeometryValidationError(
                "incomplete_junction_binding",
                "space_id and level_id must be provided together",
                entity_id=junction_id,
            )
        object.__setattr__(
            self,
            "open_boundary",
            _strict_bool(self.open_boundary, "open_boundary", entity_id=junction_id),
        )
        object.__setattr__(
            self,
            "semantic_tags",
            frozenset(
                str(tag).strip() for tag in self.semantic_tags if str(tag).strip()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.junction_id,
            "position": list(self.position),
            "chamber_id": self.chamber_id,
            "opening_id": self.opening_id,
            "space_id": self.space_id,
            "level_id": self.level_id,
            "open_boundary": self.open_boundary,
            "semantic_tags": sorted(self.semantic_tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageJunctionSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "position",
                "chamber_id",
                "opening_id",
                "space_id",
                "level_id",
                "open_boundary",
                "semantic_tags",
            },
            entity_id=data.get("id"),
        )
        return cls(
            junction_id=data["id"],
            position=tuple(data["position"]),
            chamber_id=data.get("chamber_id"),
            opening_id=data.get("opening_id"),
            space_id=data.get("space_id"),
            level_id=data.get("level_id"),
            open_boundary=data.get("open_boundary", False),
            semantic_tags=frozenset(data.get("semantic_tags", [])),
        )


@dataclass(frozen=True)
class PassageSegmentSpec:
    """A variable-profile semantic edge between two passage junctions."""

    segment_id: str
    start_junction_id: str
    end_junction_id: str
    path: tuple[Point3, ...]
    cross_sections: tuple[PassageCrossSectionSpec, ...]
    profile: PassageProfile = PassageProfile.ELLIPSE
    floor_mode: PassageFloorMode = PassageFloorMode.NATURAL
    capabilities: frozenset[str] = frozenset({"walk"})
    roughness_seed: int | None = None

    def __post_init__(self) -> None:
        segment_id = _identifier(self.segment_id, "segment_id")
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(
            self,
            "start_junction_id",
            _identifier(self.start_junction_id, "start_junction_id"),
        )
        object.__setattr__(
            self,
            "end_junction_id",
            _identifier(self.end_junction_id, "end_junction_id"),
        )
        if self.start_junction_id == self.end_junction_id:
            raise GeometryValidationError(
                "invalid_passage_segment",
                "start and end junctions must differ",
                entity_id=segment_id,
            )
        path = tuple(
            _point(item, f"path[{index}]", entity_id=segment_id)
            for index, item in enumerate(self.path)
        )
        if len(path) < 2:
            raise GeometryValidationError(
                "invalid_passage_path",
                "path requires at least two points",
                entity_id=segment_id,
            )
        for index, (start, end) in enumerate(zip(path, path[1:])):
            if math.dist(start, end) <= GEOMETRY_TOLERANCE:
                raise GeometryValidationError(
                    "invalid_passage_path",
                    f"path span {index} has zero length",
                    entity_id=segment_id,
                )
        sections = tuple(self.cross_sections)
        if len(sections) < 2:
            raise GeometryValidationError(
                "invalid_cross_section",
                "at least two cross-sections are required",
                entity_id=segment_id,
            )
        stations = [section.station for section in sections]
        if any(second <= first for first, second in zip(stations, stations[1:])):
            raise GeometryValidationError(
                "invalid_cross_section",
                "cross-section stations must be strictly increasing",
                entity_id=segment_id,
            )
        if (
            abs(stations[0]) > _REFERENCE_TOLERANCE
            or abs(stations[-1] - 1.0) > _REFERENCE_TOLERANCE
        ):
            raise GeometryValidationError(
                "invalid_cross_section",
                "cross-sections must include stations 0 and 1",
                entity_id=segment_id,
            )
        capabilities = frozenset(
            str(item).strip() for item in self.capabilities if str(item).strip()
        )
        allowed_capabilities = {"walk", "crawl", "climb", "swim", "fly"}
        if not capabilities or not capabilities <= allowed_capabilities:
            raise GeometryValidationError(
                "invalid_capabilities",
                "capabilities must use walk, crawl, climb, swim, or fly",
                entity_id=segment_id,
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "cross_sections", sections)
        object.__setattr__(self, "profile", PassageProfile(self.profile))
        object.__setattr__(self, "floor_mode", PassageFloorMode(self.floor_mode))
        object.__setattr__(self, "capabilities", capabilities)
        if self.roughness_seed is not None:
            object.__setattr__(
                self,
                "roughness_seed",
                _strict_int(
                    self.roughness_seed, "roughness_seed", entity_id=segment_id
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.segment_id,
            "start_junction_id": self.start_junction_id,
            "end_junction_id": self.end_junction_id,
            "path": [list(point) for point in self.path],
            "cross_sections": [section.to_dict() for section in self.cross_sections],
            "profile": self.profile.value,
            "floor_mode": self.floor_mode.value,
            "capabilities": sorted(self.capabilities),
            "roughness_seed": self.roughness_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageSegmentSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "start_junction_id",
                "end_junction_id",
                "path",
                "cross_sections",
                "profile",
                "floor_mode",
                "capabilities",
                "roughness_seed",
            },
            entity_id=data.get("id"),
        )
        return cls(
            segment_id=data["id"],
            start_junction_id=data["start_junction_id"],
            end_junction_id=data["end_junction_id"],
            path=tuple(tuple(point) for point in data["path"]),
            cross_sections=tuple(
                PassageCrossSectionSpec.from_dict(item)
                for item in data["cross_sections"]
            ),
            profile=PassageProfile(data.get("profile", "ellipse")),
            floor_mode=PassageFloorMode(data.get("floor_mode", "natural")),
            capabilities=frozenset(data.get("capabilities", ["walk"])),
            roughness_seed=data.get("roughness_seed"),
        )


@dataclass(frozen=True)
class PassageNetworkSpec:
    """A topology-preserving graph of passage junctions and segments."""

    network_id: str
    region_id: str
    junctions: tuple[PassageJunctionSpec, ...]
    segments: tuple[PassageSegmentSpec, ...]

    def __post_init__(self) -> None:
        network_id = _identifier(self.network_id, "network_id")
        object.__setattr__(self, "network_id", network_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        junctions = tuple(sorted(self.junctions, key=lambda item: item.junction_id))
        segments = tuple(sorted(self.segments, key=lambda item: item.segment_id))
        _unique_ids(junctions, "junction_id", "junction")
        _unique_ids(segments, "segment_id", "passage_segment")
        junction_by_id = {item.junction_id: item for item in junctions}
        for segment in segments:
            if segment.start_junction_id not in junction_by_id:
                raise GeometryValidationError(
                    "unknown_passage_junction",
                    f"unknown start junction '{segment.start_junction_id}'",
                    entity_id=segment.segment_id,
                )
            if segment.end_junction_id not in junction_by_id:
                raise GeometryValidationError(
                    "unknown_passage_junction",
                    f"unknown end junction '{segment.end_junction_id}'",
                    entity_id=segment.segment_id,
                )
            start = junction_by_id[segment.start_junction_id].position
            end = junction_by_id[segment.end_junction_id].position
            if math.dist(segment.path[0], start) > _REFERENCE_TOLERANCE:
                raise GeometryValidationError(
                    "passage_endpoint_mismatch",
                    "path start does not match its start junction position",
                    entity_id=segment.segment_id,
                )
            if math.dist(segment.path[-1], end) > _REFERENCE_TOLERANCE:
                raise GeometryValidationError(
                    "passage_endpoint_mismatch",
                    "path end does not match its end junction position",
                    entity_id=segment.segment_id,
                )
        object.__setattr__(self, "junctions", junctions)
        object.__setattr__(self, "segments", segments)

    def degree(self, junction_id: str) -> int:
        if junction_id not in {item.junction_id for item in self.junctions}:
            raise KeyError(junction_id)
        return sum(
            junction_id in {segment.start_junction_id, segment.end_junction_id}
            for segment in self.segments
        )

    def reachable(self, start_junction_id: str) -> frozenset[str]:
        known = {item.junction_id for item in self.junctions}
        if start_junction_id not in known:
            raise KeyError(start_junction_id)
        adjacency: dict[str, set[str]] = {identifier: set() for identifier in known}
        for segment in self.segments:
            adjacency[segment.start_junction_id].add(segment.end_junction_id)
            adjacency[segment.end_junction_id].add(segment.start_junction_id)
        visited: set[str] = set()
        pending = [start_junction_id]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(sorted(adjacency[current] - visited, reverse=True))
        return frozenset(visited)

    def to_connector_spec(
        self, segment_id: str, *, connector_id: str | None = None
    ) -> ConnectorSpec:
        """Adapt one bound passage edge to SceneSmith's shared topology contract."""

        segment_by_id = {item.segment_id: item for item in self.segments}
        if segment_id not in segment_by_id:
            raise KeyError(segment_id)
        segment = segment_by_id[segment_id]
        junction_by_id = {item.junction_id: item for item in self.junctions}
        start = junction_by_id[segment.start_junction_id]
        end = junction_by_id[segment.end_junction_id]
        if any(
            value is None
            for value in (start.space_id, start.level_id, end.space_id, end.level_id)
        ):
            raise GeometryValidationError(
                "unbound_passage_connector",
                "connector endpoints require junction space_id and level_id bindings",
                entity_id=segment.segment_id,
            )
        return ConnectorSpec(
            connector_id=connector_id or segment.segment_id,
            connector_type=ConnectorType.NATURAL_PASSAGE,
            start=ConnectorEndpoint(
                start.space_id, start.level_id, start.position  # type: ignore[arg-type]
            ),
            end=ConnectorEndpoint(
                end.space_id, end.level_id, end.position  # type: ignore[arg-type]
            ),
            width=min(item.width for item in segment.cross_sections),
            clearance_height=min(item.height for item in segment.cross_sections),
            required_capabilities=segment.capabilities,
            parameters={
                "geometry_embedded": True,
                "waypoints": [list(point) for point in segment.path[1:-1]],
            },
        )

    @property
    def dead_ends(self) -> frozenset[str]:
        return frozenset(
            junction.junction_id
            for junction in self.junctions
            if self.degree(junction.junction_id) == 1
        )

    @property
    def cycle_rank(self) -> int:
        if not self.junctions:
            return 0
        unseen = {item.junction_id for item in self.junctions}
        components = 0
        while unseen:
            start = min(unseen)
            reached = set(self.reachable(start))
            unseen -= reached
            components += 1
        return max(0, len(self.segments) - len(self.junctions) + components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.network_id,
            "region_id": self.region_id,
            "junctions": [item.to_dict() for item in self.junctions],
            "segments": [item.to_dict() for item in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageNetworkSpec":
        _reject_unknown_fields(
            data,
            {"id", "region_id", "junctions", "segments"},
            entity_id=data.get("id"),
        )
        return cls(
            network_id=data["id"],
            region_id=data["region_id"],
            junctions=tuple(
                PassageJunctionSpec.from_dict(item)
                for item in data.get("junctions", [])
            ),
            segments=tuple(
                PassageSegmentSpec.from_dict(item) for item in data.get("segments", [])
            ),
        )
