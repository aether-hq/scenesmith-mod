"""Portals, connectors, and cross-reference validation."""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    HeightfieldSpec,
    LevelSpec,
    PlatformSpec,
    Transform3D,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    UnknownConnectorEndpointError,
    UnknownLevelError,
    UnsafeConnectorError,
    _finite,
    _point3,
    _require_id,
)


class PortalType(str, Enum):
    DOOR = "door"
    OPEN = "open"
    WINDOW = "window"
    ARCH = "arch"
    CAVE_MOUTH = "cave_mouth"


@dataclass(frozen=True)
class PortalSpec:
    """An aperture connecting two spaces or one space to the exterior."""

    portal_id: str
    portal_type: PortalType
    source_space_id: str
    target_space_id: str | None = None
    transform: Transform3D = field(default_factory=Transform3D)
    width: float = 1.0
    height: float = 2.1
    boundary_loop_index: int | None = None
    boundary_edge_index: int | None = None
    position_along: float | None = None
    sill_height: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "portal_id", _require_id(self.portal_id, "portal_id"))
        object.__setattr__(
            self,
            "source_space_id",
            _require_id(self.source_space_id, "source_space_id"),
        )
        if self.target_space_id is not None:
            object.__setattr__(
                self,
                "target_space_id",
                _require_id(self.target_space_id, "target_space_id"),
            )
        object.__setattr__(self, "portal_type", PortalType(self.portal_type))
        width = _finite(self.width, "width", entity_id=self.portal_id)
        height = _finite(self.height, "height", entity_id=self.portal_id)
        sill_height = _finite(self.sill_height, "sill_height", entity_id=self.portal_id)
        if width <= 0 or height <= 0:
            raise GeometryValidationError(
                "invalid_portal_size",
                f"width and height must be positive; got {width} × {height}",
                entity_id=self.portal_id,
            )
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        if (self.boundary_loop_index is None) != (self.boundary_edge_index is None):
            raise GeometryValidationError(
                "invalid_portal_boundary",
                "boundary_loop_index and boundary_edge_index must be set together",
                entity_id=self.portal_id,
            )
        if self.boundary_loop_index is not None:
            loop_index = int(self.boundary_loop_index)
            edge_index = int(self.boundary_edge_index)
            if loop_index < 0 or edge_index < 0 or self.position_along is None:
                raise GeometryValidationError(
                    "invalid_portal_boundary",
                    "boundary indices and position_along must be non-negative",
                    entity_id=self.portal_id,
                )
            position_along = _finite(
                self.position_along, "position_along", entity_id=self.portal_id
            )
            if position_along < 0:
                raise GeometryValidationError(
                    "invalid_portal_boundary",
                    "position_along must be non-negative",
                    entity_id=self.portal_id,
                )
            object.__setattr__(self, "boundary_loop_index", loop_index)
            object.__setattr__(self, "boundary_edge_index", edge_index)
            object.__setattr__(self, "position_along", position_along)
        object.__setattr__(self, "sill_height", sill_height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.portal_id,
            "type": self.portal_type.value,
            "source_space_id": self.source_space_id,
            "target_space_id": self.target_space_id,
            "transform": self.transform.to_dict(),
            "width": self.width,
            "height": self.height,
            "boundary_loop_index": self.boundary_loop_index,
            "boundary_edge_index": self.boundary_edge_index,
            "position_along": self.position_along,
            "sill_height": self.sill_height,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortalSpec":
        return cls(
            portal_id=data["id"],
            portal_type=PortalType(data["type"]),
            source_space_id=data["source_space_id"],
            target_space_id=data.get("target_space_id"),
            transform=Transform3D.from_dict(data.get("transform")),
            width=data.get("width", 1.0),
            height=data.get("height", 2.1),
            boundary_loop_index=data.get("boundary_loop_index"),
            boundary_edge_index=data.get("boundary_edge_index"),
            position_along=data.get("position_along"),
            sill_height=data.get("sill_height", 0.0),
        )


class ConnectorType(str, Enum):
    STAIRS_STRAIGHT = "stairs_straight"
    STAIRS_L = "stairs_l"
    STAIRS_U = "stairs_u"
    STAIRS_SPIRAL = "stairs_spiral"
    RAMP = "ramp"
    LADDER = "ladder"
    ELEVATOR = "elevator"
    SHAFT = "shaft"
    NATURAL_PASSAGE = "natural_passage"


_AUTO_CONNECTOR_CAPABILITIES = frozenset({"__scenesmith_auto__"})


@dataclass(frozen=True)
class ConnectorEndpoint:
    """One end of a vertical or non-planar connection."""

    space_id: str
    level_id: str
    position: Point3

    def __post_init__(self) -> None:
        object.__setattr__(self, "space_id", _require_id(self.space_id, "space_id"))
        object.__setattr__(self, "level_id", _require_id(self.level_id, "level_id"))
        object.__setattr__(
            self,
            "position",
            _point3(self.position, "position", entity_id=self.space_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "level_id": self.level_id,
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConnectorEndpoint":
        return cls(
            space_id=data["space_id"],
            level_id=data["level_id"],
            position=tuple(data["position"]),
        )


@dataclass(frozen=True)
class ConnectorSpec:
    """Semantic connector between two structural spaces or elevation patches."""

    connector_id: str
    connector_type: ConnectorType
    start: ConnectorEndpoint
    end: ConnectorEndpoint
    width: float = 1.0
    clearance_height: float = 2.1
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required_capabilities: frozenset[str] = field(
        default_factory=lambda: _AUTO_CONNECTOR_CAPABILITIES
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "connector_id", _require_id(self.connector_id, "connector_id")
        )
        connector_type = ConnectorType(self.connector_type)
        width = _finite(self.width, "width", entity_id=self.connector_id)
        clearance = _finite(
            self.clearance_height,
            "clearance_height",
            entity_id=self.connector_id,
        )
        if width <= 0 or clearance <= 0:
            raise UnsafeConnectorError(
                f"width and clearance_height must be positive; got {width} and "
                f"{clearance}",
                entity_id=self.connector_id,
            )
        if math.dist(self.start.position, self.end.position) <= GEOMETRY_TOLERANCE:
            raise UnsafeConnectorError(
                "start and end positions must differ", entity_id=self.connector_id
            )
        raw_capabilities = self.required_capabilities
        if raw_capabilities == _AUTO_CONNECTOR_CAPABILITIES:
            raw_capabilities = (
                frozenset({"climb"})
                if connector_type in {ConnectorType.LADDER, ConnectorType.SHAFT}
                else frozenset({"walk"})
            )
        capabilities = frozenset(
            _require_id(capability, "required capability")
            for capability in raw_capabilities
        )
        if not capabilities:
            raise UnsafeConnectorError(
                "at least one required capability is required",
                entity_id=self.connector_id,
            )
        parameters = dict(self.parameters)
        if "geometry_embedded" in parameters:
            embedded = parameters["geometry_embedded"]
            if not isinstance(embedded, bool):
                raise UnsafeConnectorError(
                    "parameters.geometry_embedded must be a boolean",
                    entity_id=self.connector_id,
                )
            if connector_type not in {
                ConnectorType.NATURAL_PASSAGE,
                ConnectorType.SHAFT,
            }:
                raise UnsafeConnectorError(
                    "parameters.geometry_embedded is only valid for natural_passage "
                    "or shaft connectors",
                    entity_id=self.connector_id,
                )
        if parameters.get("geometry_embedded"):
            waypoints = parameters.get("waypoints", ())
            if not isinstance(waypoints, (list, tuple)):
                raise UnsafeConnectorError(
                    "embedded connector parameters.waypoints must be an array",
                    entity_id=self.connector_id,
                )
            for index, waypoint in enumerate(waypoints):
                _point3(
                    waypoint,
                    f"parameters.waypoints[{index}]",
                    entity_id=self.connector_id,
                )
        object.__setattr__(self, "connector_type", connector_type)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "clearance_height", clearance)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "required_capabilities", capabilities)

    @property
    def rise(self) -> float:
        return self.end.position[2] - self.start.position[2]

    @property
    def horizontal_run(self) -> float:
        return math.dist(self.start.position[:2], self.end.position[:2])

    def validate_straight_access(
        self,
        *,
        max_ramp_slope: float = 1.0 / 12.0,
        min_stair_rise: float = 0.10,
        max_stair_rise: float = 0.20,
        min_stair_run: float = 0.22,
    ) -> None:
        """Validate basic straight-stair/ramp parameters before compilation."""

        rise = abs(self.rise)
        run = self.horizontal_run
        if self.connector_type == ConnectorType.RAMP:
            if run <= GEOMETRY_TOLERANCE:
                raise UnsafeConnectorError(
                    "a ramp must have non-zero horizontal run",
                    entity_id=self.connector_id,
                )
            slope = rise / run
            if slope > max_ramp_slope + GEOMETRY_TOLERANCE:
                raise UnsafeConnectorError(
                    f"ramp slope {slope:.6g} exceeds maximum {max_ramp_slope:.6g}",
                    entity_id=self.connector_id,
                )
        elif self.connector_type == ConnectorType.STAIRS_STRAIGHT:
            riser_count = self.parameters.get("riser_count")
            if not isinstance(riser_count, int) or isinstance(riser_count, bool):
                raise UnsafeConnectorError(
                    "straight stairs require integer parameters.riser_count",
                    entity_id=self.connector_id,
                )
            if riser_count <= 0:
                raise UnsafeConnectorError(
                    "riser_count must be positive", entity_id=self.connector_id
                )
            per_riser = rise / riser_count
            per_tread = run / riser_count
            if not min_stair_rise <= per_riser <= max_stair_rise:
                raise UnsafeConnectorError(
                    f"riser height {per_riser:.6g} is outside "
                    f"[{min_stair_rise}, {max_stair_rise}]",
                    entity_id=self.connector_id,
                )
            if per_tread < min_stair_run:
                raise UnsafeConnectorError(
                    f"tread run {per_tread:.6g} is below minimum {min_stair_run}",
                    entity_id=self.connector_id,
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.connector_id,
            "type": self.connector_type.value,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "width": self.width,
            "clearance_height": self.clearance_height,
            "parameters": dict(self.parameters),
            "required_capabilities": sorted(self.required_capabilities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConnectorSpec":
        connector_type = ConnectorType(data["type"])
        default_capabilities = (
            frozenset({"climb"})
            if connector_type in {ConnectorType.LADDER, ConnectorType.SHAFT}
            else frozenset({"walk"})
        )
        return cls(
            connector_id=data["id"],
            connector_type=connector_type,
            start=ConnectorEndpoint.from_dict(data["start"]),
            end=ConnectorEndpoint.from_dict(data["end"]),
            width=data.get("width", 1.0),
            clearance_height=data.get("clearance_height", 2.1),
            parameters=data.get("parameters", {}),
            required_capabilities=frozenset(
                data.get("required_capabilities", default_capabilities)
            ),
        )


def validate_structural_references(
    *,
    levels: Sequence[LevelSpec],
    space_level_ids: Mapping[str, str],
    connectors: Sequence[ConnectorSpec] = (),
    portals: Sequence[PortalSpec] = (),
    structural_meshes: Sequence[StructuralMeshSpec] = (),
    platforms: Sequence[PlatformSpec] = (),
    heightfields: Sequence[HeightfieldSpec] = (),
) -> None:
    """Validate unique IDs and all level/space references in a semantic layout."""

    level_ids = [level.level_id for level in levels]
    if len(level_ids) != len(set(level_ids)):
        raise GeometryValidationError("duplicate_level", "level IDs must be unique")
    known_levels = set(level_ids)
    if not known_levels:
        raise GeometryValidationError("missing_level", "at least one level is required")

    known_spaces = set(space_level_ids)
    for space_id, level_id in space_level_ids.items():
        _require_id(space_id, "space_id")
        if level_id not in known_levels:
            raise UnknownLevelError(level_id, entity_id=space_id)

    mesh_ids: set[str] = set()
    for mesh in structural_meshes:
        if mesh.mesh_id in mesh_ids:
            raise GeometryValidationError(
                "duplicate_structural_mesh",
                "structural mesh IDs must be unique",
                entity_id=mesh.mesh_id,
            )
        mesh_ids.add(mesh.mesh_id)
        if mesh.space_id not in known_spaces:
            raise GeometryValidationError(
                "unknown_mesh_space",
                f"space '{mesh.space_id}' is not defined",
                entity_id=mesh.mesh_id,
            )

    platform_ids: set[str] = set()
    for platform in platforms:
        if platform.platform_id in platform_ids:
            raise GeometryValidationError(
                "duplicate_platform",
                "platform IDs must be unique",
                entity_id=platform.platform_id,
            )
        platform_ids.add(platform.platform_id)
        if platform.space_id not in known_spaces:
            raise GeometryValidationError(
                "unknown_platform_space",
                f"space '{platform.space_id}' is not defined",
                entity_id=platform.platform_id,
            )

    heightfield_ids: set[str] = set()
    for heightfield in heightfields:
        if heightfield.heightfield_id in heightfield_ids:
            raise GeometryValidationError(
                "duplicate_heightfield",
                "heightfield IDs must be unique",
                entity_id=heightfield.heightfield_id,
            )
        heightfield_ids.add(heightfield.heightfield_id)
        if heightfield.space_id not in known_spaces:
            raise GeometryValidationError(
                "unknown_heightfield_space",
                f"space '{heightfield.space_id}' is not defined",
                entity_id=heightfield.heightfield_id,
            )

    connector_ids: set[str] = set()
    for connector in connectors:
        if connector.connector_id in connector_ids:
            raise GeometryValidationError(
                "duplicate_connector",
                "connector IDs must be unique",
                entity_id=connector.connector_id,
            )
        connector_ids.add(connector.connector_id)
        # One tall semantic space (an atrium, library hall, stage, etc.) may
        # contain several structural datums. Internal stairs therefore retain
        # their authored endpoint levels even though the enclosing room has one
        # canonical topology level. Cross-space connectors remain strict.
        is_internal_multilevel_connector = (
            connector.start.space_id == connector.end.space_id
            and connector.start.level_id != connector.end.level_id
        )
        for endpoint in (connector.start, connector.end):
            if endpoint.space_id not in known_spaces:
                raise UnknownConnectorEndpointError(
                    endpoint.space_id, connector_id=connector.connector_id
                )
            if endpoint.level_id not in known_levels:
                raise UnknownLevelError(
                    endpoint.level_id, entity_id=connector.connector_id
                )
            expected_level = space_level_ids[endpoint.space_id]
            if (
                endpoint.level_id != expected_level
                and not is_internal_multilevel_connector
            ):
                raise GeometryValidationError(
                    "connector_level_mismatch",
                    f"endpoint space '{endpoint.space_id}' belongs to level "
                    f"'{expected_level}', not '{endpoint.level_id}'",
                    entity_id=connector.connector_id,
                )

    portal_ids: set[str] = set()
    for portal in portals:
        if portal.portal_id in portal_ids:
            raise GeometryValidationError(
                "duplicate_portal",
                "portal IDs must be unique",
                entity_id=portal.portal_id,
            )
        portal_ids.add(portal.portal_id)
        for space_id in (portal.source_space_id, portal.target_space_id):
            if space_id is not None and space_id not in known_spaces:
                raise GeometryValidationError(
                    "unknown_portal_endpoint",
                    f"space '{space_id}' is not defined",
                    entity_id=portal.portal_id,
                )
