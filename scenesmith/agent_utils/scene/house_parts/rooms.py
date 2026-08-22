"""House layout and room geometry data structures."""

import logging
import math

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    ElevationProfile,
    Footprint2D,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    PortalSpec,
    PortalType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    InvalidTransformError,
    require_safe_identifier,
)

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.scene.house_parts.openings import (
    ConnectionType,
    OpeningType,
    PlacedRoom,
    WallDirection,
    _finite_xy_position,
)


@dataclass
class RoomSpec:
    """Specification for a single room in a house layout.

    Contains design-level information about a room that can be used by floor
    plan generators to create geometry. This is the input to geometry generation,
    not the output.
    """

    room_id: str
    """Unique identifier for the room (e.g., 'main', 'living_room', 'bedroom_1')."""

    room_type: str = "room"
    """Type of room (e.g., 'living_room', 'bedroom', 'kitchen', 'bathroom')."""

    prompt: str = ""
    """Text description/prompt for this room."""

    position: tuple[float, float] = (0.0, 0.0)
    """Position of room origin in house coordinates (x, y)."""

    width: float = 5.0
    """Room width in meters (y-dimension)."""

    length: float = 5.0
    """Room length in meters (x-dimension)."""

    connections: dict[str, ConnectionType] = field(default_factory=dict)
    """Room connections: maps room_id to ConnectionType (DOOR or OPEN)."""

    exterior_walls: set[WallDirection] = field(default_factory=set)
    """Walls that MUST remain exterior (no rooms can be placed adjacent to them).

    Use this for rooms like hallways, receptions, or lobbies that need
    external door access on specific walls. The placement algorithm will
    create clearance zones around these walls to prevent other rooms from
    blocking them.
    """

    level_id: str = "ground"
    """Level grouping for v2 layouts. Missing v1 values migrate to ground."""

    elevation: float | None = None
    """Optional absolute floor elevation override; otherwise use the level datum."""

    yaw: float = 0.0
    """Room yaw in radians. Legacy rectangular placement uses zero."""

    footprint: Footprint2D | None = None
    """Optional arbitrary footprint in local coordinates."""

    floor_footprint: Footprint2D | None = None
    """Optional floor slab footprint; use holes for stairs/atria openings."""

    ceiling_footprint: Footprint2D | None = None
    """Optional ceiling slab footprint, independent of floor openings."""

    floor_profile: ElevationProfile = field(default_factory=ElevationProfile)
    """Floor elevation surface in room-local coordinates."""

    ceiling_profile: ElevationProfile | None = None
    """Optional ceiling profile; None uses the layout/level nominal height."""

    has_overhead_cover: bool = True
    """Whether the space has a roof/ceiling outside authored ceiling holes."""

    def __post_init__(self) -> None:
        self.room_id = require_safe_identifier(self.room_id, "room_id")
        self.position = _finite_xy_position(self.position, entity_id=self.room_id)
        self.level_id = self.level_id.strip()
        if not self.level_id:
            raise ValueError("RoomSpec.level_id must not be empty")
        if self.elevation is not None and not math.isfinite(self.elevation):
            raise InvalidTransformError(
                f"elevation must be finite; got {self.elevation!r}",
                entity_id=self.room_id,
            )
        if not math.isfinite(self.yaw):
            raise InvalidTransformError(
                f"yaw must be finite; got {self.yaw!r}", entity_id=self.room_id
            )
        boundary = self.footprint or Footprint2D.rectangle(self.length, self.width)
        for label, slab_footprint in (
            ("floor", self.floor_footprint),
            ("ceiling", self.ceiling_footprint),
        ):
            if slab_footprint is not None and slab_footprint.outer != boundary.outer:
                raise GeometryValidationError(
                    "slab_boundary_mismatch",
                    f"{label}_footprint outer loop must match room '{self.room_id}' "
                    "boundary; add holes for slab openings",
                    entity_id=self.room_id,
                )

    def to_dict(self) -> dict:
        """Serialize room spec to dictionary."""
        return {
            "id": self.room_id,
            "type": self.room_type,
            "position": list(self.position),
            "width": self.width,
            "length": self.length,
            "prompt": self.prompt,
            "connections": {k: v.value for k, v in self.connections.items()},
            "exterior_walls": [w.value for w in self.exterior_walls],
            "level_id": self.level_id,
            "elevation": self.elevation,
            "yaw": self.yaw,
            "footprint": self.footprint.to_dict() if self.footprint else None,
            "floor_footprint": (
                self.floor_footprint.to_dict() if self.floor_footprint else None
            ),
            "ceiling_footprint": (
                self.ceiling_footprint.to_dict() if self.ceiling_footprint else None
            ),
            "floor_profile": self.floor_profile.to_dict(),
            "ceiling_profile": (
                self.ceiling_profile.to_dict() if self.ceiling_profile else None
            ),
            "has_overhead_cover": self.has_overhead_cover,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoomSpec":
        """Deserialize room spec from dictionary."""
        connections = {}
        if "connections" in data:
            connections = {k: ConnectionType(v) for k, v in data["connections"].items()}
        exterior_walls: set[WallDirection] = set()
        if "exterior_walls" in data:
            exterior_walls = {WallDirection(w) for w in data["exterior_walls"]}
        return cls(
            room_id=data["id"],
            room_type=data.get("type", "room"),
            prompt=data.get("prompt", ""),
            position=tuple(data.get("position", [0.0, 0.0])),
            width=data.get("width", 5.0),
            length=data.get("length", 6.0),
            connections=connections,
            exterior_walls=exterior_walls,
            level_id=data.get("level_id", "ground"),
            elevation=data.get("elevation"),
            yaw=data.get("yaw", 0.0),
            footprint=(
                Footprint2D.from_dict(data["footprint"])
                if data.get("footprint")
                else None
            ),
            floor_footprint=(
                Footprint2D.from_dict(data["floor_footprint"])
                if data.get("floor_footprint")
                else None
            ),
            ceiling_footprint=(
                Footprint2D.from_dict(data["ceiling_footprint"])
                if data.get("ceiling_footprint")
                else None
            ),
            floor_profile=ElevationProfile.from_dict(data.get("floor_profile")),
            ceiling_profile=(
                ElevationProfile.from_dict(data["ceiling_profile"])
                if data.get("ceiling_profile")
                else None
            ),
            has_overhead_cover=bool(data.get("has_overhead_cover", True)),
        )


def legacy_openings_to_boundary_portals(
    room_spec: RoomSpec, placed_room: PlacedRoom, wall_height: float
) -> list[PortalSpec]:
    """Map rectangular cardinal openings to v2 boundary-local apertures."""

    rectangle = Footprint2D.rectangle(room_spec.length, room_spec.width)
    boundary = room_spec.footprint or rectangle
    if boundary.outer != rectangle.outer:
        if any(wall.openings for wall in placed_room.walls):
            raise ValueError(
                f"Room '{room_spec.room_id}' has cardinal legacy openings on an "
                "arbitrary footprint. Author them as boundary-local PortalSpec entries."
            )
        return []
    edge_map = {
        WallDirection.SOUTH: (0, False),
        WallDirection.EAST: (1, False),
        WallDirection.NORTH: (2, True),
        WallDirection.WEST: (3, True),
    }
    portal_type_map = {
        OpeningType.DOOR: PortalType.DOOR,
        OpeningType.WINDOW: PortalType.WINDOW,
        OpeningType.OPEN: PortalType.OPEN,
    }
    portals: list[PortalSpec] = []
    for wall in placed_room.walls:
        edge_index, reverse = edge_map[wall.direction]
        for opening in wall.openings:
            center = opening.position_along_wall + opening.width / 2.0
            if reverse:
                center = wall.length - center
            portals.append(
                PortalSpec(
                    portal_id=opening.opening_id,
                    portal_type=portal_type_map[opening.opening_type],
                    source_space_id=room_spec.room_id,
                    width=opening.width,
                    height=(
                        wall_height - opening.sill_height
                        if opening.opening_type == OpeningType.OPEN
                        else opening.height
                    ),
                    boundary_loop_index=0,
                    boundary_edge_index=edge_index,
                    position_along=center,
                    sill_height=opening.sill_height,
                )
            )
    return portals
