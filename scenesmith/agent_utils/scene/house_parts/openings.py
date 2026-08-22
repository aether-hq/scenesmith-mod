"""House layout and room geometry data structures."""

import hashlib
import json
import logging
import math

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from scenesmith.agent_utils.structure.geometry_models.surface_models import Footprint2D
from scenesmith.agent_utils.structure.geometry_models.validation import (
    InvalidTransformError,
    require_safe_identifier,
)
from scenesmith.utils.geometry.material import Material

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject

console_logger = logging.getLogger(__name__)


def _finite_xy_position(value: Any, *, entity_id: str) -> tuple[float, float]:
    """Normalize a room min-corner position with a typed diagnostic."""

    try:
        coordinates = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidTransformError(
            f"position must contain two finite coordinates; got {value!r}",
            entity_id=entity_id,
        ) from exc
    if len(coordinates) != 2 or not all(math.isfinite(v) for v in coordinates):
        raise InvalidTransformError(
            f"position must contain two finite coordinates; got {value!r}",
            entity_id=entity_id,
        )
    return coordinates


def compute_wall_normals(walls: list["SceneObject"]) -> dict[str, np.ndarray]:
    """Compute room-facing normals for wall objects.

    Normals point from wall center toward room center (0, 0) in the XY plane,
    creating vectors that indicate the "inside" direction of each wall. These
    are used for snap-to-wall orientation calculations.

    Args:
        walls: List of wall SceneObjects.

    Returns:
        Dict mapping wall name to normalized 2D normal vector (X, Y).
    """
    # Room center is at origin (0, 0) for rectangular rooms.
    room_center = np.array([0.0, 0.0])

    wall_normals = {}
    for wall in walls:
        # Wall center position in XY plane.
        wall_center_2d = wall.transform.translation()[:2]

        # Normal points from wall toward room center.
        normal_2d = room_center - wall_center_2d

        # Normalize to unit vector.
        normal_length = np.linalg.norm(normal_2d)
        if normal_length > 1e-6:
            normal_2d = normal_2d / normal_length
        else:
            console_logger.warning(
                f"Wall {wall.name} is at room center, cannot compute normal"
            )
            normal_2d = np.array([0.0, 0.0])

        wall_normals[wall.name] = normal_2d

    return wall_normals


class WallDirection(Enum):
    """Cardinal direction for room walls."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    def get_inward_normal(self) -> tuple[float, float]:
        """Get unit normal vector pointing INTO the room.

        Returns:
            (nx, ny) unit vector pointing into room interior.
        """
        if self == WallDirection.NORTH:
            return (0.0, -1.0)
        elif self == WallDirection.SOUTH:
            return (0.0, 1.0)
        elif self == WallDirection.EAST:
            return (-1.0, 0.0)
        else:  # WEST
            return (1.0, 0.0)


class OpeningType(Enum):
    """Type of wall opening."""

    DOOR = "door"
    WINDOW = "window"
    OPEN = "open"  # Open floor plan connection (no wall, floor-to-ceiling).


class WindowShape(Enum):
    """Supported window and wall-cutout silhouettes."""

    RECTANGULAR = "rectangular"
    ARCHED = "arched"


@dataclass
class Opening:
    """Opening (door/window/open connection) in a wall.

    Stored in Wall.openings list. Created when Door, Window, or open connection
    is added. For OPEN type, height is ignored at render time - uses wall_height.
    """

    opening_id: str
    """References Door.id or Window.id."""

    opening_type: OpeningType
    """Type of opening (door or window)."""

    position_along_wall: float
    """Meters from wall start_point."""

    width: float
    """Width of opening in meters."""

    height: float
    """Height of opening in meters."""

    sill_height: float = 0.0

    shape: WindowShape = WindowShape.RECTANGULAR

    def to_dict(self) -> dict:
        """Serialize opening to dictionary."""
        return {
            "opening_id": self.opening_id,
            "opening_type": self.opening_type.value,
            "position_along_wall": self.position_along_wall,
            "width": self.width,
            "height": self.height,
            "sill_height": self.sill_height,
            "shape": self.shape.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Opening":
        """Deserialize opening from dictionary."""
        return cls(
            opening_id=data["opening_id"],
            opening_type=OpeningType(data["opening_type"]),
            position_along_wall=data["position_along_wall"],
            width=data["width"],
            height=data["height"],
            sill_height=data.get("sill_height", 0.0),
            shape=WindowShape(data.get("shape", WindowShape.RECTANGULAR.value)),
        )

    """Height from floor to bottom (0 for doors, >0 for windows)."""


@dataclass
class ClearanceOpeningData:
    """Opening data for clearance zone physics checks and label rendering.

    This extends the basic Opening data with computed world-space coordinates
    and clearance zone bounds for physics validation.
    """

    opening_id: str
    """Unique identifier from source Opening."""

    opening_type: str
    """Type: 'door', 'window', or 'open'."""

    wall_direction: str
    """Cardinal direction: 'north', 'south', 'east', 'west'."""

    center_world: list[float]
    """World coordinates [x, y, z] for label positioning."""

    width: float
    """Opening width in meters."""

    sill_height: float
    """Height from floor to bottom (0 for doors, >0 for windows)."""

    height: float
    """Height of opening in meters."""

    clearance_bbox_min: list[float] | None
    """Clearance zone AABB minimum [x, y, z], or None for OPEN type."""

    clearance_bbox_max: list[float] | None
    """Clearance zone AABB maximum [x, y, z], or None for OPEN type."""

    wall_start: list[float]
    """Wall start point [x, y] for open connection sweep."""

    wall_end: list[float]
    """Wall end point [x, y] for open connection sweep."""

    position_along_wall: float
    """Distance from wall start to opening center."""

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "opening_id": self.opening_id,
            "opening_type": self.opening_type,
            "wall_direction": self.wall_direction,
            "center_world": self.center_world,
            "width": self.width,
            "sill_height": self.sill_height,
            "height": self.height,
            "clearance_bbox_min": self.clearance_bbox_min,
            "clearance_bbox_max": self.clearance_bbox_max,
            "wall_start": self.wall_start,
            "wall_end": self.wall_end,
            "position_along_wall": self.position_along_wall,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClearanceOpeningData":
        """Deserialize from dictionary."""
        return cls(
            opening_id=data["opening_id"],
            opening_type=data["opening_type"],
            wall_direction=data["wall_direction"],
            center_world=data["center_world"],
            width=data["width"],
            sill_height=data.get("sill_height", 0.0),
            height=data["height"],
            clearance_bbox_min=data.get("clearance_bbox_min"),
            clearance_bbox_max=data.get("clearance_bbox_max"),
            wall_start=data["wall_start"],
            wall_end=data["wall_end"],
            position_along_wall=data["position_along_wall"],
        )


@dataclass
class Door:
    """Door in house layout.

    Interior doors create openings in both rooms' walls.
    """

    id: str
    """Unique door identifier."""

    boundary_label: str
    """ASCII label from HouseLayout.boundary_labels (e.g., 'A')."""

    position_segment: str
    """'left', 'center', or 'right'."""

    position_exact: float
    """Computed meters from boundary start (randomized within segment)."""

    door_type: str
    """'exterior' or 'interior'."""

    room_a: str
    """First room (or exterior-facing room for exterior doors)."""

    room_b: str | None = None
    """Second room (None if exterior door)."""

    width: float = 1.0
    """Designer-chosen within config range (0.9-1.9m)."""

    height: float = 2.1
    """Designer-chosen within config range (2.0-2.4m)."""

    def to_dict(self) -> dict:
        """Serialize door to dictionary."""
        return {
            "id": self.id,
            "boundary_label": self.boundary_label,
            "position_segment": self.position_segment,
            "position_exact": self.position_exact,
            "door_type": self.door_type,
            "room_a": self.room_a,
            "room_b": self.room_b,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Door":
        """Deserialize door from dictionary."""
        return cls(
            id=data["id"],
            boundary_label=data["boundary_label"],
            position_segment=data["position_segment"],
            position_exact=data["position_exact"],
            door_type=data["door_type"],
            room_a=data["room_a"],
            room_b=data.get("room_b"),
            width=data.get("width", 1.0),
            height=data.get("height", 2.1),
        )


@dataclass
class Window:
    """Window in house layout. Only on exterior walls."""

    id: str
    """Unique window identifier."""

    boundary_label: str
    """ASCII label from HouseLayout.boundary_labels (for agent display only)."""

    position_along_wall: float
    """Meters from wall start (left edge)."""

    room_id: str
    """Room this window belongs to."""

    wall_direction: WallDirection | None = None
    """Stable wall identifier (used instead of boundary_label for lookup)."""

    width: float = 1.2
    """Designer-chosen: small (0.6m) to picture window (3.0m)."""

    height: float = 1.2
    """Designer-chosen: small (0.6m) to floor-to-ceiling (2.0m)."""

    sill_height: float = 0.9
    """Designer-chosen: height from floor (typically 0.9m)."""

    shape: WindowShape = WindowShape.RECTANGULAR
    """Visible frame and wall-opening silhouette."""

    def to_dict(self) -> dict:
        """Serialize window to dictionary."""
        return {
            "id": self.id,
            "boundary_label": self.boundary_label,
            "position_along_wall": self.position_along_wall,
            "room_id": self.room_id,
            "wall_direction": (
                self.wall_direction.value if self.wall_direction else None
            ),
            "width": self.width,
            "height": self.height,
            "sill_height": self.sill_height,
            "shape": self.shape.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Window":
        """Deserialize window from dictionary."""
        dir_str = data.get("wall_direction")
        return cls(
            id=data["id"],
            boundary_label=data["boundary_label"],
            position_along_wall=data["position_along_wall"],
            room_id=data["room_id"],
            wall_direction=WallDirection(dir_str) if dir_str else None,
            width=data.get("width", 1.2),
            height=data.get("height", 1.2),
            sill_height=data.get("sill_height", 0.9),
            shape=WindowShape(data.get("shape", WindowShape.RECTANGULAR.value)),
        )


@dataclass
class RoomMaterials:
    """Materials for a room's surfaces."""

    wall_material: Material | None = None
    """Wall material with PBR textures."""

    floor_material: Material | None = None
    """Floor material with PBR textures."""

    def to_dict(self) -> dict:
        """Serialize room materials to dictionary."""
        return {
            "wall_material": (
                self.wall_material.to_dict() if self.wall_material else None
            ),
            "floor_material": (
                self.floor_material.to_dict() if self.floor_material else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoomMaterials":
        """Deserialize room materials from dictionary."""
        return cls(
            wall_material=(
                Material.from_dict(data["wall_material"])
                if data.get("wall_material")
                else None
            ),
            floor_material=(
                Material.from_dict(data["floor_material"])
                if data.get("floor_material")
                else None
            ),
        )


@dataclass
class Wall:
    """A room's wall in one direction.

    Each room has exactly 4 walls (N/S/E/W). Wall geometry is computed from
    room position + dimensions. A wall can face multiple rooms in T-junction
    layouts. This preserves 4-wall-per-room structure for rendering logic.
    """

    wall_id: str
    """Format: '{room_id}_{direction}' e.g., 'living_room_north'."""

    room_id: str
    """Room that owns this wall."""

    direction: WallDirection
    """Cardinal direction of this wall."""

    start_point: tuple[float, float]
    """(x, y) start of wall segment in global coordinates."""

    end_point: tuple[float, float]
    """(x, y) end of wall segment in global coordinates."""

    length: float
    """Wall length (computable from points but stored for convenience)."""

    is_exterior: bool = True
    """True if wall faces outside the house."""

    faces_rooms: list[str] = field(default_factory=list)
    """Room IDs on other side (empty if exterior, can be >1 for T-junctions)."""

    openings: list[Opening] = field(default_factory=list)
    """Openings (doors/windows) in this wall."""

    def cache_key(self, wall_height: float, material: Material | None = None) -> str:
        """Generate cache key for wall GLTF caching.

        The cache key includes all properties that affect the wall's geometry
        and appearance. If any of these change, the wall GLTF must be regenerated.

        Args:
            wall_height: Wall height in meters (from layout).
            material: Wall material (affects texture).

        Returns:
            SHA-256 hash string (first 16 chars) for cache lookup.
        """
        # Build state dict with all rendering-relevant properties.
        state = {
            "wall_id": self.wall_id,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "direction": self.direction.value,
            "is_exterior": self.is_exterior,
            "wall_height": wall_height,
            "material": str(material.path) if material else None,
            "openings": [o.to_dict() for o in self.openings],
        }
        content_json = json.dumps(state, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize wall to dictionary."""
        return {
            "wall_id": self.wall_id,
            "room_id": self.room_id,
            "direction": self.direction.value,
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "length": self.length,
            "is_exterior": self.is_exterior,
            "faces_rooms": self.faces_rooms,
            "openings": [opening.to_dict() for opening in self.openings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Wall":
        """Deserialize wall from dictionary."""
        return cls(
            wall_id=data["wall_id"],
            room_id=data["room_id"],
            direction=WallDirection(data["direction"]),
            start_point=tuple(data["start_point"]),
            end_point=tuple(data["end_point"]),
            length=data["length"],
            is_exterior=data.get("is_exterior", True),
            faces_rooms=data.get("faces_rooms", []),
            openings=[Opening.from_dict(o) for o in data.get("openings", [])],
        )


@dataclass
class PlacedRoom:
    """Room with computed position (derived from RoomSpec).

    Regenerated when spec changes via placement algorithm.
    """

    room_id: str
    """Unique identifier matching RoomSpec.room_id."""

    position: tuple[float, float]
    """(x, y) min corner in global coordinates."""

    width: float
    """Room width in meters."""

    depth: float
    """Room depth in meters."""

    walls: list[Wall] = field(default_factory=list)
    """Exactly 4 walls: N, S, E, W."""

    level_id: str = "ground"
    """Level grouping for topology and editing (v2; defaults to v1 ground)."""

    elevation: float | None = None
    """Optional absolute world Z; otherwise inherited from the assigned level."""

    yaw: float = 0.0
    """Room-local yaw in radians around +Z."""

    footprint: Footprint2D | None = None
    """Optional arbitrary local footprint; None retains the legacy rectangle."""

    def __post_init__(self) -> None:
        self.room_id = require_safe_identifier(self.room_id, "room_id")
        self.position = _finite_xy_position(self.position, entity_id=self.room_id)
        self.level_id = self.level_id.strip()
        if not self.level_id:
            raise ValueError("PlacedRoom.level_id must not be empty")
        if self.elevation is not None and not math.isfinite(self.elevation):
            raise InvalidTransformError(
                f"elevation must be finite; got {self.elevation!r}",
                entity_id=self.room_id,
            )
        if not math.isfinite(self.yaw):
            raise InvalidTransformError(
                f"yaw must be finite; got {self.yaw!r}", entity_id=self.room_id
            )

    def to_dict(self) -> dict:
        """Serialize placed room to dictionary."""
        return {
            "room_id": self.room_id,
            "position": list(self.position),
            "width": self.width,
            "depth": self.depth,
            "walls": [wall.to_dict() for wall in self.walls],
            "level_id": self.level_id,
            "elevation": self.elevation,
            "yaw": self.yaw,
            "footprint": self.footprint.to_dict() if self.footprint else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlacedRoom":
        """Deserialize placed room from dictionary."""
        return cls(
            room_id=data["room_id"],
            position=tuple(data["position"]),
            width=data["width"],
            depth=data["depth"],
            walls=[Wall.from_dict(w) for w in data.get("walls", [])],
            level_id=data.get("level_id", "ground"),
            elevation=data.get("elevation"),
            yaw=data.get("yaw", 0.0),
            footprint=(
                Footprint2D.from_dict(data["footprint"])
                if data.get("footprint")
                else None
            ),
        )


class ConnectionType(str, Enum):
    """Type of connection between adjacent rooms."""

    DOOR = "DOOR"
    """Rooms share a wall with a door opening."""

    OPEN = "OPEN"
    """Rooms have no wall between them (open floor plan)."""
