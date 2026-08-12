"""House layout and room geometry data structures."""

import hashlib
import json
import logging
import math
import os
import time
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omegaconf import DictConfig

from scenesmith.agent_utils.semantic_environments import SemanticEnvironmentSpec
from scenesmith.agent_utils.structural_geometry import (
    SCHEMA_VERSION,
    ConnectorSpec,
    ConnectorType,
    ElevationProfile,
    Footprint2D,
    GeometryValidationError,
    HeightfieldSpec,
    InvalidTransformError,
    LevelSpec,
    PlatformSpec,
    PortalSpec,
    PortalType,
    StructuralMeshSpec,
    StructuralSurface,
    SurfaceRole,
    Transform3D,
    default_ground_level,
    validate_structural_references,
)
from scenesmith.utils.material import Material
from scenesmith.utils.package_utils import create_package_xml
from scenesmith.utils.path_utils import safe_relative_path

if TYPE_CHECKING:
    from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject
    from scenesmith.agent_utils.structural_compiler import CompiledStructurePaths

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

    def to_dict(self) -> dict:
        """Serialize opening to dictionary."""
        return {
            "opening_id": self.opening_id,
            "opening_type": self.opening_type.value,
            "position_along_wall": self.position_along_wall,
            "width": self.width,
            "height": self.height,
            "sill_height": self.sill_height,
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

    def __post_init__(self) -> None:
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


@dataclass
class RoomGeometry:
    """Generated 3D geometry for a single room.

    Contains the physical structural elements (walls, floor) and their SDFormat
    representations. This is the the actual 3D geometry that gets loaded into Drake for
    simulation.

    Contrast with RoomSpec which is the design input (dimensions, position).
    """

    sdf_tree: ET.ElementTree
    """The SDF tree of the full room geometry."""

    sdf_path: Path
    """Path to the SDF file containing floor, walls, doors, windows, etc."""

    walls: list["SceneObject"] = field(default_factory=list)
    """Wall objects (immutable architectural elements)."""

    floor: "SceneObject | None" = None
    """Floor object (immutable architectural element, optional placement surface)."""

    wall_normals: dict[str, np.ndarray] = field(default_factory=dict)
    """Pre-computed room-facing normals for walls.

    Key: wall name (e.g., "north_wall", "south_wall", "east_wall", "west_wall")
    Value: 2D normalized normal vector (X, Y) pointing from wall center toward room center
    """

    width: float = 0.0
    """Room width in meters (y-dimension)."""

    length: float = 0.0
    """Room length in meters (x-dimension)."""

    wall_height: float = 2.5
    """Wall height in meters (needed for wall height violation check)."""

    wall_thickness: float = 0.05
    """Wall thickness in meters (needed for wall surface offset from room boundary)."""

    openings: list["ClearanceOpeningData"] = field(default_factory=list)
    """All door/window/open openings with physics and rendering data."""

    footprint: Footprint2D | None = None
    """Compiled local footprint, when geometry is not a legacy rectangle."""

    floor_footprint: Footprint2D | None = None
    """Compiled floor slab footprint, including local openings."""

    ceiling_footprint: Footprint2D | None = None
    """Compiled ceiling slab footprint, including local openings."""

    floor_profile: ElevationProfile = field(default_factory=ElevationProfile)
    """Compiled floor elevation profile."""

    ceiling_profile: ElevationProfile | None = None
    """Compiled ceiling profile, if explicitly authored."""

    structural_surfaces: list[StructuralSurface] = field(default_factory=list)
    """First-class support, attachment, overhead, and traversable patches."""

    structural_surface_path: Path | None = None
    """Sidecar containing explicit surface boundaries, normals, and mesh groups."""

    additional_structural_surface_paths: list[Path] = field(default_factory=list)
    """Room-local platform, terrain, or freeform surface sidecars."""

    def content_hash(self) -> str:
        """Generate content hash for this floor plan."""
        floor_plan_dict = {
            "sdf_path": str(self.sdf_path) if self.sdf_path else "",
        }

        # Hash SDF file content.
        sdf_path_str = floor_plan_dict["sdf_path"]
        if sdf_path_str:
            try:
                path = Path(sdf_path_str)
                if path.exists():
                    # SDF files are XML-based text files, so read as UTF-8.
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    floor_plan_dict["sdf_path_content_hash"] = hashlib.sha256(
                        content.encode()
                    ).hexdigest()
                else:
                    floor_plan_dict["sdf_path_content_hash"] = ""
            except Exception as e:
                console_logger.warning(
                    f"Could not hash file content for {sdf_path_str}: {e}"
                )
                floor_plan_dict["sdf_path_content_hash"] = ""

        # Add walls and floor content hashes.
        floor_plan_dict["walls"] = [wall.content_hash() for wall in self.walls]
        floor_plan_dict["floor"] = self.floor.content_hash() if self.floor else None
        floor_plan_dict["footprint"] = (
            self.footprint.to_dict() if self.footprint is not None else None
        )
        floor_plan_dict["floor_footprint"] = (
            self.floor_footprint.to_dict() if self.floor_footprint is not None else None
        )
        floor_plan_dict["ceiling_footprint"] = (
            self.ceiling_footprint.to_dict()
            if self.ceiling_footprint is not None
            else None
        )
        floor_plan_dict["floor_profile"] = self.floor_profile.to_dict()
        floor_plan_dict["ceiling_profile"] = (
            self.ceiling_profile.to_dict() if self.ceiling_profile is not None else None
        )
        floor_plan_dict["structural_surfaces"] = [
            surface.to_dict() for surface in self.structural_surfaces
        ]
        sidecar_paths = [
            path
            for path in (
                self.structural_surface_path,
                *self.additional_structural_surface_paths,
            )
            if path is not None
        ]
        floor_plan_dict["structural_sidecars"] = []
        for path in sidecar_paths:
            path = Path(path)
            floor_plan_dict["structural_sidecars"].append(
                {
                    "path": str(path),
                    "content_hash": (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        if path.exists()
                        else ""
                    ),
                }
            )

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(floor_plan_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_dict(self, scene_dir: Path | None = None) -> dict[str, Any]:
        """Serialize RoomGeometry to dictionary.

        Args:
            scene_dir: Optional scene directory for path relativization.
                       If None, paths are stored as absolute paths.

        Returns:
            Dictionary containing floor plan state (excluding sdf_tree which
            will be re-parsed from file).
        """
        # Convert paths (relative or absolute).
        sdf_path_str = (
            safe_relative_path(self.sdf_path, scene_dir) if self.sdf_path else None
        )

        # Serialize floor if present.
        floor_data = None
        if self.floor:
            floor_data = self.floor.to_dict(scene_dir=scene_dir)

        # Serialize walls.
        walls_data = [w.to_dict(scene_dir=scene_dir) for w in self.walls]

        # Serialize wall_normals (convert numpy arrays to lists).
        wall_normals_data = {}
        for wall_name, normal_vec in self.wall_normals.items():
            wall_normals_data[wall_name] = normal_vec.tolist()

        return {
            "sdf_path": sdf_path_str,
            "walls": walls_data,
            "width": self.width,
            "length": self.length,
            "wall_height": self.wall_height,
            "wall_thickness": self.wall_thickness,
            "openings": [o.to_dict() for o in self.openings],
            "floor": floor_data,
            "wall_normals": wall_normals_data,
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
            "structural_surfaces": [
                surface.to_dict() for surface in self.structural_surfaces
            ],
            "structural_surface_path": (
                safe_relative_path(self.structural_surface_path, scene_dir)
                if self.structural_surface_path
                else None
            ),
            "additional_structural_surface_paths": [
                safe_relative_path(path, scene_dir)
                for path in self.additional_structural_surface_paths
            ],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], scene_dir: Path | None = None
    ) -> "RoomGeometry":
        """Deserialize RoomGeometry from dictionary.

        Args:
            data: Dictionary containing floor plan state.
            scene_dir: Optional scene directory for path resolution.

        Returns:
            RoomGeometry instance reconstructed from dictionary.

        Raises:
            ValueError: If SDF file is missing (fail-fast for research codebase).

        Note:
            sdf_tree is re-parsed from the sdf_path file.
        """
        # Import here to avoid circular import.
        from scenesmith.agent_utils.room import SceneObject

        # Resolve paths relative to scene_dir.
        sdf_path = None
        if data["sdf_path"]:
            sdf_path = (
                scene_dir / data["sdf_path"] if scene_dir else Path(data["sdf_path"])
            )

        # Re-parse sdf_tree from file.
        if not sdf_path or not sdf_path.exists():
            raise ValueError(
                f"SDF file not found at {sdf_path}. Floor plan cannot be restored "
                "without SDF file."
            )
        sdf_tree = ET.parse(sdf_path)

        # Restore floor if present.
        floor = None
        if data.get("floor"):
            floor = SceneObject.from_dict(data["floor"], scene_dir=scene_dir)

        # Restore walls if present.
        walls = []
        if "walls" in data:
            walls = [
                SceneObject.from_dict(w, scene_dir=scene_dir) for w in data["walls"]
            ]

        # Restore wall_normals (convert lists back to numpy arrays).
        wall_normals = {}
        if "wall_normals" in data:
            for wall_name, normal_list in data["wall_normals"].items():
                wall_normals[wall_name] = np.array(normal_list)

        return cls(
            sdf_tree=sdf_tree,
            sdf_path=sdf_path,
            walls=walls,
            floor=floor,
            wall_normals=wall_normals,
            width=data["width"],
            length=data["length"],
            wall_height=data.get("wall_height", 2.5),
            wall_thickness=data.get("wall_thickness", 0.05),
            openings=[
                ClearanceOpeningData.from_dict(o) for o in data.get("openings", [])
            ],
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
            structural_surfaces=[
                StructuralSurface.from_dict(surface)
                for surface in data.get("structural_surfaces", [])
            ],
            structural_surface_path=(
                scene_dir / data["structural_surface_path"]
                if scene_dir and data.get("structural_surface_path")
                else (
                    Path(data["structural_surface_path"])
                    if data.get("structural_surface_path")
                    else None
                )
            ),
            additional_structural_surface_paths=[
                scene_dir / path if scene_dir is not None else Path(path)
                for path in data.get("additional_structural_surface_paths", [])
            ],
        )


@dataclass
class HouseLayout:
    """Layout specification for a house with one or more rooms.

    HouseLayout is the unified data structure for both room mode (single room)
    and house mode (multiple rooms). Room mode is simply a HouseLayout with
    one room. This eliminates separate code paths for the two modes.

    The floor plan generator receives a HouseLayout and populates the
    room_geometries dict with generated geometry for each room. Following stage agents
    don't interact with HouseLayout directly - they receive RoomScene instances
    with RoomGeometry.
    """

    schema_version: int = SCHEMA_VERSION
    """Serialized semantic schema version (v1 inputs migrate to v2)."""

    wall_height: float = 2.5
    """Wall height in meters (default 2.5m, agent can override via set_wall_height)."""

    house_prompt: str = ""
    """Original user prompt for the house/room."""

    room_specs: list[RoomSpec] = field(default_factory=list)
    """Specifications for each room in the house."""

    levels: list[LevelSpec] = field(default_factory=lambda: [default_ground_level()])
    """Vertical datums. Legacy layouts contain one zero-elevation ground level."""

    connectors: list[ConnectorSpec] = field(default_factory=list)
    """Stairs, ramps, ladders, lifts, shafts, and natural passages."""

    connector_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled connector SDF paths keyed by connector ID."""

    structural_meshes: list[StructuralMeshSpec] = field(default_factory=list)
    """Imported/freeform structural meshes, including cavern chambers/tunnels."""

    structural_mesh_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled structural-mesh SDF paths keyed by mesh ID."""

    semantic_environment: SemanticEnvironmentSpec | None = None
    """LLM-authored chambers and passage graphs compiled as navigable voids."""

    semantic_environment_geometry_path: Path | None = None
    """Compiled SDF path for the semantic environment shell."""

    platforms: list[PlatformSpec] = field(default_factory=list)
    """Raised/sunken platforms, mezzanines, balconies, bridges, and catwalks."""

    platform_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled platform SDF paths keyed by platform ID."""

    heightfields: list[HeightfieldSpec] = field(default_factory=list)
    """Sampled terrain and organic floor surfaces."""

    heightfield_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled heightfield SDF paths keyed by heightfield ID."""

    portals: list[PortalSpec] = field(default_factory=list)
    """General apertures; legacy doors/windows remain available during migration."""

    room_geometries: dict[str, RoomGeometry] = field(default_factory=dict)
    """Generated room geometry for each room (room_id -> RoomGeometry)."""

    house_dir: Path | None = None
    """Directory for house-level outputs."""

    # Placed rooms (derived from specs via placement algorithm).
    placed_rooms: list[PlacedRoom] = field(default_factory=list)
    """Rooms with computed positions after placement algorithm."""

    # Doors and windows.
    doors: list[Door] = field(default_factory=list)
    """All doors in the house."""

    windows: list[Window] = field(default_factory=list)
    """All windows in the house."""

    # Materials per room (interior walls + floors).
    room_materials: dict[str, RoomMaterials] = field(default_factory=dict)
    """Materials for each room (room_id -> RoomMaterials)."""

    # Exterior shell material (consistent for entire house).
    exterior_material: Material | None = None
    """Exterior material (brick, siding, etc.) with PBR textures."""

    # Validation state.
    placement_valid: bool = False
    """True if room placement satisfies all constraints."""

    connectivity_valid: bool = False
    """True if all rooms are reachable from exterior via doors."""

    # ASCII boundary labels (generated dynamically).
    boundary_labels: dict[str, tuple[str, str | None, str | None]] = field(
        default_factory=dict
    )
    """Maps label (A, B, C...) to (room_a, room_b, direction).

    For interior walls: (room_a, room_b, None) - direction not needed.
    For exterior walls: (room_a, None, direction) - direction is wall facing (north, south, etc).
    """

    def __post_init__(self) -> None:
        """Normalize v2 defaults and create package metadata when needed."""
        if not self.levels:
            self.levels = [default_ground_level()]
        if self.schema_version not in (1, SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported house layout schema_version={self.schema_version}; "
                f"supported versions are 1 and {SCHEMA_VERSION}"
            )
        if self.house_dir is not None:
            package_xml_path = self.house_dir / "package.xml"
            if not package_xml_path.exists():
                self.house_dir.mkdir(parents=True, exist_ok=True)
                create_package_xml(self.house_dir)
                console_logger.debug(
                    f"Created package.xml at {package_xml_path} for scene portability"
                )

    def get_level(self, level_id: str) -> LevelSpec | None:
        """Get a structural level by stable ID."""
        for level in self.levels:
            if level.level_id == level_id:
                return level
        return None

    def get_room_elevation(self, room_id: str) -> float:
        """Resolve a room's absolute floor Z using placement/spec/level precedence."""
        placed_room = self.get_placed_room(room_id)
        if placed_room is not None and placed_room.elevation is not None:
            return placed_room.elevation
        room_spec = self.get_room_spec(room_id)
        if room_spec is not None:
            if room_spec.elevation is not None:
                return room_spec.elevation
            level = self.get_level(room_spec.level_id)
            if level is not None:
                return level.elevation
        return 0.0

    def validate_structure(self) -> None:
        """Validate v2 level, room, portal, and connector references."""
        space_level_ids = {spec.room_id: spec.level_id for spec in self.room_specs}
        validate_structural_references(
            levels=self.levels,
            space_level_ids=space_level_ids,
            connectors=self.connectors,
            portals=self.portals,
            structural_meshes=self.structural_meshes,
            platforms=self.platforms,
            heightfields=self.heightfields,
        )
        if self.semantic_environment is not None:
            self.semantic_environment.validate_layout_bindings(
                space_level_ids=space_level_ids,
                level_ids=[level.level_id for level in self.levels],
            )
        replacement_spaces = [
            mesh.space_id for mesh in self.structural_meshes if mesh.replaces_room_shell
        ]
        duplicate_spaces = sorted(
            space_id
            for space_id in set(replacement_spaces)
            if replacement_spaces.count(space_id) > 1
        )
        if duplicate_spaces:
            raise GeometryValidationError(
                "duplicate_room_shell",
                "only one structural mesh may replace each room shell; duplicates: "
                + ", ".join(duplicate_spaces),
            )
        replacement_floor_spaces = [
            heightfield.space_id
            for heightfield in self.heightfields
            if heightfield.replaces_floor
        ]
        duplicate_floor_spaces = sorted(
            space_id
            for space_id in set(replacement_floor_spaces)
            if replacement_floor_spaces.count(space_id) > 1
        )
        if duplicate_floor_spaces:
            raise GeometryValidationError(
                "duplicate_floor_replacement",
                "only one heightfield may replace each room floor; duplicates: "
                + ", ".join(duplicate_floor_spaces),
            )

    def compile_semantic_environment(
        self,
        output_dir: Path,
        *,
        voxel_size: float = 0.5,
        max_cells: int = 2_000_000,
        max_triangles: int = 500_000,
        structure_id: str = "semantic_environment",
    ) -> "CompiledStructurePaths":
        """Compile authored chambers and passage networks into one SDF shell."""

        if self.semantic_environment is None:
            raise ValueError("No semantic environment has been defined.")
        from scenesmith.agent_utils.semantic_environment_compiler import (
            SemanticCompileOptions,
            compile_semantic_environment,
        )
        from scenesmith.agent_utils.structural_compiler import write_compiled_structure

        compiled = compile_semantic_environment(
            self.semantic_environment,
            options=SemanticCompileOptions(
                voxel_size=voxel_size,
                max_cells=max_cells,
                max_triangles=max_triangles,
                structure_id=structure_id,
            ),
        )
        paths = write_compiled_structure(compiled, output_dir)
        self.semantic_environment_geometry_path = paths.sdf_path
        return paths

    def build_topology(self):
        """Build the capability-aware semantic topology for this layout."""

        from scenesmith.agent_utils.structural_topology import StructuralTopology

        self.validate_structure()
        return StructuralTopology.build(
            space_ids=self.room_ids,
            portals=self.portals,
            connectors=self.connectors,
        )

    def build_structural_surface_index(self, *, include_connectors: bool = True):
        """Build one house-frame query index from all compiled room surfaces."""

        from scenesmith.agent_utils.structural_surfaces import (
            StructuralSurfaceIndex,
            load_surface_patches,
            transform_surface_patches,
        )

        patches = []
        for room_id, geometry in self.room_geometries.items():
            placed = self.get_placed_room(room_id)
            if placed is None:
                continue
            paths = {
                Path(path)
                for path in (
                    geometry.structural_surface_path,
                    *geometry.additional_structural_surface_paths,
                )
                if path is not None and Path(path).exists()
            }
            room_transform = Transform3D(
                translation=(
                    placed.position[0] + placed.width / 2.0,
                    placed.position[1] + placed.depth / 2.0,
                    self.get_room_elevation(room_id),
                ),
                rotation_rpy=(0.0, 0.0, placed.yaw),
            )
            for path in sorted(paths, key=str):
                patches.extend(
                    transform_surface_patches(
                        load_surface_patches(path), room_transform
                    )
                )
        if include_connectors:
            for connector_id, sdf_path in sorted(self.connector_geometry_paths.items()):
                surface_path = Path(sdf_path).with_suffix(".surfaces.json")
                if not surface_path.exists():
                    raise ValueError(
                        f"compiled connector '{connector_id}' is missing surface "
                        f"sidecar {surface_path}"
                    )
                patches.extend(load_surface_patches(surface_path))
            if self.semantic_environment_geometry_path is not None:
                surface_path = self.semantic_environment_geometry_path.with_suffix(
                    ".surfaces.json"
                )
                if not surface_path.exists():
                    raise ValueError(
                        "compiled semantic environment is missing surface sidecar "
                        f"{surface_path}"
                    )
                patches.extend(load_surface_patches(surface_path))
        return StructuralSurfaceIndex(patches)

    @staticmethod
    def _connector_centerline_samples(
        connector: ConnectorSpec, *, sample_spacing: float
    ) -> tuple[tuple[float, float, float], ...]:
        """Sample a connector centerline in the house structural frame."""

        if not math.isfinite(sample_spacing) or sample_spacing <= 0:
            raise ValueError("sample_spacing must be finite and positive")
        if connector.connector_type.value == "stairs_spiral":
            center = connector.parameters.get("center")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError(
                    f"spiral connector '{connector.connector_id}' has no center"
                )
            low, high = (
                (connector.start.position, connector.end.position)
                if connector.start.position[2] <= connector.end.position[2]
                else (connector.end.position, connector.start.position)
            )
            center_x, center_y = float(center[0]), float(center[1])
            radius = math.hypot(low[0] - center_x, low[1] - center_y)
            turns = float(connector.parameters.get("turns", 1.0))
            direction = (
                1.0
                if str(connector.parameters.get("direction", "ccw")).lower() == "ccw"
                else -1.0
            )
            total_angle = direction * math.tau * turns
            start_angle = math.atan2(low[1] - center_y, low[0] - center_x)
            length = math.hypot(radius * total_angle, high[2] - low[2])
            count = max(1, math.ceil(length / sample_spacing))
            return tuple(
                (
                    center_x + radius * math.cos(start_angle + total_angle * i / count),
                    center_y + radius * math.sin(start_angle + total_angle * i / count),
                    low[2] + (high[2] - low[2]) * i / count,
                )
                for i in range(count + 1)
            )

        raw_waypoints = connector.parameters.get("waypoints", ())
        waypoints = tuple(
            tuple(float(value) for value in point) for point in raw_waypoints
        )
        points = (connector.start.position, *waypoints, connector.end.position)
        samples: list[tuple[float, float, float]] = []
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            length = math.dist(start, end)
            count = max(1, math.ceil(length / sample_spacing))
            samples.extend(
                tuple(
                    start[axis] + (end[axis] - start[axis]) * i / count
                    for axis in range(3)
                )
                for i in range(0 if segment_index == 0 else 1, count + 1)
            )
        return tuple(samples)

    def geometrically_blocked_connectors(
        self,
        *,
        capabilities: tuple[str, ...] = ("walk",),
        agent_height: float = 1.8,
        agent_radius: float = 0.25,
        max_step_height: float = 0.3,
        sample_spacing: float = 0.15,
    ) -> frozenset[str]:
        """Veto semantically walkable connectors that fail local clearance."""

        missing_geometry = [
            connector.connector_id
            for connector in self.connectors
            if not self._connector_geometry_is_embedded(connector)
            and connector.connector_id not in self.connector_geometry_paths
        ]
        if missing_geometry:
            raise ValueError("compile_connectors() before checking route clearance")
        index = self.build_structural_surface_index(include_connectors=True)
        available = frozenset(capabilities)
        blocked: set[str] = set()
        for connector in self.connectors:
            if not connector.required_capabilities.issubset(available):
                continue
            # Climb-only ladders use a different body model; this local support
            # sampler is intentionally restricted to walkable connectors.
            if "walk" not in connector.required_capabilities:
                continue
            if connector.width / 2.0 + 1e-9 < agent_radius:
                blocked.add(connector.connector_id)
                continue
            for x, y, z in self._connector_centerline_samples(
                connector, sample_spacing=sample_spacing
            ):
                clearance = index.clearance_at(
                    x,
                    y,
                    agent_height=agent_height,
                    agent_radius=0.0,
                    reference_z=z + max_step_height,
                    max_drop=max_step_height * 1.5,
                )
                if not clearance.fits:
                    blocked.add(connector.connector_id)
                    break
        return frozenset(blocked)

    @staticmethod
    def _connector_geometry_is_embedded(connector: ConnectorSpec) -> bool:
        """Whether a connector centerline is embodied by imported room geometry.

        Natural passages and shafts frequently are not useful as additive mesh
        primitives: the tunnel/chimney is already part of a scanned or authored
        cavern shell.  Such connectors still participate in semantic topology and
        route-clearance checks, but must not produce a duplicate simulation model.
        """

        return (
            connector.connector_type
            in {
                ConnectorType.NATURAL_PASSAGE,
                ConnectorType.SHAFT,
            }
            and connector.parameters.get("geometry_embedded") is True
        )

    def compile_connectors(self, output_dir: Path) -> dict[str, Path]:
        """Compile all supported semantic connectors into simulation assets."""
        from scenesmith.agent_utils.structural_compiler import (
            compile_connector,
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        for connector in self.connectors:
            if self._connector_geometry_is_embedded(connector):
                continue
            connector_output = output_dir / connector.connector_id
            paths = write_compiled_structure(
                compile_connector(connector), connector_output
            )
            compiled_paths[connector.connector_id] = paths.sdf_path
        self.connector_geometry_paths = compiled_paths
        return dict(compiled_paths)

    def compile_polygon_rooms(self, output_dir: Path) -> dict[str, Path]:
        """Compile explicitly polygonal rooms into room-compatible SDF assets.

        Polygon coordinates are authored in a min-corner-local convention, like
        legacy ``RoomSpec.position``.  The compiler recenters them because room
        models are welded to a frame at the footprint bounding-box center.
        """

        from scenesmith.agent_utils.structural_compiler import (
            compile_polygon_space,
            write_compiled_structure,
        )

        compiled_paths: dict[str, Path] = {}
        for room_spec in self.room_specs:
            if (
                room_spec.footprint is None
                and room_spec.floor_footprint is None
                and room_spec.ceiling_footprint is None
                and room_spec.floor_profile == ElevationProfile()
                and room_spec.ceiling_profile is None
                and abs(room_spec.yaw) <= 1e-9
                and not any(
                    heightfield.space_id == room_spec.room_id
                    and heightfield.replaces_floor
                    for heightfield in self.heightfields
                )
            ):
                continue
            source_footprint = room_spec.footprint or Footprint2D.rectangle(
                room_spec.length, room_spec.width
            )
            min_x, min_y, max_x, max_y = source_footprint.bounds
            footprint_width = max_x - min_x
            footprint_depth = max_y - min_y
            if not math.isclose(
                footprint_width, room_spec.length, rel_tol=0.0, abs_tol=1e-6
            ) or not math.isclose(
                footprint_depth, room_spec.width, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"Room '{room_spec.room_id}' footprint bounds are "
                    f"{footprint_width:g} × {footprint_depth:g}, but width/depth "
                    f"are {room_spec.length:g} × {room_spec.width:g}"
                )

            local_footprint = source_footprint.centered_on_bounds()
            local_floor_footprint = (
                room_spec.floor_footprint or source_footprint
            ).centered_on_bounds()
            local_ceiling_footprint = (
                room_spec.ceiling_footprint or source_footprint
            ).centered_on_bounds()
            authored_portals = [
                portal
                for portal in self.portals
                if portal.source_space_id == room_spec.room_id
                and portal.boundary_loop_index is not None
            ]
            placed_room = self.get_placed_room(room_spec.room_id)
            legacy_portals = (
                legacy_openings_to_boundary_portals(
                    room_spec, placed_room, self.wall_height
                )
                if placed_room is not None
                else []
            )
            authored_ids = {portal.portal_id for portal in authored_portals}
            compiled = compile_polygon_space(
                structure_id=f"room_geometry_{room_spec.room_id}",
                footprint=local_footprint,
                floor_footprint=local_floor_footprint,
                ceiling_footprint=local_ceiling_footprint,
                include_floor=not any(
                    heightfield.space_id == room_spec.room_id
                    and heightfield.replaces_floor
                    for heightfield in self.heightfields
                ),
                floor_profile=room_spec.floor_profile,
                ceiling_profile=room_spec.ceiling_profile,
                wall_height=self.wall_height,
                portals=[
                    *authored_portals,
                    *(
                        portal
                        for portal in legacy_portals
                        if portal.portal_id not in authored_ids
                    ),
                ],
            )
            paths = write_compiled_structure(
                compiled,
                output_dir / room_spec.room_id,
                model_name="room_geometry",
                link_name="room_geometry_body_link",
            )
            geometry = RoomGeometry(
                sdf_tree=ET.parse(paths.sdf_path),
                sdf_path=paths.sdf_path,
                width=footprint_depth,
                length=footprint_width,
                wall_height=self.wall_height,
                footprint=local_footprint,
                floor_footprint=local_floor_footprint,
                ceiling_footprint=local_ceiling_footprint,
                floor_profile=room_spec.floor_profile,
                ceiling_profile=room_spec.ceiling_profile,
                structural_surfaces=[patch.surface for patch in compiled.surfaces],
                structural_surface_path=paths.surfaces_path,
                wall_normals={
                    patch.surface.surface_id: np.asarray(patch.normal[:2])
                    for patch in compiled.surfaces
                    if SurfaceRole.BOUNDARY in patch.surface.roles
                },
            )
            self.set_room_geometry(room_spec.room_id, geometry)
            compiled_paths[room_spec.room_id] = paths.sdf_path
        return compiled_paths

    def compile_structural_meshes(
        self, output_dir: Path, *, repair: bool = False
    ) -> dict[str, Path]:
        """Compile cavern/freeform meshes into validated SDF and surface assets."""

        from scenesmith.agent_utils.structural_compiler import (
            compile_structural_mesh,
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        # Replacement shells must establish RoomGeometry before additive mesh
        # sidecars are attached, independent of authoring array order.
        ordered_meshes = sorted(
            self.structural_meshes, key=lambda mesh: not mesh.replaces_room_shell
        )
        for mesh_spec in ordered_meshes:
            compiled = compile_structural_mesh(mesh_spec, repair=repair)
            paths = write_compiled_structure(
                compiled,
                output_dir / mesh_spec.mesh_id,
                model_name=("room_geometry" if mesh_spec.replaces_room_shell else None),
                link_name=(
                    "room_geometry_body_link"
                    if mesh_spec.replaces_room_shell
                    else "structure_link"
                ),
            )
            compiled_paths[mesh_spec.mesh_id] = paths.sdf_path
            if mesh_spec.replaces_room_shell:
                bounds_min, bounds_max = compiled.visual_mesh.bounds
                self.set_room_geometry(
                    mesh_spec.space_id,
                    RoomGeometry(
                        sdf_tree=ET.parse(paths.sdf_path),
                        sdf_path=paths.sdf_path,
                        width=bounds_max[1] - bounds_min[1],
                        length=bounds_max[0] - bounds_min[0],
                        wall_height=bounds_max[2] - bounds_min[2],
                        structural_surfaces=[
                            patch.surface for patch in compiled.surfaces
                        ],
                        structural_surface_path=paths.surfaces_path,
                        wall_normals={
                            patch.surface.surface_id: np.asarray(patch.normal[:2])
                            for patch in compiled.surfaces
                            if SurfaceRole.BOUNDARY in patch.surface.roles
                        },
                    ),
                )
            else:
                self._attach_structural_sidecar(mesh_spec.space_id, paths.surfaces_path)
        self.structural_mesh_geometry_paths = compiled_paths
        return dict(compiled_paths)

    def compile_platforms(self, output_dir: Path) -> dict[str, Path]:
        """Compile authored platforms and open edges into static SDF assets."""

        from scenesmith.agent_utils.structural_compiler import (
            compile_platform,
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        for platform in self.platforms:
            paths = write_compiled_structure(
                compile_platform(platform), output_dir / platform.platform_id
            )
            compiled_paths[platform.platform_id] = paths.sdf_path
            self._attach_structural_sidecar(platform.space_id, paths.surfaces_path)
        self.platform_geometry_paths = compiled_paths
        return dict(compiled_paths)

    def compile_heightfields(self, output_dir: Path) -> dict[str, Path]:
        """Compile sampled terrain/floors into SDF and semantic sidecars."""

        from scenesmith.agent_utils.structural_compiler import (
            compile_heightfield,
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        for heightfield in self.heightfields:
            paths = write_compiled_structure(
                compile_heightfield(heightfield),
                output_dir / heightfield.heightfield_id,
            )
            compiled_paths[heightfield.heightfield_id] = paths.sdf_path
            self._attach_structural_sidecar(heightfield.space_id, paths.surfaces_path)
        self.heightfield_geometry_paths = compiled_paths
        return dict(compiled_paths)

    def _attach_structural_sidecar(self, room_id: str, path: Path) -> None:
        """Attach one room-local compiled surface sidecar without duplication."""

        geometry = self.get_room_geometry(room_id)
        if geometry is None:
            return
        if path not in geometry.additional_structural_surface_paths:
            geometry.additional_structural_surface_paths.append(path)

    def get_room_spec(self, room_id: str) -> RoomSpec | None:
        """Get room specification by ID.

        Args:
            room_id: The room ID to look up.

        Returns:
            RoomSpec if found, None otherwise.
        """
        for spec in self.room_specs:
            if spec.room_id == room_id:
                return spec
        return None

    def get_placed_room(self, room_id: str) -> PlacedRoom | None:
        """Get placed room by ID.

        Args:
            room_id: The room ID to look up.

        Returns:
            PlacedRoom if found, None otherwise.
        """
        for placed_room in self.placed_rooms:
            if placed_room.room_id == room_id:
                return placed_room
        return None

    def get_room_geometry(self, room_id: str) -> RoomGeometry | None:
        """Get generated geometry for a room.

        Args:
            room_id: The room ID to look up.

        Returns:
            RoomGeometry if generated, None otherwise.
        """
        return self.room_geometries.get(room_id)

    def set_room_geometry(self, room_id: str, geometry: RoomGeometry) -> None:
        """Store generated geometry for a room.

        Args:
            room_id: The room ID.
            geometry: The generated RoomGeometry.

        Raises:
            ValueError: If room_id is not in room_specs.
        """
        if not any(spec.room_id == room_id for spec in self.room_specs):
            raise ValueError(f"Unknown room_id: {room_id}")
        self.room_geometries[room_id] = geometry

    def invalidate_room_geometry(self, room_id: str) -> bool:
        """Invalidate cached geometry for a specific room.

        Call this when room properties change (dimensions, walls, materials,
        openings) to force regeneration on next render.

        Args:
            room_id: The room ID to invalidate.

        Returns:
            True if geometry was invalidated, False if room had no cached geometry.
        """
        if room_id in self.room_geometries:
            del self.room_geometries[room_id]
            return True
        return False

    def invalidate_all_room_geometries(self) -> int:
        """Invalidate all cached room geometries.

        Call this when global properties change (wall_height, exterior materials)
        or when the entire layout is regenerated.

        Returns:
            Number of rooms that had cached geometry invalidated.
        """
        count = len(self.room_geometries)
        self.room_geometries.clear()
        return count

    @property
    def room_ids(self) -> list[str]:
        """Get list of all room IDs in order."""
        return [spec.room_id for spec in self.room_specs]

    def to_dict(self, scene_dir: Path | None = None) -> dict[str, Any]:
        """Serialize HouseLayout to dictionary for JSON export.

        Args:
            scene_dir: Optional scene directory for path relativization.
                       If None, paths are stored as absolute paths.

        Returns:
            Dictionary suitable for saving as house_layout.json.
        """
        # Serialize placed_rooms if present.
        placed_rooms_data = None
        if self.placed_rooms is not None:
            placed_rooms_data = [placed.to_dict() for placed in self.placed_rooms]

        # Serialize room_geometries if present.
        room_geometries_data = {}
        for room_id, geometry in self.room_geometries.items():
            room_geometries_data[room_id] = geometry.to_dict(scene_dir=scene_dir)

        return {
            "schema_version": SCHEMA_VERSION,
            "wall_height": self.wall_height,
            "house_prompt": self.house_prompt,
            "rooms": [spec.to_dict() for spec in self.room_specs],
            "levels": [level.to_dict() for level in self.levels],
            "connectors": [connector.to_dict() for connector in self.connectors],
            "connector_geometry_paths": {
                connector_id: safe_relative_path(path, scene_dir)
                for connector_id, path in self.connector_geometry_paths.items()
            },
            "structural_meshes": [
                {
                    **mesh.to_dict(),
                    "mesh_path": safe_relative_path(Path(mesh.mesh_path), scene_dir),
                }
                for mesh in self.structural_meshes
            ],
            "structural_mesh_geometry_paths": {
                mesh_id: safe_relative_path(path, scene_dir)
                for mesh_id, path in self.structural_mesh_geometry_paths.items()
            },
            "semantic_environment": (
                self.semantic_environment.to_dict()
                if self.semantic_environment is not None
                else None
            ),
            "semantic_environment_geometry_path": (
                safe_relative_path(self.semantic_environment_geometry_path, scene_dir)
                if self.semantic_environment_geometry_path is not None
                else None
            ),
            "platforms": [platform.to_dict() for platform in self.platforms],
            "platform_geometry_paths": {
                platform_id: safe_relative_path(path, scene_dir)
                for platform_id, path in self.platform_geometry_paths.items()
            },
            "heightfields": [
                heightfield.to_dict() for heightfield in self.heightfields
            ],
            "heightfield_geometry_paths": {
                heightfield_id: safe_relative_path(path, scene_dir)
                for heightfield_id, path in self.heightfield_geometry_paths.items()
            },
            "portals": [portal.to_dict() for portal in self.portals],
            "placed_rooms": placed_rooms_data,
            "doors": [door.to_dict() for door in self.doors],
            "windows": [window.to_dict() for window in self.windows],
            "room_materials": {
                room_id: materials.to_dict()
                for room_id, materials in self.room_materials.items()
            },
            "exterior_material": (
                self.exterior_material.to_dict() if self.exterior_material else None
            ),
            "placement_valid": self.placement_valid,
            "connectivity_valid": self.connectivity_valid,
            "boundary_labels": {k: list(v) for k, v in self.boundary_labels.items()},
            "room_geometries": room_geometries_data,
        }

    def to_drake_directive(self, base_dir: Path | None = None) -> str:
        """Generate a Drake directive string for all room geometries.

        Creates a directive that includes all room geometry SDFs, with a
        house_frame at the root and room frames as children. Each room
        geometry is welded to its room frame.

        Args:
            base_dir: If provided, SDF paths are relative to this directory
                (for portable directives). The directive YAML file should be
                saved in this directory for Drake to resolve paths correctly.
                If None, absolute paths with file:// scheme are used.

        Returns:
            Drake directive in YAML format.

        Raises:
            ValueError: If no room geometries have been generated.
        """
        if (
            not self.room_geometries
            and not self.structural_meshes
            and self.semantic_environment is None
            and not self.platforms
            and not self.heightfields
        ):
            raise ValueError(
                "No room or freeform structural geometries have been defined. "
                "Generate or compile structural geometry first."
            )

        def format_sdf_path(sdf_path: Path | str | None) -> str:
            """Format SDF path as package:// URI or absolute file:// URI."""
            if sdf_path is None:
                return ""
            sdf_path = Path(sdf_path)
            if base_dir is not None:
                # Use package://scene/ for portable scenes.
                # Drake resolves this via PackageMap (set ROS_PACKAGE_PATH or
                # call parser.package_map().Add("scene", scene_dir)).
                rel_path = os.path.relpath(sdf_path, base_dir)
                return f"package://scene/{rel_path}"
            else:
                return f"file://{sdf_path.absolute()}"

        # Build lookup from room_id to PlacedRoom for positions.
        placed_room_lookup = {room.room_id: room for room in self.placed_rooms}

        directive = """directives:
- add_frame:
    name: house_frame
    X_PF:
      base_frame: world
      translation: [0, 0, 0]"""

        for room_id, room_geometry in self.room_geometries.items():
            # Get room position from placed_rooms (not room_specs).
            placed_room = placed_room_lookup.get(room_id)
            if placed_room is None:
                console_logger.warning(
                    f"Room '{room_id}' not found in placed_rooms, skipping"
                )
                continue

            # PlacedRoom.position is (x, y) of min corner.
            # Room geometry is centered at origin, so translate to room center.
            room_center_x = placed_room.position[0] + placed_room.width / 2
            room_center_y = placed_room.position[1] + placed_room.depth / 2
            room_center_z = self.get_room_elevation(room_id)
            room_yaw_deg = placed_room.yaw * 180.0 / np.pi

            room_frame_name = f"room_{room_id}_frame"
            model_name = f"room_geometry_{room_id}"
            room_geom_path = format_sdf_path(room_geometry.sdf_path)

            # Add room frame as child of house_frame.
            directive += f"""
- add_frame:
    name: {room_frame_name}
    X_PF:
      base_frame: house_frame
      translation: [{room_center_x}, {room_center_y}, {room_center_z}]
      rotation: !AngleAxis
        angle_deg: {room_yaw_deg}
        axis: [0, 0, 1]
- add_model:
    name: {model_name}
    file: {room_geom_path}
- add_weld:
    parent: {room_frame_name}
    child: {model_name}::room_geometry_body_link"""

        directive += self._connector_drake_directives(base_dir=base_dir)
        directive += self._structural_mesh_drake_directives(base_dir=base_dir)
        directive += self._semantic_environment_drake_directive(base_dir=base_dir)
        directive += self._platform_drake_directives(base_dir=base_dir)
        directive += self._heightfield_drake_directives(base_dir=base_dir)

        return directive

    def _connector_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for compiled structural connectors."""
        if not self.connectors:
            return ""
        missing = [
            connector.connector_id
            for connector in self.connectors
            if not self._connector_geometry_is_embedded(connector)
            if connector.connector_id not in self.connector_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Connector geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_connectors() before exporting the house."
            )

        directives = ""
        for connector in self.connectors:
            if self._connector_geometry_is_embedded(connector):
                continue
            sdf_path = self.connector_geometry_paths[connector.connector_id]
            if base_dir is None:
                formatted_path = f"file://{sdf_path.absolute()}"
            else:
                formatted_path = (
                    f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
                )
            model_name = f"structure_{connector.connector_id}"
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: house_frame
    child: {model_name}::structure_link"""
        return directives

    def _platform_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for platforms in room-local frames."""

        if not self.platforms:
            return ""
        missing = [
            platform.platform_id
            for platform in self.platforms
            if platform.platform_id not in self.platform_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Platform geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_platforms() before exporting the house."
            )
        directives = ""
        placed_ids = {room.room_id for room in self.placed_rooms}
        for platform in self.platforms:
            sdf_path = self.platform_geometry_paths[platform.platform_id]
            formatted_path = (
                f"file://{sdf_path.absolute()}"
                if base_dir is None
                else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
            )
            model_name = f"structure_{platform.platform_id}"
            parent_frame = (
                f"room_{platform.space_id}_frame"
                if platform.space_id in placed_ids
                else "house_frame"
            )
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::structure_link"""
        return directives

    def _structural_mesh_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for compiled freeform structures."""

        if not self.structural_meshes:
            return ""
        missing = [
            mesh.mesh_id
            for mesh in self.structural_meshes
            if mesh.mesh_id not in self.structural_mesh_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Structural mesh geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_structural_meshes() before exporting the house."
            )
        directives = ""
        placed_ids = {room.room_id for room in self.placed_rooms}
        for mesh in self.structural_meshes:
            if mesh.replaces_room_shell:
                # Its room-compatible SDF is already emitted by the standard
                # room directive and welded to the room frame there.
                continue
            sdf_path = self.structural_mesh_geometry_paths[mesh.mesh_id]
            formatted_path = (
                f"file://{sdf_path.absolute()}"
                if base_dir is None
                else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
            )
            model_name = f"structure_{mesh.mesh_id}"
            parent_frame = (
                f"room_{mesh.space_id}_frame"
                if mesh.space_id in placed_ids
                else "house_frame"
            )
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::structure_link"""
        return directives

    def _semantic_environment_drake_directive(
        self, base_dir: Path | None = None
    ) -> str:
        """Generate a house-frame directive for compiled semantic void geometry."""

        if self.semantic_environment is None:
            return ""
        if self.semantic_environment_geometry_path is None:
            raise ValueError(
                "Semantic environment geometry has not been compiled. "
                "Call compile_semantic_environment() before exporting the house."
            )
        sdf_path = self.semantic_environment_geometry_path
        formatted_path = (
            f"file://{sdf_path.absolute()}"
            if base_dir is None
            else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
        )
        return f"""
- add_model:
    name: structure_semantic_environment
    file: {formatted_path}
- add_weld:
    parent: house_frame
    child: structure_semantic_environment::structure_link"""

    def _heightfield_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for room-local heightfields."""

        if not self.heightfields:
            return ""
        missing = [
            heightfield.heightfield_id
            for heightfield in self.heightfields
            if heightfield.heightfield_id not in self.heightfield_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Heightfield geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_heightfields() before exporting the house."
            )
        directives = ""
        placed_ids = {room.room_id for room in self.placed_rooms}
        for heightfield in self.heightfields:
            sdf_path = self.heightfield_geometry_paths[heightfield.heightfield_id]
            formatted_path = (
                f"file://{sdf_path.absolute()}"
                if base_dir is None
                else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
            )
            model_name = f"structure_{heightfield.heightfield_id}"
            parent_frame = (
                f"room_{heightfield.space_id}_frame"
                if heightfield.space_id in placed_ids
                else "house_frame"
            )
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::structure_link"""
        return directives

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], house_dir: Path | None = None
    ) -> "HouseLayout":
        """Restore HouseLayout from dictionary.

        Args:
            data: Dictionary from to_dict() or house_layout.json.
            house_dir: Directory for house outputs.

        Returns:
            Restored HouseLayout instance.
        """
        input_schema_version = data.get("schema_version", 1)
        room_specs = [RoomSpec.from_dict(r) for r in data.get("rooms", [])]
        levels = [LevelSpec.from_dict(level) for level in data.get("levels", [])] or [
            default_ground_level()
        ]
        connectors = [
            ConnectorSpec.from_dict(connector)
            for connector in data.get("connectors", [])
        ]
        structural_meshes = []
        for mesh_data in data.get("structural_meshes", []):
            resolved_mesh_data = dict(mesh_data)
            if house_dir is not None:
                mesh_path = Path(resolved_mesh_data["mesh_path"])
                if not mesh_path.is_absolute():
                    resolved_mesh_data["mesh_path"] = str(house_dir / mesh_path)
            structural_meshes.append(StructuralMeshSpec.from_dict(resolved_mesh_data))
        platforms = [
            PlatformSpec.from_dict(platform) for platform in data.get("platforms", [])
        ]
        heightfields = [
            HeightfieldSpec.from_dict(heightfield)
            for heightfield in data.get("heightfields", [])
        ]
        portals = [PortalSpec.from_dict(portal) for portal in data.get("portals", [])]
        connector_geometry_paths = {
            connector_id: (house_dir / path if house_dir is not None else Path(path))
            for connector_id, path in data.get("connector_geometry_paths", {}).items()
        }
        structural_mesh_geometry_paths = {
            mesh_id: (house_dir / path if house_dir is not None else Path(path))
            for mesh_id, path in data.get("structural_mesh_geometry_paths", {}).items()
        }
        semantic_environment = (
            SemanticEnvironmentSpec.from_dict(data["semantic_environment"])
            if data.get("semantic_environment")
            else None
        )
        semantic_environment_geometry_path = None
        if data.get("semantic_environment_geometry_path"):
            environment_path = Path(data["semantic_environment_geometry_path"])
            semantic_environment_geometry_path = (
                house_dir / environment_path
                if house_dir is not None and not environment_path.is_absolute()
                else environment_path
            )
        platform_geometry_paths = {
            platform_id: (house_dir / path if house_dir is not None else Path(path))
            for platform_id, path in data.get("platform_geometry_paths", {}).items()
        }
        heightfield_geometry_paths = {
            heightfield_id: (house_dir / path if house_dir is not None else Path(path))
            for heightfield_id, path in data.get(
                "heightfield_geometry_paths", {}
            ).items()
        }
        doors = [Door.from_dict(d) for d in data.get("doors", [])]
        windows = [Window.from_dict(w) for w in data.get("windows", [])]

        # Restore room materials.
        room_materials = {
            room_id: RoomMaterials.from_dict(mat_data)
            for room_id, mat_data in data.get("room_materials", {}).items()
        }

        # Restore exterior material.
        exterior_material = None
        if data.get("exterior_material"):
            exterior_material = Material.from_dict(data["exterior_material"])

        # Restore boundary labels.
        boundary_labels = {
            label: tuple(room_pair)
            for label, room_pair in data.get("boundary_labels", {}).items()
        }

        # Restore placed_rooms if present.
        placed_rooms = None
        if data.get("placed_rooms") is not None:
            placed_rooms = [PlacedRoom.from_dict(p) for p in data["placed_rooms"]]

        # Restore room_geometries if present.
        room_geometries = {}
        if data.get("room_geometries"):
            for room_id, geom_data in data["room_geometries"].items():
                room_geometries[room_id] = RoomGeometry.from_dict(
                    geom_data, scene_dir=house_dir
                )

        layout = cls(
            schema_version=(
                SCHEMA_VERSION if input_schema_version == 1 else input_schema_version
            ),
            wall_height=data.get("wall_height", 2.5),
            house_prompt=data.get("house_prompt", ""),
            room_specs=room_specs,
            levels=levels,
            connectors=connectors,
            connector_geometry_paths=connector_geometry_paths,
            structural_meshes=structural_meshes,
            structural_mesh_geometry_paths=structural_mesh_geometry_paths,
            semantic_environment=semantic_environment,
            semantic_environment_geometry_path=semantic_environment_geometry_path,
            platforms=platforms,
            platform_geometry_paths=platform_geometry_paths,
            heightfields=heightfields,
            heightfield_geometry_paths=heightfield_geometry_paths,
            portals=portals,
            house_dir=house_dir,
            room_geometries=room_geometries,
            placed_rooms=placed_rooms,
            doors=doors,
            windows=windows,
            room_materials=room_materials,
            exterior_material=exterior_material,
            placement_valid=data.get("placement_valid", False),
            connectivity_valid=data.get("connectivity_valid", False),
            boundary_labels=boundary_labels,
        )
        layout.validate_structure()
        return layout

    def content_hash(self) -> str:
        """Generate deterministic hash of layout state for render caching.

        Creates a SHA-256 hash of all layout properties that affect rendering.
        Identical layouts produce identical hashes. Used to cache final renders.

        Returns:
            SHA-256 hash string (first 16 chars) of layout content.
        """
        # Build comprehensive state dict.
        state = {
            "schema_version": SCHEMA_VERSION,
            "wall_height": self.wall_height,
            "levels": [level.to_dict() for level in self.levels],
            "connectors": [connector.to_dict() for connector in self.connectors],
            "connector_geometry_paths": {
                connector_id: str(path)
                for connector_id, path in self.connector_geometry_paths.items()
            },
            "structural_meshes": [mesh.to_dict() for mesh in self.structural_meshes],
            "structural_mesh_geometry_paths": {
                mesh_id: str(path)
                for mesh_id, path in self.structural_mesh_geometry_paths.items()
            },
            "semantic_environment": (
                self.semantic_environment.to_dict()
                if self.semantic_environment is not None
                else None
            ),
            "semantic_environment_geometry_path": (
                str(self.semantic_environment_geometry_path)
                if self.semantic_environment_geometry_path is not None
                else None
            ),
            "platforms": [platform.to_dict() for platform in self.platforms],
            "platform_geometry_paths": {
                platform_id: str(path)
                for platform_id, path in self.platform_geometry_paths.items()
            },
            "heightfields": [
                heightfield.to_dict() for heightfield in self.heightfields
            ],
            "heightfield_geometry_paths": {
                heightfield_id: str(path)
                for heightfield_id, path in self.heightfield_geometry_paths.items()
            },
            "portals": [portal.to_dict() for portal in self.portals],
            "placed_rooms": [
                {
                    "room_id": r.room_id,
                    "position": r.position,
                    "width": r.width,
                    "depth": r.depth,
                    "level_id": r.level_id,
                    "elevation": self.get_room_elevation(r.room_id),
                    "yaw": r.yaw,
                    "footprint": r.footprint.to_dict() if r.footprint else None,
                    # Include wall cache keys for each wall.
                    "walls": [
                        w.cache_key(
                            wall_height=self.wall_height,
                            material=self._get_wall_material(r.room_id),
                        )
                        for w in r.walls
                    ],
                }
                for r in self.placed_rooms
            ],
            "room_materials": {
                room_id: {
                    "wall": str(m.wall_material.path) if m.wall_material else None,
                    "floor": str(m.floor_material.path) if m.floor_material else None,
                }
                for room_id, m in self.room_materials.items()
            },
            "exterior_material": (
                str(self.exterior_material.path) if self.exterior_material else None
            ),
        }
        content_json = json.dumps(state, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()[:16]

    def _get_wall_material(self, room_id: str) -> Material | None:
        """Get wall material for a room.

        Args:
            room_id: Room to get wall material for.

        Returns:
            Material or None if using default.
        """
        room_materials = self.room_materials.get(room_id)
        if room_materials:
            return room_materials.wall_material
        return None


@dataclass
class HouseScene:
    """Complete house scene: layout + populated rooms.

    Always use HouseScene as the top-level container. Room mode is just a
    HouseScene with a single room (room_id="main"). This unified model avoids
    code duplication between modes.

    HouseScene contains the HouseLayout (floor plan data) and populated
    RoomScene instances.
    """

    layout: HouseLayout
    """House layout containing room specs, geometry, and doors/windows."""

    rooms: dict[str, "RoomScene"] = field(default_factory=dict)
    """Dictionary mapping room_id to RoomScene instances."""

    @property
    def house_dir(self) -> Path:
        """Base directory for the house (from layout)."""
        if self.layout.house_dir is None:
            raise ValueError("HouseLayout.house_dir is not set")
        return self.layout.house_dir

    def _get_room_position(self, room_id: str) -> tuple[float, float]:
        """Get legacy XY room-center position from the full v2 transform."""

        x, y, _, _ = self._get_room_transform(room_id)
        return (x, y)

    def _get_room_transform(self, room_id: str) -> tuple[float, float, float, float]:
        """Get room center XYZ and yaw from layout.

        Room geometry is centered at origin, so we need the center position
        (not corner) when placing rooms in the combined directive.

        Args:
            room_id: Room ID to look up.

        Returns:
            (x, y, z, yaw) tuple. Returns identity if room is not found.
        """
        for placed in self.layout.placed_rooms:
            if placed.room_id == room_id:
                # Convert from corner to center position.
                center_x = placed.position[0] + placed.width / 2
                center_y = placed.position[1] + placed.depth / 2
                return (
                    center_x,
                    center_y,
                    self.layout.get_room_elevation(room_id),
                    placed.yaw,
                )
        # Default to origin for single room mode or if placement not done.
        return (0.0, 0.0, 0.0, 0.0)

    def add_room(self, room: "RoomScene") -> None:
        """Add a room to the house.

        Args:
            room: RoomScene to add. room.room_id must be unique within this house.

        Raises:
            ValueError: If a room with the same room_id already exists.
        """
        if room.room_id in self.rooms:
            raise ValueError(f"Room with id '{room.room_id}' already exists")
        self.rooms[room.room_id] = room

    def get_room(self, room_id: str) -> "RoomScene | None":
        """Get a room by ID.

        Args:
            room_id: The room ID to look up.

        Returns:
            RoomScene if found, None otherwise.
        """
        return self.rooms.get(room_id)

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize HouseScene to dictionary for checkpointing.

        Returns:
            Dictionary containing complete house state including layout.
        """
        rooms_dict = {}
        for room_id, room in self.rooms.items():
            rooms_dict[room_id] = room.to_state_dict()

        return {
            "layout": self.layout.to_dict(scene_dir=self.house_dir),
            "rooms": rooms_dict,
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], house_dir: Path
    ) -> "HouseScene":
        """Create HouseScene from serialized dictionary.

        Args:
            state_dict: State dictionary from to_state_dict().
            house_dir: Base directory for the house (needed for path resolution).

        Returns:
            Restored HouseScene instance.
        """
        # Import here to avoid circular import.
        from scenesmith.agent_utils.room import RoomScene

        # Restore layout.
        layout = HouseLayout.from_dict(state_dict["layout"], house_dir=house_dir)

        # Create HouseScene with restored layout.
        house_scene = cls(layout=layout)

        # Restore rooms.
        for room_id, room_data in state_dict["rooms"].items():
            room_dir = house_dir / f"room_{room_id}"
            room = RoomScene(
                room_geometry=None,  # Will be restored.
                scene_dir=room_dir,
                room_id=room_id,
            )
            room.restore_from_state_dict(room_data)
            house_scene.rooms[room_id] = room

        return house_scene

    def assemble(
        self,
        cfg: dict | DictConfig | None = None,
        output_name: str = "combined_house",
        include_object_types: "list[ObjectType] | None" = None,
    ) -> Path:
        """Assemble all rooms into combined house outputs.

        Creates the output directory with:
        - house.dmd.yaml: Drake directive with furniture as free bodies
          (only wall/ceiling-mounted objects welded)
        - house_furniture_welded.dmd.yaml: Drake directive with furniture welded
        - house_state.json: Combined state for all rooms
        - sceneeval_state.json: Combined SceneEval format
        - house.blend: Blender file for visualization (uses house.dmd.yaml)

        Single room is treated as a house with one room at identity transform.

        Note: Composite manipulands (stacks, piles) are always free bodies in
        both output files. This is only for final output - internal simulation
        still uses welded furniture and composites for physics.

        Args:
            cfg: Configuration (dict or OmegaConf). Required for blend export.
                If None, blend file will not be generated.
            output_name: Name of output directory (default: "combined_house").
                Use "combined_house_after_furniture" for intermediate saves.
            include_object_types: If provided, only include objects of these
                types in the output. Useful for intermediate snapshots.

        Returns:
            Path to the output directory.
        """
        combined_dir = self.house_dir / output_name
        combined_dir.mkdir(parents=True, exist_ok=True)

        # Generate house.dmd.yaml: furniture as free bodies, composites as free bodies.
        directive_free = self._generate_combined_directive(
            include_object_types=include_object_types,
            weld_furniture=False,
            weld_composite_members=False,
        )
        directive_path_free = combined_dir / "house.dmd.yaml"
        with open(directive_path_free, "w") as f:
            f.write(directive_free)
        console_logger.info(
            f"Saved Drake directive (furniture free): {directive_path_free}"
        )

        # Generate house_furniture_welded.dmd.yaml: furniture welded, composites free.
        directive_welded = self._generate_combined_directive(
            include_object_types=include_object_types,
            weld_furniture=True,
            weld_composite_members=False,
        )
        directive_path_welded = combined_dir / "house_furniture_welded.dmd.yaml"
        with open(directive_path_welded, "w") as f:
            f.write(directive_welded)
        console_logger.info(
            f"Saved Drake directive (furniture welded): {directive_path_welded}"
        )

        # Create package.xml for portability (only once per scene).
        package_xml_path = self.house_dir / "package.xml"
        if not package_xml_path.exists():
            create_package_xml(self.house_dir)
            console_logger.info(f"Created package.xml for scene portability")

        # Save combined house state.
        state_dict = self.to_state_dict()
        state_dict["timestamp"] = time.time()
        state_path = combined_dir / "house_state.json"
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)
        console_logger.info(f"Saved combined house state: {state_path}")

        # Export combined SceneEval format.
        # Imported lazily so layout serialization and validation do not require
        # the full Drake/scene runtime to be importable.
        from scenesmith.agent_utils.sceneeval_exporter import (
            SceneEvalExportConfig,
            SceneEvalExporter,
        )

        floor_thickness = cfg["floor_plan_agent"]["floor_thickness"] if cfg else 0.1
        config = SceneEvalExportConfig(floor_thickness=floor_thickness)
        SceneEvalExporter.export_house(
            house=self, output_dir=combined_dir, config=config
        )

        # Generate combined blend file.
        if cfg is not None:
            self._export_blend(output_dir=combined_dir, cfg=cfg)

        return combined_dir

    def _generate_combined_directive(
        self,
        include_object_types: "list[ObjectType] | None" = None,
        weld_furniture: bool = True,
        weld_composite_members: bool = True,
    ) -> str:
        """Generate Drake directive combining all rooms.

        Single room is just a house with one room at identity transform.
        Multi-room uses frames to position each room at its layout position.

        Args:
            include_object_types: If provided, only include objects of these
                types. Useful for intermediate snapshots.
            weld_furniture: If True (default), weld furniture to world frame.
                If False, furniture is added as free bodies.
            weld_composite_members: If True (default), weld composite manipuland
                members (stacks, piles) to their base. If False, all members
                are free bodies.

        Returns:
            Drake directive YAML string with package://scene/ URIs for portability.
        """
        directive = """directives:
- add_frame:
    name: house_frame
    X_PF:
      base_frame: world
      translation: [0, 0, 0]"""

        for room_id, room in self.rooms.items():
            geometry_name = f"room_geometry_{room_id}"
            room_frame_name = f"room_{room_id}_frame"

            # Get full room transform from the v2 layout.
            pos_x, pos_y, pos_z, yaw = self._get_room_transform(room_id)
            yaw_deg = yaw * 180.0 / np.pi

            # Add room frame as child of house_frame.
            directive += f"""
- add_frame:
    name: {room_frame_name}
    X_PF:
      base_frame: house_frame
      translation: [{pos_x}, {pos_y}, {pos_z}]
      rotation: !AngleAxis
        angle_deg: {yaw_deg}
        axis: [0, 0, 1]"""

            # Get room directive with parent_frame so all objects use
            # room-local coordinates relative to the room frame.
            room_directive = room.to_drake_directive(
                weld_room_geometry=False,
                room_geometry_name=geometry_name,
                model_name_prefix=f"{room_id}_",
                include_object_types=include_object_types,
                base_dir=self.house_dir,
                weld_furniture=weld_furniture,
                weld_stack_members=weld_composite_members,
                parent_frame=room_frame_name,
            )

            # Strip the "directives:" header.
            if room_directive.startswith("directives:"):
                room_directive = room_directive[len("directives:") :]
            directive += room_directive

            # Weld room geometry to room frame (no translation needed,
            # room geometry is centered at origin).
            directive += f"""
- add_weld:
    parent: {room_frame_name}
    child: {geometry_name}::room_geometry_body_link"""

        directive += self.layout._connector_drake_directives(base_dir=self.house_dir)
        directive += self.layout._structural_mesh_drake_directives(
            base_dir=self.house_dir
        )
        directive += self.layout._semantic_environment_drake_directive(
            base_dir=self.house_dir
        )
        directive += self.layout._platform_drake_directives(base_dir=self.house_dir)
        directive += self.layout._heightfield_drake_directives(base_dir=self.house_dir)

        return directive

    def _export_blend(self, output_dir: Path, cfg: dict | DictConfig) -> None:
        """Export Blender file for all rooms to combined directory.

        Uses the combined directive for both single and multi-room cases.
        Single room is just a house with one room at identity transform.

        Args:
            output_dir: Directory to save house.blend.
            cfg: Configuration with rendering settings (dict or OmegaConf).
        """
        from scenesmith.agent_utils.rendering import save_directive_as_blend

        directive_path = output_dir / "house.dmd.yaml"
        if not directive_path.exists():
            console_logger.error("Combined directive not found, skipping house.blend")
            return

        blend_output_path = output_dir / "house.blend"
        rendering_cfg = cfg["furniture_agent"]["rendering"]

        try:
            save_directive_as_blend(
                directive_path=directive_path,
                output_path=blend_output_path,
                blender_server_host=rendering_cfg["blender_server_host"],
                blender_server_port_range=tuple(
                    rendering_cfg["blender_server_port_range"]
                ),
                server_startup_delay=rendering_cfg["server_startup_delay"],
                port_cleanup_delay=rendering_cfg["port_cleanup_delay"],
                scene_dir=self.house_dir,
            )
            console_logger.info(f"Saved combined blend file: {blend_output_path}")
        except Exception as e:
            console_logger.error(f"Failed to export combined .blend file: {e}")
