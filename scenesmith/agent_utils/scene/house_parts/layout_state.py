"""House layout and room geometry data structures."""

import logging

from pathlib import Path
from typing import TYPE_CHECKING, Any

from scenesmith.agent_utils.structure.geometry_models.common import SCHEMA_VERSION
from scenesmith.utils.path_utils import safe_relative_path

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec


class HouseLayoutStateMixin:
    """Room lookup, geometry invalidation, and dictionary serialization."""

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
            "semantic_environment_source_hash": self.semantic_environment_source_hash,
            "semantic_detail_geometry_paths": {
                detail_id: safe_relative_path(path, scene_dir)
                for detail_id, path in self.semantic_detail_geometry_paths.items()
            },
            "semantic_detail_source_hash": self.semantic_detail_source_hash,
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
