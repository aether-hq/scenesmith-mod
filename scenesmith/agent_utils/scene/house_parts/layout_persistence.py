"""House layout and room geometry data structures."""

import hashlib
import json
import logging

from pathlib import Path
from typing import TYPE_CHECKING, Any

from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structure.geometry_models.common import SCHEMA_VERSION
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    HeightfieldSpec,
    LevelSpec,
    PlatformSpec,
    default_ground_level,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorSpec,
    PortalSpec,
)
from scenesmith.utils.geometry.material import Material

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.house import HouseLayout

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.scene.house_parts.openings import (
    Door,
    PlacedRoom,
    RoomMaterials,
    Window,
)
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec


class HouseLayoutPersistenceMixin:
    """Layout restoration, hashing, and wall-material lookup."""

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
        semantic_detail_geometry_paths = {
            detail_id: (
                house_dir / path
                if house_dir is not None and not Path(path).is_absolute()
                else Path(path)
            )
            for detail_id, path in data.get(
                "semantic_detail_geometry_paths", {}
            ).items()
        }
        semantic_environment_source_hash = data.get("semantic_environment_source_hash")
        semantic_detail_source_hash = data.get("semantic_detail_source_hash")
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
            semantic_environment_source_hash=semantic_environment_source_hash,
            semantic_detail_geometry_paths=semantic_detail_geometry_paths,
            semantic_detail_source_hash=semantic_detail_source_hash,
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
            "semantic_environment_source_hash": self.semantic_environment_source_hash,
            "semantic_detail_geometry_paths": {
                detail_id: str(path)
                for detail_id, path in self.semantic_detail_geometry_paths.items()
            },
            "semantic_detail_source_hash": self.semantic_detail_source_hash,
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
