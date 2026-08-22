import hashlib
import json
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry

from pydrake.all import RigidTransform

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.scene.room_parts.room_directive_mixin import (
    RoomDirectiveMixin,
)
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
    _int_to_base36,
)


@dataclass
class RoomScene(RoomDirectiveMixin):
    """
    Central state manager and single source of truth for a single room's composition.

    Maintains all objects in the room with transactional-like operations (add, remove,
    move, replace) ensuring consistency. Generates Drake simulation directives for
    rendering and physics simulation.
    """

    room_geometry: "RoomGeometry"
    """The generated 3D geometry for this room (walls, floor, SDF)."""

    scene_dir: Path
    """Base directory for the room (all paths are relative to this)."""

    room_id: str = "main"
    """Unique identifier for this room within a house. Default 'main' for room mode."""

    room_type: str = "room"
    """Type of room (e.g., 'living_room', 'bedroom'). Default 'room' for room mode."""

    objects: dict[UniqueID, SceneObject] = field(default_factory=dict)
    """Dictionary mapping object IDs to SceneObject instances."""

    text_description: str = ""
    """Text description of the overall room."""

    action_log_path: Path | None = None
    """Path to action log file for scene replication/replay."""

    _surface_id_counter: int = field(default=0, init=False, repr=False)
    """Counter for generating sequential surface IDs (S_0, S_1, etc.)."""

    def add_object(self, obj: SceneObject) -> None:
        """Add an object to the scene."""
        self.objects[obj.object_id] = obj

    def remove_object(self, object_id: UniqueID) -> bool:
        """Remove an object from the scene. Returns True if removed."""
        if object_id in self.objects:
            del self.objects[object_id]
            return True
        return False

    def get_object(self, object_id: UniqueID) -> SceneObject | None:
        """Get an object by ID.

        Searches both scene.objects and scene.room_geometry.floor to support
        floor as a placement target for manipulands.

        Args:
            object_id: Unique identifier for the object.

        Returns:
            SceneObject if found, None otherwise.
        """
        # Check regular objects first.
        obj = self.objects.get(object_id)
        if obj:
            return obj

        # Check floor if available.
        if (
            self.room_geometry
            and self.room_geometry.floor
            and self.room_geometry.floor.object_id == object_id
        ):
            return self.room_geometry.floor

        return None

    def generate_unique_id(self, name: str) -> UniqueID:
        """Generate a unique ID that doesn't conflict with existing scene objects.

        Uses base-36 sequential numbering (0-9, a-z) for compact IDs.

        Args:
            name: Human-readable name for the object.

        Returns:
            UniqueID that is guaranteed unique within this scene.
        """
        return UniqueID.generate_unique(name, self.objects)

    def generate_surface_id(self) -> UniqueID:
        """Generate next sequential surface ID using base-36 encoding.

        Returns:
            UniqueID in format S_0, S_1, ..., S_9, S_a, ..., S_z, S_10, etc.
        """
        suffix = _int_to_base36(self._surface_id_counter)
        surface_id = UniqueID(f"S_{suffix}")
        self._surface_id_counter += 1
        return surface_id

    def move_object(self, object_id: UniqueID, new_transform: RigidTransform) -> bool:
        """Move an object to a new position. Returns True if successful."""
        if object_id not in self.objects:
            return False
        self.objects[object_id].transform = new_transform
        return True

    def get_objects_by_type(self, object_type: ObjectType) -> list[SceneObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects.values() if obj.object_type == object_type]

    def get_manipulands(self) -> list[SceneObject]:
        """Get all manipuland objects in the scene.

        Returns:
            List of SceneObject instances with object_type=MANIPULAND.
        """
        return self.get_objects_by_type(ObjectType.MANIPULAND)

    def get_objects_on_surface(self, surface_id: UniqueID) -> list[SceneObject]:
        """Get all objects placed on a specific support surface.

        Args:
            surface_id: The ID of the support surface to query.

        Returns:
            List of SceneObject instances placed on the specified surface.
        """
        return [
            obj
            for obj in self.objects.values()
            if obj.placement_info and obj.placement_info.parent_surface_id == surface_id
        ]

    def content_hash(self) -> str:
        """
        Generate deterministic content hash of entire Scene state.

        This creates a SHA-256 hash of all scene content including floor plan,
        objects, positions, and metadata. Identical scenes will produce identical
        hashes regardless of object creation order or identity.

        Returns:
            str: SHA-256 hash string of scene content for caching.
        """
        # Collect all content for hashing by delegating to individual class methods.
        content_dict = {
            "room_geometry": self.room_geometry.content_hash(),
            "objects": self._hash_objects(),
            "text_description": self.text_description,
        }

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(content_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_state_dict(self) -> dict[str, Any]:
        """
        Return complete scene state as a dictionary for checkpointing.

        Serializes all scene data including room geometry, objects with full
        metadata needed for restoration via restore_from_state_dict(). All paths
        are saved relative to self.scene_dir for portability.

        Returns:
            Dictionary containing complete scene state including room geometry.
        """
        objects_dict = {}
        for obj in self.objects.values():
            objects_dict[str(obj.object_id)] = obj.to_dict(scene_dir=self.scene_dir)

        # Serialize room geometry if present.
        room_geometry_data = None
        if self.room_geometry:
            room_geometry_data = self.room_geometry.to_dict(scene_dir=self.scene_dir)

        return {
            "room_geometry": room_geometry_data,
            "objects": objects_dict,
            "text_description": self.text_description,
        }

    def restore_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore scene to state from serialized dictionary.

        Resolves all paths relative to self.scene_dir for portability.
        Restores room geometry first, then objects, then populates
        room_geometry.walls from restored wall objects.

        Args:
            state_dict: State dictionary from to_state_dict()
        """
        # Import here to avoid circular import.
        from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry

        # Restore room geometry first if present.
        if state_dict.get("room_geometry"):
            self.room_geometry = RoomGeometry.from_dict(
                state_dict["room_geometry"], scene_dir=self.scene_dir
            )
        else:
            self.room_geometry = None

        # Clear current objects.
        self.objects.clear()

        # Restore text description.
        self.text_description = state_dict["text_description"]

        # Restore objects.
        for obj_data in state_dict["objects"].values():
            scene_object = SceneObject.from_dict(obj_data, scene_dir=self.scene_dir)
            self.objects[scene_object.object_id] = scene_object

        # Populate room_geometry.walls from restored wall objects.
        if self.room_geometry:
            self.room_geometry.walls = [
                obj
                for obj in self.objects.values()
                if obj.object_type == ObjectType.WALL
            ]

    def _hash_objects(self) -> dict:
        """Hash all scene objects using their individual content_hash methods."""
        objects_dict = {}

        # Sort objects by ID for deterministic ordering.
        for object_id in sorted(self.objects.keys(), key=str):
            obj = self.objects[object_id]
            objects_dict[str(object_id)] = obj.content_hash()

        return objects_dict
