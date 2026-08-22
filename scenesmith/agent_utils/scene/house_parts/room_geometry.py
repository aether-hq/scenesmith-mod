"""House layout and room geometry data structures."""

import hashlib
import json
import logging
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    ElevationProfile,
    Footprint2D,
    StructuralSurface,
)
from scenesmith.utils.path_utils import safe_relative_path

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.scene.house_parts.openings import ClearanceOpeningData


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

    has_overhead_cover: bool = True
    """Whether legacy wall height represents a roof/ceiling constraint."""

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
        floor_plan_dict["has_overhead_cover"] = self.has_overhead_cover
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
            "has_overhead_cover": self.has_overhead_cover,
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
        from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject

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
            has_overhead_cover=bool(data.get("has_overhead_cover", True)),
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
