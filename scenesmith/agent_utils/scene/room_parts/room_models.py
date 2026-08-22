import hashlib
import json
import logging
import uuid

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import numpy as np
import trimesh

from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.geometry.optimization import VPolytope

from scenesmith.utils.geometry.geometry_utils import (
    compute_aabb_corners,
    safe_convex_hull_2d,
)
from scenesmith.utils.geometry.sdf_utils import serialize_rigid_transform
from scenesmith.utils.path_utils import safe_relative_path

console_logger = logging.getLogger(__name__)


def _int_to_base36(num: int) -> str:
    """Convert integer to base-36 (0-9, a-z) representation.

    Args:
        num: Non-negative integer to convert.

    Returns:
        Base-36 string representation.

    Examples:
        >>> _int_to_base36(0)
        '0'
        >>> _int_to_base36(10)
        'a'
        >>> _int_to_base36(36)
        '10'
    """
    if num < 10:
        return str(num)
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while num:
        result = chars[num % 36] + result
        num //= 36
    return result


class UniqueID(str):
    """Type-safe unique identifier string."""

    @classmethod
    def generate(cls) -> "UniqueID":
        """Generate a new unique ID using UUID4."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def generate_unique(
        cls, name: str, existing_ids: set["UniqueID"] | dict[str, Any]
    ) -> "UniqueID":
        """Generate unique ID that doesn't conflict with existing IDs.

        Uses base-36 sequential numbering (0-9, a-z) for compact IDs:
        - First occurrence: single digit suffix (e.g., "chair_0")
        - 11th: single letter (e.g., "chair_a")
        - 37th+: two chars (e.g., "chair_10")

        Args:
            name: Human-readable name for the object.
            existing_ids: Set or dict of IDs to check against for uniqueness.

        Returns:
            UniqueID guaranteed not to be in existing_ids.
        """
        base_name = name.lower().replace(" ", "_")
        index = 0
        while True:
            suffix = _int_to_base36(index)
            candidate = cls(f"{base_name}_{suffix}")
            if candidate not in existing_ids:
                return candidate
            index += 1


class ObjectType(Enum):
    """Enum for different types of objects in the scene."""

    FURNITURE = "furniture"
    MANIPULAND = "manipuland"
    THIN_COVERING = "thin_covering"  # Flat textured surface (no collision geometry)
    WALL_MOUNTED = "wall_mounted"
    CEILING_MOUNTED = "ceiling_mounted"
    WALL = "wall"
    FLOOR = "floor"
    EITHER = "either"  # Ambiguous items for analysis


class AgentType(Enum):
    """Pipeline agent types.

    Order defines pipeline execution order. All agents use the same
    planner/designer/critic trio architecture.
    """

    FLOOR_PLAN = "floor_plan"
    FURNITURE = "furniture"
    WALL_MOUNTED = "wall_mounted"
    CEILING_MOUNTED = "ceiling_mounted"
    MANIPULAND = "manipuland"

    def to_object_type(self) -> ObjectType | None:
        """Convert agent type to corresponding object type.

        Returns None for FLOOR_PLAN which doesn't produce scene objects.
        """
        if self == AgentType.FLOOR_PLAN:
            return None
        return ObjectType(self.value)

    @property
    def is_placement_agent(self) -> bool:
        """Whether this agent places objects via asset router."""
        return self != AgentType.FLOOR_PLAN


@dataclass
class PlacementInfo:
    """Metadata for objects placed on support surfaces.

    Stores the surface-relative placement information (SE(2) on surface) alongside
    the world-frame transform. This enables replay, debugging, and understanding of
    the hierarchical placement structure.
    """

    parent_surface_id: UniqueID
    """ID of the support surface this object is placed on."""

    position_2d: np.ndarray
    """2D position on the support surface in surface coordinates. Shape: (2,).
    Format: [x, y] where x is left-right and y is front-back in surface frame.
    """

    rotation_2d: float
    """Rotation in radians around the surface normal (Z-axis in surface frame)."""

    placement_method: str = "surface_placement"
    """Method used for placement (e.g., 'surface_placement', 'snap_to_edge')."""


@dataclass
class SupportSurface:
    """Represents a support surface where objects can be placed.

    A support surface is a flat surface that can be used to place objects on.
    It is represented by an axis-aligned bounding box and a transform, with optional
    sub-mesh geometry for accurate visualization and future polygon-based placement.
    """

    surface_id: UniqueID
    """Unique identifier for the support surface."""

    bounding_box_min: np.ndarray
    """Minimum corner of the 3D axis-aligned bounding box, in the surface's local
    coordinate frame (origin at surface position, see transform). Shape: (3,).
    """

    bounding_box_max: np.ndarray
    """Maximum corner of the 3D axis-aligned bounding box, in the surface's local
    coordinate frame (origin at surface position, see transform). Shape: (3,).
    """

    transform: RigidTransform
    """Pose of the support surface in the world frame."""

    mesh: "trimesh.Trimesh | None" = None
    """Simplified triangle mesh representing the support surface geometry.
    Mesh is in Z-up coordinate system (Drake/Blender standard), flattened to 2D plane.
    Ready for direct rendering without additional coordinate transformations.
    If None, only bounding box representation is available.

    Note: Future optimization could store mesh_path instead if memory becomes an issue,
    but storing the mesh directly simplifies the API and enables better mesh simplification
    during extraction.
    """

    link_name: str | None = None
    """For articulated objects: name of the link this surface belongs to.
    Used to apply FK transforms when rendering with joints open.
    None for non-articulated objects or if link association failed.
    """

    @property
    def area(self) -> float:
        """Compute surface area from XY bounding box dimensions.

        Returns:
            Surface area in square meters.
        """
        width = self.bounding_box_max[0] - self.bounding_box_min[0]
        depth = self.bounding_box_max[1] - self.bounding_box_min[1]
        return float(width * depth)

    def content_hash(self) -> str:
        """Generate content hash for this support surface."""
        content_dict = {
            "surface_id": str(self.surface_id),
            "bounding_box_min": [
                float(self.bounding_box_min[0]),
                float(self.bounding_box_min[1]),
                float(self.bounding_box_min[2]),
            ],
            "bounding_box_max": [
                float(self.bounding_box_max[0]),
                float(self.bounding_box_max[1]),
                float(self.bounding_box_max[2]),
            ],
            "transform": serialize_rigid_transform(self.transform),
        }

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(content_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_world_pose(
        self, position_2d: np.ndarray, rotation_2d: float, z_offset: float = 0.0
    ) -> RigidTransform:
        """Convert surface-relative SE(2) pose to world SE(3) pose.

        Takes a 2D position and rotation on the support surface and converts it to
        a full 3D pose in world coordinates. This is the key transformation for
        manipuland placement.

        Args:
            position_2d: 2D position on surface [x, y] in surface frame (meters).
            rotation_2d: Rotation around surface normal in radians.
            z_offset: Vertical offset from surface plane (meters). Use negative
                values to place objects below the surface plane (e.g., to
                compensate for gravity settling offset for thin coverings).

        Returns:
            RigidTransform representing the object's pose in world coordinates.
        """
        # Create surface-relative pose.
        # Z=0 is the surface plane, z_offset adjusts from there.
        surface_relative_pose = RigidTransform(
            p=[float(position_2d[0]), float(position_2d[1]), z_offset],
            rpy=RollPitchYaw([0.0, 0.0, rotation_2d]),
        )

        # Compose with surface transform to get world pose.
        world_pose = self.transform @ surface_relative_pose

        return world_pose

    def from_world_pose(
        self, world_transform: RigidTransform
    ) -> tuple[np.ndarray, float]:
        """Convert world SE(3) pose back to surface-relative SE(2).

        Inverse of to_world_pose(). Used after physics resolution to update
        placement_info with new positions.

        Args:
            world_transform: Object's pose in world coordinates.

        Returns:
            Tuple of (position_2d, rotation_2d) where:
            - position_2d: 2D position [x, y] in surface frame (meters)
            - rotation_2d: Rotation around surface normal in radians
        """
        # Compute surface-relative transform.
        surface_relative = self.transform.inverse() @ world_transform

        # Extract 2D position (XY in surface frame).
        position_2d = surface_relative.translation()[:2].copy()

        # Extract yaw rotation (rotation around surface normal / Z-axis).
        rotation_2d = RollPitchYaw(surface_relative.rotation()).yaw_angle()

        return position_2d, rotation_2d

    def contains_point_2d(self, position_2d: np.ndarray) -> bool:
        """Check if a 2D point lies within the surface convex hull.

        Uses the surface mesh's convex hull for accurate placement bounds on
        non-rectangular surfaces. This prevents placing objects outside the actual
        support surface geometry.

        Args:
            position_2d: 2D position to check [x, y] in surface frame (meters).

        Returns:
            True if the point is within the convex hull, False otherwise.

        Raises:
            ValueError: If surface has no mesh (mesh is required for convex hull).
        """
        assert position_2d.shape == (
            2,
        ), f"Expected 2D position, got shape {position_2d.shape}"

        # Fallback to AABB bounds check if no mesh geometry available.
        # This is the case for HSSD pre-validated surfaces.
        if self.mesh is None:
            console_logger.debug(
                f"Surface {self.surface_id} has no mesh geometry, using AABB bounds check"
            )
            in_x = (
                self.bounding_box_min[0] <= position_2d[0] <= self.bounding_box_max[0]
            )
            in_y = (
                self.bounding_box_min[1] <= position_2d[1] <= self.bounding_box_max[1]
            )
            return in_x and in_y

        # Both position_2d and self.mesh.vertices are in surface-local coordinates.
        # The mesh vertices were transformed to surface-local frame during surface creation.
        # No coordinate transformation needed - just use position_2d directly.
        point_xy = position_2d

        # Extract 2D vertices from mesh (XY plane in surface-local frame).
        mesh_xy_vertices = self.mesh.vertices[:, :2]

        # Compute 2D convex hull using safe wrapper.
        hull, processed_vertices = safe_convex_hull_2d(mesh_xy_vertices)
        if hull is None:
            console_logger.warning(
                f"Degenerate convex hull for surface {self.surface_id}. "
                "Falling back to AABB bounds check."
            )
            # Fallback: Check against bounding box instead.
            in_x = (
                self.bounding_box_min[0] <= position_2d[0] <= self.bounding_box_max[0]
            )
            in_y = (
                self.bounding_box_min[1] <= position_2d[1] <= self.bounding_box_max[1]
            )
            return in_x and in_y

        # Point-in-polygon test using convex hull.
        # For a convex polygon, point is inside if it's on the same side of all edges.
        # We use the cross product test for each edge.
        hull_vertices = processed_vertices[hull.vertices]
        n_vertices = len(hull_vertices)

        for i in range(n_vertices):
            # Get edge from vertex i to vertex (i+1) % n.
            v1 = hull_vertices[i]
            v2 = hull_vertices[(i + 1) % n_vertices]

            # Edge vector.
            edge = v2 - v1

            # Vector from v1 to test point.
            to_point = point_xy - v1

            # Cross product (2D): edge × to_point.
            # Positive means point is to the left of edge (inside for CCW hull).
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]

            # If point is to the right of any edge, it's outside the polygon.
            # Use small epsilon to allow points on boundary.
            epsilon = 1e-9
            if cross < -epsilon:
                return False

        return True

    def get_xy_convex_hull(self) -> VPolytope:
        """Get the XY boundary of this surface in world frame as a Drake VPolytope.

        Transforms surface boundary from local (surface) frame to world frame,
        then projects to world XY plane. This is critical for correct IK
        constraints when the surface has non-zero yaw rotation.

        Uses the convex hull of the surface mesh for accurate representation
        of round/irregular surfaces. Falls back to bounding box if no mesh.

        Returns:
            VPolytope representing the 2D XY boundary of the surface in world frame.
        """
        R = self.transform.rotation().matrix()
        t = self.transform.translation()

        if self.mesh is None:
            # No mesh - use axis-aligned bounding box corners in local frame.
            bbox_min = self.bounding_box_min
            bbox_max = self.bounding_box_max
            corners_local = np.array(
                [
                    [bbox_min[0], bbox_min[1], 0.0],
                    [bbox_max[0], bbox_min[1], 0.0],
                    [bbox_max[0], bbox_max[1], 0.0],
                    [bbox_min[0], bbox_max[1], 0.0],
                ]
            )
            # Transform to world frame.
            corners_world = (R @ corners_local.T).T + t
            corners_world_xy = corners_world[:, :2]

            hull, processed_vertices = safe_convex_hull_2d(corners_world_xy)
            if hull is None:
                # Degenerate - use AABB of transformed corners.
                lb = corners_world_xy.min(axis=0)
                ub = corners_world_xy.max(axis=0)
                return VPolytope.MakeBox(lb=lb, ub=ub)

            hull_vertices = processed_vertices[hull.vertices]
            return VPolytope(vertices=hull_vertices.T)

        # Transform mesh vertices to world frame.
        mesh_world = (R @ self.mesh.vertices.T).T + t
        mesh_world_xy = mesh_world[:, :2]

        hull, processed_vertices = safe_convex_hull_2d(mesh_world_xy)
        if hull is None:
            # Degenerate hull - fall back to AABB of transformed mesh.
            lb = mesh_world_xy.min(axis=0)
            ub = mesh_world_xy.max(axis=0)
            return VPolytope.MakeBox(lb=lb, ub=ub)

        hull_vertices = processed_vertices[hull.vertices]
        # VPolytope expects 2xN array (dim x num_vertices).
        return VPolytope(vertices=hull_vertices.T)


@dataclass
class SceneObject:
    """Represents a single object in the scene."""

    object_id: UniqueID
    """Unique identifier for the object."""

    object_type: ObjectType
    """Type of object (furniture or manipuland)."""

    name: str
    """Human-readable name of the object (e.g., 'Dining Table')."""

    description: str
    """Text description used for asset generation (e.g., 'A wooden table')."""

    transform: RigidTransform
    """3D pose of the object in world coordinates."""

    geometry_path: Path | None = None
    """Path to the 3D geometry file (e.g., GLB, OBJ)."""

    sdf_path: Path | None = None
    """Path to the Drake SDF file for simulation."""

    image_path: Path | None = None
    """Path to the reference image used for asset generation."""

    support_surfaces: list[SupportSurface] = field(default_factory=list)
    """Support surfaces where other objects can be placed on this object."""

    placement_info: PlacementInfo | None = None
    """Placement metadata for objects placed on support surfaces.

    For manipulands placed on furniture surfaces, this stores the surface-relative
    placement information (parent surface, 2D position, rotation). For furniture
    placed directly on the floor, this is None.
    """

    metadata: dict[str, str | float | bool] = field(default_factory=dict)
    """Additional metadata for the object (e.g., dimensions, material)."""

    bbox_min: np.ndarray | None = None
    """Object-frame AABB minimum corner (x, y, z)."""

    bbox_max: np.ndarray | None = None
    """Object-frame AABB maximum corner (x, y, z)."""

    immutable: bool = False
    """Whether this object is immutable (cannot be moved or removed)."""

    scale_factor: float = 1.0
    """Cumulative scale factor applied to this object's asset (1.0 = original size)."""

    def apply_scale(self, new_scale: float) -> None:
        """Apply scale factor to this object's bounding box and invalidate surfaces.

        This updates the object-frame bounding box and invalidates support surfaces
        (which will need to be recomputed after rescaling).

        Args:
            new_scale: Scale multiplier to apply (e.g., 1.5 = 50% larger).
        """
        if self.bbox_min is not None:
            self.bbox_min = self.bbox_min * new_scale
        if self.bbox_max is not None:
            self.bbox_max = self.bbox_max * new_scale

        self.support_surfaces = []  # Invalidate - must be recomputed.
        self.scale_factor = self.scale_factor * new_scale

    def compute_world_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute world-frame AABB from object-frame bounds and transform.

        Returns:
            Tuple of (world_bbox_min, world_bbox_max) or None if no bounds available.
        """
        if self.bbox_min is None or self.bbox_max is None:
            return None

        # Generate all 8 corners of the object-frame bounding box.
        corners = compute_aabb_corners(self.bbox_min, self.bbox_max)

        # Transform all corners to world coordinates.
        world_corners = []
        for corner in corners:
            world_corner = self.transform @ corner
            world_corners.append(world_corner)

        world_corners = np.array(world_corners)

        # Find min and max in each dimension.
        world_bbox_min = np.min(world_corners, axis=0)
        world_bbox_max = np.max(world_corners, axis=0)

        return world_bbox_min, world_bbox_max

    def content_hash(self) -> str:
        """Generate content hash for this scene object."""
        obj_dict = {
            "object_id": str(self.object_id),
            "name": self.name,
            "description": self.description,
            "object_type": self.object_type.value if self.object_type else "",
            "transform": serialize_rigid_transform(self.transform),
            "geometry_path": str(self.geometry_path) if self.geometry_path else "",
            "sdf_path": str(self.sdf_path) if self.sdf_path else "",
            "image_path": str(self.image_path) if self.image_path else "",
            "support_surfaces": [surf.content_hash() for surf in self.support_surfaces],
            "placement_info": (
                {
                    "parent_surface_id": str(self.placement_info.parent_surface_id),
                    "position_2d": self.placement_info.position_2d.tolist(),
                    "rotation_2d": float(self.placement_info.rotation_2d),
                    "placement_method": self.placement_info.placement_method,
                }
                if self.placement_info
                else None
            ),
            "metadata": dict(sorted(self.metadata.items())),  # Sort for determinism
            "bbox_min": self.bbox_min.tolist() if self.bbox_min is not None else None,
            "bbox_max": self.bbox_max.tolist() if self.bbox_max is not None else None,
            "immutable": self.immutable,
            "scale_factor": self.scale_factor,
        }

        # Hash file contents if they exist.
        for path_key in ["geometry_path", "sdf_path"]:
            path_str = obj_dict[path_key]
            if path_str:
                try:
                    path = Path(path_str)
                    if path.exists():
                        # Determine if file is binary or text based on extension.
                        binary_extensions = {".glb", ".obj", ".ply", ".stl"}
                        is_binary = path.suffix.lower() in binary_extensions

                        if is_binary:
                            # Read binary files in binary mode.
                            with open(path, "rb") as f:
                                content = f.read()
                            obj_dict[f"{path_key}_content_hash"] = hashlib.sha256(
                                content
                            ).hexdigest()
                        else:
                            # Read text files (SDF, XML) in text mode.
                            with open(path, "r", encoding="utf-8") as f:
                                content = f.read()
                            obj_dict[f"{path_key}_content_hash"] = hashlib.sha256(
                                content.encode()
                            ).hexdigest()
                    else:
                        obj_dict[f"{path_key}_content_hash"] = ""
                except Exception as e:
                    console_logger.warning(
                        f"Could not hash file content for {path_str}: {e}"
                    )
                    obj_dict[f"{path_key}_content_hash"] = ""

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(obj_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_dict(self, scene_dir: Path | None = None) -> dict[str, Any]:
        """
        Serialize SceneObject to dictionary.

        Args:
            scene_dir: Optional scene directory for path relativization.
                       If None, paths are stored as absolute paths.

        Returns:
            Dictionary containing complete object state.
        """
        # Serialize support surfaces.
        support_surfaces_data = []
        for surf in self.support_surfaces:
            surf_dict = {
                "surface_id": str(surf.surface_id),
                "bounding_box_min": surf.bounding_box_min.tolist(),
                "bounding_box_max": surf.bounding_box_max.tolist(),
                "transform": serialize_rigid_transform(surf.transform),
                "link_name": surf.link_name,  # For articulated FK transforms.
            }
            # Serialize mesh data if present.
            if surf.mesh is not None:
                surf_dict["mesh"] = {
                    "vertices": surf.mesh.vertices.tolist(),
                    "faces": surf.mesh.faces.tolist(),
                }
            support_surfaces_data.append(surf_dict)

        # Convert paths (relative or absolute).
        geometry_path_str = (
            safe_relative_path(self.geometry_path, scene_dir)
            if self.geometry_path
            else None
        )
        sdf_path_str = (
            safe_relative_path(self.sdf_path, scene_dir) if self.sdf_path else None
        )
        image_path_str = (
            safe_relative_path(self.image_path, scene_dir) if self.image_path else None
        )

        return {
            "object_id": str(self.object_id),
            "object_type": self.object_type.value,
            "name": self.name,
            "description": self.description,
            "transform": serialize_rigid_transform(self.transform),
            "geometry_path": geometry_path_str,
            "sdf_path": sdf_path_str,
            "image_path": image_path_str,
            "support_surfaces": support_surfaces_data,
            "placement_info": (
                {
                    "parent_surface_id": str(self.placement_info.parent_surface_id),
                    "position_2d": self.placement_info.position_2d.tolist(),
                    "rotation_2d": float(self.placement_info.rotation_2d),
                    "placement_method": self.placement_info.placement_method,
                }
                if self.placement_info
                else None
            ),
            "metadata": self.metadata,
            "bbox_min": self.bbox_min.tolist() if self.bbox_min is not None else None,
            "bbox_max": self.bbox_max.tolist() if self.bbox_max is not None else None,
            "immutable": self.immutable,
            "scale_factor": self.scale_factor,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], scene_dir: Path | None = None
    ) -> "SceneObject":
        """
        Deserialize SceneObject from dictionary.

        Args:
            data: Dictionary containing object state.
            scene_dir: Optional scene directory for path resolution.
                       If None, paths are treated as absolute.

        Returns:
            Reconstructed SceneObject instance.
        """
        # Reconstruct transform.
        transform_data = data["transform"]
        translation = np.array(transform_data["translation"])
        rotation_wxyz = transform_data["rotation_wxyz"]
        quaternion = Quaternion(wxyz=rotation_wxyz)
        rotation_matrix = RotationMatrix(quaternion)
        transform = RigidTransform(rotation_matrix, translation)

        # Reconstruct support surfaces.
        support_surfaces = []
        for surf_data in data["support_surfaces"]:
            surf_transform_data = surf_data["transform"]
            surf_translation = np.array(surf_transform_data["translation"])
            surf_rotation_wxyz = surf_transform_data["rotation_wxyz"]
            surf_quaternion = Quaternion(wxyz=surf_rotation_wxyz)
            surf_rotation_matrix = RotationMatrix(surf_quaternion)
            surf_transform = RigidTransform(surf_rotation_matrix, surf_translation)

            # Reconstruct mesh if present.
            mesh = None
            if "mesh" in surf_data and surf_data["mesh"] is not None:
                import trimesh

                mesh = trimesh.Trimesh(
                    vertices=np.array(surf_data["mesh"]["vertices"]),
                    faces=np.array(surf_data["mesh"]["faces"]),
                )

            support_surface = SupportSurface(
                surface_id=UniqueID(surf_data["surface_id"]),
                bounding_box_min=np.array(surf_data["bounding_box_min"]),
                bounding_box_max=np.array(surf_data["bounding_box_max"]),
                transform=surf_transform,
                mesh=mesh,
                link_name=surf_data.get("link_name"),  # For articulated FK transforms.
            )
            support_surfaces.append(support_surface)

        # Resolve paths.
        geometry_path = None
        if data["geometry_path"]:
            geometry_path = (
                scene_dir / data["geometry_path"]
                if scene_dir
                else Path(data["geometry_path"])
            )

        sdf_path = None
        if data["sdf_path"]:
            sdf_path = (
                scene_dir / data["sdf_path"] if scene_dir else Path(data["sdf_path"])
            )

        image_path = None
        if data["image_path"]:
            image_path = (
                scene_dir / data["image_path"]
                if scene_dir
                else Path(data["image_path"])
            )

        # Reconstruct placement_info.
        placement_info = None
        if data.get("placement_info"):
            placement_data = data["placement_info"]
            placement_info = PlacementInfo(
                parent_surface_id=UniqueID(placement_data["parent_surface_id"]),
                position_2d=np.array(placement_data["position_2d"]),
                rotation_2d=float(placement_data["rotation_2d"]),
                placement_method=placement_data["placement_method"],
            )

        return cls(
            object_id=UniqueID(data["object_id"]),
            object_type=ObjectType(data["object_type"]),
            name=data["name"],
            description=data["description"],
            transform=transform,
            geometry_path=geometry_path,
            sdf_path=sdf_path,
            image_path=image_path,
            support_surfaces=support_surfaces,
            placement_info=placement_info,
            metadata=data["metadata"],
            bbox_min=np.array(data["bbox_min"]) if data["bbox_min"] else None,
            bbox_max=np.array(data["bbox_max"]) if data["bbox_max"] else None,
            immutable=data["immutable"],
            scale_factor=data.get("scale_factor", 1.0),
        )
