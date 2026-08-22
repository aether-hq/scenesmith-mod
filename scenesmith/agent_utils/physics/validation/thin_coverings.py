"""Physics validation for scenes using Drake."""

import logging

from dataclasses import dataclass

import numpy as np

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    AgentType,
    ObjectType,
    UniqueID,
)

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.validation.collision_filtering import (
    _get_furniture_id_for_manipuland,
)


def _get_thin_covering_owner_agent(
    covering_id: str, scene: RoomScene
) -> AgentType | None:
    """Determine which agent type owns a thin covering.

    Args:
        covering_id: UniqueID of the thin covering.
        scene: RoomScene for object lookups.

    Returns:
        AgentType that owns the covering, or None if unknown.
    """
    scene_obj = scene.objects.get(UniqueID(covering_id))
    if not scene_obj:
        return None

    # Wall coverings are owned by wall agent.
    if scene_obj.metadata.get("is_wall_covering", False):
        return AgentType.WALL_MOUNTED

    # Check object type - this tells us which agent placed it.
    if scene_obj.object_type == ObjectType.MANIPULAND:
        return AgentType.MANIPULAND
    elif scene_obj.object_type == ObjectType.FURNITURE:
        # Floor coverings (rugs, carpets) are placed by furniture agent.
        return AgentType.FURNITURE

    return None


def filter_thin_covering_overlaps_by_agent(
    overlaps: list["ThinCoveringOverlap"],
    scene: RoomScene,
    agent_type: AgentType,
    current_furniture_id: UniqueID | None = None,
) -> list["ThinCoveringOverlap"]:
    """Filter thin covering overlaps to show only those relevant to the agent.

    Args:
        overlaps: List of thin covering overlaps to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.
        current_furniture_id: For ManipulandAgent, the furniture being populated.

    Returns:
        Filtered list of overlaps.
    """
    filtered = []
    for overlap in overlaps:
        owner_a = _get_thin_covering_owner_agent(
            covering_id=overlap.covering_a_id, scene=scene
        )
        owner_b = _get_thin_covering_owner_agent(
            covering_id=overlap.covering_b_id, scene=scene
        )

        # Check if at least one covering belongs to this agent.
        if owner_a == agent_type or owner_b == agent_type:
            # For manipuland agent, additionally check if on current furniture.
            if agent_type == AgentType.MANIPULAND and current_furniture_id is not None:
                # Check if at least one covering is on current furniture.
                def is_on_current_furniture(covering_id: str) -> bool:
                    obj = scene.objects.get(UniqueID(covering_id))
                    if not obj or not obj.placement_info:
                        return False
                    parent_id = _get_furniture_id_for_manipuland(
                        manipuland_id=covering_id, scene=scene
                    )
                    return parent_id == str(current_furniture_id)

                if is_on_current_furniture(
                    overlap.covering_a_id
                ) or is_on_current_furniture(overlap.covering_b_id):
                    filtered.append(overlap)
            else:
                filtered.append(overlap)

    if len(filtered) < len(overlaps):
        console_logger.debug(
            f"Filtered thin covering overlaps by agent type {agent_type.value}: "
            f"{len(overlaps)} -> {len(filtered)}"
        )

    return filtered


def filter_thin_covering_boundary_violations_by_agent(
    violations: list["ThinCoveringBoundaryViolation"],
    agent_type: AgentType,
) -> list["ThinCoveringBoundaryViolation"]:
    """Filter thin covering boundary violations by agent type.

    Only floor coverings can have boundary violations (extending beyond room walls).
    Wall coverings and surface coverings don't have floor boundary constraints.

    Args:
        violations: List of boundary violations to filter.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations (only FurnitureAgent sees floor covering violations).
    """
    # Only furniture agent places floor coverings, which are subject to room boundaries.
    if agent_type != AgentType.FURNITURE:
        return []

    # For furniture agent, show all floor covering boundary violations.
    return violations


@dataclass
class ThinCoveringOverlap:
    """Represents an overlap between two thin covering objects."""

    covering_a_name: str
    """Name of the first thin covering."""

    covering_a_id: str
    """UniqueID as string of the first thin covering."""

    covering_b_name: str
    """Name of the second thin covering."""

    covering_b_id: str
    """UniqueID as string of the second thin covering."""

    def to_description(self) -> str:
        """Format for human/VLM consumption."""
        return (
            f"Thin covering '{self.covering_a_name}' [{self.covering_a_id}] overlaps "
            f"with '{self.covering_b_name}' [{self.covering_b_id}]"
        )


@dataclass
class ThinCoveringBoundaryViolation:
    """Represents a thin covering extending beyond floor plan boundaries."""

    covering_id: str
    """UniqueID as string of the violating thin covering."""

    exceeded_boundaries: list[str]
    """List of boundary names exceeded, e.g., ["north", "east"]."""

    def to_description(self) -> str:
        """Format for human/VLM consumption."""
        boundaries_str = ", ".join(self.exceeded_boundaries)
        suffix = "boundaries" if len(self.exceeded_boundaries) > 1 else "boundary"
        return f"Thin covering [{self.covering_id}] extends beyond {boundaries_str} {suffix}"


def _get_obb_corners_2d(
    center_x: float, center_y: float, half_w: float, half_d: float, yaw: float
) -> np.ndarray:
    """Compute 2D OBB corner points given center, half-extents, and rotation.

    Args:
        center_x: X position of center.
        center_y: Y position of center.
        half_w: Half-width (X extent before rotation).
        half_d: Half-depth (Y extent before rotation).
        yaw: Rotation angle in radians (around Z axis).

    Returns:
        Array of shape (4, 2) with corner points in counter-clockwise order.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Local corners (before rotation).
    local_corners = np.array(
        [
            [-half_w, -half_d],
            [+half_w, -half_d],
            [+half_w, +half_d],
            [-half_w, +half_d],
        ]
    )

    # Rotation matrix.
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    # Transform to world space.
    world_corners = (rot @ local_corners.T).T + np.array([center_x, center_y])
    return world_corners


def _obb_overlap_2d(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """Check if two 2D OBBs overlap using Separating Axis Theorem.

    Args:
        corners_a: Array of shape (4, 2) with corners of first OBB.
        corners_b: Array of shape (4, 2) with corners of second OBB.

    Returns:
        True if OBBs overlap, False otherwise.
    """

    def get_axes(corners: np.ndarray) -> list[np.ndarray]:
        """Get edge normal axes for SAT test."""
        axes = []
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            # Perpendicular (normal) to edge.
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        return axes

    def project(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        """Project corners onto axis, return min and max."""
        dots = corners @ axis
        return dots.min(), dots.max()

    # Test all 4 axes (2 from each OBB).
    for axis in get_axes(corners_a) + get_axes(corners_b):
        min_a, max_a = project(corners_a, axis)
        min_b, max_b = project(corners_b, axis)

        # Check for separation on this axis.
        if max_a < min_b or max_b < min_a:
            return False

    return True


def compute_thin_covering_overlaps(scene: RoomScene) -> list[ThinCoveringOverlap]:
    """Check for overlapping floor/manipuland thin coverings using 2D OBB intersection.

    Floor thin coverings (rugs, carpets) and manipuland thin coverings
    (tablecloths, placemats) have no collision geometry (purely decorative),
    so they won't appear in Drake collision detection. This function performs
    a separate check for thin covering overlaps using oriented bounding boxes.

    Wall thin coverings (paintings, posters) are excluded - they have collision
    geometry and use Drake collision detection instead.

    Args:
        scene: RoomScene containing thin covering objects.

    Returns:
        List of ThinCoveringOverlap objects representing detected overlaps.
    """
    coverings = []
    for obj_id, obj in scene.objects.items():
        if obj.metadata.get("asset_source") != "thin_covering":
            continue

        # Wall coverings use Drake collision detection (have collision geometry).
        # Skip them here - only check floor/manipuland thin coverings.
        if obj.metadata.get("is_wall_covering", False):
            continue

        width = obj.metadata.get("width_m")
        depth = obj.metadata.get("depth_m")
        if width is None or depth is None:
            console_logger.error(
                f"Thin covering '{obj.name}' [{obj_id}] missing width_m/depth_m "
                "metadata, skipping overlap check"
            )
            continue

        transform = obj.transform
        pos = transform.translation()

        rot_matrix = transform.rotation().matrix()
        yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

        corners = _get_obb_corners_2d(
            center_x=pos[0],
            center_y=pos[1],
            half_w=width / 2.0,
            half_d=depth / 2.0,
            yaw=yaw,
        )

        coverings.append({"id": obj_id, "name": obj.name, "corners": corners})

    overlaps: list[ThinCoveringOverlap] = []
    for i in range(len(coverings)):
        for j in range(i + 1, len(coverings)):
            a = coverings[i]
            b = coverings[j]

            if _obb_overlap_2d(a["corners"], b["corners"]):
                overlaps.append(
                    ThinCoveringOverlap(
                        covering_a_name=a["name"],
                        covering_a_id=str(a["id"]),
                        covering_b_name=b["name"],
                        covering_b_id=str(b["id"]),
                    )
                )

    if overlaps:
        console_logger.info(f"Found {len(overlaps)} thin covering overlap(s)")
        for overlap in overlaps:
            console_logger.info(f"  {overlap.to_description()}")

    return overlaps


def compute_thin_covering_boundary_violations(
    scene: RoomScene, wall_thickness: float
) -> list[ThinCoveringBoundaryViolation]:
    """Check if floor thin coverings extend beyond the usable floor area.

    Floor thin coverings (rugs, carpets) have no collision geometry (purely
    decorative), so they won't appear in Drake collision detection. This
    function checks if thin coverings extend beyond floor plan boundaries,
    accounting for wall thickness.

    Wall thin coverings (paintings, posters) are excluded - they're mounted
    on walls, not on floors, and use Drake collision detection.

    The usable floor area is smaller than the room dimensions because walls
    extend inward by wall_thickness/2 on each side.

    Args:
        scene: RoomScene containing thin covering objects and room_geometry.
        wall_thickness: Wall thickness in meters. Walls extend inward by
            wall_thickness/2 from the room boundary.

    Returns:
        List of ThinCoveringBoundaryViolation objects representing detected violations.
    """
    # Compute usable floor bounds (inside walls).
    half_wall = wall_thickness / 2.0
    room_length = scene.room_geometry.length  # x-dimension
    room_width = scene.room_geometry.width  # y-dimension

    inner_min_x = -room_length / 2.0 + half_wall
    inner_max_x = room_length / 2.0 - half_wall
    inner_min_y = -room_width / 2.0 + half_wall
    inner_max_y = room_width / 2.0 - half_wall

    violations: list[ThinCoveringBoundaryViolation] = []

    for obj_id, obj in scene.objects.items():
        if obj.metadata.get("asset_source") != "thin_covering":
            continue

        # Wall coverings are on walls, not floors - skip floor boundary check.
        if obj.metadata.get("is_wall_covering", False):
            continue

        width = obj.metadata.get("width_m")
        depth = obj.metadata.get("depth_m")
        shape = obj.metadata.get("shape", "rectangular")

        if width is None or depth is None:
            raise ValueError(
                f"Thin covering [{obj_id}] missing width_m/depth_m metadata. "
                f"This is a bug in thin covering creation."
            )

        transform = obj.transform
        pos = transform.translation()
        center_x, center_y = pos[0], pos[1]

        exceeded_boundaries: list[str] = []

        if shape == "circular":
            # For circular thin coverings, use radius = min(width, depth) / 2.
            radius = min(width, depth) / 2.0

            if center_x - radius < inner_min_x:
                exceeded_boundaries.append("west")
            if center_x + radius > inner_max_x:
                exceeded_boundaries.append("east")
            if center_y - radius < inner_min_y:
                exceeded_boundaries.append("south")
            if center_y + radius > inner_max_y:
                exceeded_boundaries.append("north")
        else:
            # For rectangular thin coverings, compute OBB corners.
            rot_matrix = transform.rotation().matrix()
            yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

            corners = _get_obb_corners_2d(
                center_x=center_x,
                center_y=center_y,
                half_w=width / 2.0,
                half_d=depth / 2.0,
                yaw=yaw,
            )

            # Check if any corner exceeds bounds.
            min_corner_x = np.min(corners[:, 0])
            max_corner_x = np.max(corners[:, 0])
            min_corner_y = np.min(corners[:, 1])
            max_corner_y = np.max(corners[:, 1])

            if min_corner_x < inner_min_x:
                exceeded_boundaries.append("west")
            if max_corner_x > inner_max_x:
                exceeded_boundaries.append("east")
            if min_corner_y < inner_min_y:
                exceeded_boundaries.append("south")
            if max_corner_y > inner_max_y:
                exceeded_boundaries.append("north")

        if exceeded_boundaries:
            # Sort for consistent output.
            exceeded_boundaries.sort()
            violations.append(
                ThinCoveringBoundaryViolation(
                    covering_id=str(obj_id),
                    exceeded_boundaries=exceeded_boundaries,
                )
            )

    if violations:
        console_logger.info(
            f"Found {len(violations)} thin covering boundary violation(s)"
        )
        for v in violations:
            console_logger.info(f"  {v.to_description()}")

    return violations
