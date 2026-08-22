"""Fill container utilities using physics simulation.

This module provides functionality for:
- Computing container interior bounds using top rim heuristic.
- Computing fill object spawn transforms.
- Resolving initial fill object collisions using NLP projection.
- Simulating fill objects dropping into containers using Drake physics.
"""

import logging

import numpy as np
import trimesh

from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix

console_logger = logging.getLogger(__name__)

from scenesmith.manipuland_agents.tools.fill.bounds import ContainerInteriorBounds
from scenesmith.manipuland_agents.tools.fill.packing import _point_in_polygon


def _compute_object_extents(
    collision_meshes: list[trimesh.Trimesh],
) -> tuple[float, float, float]:
    """Compute axis-aligned bounding box extents from collision meshes.

    Returns:
        Tuple of (dx, dy, dz) extents along each axis.
    """
    all_vertices = np.vstack([m.vertices for m in collision_meshes])
    mins = all_vertices.min(axis=0)
    maxs = all_vertices.max(axis=0)
    return float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2])


def _should_flip_for_thick_end_up(
    vertices: np.ndarray, rotation: RotationMatrix
) -> bool:
    """Determine if object should be flipped 180° to put thick end up.

    After orienting an elongated object vertically, compares the XY footprint
    of the top half vs bottom half. If the bottom half has a larger footprint
    (thicker), the object should be flipped.

    This is important for asymmetric objects like utensils (pan flippers,
    spatulas, spoons) where the thick/business end should point upward when
    placed in a container like a utensil crock.

    Args:
        vertices: Original mesh vertices (Nx3).
        rotation: Rotation already applied to align longest axis with Z.

    Returns:
        True if object should be flipped 180° around X axis.
    """
    # Apply rotation to vertices.
    rotated = (rotation.matrix() @ vertices.T).T

    # Find center Z.
    z_min, z_max = rotated[:, 2].min(), rotated[:, 2].max()
    center_z = (z_min + z_max) / 2

    # Split into top and bottom halves.
    top_verts = rotated[rotated[:, 2] > center_z]
    bottom_verts = rotated[rotated[:, 2] <= center_z]

    # Handle edge cases.
    if len(top_verts) < 3 or len(bottom_verts) < 3:
        return False  # Not enough vertices to analyze.

    # Compute XY bounding box area for each half.
    def xy_area(verts: np.ndarray) -> float:
        x_extent = verts[:, 0].max() - verts[:, 0].min()
        y_extent = verts[:, 1].max() - verts[:, 1].min()
        return x_extent * y_extent

    top_area = xy_area(top_verts)
    bottom_area = xy_area(bottom_verts)

    # Flip if bottom is significantly thicker (10% threshold avoids flipping
    # symmetric objects).
    return bottom_area > top_area * 1.1


def _compute_fill_object_rotation(
    extents: tuple[float, float, float],
    container_interior: ContainerInteriorBounds,
    vertices: np.ndarray | None = None,
    aspect_ratio_threshold: float = 2.0,
) -> RotationMatrix:
    """Compute rotation for fill object based on shape and container geometry.

    Algorithm:
    1. Detect non-cubic objects using max/min ratio >= threshold.
    2. Stand up: rotate so shortest axis becomes horizontal.
    3. Align: apply yaw to align longest horizontal axis with container length.
    4. Apply thick-end-up flip if needed for asymmetric objects.

    Cubic objects (aspect ratio < threshold) return identity rotation.

    Args:
        extents: Object extents (dx, dy, dz).
        container_interior: Container interior bounds for alignment.
        vertices: Optional mesh vertices for thick-end-up analysis.
        aspect_ratio_threshold: Ratio of longest/shortest to be considered
            non-cubic and needing orientation.

    Returns:
        RotationMatrix to apply to the object.
    """
    dx, dy, dz = extents
    dims = [dx, dy, dz]

    # Sort dimensions to find shortest/middle/longest.
    sorted_with_idx = sorted(enumerate(dims), key=lambda x: x[1])
    shortest_idx, shortest_val = sorted_with_idx[0]
    middle_idx, middle_val = sorted_with_idx[1]
    longest_idx, longest_val = sorted_with_idx[2]

    # Check if non-cubic (needs special orientation).
    aspect_ratio = longest_val / shortest_val if shortest_val > 0 else 1.0
    if aspect_ratio < aspect_ratio_threshold:
        # Cubic object - no rotation needed.
        return RotationMatrix()

    # Step 1: Stand up - rotate so shortest axis becomes horizontal.
    # In Drake, Z is up by default. We want shortest axis in XY plane.
    if shortest_idx == 2:
        # Z is shortest (e.g., plate lying flat). Rotate 90° around X.
        # This makes: X→X, Y→Z (vertical), Z→-Y (horizontal).
        base_rotation = RotationMatrix(RollPitchYaw([np.pi / 2, 0.0, 0.0]))
        # After rotation: object's X is world X, object's Y is world Z,
        # object's Z is world -Y.
        # Horizontal axes are now: original X and original Z.
        horizontal_dims = [(dims[0], 0), (dims[2], 2)]  # X and Z extents.
    elif shortest_idx == 1:
        # Y is shortest. Rotate 90° around Z then 90° around X.
        # Simpler: rotate -90° around X to make Y horizontal.
        # Y→-Z (horizontal), Z→Y (still horizontal), X→X.
        # Actually: rotate 90° around Z to swap X and Y, then proceed.
        # Simplest: just rotate so Y ends up horizontal.
        # Rotate around Z by 90°: X→Y, Y→-X, Z→Z. Then around X by 90°.
        # Let's use: rotate around Z by 90° to make Y the new X (horizontal).
        base_rotation = RotationMatrix(RollPitchYaw([0.0, 0.0, np.pi / 2]))
        # After: X→Y, Y→-X, Z→Z. Shortest (Y) is now along -X (horizontal).
        # Vertical axis is Z. Horizontal axes are: original Y (now -X) and Z.
        horizontal_dims = [(dims[1], 1), (dims[2], 2)]
    else:
        # X is shortest. Already horizontal in default orientation.
        base_rotation = RotationMatrix()
        # Vertical axis is Z. Horizontal axes are X and Y.
        # Wait, if X is shortest, what's vertical? We need to stand up.
        # "Stand up" means the tall axis should be vertical.
        # If X is shortest, then Y or Z is longest. Z might already be vertical.
        # We want shortest horizontal, longest vertical if possible.
        # If X is shortest and Z is longest: Z is already vertical, good.
        # If X is shortest and Y is longest: rotate to make Y vertical.
        if longest_idx == 1:
            # Y is longest, X is shortest. Rotate 90° around X to make Y vertical.
            base_rotation = RotationMatrix(RollPitchYaw([np.pi / 2, 0.0, 0.0]))
            # After: X→X, Y→Z, Z→-Y. Horizontal axes: X and Z (original).
            horizontal_dims = [(dims[0], 0), (dims[2], 2)]
        else:
            # Z is longest (or middle). Z is vertical, X is horizontal.
            base_rotation = RotationMatrix()
            # Horizontal axes: X and Y.
            horizontal_dims = [(dims[0], 0), (dims[1], 1)]

    # Step 2: Align longest horizontal axis with container length direction.
    # Determine container length direction from hull.
    hull = container_interior.hull_vertices_2d
    x_extent = float(hull[:, 0].max() - hull[:, 0].min())
    y_extent = float(hull[:, 1].max() - hull[:, 1].min())
    container_length_along_x = x_extent >= y_extent

    # Find longest horizontal dimension after base rotation.
    longest_horiz_val, longest_horiz_orig_idx = max(horizontal_dims, key=lambda x: x[0])

    # Determine yaw to align longest horizontal with container length.
    # After base_rotation, we need to know where the longest horizontal axis ended up.
    # Apply base rotation to unit vectors to find current orientation.
    R = base_rotation.matrix()
    orig_axes = np.eye(3)
    rotated_axes = R @ orig_axes  # Columns are rotated X, Y, Z axes.

    # Find which world axis the longest original horizontal axis aligns with.
    longest_horiz_world_dir = rotated_axes[:, longest_horiz_orig_idx]

    # We want this direction to align with container length (X or Y).
    # Compute yaw angle needed.
    # Current direction in XY plane:
    current_angle = np.arctan2(longest_horiz_world_dir[1], longest_horiz_world_dir[0])
    # Target direction:
    target_angle = 0.0 if container_length_along_x else np.pi / 2

    yaw = target_angle - current_angle

    # Apply yaw rotation.
    yaw_rotation = RotationMatrix(RollPitchYaw([0.0, 0.0, yaw]))
    rotation = yaw_rotation @ base_rotation

    # Step 3: Apply thick-end-up check for asymmetric objects.
    if vertices is not None and _should_flip_for_thick_end_up(vertices, rotation):
        # Flip 180° around X axis to put thick end up.
        flip = RotationMatrix(RollPitchYaw([np.pi, 0.0, 0.0]))
        rotation = flip @ rotation

    return rotation


def compute_fill_spawn_transforms(
    fill_collision_meshes: list[list[trimesh.Trimesh]],
    container_interior: ContainerInteriorBounds,
    container_transform: RigidTransform,
    spawn_height_above_rim: float = 0.1,
    height_stagger_fraction: float = 0.75,
    min_height_stagger: float = 0.02,
    rng: np.random.Generator | None = None,
) -> list[RigidTransform]:
    """Compute initial spawn transforms for fill objects.

    Fill objects spawn at staggered heights above the container rim to prevent
    initial overlaps. Each subsequent object spawns higher based on its actual
    post-rotation height (not diagonal, since rotation is deterministic).

    Objects are oriented based on their shape and container geometry:
    - Non-cubic objects (max/min >= 2): rotated to stand up with shortest axis
      horizontal, longest horizontal axis aligned with container length.
    - Cubic objects: no rotation applied.

    Args:
        fill_collision_meshes: List of collision mesh lists for each fill object.
        container_interior: Container interior bounds.
        container_transform: Transform of container in world frame.
        spawn_height_above_rim: Base height above container top for first object.
        height_stagger_fraction: Fraction of post-rotation height for Z spacing.
        min_height_stagger: Minimum stagger between objects (meters).
        rng: Random number generator (uses default if None).

    Returns:
        List of world transforms for each fill object.
    """
    if rng is None:
        rng = np.random.default_rng()

    transforms = []
    hull = container_interior.hull_vertices_2d

    # Compute bounding box of hull for rejection sampling.
    hull_min = hull.min(axis=0)
    hull_max = hull.max(axis=0)

    # Track current spawn Z level (accumulates with per-object stagger).
    current_z = container_interior.top_z + spawn_height_above_rim

    for obj_index, meshes in enumerate(fill_collision_meshes):
        # Compute object extents.
        extents = _compute_object_extents(meshes)
        all_vertices = np.vstack([m.vertices for m in meshes])

        # Generate random XY within hull using rejection sampling.
        max_attempts = 100
        for _ in range(max_attempts):
            x = rng.uniform(hull_min[0], hull_max[0])
            y = rng.uniform(hull_min[1], hull_max[1])
            point = np.array([x, y])

            # Check if point is inside hull.
            if _point_in_polygon(point=point, polygon=hull):
                break
        else:
            # Fallback to centroid if rejection sampling fails.
            x, y = container_interior.centroid_2d
            console_logger.warning("Rejection sampling failed, using centroid")

        # Compute rotation based on object shape and container geometry.
        rotation = _compute_fill_object_rotation(
            extents=extents,
            container_interior=container_interior,
            vertices=all_vertices,
        )

        # Compute post-rotation bounding box.
        rotated_vertices = (rotation.matrix() @ all_vertices.T).T
        rotated_z_min = float(rotated_vertices[:, 2].min())
        rotated_z_max = float(rotated_vertices[:, 2].max())
        rotated_height = rotated_z_max - rotated_z_min

        # Compute spawn Z: object's z_min sits at current layer.
        spawn_z = current_z - rotated_z_min

        # Create local transform (in container frame).
        local_transform = RigidTransform(rotation, [float(x), float(y), float(spawn_z)])

        # Convert to world frame.
        world_transform = container_transform @ local_transform
        transforms.append(world_transform)

        # Stagger by actual height (rotation is deterministic, not random).
        stagger = max(min_height_stagger, rotated_height * height_stagger_fraction)
        current_z += stagger

    return transforms
