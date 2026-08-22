"""Fill container utilities using physics simulation.

This module provides functionality for:
- Computing container interior bounds using top rim heuristic.
- Computing fill object spawn transforms.
- Resolving initial fill object collisions using NLP projection.
- Simulating fill objects dropping into containers using Drake physics.
"""

import logging

from dataclasses import dataclass

import numpy as np
import trimesh

from pydrake.all import RigidTransform
from scipy.spatial import ConvexHull, QhullError

console_logger = logging.getLogger(__name__)


@dataclass
class ContainerInteriorBounds:
    """Interior bounds of a container for fill object spawning."""

    hull_vertices_2d: np.ndarray
    """2D convex hull vertices of container opening in XY plane."""
    centroid_2d: np.ndarray
    """Centroid of the hull in XY."""
    top_z: float
    """Z coordinate of container top (rim)."""
    bottom_z: float
    """Z coordinate of container bottom (interior floor)."""


@dataclass
class FillSimulationResult:
    """Result of physics simulation for fill container operation."""

    inside_indices: list[int]
    """Indices of new fill objects that stayed inside the container."""
    outside_indices: list[int]
    """Indices of new fill objects that fell outside (to catch floor)."""
    final_transforms: list[RigidTransform]
    """Final transforms for new fill objects after simulation."""
    settled_final_transforms: list[RigidTransform] | None = None
    """Updated transforms for settled objects (may have shifted)."""
    settled_fell_out_indices: list[int] | None = None
    """Indices (into settled list) of settled objects pushed out of container."""
    error_message: str | None = None
    """Error message if simulation failed."""


def compute_container_interior_bounds(
    collision_meshes: list[trimesh.Trimesh],
    top_rim_height_fraction: float = 0.15,
    interior_scale: float = 0.95,
) -> ContainerInteriorBounds:
    """Compute interior bounds of a container using top rim heuristic.

    The algorithm:
    1. Find container's max Z (top of container).
    2. Select vertices within top N% of container height.
    3. Project those vertices to XY plane.
    4. Compute convex hull of projected points.
    5. Scale hull by interior_scale to stay away from walls.

    This gives the opening of the container, avoiding handles/decorations.

    Args:
        collision_meshes: Container collision mesh pieces.
        top_rim_height_fraction: Fraction of height for rim detection.
        interior_scale: Scale factor to shrink interior bounds.

    Returns:
        ContainerInteriorBounds with hull vertices and Z bounds.

    Raises:
        ValueError: If collision meshes are empty or hull computation fails.
    """
    if not collision_meshes:
        raise ValueError("No collision meshes provided")

    # Combine all vertices.
    all_vertices = np.vstack([m.vertices for m in collision_meshes])

    # Find Z extent.
    z_min = float(all_vertices[:, 2].min())
    z_max = float(all_vertices[:, 2].max())
    height = z_max - z_min

    if height <= 0:
        raise ValueError("Container has zero or negative height")

    # Select vertices in top N% of container height.
    z_threshold = z_max - (top_rim_height_fraction * height)
    top_vertices = all_vertices[all_vertices[:, 2] >= z_threshold]

    if len(top_vertices) < 3:
        # Fallback: use all vertices if not enough in top region.
        console_logger.warning(
            f"Only {len(top_vertices)} vertices in top rim region, using all vertices"
        )
        top_vertices = all_vertices

    # Project to XY plane.
    xy_vertices = top_vertices[:, :2]

    # Compute convex hull.
    try:
        hull = ConvexHull(xy_vertices)
    except QhullError as e:
        raise ValueError(f"Failed to compute container interior hull: {e}")

    hull_vertices = xy_vertices[hull.vertices]

    # Compute centroid.
    centroid = hull_vertices.mean(axis=0)

    # Scale hull toward centroid to create interior bounds.
    scaled_hull = centroid + interior_scale * (hull_vertices - centroid)

    return ContainerInteriorBounds(
        hull_vertices_2d=scaled_hull,
        centroid_2d=centroid,
        top_z=z_max,
        bottom_z=z_min,
    )
