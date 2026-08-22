"""Support surface extraction using HSM face clustering algorithm.

This module implements the support surface identification algorithm from the HSM
paper (https://arxiv.org/abs/2503.16848v2).

The algorithm clusters mesh faces by normal similarity, fits planes to clusters,
classifies surfaces as horizontal/vertical, and extracts horizontal support
surfaces for manipuland placement.

We slightly modified the algorithm to make it more robust for our lower-quality
generated rather than artist designed furniture meshes.
"""

from __future__ import annotations

import logging
import time

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from scenesmith.agent_utils.geometry.support_surfaces.models import (
    ExtractedPlane,
    SupportSurfaceExtractionConfig,
)
from scenesmith.agent_utils.geometry.support_surfaces.plane_extraction import (
    _cluster_faces_by_normal,
    _compute_convex_hull_min_width,
    _compute_surface_bounds,
    _create_flattened_surface_mesh,
    _create_surface_transform,
    _fit_plane_to_cluster,
    _load_and_prepare_mesh,
    _split_clusters_by_height,
)
from scenesmith.utils.geometry.geometry_utils import safe_convex_hull_2d

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.room_parts.room_models import SupportSurface

console_logger = logging.getLogger(__name__)


def _create_support_surface_from_plane(
    plane: ExtractedPlane,
    mesh: trimesh.Trimesh,
    surface_index: int,
    config: SupportSurfaceExtractionConfig,
) -> "SupportSurface" | None:
    """Create SupportSurface from plane with bounds, clearance, and filtering.

    Computes surface transform, bounds, and clearance via ray-casting.
    Filters out surfaces that are too small, have insufficient clearance,
    or are thin slivers.

    Args:
        plane: Extracted plane to convert to support surface.
        mesh: Source mesh containing the surface geometry.
        surface_index: Index for unique surface ID.
        config: Configuration with filtering thresholds.

    Returns:
        SupportSurface if plane passes all filters, None otherwise.
    """
    from scenesmith.agent_utils.scene.room_parts.room_models import (
        SupportSurface,
        UniqueID,
    )

    # Apply surface offset for gravity settling.
    # This lifts the surface origin above the mesh by config.surface_offset_m.
    # Objects placed at Z=0 in surface-local frame will be at this offset height.
    offset_centroid = plane.centroid.copy()
    offset_centroid[2] += config.surface_offset_m

    # Validate plane geometry before creating transform.
    # Check centroid for NaN/Inf.
    if not np.all(np.isfinite(offset_centroid)):
        console_logger.debug(
            f"  ✗ Rejected: centroid contains NaN/Inf: {offset_centroid}"
        )
        return None

    # Check normal for NaN/Inf.
    if not np.all(np.isfinite(plane.normal)):
        console_logger.debug(f"  ✗ Rejected: normal contains NaN/Inf: {plane.normal}")
        return None

    # Check normal magnitude to prevent division by zero.
    normal_magnitude = np.linalg.norm(plane.normal)
    if normal_magnitude < 1e-9:
        console_logger.debug(
            f"  ✗ Rejected: normal magnitude too small: {normal_magnitude:.2e}"
        )
        return None

    # Create surface transform with offset centroid.
    transform = _create_surface_transform(centroid=offset_centroid, normal=plane.normal)

    # Create flattened mesh for this surface (needed for ray-casting).
    flattened_mesh = _create_flattened_surface_mesh(
        mesh=mesh,
        face_indices=plane.face_indices,
    )

    # Compute surface bounds (uses ray-casting with flattened mesh).
    bounds_min, bounds_max, clearance = _compute_surface_bounds(
        mesh=mesh,
        plane=plane,
        transform=transform,
        simplified_mesh=flattened_mesh,
        config=config,
    )

    # Adjust clearance to account for surface offset.
    # The clearance is computed from the original plane, but the surface origin
    # has been moved up by surface_offset_m. We need to reduce the clearance
    # by this amount to reflect the actual available space above the offset surface.
    clearance_adjusted = clearance - config.surface_offset_m

    # Recompute bounds_max with adjusted clearance.
    # bounds_max[2] was computed as z_min + clearance, but should be z_min +
    # clearance_adjusted.
    bounds_max = bounds_max.copy()
    bounds_max[2] = config.surface_offset_m + clearance_adjusted

    # Skip surfaces with insufficient clearance (internal surfaces).
    if clearance_adjusted < config.min_clearance_m:
        console_logger.debug(
            f"  ✗ Rejected: insufficient clearance {clearance_adjusted:.3f}m "
            f"(< {config.min_clearance_m}m threshold)"
        )
        return None

    # Skip thin slivers by checking inscribed radius and aspect ratio.
    # Compute 2D convex hull of surface vertices in XY plane.
    xy_vertices = flattened_mesh.vertices[:, :2]

    # Handle degenerate geometry gracefully using safe wrapper.
    hull, processed_vertices = safe_convex_hull_2d(xy_vertices)
    if hull is None:
        console_logger.debug(
            f"  ✗ Rejected: degenerate geometry "
            f"(ConvexHull failed - likely collinear/duplicate vertices)"
        )
        return None
    hull_vertices = processed_vertices[hull.vertices]

    # Compute minimum width of convex hull (handles diagonal surfaces correctly).
    # The inscribed radius approximation is half the minimum width.
    hull_min_width = _compute_convex_hull_min_width(hull_vertices)
    min_inscribed_radius = hull_min_width / 2.0

    if min_inscribed_radius < config.min_inscribed_radius_m:
        console_logger.debug(
            f"  ✗ Rejected: inscribed radius {min_inscribed_radius:.3f}m "
            f"(< {config.min_inscribed_radius_m}m threshold)"
        )
        return None

    # Transform mesh vertices to surface-local coordinates.
    # The mesh vertices are currently in mesh-local frame (furniture geometry).
    # We need them in surface-local frame for convex hull validation in contains_point_2d.
    # Use the inverse of the surface transform (which maps surface → mesh).
    transform_matrix_inv = transform.inverse().GetAsMatrix4()
    vertices_local = []
    for v in flattened_mesh.vertices:
        v_hom = np.append(v, 1.0)
        v_local_hom = transform_matrix_inv @ v_hom
        vertices_local.append(v_local_hom[:3])

    # Create new mesh with surface-local vertices.
    mesh_local = trimesh.Trimesh(
        vertices=np.array(vertices_local),
        faces=flattened_mesh.faces,
    )

    # Create surface.
    # Note: surface_id is temporary - replaced by scene.generate_surface_id() when added.
    surface = SupportSurface(
        surface_id=UniqueID(f"surface_{surface_index}"),
        bounding_box_min=bounds_min,
        bounding_box_max=bounds_max,
        transform=transform,
        mesh=mesh_local,
    )

    return surface


def extract_support_surfaces_from_mesh(
    mesh_path: Path,
    config: SupportSurfaceExtractionConfig | None = None,
) -> list["SupportSurface"]:
    """Extract all horizontal support surfaces from furniture mesh.

    Implements HSM face clustering algorithm (https://arxiv.org/abs/2503.16848v2).

    Args:
        mesh_path: Path to visual mesh (.gltf or .glb).
        config: Algorithm parameters (defaults to HSM values).

    Returns:
        List of SupportSurface objects, sorted by area (largest first).

    Raises:
        FileNotFoundError: If mesh file doesn't exist.
        ValueError: If mesh loading fails or mesh has no faces.

    Algorithm:
        1. Load mesh and convert Y-up (GLTF) to Z-up (Drake)
        2. Cluster faces by normal similarity
        3. Fit plane to each cluster
        4. Classify surfaces as horizontal or vertical
        5. Filter by minimum area threshold
        6. Create SupportSurface with bounds, transform, and clearance
        7. Filter by minimum clearance and inscribed radius
        8. Sort by area descending
    """

    start_time = time.time()
    if config is None:
        config = SupportSurfaceExtractionConfig()

    mesh = _load_and_prepare_mesh(mesh_path=mesh_path)

    # Cluster faces by normal similarity.
    clusters = _cluster_faces_by_normal(mesh=mesh, config=config)

    # Split clusters by height to separate multi-level surfaces.
    clusters = _split_clusters_by_height(clusters=clusters, mesh=mesh, config=config)

    # Fit planes to clusters.
    planes = []
    for cluster in clusters:
        try:
            plane = _fit_plane_to_cluster(mesh=mesh, cluster=cluster, config=config)
            planes.append(plane)
        except ValueError as e:
            # Skip degenerate clusters (e.g., faces with zero normals).
            console_logger.debug(f"Skipping degenerate cluster: {e}")
            continue

    # Separate horizontal and vertical surfaces.
    # Only keep upward-facing horizontal surfaces for gravity-based placement.
    horizontal_planes_all = [plane for plane in planes if plane.is_horizontal]
    horizontal_planes_downward = [
        plane for plane in horizontal_planes_all if not plane.is_upward_facing
    ]
    horizontal_planes = [
        plane for plane in planes if plane.is_horizontal and plane.is_upward_facing
    ]
    vertical_planes = [plane for plane in planes if not plane.is_horizontal]

    # Log filtering details.
    if horizontal_planes_downward:
        normals = [
            f"[{p.normal[0]:.2f}, {p.normal[1]:.2f}, {p.normal[2]:.2f}]"
            for p in horizontal_planes_downward
        ]
        console_logger.debug(
            f"Filtered out {len(horizontal_planes_downward)} downward-facing surfaces "
            f"(normals: {normals})"
        )

    console_logger.debug(
        f"Clustering created {len(planes)} planes: {len(horizontal_planes)} horizontal "
        f"(upward), {len(horizontal_planes_downward)} horizontal (downward), "
        f"{len(vertical_planes)} vertical"
    )

    # Pre-filter by plane mesh area to remove tiny surfaces early.
    # This catches thin dividers/slivers that would otherwise pass bbox area filters
    # because their convex hulls span the full width of the furniture.
    large_planes = [
        plane for plane in horizontal_planes if plane.area >= config.min_surface_area_m2
    ]

    console_logger.debug(
        f"After mesh area filter (>= {config.min_surface_area_m2}m²): "
        f"{len(large_planes)}/{len(horizontal_planes)} planes remain"
    )

    # Convert planes to SupportSurface objects.
    # Also filter by bbox area after creation to handle convex hull edge cases.
    surfaces = []
    for i, plane in enumerate(large_planes):
        console_logger.debug(
            f"Processing plane {i}: area={plane.area:.4f}m², "
            f"centroid_z={plane.centroid[2]:.3f}m, is_horizontal={plane.is_horizontal}"
        )
        surface = _create_support_surface_from_plane(
            plane=plane, mesh=mesh, surface_index=i, config=config
        )
        if surface is not None:
            # Filter by bounding box area (surface.area), not plane area.
            if surface.area >= config.min_surface_area_m2:
                surfaces.append(surface)
                console_logger.debug(
                    f"  → Created surface {i} (bbox area={surface.area:.4f}m²)"
                )
            else:
                console_logger.debug(
                    f"  → Rejected: bbox area {surface.area:.4f}m² < "
                    f"{config.min_surface_area_m2}m² threshold"
                )
        else:
            console_logger.debug(f"  → Rejected by _create_support_surface_from_plane")

    # Sort by bounding box area (largest first).
    surfaces.sort(key=lambda s: s.area, reverse=True)

    console_logger.info(
        f"Extracted {len(surfaces)} support surfaces for {mesh_path.name} in "
        f"{time.time() - start_time:.2f} seconds"
    )

    return surfaces
