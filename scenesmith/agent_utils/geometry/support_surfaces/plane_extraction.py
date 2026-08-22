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

from collections import deque
from pathlib import Path

import numpy as np
import trimesh

from pydrake.all import RigidTransform, RotationMatrix

from scenesmith.agent_utils.geometry.support_surfaces.models import (
    ExtractedPlane,
    FaceCluster,
    SupportSurfaceExtractionConfig,
)

console_logger = logging.getLogger(__name__)


def _cluster_faces_by_normal(
    mesh: trimesh.Trimesh, config: SupportSurfaceExtractionConfig
) -> list[FaceCluster]:
    """Cluster faces by normal similarity using HSM algorithm.

    Algorithm from HSM paper Section A.2:
    1. Sort faces by area (largest first)
    2. While unclustered faces remain:
       a. Select largest unclustered face as seed
       b. Grow cluster via breadth-first search:
          - Adjacent face: dot(normal, seed_normal) >= t_adj
          - Cluster membership: dot(normal, cluster_normal) >= t_norm
       c. Use seed normal as cluster normal (HSM Algorithm 1)
    3. Return clusters

    Args:
        mesh: Input triangle mesh.
        config: Algorithm configuration parameters.

    Returns:
        List of face clusters, ordered by total area (largest first).
    """
    num_faces = len(mesh.faces)
    unclustered = set(range(num_faces))
    clusters = []

    # Precompute face properties.
    face_normals = mesh.face_normals  # (num_faces, 3).
    face_areas = mesh.area_faces  # (num_faces,).

    # Build adjacency graph.
    adjacency = {}
    for face_a, face_b in mesh.face_adjacency:
        adjacency.setdefault(face_a, []).append(face_b)
        adjacency.setdefault(face_b, []).append(face_a)

    # Sort faces by area for processing (largest first).
    sorted_faces = np.argsort(-face_areas)
    # Convert to list for efficient iteration while removing elements.
    sorted_unclustered = [face for face in sorted_faces]

    while unclustered:
        # Select largest unclustered face as seed (O(1) amortized).
        seed_face = None
        while sorted_unclustered:
            candidate = sorted_unclustered.pop(0)
            if candidate in unclustered:
                seed_face = candidate
                break

        if seed_face is None:
            break

        # Initialize cluster with seed.
        cluster_faces = {seed_face}
        unclustered.remove(seed_face)

        seed_normal = face_normals[seed_face]

        # Grow cluster via breadth-first search.
        queue = deque([seed_face])

        while queue:
            current_face = queue.popleft()

            # Check adjacent faces.
            for neighbor_face in adjacency.get(current_face, []):
                if neighbor_face not in unclustered:
                    continue

                neighbor_normal = face_normals[neighbor_face]

                # Check adjacent threshold (similarity to current face).
                dot_with_current = np.dot(neighbor_normal, face_normals[current_face])
                if dot_with_current < config.normal_adjacent_threshold:
                    continue

                # Check cluster threshold (similarity to seed - HSM Algorithm 1 line 17).
                dot_with_seed = np.dot(neighbor_normal, seed_normal)
                if dot_with_seed < config.normal_cluster_threshold:
                    continue

                # Add to cluster.
                cluster_faces.add(neighbor_face)
                unclustered.remove(neighbor_face)
                queue.append(neighbor_face)

        # Compute total area from clustered faces.
        cluster_area = np.sum(face_areas[list(cluster_faces)])

        # Create cluster.
        cluster = FaceCluster(
            face_indices=np.array(list(cluster_faces)),
            mean_normal=seed_normal,  # Use seed normal (HSM Algorithm 1).
            total_area=cluster_area,
        )
        clusters.append(cluster)

    # Sort clusters by area (largest first).
    clusters.sort(key=lambda c: c.total_area, reverse=True)

    console_logger.debug(f"Clustered {num_faces} faces into {len(clusters)} clusters")

    return clusters


def _split_clusters_by_height(
    clusters: list[FaceCluster],
    mesh: trimesh.Trimesh,
    config: SupportSurfaceExtractionConfig,
) -> list[FaceCluster]:
    """Split face clusters that span multiple height levels.

    Multi-level furniture (desks with shelves, bookcases) can have topologically
    connected surfaces at different heights that get merged into a single cluster
    by normal-based clustering. This function splits such clusters by Z-height.

    Args:
        clusters: Face clusters from normal-based clustering.
        mesh: Source triangle mesh.
        config: Algorithm configuration with height_tolerance_m.

    Returns:
        Split clusters, one per height level. Clusters within height_tolerance_m
        are kept together.
    """
    split_clusters = []
    num_split = 0

    for cluster in clusters:
        # Get Z-positions of face centroids in this cluster.
        face_centroids_z = []
        for face_idx in cluster.face_indices:
            face_verts = mesh.vertices[mesh.faces[face_idx]]
            centroid_z = face_verts[:, 2].mean()  # Z component in Z-up coords.
            face_centroids_z.append(centroid_z)

        face_centroids_z = np.array(face_centroids_z)

        # Check if cluster spans multiple height levels.
        z_range = face_centroids_z.max() - face_centroids_z.min()

        # If cluster has small vertical extent, keep as-is.
        if z_range <= config.height_tolerance_m:
            split_clusters.append(cluster)
            continue

        # Split cluster by Z-layers using binning approach.
        # Group faces whose centroids are within height_tolerance_m.
        layers = {}  # {representative_z: [(face_idx, centroid_z), ...]}.

        for i, face_idx in enumerate(cluster.face_indices):
            z = face_centroids_z[i]

            # Find existing layer within height_tolerance_m.
            assigned = False
            for layer_z in layers.keys():
                if abs(z - layer_z) <= config.height_tolerance_m:
                    layers[layer_z].append((face_idx, z))
                    assigned = True
                    break

            # Create new layer if no match found.
            if not assigned:
                layers[z] = [(face_idx, z)]

        # Create sub-clusters for each height layer.
        if len(layers) > 1:
            num_split += 1

        for layer_faces in layers.values():
            face_indices = np.array([f[0] for f in layer_faces])

            # Recompute area and mean normal for sub-cluster.
            layer_area = mesh.area_faces[face_indices].sum()
            layer_normals = mesh.face_normals[face_indices]
            # Area-weighted mean normal.
            face_areas = mesh.area_faces[face_indices]
            mean_normal = np.average(layer_normals, weights=face_areas, axis=0)

            sub_cluster = FaceCluster(
                face_indices=face_indices,
                mean_normal=mean_normal,
                total_area=layer_area,
            )
            split_clusters.append(sub_cluster)

    if num_split > 0:
        console_logger.debug(
            f"Split {num_split} clusters by height into {len(split_clusters)} total "
            f"(was {len(clusters)})"
        )

    return split_clusters


def _fit_plane_to_cluster(
    mesh: trimesh.Trimesh, cluster: FaceCluster, config: SupportSurfaceExtractionConfig
) -> ExtractedPlane:
    """Fit plane to face cluster using OBB (oriented bounding box).

    Matches HSM paper methodology (Section A.2).

    Algorithm:
    1. Extract face centroids weighted by area
    2. Compute weighted mean centroid
    3. Optionally compute height offset to max Z percentile
    4. Fit OBB to cluster submesh
    5. Normal = OBB axis with largest Z component (upward-facing)
    6. Apply height offset to move centroid to surface top

    Args:
        mesh: Input triangle mesh.
        cluster: Face cluster to fit plane to.
        config: Algorithm configuration.

    Returns:
        Extracted plane with normal, centroid, and planarity score.
    """
    # Extract centroids of faces in cluster.
    face_indices = cluster.face_indices
    face_areas = mesh.area_faces[face_indices]

    # Compute centroids: mean of triangle vertices.
    centroids = np.mean(mesh.vertices[mesh.faces[face_indices]], axis=1)  # (N, 3).

    # Weighted mean centroid (by area).
    # Handle zero-area faces (degenerate triangles).
    total_area = np.sum(face_areas)
    if total_area < 1e-10:
        # All faces are degenerate, use unweighted mean.
        mean_centroid = np.mean(centroids, axis=0)
    else:
        mean_centroid = np.average(centroids, weights=face_areas, axis=0)

    # Compute offset to max Z in cluster if enabled.
    # We'll use this offset to adjust the plane height later.
    height_offset = 0.0
    if config.use_max_z_for_surface_height:
        # Use percentile instead of max to filter outliers.
        max_z = np.percentile(centroids[:, 2], config.max_z_percentile)
        height_offset = max_z - mean_centroid[2]

    # Check for degenerate cluster (zero normal).
    if np.linalg.norm(cluster.mean_normal) < 1e-10:
        raise ValueError(
            f"Cluster has degenerate normal (magnitude < 1e-10). "
            f"Cluster size: {len(face_indices)}"
        )

    # Fit OBB (oriented bounding box) to cluster.
    # For small clusters or when OBB fails, fall back to cluster mean normal.
    if len(face_indices) >= 3:
        try:
            # Create submesh from cluster faces for OBB fitting.
            submesh = mesh.submesh([face_indices], append=True)

            # Get OBB transform.
            obb = submesh.bounding_box_oriented
            obb_transform = obb.primitive.transform

            # Extract rotation matrix (top-left 3x3).
            # The columns are the principal axes of the OBB.
            rotation = obb_transform[:3, :3]

            # Find axis with largest Z component (most vertical).
            # This is the surface normal direction.
            z_components = np.abs(rotation[2, :])
            normal_axis_idx = np.argmax(z_components)
            normal = rotation[:, normal_axis_idx].copy()

            # Validate OBB produced finite values.
            if not np.all(np.isfinite(normal)):
                raise ValueError("OBB produced non-finite normal.")
        except Exception:
            # OBB failed (degenerate geometry), use cluster mean normal.
            normal = cluster.mean_normal.copy()
    else:
        # Too few faces for reliable OBB, use cluster mean normal.
        normal = cluster.mean_normal.copy()

    # Normalize.
    normal /= np.linalg.norm(normal)

    # Check for NaN/inf from degenerate faces (zero normals).
    if not np.all(np.isfinite(normal)):
        raise ValueError(
            f"Cluster has degenerate normal after normalization (NaN/inf detected). "
            f"Cluster size: {len(face_indices)}, original normal norm: "
            f"{np.linalg.norm(cluster.mean_normal):.6f}"
        )

    # Classify as horizontal based on absolute Z component.
    # Uses abs() to detect both upward and downward horizontal surfaces.
    # Downward-facing surfaces will be filtered out later for gravity-based placement.
    is_horizontal = abs(normal[2]) >= config.horizontal_normal_z_min

    # Track original normal direction using cluster mean normal (from mesh faces).
    # Cannot use OBB normal because OBB orientation is arbitrary and may be flipped.
    # Upward-facing surfaces have positive Z component in Drake Z-up coordinates.
    # This filters out bottom surfaces (e.g., underside of shelf planks).
    is_upward_facing = cluster.mean_normal[2] > 0

    # Ensure upward-pointing for downstream processing (transform, bounds).
    # This is only for geometry, not for classification.
    if normal[2] < 0:
        normal = -normal

    # Apply height offset to move plane to max Z if enabled.
    adjusted_centroid = mean_centroid.copy()
    adjusted_centroid[2] += height_offset

    plane = ExtractedPlane(
        normal=normal,
        centroid=adjusted_centroid,
        face_indices=face_indices,
        area=cluster.total_area,
        is_horizontal=is_horizontal,
        is_upward_facing=is_upward_facing,
    )

    return plane


def _create_surface_transform(
    centroid: np.ndarray, normal: np.ndarray
) -> RigidTransform:
    """Create RigidTransform for surface frame in Drake Z-up coordinates.

    Input is in Z-up coordinates, output is in Z-up coordinates.

    Surface frame (Z-up):
    - Origin: centroid
    - Z-axis: normal (upward in Z-up)
    - Y-axis: Z × world_x (perpendicular to normal and world X)
    - X-axis: Y × Z (recomputed for orthogonality, right-hand rule)

    Args:
        centroid: Surface center position in Z-up coordinates (3,).
        normal: Surface normal in Z-up coordinates (3,), points upward (Z+).

    Returns:
        RigidTransform from surface frame to world frame in Drake Z-up coordinates.
    """
    # Z-axis = normal (should be upward in Z-up coordinate system).
    z_axis = normal / np.linalg.norm(normal)

    # Build right-handed coordinate frame with Z pointing up.
    # Choose initial X direction in XY plane, then compute Y = Z × X, then X = Y × Z.
    world_x = np.array([1.0, 0.0, 0.0])

    # Compute Y-axis: Y = Z × world_x (perpendicular to both).
    y_axis = np.cross(z_axis, world_x)

    # Handle degenerate case where world_x is parallel to z_axis.
    if np.linalg.norm(y_axis) < 1e-6:
        # Use world Y instead.
        world_y = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, world_y)

    y_axis /= np.linalg.norm(y_axis)

    # Recompute X-axis: X = Y × Z (ensures right-handed orthogonal system).
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Create rotation matrix in Z-up coordinates [X Y Z] (column vectors).
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # Create RigidTransform in Z-up coordinates.
    transform = RigidTransform(
        R=RotationMatrix(rotation_matrix),
        p=centroid,
    )

    return transform


def _compute_convex_hull_min_width(hull_vertices: np.ndarray) -> float:
    """Compute minimum width of 2D convex hull.

    The minimum width is the smallest perpendicular distance across the hull,
    found by checking the "height" when the hull rests on each edge. This
    correctly handles diagonal surfaces that axis-aligned bounding boxes miss.

    Args:
        hull_vertices: 2D vertices of convex hull, shape (N, 2).

    Returns:
        Minimum width in meters, or inf if degenerate.
    """
    min_width = float("inf")
    n = len(hull_vertices)

    for i in range(n):
        # Edge from vertex i to vertex i+1.
        p1 = hull_vertices[i]
        p2 = hull_vertices[(i + 1) % n]

        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue

        # Unit normal perpendicular to edge.
        normal = np.array([-edge[1], edge[0]]) / edge_len

        # Max distance from edge to any vertex (the "height" on this edge).
        max_dist = 0.0
        for j in range(n):
            dist = abs(np.dot(hull_vertices[j] - p1, normal))
            max_dist = max(max_dist, dist)

        min_width = min(min_width, max_dist)

    return min_width


def _compute_clearance_via_raycasting(
    surface_mesh: trimesh.Trimesh,
    full_mesh: trimesh.Trimesh,
    config: SupportSurfaceExtractionConfig,
    default_clearance: float,
) -> float:
    """Compute clearance by ray-casting upward from surface vertices.

    Casts vertical rays (+Z direction) from each vertex in the surface mesh
    to find intersections with the full mesh geometry. Uses percentile-based
    clearance (default 10th percentile) to filter edge effects where rays hit
    nearby vertical walls (shelf dividers) at very short distances.

    Args:
        surface_mesh: Flattened support surface mesh (vertices at surface height).
        full_mesh: Complete furniture mesh to ray-cast against.
        config: Algorithm configuration parameters.
        default_clearance: Clearance to use when no obstacles found above.

    Returns:
        Clearance distance in meters based on config.clearance_percentile.
    """
    # Get surface vertices as ray origins, with Z offset to avoid self-intersection.
    # The surface mesh is flattened to a single Z height (the surface plane).
    # Without offset, rays immediately hit the surface faces they originate from.
    ray_offset_z = config.self_intersection_threshold_m
    ray_origins = surface_mesh.vertices.copy()  # Shape: (N, 3).
    ray_origins[:, 2] += ray_offset_z

    # Ray directions: straight up along Z-axis.
    ray_directions = np.tile([0, 0, 1], (len(ray_origins), 1))  # Shape: (N, 3).

    # Cast rays against full mesh.
    # Returns locations of intersections and indices of hit triangles.
    locations, index_ray, index_tri = full_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,  # Only need first hit above each vertex.
    )

    console_logger.debug(
        f"Ray-casting: {len(ray_origins)} rays, {len(locations)} hits, "
        f"ray Z: [{np.min(ray_origins[:, 2]):.3f}, {np.max(ray_origins[:, 2]):.3f}], "
        f"mesh Z: [{np.min(full_mesh.vertices[:, 2]):.3f}, "
        f"{np.max(full_mesh.vertices[:, 2]):.3f}]"
        + (
            f", hit Z: [{np.min(locations[:, 2]):.3f}, {np.max(locations[:, 2]):.3f}]"
            if len(locations) > 0
            else ""
        )
    )

    if len(locations) == 0:
        # No intersections - use default clearance for top surfaces.
        console_logger.debug(f"No hits, using default clearance={default_clearance}m")
        return default_clearance

    # Compute distances from ray origins to intersection points.
    # index_ray tells us which ray (vertex) each intersection belongs to.
    # Add ray_offset_z to get true clearance from original surface (not offset origin).
    distances = (
        np.linalg.norm(locations - ray_origins[index_ray], axis=1) + ray_offset_z
    )

    # Filter out tiny distances (self-intersections or numerical noise).
    # With the offset, self-intersections should no longer occur, but filter anyway.
    valid_distances = distances[distances > config.self_intersection_threshold_m]

    # Include non-hit rays with default clearance for correct percentile semantics.
    # The percentile should represent "X% of rays have clearance this low or lower",
    # not "Xth percentile of only the rays that hit something". Non-hit rays have
    # infinite clearance; we use default_clearance as a practical upper bound.
    num_non_hits = len(ray_origins) - len(valid_distances)
    all_distances = np.concatenate(
        [
            valid_distances,
            np.full(num_non_hits, default_clearance),
        ]
    )

    # Use percentile over all rays (hits + non-hits) to compute clearance.
    percentile_clearance = float(
        np.percentile(all_distances, config.clearance_percentile)
    )
    capped_clearance = min(percentile_clearance, config.max_measured_clearance_m)

    console_logger.debug(
        f"Clearance: {len(valid_distances)} hits + {num_non_hits} non-hits = "
        f"{len(all_distances)} rays, "
        f"p{config.clearance_percentile:.0f}={percentile_clearance:.3f}m, "
        f"capped={capped_clearance:.3f}m"
    )
    return capped_clearance


def _compute_surface_bounds(
    mesh: trimesh.Trimesh,
    plane: ExtractedPlane,
    transform: RigidTransform,
    simplified_mesh: trimesh.Trimesh,
    config: SupportSurfaceExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute 3D AABB for surface in surface-local frame.

    Algorithm:
    1. Extract vertices of faces in cluster
    2. Transform vertices to surface frame
    3. Compute 2D AABB in XY plane (surface coordinates)
    4. Ray-cast upward from surface to find clearance height
    5. Center bounding box around surface frame origin in XY

    Clearance is computed by ray-casting vertically upward from each vertex
    in the simplified surface mesh to find intersections with the full mesh.
    This handles thick meshes, tilted obstacles, and complex geometry.

    Args:
        mesh: Input triangle mesh (source geometry for surface vertices).
        plane: Extracted plane.
        transform: Surface frame transform.
        simplified_mesh: Flattened surface mesh for ray-casting.
        config: Algorithm configuration.

    Returns:
        Tuple of (bounds_min, bounds_max, clearance) in surface-local frame.
        bounds_min and bounds_max are (3,) arrays, clearance is float in meters.
    """
    # Extract vertices of faces in this surface.
    face_vertices = mesh.vertices[mesh.faces[plane.face_indices]]  # (N, 3, 3).
    vertices = face_vertices.reshape(-1, 3)  # Flatten to (N*3, 3).

    # Transform to surface frame.
    transform_matrix = transform.GetAsMatrix4()  # 4x4 homogeneous.
    transform_inv = np.linalg.inv(transform_matrix)

    # Convert to homogeneous coordinates.
    vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])

    # Transform to surface frame.
    vertices_surface = (transform_inv @ vertices_hom.T).T[:, :3]

    # Compute 2D bounds in XY plane.
    xy_min = np.min(vertices_surface[:, :2], axis=0)
    xy_max = np.max(vertices_surface[:, :2], axis=0)

    # Center bounding box around origin in XY.
    xy_half_extents = (xy_max - xy_min) / 2

    xy_min_centered = -xy_half_extents
    xy_max_centered = xy_half_extents

    # Z bounds: surface offset to ray-cast clearance height.
    z_min = config.surface_offset_m

    # Ray-cast to find actual clearance above this surface.
    clearance = _compute_clearance_via_raycasting(
        surface_mesh=simplified_mesh,
        full_mesh=mesh,
        config=config,
        default_clearance=config.top_surface_clearance_m,
    )

    z_max = z_min + clearance

    # Combine into 3D bounds.
    bounds_min = np.array([xy_min_centered[0], xy_min_centered[1], z_min])
    bounds_max = np.array([xy_max_centered[0], xy_max_centered[1], z_max])

    return bounds_min, bounds_max, clearance


def _create_flattened_surface_mesh(
    mesh: trimesh.Trimesh, face_indices: np.ndarray
) -> trimesh.Trimesh:
    """Create flattened support surface mesh in Z-up coordinates.

    Extracts submesh and flattens to target height for visualization.
    Output mesh is ready for direct rendering without additional transforms.

    Args:
        mesh: Source mesh in Z-up coordinates (already transformed).
        face_indices: Indices of faces that comprise the support surface.

    Returns:
        Flattened trimesh.Trimesh in Z-up coordinates at horizontal plane.
    """
    # Extract sub-mesh using face indices.
    submesh = mesh.submesh([face_indices], append=True)

    # Get target Z height as maximum Z of surface vertices.
    # Using centroid Z would cause ray-casting self-intersection because some
    # faces lie above the average. Max Z ensures surface is at the top.
    face_vertices = mesh.vertices[mesh.faces[face_indices]]
    target_z = np.max(face_vertices[:, :, 2])

    # Flatten to horizontal plane: Set all Z coordinates to target height.
    # In Z-up coordinates, horizontal surfaces have constant Z.
    # The mesh stores vertices in global coordinates for visualization.
    vertices = submesh.vertices.copy()
    vertices[:, 2] = target_z
    submesh.vertices = vertices

    return submesh


def _load_and_prepare_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load mesh and prepare it for surface extraction.

    Handles:
    - File validation and loading.
    - Scene concatenation (multiple geometries).
    - Y-up to Z-up coordinate conversion (GLTF → Drake).
    - Vertex merging for proper adjacency detection.

    Args:
        mesh_path: Path to mesh file (GLTF, OBJ, etc.).

    Returns:
        Prepared mesh in Z-up coordinates with merged vertices.

    Raises:
        FileNotFoundError: If mesh_path does not exist.
        ValueError: If mesh cannot be loaded or has no faces.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    console_logger.info(f"Extracting support surfaces from {mesh_path.name}")

    mesh = trimesh.load(str(mesh_path), force="mesh")

    # Handle Scene objects (multiple geometries).
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [
                geom
                for geom in mesh.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
            ]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as Trimesh: {mesh_path}")

    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    console_logger.debug(
        f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
    )

    # Convert from Y-up (GLTF) to Z-up (Drake/Blender) immediately.
    # All subsequent processing happens in Z-up coordinates.
    # Transformation: (X, Y, Z) → (X, -Z, Y).
    # This ensures positive Y (up in GLTF) becomes positive Z (up in Drake).
    transform_y_to_z = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    mesh.apply_transform(transform_y_to_z)

    # Merge duplicate vertices for proper adjacency detection.
    # GLTF format often stores duplicate vertices per triangle.
    # Use digits_vertex=4 for ~0.0001m precision (matching Blender defaults).
    # Done after Z-up transform to ensure all processing happens in Z-up.
    vertices_before = len(mesh.vertices)
    mesh.merge_vertices(digits_vertex=4)
    console_logger.debug(
        f"Merged vertices: {vertices_before} → {len(mesh.vertices)} "
        f"({100 * (1 - len(mesh.vertices) / vertices_before):.1f}% reduction)"
    )

    return mesh
