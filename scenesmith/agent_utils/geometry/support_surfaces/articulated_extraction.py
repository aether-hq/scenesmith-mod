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

from scenesmith.agent_utils.geometry.support_surfaces.mesh_extraction import (
    extract_support_surfaces_from_mesh,
)
from scenesmith.agent_utils.geometry.support_surfaces.models import (
    SupportSurfaceExtractionConfig,
)
from scenesmith.agent_utils.geometry.support_surfaces.plane_extraction import (
    _load_and_prepare_mesh,
)

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.room_parts.room_models import SupportSurface

console_logger = logging.getLogger(__name__)


def _parse_sdf_mesh_to_link(sdf_path: Path) -> dict[str, str]:
    """Parse SDF to build mesh filename -> link name mapping.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        Dict mapping mesh filename (e.g., 'P_7b614f8bbcce8e3f.gltf') to
        link name (e.g., 'E_drawer_1').
    """
    import xml.etree.ElementTree as ET

    mesh_to_link: dict[str, str] = {}
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    for link in root.iter("link"):
        link_name = link.get("name")
        if not link_name:
            continue

        visual_count = 0
        for visual in link.iter("visual"):
            for uri in visual.iter("uri"):
                if uri.text and uri.text.endswith(".gltf"):
                    visual_count += 1
                    # Extract filename from URI (may include subdir path).
                    mesh_filename = Path(uri.text).name
                    mesh_to_link[mesh_filename] = link_name

        if visual_count > 1:
            console_logger.warning(
                f"Link '{link_name}' has {visual_count} visual meshes; "
                f"surfaces from all meshes will be assigned to this link"
            )

    return mesh_to_link


def extract_support_surfaces_articulated(
    sdf_dir: Path,
    config: SupportSurfaceExtractionConfig | None = None,
    sdf_path: Path | None = None,
) -> list["SupportSurface"]:
    """Extract support surfaces from articulated object with per-link association.

    For articulated objects (e.g., furniture with drawers/doors), this function:
    1. Extracts surfaces from each link mesh (for correct link association)
    2. Re-computes clearance against combined mesh (for accurate filtering)
    3. Filters surfaces that don't meet clearance threshold

    Args:
        sdf_dir: Directory containing articulated object files.
        config: Surface extraction configuration. If None, uses defaults.
        sdf_path: Path to the SDF file for mesh-to-link mapping. Required for
            accurate link name resolution with ArtVIP assets.

    Returns:
        List of SupportSurface objects with link_name populated.

    Raises:
        FileNotFoundError: If no link meshes found.
    """
    from scenesmith.agent_utils.scene.room_parts.room_models import SupportSurface

    config = config or SupportSurfaceExtractionConfig()
    start_time = time.time()

    # Build mesh-to-link mapping from SDF for accurate link name resolution.
    mesh_to_link: dict[str, str] = {}
    if sdf_path and sdf_path.exists():
        mesh_to_link = _parse_sdf_mesh_to_link(sdf_path)
        console_logger.debug(
            f"Built mesh-to-link mapping with {len(mesh_to_link)} entries"
        )

    # Find per-link mesh files - check multiple locations/patterns.
    link_gltfs: list[Path] = []

    # Pattern 1: *_combined.gltf at top level (PartNet-Mobility converted).
    link_gltfs.extend(
        f for f in sdf_dir.glob("*_combined.gltf") if f.name != "combined_scene.gltf"
    )

    # Pattern 2: visual/ subdirectory (PartNet-Mobility raw).
    visual_dir = sdf_dir / "visual"
    if visual_dir.exists():
        link_gltfs.extend(visual_dir.glob("*_visual.gltf"))

    # Pattern 3: *_meshes/ subdirectory (ArtVIP).
    for meshes_subdir in sdf_dir.glob("*_meshes"):
        link_gltfs.extend(
            f for f in meshes_subdir.glob("*.gltf") if f.name != "combined_scene.gltf"
        )

    link_gltfs = sorted(set(link_gltfs))  # Dedupe and sort.

    if not link_gltfs:
        # Fallback to combined mesh extraction.
        combined_path = sdf_dir / "combined_scene.gltf"
        # Also check *_meshes subdirectory for combined mesh (ArtVIP).
        if not combined_path.exists():
            for meshes_subdir in sdf_dir.glob("*_meshes"):
                alt_combined = meshes_subdir / "combined_scene.gltf"
                if alt_combined.exists():
                    combined_path = alt_combined
                    break
        if combined_path.exists():
            console_logger.warning(f"No per-link meshes in {sdf_dir}, using combined")
            return extract_support_surfaces_from_mesh(combined_path, config)
        raise FileNotFoundError(f"No link meshes found in {sdf_dir}")

    # Load combined mesh for clearance re-computation.
    combined_path = sdf_dir / "combined_scene.gltf"
    # Also check *_meshes subdirectory (ArtVIP).
    if not combined_path.exists():
        for meshes_subdir in sdf_dir.glob("*_meshes"):
            alt_combined = meshes_subdir / "combined_scene.gltf"
            if alt_combined.exists():
                combined_path = alt_combined
                break
    combined_mesh = None
    if combined_path.exists():
        combined_mesh = _load_and_prepare_mesh(combined_path)
        console_logger.debug("Loaded combined mesh for clearance computation")

    console_logger.info(f"Extracting surfaces from {len(link_gltfs)} link meshes")

    all_surfaces: list[SupportSurface] = []

    for link_gltf in link_gltfs:
        # Derive link_name from SDF mapping, fallback to filename.
        mesh_filename = link_gltf.name
        if mesh_filename in mesh_to_link:
            link_name = mesh_to_link[mesh_filename]
        else:
            # Fallback: strip _combined/_visual suffix.
            stem = link_gltf.stem
            link_name = stem.replace("_combined", "").replace("_visual", "")

        console_logger.debug(f"Processing mesh '{mesh_filename}' -> link '{link_name}'")

        # Extract surfaces from this link mesh using the standard algorithm.
        try:
            link_surfaces = extract_support_surfaces_from_mesh(
                mesh_path=link_gltf, config=config
            )
        except (FileNotFoundError, ValueError) as e:
            console_logger.warning(f"Failed to extract from {link_gltf.name}: {e}")
            continue

        # Tag each surface with its source link.
        for surface in link_surfaces:
            surface.link_name = link_name

        console_logger.info(
            f"Link '{link_name}': {len(link_surfaces)} surfaces extracted"
        )
        all_surfaces.extend(link_surfaces)

    # Log per-link surface counts.
    link_counts = {}
    for surface in all_surfaces:
        link_counts[surface.link_name] = link_counts.get(surface.link_name, 0) + 1
    console_logger.info(
        f"Per-link extraction: {len(all_surfaces)} surfaces from "
        f"{len(link_counts)} links: {link_counts}"
    )

    # Re-compute clearance against combined mesh and filter.
    if combined_mesh is not None:
        filtered_surfaces = []
        for surface in all_surfaces:
            # Create flattened mesh at surface height for ray-casting.
            surface_z = surface.transform.translation()[2]
            # Create a simple grid of points covering the surface bbox.
            bounds_min = surface.bounding_box_min
            bounds_max = surface.bounding_box_max
            # Transform bbox corners to world frame.
            transform_matrix = surface.transform.GetAsMatrix4()
            local_corners = np.array(
                [
                    [bounds_min[0], bounds_min[1], 0],
                    [bounds_max[0], bounds_min[1], 0],
                    [bounds_max[0], bounds_max[1], 0],
                    [bounds_min[0], bounds_max[1], 0],
                ]
            )
            corners_hom = np.column_stack([local_corners, np.ones(4)])
            world_corners = (transform_matrix @ corners_hom.T).T[:, :3]

            # Sample points on surface for ray-casting.
            n_samples = 25  # 5x5 grid.
            u = np.linspace(0, 1, 5)
            v = np.linspace(0, 1, 5)
            ray_origins = []
            for ui in u:
                for vi in v:
                    pt = (
                        (1 - ui) * (1 - vi) * world_corners[0]
                        + ui * (1 - vi) * world_corners[1]
                        + ui * vi * world_corners[2]
                        + (1 - ui) * vi * world_corners[3]
                    )
                    ray_origins.append(pt)
            ray_origins = np.array(ray_origins)
            ray_origins[:, 2] = surface_z + config.self_intersection_threshold_m

            # Cast rays upward.
            ray_directions = np.tile([0, 0, 1], (n_samples, 1))
            locations, index_ray, _ = combined_mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False,
            )

            if len(locations) > 0:
                distances = (
                    np.linalg.norm(locations - ray_origins[index_ray], axis=1)
                    + config.self_intersection_threshold_m
                )
                clearance = float(np.percentile(distances, config.clearance_percentile))
            else:
                clearance = config.top_surface_clearance_m

            # Apply clearance filter.
            if clearance >= config.min_clearance_m:
                filtered_surfaces.append(surface)
                console_logger.debug(
                    f"Surface {surface.surface_id} (link={surface.link_name}): "
                    f"clearance={clearance:.3f}m >= {config.min_clearance_m}m ✓"
                )
            else:
                console_logger.debug(
                    f"Surface {surface.surface_id} (link={surface.link_name}): "
                    f"clearance={clearance:.3f}m < {config.min_clearance_m}m ✗ filtered"
                )

        all_surfaces = filtered_surfaces
        console_logger.info(
            f"After clearance re-computation: {len(all_surfaces)} surfaces remain"
        )

    # Sort by bounding box area (largest first).
    all_surfaces.sort(key=lambda s: s.area, reverse=True)

    console_logger.info(
        f"Extracted {len(all_surfaces)} surfaces from articulated object in "
        f"{time.time() - start_time:.2f}s"
    )

    return all_surfaces


def load_link_meshes(sdf_dir: Path) -> dict[str, trimesh.Trimesh]:
    """Load per-link meshes from articulated object directory.

    Articulated objects (from PartNet-Mobility) have per-link mesh files named
    `{link_name}_combined.gltf`. This function loads them for surface-link
    association.

    Args:
        sdf_dir: Directory containing the articulated object files.

    Returns:
        Mapping from link name to loaded mesh. Empty dict if no link meshes found.
    """
    link_meshes: dict[str, trimesh.Trimesh] = {}

    for gltf_file in sdf_dir.glob("*_combined.gltf"):
        # Skip the merged mesh used for extraction.
        if gltf_file.name == "combined_scene.gltf":
            continue

        link_name = gltf_file.stem.replace("_combined", "")
        try:
            mesh = trimesh.load(gltf_file)

            # Handle Scene objects (multi-mesh gltf files).
            if isinstance(mesh, trimesh.Scene):
                # Concatenate all geometries into single mesh.
                meshes = [
                    g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    console_logger.warning(
                        f"No trimesh geometries in {gltf_file.name}, skipping"
                    )
                    continue

            # Convert Y-up (GLTF) to Z-up (Drake) to match combined mesh.
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            )

            link_meshes[link_name] = mesh
            console_logger.debug(
                f"Loaded link mesh '{link_name}' with {len(mesh.vertices)} vertices"
            )
        except Exception as e:
            console_logger.warning(f"Failed to load link mesh {gltf_file.name}: {e}")

    if link_meshes:
        console_logger.debug(f"Loaded {len(link_meshes)} link meshes from {sdf_dir}")

    return link_meshes
