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

from dataclasses import dataclass

import numpy as np

from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


@dataclass
class SupportSurfaceExtractionConfig:
    """HSM algorithm parameters from paper Section A.2.

    Input meshes in Y-up (GLTF) are converted to Z-up immediately.
    All surface detection happens in Z-up coordinates (Drake convention).
    """

    normal_cluster_threshold: float = 0.9
    """HSM t_norm: Minimum dot product for face normal similarity in cluster."""

    normal_adjacent_threshold: float = 0.95
    """HSM t_adj: Minimum dot product for adjacent face similarity."""

    horizontal_normal_z_min: float = 0.95
    """HSM t_hzn (adapted to Z-up): Minimum Z component for horizontal normals."""

    vertical_normal_z_max: float = 0.05
    """HSM t_vert (adapted to Z-up): Maximum Z component for vertical normals."""

    min_surface_area_m2: float = 0.003
    """HSM MIN_AREA: 30 cm² minimum area (filters tiny surfaces)."""

    min_area_ratio: float = 0.20
    """Minimum mesh_area/bbox_area ratio (filters mesh artifacts)."""

    min_clearance_m: float = 0.05
    """5 cm minimum clearance above surface (filters internal surfaces)."""

    min_inscribed_radius_m: float = 0.10
    """10 cm minimum inscribed radius (filters thin slivers)."""

    height_tolerance_m: float = 0.05
    """5 cm height tolerance for grouping surfaces at same level."""

    self_intersection_threshold_m: float = 0.001
    """1mm threshold for filtering ray-casting self-hits."""

    max_measured_clearance_m: float = 5.0
    """Maximum clearance to measure via ray-casting (cap for efficiency)."""

    top_surface_clearance_m: float = 0.5
    """HSM h_top: 50 cm default clearance for top surfaces."""

    surface_offset_m: float = 0.01
    """Offset above mesh surface for gravity settling."""

    use_max_z_for_surface_height: bool = True
    """Use maximum Z in cluster instead of mean for surface height."""

    max_z_percentile: float = 98.0
    """Percentile for maximum Z (98 filters top 2% outliers)."""

    clearance_percentile: float = 10.0
    """Percentile for clearance calculation. Edge rays often hit nearby vertical
    walls (shelf dividers) at very short distances, while center rays measure the
    actual usable clearance. The 10th percentile filters these edge outliers while
    remaining conservative (not using median/50th)."""

    recompute_hssd_surfaces: bool = False
    """Recompute HSSD surfaces using HSM instead of loading from JSON."""

    use_catalog_aabb_fast_path: bool = True
    """Use bounded canonical AABB planes for catalog assets instead of dense HSM."""

    aabb_inset_ratio: float = 0.08
    """Inset each side of an AABB support plane to avoid object edges."""

    bed_surface_height_ratio: float = 0.48
    """Approximate mattress height within a canonical bed AABB."""

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SupportSurfaceExtractionConfig":
        """Create config from Hydra/OmegaConf nested structure.

        Args:
            cfg: Support surface extraction config subtree.

        Returns:
            SupportSurfaceExtractionConfig instance.
        """
        # ``fast_path`` was added after the original configuration contract.
        # Resumed runs and downstream callers may still supply the older shape,
        # so use the dataclass defaults instead of rejecting a valid config.
        fast_path = cfg.get("fast_path") or {}
        return cls(
            # Face clustering parameters.
            normal_cluster_threshold=cfg.face_clustering.normal_cluster_threshold,
            normal_adjacent_threshold=cfg.face_clustering.normal_adjacent_threshold,
            horizontal_normal_z_min=cfg.face_clustering.horizontal_normal_z_min,
            vertical_normal_z_max=cfg.face_clustering.vertical_normal_z_max,
            # Filtering parameters.
            min_surface_area_m2=cfg.filtering.min_surface_area_m2,
            min_area_ratio=cfg.filtering.min_area_ratio,
            min_inscribed_radius_m=cfg.filtering.min_inscribed_radius_m,
            # Clearance parameters.
            min_clearance_m=cfg.clearance.min_clearance_m,
            max_measured_clearance_m=cfg.clearance.max_measured_clearance_m,
            top_surface_clearance_m=cfg.clearance.top_surface_clearance_m,
            self_intersection_threshold_m=cfg.clearance.self_intersection_threshold_m,
            clearance_percentile=cfg.clearance.clearance_percentile,
            # Height parameters.
            surface_offset_m=cfg.height.surface_offset_m,
            use_max_z_for_surface_height=cfg.height.use_max_z_for_surface_height,
            max_z_percentile=cfg.height.max_z_percentile,
            height_tolerance_m=cfg.height.height_tolerance_m,
            # HSSD surface handling.
            recompute_hssd_surfaces=cfg.hssd.recompute_surfaces,
            # Deterministic catalog fast path.
            use_catalog_aabb_fast_path=fast_path.get("enabled", True),
            aabb_inset_ratio=fast_path.get("aabb_inset_ratio", 0.08),
            bed_surface_height_ratio=fast_path.get("bed_surface_height_ratio", 0.48),
        )


@dataclass
class FaceCluster:
    """Group of mesh faces with similar normals."""

    face_indices: np.ndarray
    """Mesh face indices in this cluster."""

    mean_normal: np.ndarray
    """Average normal vector (3,)."""

    total_area: float
    """Sum of face areas in cluster."""


@dataclass
class ExtractedPlane:
    """Fitted plane from face cluster."""

    normal: np.ndarray
    """Unit normal vector (3,)."""

    centroid: np.ndarray
    """Plane centroid position (3,)."""

    face_indices: np.ndarray
    """Source mesh face indices."""

    area: float
    """Total surface area."""

    is_horizontal: bool
    """Surface classification result."""

    is_upward_facing: bool
    """Whether surface was originally upward-facing (before normal flip)."""
