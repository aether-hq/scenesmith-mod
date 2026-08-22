import logging
import time

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from pydrake.all import RigidTransform

from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

HSSD_CANONICAL_CONVERSION_VERSION = 4

from scenesmith.agent_utils.assets.asset_models import AssetPathConfig


class AssetRegistryMixin:
    """SDF discovery, scene-object construction, bounds, and registry access."""

    def _find_sdf_file(self, sdf_dir: Path) -> Path:
        """Find the generated SDF file in the asset directory.

        Args:
            sdf_dir: Directory containing the generated SDF file.

        Returns:
            Path to the SDF file.

        Raises:
            RuntimeError: If no SDF file or multiple SDF files are found.
        """
        # First try direct search in the directory.
        sdf_files = list(sdf_dir.glob("*.sdf"))

        # If not found, search recursively (create_drake_asset_from_geometry creates
        # nested dirs).
        if not sdf_files:
            sdf_files = list(sdf_dir.glob("**/*.sdf"))

        if not sdf_files:
            raise RuntimeError(f"No SDF file generated in {sdf_dir}")
        if len(sdf_files) > 1:
            raise RuntimeError(f"Multiple SDF files generated in {sdf_dir}")
        return sdf_files[0].absolute()

    def _create_scene_object(
        self,
        config: AssetPathConfig,
        object_type: ObjectType,
        sdf_path: Path,
        final_gltf_path: Path,
        bbox_min: np.ndarray | None = None,
        bbox_max: np.ndarray | None = None,
        additional_metadata: dict | None = None,
        scale_factor: float = 1.0,
    ) -> SceneObject:
        """Convert assets to SceneObject (supports both generated and HSSD).

        Args:
            config: Asset path configuration containing metadata and paths.
            object_type: Type of object.
            sdf_path: Path to the generated SDF file.
            final_gltf_path: Path to the final scaled GLTF mesh file.
            bbox_min: Minimum corner of object-frame bounding box.
            bbox_max: Maximum corner of object-frame bounding box.
            additional_metadata: Optional metadata to merge into the object's
                metadata dict. Useful for HSSD assets to add {"asset_source": "hssd"}.
            scale_factor: Initial uniform scale factor applied during mesh scaling.
                This is needed to correctly scale HSSD pre-computed support surfaces.

        Returns:
            Complete SceneObject ready for scene placement.
        """
        # Base metadata common to all assets.
        metadata = {"generation_timestamp": time.time()}

        # Merge additional metadata (for HSSD: {"asset_source": "hssd"}).
        if additional_metadata:
            metadata.update(additional_metadata)

        scene_obj = SceneObject(
            object_id=self.registry.generate_unique_id(config.short_name),
            object_type=object_type,
            name=config.short_name,
            description=config.description,
            transform=RigidTransform(),  # Will be set during placement.
            geometry_path=final_gltf_path,
            sdf_path=sdf_path,
            image_path=config.image_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            metadata=metadata,
            scale_factor=scale_factor,
        )

        # Register the asset for reuse.
        self.registry.register(scene_obj)

        return scene_obj

    def get_asset_by_id(self, asset_id: UniqueID) -> SceneObject | None:
        """Get a registered asset by ID.

        Args:
            asset_id: Unique identifier of the asset.

        Returns:
            SceneObject if found, None otherwise.
        """
        return self.registry.get(asset_id)

    def list_available_assets(self) -> list[SceneObject]:
        """List all assets available for reuse.

        Returns:
            List of all registered SceneObjects.
        """
        return self.registry.list_all()

    def _extract_bounds_from_visual_mesh(
        self, sdf_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract AABB from the visual GLTF mesh after conversion.

        Args:
            sdf_path: Path to the SDF file.

        Returns:
            Tuple of (bbox_min, bbox_max) arrays.

        Raises:
            FileNotFoundError: If GLTF file is not found.
            ValueError: If mesh cannot be loaded or is invalid.
        """
        # Pattern: {sdf_dir}/{asset_name}/{asset_name}.gltf
        gltf_path = sdf_path.with_suffix(".gltf")

        if not gltf_path.exists():
            raise FileNotFoundError(
                f"Visual GLTF not found at expected path: {gltf_path}"
            )

        # Load mesh using trimesh.
        mesh = trimesh.load(gltf_path, force="mesh")

        # Handle Scene objects (multiple meshes).
        if isinstance(mesh, trimesh.Scene):
            combined_mesh = trimesh.Trimesh()
            for geom in mesh.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
            mesh = combined_mesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load valid mesh from {gltf_path}")

        # Extract bounds.
        bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        bbox_min = bounds[0]
        bbox_max = bounds[1]

        console_logger.debug(
            f"Extracted bounds from {gltf_path}: min={bbox_min}, max={bbox_max}"
        )

        return bbox_min, bbox_max

    def clear_asset_registry(self) -> None:
        """Clear the asset registry."""
        self.registry.clear()
