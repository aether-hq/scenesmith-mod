import logging
import re

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from scenesmith.agent_utils.geometry.mesh_canonicalization import canonicalize_mesh
from scenesmith.agent_utils.geometry.mesh_utils import load_mesh_as_trimesh
from scenesmith.agent_utils.geometry.sdf_generator import generate_drake_sdf
from scenesmith.agent_utils.geometry_generation_server.pipelines.sam_provider import (
    sam_provider_config_from_mapping,
    validate_sam_provider_config,
)
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.objaverse_retrieval_server import ObjaverseRetrievalClient
from scenesmith.agent_utils.objaverse_retrieval_server.dataclasses import (
    ObjaverseRetrievalServerRequest,
)
from scenesmith.agent_utils.physics.mesh_physics_analyzer import (
    MeshPhysicsAnalysis,
    analyze_mesh_orientation_and_material,
)
from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

HSSD_CANONICAL_CONVERSION_VERSION = 4

from scenesmith.agent_utils.assets.asset_models import (
    AssetGenerationRequest,
    AssetGenerationResult,
    FailedAsset,
)


class AssetRetrievalMixin:
    """Filename, collision, validation, HSSD, and Objaverse retrieval."""

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = 50) -> str:
        """Sanitize a name for use as a filename.

        Args:
            name: Name to sanitize.
            max_length: Maximum length for the filename.

        Returns:
            Filesystem-safe filename string.
        """
        # Replace problematic characters with underscores.
        sanitized = re.sub(r"[^\w\-_.]", "_", name)
        # Remove consecutive underscores.
        sanitized = re.sub(r"_+", "_", sanitized)
        # Trim to max length.
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip("_")
        return sanitized

    def _generate_collision_geometry(self, mesh_path: Path) -> list[trimesh.Trimesh]:
        """Generate collision geometry using the configured convex decomposition method.

        Args:
            mesh_path: Path to the mesh file (GLTF/GLB/OBJ).

        Returns:
            List of convex trimesh objects from the decomposition.

        Raises:
            RuntimeError: If collision client is not available.
        """
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        # Build parameter dict based on method.
        if self.collision_method == "coacd":
            return self.collision_client.generate_collision_geometry(
                mesh_path=mesh_path,
                method="coacd",
                threshold=self.collision_coacd_cfg.threshold,
                max_convex_hull=self.collision_coacd_cfg.max_convex_hull,
                preprocess_mode=self.collision_coacd_cfg.preprocess_mode,
                preprocess_resolution=self.collision_coacd_cfg.preprocess_resolution,
                resolution=self.collision_coacd_cfg.resolution,
                mcts_nodes=self.collision_coacd_cfg.mcts_nodes,
                mcts_iterations=self.collision_coacd_cfg.mcts_iterations,
                mcts_max_depth=self.collision_coacd_cfg.mcts_max_depth,
                pca=self.collision_coacd_cfg.pca,
                merge=self.collision_coacd_cfg.merge,
                decimate=self.collision_coacd_cfg.decimate,
                max_ch_vertex=self.collision_coacd_cfg.max_ch_vertex,
                extrude=self.collision_coacd_cfg.extrude,
                extrude_margin=self.collision_coacd_cfg.extrude_margin,
                apx_mode=self.collision_coacd_cfg.apx_mode,
                seed=self.collision_coacd_cfg.seed,
            )
        else:
            # V-HACD method.
            return self.collision_client.generate_collision_geometry(
                mesh_path=mesh_path,
                method="vhacd",
                max_convex_hulls=self.collision_vhacd_cfg.max_convex_hulls,
                vhacd_resolution=self.collision_vhacd_cfg.resolution,
                max_recursion_depth=self.collision_vhacd_cfg.max_recursion_depth,
                max_num_vertices_per_ch=self.collision_vhacd_cfg.max_num_vertices_per_ch,
                min_volume_percent_error=self.collision_vhacd_cfg.min_volume_percent_error,
                shrink_wrap=self.collision_vhacd_cfg.shrink_wrap,
                fill_mode=self.collision_vhacd_cfg.fill_mode,
                min_edge_length=self.collision_vhacd_cfg.min_edge_length,
                find_best_plane=self.collision_vhacd_cfg.find_best_plane,
            )

    def _validate_sam3d_config(self) -> None:
        """Validate SAM3D configuration at startup.

        Raises:
            ValueError: If SAM3D configuration is invalid or missing required fields.
            FileNotFoundError: If checkpoint files do not exist.
        """
        if "sam3d" not in self.cfg.asset_manager:
            raise ValueError(
                "SAM3D backend selected but 'sam3d' configuration is missing. "
                "Add 'sam3d' section to asset_manager config."
            )

        sam3d_cfg = self.cfg.asset_manager.sam3d

        provider_config = sam_provider_config_from_mapping(sam3d_cfg)
        provider = validate_sam_provider_config(provider_config)

        # Validate mode field.
        mode = sam3d_cfg.mode
        if mode not in ["foreground", "object_description"]:
            raise ValueError(
                f"Invalid SAM3D mode: {mode}. "
                "Must be 'foreground' or 'object_description'."
            )

        # Validate threshold.
        threshold = sam3d_cfg.threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Invalid SAM3D threshold: {threshold}. Must be between 0.0 and 1.0."
            )

        console_logger.info(
            f"SAM3D configuration validated successfully (provider={provider}, "
            f"mode={mode}, threshold={threshold})"
        )

    def _retrieve_hssd_assets(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Retrieve assets from HSSD library using server client.

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with retrieved assets.
        """
        if self.hssd_client is None:
            raise RuntimeError("HSSD retrieval client not initialized")
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Retrieving {len(request.object_descriptions)} assets from HSSD server"
        )

        # Create asset path configurations for output directories.
        asset_path_configs = self._create_asset_paths(
            object_descriptions=request.object_descriptions,
            short_names=request.short_names,
        )

        # Ensure output directories exist.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Create batch requests for HSSD server with client-specified output dirs.
        retrieval_requests = [
            HssdRetrievalServerRequest(
                object_description=desc,
                object_type=request.object_type.value,
                desired_dimensions=tuple(dims) if dims else None,
                output_dir=str(config.sdf_dir),
                scene_id=request.scene_id,
            )
            for desc, dims, config in zip(
                request.object_descriptions,
                request.desired_dimensions,
                asset_path_configs,
            )
        ]

        successful_objects: list[SceneObject] = []
        failed_assets: list[FailedAsset] = []

        # Submit batch to server and process streaming responses.
        for index, response in self.hssd_client.retrieve_objects(retrieval_requests):
            desc = request.object_descriptions[index]
            short_name = request.short_names[index]
            config = asset_path_configs[index]

            try:
                console_logger.info(
                    "Processing HSSD response "
                    f"{index+1}/{len(request.object_descriptions)}: '{desc}'"
                )

                # Server returns mesh path (already exported to our output_dir).
                if not response.results:
                    raise ValueError("No results returned from HSSD server")

                result = response.results[0]  # Get top result.
                server_mesh_path = Path(result.mesh_path)
                mesh_id = result.hssd_id

                # Server exported to our specified output_dir, convert GLB to GLTF if
                # needed. Uses BlenderServer for crash isolation.
                if server_mesh_path.suffix.lower() == ".glb":
                    # Server exported GLB, convert to GLTF with Y-up coordinates.
                    gltf_path = server_mesh_path.with_suffix(".gltf")
                    self.blender_server.convert_glb_to_gltf(
                        input_path=server_mesh_path,
                        output_path=gltf_path,
                        export_yup=True,
                    )
                    server_mesh_path.unlink()  # Remove GLB after conversion.
                else:
                    # Already GLTF, use as-is.
                    gltf_path = server_mesh_path

                # Run VLM analysis for material and mass estimation.
                # Use HSSD-specific prompts and only side views to constrain
                # rotation to Z-axis. Orientation (Z-up) is correct from HSSD
                # transformation pipeline.
                # Create debug directory for saving multi-view physics analysis images.
                debug_dir = self.debug_dir / short_name

                console_logger.info(
                    f"Running VLM analysis for HSSD material/mass: {short_name}"
                )
                vlm_physics = analyze_mesh_orientation_and_material(
                    mesh_path=gltf_path,
                    vlm_service=self.vlm_service,
                    cfg=self.cfg,
                    elevation_degrees=self.side_view_elevation_degrees,
                    blender_server=self.blender_server,
                    num_side_views=self.num_side_views_for_physics_analysis,
                    prompt_type="hssd",
                    include_vertical_views=False,
                    debug_output_dir=debug_dir,
                )
                console_logger.info(
                    f"VLM analysis complete: material={vlm_physics.material}, "
                    f"mass={vlm_physics.mass_kg}kg, front={vlm_physics.front_axis}"
                )

                # Use VLM's material, mass, and front axis determination.
                # up_axis is always Z for HSSD (validated by VLM).
                physics_analysis = MeshPhysicsAnalysis(
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    material=vlm_physics.material,
                    mass_kg=vlm_physics.mass_kg,
                    mass_range_kg=vlm_physics.mass_range_kg,
                )

                # Canonicalize mesh orientation to align with scenesmith canonical
                # (Z-up, Y-forward). For HSSD objects already with front=+Y, this is
                # a no-op (fast return). Otherwise, applies Z-rotation to align front.
                console_logger.info(
                    f"Canonicalizing HSSD mesh: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis} → +Y"
                )
                final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
                canonicalize_mesh(
                    gltf_path=gltf_path,
                    output_path=final_gltf_path,
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    blender_server=self.blender_server,
                    object_type=request.object_type,
                )

                # Generate collision geometry via convex decomposition server.
                collision_pieces = self._generate_collision_geometry(final_gltf_path)

                # Load mesh for bounding box calculation.
                mesh = load_mesh_as_trimesh(final_gltf_path, force_merge=True)

                sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
                generate_drake_sdf(
                    visual_mesh_path=final_gltf_path,
                    collision_pieces=collision_pieces,
                    physics_analysis=physics_analysis,
                    output_path=sdf_path,
                    asset_name=config.short_name,
                )

                # Extract bounding box from Y-up GLTF.
                bounds = mesh.bounds  # In Y-up coordinates (GLTF native format).

                # Transform from Y-up (GLTF) to Z-up (Drake) coordinate system.
                # Y-up → Z-up transformation: (x, y, z) → (x, -z, y)
                # Maps: X→X (right), Y→Z (up), Z→-Y (forward with sign flip).
                bbox_min_yup = bounds[0]
                bbox_max_yup = bounds[1]

                # Apply coordinate transformation.
                bbox_min = np.array(
                    [bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]]
                )
                bbox_max = np.array(
                    [bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]]
                )

                # Ensure min < max after transformation (negation can swap order).
                bbox_min, bbox_max = (
                    np.minimum(bbox_min, bbox_max),
                    np.maximum(bbox_min, bbox_max),
                )

                # Create SceneObject using shared helper.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={
                        "asset_source": "hssd",
                        "hssd_mesh_id": mesh_id,
                    },
                )

                successful_objects.append(scene_obj)

                console_logger.info(
                    f"HSSD asset retrieved successfully: {config.short_name}"
                )

            except Exception as e:
                console_logger.error(
                    f"Failed to process HSSD asset '{desc}': {e}", exc_info=True
                )
                failed_assets.append(
                    FailedAsset(index=index, description=desc, error_message=str(e))
                )

        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def _retrieve_objaverse_assets(
        self,
        request: AssetGenerationRequest,
        client: ObjaverseRetrievalClient | None = None,
        source_label: str = "objaverse",
    ) -> AssetGenerationResult:
        """Retrieve assets from Objaverse (ObjectThor) library using server client.

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with retrieved assets.
        """
        retrieval_client = client or self.objaverse_client
        if retrieval_client is None:
            raise RuntimeError(f"{source_label} retrieval client not initialized")
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Retrieving {len(request.object_descriptions)} assets from "
            f"{source_label} server"
        )

        # Create asset path configurations for output directories.
        asset_path_configs = self._create_asset_paths(
            object_descriptions=request.object_descriptions,
            short_names=request.short_names,
        )

        # Ensure output directories exist.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Create batch requests for Objaverse server with client-specified output dirs.
        retrieval_requests = [
            ObjaverseRetrievalServerRequest(
                object_description=desc,
                object_type=request.object_type.value,
                desired_dimensions=tuple(dims) if dims else None,
                output_dir=str(config.sdf_dir),
                scene_id=request.scene_id,
            )
            for desc, dims, config in zip(
                request.object_descriptions,
                request.desired_dimensions,
                asset_path_configs,
            )
        ]

        successful_objects: list[SceneObject] = []
        failed_assets: list[FailedAsset] = []

        # Submit batch to server and process streaming responses.
        for index, response in retrieval_client.retrieve_objects(retrieval_requests):
            desc = request.object_descriptions[index]
            short_name = request.short_names[index]
            config = asset_path_configs[index]

            try:
                console_logger.info(
                    "Processing Objaverse response "
                    f"{index+1}/{len(request.object_descriptions)}: '{desc}'"
                )

                # Server returns mesh path (already exported to our output_dir).
                if not response.results:
                    raise ValueError("No results returned from Objaverse server")

                result = response.results[0]  # Get top result.
                server_mesh_path = Path(result.mesh_path)
                mesh_id = result.objaverse_uid

                # Server exported to our specified output_dir, convert GLB to GLTF if
                # needed. Uses BlenderServer for crash isolation.
                if server_mesh_path.suffix.lower() == ".glb":
                    # Server exported GLB, convert to GLTF with Y-up coordinates.
                    gltf_path = server_mesh_path.with_suffix(".gltf")
                    self.blender_server.convert_glb_to_gltf(
                        input_path=server_mesh_path,
                        output_path=gltf_path,
                        export_yup=True,
                    )
                    server_mesh_path.unlink()  # Remove GLB after conversion.
                else:
                    # Already GLTF, use as-is.
                    gltf_path = server_mesh_path

                # Run VLM analysis for orientation, material and mass estimation.
                console_logger.info(
                    f"Running VLM analysis for Objaverse orientation/material/mass: "
                    f"{short_name}"
                )
                vlm_physics = analyze_mesh_orientation_and_material(
                    mesh_path=gltf_path,
                    vlm_service=self.vlm_service,
                    cfg=self.cfg,
                    elevation_degrees=self.side_view_elevation_degrees,
                    blender_server=self.blender_server,
                    num_side_views=self.num_side_views_for_physics_analysis,
                    prompt_type="generated",  # Full VLM analysis (not pre-canonicalized).
                    include_vertical_views=True,
                    debug_output_dir=self.debug_dir / short_name,
                )
                console_logger.info(
                    f"VLM analysis complete: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis}, material={vlm_physics.material}, "
                    f"mass={vlm_physics.mass_kg}kg"
                )

                # Use VLM's orientation, material, and mass determination.
                physics_analysis = MeshPhysicsAnalysis(
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    material=vlm_physics.material,
                    mass_kg=vlm_physics.mass_kg,
                    mass_range_kg=vlm_physics.mass_range_kg,
                )

                # Canonicalize mesh orientation to align with scenesmith canonical
                # (Z-up, Y-forward).
                console_logger.info(
                    f"Canonicalizing Objaverse mesh: up={vlm_physics.up_axis}, "
                    f"front={vlm_physics.front_axis} → +Y"
                )
                final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
                canonicalize_mesh(
                    gltf_path=gltf_path,
                    output_path=final_gltf_path,
                    up_axis=vlm_physics.up_axis,
                    front_axis=vlm_physics.front_axis,
                    blender_server=self.blender_server,
                    object_type=request.object_type,
                )

                # Generate collision geometry via collision server.
                collision_pieces = self._generate_collision_geometry(final_gltf_path)

                # Load mesh for bounding box calculation.
                mesh = load_mesh_as_trimesh(final_gltf_path, force_merge=True)

                sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
                generate_drake_sdf(
                    visual_mesh_path=final_gltf_path,
                    collision_pieces=collision_pieces,
                    physics_analysis=physics_analysis,
                    output_path=sdf_path,
                    asset_name=config.short_name,
                )

                # Extract bounding box from Y-up GLTF.
                bounds = mesh.bounds  # In Y-up coordinates (GLTF native format).

                # Transform from Y-up (GLTF) to Z-up (Drake) coordinate system.
                # Y-up → Z-up transformation: (x, y, z) → (x, -z, y)
                bbox_min_yup = bounds[0]
                bbox_max_yup = bounds[1]

                # Apply coordinate transformation.
                bbox_min = np.array(
                    [bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]]
                )
                bbox_max = np.array(
                    [bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]]
                )

                # Ensure min < max after transformation (negation can swap order).
                bbox_min, bbox_max = (
                    np.minimum(bbox_min, bbox_max),
                    np.maximum(bbox_min, bbox_max),
                )

                # Create SceneObject using shared helper.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={
                        "asset_source": result.asset_source,
                        "objaverse_mesh_id": mesh_id,
                        "catalog_id": mesh_id,
                        "license": result.license,
                    },
                )

                successful_objects.append(scene_obj)

                console_logger.info(
                    f"Objaverse asset retrieved successfully: {config.short_name}"
                )

            except Exception as e:
                console_logger.error(
                    f"Failed to process Objaverse asset '{desc}': {e}", exc_info=True
                )
                failed_assets.append(
                    FailedAsset(index=index, description=desc, error_message=str(e))
                )

        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )
