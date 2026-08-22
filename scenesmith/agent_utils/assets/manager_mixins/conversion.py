import logging
import shutil
import time

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.asset_router.dataclasses import (
    ArticulatedGeometry,
    AssetItem,
    GeneratedGeometry,
)
from scenesmith.agent_utils.geometry.sdf_generator import add_self_collision_filter
from scenesmith.agent_utils.geometry.sdf_mesh_utils import (
    combine_sdf_meshes_at_joint_angles,
)
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationError,
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.geometry_generation_server.pipelines.sam_provider import (
    sam_provider_config_from_mapping,
)
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, SceneObject
from scenesmith.agent_utils.structure.thin_covering_generator import (
    infer_thin_covering_shape,
)

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

HSSD_CANONICAL_CONVERSION_VERSION = 4

from scenesmith.agent_utils.assets.asset_models import (
    AssetGenerationRequest,
    AssetPathConfig,
    FailedAsset,
)


class AssetConversionMixin:
    """Validated geometry conversion and asset-batch processing."""

    def _generate_geometry_with_validation(
        self, item: AssetItem, request: AssetGenerationRequest
    ) -> GeneratedGeometry | ArticulatedGeometry | None:
        """Generate/retrieve validated geometry for a single item. Thread-safe.

        This method only performs HTTP-based operations (geometry server, HSSD server,
        BlenderServer for validation rendering) and is safe to call from worker threads.

        Args:
            item: The asset item to generate/retrieve.
            request: Original request (for style_context).

        Returns:
            GeneratedGeometry or ArticulatedGeometry if successful,
            None if all strategies/candidates exhausted.
        """
        return self.router.generate_with_validation(
            item=item,
            geometry_client=self.geometry_client,
            image_generator=self.image_generator,
            images_dir=self.images_dir,
            geometry_dir=self.geometry_dir,
            debug_dir=self.debug_dir,
            style_context=request.style_context,
            hssd_client=self.hssd_client,
            objaverse_client=self.objaverse_client,
            polyhaven_client=self.polyhaven_client,
            articulated_client=self.articulated_client,
            materials_client=self.materials_client,
            scene_id=request.scene_id,
        )

    def _convert_generated_to_scene_object(
        self,
        item: "AssetItem",
        generated: "GeneratedGeometry",
        request: AssetGenerationRequest,
    ) -> SceneObject:
        """Convert validated geometry to SceneObject. Must run on main thread.

        This method uses bpy for GLB→GLTF conversion and must be called from the
        main thread, not from ThreadPoolExecutor workers.

        Args:
            item: The asset item that was generated.
            generated: The validated geometry from router.
            request: Original request (for object_type).

        Returns:
            SceneObject ready for scene placement.

        Raises:
            Exception: If mesh conversion or SDF generation fails.
        """
        # Derive base_name from geometry path (already has unique timestamp or HSSD ID).
        base_name = generated.geometry_path.stem

        config = AssetPathConfig(
            description=item.description,
            short_name=item.short_name,
            image_path=generated.image_path,
            geometry_path=generated.geometry_path,
            sdf_dir=self.sdf_dir / base_name,
        )
        config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Thin coverings use simplified conversion: no VLM analysis.
        # Wall thin coverings (paintings, posters) get collision geometry.
        if generated.asset_source == "thin_covering":
            is_wall_covering = request.object_type == ObjectType.WALL_MOUNTED

            # Only add collision for wall coverings (paintings, posters).
            collision_dims = None
            collision_shape = "rectangular"
            if is_wall_covering and item.dimensions:
                # Wall covering dims: (width, depth, height) where depth is thickness.
                thickness = (
                    self.cfg.asset_manager.router.strategies.thin_covering.thickness_m
                )
                collision_dims = (item.dimensions[0], thickness, item.dimensions[2])
                collision_shape = infer_thin_covering_shape(item.description)

            sdf_path, final_gltf_path, bbox_min, bbox_max = (
                self._convert_thin_covering_to_simulation_asset(
                    geometry_path=generated.geometry_path,
                    config=config,
                    collision_dims=collision_dims,
                    collision_shape=collision_shape,
                )
            )
            initial_scale = 1.0  # Thin coverings don't scale the mesh.
        else:
            # Convert validated geometry to simulation asset (physics analysis, SDF).
            sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale = (
                self._convert_mesh_to_simulation_asset(
                    geometry_path=generated.geometry_path,
                    config=config,
                    object_type=request.object_type,
                    desired_dimensions=item.dimensions,
                    asset_source=generated.asset_source,
                    canonical_up=generated.canonical_up,
                    canonical_front=generated.canonical_front,
                )
            )

        # Build additional metadata using explicit asset_source from GeneratedGeometry.
        additional_metadata = {"asset_source": generated.asset_source}
        additional_metadata.update(
            self._asset_conversion_metadata(generated.asset_source)
        )
        if generated.hssd_id is not None:
            additional_metadata["hssd_mesh_id"] = generated.hssd_id
        if generated.objaverse_uid is not None:
            additional_metadata["objaverse_mesh_id"] = generated.objaverse_uid
        if generated.catalog_id is not None:
            additional_metadata["catalog_id"] = generated.catalog_id
        if generated.license is not None:
            additional_metadata["license"] = generated.license
        if generated.ontology_path is not None:
            additional_metadata["ontology_path"] = generated.ontology_path
        if generated.placement_classes:
            additional_metadata["placement_classes"] = list(generated.placement_classes)
        if generated.canonical_up is not None:
            additional_metadata["canonical_up"] = generated.canonical_up
        if generated.canonical_front is not None:
            additional_metadata["canonical_front"] = generated.canonical_front
        if generated.support_zones:
            additional_metadata["support_zones"] = list(generated.support_zones)
        if generated.clearance_zones:
            additional_metadata["clearance_zones"] = list(generated.clearance_zones)
        if generated.quality_score is not None:
            additional_metadata["asset_quality_score"] = generated.quality_score
        if generated.thumbnail is not None:
            additional_metadata["asset_thumbnail"] = generated.thumbnail
        if generated.catalog_semantics is not None:
            additional_metadata["catalog_semantics"] = generated.catalog_semantics

        # Add thin_covering-specific metadata for physics validation.
        if generated.asset_source == "thin_covering":
            additional_metadata["width_m"] = item.dimensions[0]
            additional_metadata["depth_m"] = item.dimensions[1]
            additional_metadata["shape"] = infer_thin_covering_shape(item.description)
            # Wall coverings use Drake collision; floor/manipuland use 2D OBB overlap.
            additional_metadata["is_wall_covering"] = (
                request.object_type == ObjectType.WALL_MOUNTED
            )

        # Keep original object_type - thin coverings are identified via asset_source
        # metadata, not object_type. This preserves semantic category (FURNITURE,
        # WALL_MOUNTED, MANIPULAND) for stage-based filtering in snapshots.
        object_type = request.object_type

        # Create SceneObject.
        return self._create_scene_object(
            config=config,
            object_type=object_type,
            sdf_path=sdf_path,
            final_gltf_path=final_gltf_path,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            additional_metadata=additional_metadata,
            scale_factor=initial_scale,
        )

    def _convert_articulated_to_scene_object(
        self, articulated: ArticulatedGeometry, request: AssetGenerationRequest
    ) -> SceneObject:
        """Convert articulated retrieval result to SceneObject.

        Unlike generated assets, articulated objects already have:
        - Pre-processed SDF with links and joints
        - Bounding box at default pose (joints=0)
        - No need for VLM analysis or mesh canonicalization

        We combine the visual meshes at default pose for geometry_path (needed
        for collision checks, support surface extraction, snapping).

        Args:
            articulated: The articulated geometry from router.
            request: Original request (for object_type).

        Returns:
            SceneObject ready for scene placement.
        """
        item = articulated.item
        safe_name = self._sanitize_filename(item.short_name)
        timestamp = int(time.time())
        base_name = f"{safe_name}_{timestamp}"

        # Create output directory for combined geometry.
        output_dir = self.geometry_dir / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy articulated SDF directory to output for replay and export.
        # The SDF references meshes via relative paths, so we copy the entire directory.
        source_sdf_dir = articulated.sdf_path.parent
        dest_sdf_dir = self.sdf_dir / base_name
        console_logger.info(
            f"Copying articulated SDF directory from {source_sdf_dir} to {dest_sdf_dir}"
        )
        shutil.copytree(source_sdf_dir, dest_sdf_dir)
        copied_sdf_path = dest_sdf_dir / articulated.sdf_path.name

        # Add self-collision filtering if enabled.
        if self.cfg.asset_manager.articulated.enable_self_collision_filtering:
            add_self_collision_filter(copied_sdf_path)

        # Fix ArtVIP texture paths: GLTF files reference textures with relative paths,
        # but textures are in *_meshes/ subdirectories. Copy textures to parent dir.
        for meshes_subdir in dest_sdf_dir.glob("*_meshes"):
            for texture_file in meshes_subdir.glob("*.png"):
                dest_texture = dest_sdf_dir / texture_file.name
                if not dest_texture.exists():
                    shutil.copy2(texture_file, dest_texture)

        # Combine SDF visual meshes at default pose (joints=0) for geometry operations.
        console_logger.info(
            f"Combining articulated meshes at default pose for '{item.description}'"
        )
        combined_mesh = combine_sdf_meshes_at_joint_angles(
            copied_sdf_path, use_max_angles=False
        )

        # Save combined mesh as GLTF for collision checks, snapping, etc.
        combined_gltf_path = output_dir / f"{safe_name}_combined.gltf"
        combined_mesh.export(combined_gltf_path)

        console_logger.info(
            f"Articulated asset combined mesh saved to {combined_gltf_path}"
        )

        # Build metadata for provenance tracking.
        metadata = {
            "asset_source": "articulated",
            "articulated_source": articulated.source,
            "articulated_id": articulated.object_id,
            "is_articulated": True,
            "generation_timestamp": time.time(),
        }

        # Create SceneObject with copied SDF path and combined geometry.
        scene_obj = SceneObject(
            object_id=self.registry.generate_unique_id(item.short_name),
            object_type=request.object_type,
            name=item.short_name,
            description=item.description,
            transform=RigidTransform(),  # Will be set during placement.
            geometry_path=combined_gltf_path,
            sdf_path=copied_sdf_path,
            image_path=None,  # No generated image for articulated assets.
            bbox_min=np.array(articulated.bounding_box_min),
            bbox_max=np.array(articulated.bounding_box_max),
            metadata=metadata,
        )

        # Register the asset for reuse.
        self.registry.register(scene_obj)

        console_logger.info(
            f"Articulated asset registered: {item.short_name} "
            f"(source={articulated.source}, id={articulated.object_id})"
        )

        return scene_obj

    def _create_asset_paths(
        self, object_descriptions: list[str], short_names: list[str]
    ) -> list[AssetPathConfig]:
        """Create file paths and identifiers for each asset to be generated.

        Args:
            object_descriptions: List of object descriptions to generate.
            short_names: List of short names for filesystem-safe file naming.

        Returns:
            List of AssetPathConfig objects containing asset paths and metadata.
        """
        asset_paths = []
        for desc, short_name in zip(object_descriptions, short_names):
            # Use sanitized short name for file naming.
            safe_name = self._sanitize_filename(short_name)
            timestamp = int(time.time())
            base_name = f"{safe_name}_{timestamp}"

            asset_paths.append(
                AssetPathConfig(
                    description=desc,
                    short_name=short_name,
                    image_path=self.images_dir / f"{base_name}.png",
                    geometry_path=self.geometry_dir / f"{base_name}.glb",
                    sdf_dir=self.sdf_dir / base_name,
                )
            )
        return asset_paths

    def _generate_images(
        self,
        request: AssetGenerationRequest,
        asset_paths_configs: list[AssetPathConfig],
    ) -> None:
        """Generate images for all assets using the image generator.

        Args:
            request: Asset generation request with style and operation details.
            asset_paths_configs: List of asset path configurations.
        """
        style_prompt = request.style_context or "Modern style"
        console_logger.info(f"Generating {len(request.object_descriptions)} images")
        console_logger.debug(f"Style prompt: {style_prompt}")

        output_paths = [config.image_path for config in asset_paths_configs]

        start_time = time.time()
        self.image_generator.generate_images(
            style_prompt=style_prompt,
            object_descriptions=request.object_descriptions,
            output_paths=output_paths,
        )

        elapsed = time.time() - start_time
        console_logger.info(
            f"Generated {len(request.object_descriptions)} images in "
            f"{elapsed:.2f} seconds"
        )

    def _process_assets_to_scene_objects(
        self, request: AssetGenerationRequest, asset_path_configs: list[AssetPathConfig]
    ) -> tuple[list[SceneObject], list[FailedAsset]]:
        """Convert generated images to 3D assets and create SceneObjects.

        Uses batch processing to optimize GPU utilization by pipelining geometry
        generation and Drake SDF conversion. Handles failures gracefully by
        collecting failed assets instead of raising exceptions, allowing all
        generated geometries to be processed.

        Args:
            request: Asset generation request.
            asset_path_configs: List of asset path configurations.

        Returns:
            Tuple of (successful_objects, failed_assets). The successful_objects
            list contains SceneObject instances ready for placement. The failed_assets
            list contains FailedAsset instances with error details.
        """
        if not asset_path_configs:
            return [], []

        # Create Drake asset directories for all configs.
        for config in asset_path_configs:
            config.sdf_dir.mkdir(parents=True, exist_ok=True)

        # Prepare batch geometry generation requests.
        geometry_requests = []
        for config in asset_path_configs:
            expected_filename = config.geometry_path.name

            # Extract backend configuration.
            backend = self.cfg.asset_manager.backend

            # Prepare SAM3D config if backend is sam3d.
            sam3d_config = None
            if backend == "sam3d":
                sam3d_cfg = self.cfg.asset_manager.sam3d
                mode = sam3d_cfg.mode
                sam3d_config = sam_provider_config_from_mapping(
                    sam3d_cfg,
                    object_description=(
                        config.description if mode == "object_description" else None
                    ),
                )

            geometry_request = GeometryGenerationServerRequest(
                image_path=str(config.image_path),
                output_dir=str(self.geometry_dir),
                prompt=config.description,
                debug_folder=str(self.debug_dir),
                output_filename=expected_filename,
                backend=backend,
                sam3d_config=sam3d_config,
                scene_id=request.scene_id,
            )
            geometry_requests.append(geometry_request)

        console_logger.info(
            f"Submitting batch geometry generation for {len(geometry_requests)} assets"
        )

        # Initialize result tracking.
        scene_objects = []
        failed_assets = []

        # Process batch results as they stream back.
        # This enables pipelining: Drake conversion for asset N while GPU processes
        # asset N+1.
        for index, result in self.geometry_client.generate_geometries(
            geometry_requests
        ):
            # Handle geometry generation failures.
            if isinstance(result, GeometryGenerationError):
                console_logger.error(
                    f"Geometry generation failed for asset {index + 1}/"
                    f"{len(asset_path_configs)} ({asset_path_configs[index].description}): "
                    f"{result.error_message}"
                )
                failed_assets.append(
                    FailedAsset(
                        index=index,
                        description=asset_path_configs[index].description,
                        error_message=result.error_message,
                    )
                )
                continue

            try:
                config = asset_path_configs[index]
                server_geometry_path = Path(result.geometry_path)

                console_logger.info(
                    f"Converting asset {index + 1}/{len(asset_path_configs)} to Drake "
                    f"format: {config.description}"
                )

                # Process mesh: VLM → canonicalize → scale → collision → SDF.
                sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale = (
                    self._convert_mesh_to_simulation_asset(
                        geometry_path=server_geometry_path,
                        config=config,
                        object_type=request.object_type,
                        desired_dimensions=request.desired_dimensions[index],
                    )
                )

                # Create the SceneObject.
                scene_obj = self._create_scene_object(
                    config=config,
                    object_type=request.object_type,
                    sdf_path=sdf_path,
                    final_gltf_path=final_gltf_path,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    additional_metadata={"asset_source": "generated"},
                    scale_factor=initial_scale,
                )

                scene_objects.append(scene_obj)
                console_logger.info(
                    f"Successfully generated asset {index + 1}/{len(asset_path_configs)}: "
                    f"{config.description}"
                )

            except Exception as e:
                # Log failure but continue processing remaining assets.
                console_logger.error(
                    f"Failed to process asset {index + 1}/{len(asset_path_configs)} "
                    f"({asset_path_configs[index].description}): {e}",
                    exc_info=True,
                )
                failed_assets.append(
                    FailedAsset(
                        index=index,
                        description=asset_path_configs[index].description,
                        error_message=str(e),
                    )
                )

        # Log summary.
        if failed_assets:
            console_logger.warning(
                f"Asset generation completed with {len(failed_assets)} failure(s) "
                f"and {len(scene_objects)} success(es)"
            )
        else:
            console_logger.info(
                f"Successfully processed all {len(scene_objects)} assets"
            )

        return scene_objects, failed_assets
