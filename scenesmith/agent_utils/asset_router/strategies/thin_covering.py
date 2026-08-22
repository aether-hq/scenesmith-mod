"""Asset router for LLM-advised asset generation."""

import json
import logging
import time

from pathlib import Path
from typing import TYPE_CHECKING

from scenesmith.agent_utils.asset_router.dataclasses import (
    AssetItem,
    GeneratedGeometry,
    ValidationResult,
)
from scenesmith.agent_utils.assets.material_generator import (
    MaterialGenerator,
    MaterialGeneratorConfig,
)
from scenesmith.agent_utils.blender.renderer import MATERIAL_VALIDATION_LIGHT_ENERGY
from scenesmith.agent_utils.materials_retrieval_server import (
    MaterialsRetrievalServerRequest,
)
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.agent_utils.structure.thin_covering_generator import (
    create_circular_thin_covering_glb,
    create_rectangular_thin_covering_glb,
    infer_thin_covering_shape,
)
from scenesmith.prompts import AssetRouterPrompts, prompt_manager
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.assets.image_generation import BaseImageGenerator
    from scenesmith.agent_utils.materials_retrieval_server import (
        MaterialsRetrievalClient,
    )

console_logger = logging.getLogger(__name__)


class ThinCoveringAssetStrategyMixin:
    """Retrieved and generated thin-covering strategy."""

    def _try_thin_covering_strategy(
        self,
        item: AssetItem,
        max_retries: int,
        materials_client: "MaterialsRetrievalClient | None",
        image_generator: "BaseImageGenerator | None",
        geometry_dir: Path,
        debug_dir: Path,
        scene_id: str | None = None,
    ) -> GeneratedGeometry | None:
        """Try the thin covering strategy for procedural textured surface generation.

        Generates thin textured meshes with PBR materials from the materials
        retrieval server. For floor agents, creates horizontal textured surfaces
        (rugs). For wall agents, creates vertical wall-mounted surfaces (paintings).

        Falls back to AI texture generation if all retrieval candidates fail
        validation and the generator config is enabled.

        Args:
            item: The asset item to generate.
            max_retries: Number of material candidates to try with VLM validation.
                0 = no validation (use first material).
            materials_client: Client for materials retrieval server.
            image_generator: Image generator for fallback texture generation.
            geometry_dir: Directory to save generated GLTF.
            debug_dir: Directory to save debug outputs (validation renders).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry if successful, None if retrieval/validation fails.
        """
        if materials_client is None:
            console_logger.warning(
                f"Materials client not available for '{item.description}'"
            )
            return None

        # Wall agent uses vertical geometry (wall-mounted).
        is_wall_mode = self.agent_type == AgentType.WALL_MOUNTED

        # Infer shape from description (circular vs rectangular).
        shape = infer_thin_covering_shape(item.description)

        # Extract dimensions - wall mode uses width/height, floor mode uses width/depth.
        if is_wall_mode:
            if not item.dimensions or len(item.dimensions) < 3:
                console_logger.warning(
                    f"Wall thin covering '{item.description}' missing dimensions "
                    f"(need width, depth, height)"
                )
                return None
            width = item.dimensions[0]
            height = item.dimensions[2]
            console_logger.info(
                f"Generating {shape} wall thin covering '{item.description}' "
                f"({width:.2f}m x {height:.2f}m)"
            )
        else:
            if not item.dimensions or len(item.dimensions) < 2:
                console_logger.warning(
                    f"Floor thin covering '{item.description}' missing dimensions "
                    f"(need width, depth)"
                )
                return None
            width = item.dimensions[0]
            depth = item.dimensions[1]
            console_logger.info(
                f"Generating {shape} floor thin covering '{item.description}' "
                f"({width:.2f}m x {depth:.2f}m)"
            )

        # Request materials from server.
        request = MaterialsRetrievalServerRequest(
            material_description=item.description,
            output_dir=str(geometry_dir),
            scene_id=scene_id,
            num_candidates=max(1, max_retries),
        )

        try:
            # Fetch materials (single request returns single response).
            responses = list(materials_client.retrieve_materials([request]))
            if not responses:
                console_logger.warning(
                    f"No material response for thin covering '{item.description}'"
                )
                return None
            _, response = responses[0]
            materials = response.results
        except Exception as e:
            console_logger.error(
                f"Materials retrieval failed for thin covering '{item.description}': {e}"
            )
            return None

        if not materials:
            console_logger.warning(
                f"No materials found for thin covering '{item.description}'"
            )
            return None
        console_logger.info(
            f"Got {len(materials)} material candidates for '{item.description}'"
        )

        thin_covering_cfg = self.cfg.asset_manager.router.strategies.thin_covering
        thickness = thin_covering_cfg.thickness_m
        # single_image (artwork) uses cover mode, tileable uses configured scale.
        is_single_image = item.thin_covering_type == "single_image"
        texture_scale = None if is_single_image else thin_covering_cfg.texture_scale

        # If max_retries=0, use first material without validation.
        if max_retries == 0:
            material = materials[0]
            console_logger.info(
                f"Using first material without validation: {material.material_id}"
            )
            return self._generate_thin_covering_geometry(
                item=item,
                material_path=Path(material.material_path),
                width=width,
                second_dim=height if is_wall_mode else depth,
                thickness=thickness,
                geometry_dir=geometry_dir,
                is_wall_mode=is_wall_mode,
                shape=shape,
                texture_scale=texture_scale,
            )

        # Validation loop: try each material until one passes.
        for i, material in enumerate(materials):
            console_logger.info(
                f"Validating material {i + 1}/{len(materials)}: "
                f"{material.material_id} (score={material.similarity_score:.3f})"
            )

            # Generate geometry with this material.
            result = self._generate_thin_covering_geometry(
                item=item,
                material_path=Path(material.material_path),
                width=width,
                second_dim=height if is_wall_mode else depth,
                thickness=thickness,
                geometry_dir=geometry_dir,
                is_wall_mode=is_wall_mode,
                shape=shape,
                texture_scale=texture_scale,
            )

            if result is None:
                continue

            # Validate with VLM.
            validation = self._validate_thin_covering(
                mesh_path=result.geometry_path,
                description=item.description,
                debug_dir=debug_dir,
                is_wall_mode=is_wall_mode,
            )
            if validation.is_acceptable:
                console_logger.info(
                    f"Validation passed for '{item.description}': "
                    f"{validation.reason}"
                )
                return result

            console_logger.info(
                f"Validation failed for '{item.description}': "
                f"{validation.reason}. Suggestions: {validation.suggestions}"
            )

        console_logger.warning(
            f"All {len(materials)} materials failed validation for "
            f"thin covering '{item.description}'"
        )

        # Fallback to AI texture generation if enabled and image_generator available.
        return self._try_generated_thin_covering(
            item=item,
            image_generator=image_generator,
            width=width,
            second_dim=height if is_wall_mode else depth,
            thickness=thickness,
            geometry_dir=geometry_dir,
            debug_dir=debug_dir,
            is_wall_mode=is_wall_mode,
            shape=shape,
        )

    def _try_generated_thin_covering(
        self,
        item: AssetItem,
        image_generator: "BaseImageGenerator | None",
        width: float,
        second_dim: float,
        thickness: float,
        geometry_dir: Path,
        debug_dir: Path,
        is_wall_mode: bool,
        shape: str,
    ) -> GeneratedGeometry | None:
        """Try generating thin covering texture with AI when retrieval fails.

        Creates MaterialGenerator locally (lightweight, no GPU/server resources).

        Args:
            item: The asset item to generate.
            image_generator: Image generator for texture generation.
            width: Width in meters.
            second_dim: Height (wall) or depth (floor) in meters.
            thickness: Thickness in meters.
            geometry_dir: Directory to save generated GLTF.
            debug_dir: Directory for debug outputs.
            is_wall_mode: True for wall coverings, False for floor.
            shape: "rectangular" or "circular".

        Returns:
            GeneratedGeometry if successful, None otherwise.
        """
        # Check if generator is enabled in config.
        gen_cfg = self.cfg.asset_manager.router.strategies.thin_covering.generator
        if not gen_cfg.enabled:
            console_logger.info("Thin covering generator disabled in config")
            return None

        if image_generator is None:
            console_logger.warning(
                "Image generator not available for thin covering fallback"
            )
            return None

        console_logger.info(f"Trying AI texture generation for '{item.description}'")

        # Create output directory for generated materials.
        generated_materials_dir = geometry_dir / "generated_materials"
        generated_materials_dir.mkdir(exist_ok=True)

        # Create MaterialGenerator locally.
        material_generator = MaterialGenerator(
            config=MaterialGeneratorConfig(
                enabled=True,
                backend=gen_cfg.backend,
                max_retries=gen_cfg.max_retries,
                default_roughness=gen_cfg.default_roughness,
                texture_scale=gen_cfg.texture_scale,
            ),
            output_dir=generated_materials_dir,
            image_generator=image_generator,
        )

        # Determine if single image (artwork) or tileable texture.
        is_single_image = item.thin_covering_type == "single_image"

        for retry in range(material_generator.config.max_retries):
            console_logger.info(
                f"Generation attempt {retry + 1}/{material_generator.config.max_retries}"
            )

            if is_single_image:
                # For circular shapes, always use square (1:1) since circular UV mapping
                # expects square textures. For rectangular, use actual dimensions.
                if shape == "circular":
                    generated = material_generator.generate_artwork(
                        description=item.description, width=1.0, height=1.0
                    )
                else:
                    generated = material_generator.generate_artwork(
                        description=item.description, width=width, height=second_dim
                    )
            else:
                # Tileable textures always use square - tiling handles non-square surfaces.
                generated = material_generator.generate_material(
                    description=item.description
                )

            if generated is None:
                continue

            result = self._generate_thin_covering_geometry(
                item=item,
                material_path=generated.path,
                width=width,
                second_dim=second_dim,
                thickness=thickness,
                geometry_dir=geometry_dir,
                is_wall_mode=is_wall_mode,
                shape=shape,
                texture_scale=generated.texture_scale,
            )

            if result is None:
                continue

            # Validate with VLM.
            validation = self._validate_thin_covering(
                mesh_path=result.geometry_path,
                description=item.description,
                debug_dir=debug_dir,
                is_wall_mode=is_wall_mode,
            )

            if validation.is_acceptable:
                console_logger.info(
                    f"Generated material passed validation: {validation.reason}"
                )
                return result

            console_logger.info(
                f"Generated material failed validation: {validation.reason}"
            )

        console_logger.warning(
            f"All {material_generator.config.max_retries} generation attempts "
            f"failed for '{item.description}'"
        )
        return None

    def _generate_thin_covering_geometry(
        self,
        item: AssetItem,
        material_path: Path,
        width: float,
        second_dim: float,
        thickness: float,
        geometry_dir: Path,
        is_wall_mode: bool,
        texture_scale: float | None,
        shape: str = "rectangular",
    ) -> GeneratedGeometry | None:
        """Generate thin covering GLTF mesh with given material.

        Creates either a horizontal floor covering (rug) or vertical wall
        covering (painting/mirror) depending on is_wall_mode.

        Args:
            item: The asset item.
            material_path: Path to material folder with PBR textures.
            width: Width in meters (X dimension).
            second_dim: Height for wall mode, depth for floor mode.
            thickness: Thickness in meters.
            geometry_dir: Directory to save the GLB.
            is_wall_mode: True for vertical wall surfaces, False for floor.
            texture_scale: Meters per texture tile (None = cover mode, no tiling).
            shape: "rectangular" or "circular" (applies to both floor and wall).

        Returns:
            GeneratedGeometry if successful, None on error.
        """
        timestamp = int(time.time())
        base_name = f"{item.short_name}_{timestamp}"
        glb_path = geometry_dir / f"{base_name}.glb"

        try:
            if shape == "circular":
                # Circular covering (floor or wall).
                radius = min(width, second_dim) / 2.0
                create_circular_thin_covering_glb(
                    radius=radius,
                    thickness=thickness,
                    material_folder=material_path,
                    output_path=glb_path,
                    texture_scale=texture_scale,
                    is_wall=is_wall_mode,
                )
            else:
                # Rectangular covering (floor or wall).
                create_rectangular_thin_covering_glb(
                    width=width,
                    second_dim=second_dim,
                    thickness=thickness,
                    material_folder=material_path,
                    output_path=glb_path,
                    texture_scale=texture_scale,
                    is_wall=is_wall_mode,
                )

            console_logger.info(f"Generated thin covering mesh: {glb_path}")

            return GeneratedGeometry(
                geometry_path=glb_path, item=item, asset_source="thin_covering"
            )

        except Exception as e:
            console_logger.error(f"Failed to generate thin covering mesh: {e}")
            return None

    def _validate_thin_covering(
        self, mesh_path: Path, description: str, debug_dir: Path, is_wall_mode: bool
    ) -> ValidationResult:
        """Validate thin covering material matches description using VLM.

        Renders appropriate view (top-down for floor, frontal for wall) and
        validates that the texture pattern matches the description.

        Args:
            mesh_path: Path to the thin covering GLB mesh.
            description: Original description for validation.
            debug_dir: Directory to save validation renders.
            is_wall_mode: True for wall coverings, False for floor coverings.

        Returns:
            ValidationResult with is_acceptable, reason, and suggestions.
        """
        validation_dir = debug_dir / f"{mesh_path.stem}_thin_covering_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        # Render appropriate view based on orientation.
        image_path = self._render_thin_covering_for_validation(
            mesh_path=mesh_path, output_dir=validation_dir, is_wall_mode=is_wall_mode
        )

        if image_path is None:
            return ValidationResult(
                is_acceptable=False,
                reason="Failed to render thin covering for validation",
                suggestions=["Check Blender server availability"],
            )

        image_base64 = encode_image_to_base64(image_path)

        system_prompt = prompt_manager.get_prompt(
            AssetRouterPrompts.THIN_COVERING_VALIDATION_PROMPT
        )
        user_prompt = (
            f"Validate this {'wall' if is_wall_mode else 'floor'} "
            f"covering texture for the description: {description}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            },
        ]

        model = self.cfg.openai.model
        reasoning_effort = self.cfg.openai.reasoning_effort.asset_validation
        verbosity = self.cfg.openai.verbosity.asset_validation
        vision_detail = self.cfg.openai.vision_detail

        try:
            start_time = time.time()
            response_text = self.vlm_service.create_completion(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                response_format={"type": "json_object"},
                vision_detail=vision_detail,
            )
            elapsed = time.time() - start_time
            response_json = json.loads(response_text)
            console_logger.info(
                f"Thin covering validation completed in {elapsed:.1f}s for "
                f"'{description}':\n{response_json}"
            )
        except Exception as e:
            console_logger.error(f"Thin covering VLM validation failed: {e}")
            return ValidationResult(
                is_acceptable=False,
                reason=f"Validation call failed: {e}",
                suggestions=["Retry validation"],
            )

        return ValidationResult(
            is_acceptable=response_json.get("is_acceptable", False),
            reason=response_json.get("reason", "Unknown"),
            suggestions=response_json.get("suggestions", []),
        )

    def _render_thin_covering_for_validation(
        self, mesh_path: Path, output_dir: Path, is_wall_mode: bool
    ) -> Path | None:
        """Render thin covering for VLM validation.

        For floor coverings, renders top-down view. For wall coverings,
        renders frontal view.

        Args:
            mesh_path: Path to the thin covering GLB mesh.
            output_dir: Directory to save rendered images.
            is_wall_mode: True for wall coverings (frontal), False for floor (top-down).

        Returns:
            Path to rendered image, or None if rendering failed.
        """
        # Elevation from config.
        elevation = self.side_view_elevation_degrees

        if is_wall_mode:
            # Wall covering: frontal view from +Y (object front face is at +Y).
            num_side_views = 1
            include_vertical = False
            # Start azimuth at 90° to position camera at +Y.
            start_azimuth = 90.0
        else:
            # Floor covering: top-down view.
            num_side_views = 0
            include_vertical = True
            start_azimuth = 0.0

        # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
        # due to GPU/OpenGL state corruption from fork.
        # Disable coordinate frame for cleaner validation renders.
        # Use lower light energy to avoid washing out material colors.
        if self.blender_server is None or not self.blender_server.is_running():
            raise RuntimeError(
                "BlenderServer required for thin covering validation. "
                "Forked workers cannot safely use embedded bpy."
            )
        image_paths = self.blender_server.render_multiview_for_analysis(
            mesh_path=mesh_path,
            output_dir=output_dir,
            elevation_degrees=elevation,
            num_side_views=num_side_views,
            include_vertical_views=include_vertical,
            start_azimuth_degrees=start_azimuth,
            show_coordinate_frame=False,
            light_energy=MATERIAL_VALIDATION_LIGHT_ENERGY,
            taa_samples=self.validation_taa_samples,
        )

        return image_paths[0] if image_paths else None
