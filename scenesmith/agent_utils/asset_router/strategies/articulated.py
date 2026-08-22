"""Asset router for LLM-advised asset generation."""

import json
import logging
import time

from pathlib import Path
from typing import TYPE_CHECKING

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalClient,
    ArticulatedRetrievalServerRequest,
)
from scenesmith.agent_utils.articulated_retrieval_server.dataclasses import (
    ArticulatedRetrievalResult,
)
from scenesmith.agent_utils.asset_router.dataclasses import (
    ArticulatedGeometry,
    AssetItem,
    ValidationResult,
)
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.prompts import AssetRouterPrompts, prompt_manager
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)


class ArticulatedAssetStrategyMixin:
    """Articulated retrieval, validation, and rendering strategy."""

    def _try_articulated_strategy(
        self,
        item: AssetItem,
        max_retries: int,
        debug_dir: Path,
        articulated_client: "ArticulatedRetrievalClient | None",
    ) -> ArticulatedGeometry | None:
        """Try the articulated strategy for objects with doors/drawers/etc.

        Retrieves pre-processed SDF assets from articulated object libraries
        (PartNet-Mobility, ArtVIP) using CLIP semantic matching and bounding
        box ranking via the articulated retrieval server.

        Args:
            item: The asset item to retrieve.
            max_retries: Number of candidates to try (from router config).
            debug_dir: Directory to save debug outputs (validation renders).
            articulated_client: Client for articulated retrieval server.

        Returns:
            ArticulatedGeometry if successful, None if no suitable candidate found.
        """
        if articulated_client is None:
            console_logger.warning(
                f"Articulated client not available for '{item.description}'"
            )
            return None

        # Map EITHER to concrete type based on which agent is calling.
        object_type = item.object_type.value.upper()
        if object_type == "EITHER":
            if self.agent_type == AgentType.FURNITURE:
                object_type = "FURNITURE"
            elif self.agent_type == AgentType.WALL_MOUNTED:
                object_type = "WALL_MOUNTED"
            else:
                object_type = "MANIPULAND"
            console_logger.debug(
                f"Mapped 'EITHER' to '{object_type}' for {self.agent_type} agent"
            )

        # Create output directory for retrieved meshes.
        articulated_output_dir = debug_dir / "articulated_meshes"
        articulated_output_dir.mkdir(parents=True, exist_ok=True)

        # Request enough candidates for validation retries.
        num_candidates = max(1, max_retries)
        request = ArticulatedRetrievalServerRequest(
            object_description=item.description,
            object_type=object_type,
            output_dir=str(articulated_output_dir),
            desired_dimensions=tuple(item.dimensions) if item.dimensions else None,
            num_candidates=num_candidates,
        )

        # Fetch candidates via server.
        try:
            responses = list(articulated_client.retrieve_objects([request]))
            if not responses:
                console_logger.warning(
                    f"No articulated response for '{item.description}'"
                )
                return None

            _, response = responses[0]
            candidates = response.results

        except Exception as e:
            console_logger.error(
                f"Articulated retrieval failed for '{item.description}': {e}"
            )
            return None

        if not candidates:
            console_logger.warning(
                f"No articulated candidates found for '{item.description}'"
            )
            return None

        console_logger.info(
            f"Got {len(candidates)} articulated candidates for '{item.description}'"
        )

        # If max_retries=0, return first candidate without validation.
        if max_retries == 0:
            candidate = candidates[0]
            console_logger.info(
                f"Returning first articulated candidate without validation: "
                f"{candidate.object_id}"
            )
            return self._result_to_articulated_geometry(result=candidate, item=item)

        # Validation loop: try each candidate until one passes.
        for i, candidate in enumerate(candidates):
            console_logger.info(
                f"Validating articulated candidate {i + 1}/{len(candidates)}: "
                f"{candidate.object_id} (clip={candidate.clip_score:.3f}, "
                f"bbox={candidate.bbox_score:.3f})"
            )

            # Validate with VLM.
            validation = self._validate_articulated_result(
                result=candidate, description=item.description, debug_dir=debug_dir
            )
            if validation.is_acceptable:
                console_logger.info(
                    f"Articulated validation passed for '{item.description}': "
                    f"{validation.reason}"
                )
                return self._result_to_articulated_geometry(candidate, item)

            console_logger.info(
                f"Articulated validation failed for '{item.description}': "
                f"{validation.reason}. Suggestions: {validation.suggestions}"
            )

        console_logger.warning(
            f"All {len(candidates)} articulated candidates failed validation "
            f"for '{item.description}'"
        )
        return None

    def _result_to_articulated_geometry(
        self, result: ArticulatedRetrievalResult, item: AssetItem
    ) -> ArticulatedGeometry:
        """Convert a server retrieval result to ArticulatedGeometry.

        Args:
            result: The retrieval result from the server.
            item: The original asset item.

        Returns:
            ArticulatedGeometry with result data.
        """
        return ArticulatedGeometry(
            sdf_path=Path(result.sdf_path),
            item=item,
            source=result.source,
            object_id=result.object_id,
            bounding_box_min=result.bounding_box_min,
            bounding_box_max=result.bounding_box_max,
        )

    def _validate_articulated_result(
        self, result: ArticulatedRetrievalResult, description: str, debug_dir: Path
    ) -> ValidationResult:
        """Validate an articulated result using VLM.

        The server has already exported a combined mesh, so we use that directly
        for validation rendering.

        Args:
            result: The retrieval result from the server.
            description: Original description to validate against.
            debug_dir: Directory to save rendered images.

        Returns:
            ValidationResult with acceptance decision and reasoning.
        """
        # Create validation directory for this result.
        validation_dir = debug_dir / f"articulated_{result.object_id}_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        # The server has already exported the combined mesh.
        mesh_path = Path(result.mesh_path)
        if not mesh_path.exists():
            console_logger.error(f"Combined mesh not found: {mesh_path}")
            return ValidationResult(
                is_acceptable=False,
                reason="Combined mesh file not found",
                suggestions=["Check server mesh export"],
            )

        # Render the mesh for validation.
        try:
            image_paths = self._render_mesh_for_validation(
                mesh_path=mesh_path, output_dir=validation_dir
            )
        except Exception as e:
            console_logger.error(
                f"Failed to render articulated object for validation: {e}"
            )
            return ValidationResult(
                is_acceptable=False,
                reason=f"Rendering failed: {e}",
                suggestions=["Check mesh file validity"],
            )

        if not image_paths:
            console_logger.error(
                f"No images rendered for articulated result {result.object_id}"
            )
            return ValidationResult(
                is_acceptable=False,
                reason="No images rendered",
                suggestions=["Check mesh visual geometry"],
            )

        # Encode images for VLM.
        encoded_images = [encode_image_to_base64(img) for img in image_paths]

        # Select prompt based on lenient validation flag.
        use_lenient = (
            self.cfg.asset_manager.router.strategies.articulated.use_lenient_validation
        )
        if use_lenient:
            prompt_name = AssetRouterPrompts.ASSET_VALIDATION_LENIENT
        else:
            prompt_name = AssetRouterPrompts.ASSET_VALIDATION

        # Build prompt with template variables.
        prompt = prompt_manager.get_prompt(
            prompt_name=prompt_name,
            description=description,
            num_images=len(image_paths),
        )

        # Build message with images.
        user_content = [{"type": "text", "text": prompt}]
        for img_base64 in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                }
            )

        messages = [{"role": "user", "content": user_content}]

        # Call VLM for validation.
        openai_config = self.cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.asset_validation
        verbosity = openai_config.verbosity.asset_validation
        vision_detail = openai_config.vision_detail

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
                f"Articulated validation completed in {elapsed:.1f}s for "
                f"'{description}':\n{response_json}"
            )
        except Exception as e:
            console_logger.error(f"VLM validation failed: {e}")
            return ValidationResult(
                is_acceptable=False,
                reason=f"Validation call failed: {e}",
                suggestions=["Retry validation"],
            )

        # Parse response.
        return ValidationResult(
            is_acceptable=response_json.get("is_acceptable", False),
            reason=response_json.get("reason", "Unknown"),
            suggestions=response_json.get("suggestions", []),
        )

    def _render_mesh_for_validation(
        self, mesh_path: Path, output_dir: Path
    ) -> list[Path]:
        """Render a mesh for VLM validation.

        Args:
            mesh_path: Path to the mesh file (GLB format).
            output_dir: Directory to save rendered images.

        Returns:
            List of paths to rendered images.
        """
        # Use lower light energy for articulated objects (more reflective materials).
        from scenesmith.agent_utils.blender.renderer import ARTICULATED_LIGHT_ENERGY

        # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
        # due to GPU/OpenGL state corruption from fork.
        # Disable coordinate frame for cleaner validation renders.
        if self.blender_server is None or not self.blender_server.is_running():
            raise RuntimeError(
                "BlenderServer required for articulated asset validation. "
                "Forked workers cannot safely use embedded bpy."
            )
        image_paths = self.blender_server.render_multiview_for_analysis(
            mesh_path=mesh_path,
            output_dir=output_dir,
            elevation_degrees=self.side_view_elevation_degrees,
            num_side_views=4,
            include_vertical_views=True,
            light_energy=ARTICULATED_LIGHT_ENERGY,
            show_coordinate_frame=False,
            taa_samples=self.validation_taa_samples,
        )

        return image_paths
