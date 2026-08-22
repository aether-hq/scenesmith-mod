"""Asset router for LLM-advised asset generation."""

import json
import logging
import tempfile
import time

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalClient,
)
from scenesmith.agent_utils.asset_router.dataclasses import (
    AnalysisResult,
    ArticulatedGeometry,
    AssetItem,
    GeneratedGeometry,
    ValidationResult,
)
from scenesmith.agent_utils.asset_router.strategies.articulated import (
    ArticulatedAssetStrategyMixin,
)
from scenesmith.agent_utils.asset_router.strategies.generated import (
    GeneratedAssetStrategyMixin,
)
from scenesmith.agent_utils.asset_router.strategies.thin_covering import (
    ThinCoveringAssetStrategyMixin,
)
from scenesmith.agent_utils.llm.vlm_service import VLMService
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType, ObjectType
from scenesmith.prompts import AssetRouterPrompts, prompt_manager
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.assets.image_generation import BaseImageGenerator
    from scenesmith.agent_utils.blender import BlenderServer
    from scenesmith.agent_utils.geometry_generation_server.client import (
        GeometryGenerationClient,
    )
    from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalClient
    from scenesmith.agent_utils.materials_retrieval_server import (
        MaterialsRetrievalClient,
    )
    from scenesmith.agent_utils.objaverse_retrieval_server import (
        ObjaverseRetrievalClient,
    )

console_logger = logging.getLogger(__name__)


class AssetRouter(
    ArticulatedAssetStrategyMixin,
    ThinCoveringAssetStrategyMixin,
    GeneratedAssetStrategyMixin,
):
    """Routes asset generation requests through LLM analysis and validation.

    This implements a hybrid "LLM-Advised Deterministic Loop" architecture:
    - LLM handles semantic decisions (routing, composite detection, validation)
    - Deterministic Python handles execution (retry logic, fallback chains)
    """

    def __init__(
        self,
        agent_type: AgentType,
        vlm_service: VLMService,
        cfg: DictConfig,
        blender_server: "BlenderServer | None" = None,
    ) -> None:
        """Initialize the asset router.

        Args:
            agent_type: Type of placement agent.
            vlm_service: VLM service for analysis and validation.
            cfg: Configuration with OpenAI and router settings.
            blender_server: Optional BlenderServer for thread-safe validation rendering.
                When provided and running, validation uses HTTP requests to the server
                instead of direct BlenderRenderer calls. This is required for parallel
                generation since bpy (Blender Python API) requires main thread execution.

        Raises:
            ValueError: If agent_type is not a placement agent.
        """
        if not agent_type.is_placement_agent:
            raise ValueError(
                f"AssetRouter requires a placement agent, got {agent_type.value}"
            )
        self.agent_type = agent_type
        self.vlm_service = vlm_service
        self.cfg = cfg
        self.blender_server = blender_server
        self.side_view_elevation_degrees = cfg.asset_manager.side_view_elevation_degrees
        self.validation_taa_samples = cfg.asset_manager.validation_taa_samples

    def analyze_request(
        self, description: str, dimensions: list[float]
    ) -> AnalysisResult:
        """Analyze an asset request using VLM.

        Calls the appropriate analysis prompt (furniture or manipuland) to:
        - Extract valid items for this agent type
        - Filter out items for other agents (e.g., furniture agent discards manipulands)
        - Split composite requests into individual items
        - Select appropriate generation strategies

        Retries if parsing fails for all items (e.g., invalid object_type values).

        Args:
            description: Object description from the designer.
            dimensions: Desired dimensions [width, depth, height] in meters.

        Returns:
            AnalysisResult with extracted items and any modifications.
        """
        # Select prompt based on agent type.
        if self.agent_type == AgentType.FURNITURE:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_FURNITURE
        elif self.agent_type == AgentType.WALL_MOUNTED:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_WALL
        elif self.agent_type == AgentType.CEILING_MOUNTED:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_CEILING
        else:
            prompt_enum = AssetRouterPrompts.REQUEST_ANALYSIS_MANIPULAND

        # Render prompt with template variables.
        prompt = prompt_manager.get_prompt(
            prompt_name=prompt_enum, description=description, dimensions=dimensions
        )

        # Call VLM for analysis.
        messages = [{"role": "user", "content": prompt}]

        openai_config = self.cfg.openai
        model = openai_config.model
        reasoning_effort = openai_config.reasoning_effort.asset_analysis
        verbosity = openai_config.verbosity.asset_analysis

        # Retry loop for parsing failures (e.g., LLM returns invalid object_type).
        max_retries = self.cfg.asset_manager.router.analysis_max_retries
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response_text = self.vlm_service.create_completion(
                    model=model,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    response_format={"type": "json_object"},
                )
                elapsed = time.time() - start_time
                response_json = json.loads(response_text)
                console_logger.info(
                    f"Router analysis completed in {elapsed:.1f}s:\n{response_json}"
                )
            except Exception as e:
                console_logger.error(f"VLM analysis failed: {e}")
                return AnalysisResult(
                    items=[],
                    original_description=None,
                    discarded_manipulands=None,
                    error=f"Analysis failed: {e}",
                )

            # Parse response into AnalysisResult.
            result = self._parse_analysis_response(response_json)

            # Check if parsing succeeded or if there's nothing to parse.
            raw_items = response_json.get("items", [])
            if result.items or not raw_items or result.error:
                # Success: got valid items, no items to parse, or explicit error.
                return result

            # All items failed to parse - retry VLM call.
            console_logger.warning(
                f"All {len(raw_items)} items failed to parse "
                f"(attempt {attempt + 1}/{max_retries}), retrying VLM call..."
            )

        # All retries exhausted.
        return AnalysisResult(
            items=[],
            original_description=description,
            discarded_manipulands=None,
            error=f"Failed to parse valid items after {max_retries} attempts",
        )

    def _parse_analysis_response(self, response: dict) -> AnalysisResult:
        """Parse VLM analysis response into AnalysisResult.

        Args:
            response: JSON response from VLM.

        Returns:
            Parsed AnalysisResult.
        """
        # Check for error in response.
        if "error" in response and response["error"]:
            return AnalysisResult(
                items=[],
                original_description=response.get("original_description"),
                discarded_manipulands=response.get("discarded_manipulands"),
                error=response["error"],
            )

        # Parse items.
        items = []
        for item_data in response.get("items", []):
            try:
                object_type = ObjectType(item_data["object_type"].lower())
                items.append(
                    AssetItem(
                        description=item_data["description"],
                        short_name=item_data["short_name"],
                        dimensions=item_data["dimensions"],
                        object_type=object_type,
                        strategies=item_data["strategies"],
                        thin_covering_type=item_data.get("thin_covering_type"),
                    )
                )
            except (KeyError, ValueError) as e:
                console_logger.warning(f"Failed to parse item: {item_data}, error: {e}")
                continue

        return AnalysisResult(
            items=items,
            original_description=response.get("original_description"),
            discarded_manipulands=response.get("discarded_manipulands"),
            error=None,
        )

    def validate_asset(
        self,
        mesh_path: Path,
        description: str,
        output_dir: Path | None = None,
        use_lenient: bool = False,
        ignore_appearance: bool = False,
    ) -> ValidationResult:
        """Validate a generated asset using VLM.

        Renders multi-view images of the mesh and asks VLM to verify:
        - Correct object type (matches description)
        - Style matches (if specified in description)
        - Single object (not multiple objects)
        - Completeness (no missing parts)
        - Reasonable proportions

        Args:
            mesh_path: Path to the generated mesh file.
            description: Original description to validate against.
            output_dir: Optional directory to save rendered images.
            use_lenient: If True, use lenient validation prompt. Lenient validation
                accepts minor imperfections common in library assets.
            ignore_appearance: Judge geometry only. This is required for providers
                such as SAM3D MLX that intentionally omit color baking.

        Returns:
            ValidationResult with acceptance decision and reasoning.
        """
        # Determine output directory for rendered images.
        if output_dir is not None:
            render_dir = output_dir
            render_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp(prefix="asset_validation_")
            render_dir = Path(temp_dir)

        # Render multi-view images via BlenderServer.
        # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
        # due to GPU/OpenGL state corruption from fork.
        # Disable coordinate frame for cleaner validation renders.
        try:
            if self.blender_server is None or not self.blender_server.is_running():
                raise RuntimeError(
                    "BlenderServer required for asset validation. "
                    "Forked workers cannot safely use embedded bpy."
                )
            image_paths = self.blender_server.render_multiview_for_analysis(
                mesh_path=mesh_path,
                output_dir=render_dir,
                elevation_degrees=self.side_view_elevation_degrees,
                num_side_views=4,
                include_vertical_views=True,
                show_coordinate_frame=False,
                taa_samples=self.validation_taa_samples,
            )
            console_logger.debug(f"Rendered {len(image_paths)} images for validation")
        except Exception as e:
            console_logger.error(f"Failed to render mesh for validation: {e}")
            return ValidationResult(
                is_acceptable=False,
                reason=f"Rendering failed: {e}",
                suggestions=["Check mesh file validity"],
            )

        # Encode images for VLM.
        encoded_images = [encode_image_to_base64(img) for img in image_paths]

        # Build prompt with template variables.
        prompt_name = (
            AssetRouterPrompts.ASSET_VALIDATION_LENIENT
            if use_lenient
            else AssetRouterPrompts.ASSET_VALIDATION
        )
        prompt = prompt_manager.get_prompt(
            prompt_name=prompt_name,
            description=description,
            num_images=len(image_paths),
        )
        if ignore_appearance:
            prompt += (
                "\n\nPROVIDER LIMITATION: This geometry backend intentionally does "
                "not bake colors, materials, or textures, so the mesh may render "
                "uniform white or gray. Ignore requested color, finish, material, "
                "texture, and appearance-only style cues. Do not reject for those "
                "reasons. Validate only object identity, required physical parts, "
                "single-object composition, completeness, and proportions."
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
                f"Router validation completed in {elapsed:.1f}s for "
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

    def validate_item_types(self, items: list[AssetItem]) -> str | None:
        """Validate that all items match this agent's type.

        This is a safety check for LLM errors - the analysis prompt should
        already filter correctly, but this catches unexpected responses.

        Args:
            items: List of items from analysis.

        Returns:
            Error message if validation fails, None if all items are valid.
        """
        for item in items:
            # EITHER type is allowed in both agents.
            if item.object_type == ObjectType.EITHER:
                continue

            # Check if item type matches agent type.
            if item.object_type != self.agent_type.to_object_type():
                console_logger.warning(
                    f"LLM analysis returned wrong type: {item.object_type} "
                    f"for {self.agent_type.value} agent"
                )
                return (
                    f"Item '{item.description}' has type {item.object_type.value}, "
                    f"but this is the {self.agent_type} agent."
                )

        return None

    def generate_with_validation(
        self,
        item: AssetItem,
        geometry_client: "GeometryGenerationClient | None",
        image_generator: "BaseImageGenerator | None",
        images_dir: Path | None,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        hssd_client: "HssdRetrievalClient | None" = None,
        objaverse_client: "ObjaverseRetrievalClient | None" = None,
        polyhaven_client: "ObjaverseRetrievalClient | None" = None,
        articulated_client: "ArticulatedRetrievalClient | None" = None,
        materials_client: "MaterialsRetrievalClient | None" = None,
        scene_id: str | None = None,
    ) -> GeneratedGeometry | ArticulatedGeometry | None:
        """Generate or retrieve geometry for item with validation and retry.

        For generated assets: Tries each strategy in item.strategies. For each
        strategy, validation is controlled by the config's max_retries.

        For HSSD/Objaverse assets: Retrieves top-k candidates and validates each
        until one passes or all fail.

        Args:
            item: The asset item to generate/retrieve.
            geometry_client: Client for geometry generation server (for generated).
            image_generator: Image generator for creating reference images (for generated).
            images_dir: Directory to save generated images (for generated).
            geometry_dir: Directory to save generated/retrieved geometry.
            debug_dir: Directory to save debug outputs (validation renders).
            style_context: Optional style context for image generation.
            hssd_client: Client for HSSD retrieval server (for HSSD).
            objaverse_client: Client for Objaverse retrieval server (for Objaverse).
            polyhaven_client: Generic catalog client backed by the Poly Haven index.
            articulated_client: Client for articulated retrieval server.
            materials_client: Client for materials retrieval server (for thin coverings).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry if successful, None if all strategies/candidates exhausted.
        """
        for strategy in item.strategies:
            # Get strategy config.
            strategies_cfg = self.cfg.asset_manager.router.strategies
            if not hasattr(strategies_cfg, strategy):
                console_logger.warning(f"Strategy '{strategy}' not in config, skipping")
                continue

            strategy_cfg = getattr(strategies_cfg, strategy)
            if hasattr(strategy_cfg, "enabled") and not strategy_cfg.enabled:
                console_logger.warning(f"Strategy '{strategy}' disabled, skipping")
                continue

            max_retries = strategy_cfg.max_retries
            console_logger.info(
                f"Trying strategy '{strategy}' for '{item.description}' "
                f"(max_retries={max_retries})"
            )

            # Dispatch to strategy-specific helper.
            if strategy == "generated":
                result = self._try_generated_strategy(
                    item=item,
                    max_retries=max_retries,
                    geometry_client=geometry_client,
                    hssd_client=hssd_client,
                    objaverse_client=objaverse_client,
                    polyhaven_client=polyhaven_client,
                    image_generator=image_generator,
                    images_dir=images_dir,
                    geometry_dir=geometry_dir,
                    debug_dir=debug_dir,
                    style_context=style_context,
                    scene_id=scene_id,
                )
            elif strategy == "articulated":
                result = self._try_articulated_strategy(
                    item=item,
                    max_retries=max_retries,
                    debug_dir=debug_dir,
                    articulated_client=articulated_client,
                )
            elif strategy == "thin_covering":
                # Thin covering: textured flat surface for floors (e.g, rugs),
                # manipulands (e.g., tablecloths), and walls (e.g., posters).
                # Strategy auto-detects orientation based on agent_type.
                result = self._try_thin_covering_strategy(
                    item=item,
                    max_retries=max_retries,
                    materials_client=materials_client,
                    image_generator=image_generator,
                    geometry_dir=geometry_dir,
                    debug_dir=debug_dir,
                    scene_id=scene_id,
                )
            else:
                console_logger.warning(
                    f"Unknown strategy '{strategy}' for '{item.description}'"
                )
                continue

            if result is not None:
                return result

        console_logger.warning(f"All strategies exhausted for '{item.description}'")
        return None
