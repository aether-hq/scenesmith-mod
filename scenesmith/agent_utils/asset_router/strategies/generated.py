"""Asset router for LLM-advised asset generation."""

import logging
import time

from pathlib import Path
from typing import TYPE_CHECKING

from scenesmith.agent_utils.asset_router.dataclasses import AssetItem, GeneratedGeometry
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationError,
)
from scenesmith.agent_utils.geometry_generation_server.pipelines.sam_provider import (
    sam_provider_config_from_mapping,
)
from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
)
from scenesmith.agent_utils.objaverse_retrieval.data_loader import (
    convert_objathor_asset_to_glb,
)
from scenesmith.agent_utils.objaverse_retrieval_server.dataclasses import (
    ObjaverseRetrievalServerRequest,
)
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType

if TYPE_CHECKING:
    from scenesmith.agent_utils.assets.image_generation import BaseImageGenerator
    from scenesmith.agent_utils.geometry_generation_server.client import (
        GeometryGenerationClient,
    )
    from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalClient
    from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import (
        HssdRetrievalResult,
    )
    from scenesmith.agent_utils.objaverse_retrieval_server import (
        ObjaverseRetrievalClient,
    )
    from scenesmith.agent_utils.objaverse_retrieval_server.dataclasses import (
        ObjaverseRetrievalResult,
    )

console_logger = logging.getLogger(__name__)


class GeneratedAssetStrategyMixin:
    """Catalog candidate acquisition and generated geometry strategy."""

    def _try_generated_strategy(
        self,
        item: AssetItem,
        max_retries: int,
        geometry_client: "GeometryGenerationClient | None",
        hssd_client: "HssdRetrievalClient | None",
        objaverse_client: "ObjaverseRetrievalClient | None",
        polyhaven_client: "ObjaverseRetrievalClient | None",
        image_generator: "BaseImageGenerator | None",
        images_dir: Path | None,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        scene_id: str | None = None,
        asset_source_override: str | None = None,
        candidate_pool_size: int | None = None,
    ) -> GeneratedGeometry | None:
        """Try the generated strategy with text-to-3D or library retrieval.

        For "generated" strategy, asset_source config (general_asset_source)
        determines whether to use text-to-3D generation or library retrieval
        (HSSD or Objaverse).

        Args:
            item: The asset item to generate.
            max_retries: Number of retries. 0 means single attempt without validation.
            geometry_client: Client for geometry generation server (for text-to-3D).
            hssd_client: Client for HSSD retrieval server (for HSSD retrieval).
            objaverse_client: Client for Objaverse retrieval server (for Objaverse).
            polyhaven_client: Generic catalog client backed by the Poly Haven index.
            image_generator: Image generator for creating reference images (for text-to-3D).
            images_dir: Directory to save generated images (for text-to-3D).
            geometry_dir: Directory to save generated geometry.
            debug_dir: Directory to save debug outputs (validation renders).
            style_context: Optional style context for image generation.
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry if successful, None if all retries exhausted.
        """
        # "all" is a deterministic quality hierarchy. Articulated assets remain
        # a separate router strategy and are attempted before this strategy when
        # the analysis requests them.
        configured_source = self.cfg.asset_manager.general_asset_source
        if configured_source == "all" and asset_source_override is None:
            federated_cfg = self.cfg.asset_manager.get("federated", {})
            source_order = list(
                federated_cfg.get(
                    "source_order", ["polyhaven", "hssd", "objaverse", "generated"]
                )
            )
            catalog_retries = int(federated_cfg.get("catalog_max_retries", 2))
            global_pool_size = int(federated_cfg.get("candidate_pool_size", 12))
            for source in source_order:
                if source not in {"polyhaven", "hssd", "objaverse", "generated"}:
                    console_logger.warning(
                        "Ignoring unknown federated asset source '%s'", source
                    )
                    continue
                source_retries = (
                    max_retries if source == "generated" else max(0, catalog_retries)
                )
                console_logger.info(
                    "Federated search trying %s for '%s'", source, item.description
                )
                result = self._try_generated_strategy(
                    item=item,
                    max_retries=source_retries,
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
                    asset_source_override=source,
                    candidate_pool_size=(
                        global_pool_size if source == "polyhaven" else None
                    ),
                )
                if result is not None:
                    return result
            return None

        # Determine asset source (text-to-3D vs library retrieval).
        asset_source = asset_source_override or configured_source

        # For library retrieval, pre-fetch candidates (single server call).
        hssd_candidates: list | None = None
        objaverse_candidates: list | None = None
        polyhaven_candidates: list | None = None
        if asset_source == "hssd":
            hssd_candidates = self._fetch_hssd_candidates(
                item=item,
                hssd_client=hssd_client,
                geometry_dir=geometry_dir,
                max_retries=max_retries,
                scene_id=scene_id,
            )
            if not hssd_candidates:
                console_logger.warning(f"No HSSD candidates for '{item.description}'")
                return None
        elif asset_source == "objaverse":
            objaverse_candidates = self._fetch_objaverse_candidates(
                item=item,
                objaverse_client=objaverse_client,
                geometry_dir=geometry_dir,
                max_retries=max_retries,
                scene_id=scene_id,
            )
            if not objaverse_candidates:
                console_logger.warning(
                    f"No Objaverse candidates for '{item.description}'"
                )
                return None
        elif asset_source == "polyhaven":
            source_label = (
                "Global asset catalog" if configured_source == "all" else "Poly Haven"
            )
            polyhaven_candidates = self._fetch_objaverse_candidates(
                item=item,
                objaverse_client=polyhaven_client,
                geometry_dir=geometry_dir,
                max_retries=max_retries,
                num_candidates=candidate_pool_size,
                scene_id=scene_id,
                source_label=source_label,
            )
            if not polyhaven_candidates:
                console_logger.warning(
                    f"No {source_label} candidates for '{item.description}'"
                )
                return None

        if max_retries == 0:
            # Single attempt, no validation.
            console_logger.info(
                f"Acquiring '{item.description}' with generated (no validation)"
            )
            return self._acquire_generated_candidate(
                item=item,
                asset_source=asset_source,
                attempt=0,
                geometry_client=geometry_client,
                image_generator=image_generator,
                images_dir=images_dir,
                geometry_dir=geometry_dir,
                debug_dir=debug_dir,
                style_context=style_context,
                scene_id=scene_id,
                hssd_candidates=hssd_candidates,
                objaverse_candidates=objaverse_candidates,
                polyhaven_candidates=polyhaven_candidates,
            )

        # Validation + retry loop.
        for attempt in range(max_retries):
            console_logger.info(
                f"Attempt {attempt + 1}/{max_retries} for '{item.description}'"
            )

            result = self._acquire_generated_candidate(
                item=item,
                asset_source=asset_source,
                attempt=attempt,
                geometry_client=geometry_client,
                image_generator=image_generator,
                images_dir=images_dir,
                geometry_dir=geometry_dir,
                debug_dir=debug_dir,
                style_context=style_context,
                scene_id=scene_id,
                hssd_candidates=hssd_candidates,
                objaverse_candidates=objaverse_candidates,
                polyhaven_candidates=polyhaven_candidates,
            )

            if result is None:
                console_logger.warning(
                    f"Candidate acquisition failed for '{item.description}'"
                )
                continue

            # Validate with VLM.
            # Use lenient validation for retrieved library assets (HSSD, Objaverse) based
            # on config. Generated assets always use strict validation.
            use_lenient = False
            if asset_source == "hssd":
                use_lenient = self.cfg.asset_manager.hssd.use_lenient_validation
            elif asset_source == "objaverse":
                use_lenient = self.cfg.asset_manager.objaverse.use_lenient_validation
            elif asset_source == "polyhaven":
                use_lenient = self.cfg.asset_manager.polyhaven.use_lenient_validation

            validation_dir = debug_dir / f"{item.short_name}_validation"
            validation = self.validate_asset(
                mesh_path=result.geometry_path,
                description=item.description,
                output_dir=validation_dir,
                use_lenient=use_lenient,
                ignore_appearance=(
                    asset_source == "generated"
                    and self.cfg.asset_manager.backend == "sam3d"
                    and str(self.cfg.asset_manager.sam3d.provider).lower() == "mlx"
                ),
            )

            if validation.is_acceptable:
                console_logger.info(
                    f"Validation passed for '{item.description}': {validation.reason}"
                )
                return result

            console_logger.info(
                f"Validation failed for '{item.description}': {validation.reason}. "
                f"Suggestions: {validation.suggestions}"
            )

        return None

    def _fetch_hssd_candidates(
        self,
        item: AssetItem,
        hssd_client: "HssdRetrievalClient | None",
        geometry_dir: Path,
        max_retries: int,
        scene_id: str | None = None,
    ) -> list["HssdRetrievalResult"] | None:
        """Fetch HSSD candidates in a single server call.

        Args:
            item: The asset item to retrieve candidates for.
            hssd_client: Client for HSSD retrieval server.
            geometry_dir: Directory to save retrieved geometry.
            max_retries: Number of validation retries (determines num_candidates).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            List of HssdRetrievalResult candidates, or None if fetch failed.
        """
        if hssd_client is None:
            console_logger.error(
                f"HSSD client not provided for '{item.description}', "
                "but asset_source is 'hssd'"
            )
            return None

        console_logger.info(f"Fetching HSSD candidates for '{item.description}'")

        # Request enough candidates for all retry attempts.
        # max_retries=0 means single attempt, so we need at least 1.
        num_candidates = max(1, max_retries)

        # Map EITHER to concrete type based on which agent is calling.
        object_type = item.object_type.value
        if object_type == "either":
            object_type = (
                "furniture" if self.agent_type == AgentType.FURNITURE else "manipuland"
            )
            console_logger.debug(
                f"Mapped 'either' to '{object_type}' for {self.agent_type} agent"
            )

        request = HssdRetrievalServerRequest(
            object_description=item.description,
            object_type=object_type,
            desired_dimensions=tuple(item.dimensions) if item.dimensions else None,
            output_dir=str(geometry_dir),
            scene_id=scene_id,
            num_candidates=num_candidates,
        )

        try:
            responses = list(hssd_client.retrieve_objects([request]))
            if not responses:
                console_logger.error(f"No HSSD response for '{item.description}'")
                return None

            _, response = responses[0]

            if not response.results:
                console_logger.error(f"No HSSD results for '{item.description}'")
                return None

            console_logger.info(
                f"Got {len(response.results)} HSSD candidates for '{item.description}'"
            )
            return response.results

        except Exception as e:
            console_logger.error(f"HSSD fetch failed for '{item.description}': {e}")
            return None

    def _fetch_objaverse_candidates(
        self,
        item: AssetItem,
        objaverse_client: "ObjaverseRetrievalClient | None",
        geometry_dir: Path,
        max_retries: int,
        num_candidates: int | None = None,
        scene_id: str | None = None,
        source_label: str = "Objaverse",
    ) -> list["ObjaverseRetrievalResult"] | None:
        """Fetch Objaverse candidates in a single server call.

        Args:
            item: The asset item to retrieve candidates for.
            objaverse_client: Client for a generic catalog retrieval server.
            geometry_dir: Directory to save retrieved geometry.
            max_retries: Number of validation retries (determines num_candidates).
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            List of ObjaverseRetrievalResult candidates, or None if fetch failed.
        """
        if objaverse_client is None:
            console_logger.error(
                f"{source_label} client not provided for '{item.description}'"
            )
            return None

        console_logger.info(
            f"Fetching {source_label} candidates for '{item.description}'"
        )

        # Request enough candidates for all retry attempts.
        # max_retries=0 means single attempt, so we need at least 1.
        num_candidates = max(1, num_candidates or max_retries)

        # Map EITHER to concrete type based on which agent is calling.
        object_type = item.object_type.value
        if object_type == "either":
            object_type = (
                "furniture" if self.agent_type == AgentType.FURNITURE else "manipuland"
            )
            console_logger.debug(
                f"Mapped 'either' to '{object_type}' for {self.agent_type} agent"
            )

        request = ObjaverseRetrievalServerRequest(
            object_description=item.description,
            object_type=object_type,
            desired_dimensions=tuple(item.dimensions) if item.dimensions else None,
            output_dir=str(geometry_dir),
            scene_id=scene_id,
            num_candidates=num_candidates,
        )

        try:
            responses = list(objaverse_client.retrieve_objects([request]))
            if not responses:
                console_logger.error(
                    f"No {source_label} response for '{item.description}'"
                )
                return None

            _, response = responses[0]

            if not response.results:
                console_logger.error(
                    f"No {source_label} results for '{item.description}'"
                )
                return None

            console_logger.info(
                f"Got {len(response.results)} {source_label} candidates for "
                f"'{item.description}'"
            )
            return response.results

        except Exception as e:
            console_logger.error(
                f"{source_label} fetch failed for '{item.description}': {e}"
            )
            return None

    def _acquire_generated_candidate(
        self,
        item: AssetItem,
        asset_source: str,
        attempt: int,
        geometry_client: "GeometryGenerationClient | None",
        image_generator: "BaseImageGenerator | None",
        images_dir: Path | None,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        scene_id: str | None = None,
        hssd_candidates: list["HssdRetrievalResult"] | None = None,
        objaverse_candidates: list["ObjaverseRetrievalResult"] | None = None,
        polyhaven_candidates: list["ObjaverseRetrievalResult"] | None = None,
    ) -> GeneratedGeometry | None:
        """Acquire a single candidate based on asset source.

        For text-to-3D: generates a new mesh (attempt number ignored, randomness varies).
        For HSSD/Objaverse: returns candidate at index `attempt` from pre-fetched list.

        Args:
            item: The asset item to acquire.
            asset_source: "generated", "hssd", or "objaverse".
            attempt: Attempt number (used as index for library candidates).
            geometry_client: Client for geometry generation server.
            image_generator: Image generator for reference images.
            images_dir: Directory for generated images.
            geometry_dir: Directory for geometry files.
            debug_dir: Directory for debug outputs.
            style_context: Style context for image generation.
            scene_id: Scene identifier for scheduling.
            hssd_candidates: Pre-fetched HSSD candidates (required if asset_source="hssd").
            objaverse_candidates: Pre-fetched Objaverse candidates (if asset_source="objaverse").

        Returns:
            GeneratedGeometry if successful, None if failed or no more candidates.
        """
        if asset_source == "hssd":
            if hssd_candidates is None or attempt >= len(hssd_candidates):
                console_logger.warning(
                    f"No more HSSD candidates for '{item.description}' "
                    f"(attempt {attempt}, available {len(hssd_candidates or [])})"
                )
                return None

            candidate = hssd_candidates[attempt]
            console_logger.info(
                f"Using HSSD candidate {attempt + 1}/{len(hssd_candidates)} "
                f"for '{item.description}': {candidate.hssd_id}"
            )

            return GeneratedGeometry(
                geometry_path=Path(candidate.mesh_path),
                item=item,
                asset_source="hssd",
                hssd_id=candidate.hssd_id,
            )

        if asset_source in {"objaverse", "polyhaven"}:
            candidates = (
                polyhaven_candidates
                if asset_source == "polyhaven"
                else objaverse_candidates
            )
            source_label = (
                "Global asset catalog"
                if asset_source == "polyhaven"
                and self.cfg.asset_manager.general_asset_source == "all"
                else "Poly Haven" if asset_source == "polyhaven" else "Objaverse"
            )
            if candidates is None or attempt >= len(candidates):
                console_logger.warning(
                    f"No more {source_label} candidates for '{item.description}' "
                    f"(attempt {attempt}, available {len(candidates or [])})"
                )
                return None

            candidate = candidates[attempt]
            console_logger.info(
                f"Using {source_label} candidate {attempt + 1}/{len(candidates)} "
                f"for '{item.description}': {candidate.objaverse_uid}"
            )

            geometry_path = Path(candidate.mesh_path)
            if not geometry_path.exists() and candidate.asset_source == "objaverse":
                geometry_path = convert_objathor_asset_to_glb(
                    asset_dir=geometry_path.parent,
                    uid=candidate.source_id or candidate.objaverse_uid,
                )

            return GeneratedGeometry(
                geometry_path=geometry_path,
                item=item,
                asset_source=candidate.asset_source,
                objaverse_uid=(
                    candidate.objaverse_uid if asset_source == "objaverse" else None
                ),
                catalog_id=candidate.objaverse_uid,
                license=candidate.license,
                ontology_path=candidate.ontology_path,
                placement_classes=candidate.placement_classes,
                canonical_up=candidate.canonical_up,
                canonical_front=candidate.canonical_front,
                support_zones=candidate.support_zones,
                clearance_zones=candidate.clearance_zones,
                quality_score=candidate.quality_score,
                thumbnail=candidate.thumbnail,
                catalog_semantics=candidate.semantic_metadata,
            )

        # Text-to-3D generation.
        return self._generate_geometry(
            item=item,
            geometry_client=geometry_client,
            image_generator=image_generator,
            images_dir=images_dir,
            geometry_dir=geometry_dir,
            debug_dir=debug_dir,
            style_context=style_context,
            scene_id=scene_id,
        )

    def _generate_geometry(
        self,
        item: AssetItem,
        geometry_client: "GeometryGenerationClient",
        image_generator: "BaseImageGenerator",
        images_dir: Path,
        geometry_dir: Path,
        debug_dir: Path,
        style_context: str | None = None,
        scene_id: str | None = None,
    ) -> GeneratedGeometry | None:
        """Generate geometry for a single item.

        Args:
            item: The asset item to generate.
            geometry_client: Client for geometry generation server.
            image_generator: Image generator for creating reference images.
            images_dir: Directory to save generated images.
            geometry_dir: Directory to save generated geometry.
            debug_dir: Directory to save debug outputs (segmentation masks, etc.).
            style_context: Optional style context for image generation.
            scene_id: Optional scene identifier for fair round-robin scheduling.

        Returns:
            GeneratedGeometry with paths, or None if generation failed.
        """
        from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
            GeometryGenerationServerRequest,
        )

        # Generate unique filename with timestamp.
        timestamp = int(time.time())
        base_name = f"{item.short_name}_{timestamp}"
        image_path = images_dir / f"{base_name}.png"
        geometry_path = geometry_dir / f"{base_name}.glb"

        # Generate reference image.
        try:
            style_prompt = style_context or "Modern style"
            image_generator.generate_images(
                style_prompt=style_prompt,
                object_descriptions=[item.description],
                output_paths=[image_path],
            )
        except Exception as e:
            console_logger.error(
                f"Image generation failed for '{item.description}': {e}"
            )
            return None

        # Generate geometry from image.
        try:
            # Extract backend configuration from config.
            backend = self.cfg.asset_manager.get("backend", "hunyuan3d")

            # Prepare SAM3D config if backend is sam3d.
            sam3d_config = None
            if backend == "sam3d":
                sam3d_cfg = self.cfg.asset_manager.sam3d
                mode = sam3d_cfg.get("mode", "foreground")
                sam3d_config = sam_provider_config_from_mapping(
                    sam3d_cfg,
                    object_description=(
                        item.description if mode == "object_description" else None
                    ),
                )

            request = GeometryGenerationServerRequest(
                image_path=str(image_path),
                output_dir=str(geometry_dir),
                prompt=item.description,
                output_filename=geometry_path.name,
                debug_folder=str(debug_dir),
                backend=backend,
                sam3d_config=sam3d_config,
                scene_id=scene_id,
            )

            # Use synchronous single-item generation.
            responses = list(geometry_client.generate_geometries([request]))
            if not responses:
                console_logger.error(f"No geometry response for '{item.description}'")
                return None

            _, response = responses[0]

            # Check for generation error.
            if isinstance(response, GeometryGenerationError):
                console_logger.error(
                    f"Geometry generation error for '{item.description}': "
                    f"{response.error_message}"
                )
                return None

            actual_geometry_path = Path(response.geometry_path)

        except Exception as e:
            console_logger.error(
                f"Geometry generation failed for '{item.description}': {e}"
            )
            return None

        return GeneratedGeometry(
            geometry_path=actual_geometry_path,
            item=item,
            asset_source="generated",
            image_path=image_path,
        )
