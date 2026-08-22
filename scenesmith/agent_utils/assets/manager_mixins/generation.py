import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import TYPE_CHECKING

from scenesmith.agent_utils.asset_router.dataclasses import (
    ArticulatedGeometry,
    AssetItem,
    ModificationInfo,
)
from scenesmith.agent_utils.assets.asset_semantics import (
    catalog_candidate_is_compatible,
    catalog_candidate_satisfies_request_details,
    is_structural_architecture_request,
    tall_furniture_dimensions_are_compatible,
)
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, SceneObject

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

HSSD_CANONICAL_CONVERSION_VERSION = 4

from scenesmith.agent_utils.assets.asset_models import (
    AssetGenerationRequest,
    AssetGenerationResult,
    FailedAsset,
    _subscription_aware_worker_count,
)


class AssetGenerationMixin:
    """Model/router generation orchestration and cache quarantine."""

    def _generate_assets_with_model(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate assets using text-to-3D model (Hunyuan3D).

        This method handles the complete generation pipeline:
        - Style change detection and registry reset
        - Request validation (descriptions vs short names, dimensions)
        - Duplicate detection and deduplication
        - Asset path creation
        - Image generation via VLM
        - Mesh generation via geometry server
        - Asset processing and conversion

        Args:
            request: Asset generation request with descriptions and parameters.

        Returns:
            AssetGenerationResult with generated scene objects and metadata.
        """
        # Validate request.
        if len(request.object_descriptions) != len(request.short_names):
            raise ValueError(
                f"Mismatch between descriptions ({len(request.object_descriptions)}) "
                f"and short names ({len(request.short_names)})"
            )

        # Validate desired_dimensions.
        if len(request.desired_dimensions) != len(request.object_descriptions):
            raise ValueError(
                f"Mismatch between desired_dimensions ({len(request.desired_dimensions)}) "
                f"and object_descriptions ({len(request.object_descriptions)})"
            )

        # Detect duplicates based on (description, desired_dimensions).
        unique_items: dict[tuple[str, tuple[float, ...]], int] = {}
        duplicate_indices: dict[str, list[int]] = {}

        for i, (desc, dims) in enumerate(
            zip(request.object_descriptions, request.desired_dimensions)
        ):
            key = (desc, tuple(dims))
            if key in unique_items:
                # This is a duplicate.
                original_idx = unique_items[key]
                if desc not in duplicate_indices:
                    duplicate_indices[desc] = []
                duplicate_indices[desc].append(i)
                console_logger.warning(
                    f"Duplicate detected at index {i}: '{desc}' with dimensions "
                    f"{dims} (same as index {original_idx})"
                )
            else:
                # This is unique.
                unique_items[key] = i

        # Store duplicate info for tool feedback.
        self.last_duplicate_info = duplicate_indices if duplicate_indices else None

        # Log summary if duplicates found.
        if duplicate_indices:
            total_duplicates = sum(
                len(indices) for indices in duplicate_indices.values()
            )
            console_logger.warning(
                f"Found {total_duplicates} duplicate request(s) across "
                f"{len(duplicate_indices)} description(s). Generating only unique items."
            )

        # Build unique request lists.
        unique_indices = sorted(unique_items.values())
        unique_descriptions = [request.object_descriptions[i] for i in unique_indices]
        unique_short_names = [request.short_names[i] for i in unique_indices]
        unique_dimensions = [request.desired_dimensions[i] for i in unique_indices]

        # Create reduced request with only unique items.
        unique_request = AssetGenerationRequest(
            object_descriptions=unique_descriptions,
            short_names=unique_short_names,
            object_type=request.object_type,
            desired_dimensions=unique_dimensions,
            style_context=request.style_context,
            operation_type=request.operation_type,
            scene_id=request.scene_id,
        )

        # Create asset path configurations.
        asset_paths_configs = self._create_asset_paths(
            object_descriptions=unique_request.object_descriptions,
            short_names=unique_request.short_names,
        )

        # Generate images for all assets.
        self._generate_images(
            request=unique_request, asset_paths_configs=asset_paths_configs
        )

        # Convert images to 3D assets and create SceneObjects.
        successful_objects, failed_assets = self._process_assets_to_scene_objects(
            request=unique_request, asset_path_configs=asset_paths_configs
        )

        console_logger.info(
            f"Asset generation completed: {len(successful_objects)} unique objects "
            f"created, {len(failed_assets)} failed"
        )
        return AssetGenerationResult(
            successful_assets=successful_objects, failed_assets=failed_assets
        )

    def generate_assets(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        """Generate scene assets using configured source (generated or hssd).

        If router is enabled, analyzes requests to split composites and filter
        items before dispatching to the configured asset source.

        Args:
            request: Asset generation request with descriptions and context.

        Returns:
            AssetGenerationResult with successful assets and failure information.
        """
        console_logger.info(
            f"Starting {request.object_type.value} asset acquisition for "
            f"{len(request.object_descriptions)} items using "
            f"'{self.general_asset_source}' source. Router is "
            f"{'enabled' if self.router is not None else 'disabled'}."
        )

        # Traversable architecture is authored and compiled by the floor-plan
        # stage.  Treating it as furniture creates duplicate stairs and allows a
        # semantically unrelated catalog mesh to masquerade as structure.
        if request.object_type == ObjectType.FURNITURE:
            accepted_indices = [
                index
                for index, (description, short_name) in enumerate(
                    zip(request.object_descriptions, request.short_names)
                )
                if not is_structural_architecture_request(f"{short_name} {description}")
            ]
            if len(accepted_indices) != len(request.object_descriptions):
                rejected = [
                    request.object_descriptions[index]
                    for index in range(len(request.object_descriptions))
                    if index not in accepted_indices
                ]
                console_logger.warning(
                    "Ignored structural architecture requested through the furniture "
                    "channel (already owned by the floor plan): %s",
                    rejected,
                )
                if not accepted_indices:
                    return AssetGenerationResult(
                        successful_assets=[],
                        failed_assets=[],
                        modification_info=None,
                    )
                request = replace(
                    request,
                    object_descriptions=[
                        request.object_descriptions[index] for index in accepted_indices
                    ],
                    short_names=[
                        request.short_names[index] for index in accepted_indices
                    ],
                    desired_dimensions=[
                        request.desired_dimensions[index] for index in accepted_indices
                    ],
                )

        # If router is enabled, analyze and potentially modify the request.
        if self.router is not None:
            return self._generate_assets_with_router(request)

        # Dispatch based on asset source (router disabled).
        if self.general_asset_source == "hssd":
            return self._retrieve_hssd_assets(request)
        elif self.general_asset_source == "objaverse":
            return self._retrieve_objaverse_assets(request)
        elif self.general_asset_source == "polyhaven":
            return self._retrieve_objaverse_assets(
                request,
                client=self.polyhaven_client,
                source_label="polyhaven",
            )
        elif self.general_asset_source == "all":
            raise ValueError("Asset source 'all' requires asset_manager.router.enabled")
        elif self.general_asset_source == "generated":
            return self._generate_assets_with_model(request)
        else:
            # This should never happen due to __init__ validation.
            raise ValueError(f"Unknown asset source: {self.general_asset_source}")

    def _quarantine_incompatible_cached_assets(self) -> None:
        """Remove stale catalog aliases before they can be placed in a new scene."""

        for asset in list(self.registry.list_all()):
            request_text = f"{asset.name} {asset.description}"
            metadata = asset.metadata or {}
            catalog_text = str(
                metadata.get("catalog_semantics") or metadata.get("ontology_path") or ""
            )
            source = str(metadata.get("asset_source", "")).casefold()
            reason = ""
            compatible = True
            if is_structural_architecture_request(request_text):
                compatible = False
                reason = "architectural structure cannot be cached as furniture"
            elif (
                source == "hssd"
                and metadata.get("canonical_conversion_version")
                != HSSD_CANONICAL_CONVERSION_VERSION
            ):
                compatible = False
                reason = (
                    "stale HSSD canonical conversion version "
                    f"{metadata.get('canonical_conversion_version')!r}; expected "
                    f"{HSSD_CANONICAL_CONVERSION_VERSION}"
                )
            elif source in {"hssd", "objaverse", "polyhaven"}:
                compatible, reason = catalog_candidate_is_compatible(
                    request_text=request_text,
                    candidate_text=catalog_text,
                    quality_score=(
                        float(metadata["asset_quality_score"])
                        if metadata.get("asset_quality_score") is not None
                        else None
                    ),
                    minimum_quality=0.70,
                )
                if compatible:
                    compatible, reason = tall_furniture_dimensions_are_compatible(
                        request_text=request_text,
                        desired_dimensions=None,
                        bbox_min=getattr(asset, "bbox_min", None),
                        bbox_max=getattr(asset, "bbox_max", None),
                    )
                if compatible:
                    compatible, reason = catalog_candidate_satisfies_request_details(
                        request_text=request_text,
                        candidate_text=catalog_text,
                        supports_detail_fill=bool(metadata.get("support_zones")),
                    )
            if compatible:
                continue
            console_logger.warning(
                "Rejecting cached asset %s for '%s': %s",
                asset.object_id,
                request_text,
                reason,
            )
            self.registry.discard(asset.object_id)

    @staticmethod
    def _asset_conversion_metadata(asset_source: str) -> dict[str, int]:
        """Return source-specific geometry-contract metadata for cache reuse."""

        if asset_source.casefold() == "hssd":
            return {"canonical_conversion_version": HSSD_CANONICAL_CONVERSION_VERSION}
        return {}

    def _generate_assets_with_router(
        self, request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate assets using router for LLM-advised analysis and validation.

        Two-phase processing for thread safety:

        **Phase 1 - Parallel (thread-safe HTTP calls):**
        1. Validate request and check for style changes
        2. Deduplicate by (description, dimensions) to save LLM calls
        3. LLM analysis per unique item (split composites, select strategies)
        4. Parallel generation/retrieval via geometry or HSSD server
        5. VLM validation with retry loop (configured max_retries per strategy)

        **Phase 2 - Sequential (main thread, uses bpy):**
        6. GLB→GLTF conversion, floater removal, mesh canonicalization
        7. CoACD collision geometry, SDF generation
        8. Build SceneObjects and modification_info

        Args:
            request: Asset generation request.

        Returns:
            AssetGenerationResult with modification_info if request was modified.
        """
        # Validate request lengths.
        if len(request.object_descriptions) != len(request.short_names):
            raise ValueError(
                f"Mismatch between descriptions ({len(request.object_descriptions)}) "
                f"and short names ({len(request.short_names)})"
            )

        if len(request.desired_dimensions) != len(request.object_descriptions):
            raise ValueError(
                f"Mismatch between desired_dimensions ({len(request.desired_dimensions)}) "
                f"and object_descriptions ({len(request.object_descriptions)})"
            )

        all_items: list[AssetItem] = []
        all_discarded_manipulands: list[str] = []
        original_descriptions: list[str] = []
        had_modifications = False
        failed_assets: list[FailedAsset] = []

        # Pre-analysis deduplication: group by (description, dimensions) to save LLM calls.
        # Track duplicates for tool feedback (same format as _generate_assets_with_model).
        unique_requests: dict[tuple[str, tuple[float, ...]], int] = {}
        duplicate_indices: dict[str, list[int]] = {}

        for idx, (desc, dims) in enumerate(
            zip(request.object_descriptions, request.desired_dimensions)
        ):
            key = (desc, tuple(dims))
            if key in unique_requests:
                # Track duplicate.
                if desc not in duplicate_indices:
                    duplicate_indices[desc] = []
                duplicate_indices[desc].append(idx)
            else:
                unique_requests[key] = idx

        # Store duplicate info for tool feedback.
        self.last_duplicate_info = duplicate_indices if duplicate_indices else None

        if len(unique_requests) < len(request.object_descriptions):
            console_logger.info(
                f"Pre-analysis deduplication: {len(request.object_descriptions)} requests "
                f"-> {len(unique_requests)} unique"
            )

        deterministic_analysis = bool(
            self.cfg.asset_manager.router.get("deterministic_analysis", False)
        )
        if deterministic_analysis:
            # These tool requests already contain one atomic object, dimensions,
            # a short name, and a placement type. Reconstructing those fields with
            # an LLM added 20-60 seconds per item and blocked asyncio deadlines.
            for (desc, dims), idx in unique_requests.items():
                all_items.append(
                    self._build_deterministic_asset_item(
                        description=desc,
                        short_name=request.short_names[idx],
                        dimensions=list(dims),
                        object_type=request.object_type,
                    )
                )
            console_logger.info(
                "Deterministically routed %d structured asset requests in-process",
                len(all_items),
            )
        else:
            # API-backed analysis may run in parallel. Subscription CLIs are one
            # supervised local worker, so their requests are intentionally serialized.
            configured_workers = self.cfg.asset_manager.router.parallel_workers
            max_workers = _subscription_aware_worker_count(
                configured_workers, len(unique_requests)
            )

            console_logger.info(
                f"Analyzing {len(unique_requests)} requests "
                f"with {max_workers} workers"
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.router.analyze_request,
                        description=desc,
                        dimensions=list(dims),
                    ): (idx, desc)
                    for (desc, dims), idx in unique_requests.items()
                }

                # Each model turn owns its liveness and hard-orphan watchdog. A
                # batch-wide wall-clock timeout is incorrect because queued turns
                # have not started yet.
                for future in as_completed(futures):
                    idx, desc = futures[future]
                    try:
                        analysis = future.result()

                        if analysis.error:
                            console_logger.warning(
                                f"Router rejected '{desc}': {analysis.error}"
                            )
                            failed_assets.append(
                                FailedAsset(
                                    index=idx,
                                    description=desc,
                                    error_message=analysis.error,
                                )
                            )
                            continue

                        # Validate item types match this agent.
                        type_error = self.router.validate_item_types(analysis.items)
                        if type_error:
                            console_logger.warning(
                                f"Router type validation failed: {type_error}"
                            )
                            failed_assets.append(
                                FailedAsset(
                                    index=idx,
                                    description=desc,
                                    error_message=type_error,
                                )
                            )
                            continue

                        # Collect items and track modifications.
                        all_items.extend(analysis.items)

                        if analysis.was_modified:
                            had_modifications = True
                            original_descriptions.append(
                                analysis.original_description or desc
                            )
                            if analysis.discarded_manipulands:
                                all_discarded_manipulands.extend(
                                    analysis.discarded_manipulands
                                )

                    except Exception as e:
                        console_logger.error(
                            f"Analysis failed for '{desc}': {e}", exc_info=True
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx, description=desc, error_message=str(e)
                            )
                        )

        if not all_items:
            console_logger.warning("Router returned no items to generate")
            return AssetGenerationResult(
                successful_assets=[],
                failed_assets=failed_assets,
                modification_info=None,
            )

        # Deduplicate items by description (same description = generate once).
        unique_items: dict[str, AssetItem] = {}
        for item in all_items:
            if item.description not in unique_items:
                unique_items[item.description] = item
        console_logger.info(
            f"Router produced {len(unique_items)} unique items from "
            f"{len(request.object_descriptions)} requests"
        )

        # Generate/retrieve using router. Handles multiple asset sources internally.
        result = self._generate_items_with_validation(
            unique_items=unique_items, request=request
        )

        # Build modification_info if request was modified.
        modification_info = None
        if had_modifications:
            modification_info = ModificationInfo(
                original_description=", ".join(original_descriptions),
                resulting_descriptions=[
                    item.description for item in unique_items.values()
                ],
                discarded_manipulands=(
                    all_discarded_manipulands if all_discarded_manipulands else None
                ),
            )

        # Combine failed assets from analysis phase with those from generation phase.
        all_failed = failed_assets + result.failed_assets

        return AssetGenerationResult(
            successful_assets=result.successful_assets,
            failed_assets=all_failed,
            modification_info=modification_info,
        )

    @staticmethod
    def _build_deterministic_asset_item(
        *,
        description: str,
        short_name: str,
        dimensions: list[float],
        object_type: ObjectType,
    ) -> AssetItem:
        """Map an already-structured tool request to a retrieval strategy."""
        normalized = description.casefold()
        articulated_terms = (
            "cabinet",
            "cupboard",
            "drawer",
            "dresser",
            "wardrobe",
            "refrigerator",
            "fridge",
            "locker",
        )
        floor_covering_terms = ("rug", "carpet", "floor mat", "runner")
        wall_covering_terms = ("poster", "painting", "wall art", "tapestry")
        surface_covering_terms = ("tablecloth", "placemat", "desk mat")

        thin_covering_type = None
        if object_type == ObjectType.FURNITURE and any(
            term in normalized for term in floor_covering_terms
        ):
            strategies = ["thin_covering", "generated"]
            thin_covering_type = "tileable"
        elif object_type == ObjectType.WALL_MOUNTED and any(
            term in normalized for term in wall_covering_terms
        ):
            strategies = ["thin_covering", "generated"]
            thin_covering_type = "single_image"
        elif object_type == ObjectType.MANIPULAND and any(
            term in normalized for term in surface_covering_terms
        ):
            strategies = ["thin_covering", "generated"]
            thin_covering_type = "tileable"
        elif object_type == ObjectType.FURNITURE and any(
            term in normalized for term in articulated_terms
        ):
            strategies = ["articulated", "generated"]
        else:
            strategies = ["generated"]

        return AssetItem(
            description=description,
            short_name=short_name,
            dimensions=dimensions,
            object_type=object_type,
            strategies=strategies,
            thin_covering_type=thin_covering_type,
        )

    def _generate_items_with_validation(
        self, unique_items: dict[str, "AssetItem"], request: AssetGenerationRequest
    ) -> AssetGenerationResult:
        """Generate items with overlapped generation and conversion.

        Generates geometry via parallel HTTP calls (thread-safe) and converts each
        mesh to a simulation asset immediately as it completes. This overlaps
        GPU-bound generation with CPU-bound conversion for better resource utilization.

        The main thread runs the as_completed loop and handles conversion (bpy),
        while worker threads continue fetching geometry in parallel.

        Args:
            unique_items: Dict of description -> AssetItem to generate.
            request: Original request (for style_context, object_type).

        Returns:
            AssetGenerationResult with successful assets and failures.
        """
        failed_assets: list[FailedAsset] = []
        successful_assets: list[SceneObject] = []

        configured_workers = self.cfg.asset_manager.router.parallel_workers
        items_list = list(unique_items.items())
        # Candidate validation includes model-based mesh analysis. Running these
        # tasks in parallel against a subscription CLI only creates a hidden
        # model queue and also makes local CLIP retrievers contend for MPS. Feed
        # that pipeline serially in subscription mode; API mode remains parallel.
        max_workers = _subscription_aware_worker_count(
            configured_workers, len(items_list)
        )

        console_logger.info(
            f"Generating {len(items_list)} items with {max_workers} parallel workers "
            "(overlapping generation with conversion)"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_geometry_with_validation,
                    item=item,
                    request=request,
                ): (idx, desc, item)
                for idx, (desc, item) in enumerate(items_list)
            }

            # Retrieval/generation clients enforce per-request deadlines. Do not
            # impose a second deadline on the whole batch: later futures may be
            # healthy and merely waiting for a bounded local worker.
            for future in as_completed(futures):
                idx, desc, item = futures[future]
                try:
                    generated = future.result()
                    if generated is None:
                        console_logger.warning(f"All attempts exhausted for '{desc}'")
                        failed_assets.append(
                            FailedAsset(
                                index=idx,
                                description=desc,
                                error_message="All generation/retrieval attempts exhausted",
                            )
                        )
                        continue

                    console_logger.info(
                        f"Geometry acquired for '{desc}', converting..."
                    )

                    # Convert immediately while other geometries are still generating.
                    # This runs on main thread (bpy) while workers fetch next geometry.
                    try:
                        # Handle ArticulatedGeometry (SDF assets) vs GeneratedGeometry.
                        if isinstance(generated, ArticulatedGeometry):
                            scene_obj = self._convert_articulated_to_scene_object(
                                articulated=generated, request=request
                            )
                        else:
                            scene_obj = self._convert_generated_to_scene_object(
                                item=item, generated=generated, request=request
                            )
                        successful_assets.append(scene_obj)
                        console_logger.info(f"Successfully converted asset: '{desc}'")
                    except Exception as e:
                        console_logger.error(
                            f"Mesh conversion failed for '{desc}': {e}", exc_info=True
                        )
                        failed_assets.append(
                            FailedAsset(
                                index=idx, description=desc, error_message=str(e)
                            )
                        )

                except Exception as e:
                    console_logger.error(
                        f"Geometry generation failed for '{desc}': {e}", exc_info=True
                    )
                    failed_assets.append(
                        FailedAsset(index=idx, description=desc, error_message=str(e))
                    )

        console_logger.info(
            f"Router generation completed: {len(successful_assets)} success, "
            f"{len(failed_assets)} failed"
        )

        return AssetGenerationResult(
            successful_assets=successful_assets, failed_assets=failed_assets
        )
