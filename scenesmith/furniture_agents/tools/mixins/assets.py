import logging
import time

from scenesmith.agent_utils.assets.asset_models import (
    AssetGenerationRequest,
    AssetGenerationResult as DomainAssetGenerationResult,
)
from scenesmith.agent_utils.core.response_datatypes import (
    AssetGenerationResult,
    GeneratedAsset,
)
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType
from scenesmith.furniture_agents.tools.response_dataclasses import (
    AssetInfo,
    AvailableAssetsResult,
)

console_logger = logging.getLogger(__name__)


class FurnitureAssetOperationsMixin:
    """Furniture generation, listing, and result-message helpers."""

    def _add_duplicate_warning(self, message_parts: list[str]) -> None:
        """Add duplicate warning to message if duplicates were detected."""
        duplicate_info = self.asset_manager.last_duplicate_info
        if duplicate_info:
            total_duplicates = sum(len(indices) for indices in duplicate_info.values())
            message_parts.append("")
            message_parts.append("⚠️  DUPLICATES REMOVED:")
            message_parts.append(
                f"You requested {total_duplicates} duplicate item(s). "
                "I generated each unique item only once:"
            )
            for desc, indices in duplicate_info.items():
                count = len(indices) + 1  # +1 for the original
                message_parts.append(f"  - '{desc}' (requested {count} times)")
            message_parts.append("")
            message_parts.append(
                "REMINDER: To place multiple identical items, use "
                "add_furniture_to_scene_tool with the SAME asset_id at different "
                "positions. Do NOT generate the same asset multiple times."
            )

    def _build_partial_success_message(
        self,
        result: DomainAssetGenerationResult,
        generated_assets: list[GeneratedAsset],
    ) -> tuple[str, str]:
        """Build message for partial success case."""
        message_parts = [
            f"Generated {len(generated_assets)} asset(s) successfully, but "
            f"{len(result.failed_assets)} failed:"
        ]

        # List successful assets.
        if generated_assets:
            message_parts.append("\n✓ SUCCESSFUL:")
            for asset in generated_assets:
                message_parts.append(f"  - {asset.name} (ID: {asset.object_id})")

        # List failed assets with error details.
        message_parts.append("\n✗ FAILED:")
        failure_details = []
        for failed in result.failed_assets:
            message_parts.append(f"  - {failed.description}: {failed.error_message}")
            failure_details.append(f"- {failed.description}: {failed.error_message}")

        message_parts.append(
            "\nRECOMMENDATION: Regenerate only the failed assets with adjusted "
            "prompts if needed."
        )

        # Add duplicate warning if applicable.
        self._add_duplicate_warning(message_parts)

        return "\n".join(message_parts), "\n".join(failure_details)

    def _build_full_success_message(
        self, generated_assets: list[GeneratedAsset], object_type: ObjectType
    ) -> str:
        """Build message for full success case."""
        message_parts = [
            f"Successfully generated {len(generated_assets)} unique "
            f"{object_type.value} asset(s):"
        ]

        # List generated assets with IDs.
        for asset in generated_assets:
            message_parts.append(f"  - {asset.name} (ID: {asset.object_id})")

        # Add duplicate warning if applicable.
        self._add_duplicate_warning(message_parts)

        return "\n".join(message_parts)

    def _generate_assets_impl(self, request: AssetGenerationRequest) -> str:
        """Implementation for generating assets with partial success handling."""
        console_logger.info(
            f"Generating batch of {len(request.object_descriptions)} assets"
        )
        start_time = time.time()

        # Generate assets using the asset manager.
        result = self.asset_manager.generate_assets(request)

        # Convert successful assets to DTOs.
        generated_assets = [
            GeneratedAsset(
                name=obj.name,
                object_id=str(obj.object_id),
                description=obj.description,
                width=(
                    float(obj.bbox_max[0] - obj.bbox_min[0])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                depth=(
                    float(obj.bbox_max[1] - obj.bbox_min[1])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                height=(
                    float(obj.bbox_max[2] - obj.bbox_min[2])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
            )
            for obj in result.successful_assets
        ]

        elapsed_time = time.time() - start_time

        # Handle partial success.
        if result.has_failures:
            console_logger.warning(
                f"Asset generation completed with {len(result.failed_assets)} "
                f"failure(s) and {len(result.successful_assets)} success(es) in "
                f"{elapsed_time:.2f} seconds"
            )

            message, failure_details = self._build_partial_success_message(
                result=result, generated_assets=generated_assets
            )

            return AssetGenerationResult(
                success=False,
                assets=generated_assets,
                message=message,
                successful_count=len(generated_assets),
                failed_count=len(result.failed_assets),
                failures=failure_details,
            ).to_json()

        # All succeeded.
        console_logger.info(
            f"Successfully generated {len(generated_assets)} assets in batch in "
            f"{elapsed_time:.2f} seconds"
        )

        message = self._build_full_success_message(
            generated_assets=generated_assets, object_type=request.object_type
        )

        return AssetGenerationResult(
            success=True,
            assets=generated_assets,
            message=message,
        ).to_json()

    def _list_available_assets_impl(self) -> str:
        """List all assets available for reuse.

        Returns:
            JSON response with list of available assets.
        """
        console_logger.info("Tool called: list_available_assets")
        try:
            available_assets = self.asset_manager.list_available_assets()

            asset_infos = [
                AssetInfo.from_scene_object(asset) for asset in available_assets
            ]

            result = AvailableAssetsResult(
                success=True,
                assets=asset_infos,
                count=len(asset_infos),
                message=f"Found {len(asset_infos)} available assets for reuse",
            )

            console_logger.info(f"Listed {len(asset_infos)} available assets")
            return result.to_json()

        except Exception as e:
            result = AvailableAssetsResult(
                success=False,
                assets=[],
                count=0,
                message=f"Failed to list available assets: {e}",
            )
            return result.to_json()
