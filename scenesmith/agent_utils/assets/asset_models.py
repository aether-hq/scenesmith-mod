import logging

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scenesmith.agent_utils.asset_router.dataclasses import ModificationInfo
from scenesmith.agent_utils.assets.image_generation import AssetOperationType
from scenesmith.agent_utils.llm.llm_harness import LLMHarnessConfig
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, SceneObject

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

HSSD_CANONICAL_CONVERSION_VERSION = 4


def _subscription_aware_worker_count(
    configured_workers: int, request_count: int
) -> int:
    """Return honest concurrency for work that may call the configured LLM.

    Subscription CLIs are single interactive workers protected by a process-wide
    lock. Sending several HTTP requests at them only builds a hidden queue; it
    cannot increase throughput and used to make batch-wide timers expire while
    otherwise healthy turns were waiting. Keep API providers parallel, but feed
    subscription workers one request at a time.
    """
    if LLMHarnessConfig.from_env().uses_cli_bridge:
        return min(1, request_count)
    return min(configured_workers, request_count)


@dataclass
class AssetPathConfig:
    """Configuration for asset file paths and metadata."""

    description: str
    """Description of the object."""

    short_name: str
    """Short name for the object."""

    image_path: Path | None
    """Path to the generated image."""

    geometry_path: Path
    """Path to the generated 3D geometry."""

    sdf_dir: Path
    """Directory containing the generated SDF file."""


@dataclass
class AssetGenerationRequest:
    """Request for generating scene assets (furniture, manipulands, etc.)."""

    object_descriptions: list[str]
    """List of object descriptions to generate."""

    short_names: list[str]
    """List of short names for filesystem-safe file naming."""

    object_type: ObjectType
    """Type of objects to generate (FURNITURE, MANIPULAND, etc.)."""

    desired_dimensions: list[list[float]]
    """Desired dimensions (width, depth, height) in meters for each object.
    Agent must predict dimensions considering scene context.
    Must match the length of object_descriptions.
    """

    style_context: str | None = None
    """Style context for consistency (e.g., 'modern minimalist kitchen')."""

    operation_type: AssetOperationType = AssetOperationType.INITIAL
    """Type of generation operation."""

    scene_id: str | None = None
    """Optional scene identifier for fair round-robin scheduling on servers.

    When multiple scenes generate assets concurrently, passing scene_id ensures
    fair GPU time allocation across scenes in the geometry and HSSD servers.
    """


@dataclass
class FailedAsset:
    """Information about a failed asset generation."""

    index: int
    """Index of the failed asset in the original request."""

    description: str
    """Description of the object that failed to generate."""

    error_message: str
    """Error message describing why generation failed."""


@dataclass
class AssetGenerationResult:
    """Result of asset generation with potential partial success."""

    successful_assets: list[SceneObject]
    """List of successfully generated scene objects."""

    failed_assets: list[FailedAsset]
    """List of assets that failed during generation."""

    modification_info: ModificationInfo | None = None
    """Set when router modified the original request (split composites or filtered
    items). Contains original description, resulting items, and any discarded
    manipulands (furniture agent only). None when router is disabled or request
    was not modified.
    """

    @property
    def has_failures(self) -> bool:
        """Check if any assets failed to generate."""
        return len(self.failed_assets) > 0

    @property
    def all_succeeded(self) -> bool:
        """Check if all assets were generated successfully."""
        return len(self.failed_assets) == 0
