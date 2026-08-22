import logging

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalClient,
)
from scenesmith.agent_utils.asset_router import AssetRouter
from scenesmith.agent_utils.assets.asset_registry import AssetRegistry
from scenesmith.agent_utils.assets.image_generation import create_image_generator
from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionClient
from scenesmith.agent_utils.geometry.mesh_canonicalization import canonicalize_mesh
from scenesmith.agent_utils.geometry.mesh_utils import (
    remove_mesh_floaters,
    scale_mesh_uniformly_to_dimensions,
)
from scenesmith.agent_utils.geometry.sdf_generator import generate_drake_sdf
from scenesmith.agent_utils.geometry_generation_server.client import (
    GeometryGenerationClient,
)
from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalClient
from scenesmith.agent_utils.llm.vlm_service import VLMService
from scenesmith.agent_utils.materials_retrieval_server import MaterialsRetrievalClient
from scenesmith.agent_utils.objaverse_retrieval_server import ObjaverseRetrievalClient
from scenesmith.agent_utils.physics.mesh_physics_analyzer import (
    analyze_mesh_orientation_and_material,
)
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.utils.logging import BaseLogger

if TYPE_CHECKING:
    from scenesmith.agent_utils.asset_router import AssetRouter
    from scenesmith.agent_utils.blender import BlenderServer

console_logger = logging.getLogger(__name__)

HSSD_CANONICAL_CONVERSION_VERSION = 4

from scenesmith.agent_utils.assets.manager_mixins import (
    simulation_assets as _simulation_assets,
)
from scenesmith.agent_utils.assets.manager_mixins.conversion import AssetConversionMixin
from scenesmith.agent_utils.assets.manager_mixins.generation import AssetGenerationMixin
from scenesmith.agent_utils.assets.manager_mixins.registry import AssetRegistryMixin
from scenesmith.agent_utils.assets.manager_mixins.retrieval import AssetRetrievalMixin
from scenesmith.agent_utils.assets.manager_mixins.simulation_assets import (
    SimulationAssetConversionMixin,
)


class AssetManager(
    AssetRetrievalMixin,
    AssetGenerationMixin,
    AssetConversionMixin,
    SimulationAssetConversionMixin,
    AssetRegistryMixin,
):
    """Manages 3D asset acquisition for scene generation.

    Supports two acquisition strategies configured via `general_asset_source`:
    - "generated": Text-to-3D generation (text → image → 3D mesh)
    - "hssd": Retrieval from HSSD library

    Has two operating modes based on `router.enabled` config:

    **Router path** (router.enabled=True):
    - LLM analyzes requests to split composites and select strategies
    - Parallel HTTP calls for generation/retrieval (thread-safe)
    - Sequential bpy operations for mesh processing (main thread)
    - VLM validation with retry loop for quality control

    **Non-router path** (router.enabled=False):
    - Direct dispatch to generation or retrieval based on config
    - Batch processing without LLM analysis
    - Simpler but less flexible

    Both paths produce simulation-ready Drake SDF files with:
    - Canonical orientation (Z-up, Y-forward)
    - Convex decomposition collision geometry (CoACD or V-HACD)
    - VLM-estimated physics properties (material, mass)

    Maintains style consistency through conversational context and includes
    an asset registry to track generated assets for reuse.
    """

    def __init__(
        self,
        logger: BaseLogger,
        vlm_service: VLMService,
        blender_server: "BlenderServer | None",
        collision_client: ConvexDecompositionClient | None,
        cfg: DictConfig,
        agent_type: AgentType,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        objaverse_server_host: str = "127.0.0.1",
        objaverse_server_port: int = 7009,
        polyhaven_server_host: str = "127.0.0.1",
        polyhaven_server_port: int = 7010,
    ) -> None:
        """Initialize the asset manager.

        Args:
            logger: Logger instance for tracking operations.
            vlm_service: VLM service instance for mesh physics analysis.
            blender_server: Blender server instance for multi-view rendering.
            collision_client: Client for collision geometry generation via convex
                decomposition. Can be None for checkpoint loading (no collision
                generation needed).
            cfg: Configuration with asset_manager settings.
            agent_type: Agent type for directory organization. Assets will be
                stored in generated_assets/{agent_type.value}/.
            geometry_server_host: Host for geometry generation server.
            geometry_server_port: Port for geometry generation server.
            hssd_server_host: Host for HSSD retrieval server.
            hssd_server_port: Port for HSSD retrieval server.
            articulated_server_host: Host for articulated retrieval server.
            articulated_server_port: Port for articulated retrieval server.
            materials_server_host: Host for materials retrieval server.
            materials_server_port: Port for materials retrieval server.
            objaverse_server_host: Host for Objaverse retrieval server.
            objaverse_server_port: Port for Objaverse retrieval server.
            polyhaven_server_host: Host for Poly Haven retrieval server.
            polyhaven_server_port: Port for Poly Haven retrieval server.
        """
        self.output_dir = logger.output_dir
        self.logger = logger
        self.cfg = cfg
        self.agent_type = agent_type

        # Extract config values.
        self.num_side_views_for_physics_analysis = (
            cfg.asset_manager.num_side_views_for_physics_analysis
        )
        self.side_view_elevation_degrees = cfg.asset_manager.side_view_elevation_degrees
        self.min_mesh_dimension_meters = cfg.asset_manager.min_mesh_dimension_meters
        self.mesh_relative_dimension_threshold = (
            cfg.asset_manager.mesh_relative_dimension_threshold
        )
        # Store collision geometry configuration.
        self.collision_method = cfg.collision_geometry.method
        self.collision_coacd_cfg = cfg.collision_geometry.coacd
        self.collision_vhacd_cfg = cfg.collision_geometry.vhacd

        self.vlm_service = vlm_service
        self.blender_server = blender_server
        self.collision_client = collision_client
        self.image_generator = create_image_generator(
            backend=cfg.asset_manager.image_generation.backend,
            config=cfg.asset_manager.image_generation,
        )

        # Create agent-specific subdirectories for organization.
        generated_assets_dir = self.output_dir / "generated_assets" / agent_type.value
        self.images_dir = generated_assets_dir / "images"
        self.geometry_dir = generated_assets_dir / "geometry"
        self.sdf_dir = generated_assets_dir / "sdf"
        self.debug_dir = generated_assets_dir / "debug"

        for dir_path in [
            self.images_dir,
            self.geometry_dir,
            self.sdf_dir,
            self.debug_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize registry with auto-save to enable incremental persistence.
        registry_path = generated_assets_dir / "asset_registry.json"
        shared_registry_value = cfg.asset_manager.get("shared_registry_path")
        shared_registry_path = (
            Path(str(shared_registry_value)).expanduser()
            if shared_registry_value
            else None
        )
        self.registry = AssetRegistry(
            auto_save_path=registry_path,
            mirror_save_path=shared_registry_path,
        )
        if shared_registry_path is not None and shared_registry_path.is_file():
            try:
                self.registry.load_from_file(file_path=shared_registry_path)
                self._quarantine_incompatible_cached_assets()
                self.registry.save_to_file(file_path=registry_path)
                console_logger.info(
                    f"Loaded shared {agent_type.value} asset cache from "
                    f"{shared_registry_path}"
                )
            except Exception as exc:
                console_logger.warning(
                    f"Could not load shared asset cache {shared_registry_path}: {exc}"
                )

        # Initialize strategy-specific clients.
        self.general_asset_source = cfg.asset_manager.general_asset_source
        if self.general_asset_source not in [
            "generated",
            "hssd",
            "objaverse",
            "polyhaven",
            "all",
        ]:
            raise ValueError(f"Unknown asset source: {self.general_asset_source}")

        # Initialize geometry generation client if source is "generated".
        self.geometry_client: GeometryGenerationClient | None = None
        if self.general_asset_source in ["generated", "all"]:
            console_logger.info("Initializing geometry generation client")
            self.geometry_client = GeometryGenerationClient(
                host=geometry_server_host, port=geometry_server_port
            )

        # Initialize HSSD client if source is "hssd".
        self.hssd_client: HssdRetrievalClient | None = None
        if self.general_asset_source == "hssd":
            console_logger.info("Initializing HSSD retrieval client")
            self.hssd_client = HssdRetrievalClient(
                host=hssd_server_host, port=hssd_server_port
            )

        # Initialize Objaverse client if source is "objaverse".
        self.objaverse_client: ObjaverseRetrievalClient | None = None
        if self.general_asset_source == "objaverse":
            console_logger.info("Initializing Objaverse retrieval client")
            self.objaverse_client = ObjaverseRetrievalClient(
                host=objaverse_server_host, port=objaverse_server_port
            )

        # Direct Poly Haven and the normalized global catalog share the generic
        # catalog retrieval protocol; ``all`` points this client at the latter.
        self.polyhaven_client: ObjaverseRetrievalClient | None = None
        if self.general_asset_source in ["polyhaven", "all"]:
            catalog_label = (
                "global asset catalog"
                if self.general_asset_source == "all"
                else "Poly Haven"
            )
            console_logger.info("Initializing %s retrieval client", catalog_label)
            self.polyhaven_client = ObjaverseRetrievalClient(
                host=polyhaven_server_host, port=polyhaven_server_port
            )

        # Initialize articulated retrieval client if articulated strategy is enabled.
        self.articulated_client: ArticulatedRetrievalClient | None = None
        articulated_enabled = cfg.asset_manager.router.strategies.articulated.enabled
        if articulated_enabled:
            console_logger.info("Initializing articulated retrieval client")
            self.articulated_client = ArticulatedRetrievalClient(
                host=articulated_server_host, port=articulated_server_port
            )

        # Initialize materials retrieval client if thin_covering strategy is enabled.
        self.materials_client: MaterialsRetrievalClient | None = None
        thin_covering_enabled = (
            cfg.asset_manager.router.strategies.thin_covering.enabled
        )
        if thin_covering_enabled:
            console_logger.info("Initializing materials retrieval client")
            self.materials_client = MaterialsRetrievalClient(
                host=materials_server_host, port=materials_server_port
            )

        # Initialize asset router if enabled in config.
        self.router: "AssetRouter | None" = None
        if cfg.asset_manager.router.enabled:
            console_logger.info("Initializing asset router for LLM-advised generation")
            self.router = AssetRouter(
                agent_type=agent_type,
                vlm_service=vlm_service,
                cfg=cfg,
                blender_server=blender_server,
            )

        # Track duplicate requests from the last generate_assets call.
        self.last_duplicate_info: dict[str, list[int]] | None = None

    def _convert_mesh_to_simulation_asset(self, *args, **kwargs):
        """Preserve historic patch points for the mesh-conversion boundary."""

        _simulation_assets.remove_mesh_floaters = remove_mesh_floaters
        _simulation_assets.analyze_mesh_orientation_and_material = (
            analyze_mesh_orientation_and_material
        )
        _simulation_assets.scale_mesh_uniformly_to_dimensions = (
            scale_mesh_uniformly_to_dimensions
        )
        _simulation_assets.canonicalize_mesh = canonicalize_mesh
        _simulation_assets.generate_drake_sdf = generate_drake_sdf
        return super()._convert_mesh_to_simulation_asset(*args, **kwargs)
