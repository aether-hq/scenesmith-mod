import logging

from omegaconf import DictConfig, OmegaConf

from scenesmith.agent_utils.articulated_retrieval_server import (
    ArticulatedRetrievalServer,
)
from scenesmith.agent_utils.geometry_generation_server.execution_provider import (
    resolve_geometry_execution_provider,
)
from scenesmith.agent_utils.geometry_generation_server.service_provider import (
    GeometryService,
    resolve_geometry_service_provider,
)
from scenesmith.agent_utils.hssd_retrieval_server import HssdRetrievalServer
from scenesmith.agent_utils.materials_retrieval_server import MaterialsRetrievalServer
from scenesmith.agent_utils.objaverse_retrieval_server import ObjaverseRetrievalServer
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.ceiling_agents.stateful_ceiling_agent import StatefulCeilingAgent
from scenesmith.experiments.indoor.runtime_support import (
    _asset_config_uses_generated_geometry,
    _get_retrieval_compute_device,
    _resolve_geometry_runtime_configuration,
)
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent

console_logger = logging.getLogger(__name__)

# Pipeline stages in execution order (derived from AgentType enum).
PIPELINE_STAGES = [agent.value for agent in AgentType]

# Stage dependencies for resume from checkpoint.
# Maps start_stage to the checkpoint it needs from the previous stage.
STAGE_CHECKPOINTS = {
    "floor_plan": None,
    "furniture": None,
    "wall_mounted": "scene_after_furniture",
    "ceiling_mounted": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
}

# Maps start_stage to the asset directories it needs from previous stages.
STAGE_ASSET_DIRS = {
    "floor_plan": [],
    "furniture": [],
    "wall_mounted": ["furniture"],
    "ceiling_mounted": ["furniture", "wall_mounted"],
    "manipuland": ["furniture", "wall_mounted", "ceiling_mounted"],
}


class IndoorServiceLifecycleMixin:
    """An experiment that generates indoor scenes."""

    compatible_floor_plan_agents = {
        "stateful_floor_plan_agent": StatefulFloorPlanAgent,
    }
    compatible_furniture_agents = {
        "stateful_furniture_agent": StatefulFurnitureAgent,
    }
    compatible_manipuland_agents = {
        "stateful_manipuland_agent": StatefulManipulandAgent,
    }
    compatible_wall_agents = {
        "stateful_wall_agent": StatefulWallAgent,
    }
    compatible_ceiling_agents = {
        "stateful_ceiling_agent": StatefulCeilingAgent,
    }

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        self.geometry_server: GeometryService | None = None
        self.hssd_server: HssdRetrievalServer | None = None
        self.objaverse_server: ObjaverseRetrievalServer | None = None
        self.polyhaven_server: ObjaverseRetrievalServer | None = None
        self.articulated_server: ArticulatedRetrievalServer | None = None
        self.materials_server: MaterialsRetrievalServer | None = None
        self._retrieval_device: str | None = None

    def _resolve_retrieval_device(self) -> str:
        """Resolve and cache the configured retrieval execution provider."""

        if self._retrieval_device is None:
            self._retrieval_device = _get_retrieval_compute_device(
                requested=self.provider_selection.compute,
                policy=self.provider_selection.policy,
            )
        return self._retrieval_device

    def __del__(self):
        """Ensure servers are stopped when experiment is destroyed."""
        if self.geometry_server and self.geometry_server.is_running():
            console_logger.warning("Stopping geometry server in destructor")
            try:
                self.geometry_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop geometry server in destructor: {e}"
                )

        if self.hssd_server and self.hssd_server.is_running():
            console_logger.warning("Stopping HSSD server in destructor")
            try:
                self.hssd_server.stop()
            except Exception as e:
                console_logger.error(f"Failed to stop HSSD server in destructor: {e}")

        if self.objaverse_server and self.objaverse_server.is_running():
            console_logger.warning("Stopping Objaverse server in destructor")
            try:
                self.objaverse_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop Objaverse server in destructor: {e}"
                )

        if self.polyhaven_server and self.polyhaven_server.is_running():
            console_logger.warning("Stopping Poly Haven server in destructor")
            try:
                self.polyhaven_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop Poly Haven server in destructor: {e}"
                )

        if self.articulated_server and self.articulated_server.is_running():
            console_logger.warning("Stopping articulated server in destructor")
            try:
                self.articulated_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop articulated server in destructor: {e}"
                )

        if self.materials_server and self.materials_server.is_running():
            console_logger.warning("Stopping materials server in destructor")
            try:
                self.materials_server.stop()
            except Exception as e:
                console_logger.error(
                    f"Failed to stop materials server in destructor: {e}"
                )

    def _start_geometry_server(self) -> None:
        """Start geometry generation server (if general_asset_source == 'generated')."""
        # "all" can be catalog-only. Only allocate the expensive geometry
        # runtime when at least one agent's federated hierarchy can actually
        # fall through to generated geometry.
        agent_configs = [
            self.cfg.furniture_agent.asset_manager,
            self.cfg.wall_agent.asset_manager,
            self.cfg.ceiling_agent.asset_manager,
            self.cfg.manipuland_agent.asset_manager,
        ]
        if not any(
            _asset_config_uses_generated_geometry(
                OmegaConf.to_container(asset_config, resolve=True)
            )
            for asset_config in agent_configs
        ):
            console_logger.info(
                "Skipping geometry server; configured asset hierarchy is catalog-only"
            )
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.geometry_generation_server

        backend, sam3d_config = _resolve_geometry_runtime_configuration(self.cfg)

        service_provider = resolve_geometry_service_provider(
            self.provider_selection.geometry_service,
            external_scheme=self.provider_selection.external_scheme,
            external_auth_token=self.provider_selection.external_auth_token,
            environ={},
        )
        execution_provider = None
        if service_provider.key == "local":
            execution_provider = resolve_geometry_execution_provider(
                backend=str(backend),
                sam3d_config=sam3d_config,
                requested=self.provider_selection.geometry,
                environ={},
            )
        provider_label = (
            f"/{execution_provider.key}"
            if execution_provider is not None
            else "/remote-selected"
        )
        console_logger.info(
            "Connecting to %s geometry service (%s%s) at %s:%s",
            service_provider.key,
            backend,
            provider_label,
            server_config.host,
            server_config.port,
        )

        self.geometry_server = service_provider.connect(
            host=server_config.host,
            port=server_config.port,
            backend=backend,
            sam3d_config=sam3d_config,
            log_file=self.output_dir / "experiment.log",
            execution_provider=execution_provider,
        )

        self.geometry_server.start()
        self.geometry_server.wait_until_ready(timeout_s=30.0)
        console_logger.info("Geometry generation server ready")

    def _stop_geometry_server(self) -> None:
        """Stop the geometry generation server."""
        if self.geometry_server and self.geometry_server.is_running():
            console_logger.info("Stopping geometry generation server...")
            self.geometry_server.stop()
            console_logger.info("Geometry generation server stopped")
            self.geometry_server = None

    def _start_hssd_server(self) -> None:
        """Start HSSD retrieval server (if general_asset_source == 'hssd')."""
        # Only start if at least one agent uses HSSD strategy.
        furniture_uses_hssd = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "hssd"
        )
        manipuland_uses_hssd = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "hssd"
        )
        wall_uses_hssd = (
            self.cfg.wall_agent.asset_manager.general_asset_source == "hssd"
        )
        ceiling_uses_hssd = (
            self.cfg.ceiling_agent.asset_manager.general_asset_source == "hssd"
        )

        if not (
            furniture_uses_hssd
            or manipuland_uses_hssd
            or wall_uses_hssd
            or ceiling_uses_hssd
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.hssd_retrieval_server
        # Get HSSD data configuration from asset manager config.
        hssd_config = self.cfg.furniture_agent.asset_manager.hssd

        retrieval_device = self._resolve_retrieval_device()
        console_logger.info(
            f"Starting HSSD retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device})"
        )

        self.hssd_server = HssdRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            hssd_data_path=str(hssd_config.data_path),
            hssd_preprocessed_path=str(hssd_config.preprocessed_path),
            hssd_top_k=hssd_config.use_top_k,
            clip_device=retrieval_device,
        )

        self.hssd_server.start()
        # Longer timeout for CLIP loading.
        self.hssd_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("HSSD retrieval server ready")

    def _stop_hssd_server(self) -> None:
        """Stop the HSSD retrieval server."""
        if self.hssd_server and self.hssd_server.is_running():
            console_logger.info("Stopping HSSD retrieval server...")
            self.hssd_server.stop()
            console_logger.info("HSSD retrieval server stopped")
            self.hssd_server = None

    def _start_objaverse_server(self) -> None:
        """Start Objaverse retrieval server (if general_asset_source == 'objaverse')."""
        # Only start if at least one agent uses objaverse strategy.
        furniture_uses_objaverse = (
            self.cfg.furniture_agent.asset_manager.general_asset_source == "objaverse"
        )
        manipuland_uses_objaverse = (
            self.cfg.manipuland_agent.asset_manager.general_asset_source == "objaverse"
        )
        wall_uses_objaverse = (
            self.cfg.wall_agent.asset_manager.general_asset_source == "objaverse"
        )
        ceiling_uses_objaverse = (
            self.cfg.ceiling_agent.asset_manager.general_asset_source == "objaverse"
        )

        if not (
            furniture_uses_objaverse
            or manipuland_uses_objaverse
            or wall_uses_objaverse
            or ceiling_uses_objaverse
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.objaverse_retrieval_server
        # Get Objaverse data configuration from asset manager config.
        objaverse_config = self.cfg.furniture_agent.asset_manager.objaverse

        retrieval_device = self._resolve_retrieval_device()
        console_logger.info(
            f"Starting Objaverse retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device})"
        )

        self.objaverse_server = ObjaverseRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,
            objaverse_data_path=str(objaverse_config.data_path),
            objaverse_preprocessed_path=str(objaverse_config.preprocessed_path),
            objaverse_top_k=objaverse_config.use_top_k,
            clip_device=retrieval_device,
        )

        self.objaverse_server.start()
        # Longer timeout for CLIP loading.
        self.objaverse_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Objaverse retrieval server ready")

    def _stop_objaverse_server(self) -> None:
        """Stop the Objaverse retrieval server."""
        if self.objaverse_server and self.objaverse_server.is_running():
            console_logger.info("Stopping Objaverse retrieval server...")
            self.objaverse_server.stop()
            console_logger.info("Objaverse retrieval server stopped")
            self.objaverse_server = None

    def _start_polyhaven_server(self) -> None:
        """Start the Poly Haven catalog server when selected directly or via all."""
        sources = [
            self.cfg.furniture_agent.asset_manager.general_asset_source,
            self.cfg.manipuland_agent.asset_manager.general_asset_source,
            self.cfg.wall_agent.asset_manager.general_asset_source,
            self.cfg.ceiling_agent.asset_manager.general_asset_source,
        ]
        if not any(source in {"polyhaven", "all"} for source in sources):
            return

        server_config = self.cfg.experiment.polyhaven_retrieval_server
        uses_global_catalog = any(source == "all" for source in sources)
        catalog_config = self.cfg.furniture_agent.asset_manager.polyhaven
        data_path = (
            "data/global-assets"
            if uses_global_catalog
            else str(catalog_config.data_path)
        )
        preprocessed_path = (
            "data/global-assets/preprocessed"
            if uses_global_catalog
            else str(catalog_config.preprocessed_path)
        )
        top_k = 50 if uses_global_catalog else catalog_config.use_top_k
        # ViT-L text inference is ~50x faster on CPU than MPS on current Apple
        # Silicon/PyTorch for these tiny one-string queries (about 0.1s vs 5s).
        retrieval_device = (
            "cpu" if uses_global_catalog else self._resolve_retrieval_device()
        )
        console_logger.info(
            "Starting %s retrieval server on %s:%s (CLIP device: %s)",
            "global asset catalog" if uses_global_catalog else "Poly Haven",
            server_config.host,
            server_config.port,
            retrieval_device,
        )
        self.polyhaven_server = ObjaverseRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,
            objaverse_data_path=data_path,
            objaverse_preprocessed_path=preprocessed_path,
            objaverse_top_k=top_k,
            clip_device=retrieval_device,
        )
        self.polyhaven_server.start()
        self.polyhaven_server.wait_until_ready(timeout_s=60.0)
        console_logger.info(
            "%s retrieval server ready",
            "Global asset catalog" if uses_global_catalog else "Poly Haven",
        )

    def _stop_polyhaven_server(self) -> None:
        """Stop the Poly Haven catalog server."""
        if self.polyhaven_server and self.polyhaven_server.is_running():
            console_logger.info("Stopping Poly Haven retrieval server...")
            self.polyhaven_server.stop()
            console_logger.info("Poly Haven retrieval server stopped")
            self.polyhaven_server = None

    def _start_articulated_server(self) -> None:
        """Start articulated retrieval server (if articulated strategy is enabled)."""
        # Check if articulated strategy is enabled for any agent.
        furniture_articulated_enabled = (
            self.cfg.furniture_agent.asset_manager.router.strategies.articulated.enabled
        )
        manipuland_articulated_enabled = (
            self.cfg.manipuland_agent.asset_manager.router.strategies.articulated.enabled
        )
        wall_articulated_enabled = (
            self.cfg.wall_agent.asset_manager.router.strategies.articulated.enabled
        )
        ceiling_articulated_enabled = (
            self.cfg.ceiling_agent.asset_manager.router.strategies.articulated.enabled
        )

        if not (
            furniture_articulated_enabled
            or manipuland_articulated_enabled
            or wall_articulated_enabled
            or ceiling_articulated_enabled
        ):
            return

        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.articulated_retrieval_server

        # Get articulated data configuration from furniture agent config.
        articulated_config = self.cfg.furniture_agent.asset_manager.articulated

        retrieval_device = self._resolve_retrieval_device()
        console_logger.info(
            f"Starting articulated retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device})"
        )

        self.articulated_server = ArticulatedRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            articulated_config=articulated_config,
            clip_device=retrieval_device,
        )

        self.articulated_server.start()
        # Longer timeout for CLIP loading.
        self.articulated_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Articulated retrieval server ready")

    def _stop_articulated_server(self) -> None:
        """Stop the articulated retrieval server."""
        if self.articulated_server and self.articulated_server.is_running():
            console_logger.info("Stopping articulated retrieval server...")
            self.articulated_server.stop()
            console_logger.info("Articulated retrieval server stopped")
            self.articulated_server = None

    def _start_materials_server(self) -> None:
        """Start materials retrieval server."""
        # Get server configuration from experiment config.
        server_config = self.cfg.experiment.materials_retrieval_server

        retrieval_device = self._resolve_retrieval_device()
        console_logger.info(
            f"Starting materials retrieval server on "
            f"{server_config.host}:{server_config.port} "
            f"(CLIP device: {retrieval_device})"
        )

        self.materials_server = MaterialsRetrievalServer(
            host=server_config.host,
            port=server_config.port,
            preload_retriever=True,  # Always preload CLIP for consistent performance.
            materials_config=server_config,  # Pass DictConfig directly.
            clip_device=retrieval_device,
        )

        self.materials_server.start()
        # Longer timeout for CLIP loading.
        self.materials_server.wait_until_ready(timeout_s=60.0)
        console_logger.info("Materials retrieval server ready")

    def _stop_materials_server(self) -> None:
        """Stop the materials retrieval server."""
        if self.materials_server and self.materials_server.is_running():
            console_logger.info("Stopping materials retrieval server...")
            self.materials_server.stop()
            console_logger.info("Materials retrieval server stopped")
            self.materials_server = None
