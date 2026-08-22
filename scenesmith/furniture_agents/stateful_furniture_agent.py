"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import logging

from pathlib import Path

from omegaconf import DictConfig

from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.runtime.base_stateful_agent import BaseStatefulAgent
from scenesmith.agent_utils.scene.clearance_zones import (
    compute_door_clearance_violations,
    compute_window_clearance_violations,
)
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenesmith.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)

from scenesmith.furniture_agents.mixins.agent_setup import FurnitureAgentSetupMixin
from scenesmith.furniture_agents.mixins.room_kit_placement import (
    FurnitureRoomKitPlacementMixin,
)
from scenesmith.furniture_agents.mixins.workflow import FurnitureAgentWorkflowMixin
from scenesmith.furniture_agents.room_kit import (
    planning as _room_kit_planning,
    validation as _room_kit_validation,
)
from scenesmith.furniture_agents.room_kit.planning import (
    _normalize_dense_library_bookcases as _normalize_dense_library_bookcases_impl,
)
from scenesmith.furniture_agents.room_kit.validation import (
    _validate_furniture_collision_free as _validate_furniture_collision_free_impl,
)


def _normalize_dense_library_bookcases(*args, **kwargs):
    """Preserve historic clearance-detector patch points."""

    _room_kit_planning.compute_window_clearance_violations = (
        compute_window_clearance_violations
    )
    _room_kit_planning.compute_door_clearance_violations = (
        compute_door_clearance_violations
    )
    return _normalize_dense_library_bookcases_impl(*args, **kwargs)


def _validate_furniture_collision_free(*args, **kwargs):
    """Preserve the historic collision-computation patch point."""

    _room_kit_validation.compute_scene_collisions = compute_scene_collisions
    return _validate_furniture_collision_free_impl(*args, **kwargs)


class StatefulFurnitureAgent(
    FurnitureAgentSetupMixin,
    FurnitureRoomKitPlacementMixin,
    FurnitureAgentWorkflowMixin,
    BaseStatefulAgent,
    BaseFurnitureAgent,
):
    """Natural conversation between persistent agents with proper image injection."""

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.FURNITURE

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        num_workers: int = 1,
        render_allocation: RenderAllocation | None = None,
    ):
        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
        )
        # Initialize furniture-specific base class.
        BaseFurnitureAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
            articulated_server_host=articulated_server_host,
            articulated_server_port=articulated_server_port,
            materials_server_host=materials_server_host,
            materials_server_port=materials_server_port,
            num_workers=num_workers,
            render_allocation=render_allocation,
        )

        # Create persistent agent sessions using base class method.
        self.designer_session, self.critic_session = self._create_sessions()

        # Context image for designer initialization (furniture-specific).
        self.context_image_path: Path | None = None
        self.room_kit_brief = (
            "No semantic room kit matched; use the scene requirements."
        )

    def _preprune_and_recover_room_kit(self, *args, **kwargs):
        """Keep clearance-detector patches visible across the mixin boundary."""

        _room_kit_planning.compute_window_clearance_violations = (
            compute_window_clearance_violations
        )
        _room_kit_planning.compute_door_clearance_violations = (
            compute_door_clearance_violations
        )
        return super()._preprune_and_recover_room_kit(*args, **kwargs)
