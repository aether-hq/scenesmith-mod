"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import logging

from typing import Callable, Literal

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.floor_plan_agents.tools.floor_plan_models import (
    DoorWindowConfig,
    Result,
)
from scenesmith.floor_plan_agents.tools.materials_resolver import (
    MaterialsConfig,
    MaterialsResolver,
)
from scenesmith.floor_plan_agents.tools.mixins.materials import FloorPlanMaterialsMixin
from scenesmith.floor_plan_agents.tools.mixins.room_editing import (
    FloorPlanRoomEditingMixin,
)
from scenesmith.floor_plan_agents.tools.mixins.structural import (
    FloorPlanStructuralMixin,
)
from scenesmith.floor_plan_agents.tools.mixins.submission import (
    FloorPlanSubmissionMixin,
)
from scenesmith.floor_plan_agents.tools.mixins.tool_closures import (
    FloorPlanToolClosureMixin,
)
from scenesmith.floor_plan_agents.tools.open_plan_mixin import OpenPlanMixin
from scenesmith.floor_plan_agents.tools.submission.door_window_mixin import (
    DoorWindowMixin,
)
from scenesmith.floor_plan_agents.tools.submission.placement.models import (
    PlacementConfig,
    ScoringWeights,
)

console_logger = logging.getLogger(__name__)


class FloorPlanTools(
    DoorWindowMixin,
    OpenPlanMixin,
    FloorPlanToolClosureMixin,
    FloorPlanSubmissionMixin,
    FloorPlanStructuralMixin,
    FloorPlanRoomEditingMixin,
    FloorPlanMaterialsMixin,
):
    """Tools for floor plan designer agent.

    Follow the workflow phases:
    1. Room Layout - generate_room_specs, resize_room, add/remove_adjacency
    2. Wall Height - set_wall_height
    3. Doors - add_door, remove_door
    4. Windows - add_window, remove_window
    5. Materials - get_material, set_room_materials, set_exterior_material
    6. Validation - validate
    """

    def __init__(
        self,
        layout: HouseLayout,
        mode: Literal["room", "house"] = "room",
        materials_config: MaterialsConfig | None = None,
        min_opening_separation: float = 0.5,
        placement_timeout_seconds: float = 5.0,
        placement_scoring_weights: ScoringWeights | None = None,
        placement_exterior_wall_clearance_m: float = 20.0,
        door_window_config: DoorWindowConfig | None = None,
        wall_height_min: float = 2.0,
        wall_height_max: float = 12.0,
        room_dim_min: float = 1.5,
        room_dim_max: float = 20.0,
        checkpoint_callback: Callable[[], bool] | None = None,
    ):
        """Initialize floor plan tools.

        Args:
            layout: The HouseLayout to modify.
            mode: "room" (single room) or "house" (multi-room).
            materials_config: Materials resolver configuration.
            min_opening_separation: Minimum gap between door and window on same wall.
            placement_timeout_seconds: Backtracking search timeout for room placement.
            placement_scoring_weights: Weights for layout scoring (compactness, stability).
            placement_exterior_wall_clearance_m: Clearance zone for exterior_walls constraint.
            door_window_config: Door and window constraints configuration.
            wall_height_min: Minimum wall height in meters.
            wall_height_max: Maximum wall height in meters.
            room_dim_min: Minimum room dimension (width or depth) in meters.
            room_dim_max: Maximum room dimension (width or depth) in meters.
            checkpoint_callback: Optional durable checkpoint hook invoked only
                when the current layout is already valid.
        """
        self.layout = layout
        self.mode = mode
        self.materials_resolver = MaterialsResolver(materials_config)
        self.min_opening_separation = min_opening_separation
        self.placement_config = PlacementConfig(
            timeout_seconds=placement_timeout_seconds,
            scoring_weights=placement_scoring_weights or ScoringWeights(),
            exterior_wall_clearance_m=placement_exterior_wall_clearance_m,
        )
        self.door_window_config = door_window_config or DoorWindowConfig()
        self.wall_height_min = wall_height_min
        self.wall_height_max = wall_height_max
        self.room_dim_min = room_dim_min
        self.room_dim_max = room_dim_max
        self.checkpoint_callback = checkpoint_callback

        # Build tools dictionary using closure pattern.
        # This avoids including 'self' in OpenAI function schemas.
        self.tools = self._create_tool_closures()
        self.submit_floor_plan_tool = self._create_submit_floor_plan_tool()

    def _check_rooms_exist(self) -> Result | None:
        """Check if rooms have been defined.

        Returns:
            Error Result if no rooms, None if OK.
        """
        if not self.layout.room_specs:
            return Result(
                success=False,
                message="No rooms defined. Call generate_room_specs first.",
            )
        return None

    def _checkpoint_if_valid(self) -> bool:
        """Persist a valid layout without turning checkpoint I/O into a tool error."""

        if (
            self.checkpoint_callback is None
            or not self.layout.placement_valid
            or not self.layout.connectivity_valid
        ):
            return False
        try:
            return bool(self.checkpoint_callback())
        except Exception as exc:
            console_logger.warning("Floor-plan checkpoint hook failed: %s", exc)
            return False

    def _fail(self, message: str) -> Result:
        """Log failure and return Result with success=False.

        Args:
            message: Failure message to log and return.

        Returns:
            Result with success=False and the message.
        """
        console_logger.info(f"Tool failed: {message}")
        return Result(success=False, message=message)
