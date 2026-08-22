import logging

from typing import Any

from agents import function_tool
from omegaconf import DictConfig

from scenesmith.agent_utils.assets.asset_manager import AssetManager
from scenesmith.agent_utils.assets.asset_models import AssetGenerationRequest
from scenesmith.agent_utils.core.loop_detector import LoopDetector
from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType
from scenesmith.furniture_agents.tools.mixins import validation as _validation_mixin
from scenesmith.furniture_agents.tools.mixins.assets import (
    FurnitureAssetOperationsMixin,
)
from scenesmith.furniture_agents.tools.mixins.placement import FurniturePlacementMixin
from scenesmith.furniture_agents.tools.mixins.validation import FurnitureValidationMixin

console_logger = logging.getLogger(__name__)


class FurnitureTools(
    FurnitureValidationMixin,
    FurniturePlacementMixin,
    FurnitureAssetOperationsMixin,
):
    """
    Agent-callable tools for furniture asset generation and placement in 3D scenes.

    Provides a two-phase workflow for the designer agent:
    1. Asset Generation: Creates 3D furniture from text descriptions via the text-to-3D
       pipeline (GPT images → Hunyuan3D geometry → Drake SDF)
    2. Scene Operations: Places, moves, and removes furniture using generated assets

    Tools exposed:
    - generate_assets: Batch generate 3D furniture from descriptions
    - add_furniture_to_scene_tool: Place furniture at specific coordinates
    - move_furniture_tool: Reposition existing furniture
    - remove_furniture_tool: Delete furniture from scene
    """

    def __init__(self, scene: RoomScene, asset_manager: AssetManager, cfg: DictConfig):
        """Initialize furniture tools.

        Args:
            scene: RoomScene instance to manipulate.
            asset_manager: Asset manager for generating 3D assets.
            cfg: Configuration object containing loop detection settings.
        """
        self.scene = scene
        self.asset_manager = asset_manager
        self.cfg = cfg
        self._structural_surface_index = None

        # Initialize placement noise configuration.
        # Start with natural profile as default until planner sets it.
        self.placement_noise_config = cfg.placement_noise
        self.active_noise_profile = self.placement_noise_config.natural_profile

        # Initialize loop detector from config.
        loop_config = cfg.loop_detection
        loop_detector = LoopDetector(
            max_attempts=loop_config.max_repeated_attempts,
            window_size=loop_config.tracking_window,
            enabled=loop_config.enabled,
            default_error_factory=self._create_loop_error_response,
        )

        # Apply loop detection to implementation methods.
        self._add_furniture_to_scene_impl = loop_detector(
            self._add_furniture_to_scene_impl
        )
        self._move_furniture_impl = loop_detector(self._move_furniture_impl)
        self._remove_furniture_impl = loop_detector(self._remove_furniture_impl)

        # Create tool closures that use the protected methods.
        self.tools = self._create_tool_closures()

    def _placement_collisions_for(self, *args, **kwargs):
        """Preserve the historic patch point for collision computation."""

        _validation_mixin.compute_scene_collisions = compute_scene_collisions
        return super()._placement_collisions_for(*args, **kwargs)

    def set_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Update the active noise profile based on placement style.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        if mode == PlacementNoiseMode.NATURAL:
            self.active_noise_profile = self.placement_noise_config.natural_profile
            console_logger.info("Placement noise set to NATURAL profile")
        elif mode == PlacementNoiseMode.PERFECT:
            self.active_noise_profile = self.placement_noise_config.perfect_profile
            console_logger.info("Placement noise set to PERFECT profile")
        else:
            console_logger.warning(
                f"Unsupported noise mode {mode}, keeping current profile"
            )

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create closure-based tools that capture self."""

        @function_tool
        def generate_assets(
            object_descriptions: list[str],
            short_names: list[str],
            desired_dimensions: list[list[float]],
            style_context: str | None = None,
        ) -> str:
            """Create 3D furniture models from descriptions with specified dimensions.

            Generate floor-standing furniture items only. This tool is restricted
            to furniture that sits flat on the floor.

            DO NOT generate:
            - Manipulands (small objects meant for surfaces like books, vases, cups)
            - Carpets or rugs
            - Wall decorations
            - Architecture (stairs, ramps, ladders, platforms, mezzanines, railings,
              balconies, or bridges). The floor-plan stage already owns and compiles
              those structures; never duplicate them as furniture.

            ONLY generate furniture items that rest directly on the floor.

            You MUST specify dimensions for each object considering the
            relative sizes of other objects in the scene. Use realistic furniture
            proportions.

            Args:
                object_descriptions: List of furniture descriptions to generate
                    (e.g., "Modern oak dining table", "Leather office chair").
                short_names: List of short filesystem-safe names corresponding to
                    each description (e.g., "dining_table", "office_chair").
                desired_dimensions: List of [width, depth, height] in meters for each
                    object. Width (X-axis), depth (Y-axis), and height (Z-axis) specify
                    the object's dimensions in the room coordinate system. Width is
                    left-right, depth is front-back, height is up-down. Predict
                    dimensions considering other objects in the scene.
                    Example: [[1.8, 0.9, 0.75], [0.5, 0.5, 0.9]] for table and chair.
                style_context: Optional style context for visual consistency
                    (e.g., "modern minimalist living room").

            Returns:
                IDs and details of the created furniture models.
            """
            console_logger.info("Tool called: generate_assets")
            console_logger.info(
                f"Generating batch of {len(object_descriptions)} assets: "
                f"{object_descriptions}"
            )
            request = AssetGenerationRequest(
                object_descriptions=object_descriptions,
                short_names=short_names,
                object_type=ObjectType.FURNITURE,
                desired_dimensions=desired_dimensions,
                style_context=style_context,
                scene_id=self.scene.scene_dir.name,
            )
            return self._generate_assets_impl(request)

        @function_tool
        def add_furniture_to_scene_tool(
            asset_id: str,
            x: float,
            y: float,
            yaw: float = 0.0,
            elevation: float = 0.0,
        ) -> str:
            """Place furniture in the room at a specific floor position.

            Furniture sits flat on the structural floor nearest the requested
            elevation, with upright orientation. Use elevation=0 for the ground
            floor or the level's floor height for upper stories.

            Each placement gets a unique ID so you can move or remove it later.
            The same furniture model can be placed multiple times.

            Use 'list_available_assets' to see what furniture you can place.

            Args:
                asset_id: ID of the furniture to place.
                x: X position in the room (meters).
                y: Y position in the room (meters).
                yaw: Yaw rotation in degrees around vertical axis (default: 0.0).
                    Positive values rotate counterclockwise in top-down view.
                elevation: Structural floor elevation in meters (default: 0.0).

            Returns:
                The unique ID for this placement and confirmation of success.
            """
            return self._add_furniture_to_scene_impl(
                asset_id=asset_id,
                x=x,
                y=y,
                z=elevation,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )

        @function_tool
        def move_furniture_tool(
            object_id: str,
            x: float,
            y: float,
            yaw: float = 0.0,
            elevation: float = 0.0,
        ) -> str:
            """Move existing furniture to a new floor position.

            Furniture sits flat on the structural floor nearest the requested
            elevation, with upright orientation. Use elevation=0 for the ground
            floor or the level's floor height for upper stories.

            Use this to relocate furniture that's already in the room. You need
            the object ID from when you placed it or from 'get_current_scene_state'.

            Args:
                object_id: ID of the furniture item to move.
                x: New X position in the room (meters).
                y: New Y position in the room (meters).
                yaw: New yaw rotation in degrees around vertical axis (default: 0.0).
                    Positive values rotate counterclockwise in top-down view.
                elevation: Target structural floor elevation in meters (default: 0.0).

            Returns:
                Confirmation that the furniture was moved successfully.
            """
            return self._move_furniture_impl(
                object_id=object_id,
                x=x,
                y=y,
                z=elevation,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )

        @function_tool
        def remove_furniture_tool(object_id: str) -> str:
            """Remove furniture from the room.

            Use this to delete furniture you no longer want. You need the object ID
            from when you placed it or from 'get_current_scene_state'.

            Args:
                object_id: ID of the furniture item to remove.

            Returns:
                Confirmation that the furniture was removed successfully.
            """
            return self._remove_furniture_impl(object_id)

        @function_tool
        def list_available_assets() -> str:
            """See all furniture models you can place with their dimensions.

            This shows you all the furniture that's available for placing in the
            room, including precise dimensions (width, depth, height) to help with
            spatial planning. Use the IDs from this list with 'add_furniture_to_scene_tool'
            to actually place items. You can place the same model multiple times.

            Returns:
                List of furniture with their IDs, names, descriptions, and dimensions.
            """
            return self._list_available_assets_impl()

        @function_tool
        def rescale_furniture_tool(object_id: str, scale_factor: float) -> str:
            """Resize furniture by a uniform scale factor.

            IMPORTANT: This rescales the underlying ASSET. All instances of the same
            asset (e.g., all 4 dining chairs) will be affected. This is usually what
            you want - if one chair is too small, they all are.

            Use this when proportions are correct but size is wrong.
            For shape/proportion issues, regenerate the asset instead.

            Args:
                object_id: ID of the furniture item to rescale.
                scale_factor: Scale multiplier (e.g., 1.5 = 50% larger, 0.8 = 20% smaller).

            Returns:
                Result with new dimensions and list of affected objects.
            """
            return self._rescale_furniture_impl(object_id, scale_factor)

        return {
            "generate_assets": generate_assets,
            "add_furniture_to_scene_tool": add_furniture_to_scene_tool,
            "move_furniture_tool": move_furniture_tool,
            "remove_furniture_tool": remove_furniture_tool,
            "rescale_furniture_tool": rescale_furniture_tool,
            "list_available_assets": list_available_assets,
        }
