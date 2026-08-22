import json
import logging

from typing import Any

from agents import function_tool
from omegaconf import DictConfig

from scenesmith.agent_utils.assets.asset_manager import AssetManager
from scenesmith.agent_utils.assets.asset_models import AssetGenerationRequest
from scenesmith.agent_utils.core.loop_detector import LoopDetector
from scenesmith.agent_utils.design.placement_noise import (
    PlacementNoiseMode,
    apply_placement_noise,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SupportSurface,
    UniqueID,
)
from scenesmith.manipuland_agents.tools.manipuland_tool_models import FillAssetItem
from scenesmith.manipuland_agents.tools.mixins import placement as _placement_mixin
from scenesmith.manipuland_agents.tools.mixins.operations import (
    ManipulandOperationsMixin,
)
from scenesmith.manipuland_agents.tools.mixins.placement import ManipulandPlacementMixin
from scenesmith.manipuland_agents.tools.mixins.resolution import (
    ManipulandResolutionMixin,
)
from scenesmith.manipuland_agents.tools.mixins.validation import (
    ManipulandValidationMixin,
)

console_logger = logging.getLogger(__name__)


class ManipulandTools(
    ManipulandValidationMixin,
    ManipulandPlacementMixin,
    ManipulandOperationsMixin,
    ManipulandResolutionMixin,
):
    """Agent-callable tools for manipuland asset generation and placement.

    Provides tools for the manipuland designer agent:
    1. Asset Generation: Creates 3D manipulands from text descriptions
    2. Surface Placement: Places manipulands on support surfaces using SE(2) poses
    3. Scene Operations: Removes manipulands, queries scene state

    Tools exposed:
    - generate_manipuland_assets: Generate 3D assets via text-to-3D pipeline
    - place_manipuland_on_surface: Place manipuland on surface with SE(2) pose
    - remove_manipuland: Delete manipuland from scene
    - get_current_scene_state: Get furniture + manipulands for current surface
    - list_available_assets: List all available manipuland assets
    """

    def __init__(
        self,
        scene: RoomScene,
        asset_manager: AssetManager,
        cfg: DictConfig,
        current_furniture_id: UniqueID,
        support_surfaces: dict[str, SupportSurface],
    ):
        """Initialize manipuland tools.

        Args:
            scene: RoomScene instance to manipulate.
            asset_manager: Asset manager for generating 3D assets.
            cfg: Configuration object containing loop detection and validation settings.
            current_furniture_id: ID of furniture currently being populated.
            support_surfaces: Dictionary mapping surface_id (string) to SupportSurface.
                All surfaces for the current furniture item.
        """
        self.scene = scene
        self.asset_manager = asset_manager
        self.cfg = cfg
        self.current_furniture_id = current_furniture_id
        self.support_surfaces = support_surfaces

        # Initialize placement noise configuration.
        # Start with natural profile as default until planner sets it.
        self.placement_noise_config = cfg.placement_noise
        self.active_noise_profile = self.placement_noise_config.natural_profile

        # Initialize placement validation configuration.
        self.top_surface_overlap_tolerance = (
            cfg.placement_validation.top_surface_overlap_tolerance
        )

        # Initialize loop detector from config.
        loop_config = cfg.loop_detection
        loop_detector = LoopDetector(
            max_attempts=loop_config.max_repeated_attempts,
            window_size=loop_config.tracking_window,
            enabled=loop_config.enabled,
            default_error_factory=self._create_loop_error_response,
        )

        # Apply loop detection to implementation methods.
        self._place_manipuland_on_surface_impl = loop_detector(
            self._place_manipuland_on_surface_impl
        )
        self._move_manipuland_impl = loop_detector(self._move_manipuland_impl)
        self._remove_manipuland_impl = loop_detector(self._remove_manipuland_impl)
        self._create_stack_impl = loop_detector(self._create_stack_impl)
        self._create_pile_impl = loop_detector(self._create_pile_impl)
        self._resolve_penetrations_impl = loop_detector(self._resolve_penetrations_impl)

        # Create tool closures.
        self.tools = self._create_tool_closures()

    def _place_manipuland_on_surface_impl(self, *args, **kwargs):
        """Preserve the historic placement-noise patch point."""

        _placement_mixin.apply_placement_noise = apply_placement_noise
        return super()._place_manipuland_on_surface_impl(*args, **kwargs)

    def _move_manipuland_impl(self, *args, **kwargs):
        """Preserve the historic movement-noise patch point."""

        _placement_mixin.apply_placement_noise = apply_placement_noise
        return super()._move_manipuland_impl(*args, **kwargs)

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
        """Create tool closures that capture current furniture/surface context."""

        @function_tool
        def generate_manipuland_assets(
            object_descriptions: list[str],
            short_names: list[str],
            desired_dimensions: list[list[float]],
            style_context: str | None = None,
        ) -> str:
            """Generate 3D manipuland assets from text descriptions.

            Creates small objects like lamps, books, decorations, kitchenware, etc.
            Each object goes through: text → image → 3D geometry → Drake SDF.

            Args:
                object_descriptions: List of manipuland descriptions
                    (e.g., "Ceramic coffee mug", "Hardcover book").
                short_names: List of filesystem-safe names
                    (e.g., "coffee_mug", "book").
                desired_dimensions: List of [width, depth, height] in meters.
                    Manipulands are typically smaller: 0.05-0.3m range.
                style_context: Optional style context for visual consistency
                    (e.g., "modern kitchen", "cozy bedroom").

            Returns:
                IDs and details of created manipuland models.
            """
            console_logger.info("Tool called: generate_manipuland_assets")
            request = AssetGenerationRequest(
                object_descriptions=object_descriptions,
                short_names=short_names,
                object_type=ObjectType.MANIPULAND,
                desired_dimensions=desired_dimensions,
                style_context=style_context,
                scene_id=self.scene.scene_dir.name,
            )
            return self._generate_assets_impl(request)

        @function_tool
        def list_support_surfaces() -> str:
            """List all support surfaces available on the current furniture.

            Returns:
                JSON string with list of surfaces, each containing:
                - surface_id: Unique identifier for the surface
                - area_m2: Surface area in square meters
                - height_m: Approximate height in meters (Z coordinate of surface)
                - clearance_height: Vertical clearance in meters (max object height)
            """
            surfaces_info = []
            for surface_id_str, surface in self.support_surfaces.items():
                # Get approximate height from transform.
                height = surface.transform.translation()[2]
                surfaces_info.append(
                    {
                        "surface_id": surface_id_str,
                        "area_m2": round(surface.area, 4),
                        "height_m": round(height, 3),
                        "clearance_height": round(
                            float(
                                surface.bounding_box_max[2]
                                - surface.bounding_box_min[2]
                            ),
                            3,
                        ),
                    }
                )

            # Sort by height (top to bottom).
            surfaces_info.sort(key=lambda s: s["height_m"], reverse=True)

            result = {
                "furniture_id": str(self.current_furniture_id),
                "num_surfaces": len(surfaces_info),
                "surfaces": surfaces_info,
            }

            return json.dumps(result, indent=2)

        @function_tool
        def place_manipuland_on_surface(
            asset_id: str,
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Place manipuland on a specific support surface.

            The manipuland is placed using 2D coordinates on the surface plane.
            The system automatically converts to 3D world coordinates.

            Each placement gets a unique ID so you can move or remove it later.
            The same manipuland can be placed multiple times.

            Coordinate system:
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center
            - Rotation: degrees around surface normal (Z-axis)

            Args:
                asset_id: ID of the manipuland asset to place.
                surface_id: ID of the support surface to place on.
                    Use list_support_surfaces() to see available surfaces.
                position_x: X position on surface (meters, left-right).
                position_z: Z position on surface (meters, front-back).
                rotation_degrees: Rotation around surface normal (degrees).
                    Positive values rotate counterclockwise when viewed from above.

            Returns:
                Placement result with world pose and surface-relative pose.
            """
            return self._place_manipuland_on_surface_impl(
                asset_id=asset_id,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def move_manipuland(
            object_id: str,
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Move existing manipuland to a new position on a support surface.

            Use this to reposition manipulands or move them between surfaces.
            You need the object ID from when you placed it or from
            'get_current_scene_state'.

            Coordinate system same as placement:
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters, front-back)
            - Origin (0, 0) is at surface center
            - Rotation: degrees around surface normal (Z-axis)

            Args:
                object_id: ID of the manipuland to move.
                surface_id: ID of the target support surface.
                    Can be same or different from current surface.
                position_x: New X position on surface (meters).
                position_z: New Y position on surface (meters, front-back).
                rotation_degrees: New rotation around surface normal (degrees).
                    Positive values rotate counterclockwise when viewed from above.

            Returns:
                Result of the move operation with new world pose.
            """
            return self._move_manipuland_impl(
                object_id=object_id,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def remove_manipuland(object_id: str) -> str:
            """Remove a manipuland from the scene.

            Args:
                object_id: ID of the manipuland to remove.

            Returns:
                Result of the removal operation.
            """
            return self._remove_manipuland_impl(
                object_id=object_id,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                },
            )

        @function_tool
        def get_current_scene_state() -> str:
            """Get current scene state filtered to current furniture + manipulands.

            Shows:
            - Current furniture being populated (with dimensions)
            - Manipulands already placed on this furniture (with dimensions)
            - Current support surface bounds and clearance_height

            Does NOT show:
            - Other furniture in the scene
            - Manipulands on other furniture

            Returns:
                Scene state with furniture, manipulands, and surface info including:
                - surfaces[].clearance_height: Max object height that fits (meters)
                - manipulands[].dimensions: Object size (width, depth, height)
            """
            return self._get_current_scene_state_impl()

        @function_tool
        def list_available_assets() -> str:
            """List all available manipuland assets (from all furniture).

            This includes manipulands generated for previous furniture, enabling
            asset reuse (e.g., same plate on multiple tables).

            Returns:
                List of all available manipuland assets with IDs and descriptions.
            """
            return self._list_available_assets_impl()

        @function_tool
        def create_stack(
            asset_ids: list[str],
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Create a vertical stack of objects on a support surface.

            Stacks objects bottom-to-top. Creates a single composite object that can be
            moved or removed as a unit.

            Use cases:
            - Stack of plates: ["plate_0", "plate_0", "plate_0"]
            - Stack of books: ["book_red", "book_blue", "book_green"]
            - Mixed items: ["plate_0", "bowl_0", "cup_0"]

            Coordinate system (same as place_manipuland_on_surface):
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center

            Args:
                asset_ids: List of asset IDs to stack (bottom to top). Must have
                    at least 2 items. Use same ID multiple times for identical
                    objects. Use list_available_assets() to see available IDs.
                surface_id: ID of the support surface to place stack on.
                    Use list_support_surfaces() to see available surfaces.
                position_x: X position of stack base on surface (meters, left-right).
                position_z: Z position of stack base on surface (meters, front-back).
                rotation_degrees: Rotation around surface normal (degrees).
                    Applied to entire stack.

            Returns:
                StackCreationResult with composite object ID and height.
                On failure, includes actionable feedback in message.
            """
            return self._create_stack_impl(
                asset_ids=asset_ids,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def fill_container(
            container_asset_id: str,
            fill_asset_ids: list[str],
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Fill a container with objects.

            Places a container (bowl, basket, pen holder) at the specified position
            and fills it with objects.

            Use cases:
            - Fruit bowl with apples and oranges
            - Pen holder with pens and pencils
            - Breadbasket with rolls
            - Toy bin with toys

            Coordinate system (same as place_manipuland_on_surface):
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center

            Args:
                container_asset_id: ID of the container asset (bowl, basket, etc.).
                fill_asset_ids: List of asset IDs to put inside container.
                    Must have at least 1 item. Can use same ID multiple times.
                surface_id: ID of the support surface to place container on.
                position_x: X position of container on surface (meters, left-right).
                position_z: Z position of container on surface (meters, front-back).
                rotation_degrees: Container rotation around surface normal (degrees).

            Returns:
                FillContainerResult with composite object ID and fill count.
                On failure, includes actionable feedback in message.
            """
            return self._fill_container_impl(
                container_asset_id=container_asset_id,
                fill_asset_ids=fill_asset_ids,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def create_arrangement(
            container_asset_id: str,
            fill_assets: list[FillAssetItem],
            surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Place items at specified positions on a flat container (tray, platter, board).

            Unlike fill_container (random positions inside cavity), this places items
            at your exact x,y coordinates on a flat surface. Fails if items collide
            or fall off.

            Two coordinate systems:
            1. Furniture surface (position_x, position_z, rotation_degrees):
               Same as place_manipuland.

            2. Container local (fill_assets x, y, rotation):
               Origin at container center, in meters.
               +X = right, -X = left (when facing container front)
               +Y = front (near edge), -Y = back (far edge)
               Positions rotate with the container.

            Args:
                container_asset_id: Flat container asset ID.
                fill_assets: List of FillAssetItem with id, x, y, rotation.
                    All fields required. x/y in meters from container center.
                    rotation in degrees (use 0 if no rotation needed).
                surface_id: Furniture surface ID.
                position_x: Container X on surface (meters).
                position_z: Container Z on surface (meters).
                rotation_degrees: Container rotation (degrees).

            Returns:
                FillContainerResult. On failure: container bounds + feedback.
            """
            return self._create_arrangement_impl(
                container_asset_id=container_asset_id,
                fill_assets=fill_assets,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def create_pile(
            asset_ids: list[str], surface_id: str, position_x: float, position_z: float
        ) -> str:
            """Create a random pile of objects on a support surface.

            Drops objects in a random cluster and lets physics settle them into a
            natural, messy arrangement. Creates a composite that moves as a unit.

            Use cases:
            - Toys on floor: ["block_0", "block_0", "toy_car_1"] in kid's room
            - Dirty dishes in sink: ["plate_0", "mug_1", "bowl_2"] in built-in sink
            - Firewood by fireplace: ["log_0", "log_0", "log_0"] on hearth
            - Papers on desk: ["paper_0", "paper_0", "folder_1"] messily stacked
            - Laundry on floor: ["shirt_0", "pants_1", "sock_2"] in messy pile

            NOT for (use other tools instead):
            - Neat table settings → place items individually with place_manipuland
            - Stacked plates/bowls → use create_stack (neat, aligned stacking)
            - Glasses/cups/mugs → place individually (must stand upright!)
            - Any arrangement meant to look tidy or organized

            Pile vs Fill distinction:
            - fill_container: Container is a SEPARATE manipuland (bowl, basket, vase)
            - create_pile: Objects dropped on surface OR into BUILT-IN container (sink)

            Coordinate system (same as place_manipuland_on_surface):
            - X: left-right on surface (meters)
            - Y: front-back on surface (meters)
            - Origin (0, 0) is at surface center

            Args:
                asset_ids: List of asset IDs to pile (minimum 2 items).
                    Use same ID multiple times for identical objects.
                    Use list_available_assets() to see available IDs.
                surface_id: ID of the support surface to place pile on.
                    Use list_support_surfaces() to see available surfaces.
                position_x: X position of pile center on surface (meters, left-right).
                position_z: Z position of pile center on surface (meters, front-back).

            Returns:
                PileCreationResult with composite object ID and pile count.
                On failure, includes actionable feedback in message.
            """
            return self._create_pile_impl(
                asset_ids=asset_ids,
                surface_id=surface_id,
                position_x=position_x,
                position_z=position_z,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                    "surface_id": surface_id,
                },
            )

        @function_tool
        def rescale_manipuland(object_id: str, scale_factor: float) -> str:
            """Resize manipuland by a uniform scale factor.

            IMPORTANT: This rescales the underlying ASSET. All instances of the same
            asset will be affected. This is usually what you want - if one instance
            is too small, they all are.

            NOTE: Composite objects CANNOT be rescaled. This includes:
            - Stacks (created by create_stack)
            - Piles (created by create_pile)
            - Filled containers (created by fill_container)
            To resize items in a composite, remove the composite and recreate it with
            rescaled individual assets.

            Use this when proportions are correct but size is wrong.
            For shape/proportion issues, regenerate the asset instead.

            Args:
                object_id: ID of the manipuland to rescale.
                scale_factor: Scale multiplier (e.g., 1.5 = 50% larger, 0.8 = 20% smaller).

            Returns:
                Result with new dimensions and list of affected objects.
            """
            return self._rescale_manipuland_impl(
                object_id=object_id,
                scale_factor=scale_factor,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                },
            )

        @function_tool
        def resolve_penetrations(object_ids: list[str]) -> str:
            """Resolve collisions between specified objects on a surface.

            Spreads overlapping objects apart while keeping them on the surface.
            All orientations are preserved exactly.

            The surface is inferred from the objects - all objects must be on the
            same surface. Fails if objects are on different surfaces.

            IMPORTANT: This is a LAST RESORT tool. Always prefer manual placement
            with calculated positions that avoid overlaps. Only use this when:
            - Space is genuinely too tight for manual collision avoidance
            - You need many objects in a small area (e.g., 10 bottles on narrow shelf)
            - Multiple stacks must be grouped closer than their footprints allow

            WARNING: Objects may end up at positions you didn't intend. The solver
            spreads objects minimally, but you lose precise control over final XY
            positions. Orientations are always preserved.

            Args:
                object_ids: List of object IDs to resolve. These objects will be
                    spread apart if they overlap. All objects must be on the same
                    surface.

            Returns:
                PenetrationResolutionResult with list of moved objects and
                displacement magnitudes.
            """
            return self._resolve_penetrations_impl(
                object_ids=object_ids,
                _action_metadata={
                    "furniture_id": str(self.current_furniture_id),
                },
            )

        return {
            "list_support_surfaces": list_support_surfaces,
            "generate_manipuland_assets": generate_manipuland_assets,
            "place_manipuland_on_surface": place_manipuland_on_surface,
            "move_manipuland": move_manipuland,
            "remove_manipuland": remove_manipuland,
            "rescale_manipuland": rescale_manipuland,
            "get_current_scene_state": get_current_scene_state,
            "list_available_assets": list_available_assets,
            "create_stack": create_stack,
            "fill_container": fill_container,
            "create_arrangement": create_arrangement,
            "create_pile": create_pile,
            # "resolve_penetrations": resolve_penetrations,  # Disabled: 0% success rate in experiments
        }
