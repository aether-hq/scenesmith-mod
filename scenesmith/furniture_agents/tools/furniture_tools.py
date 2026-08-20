import copy
import json
import logging
import math
import os
import time

from typing import Any

import numpy as np

from agents import function_tool
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix

from scenesmith.agent_utils.action_logger import log_scene_action
from scenesmith.agent_utils.asset_manager import (
    AssetGenerationRequest,
    AssetGenerationResult as DomainAssetGenerationResult,
    AssetManager,
)
from scenesmith.agent_utils.contextual_solver import validate_scene_object_placement
from scenesmith.agent_utils.loop_detector import LoopDetector
from scenesmith.agent_utils.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.placement_noise import (
    PlacementNoiseMode,
    apply_placement_noise,
)
from scenesmith.agent_utils.rescale_helpers import rescale_object_common
from scenesmith.agent_utils.rescale_result import RescaleErrorType, RescaleResult
from scenesmith.agent_utils.response_datatypes import (
    AssetGenerationResult,
    GeneratedAsset,
)
from scenesmith.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    UniqueID,
    copy_scene_object_with_new_pose,
)
from scenesmith.furniture_agents.tools.response_dataclasses import (
    AssetInfo,
    AvailableAssetsResult,
    FurnitureErrorType,
    FurnitureOperationResult,
    FurniturePlacementResult,
    Position3D,
    Rotation3D,
)

console_logger = logging.getLogger(__name__)


class FurnitureTools:
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

    def _check_floor_bounds(self, x: float, y: float) -> tuple[bool, str]:
        """Check if position (center point) is within floor plan bounds.

        Args:
            x: X coordinate in meters.
            y: Y coordinate in meters.

        Returns:
            (is_valid, error_message) - error_message is empty string if valid.
        """
        room_geometry = self.scene.room_geometry

        surface_index = self._get_structural_surface_index()
        if surface_index is not None:
            pose = surface_index.support_pose(x, y)
            if pose is None:
                return (
                    False,
                    f"Position ({x:.3f}, {y:.3f}) is not on a traversable "
                    "structural support surface (it may be outside the room, "
                    "inside a hole/shaft, or on an overly steep patch).",
                )
            return True, ""

        # Floor bounds: [-length/2, length/2] × [-width/2, width/2].
        min_x = -room_geometry.length / 2
        max_x = room_geometry.length / 2
        min_y = -room_geometry.width / 2
        max_y = room_geometry.width / 2

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            error_msg = (
                f"Position ({x:.3f}, {y:.3f}) is out of floor plan bounds. "
                f"Valid bounds: X=[{min_x:.3f}, {max_x:.3f}], "
                f"Y=[{min_y:.3f}, {max_y:.3f}]"
            )
            return False, error_msg

        return True, ""

    def _get_structural_surface_index(self):
        """Lazily load compiled surface patches for irregular geometry."""

        if self._structural_surface_index is False:
            return None
        if self._structural_surface_index is not None:
            return self._structural_surface_index
        room_geometry = self.scene.room_geometry
        additional = getattr(room_geometry, "additional_structural_surface_paths", ())
        if not isinstance(additional, (list, tuple, set, frozenset)):
            additional = ()
        candidates = (
            getattr(room_geometry, "structural_surface_path", None),
            *additional,
        )
        sidecar_paths = [
            path
            for candidate in candidates
            if isinstance(candidate, (str, os.PathLike))
            and (path := os.fspath(candidate))
            and os.path.isfile(path)
        ]
        if not sidecar_paths:
            self._structural_surface_index = False
            return None
        from scenesmith.agent_utils.structural_surfaces import (
            StructuralSurfaceIndex,
            load_surface_patches,
        )

        self._structural_surface_index = StructuralSurfaceIndex(
            patch
            for sidecar_path in sidecar_paths
            for patch in load_surface_patches(sidecar_path)
        )
        return self._structural_surface_index

    def _surface_aligned_pose(
        self, x: float, y: float, yaw_degrees: float
    ) -> tuple[float, float, float, float] | None:
        """Return z/roll/pitch/yaw aligned to the support surface, if present."""

        surface_index = self._get_structural_surface_index()
        if surface_index is None:
            return None
        pose = surface_index.support_pose(x, y, yaw=math.radians(yaw_degrees))
        if pose is None:
            return None
        rotation = RotationMatrix(
            np.column_stack((pose.tangent_x, pose.tangent_y, pose.normal))
        )
        rpy = RollPitchYaw(rotation)
        return (
            pose.position[2],
            math.degrees(rpy.roll_angle()),
            math.degrees(rpy.pitch_angle()),
            math.degrees(rpy.yaw_angle()),
        )

    def _validate_spatial_envelope(self, scene_object: SceneObject) -> tuple[bool, str]:
        """Require an object's complete AABB to fit its support and enclosure.

        Structural overhead patches are authoritative when present: a missing
        overhead patch means that XY location is open air. Legacy rectangular
        rooms use ``has_overhead_cover`` and ``wall_height``.
        """
        try:
            bounds = scene_object.compute_world_bounds()
        except (TypeError, ValueError, IndexError):
            # Compatibility for legacy/test registry entries whose bbox fields
            # predate concrete numeric bounds.
            return True, ""
        if (
            bounds is None
            or not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or not all(isinstance(item, np.ndarray) for item in bounds)
        ):
            # Older hand-authored test assets may not carry bounds. Production
            # catalog/generated assets do, and are validated here.
            return True, ""
        world_min, world_max = bounds
        center_x = float((world_min[0] + world_max[0]) / 2.0)
        center_y = float((world_min[1] + world_max[1]) / 2.0)
        footprint_samples = (
            (center_x, center_y),
            (float(world_min[0]), float(world_min[1])),
            (float(world_min[0]), float(world_max[1])),
            (float(world_max[0]), float(world_min[1])),
            (float(world_max[0]), float(world_max[1])),
        )
        epsilon = 1e-4
        surface_index = self._get_structural_surface_index()
        if surface_index is not None:
            reference_z = float(world_min[2]) + 0.10
            for sample_x, sample_y in footprint_samples:
                support = surface_index.support_pose(
                    sample_x,
                    sample_y,
                    reference_z=reference_z,
                    max_drop=0.20,
                )
                if support is None:
                    return (
                        False,
                        "Furniture footprint extends beyond its traversable support "
                        f"near ({sample_x:.3f}, {sample_y:.3f}).",
                    )

            center_overhead = surface_index.overhead_pose(
                center_x, center_y, reference_z=float(world_min[2])
            )
            if center_overhead is None:
                return True, ""

            overhead_heights: list[float] = []
            for sample_x, sample_y in footprint_samples:
                overhead = surface_index.overhead_pose(
                    sample_x, sample_y, reference_z=float(world_min[2])
                )
                if overhead is None:
                    return (
                        False,
                        "Furniture footprint straddles the edge of a covered area; "
                        "move it fully under the roof or fully into open air.",
                    )
                overhead_heights.append(float(overhead.position[2]))
            lowest_overhead = min(overhead_heights)
            if float(world_max[2]) > lowest_overhead + epsilon:
                return (
                    False,
                    "Furniture exceeds the available overhead clearance: object top "
                    f"is {float(world_max[2]):.3f}m, overhead is "
                    f"{lowest_overhead:.3f}m.",
                )
            return True, ""

        room_geometry = self.scene.room_geometry
        half_length = float(room_geometry.length) / 2.0
        half_width = float(room_geometry.width) / 2.0
        if (
            float(world_min[0]) < -half_length - epsilon
            or float(world_max[0]) > half_length + epsilon
            or float(world_min[1]) < -half_width - epsilon
            or float(world_max[1]) > half_width + epsilon
        ):
            return (
                False,
                "Furniture footprint exceeds the room floor bounds: "
                f"object X=[{world_min[0]:.3f}, {world_max[0]:.3f}], "
                f"Y=[{world_min[1]:.3f}, {world_max[1]:.3f}].",
            )

        has_overhead = bool(getattr(room_geometry, "has_overhead_cover", True))
        if has_overhead:
            overhead_height = float(room_geometry.wall_height)
            if float(world_max[2]) > overhead_height + epsilon:
                return (
                    False,
                    "Furniture exceeds the available overhead clearance: object top "
                    f"is {float(world_max[2]):.3f}m, overhead is "
                    f"{overhead_height:.3f}m.",
                )
        return True, ""

    def _furniture_collisions_for(self, object_id: UniqueID) -> list[str]:
        """Return concrete furniture collisions involving one proposed pose."""

        furniture = self.scene.get_objects_by_type(ObjectType.FURNITURE)
        if not isinstance(furniture, list) or len(furniture) < 2:
            return []
        physics_cfg = self.cfg.physics_validation
        collisions = compute_scene_collisions(
            scene=self.scene,
            penetration_threshold=physics_cfg.object_penetration_threshold_m,
            floor_penetration_tolerance=physics_cfg.floor_penetration_tolerance_m,
            manipuland_furniture_tolerance_m=(
                physics_cfg.manipuland_furniture_tolerance_m
            ),
        )
        current = str(object_id)
        descriptions: list[str] = []
        for collision in collisions:
            ids = {collision.object_a_id, collision.object_b_id}
            if current not in ids:
                continue
            other_id = next((item for item in ids if item != current), None)
            if other_id is None:
                continue
            try:
                other = self.scene.get_object(UniqueID(other_id))
            except Exception:
                other = None
            if other is None or other.object_type != ObjectType.FURNITURE:
                continue
            descriptions.append(collision.to_description())
        return descriptions

    def _validate_contextual_zones(self, scene_object: SceneObject) -> tuple[bool, str]:
        """Run the fast semantic solver before invoking heavyweight physics."""

        furniture = self.scene.get_objects_by_type(ObjectType.FURNITURE)
        if not isinstance(furniture, (list, tuple)):
            # Compatibility for legacy tests and partially restored scenes. The
            # production RoomScene API always returns a concrete list.
            furniture = []
        existing = [
            item for item in furniture if item.object_id != scene_object.object_id
        ]
        result = validate_scene_object_placement(scene_object, existing)
        hard = [
            violation for violation in result.violations if violation.severity == "hard"
        ]
        soft = [
            violation for violation in result.violations if violation.severity == "soft"
        ]
        for violation in soft:
            console_logger.info(
                "Contextual placement advisory %s: %s",
                violation.code,
                violation.message,
            )
        if not hard:
            return True, ""
        payload = {
            "message": "Contextual placement constraints rejected this pose",
            "violations": [violation.to_dict() for violation in hard],
        }
        return False, json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def _create_loop_error_response(
        self, method_name: str, attempt_count: int, args: tuple, kwargs: dict
    ) -> str:
        """Create furniture-specific error response for loop detection."""
        # Extract object_id from kwargs or args if available.
        object_id = kwargs.get("object_id", "")
        if not object_id and args and len(args) > 1:
            object_id = str(args[1])  # First arg after self

        # Create context-specific diagnostic message.
        if method_name == "_remove_furniture_impl":
            base_name = object_id.rsplit("_", 1)[0] if "_" in object_id else object_id
            diagnostic_message = (
                f"Loop detected: You've tried to remove '{object_id}' {attempt_count} "
                f"times.\n\n"
                f"This means one of:\n"
                f"1. Wrong object name - missing ID postfix (e.g., using '{base_name}' "
                f"instead of '{base_name}_0', '{base_name}_1', etc.)\n"
                f"2. Object was already removed\n"
                f"3. Object doesn't exist with that ID\n\n"
                f"CRITICAL: ALL objects have sequential postfixes (_0, _1, _2, ...).\n"
                f"Base names without postfixes NEVER exist.\n\n"
                f"Recovery procedure (execute in order):\n"
                f"1. Call get_current_scene_state() to see current object IDs with "
                f"postfixes\n"
                f"2. Find objects whose names start with '{base_name}' "
                f"(e.g., '{base_name}_0', '{base_name}_1')\n"
                f"3. Use exact object_id from get_current_scene_state() including postfix\n"
                f"4. If object not in scene, report it was already removed\n\n"
                f"First call get_current_scene_state(), then retry with correct ID."
            )
            suggested_action = (
                "Call get_current_scene_state() to discover object IDs with postfixes"
            )
        elif method_name == "_move_furniture_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to move '{object_id}' {attempt_count} "
                f"times.\n\n"
                f"Causes:\n"
                f"1. Wrong object ID - IDs have sequential postfixes (_0, _1, _2, ...)\n"
                f"2. Object doesn't exist\n"
                f"3. Position/rotation causing collision or validation failure\n\n"
                f"CRITICAL: ALL objects have sequential postfixes. Base names without "
                f"postfixes NEVER exist.\n\n"
                f"Recovery procedure:\n"
                f"1. Call get_current_scene_state() to verify object exists with correct "
                f"ID\n"
                f"2. If collision issue, try different coordinates\n"
                f"3. Check for obstacles blocking the target position"
            )
            suggested_action = "Call get_current_scene_state() to verify object ID"
        else:
            diagnostic_message = (
                f"Loop detected: {attempt_count} identical calls to {method_name}"
            )
            suggested_action = (
                "Call get_current_scene_state() to refresh state, then try different "
                "approach"
            )

        return FurnitureOperationResult(
            success=False,
            message=diagnostic_message,
            object_id=object_id,
            error_type=FurnitureErrorType.LOOP_DETECTED,
            suggested_action=suggested_action,
        ).to_json()

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
        ) -> str:
            """Place furniture in the room at a specific floor position.

            Furniture sits flat on the floor at z=0 with upright orientation.
            You can only control the x, y position and yaw rotation (rotation
            around the vertical axis).

            Each placement gets a unique ID so you can move or remove it later.
            The same furniture model can be placed multiple times.

            Use 'list_available_assets' to see what furniture you can place.

            Args:
                asset_id: ID of the furniture to place.
                x: X position in the room (meters).
                y: Y position in the room (meters).
                yaw: Yaw rotation in degrees around vertical axis (default: 0.0).
                    Positive values rotate counterclockwise in top-down view.

            Returns:
                The unique ID for this placement and confirmation of success.
            """
            return self._add_furniture_to_scene_impl(
                asset_id=asset_id,
                x=x,
                y=y,
                z=0.0,
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
        ) -> str:
            """Move existing furniture to a new floor position.

            Furniture sits flat on the floor at z=0 with upright orientation.
            You can only control the x, y position and yaw rotation (rotation
            around the vertical axis).

            Use this to relocate furniture that's already in the room. You need
            the object ID from when you placed it or from 'get_current_scene_state'.

            Args:
                object_id: ID of the furniture item to move.
                x: New X position in the room (meters).
                y: New Y position in the room (meters).
                yaw: New yaw rotation in degrees around vertical axis (default: 0.0).
                    Positive values rotate counterclockwise in top-down view.

            Returns:
                Confirmation that the furniture was moved successfully.
            """
            return self._move_furniture_impl(
                object_id=object_id,
                x=x,
                y=y,
                z=0.0,
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

    @log_scene_action
    def _add_furniture_to_scene_impl(
        self,
        asset_id: str,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> str:
        """Implementation for placing an asset from the registry into the scene.

        Creates a new scene object instance with a unique object_id from the asset
        template.

        Rotations are in degrees.
        """
        console_logger.info("Tool called: add_furniture_to_scene_tool")
        try:
            console_logger.debug(f"Attempting to place asset: {asset_id}")

            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(asset_id)
            except Exception:
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=f"Invalid asset ID format: {asset_id}",
                    error_type=FurnitureErrorType.ASSET_NOT_FOUND,
                )

            # Get the asset from registry.
            original_asset = self.asset_manager.get_asset_by_id(unique_id)
            if not original_asset:
                available_assets = self.asset_manager.list_available_assets()
                available_ids = [str(a.object_id) for a in available_assets]
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=f"Asset {asset_id} not found in registry. "
                    f"Available: {available_ids}",
                    error_type=FurnitureErrorType.ASSET_NOT_FOUND,
                )

            console_logger.debug(
                f"Placing asset {asset_id} ({original_asset.name}) at position "
                f"({x}, {y}, {z}), rotation "
                f"({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)"
            )

            # Validate position is within floor plan bounds.
            is_valid, error_msg = self._check_floor_bounds(x=x, y=y)
            if not is_valid:
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=error_msg,
                    error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                )

            surface_pose = self._surface_aligned_pose(x, y, yaw)
            if surface_pose is not None:
                z, roll, pitch, yaw = surface_pose

            # Create new scene object with unique ID and specified pose.
            # Convert degrees to radians for Drake's RigidTransform.
            scene_object = copy_scene_object_with_new_pose(
                scene=self.scene,
                original=original_asset,
                x=x,
                y=y,
                z=z,
                roll=math.radians(roll),
                pitch=math.radians(pitch),
                yaw=math.radians(yaw),
            )

            # Apply placement noise for realistic variation.
            scene_object.transform = apply_placement_noise(
                transform=scene_object.transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )

            envelope_valid, envelope_message = self._validate_spatial_envelope(
                scene_object
            )
            if not envelope_valid:
                console_logger.warning("Placement rejected: %s", envelope_message)
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=envelope_message,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                )

            contextual_valid, contextual_message = self._validate_contextual_zones(
                scene_object
            )
            if not contextual_valid:
                console_logger.warning(
                    "Contextual placement rejected: %s", contextual_message
                )
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=contextual_message,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                )

            # Add to scene.
            self.scene.add_object(scene_object)

            placement_collisions = self._furniture_collisions_for(
                scene_object.object_id
            )
            if placement_collisions:
                self.scene.remove_object(scene_object.object_id)
                message = (
                    "Placement rejected because it intersects existing furniture: "
                    + "; ".join(placement_collisions)
                    + ". Move it farther away and retry."
                )
                console_logger.warning(message)
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=message,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                )

            # Log what changed.
            new_position = scene_object.transform.translation()
            new_rpy = RollPitchYaw(scene_object.transform.rotation())
            new_roll, new_pitch, new_yaw = (
                math.degrees(new_rpy.roll_angle()),
                math.degrees(new_rpy.pitch_angle()),
                math.degrees(new_rpy.yaw_angle()),
            )
            console_logger.info(
                f"Successfully placed asset '{original_asset.name}' as object "
                f"'{scene_object.object_id}' at position ({new_position[0]:.3f}, "
                f"{new_position[1]:.3f}, {new_position[2]:.3f}) and "
                f"rotation ({new_roll:.1f}°, {new_pitch:.1f}°, {new_yaw:.1f}°)"
            )

            return self._create_success_result(
                asset_id=asset_id, furniture_obj=scene_object
            )

        except Exception as e:
            console_logger.error(f"Error placing asset '{asset_id}': {e}")
            return self._create_failure_result(
                asset_id=asset_id,
                message=f"Failed to place asset: {str(e)}",
            )

    def _create_success_result(self, asset_id: str, furniture_obj: SceneObject) -> str:
        """Create success result for furniture placement."""
        position = furniture_obj.transform.translation()
        rpy = RollPitchYaw(furniture_obj.transform.rotation())

        return FurniturePlacementResult(
            success=True,
            message=(
                f"Successfully placed asset '{furniture_obj.name}' as object "
                f"'{furniture_obj.object_id}'. "
                f"Use object_id '{furniture_obj.object_id}' for remove/move operations."
            ),
            asset_id=asset_id,
            object_id=str(furniture_obj.object_id),
            position=Position3D(x=position[0], y=position[1], z=position[2]),
            rotation=Rotation3D(
                roll=math.degrees(rpy.roll_angle()),  # Convert radians to degrees
                pitch=math.degrees(rpy.pitch_angle()),
                yaw=math.degrees(rpy.yaw_angle()),
            ),
            has_geometry=bool(furniture_obj.geometry_path),
        ).to_json()

    def _create_failure_result(
        self, asset_id: str, message: str, error_type: FurnitureErrorType | None = None
    ) -> str:
        """Create failure result for furniture placement."""
        return FurniturePlacementResult(
            success=False,
            message=message,
            asset_id=asset_id,
            object_id="",
            position=Position3D(x=0.0, y=0.0, z=0.0),
            rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=0.0),
            has_geometry=False,
            error_type=error_type,
        ).to_json()

    @log_scene_action
    def _move_furniture_impl(
        self,
        object_id: str,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> str:
        """
        Implementation for moving furniture to absolute pose. Rotations are in degrees.
        """
        console_logger.info("Tool called: move_furniture_tool")
        try:
            # Convert string ID to UniqueID.
            unique_id = UniqueID(object_id)

            # Check if object exists.
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                return FurnitureOperationResult(
                    success=False,
                    message=f"Object with ID '{object_id}' not found in scene",
                    object_id=object_id,
                    error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Check if object is immutable.
            if scene_obj.immutable:
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Cannot move {scene_obj.name}: architectural element is "
                        "immutable"
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.IMMUTABLE_OBJECT,
                    suggested_action=(
                        "Walls and architectural elements cannot be repositioned"
                    ),
                ).to_json()

            # Validate position is within floor plan bounds.
            is_valid, error_msg = self._check_floor_bounds(x=x, y=y)
            if not is_valid:
                return FurnitureOperationResult(
                    success=False,
                    message=error_msg,
                    object_id=object_id,
                    error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                ).to_json()

            surface_pose = self._surface_aligned_pose(x, y, yaw)
            if surface_pose is not None:
                z, roll, pitch, yaw = surface_pose

            # Get current position and rotation.
            current_transform = scene_obj.transform
            current_position = current_transform.translation()
            current_rpy = RollPitchYaw(current_transform.rotation())

            new_position = np.array([x, y, z])
            new_rotation = np.array([roll, pitch, yaw])
            current_rotation = np.array(
                [
                    math.degrees(current_rpy.roll_angle()),
                    math.degrees(current_rpy.pitch_angle()),
                    math.degrees(current_rpy.yaw_angle()),
                ]
            )  # Current rotation in degrees for comparison

            # Check if both position and rotation are unchanged.
            position_unchanged = np.allclose(current_position, new_position, atol=1e-6)
            rotation_unchanged = np.allclose(current_rotation, new_rotation, atol=1e-6)

            if position_unchanged and rotation_unchanged:
                console_logger.info(
                    f"Furniture '{scene_obj.name}'/'{object_id}' is already at position "
                    f"({x}, {y}, {z}) and rotation ({roll}, {pitch}, {yaw}) - no "
                    "movement needed"
                )
                return FurnitureOperationResult(
                    success=False,
                    message=f"{scene_obj.name} is already at the target position and "
                    "rotation - no movement needed",
                    object_id=object_id,
                    error_type=FurnitureErrorType.NO_MOVEMENT,
                    current_position=Position3D(
                        x=current_position[0],
                        y=current_position[1],
                        z=current_position[2],
                    ),
                    attempted_position=Position3D(x=x, y=y, z=z),
                    current_rotation=Rotation3D(
                        roll=current_rotation[0],
                        pitch=current_rotation[1],
                        yaw=current_rotation[2],
                    ),
                    attempted_rotation=Rotation3D(roll=roll, pitch=pitch, yaw=yaw),
                    suggested_action="Try moving to a different position or rotation",
                ).to_json()

            # Create new transform with absolute position and rotation.
            # Convert degrees to radians for Drake's RigidTransform.
            new_rpy = RollPitchYaw(
                math.radians(roll), math.radians(pitch), math.radians(yaw)
            )
            new_transform = RigidTransform(rpy=new_rpy, p=[x, y, z])

            # Apply placement noise for realistic variation.
            new_transform = apply_placement_noise(
                transform=new_transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )

            prospective = copy.copy(scene_obj)
            prospective.transform = new_transform
            envelope_valid, envelope_message = self._validate_spatial_envelope(
                prospective
            )
            if not envelope_valid:
                console_logger.warning("Move rejected: %s", envelope_message)
                return FurnitureOperationResult(
                    success=False,
                    message=envelope_message,
                    object_id=object_id,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                    suggested_action="Choose a pose that fits the support and enclosure",
                ).to_json()

            contextual_valid, contextual_message = self._validate_contextual_zones(
                prospective
            )
            if not contextual_valid:
                console_logger.warning(
                    "Contextual move rejected: %s", contextual_message
                )
                return FurnitureOperationResult(
                    success=False,
                    message=contextual_message,
                    object_id=object_id,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                    suggested_action="Apply the machine-readable repair and retry",
                ).to_json()

            # Update object to new absolute pose.
            self.scene.move_object(object_id=unique_id, new_transform=new_transform)

            placement_collisions = self._furniture_collisions_for(unique_id)
            if placement_collisions:
                self.scene.move_object(
                    object_id=unique_id, new_transform=current_transform
                )
                message = (
                    "Move rejected because it intersects existing furniture: "
                    + "; ".join(placement_collisions)
                    + ". Try a clear position."
                )
                console_logger.warning(message)
                return FurnitureOperationResult(
                    success=False,
                    message=message,
                    object_id=object_id,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                    suggested_action="Move the object farther from the listed furniture",
                ).to_json()

            # Log what changed.
            changes = []
            if not position_unchanged:
                new_pos = new_transform.translation()
                changes.append(
                    f"position from ({current_position[0]:.3f}, "
                    f"{current_position[1]:.3f}, {current_position[2]:.3f}) to "
                    f"({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})"
                )
            if not rotation_unchanged:
                new_rpy = RollPitchYaw(new_transform.rotation())
                new_roll, new_pitch, new_yaw = (
                    math.degrees(new_rpy.roll_angle()),
                    math.degrees(new_rpy.pitch_angle()),
                    math.degrees(new_rpy.yaw_angle()),
                )
                changes.append(
                    f"rotation from ({current_rotation[0]:.3f}°, "
                    f"{current_rotation[1]:.3f}°, {current_rotation[2]:.3f}°) to "
                    f"({new_roll:.3f}°, {new_pitch:.3f}°, {new_yaw:.3f}°)"
                )

            console_logger.info(
                f"Moved furniture '{scene_obj.name}'/'{object_id}': {' and '.join(changes)}"
            )

            return FurnitureOperationResult(
                success=True,
                message=f"Successfully moved {scene_obj.name} to new position and "
                "rotation",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error moving furniture '{object_id}': {e}")
            return FurnitureOperationResult(
                success=False,
                message=f"Failed to move furniture: {str(e)}",
                object_id=object_id,
            ).to_json()

    @log_scene_action
    def _remove_furniture_impl(self, object_id: str) -> str:
        """Implementation for removing furniture."""
        console_logger.info("Tool called: remove_furniture_tool")
        try:
            # Convert string ID to UniqueID.
            unique_id = UniqueID(object_id)

            # Check if object exists.
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                base_name = (
                    object_id.rsplit("_", 1)[0] if "_" in object_id else object_id
                )
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Object with ID '{object_id}' not found in scene.\n\n"
                        f"Causes:\n"
                        f"1. Missing ID postfix - IDs have random postfixes like "
                        f"'{base_name}_a1b2c3'\n"
                        f"2. Object already removed\n"
                        f"3. Typo in object_id\n\n"
                        f"Call get_current_scene_state() to see current object IDs with "
                        f"postfixes. Find objects whose names start with '{base_name}'."
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                    suggested_action="Call get_current_scene_state() to verify object IDs",
                ).to_json()

            # Check if object is immutable.
            if scene_obj.immutable:
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Cannot remove {scene_obj.name}: architectural element is "
                        "immutable"
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.IMMUTABLE_OBJECT,
                    suggested_action=(
                        "Walls and architectural elements cannot be removed"
                    ),
                ).to_json()

            # Remove from scene.
            removed = self.scene.remove_object(unique_id)

            if not removed:
                # Log detailed information for debugging.
                scene_ids = list(self.scene.objects.keys())
                console_logger.info(
                    f"Failed to remove object '{object_id}' from scene. "
                    f"Object exists in scene (get_object succeeded) but remove_object "
                    f"returned False."
                )
                console_logger.info(f"Attempted to remove ID: {object_id}")
                console_logger.info(f"Attempted to remove ID repr: {repr(unique_id)}")
                console_logger.info(
                    f"Current scene object IDs ({len(scene_ids)}): {scene_ids}"
                )
                return FurnitureOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} exists but could not be removed from "
                        f"scene"
                    ),
                    object_id=object_id,
                    error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            console_logger.info(f"Removed furniture '{scene_obj.name}' from scene")
            return FurnitureOperationResult(
                success=True,
                message=f"Successfully removed {scene_obj.name} from scene",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing furniture '{object_id}': {e}")
            return FurnitureOperationResult(
                success=False,
                message=f"Failed to remove furniture: {str(e)}",
                object_id=object_id,
            ).to_json()

    @log_scene_action
    def _rescale_furniture_impl(self, object_id: str, scale_factor: float) -> str:
        """Implementation for rescaling furniture."""
        console_logger.info(
            f"Tool called: rescale_furniture (id={object_id}, scale={scale_factor})"
        )
        if scale_factor > 0 and scale_factor != 1.0:
            try:
                scene_object = self.scene.get_object(UniqueID(object_id))
            except Exception:
                scene_object = None
            if scene_object is not None and scene_object.sdf_path is not None:
                scene_objects = getattr(self.scene, "objects", {})
                candidates = (
                    scene_objects.values()
                    if isinstance(scene_objects, dict)
                    else (scene_object,)
                )
                for affected in candidates:
                    if affected.sdf_path != scene_object.sdf_path:
                        continue
                    prospective = copy.copy(affected)
                    if prospective.bbox_min is not None:
                        prospective.bbox_min = prospective.bbox_min * scale_factor
                    if prospective.bbox_max is not None:
                        prospective.bbox_max = prospective.bbox_max * scale_factor
                    valid, message = self._validate_spatial_envelope(prospective)
                    if not valid:
                        return RescaleResult(
                            success=False,
                            message=f"Rescale rejected: {message}",
                            object_id=object_id,
                            error_type=RescaleErrorType.RESCALE_FAILED,
                        ).to_json()
        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="furniture",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()

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
