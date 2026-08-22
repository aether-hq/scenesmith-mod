"""Tools for ceiling-mounted object generation and placement.

This module provides tools for generating and placing ceiling-mounted objects
(lights, fans, chandeliers, etc.) on the ceiling plane.
"""

import logging
import math

from pydrake.all import RollPitchYaw

from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.core.response_datatypes import AssetInfo, BoundingBox3D
from scenesmith.agent_utils.geometry.rescale_helpers import rescale_object_common
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID
from scenesmith.ceiling_agents.tools.response_dataclasses import (
    AvailableAssetsResult,
    CeilingErrorType,
    CeilingObjectInfo,
    CeilingOperationResult,
    CeilingSceneStateResult,
    PlaceCeilingObjectResult,
    RoomBoundsInfo,
)

console_logger = logging.getLogger(__name__)

CEILING_ATTACHMENT_CLEARANCE_METERS = 0.01


class CeilingSceneManagementMixin:
    @log_scene_action
    def _move_ceiling_object_impl(
        self,
        object_id: str,
        position_x: float,
        position_y: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for moving ceiling object to new position."""
        console_logger.info(
            f"Tool called: move_ceiling_object("
            f"object_id={object_id}, position_x={position_x}, "
            f"position_y={position_y}, rotation_degrees={rotation_degrees})"
        )

        try:
            # Get the existing object.
            unique_id = UniqueID(object_id)
            scene_object = self.scene.get_object(unique_id)
            if scene_object is None:
                return CeilingOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene.",
                    object_id=object_id,
                    error_type=CeilingErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Verify it's a ceiling-mounted object.
            if scene_object.object_type != ObjectType.CEILING_MOUNTED:
                return CeilingOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a ceiling-mounted object "
                        f"(type: {scene_object.object_type.value})."
                    ),
                    object_id=object_id,
                    error_type=CeilingErrorType.INVALID_OPERATION,
                ).to_json()

            # Validate position is within room bounds.
            min_x, min_y, max_x, max_y = self.room_bounds
            if not (min_x <= position_x <= max_x and min_y <= position_y <= max_y):
                return CeilingOperationResult(
                    success=False,
                    message=(
                        f"Position ({position_x:.2f}, {position_y:.2f}) is outside "
                        f"room bounds ({min_x:.2f}, {min_y:.2f}) to "
                        f"({max_x:.2f}, {max_y:.2f})"
                    ),
                    object_id=object_id,
                    error_type=CeilingErrorType.POSITION_OUT_OF_BOUNDS,
                ).to_json()

            # Convert ceiling SE(2) to world SE(3).
            world_transform = self._ceiling_transform(
                x=position_x, y=position_y, rotation_deg=rotation_degrees
            )
            if world_transform is None:
                return CeilingOperationResult(
                    success=False,
                    message=(
                        f"Position ({position_x:.2f}, {position_y:.2f}) is not "
                        "on an authored overhead attachment surface."
                    ),
                    object_id=object_id,
                    error_type=CeilingErrorType.POSITION_OUT_OF_BOUNDS,
                ).to_json()

            # Update the object's transform.
            scene_object.transform = world_transform

            console_logger.info(
                f"Moved ceiling object '{scene_object.name}' to "
                f"({position_x:.3f}, {position_y:.3f})"
            )

            return CeilingOperationResult(
                success=True,
                message=(
                    f"Successfully moved '{scene_object.name}' to "
                    f"({position_x:.3f}m, {position_y:.3f}m)"
                ),
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error moving ceiling object: {e}", exc_info=True)
            return CeilingOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
                error_type=None,
            ).to_json()

    @log_scene_action
    def _remove_ceiling_object_impl(self, object_id: str, **kwargs) -> str:
        """Implementation for removing ceiling object from scene."""
        console_logger.info(
            f"Tool called: remove_ceiling_object(object_id={object_id})"
        )

        try:
            unique_id = UniqueID(object_id)
            scene_object = self.scene.get_object(unique_id)
            if scene_object is None:
                return CeilingOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene.",
                    object_id=object_id,
                    error_type=CeilingErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Verify it's a ceiling-mounted object.
            if scene_object.object_type != ObjectType.CEILING_MOUNTED:
                return CeilingOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a ceiling-mounted object "
                        f"(type: {scene_object.object_type.value})."
                    ),
                    object_id=object_id,
                    error_type=CeilingErrorType.INVALID_OPERATION,
                ).to_json()

            # Remove from scene.
            self.scene.remove_object(unique_id)

            console_logger.info(f"Removed ceiling object '{scene_object.name}'")

            return CeilingOperationResult(
                success=True,
                message=f"Successfully removed '{scene_object.name}'.",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing ceiling object: {e}", exc_info=True)
            return CeilingOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
                error_type=None,
            ).to_json()

    @log_scene_action
    def _rescale_ceiling_object_impl(
        self, object_id: str, scale_factor: float, **kwargs
    ) -> str:
        """Implementation for rescaling a ceiling object."""
        console_logger.info(
            f"Tool called: rescale_ceiling_object("
            f"object_id={object_id}, scale_factor={scale_factor})"
        )
        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="ceiling object",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()

    def _get_current_scene_state_impl(self) -> str:
        """Implementation for getting current scene state."""
        # Build room bounds info.
        min_x, min_y, max_x, max_y = self.room_bounds
        room_info = RoomBoundsInfo(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            ceiling_height=self.ceiling_height,
        )

        # Build ceiling objects info.
        ceiling_objects_info = []
        for obj in self.scene.get_objects_by_type(ObjectType.CEILING_MOUNTED):
            # Get ceiling-local position from world transform.
            # Ceiling is at fixed Z height, so X, Y come from translation.
            translation = obj.transform.translation()
            pos_x = float(translation[0])
            pos_y = float(translation[1])

            # Extract yaw rotation from transform.
            rpy = RollPitchYaw(obj.transform.rotation())
            rot_deg = math.degrees(rpy.yaw_angle())

            # Get dimensions from bounding box.
            width = float(obj.bbox_max[0] - obj.bbox_min[0])
            depth = float(obj.bbox_max[1] - obj.bbox_min[1])
            height = float(obj.bbox_max[2] - obj.bbox_min[2])

            ceiling_objects_info.append(
                CeilingObjectInfo(
                    object_id=str(obj.object_id),
                    description=obj.description,
                    position_x=pos_x,
                    position_y=pos_y,
                    rotation_degrees=rot_deg,
                    dimensions=BoundingBox3D(width=width, depth=depth, height=height),
                )
            )

        result = CeilingSceneStateResult(
            room_bounds=room_info,
            ceiling_objects=ceiling_objects_info,
            object_count=len(ceiling_objects_info),
        )

        return result.to_json()

    def _list_available_assets_impl(self) -> str:
        """Implementation for listing available ceiling assets."""
        all_assets = self.asset_manager.list_available_assets()
        ceiling_assets = [
            asset
            for asset in all_assets
            if asset.object_type == ObjectType.CEILING_MOUNTED
        ]

        assets_info = []
        for asset in ceiling_assets:
            assets_info.append(
                AssetInfo(
                    asset_id=str(asset.object_id),
                    name=asset.name,
                    description=asset.description,
                    object_type=asset.object_type.value,
                    dimensions=BoundingBox3D(
                        width=float(asset.bbox_max[0] - asset.bbox_min[0]),
                        depth=float(asset.bbox_max[1] - asset.bbox_min[1]),
                        height=float(asset.bbox_max[2] - asset.bbox_min[2]),
                    ),
                )
            )

        result = AvailableAssetsResult(
            assets=assets_info,
            count=len(assets_info),
        )

        return result.to_json()

    def _create_placement_failure_result(
        self,
        asset_id: str,
        position_x: float,
        position_y: float,
        rotation_deg: float,
        message: str,
        error_type: CeilingErrorType | None,
    ) -> str:
        """Create a placement failure result."""
        result = PlaceCeilingObjectResult(
            success=False,
            asset_id=asset_id,
            object_id="",
            message=message,
            position_x=position_x,
            position_y=position_y,
            rotation_degrees=rotation_deg,
            error_type=error_type,
        )
        return result.to_json()
