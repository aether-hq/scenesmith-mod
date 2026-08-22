"""Tools for wall-mounted object generation and placement.

This module provides tools for generating and placing wall-mounted objects
(mirrors, artwork, shelves, clocks, etc.) on wall surfaces.
"""

import logging
import math

from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.core.response_datatypes import AssetInfo, BoundingBox3D
from scenesmith.agent_utils.geometry.rescale_helpers import rescale_object_common
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID
from scenesmith.wall_agents.tools.response_dataclasses import (
    AvailableAssetsResult,
    ExcludedRegionInfo,
    PlaceWallObjectResult,
    WallErrorType,
    WallObjectInfo,
    WallOperationResult,
    WallSceneStateResult,
    WallSurfaceInfo,
)

console_logger = logging.getLogger(__name__)


class WallSceneManagementMixin:
    @log_scene_action
    def _remove_wall_object_impl(self, object_id: str, **kwargs) -> str:
        """Implementation for removing wall object from scene."""
        console_logger.info(f"Tool called: remove_wall_object(object_id={object_id})")

        try:
            unique_id = UniqueID(object_id)
            scene_object = self.scene.get_object(unique_id)
            if scene_object is None:
                return WallOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene.",
                    object_id=object_id,
                    error_type=WallErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Verify it's a wall-mounted object.
            if scene_object.object_type != ObjectType.WALL_MOUNTED:
                return WallOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a wall-mounted object "
                        f"(type: {scene_object.object_type.value})."
                    ),
                    object_id=object_id,
                    error_type=WallErrorType.INVALID_OPERATION,
                ).to_json()

            # Remove from scene.
            self.scene.remove_object(unique_id)

            console_logger.info(f"Removed wall object '{scene_object.name}'")

            return WallOperationResult(
                success=True,
                message=f"Successfully removed '{scene_object.name}'.",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing wall object: {e}", exc_info=True)
            return WallOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
                error_type=None,
            ).to_json()

    @log_scene_action
    def _rescale_wall_object_impl(
        self, object_id: str, scale_factor: float, **kwargs
    ) -> str:
        """Implementation for rescaling a wall object."""
        console_logger.info(
            f"Tool called: rescale_wall_object("
            f"object_id={object_id}, scale_factor={scale_factor})"
        )
        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="wall object",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()

    def _get_current_scene_state_impl(self) -> str:
        """Implementation for getting current scene state."""
        # Build wall surfaces info.
        surfaces_info = []
        for surface in self.wall_surfaces:
            excluded = [
                ExcludedRegionInfo(x_min=r[0], z_min=r[1], x_max=r[2], z_max=r[3])
                for r in surface.excluded_regions
            ]
            surfaces_info.append(
                WallSurfaceInfo(
                    surface_id=str(surface.surface_id),
                    wall_id=surface.wall_id,
                    wall_direction=surface.wall_direction.value,
                    length=surface.length,
                    height=surface.height,
                    excluded_regions=excluded,
                )
            )

        # Build wall objects info.
        wall_objects_info = []
        for obj in self.scene.get_objects_by_type(ObjectType.WALL_MOUNTED):
            # Get wall-local position from placement_info.
            if obj.placement_info is None:
                raise RuntimeError(
                    f"Wall object '{obj.name}' ({obj.object_id}) has no placement_info. "
                    f"All wall objects must be placed via place_wall_object."
                )
            pos_x = float(obj.placement_info.position_2d[0])
            pos_z = float(obj.placement_info.position_2d[1])
            rot_deg = math.degrees(obj.placement_info.rotation_2d)
            surface_id = str(obj.placement_info.parent_surface_id)

            # Get dimensions from bounding box.
            width = float(obj.bbox_max[0] - obj.bbox_min[0])
            depth = float(obj.bbox_max[1] - obj.bbox_min[1])
            height = float(obj.bbox_max[2] - obj.bbox_min[2])

            wall_objects_info.append(
                WallObjectInfo(
                    object_id=str(obj.object_id),
                    description=obj.description,
                    wall_surface_id=surface_id,
                    position_x=pos_x,
                    position_z=pos_z,
                    rotation_deg=rot_deg,
                    dimensions=BoundingBox3D(width=width, depth=depth, height=height),
                )
            )

        result = WallSceneStateResult(
            wall_surfaces=surfaces_info,
            wall_objects=wall_objects_info,
            object_count=len(wall_objects_info),
        )

        return result.to_json()

    def _list_available_assets_impl(self) -> str:
        """Implementation for listing available wall assets."""
        all_assets = self.asset_manager.list_available_assets()
        wall_assets = [
            asset
            for asset in all_assets
            if asset.object_type == ObjectType.WALL_MOUNTED
        ]

        assets_info = []
        for asset in wall_assets:
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
        wall_surface_id: str,
        position_x: float,
        position_z: float,
        rotation_deg: float,
        message: str,
        error_type: WallErrorType | None,
    ) -> str:
        """Create a placement failure result."""
        result = PlaceWallObjectResult(
            success=False,
            asset_id=asset_id,
            object_id="",
            message=message,
            wall_surface_id=wall_surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_deg=rotation_deg,
            error_type=error_type,
        )
        return result.to_json()
