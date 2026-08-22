import logging
import math

from pathlib import Path

import numpy as np

from pydrake.all import RollPitchYaw

from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.design.placement_noise import apply_placement_noise
from scenesmith.agent_utils.geometry.rescale_helpers import rescale_object_common
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    PlacementInfo,
    SceneObject,
    UniqueID,
)
from scenesmith.manipuland_agents.tools.response_dataclasses import (
    ManipulandErrorType,
    ManipulandOperationResult,
    ManipulandPlacementResult,
    Position2D,
    Position3D,
    Rotation3D,
)
from scenesmith.utils.geometry.sdf_utils import (
    deserialize_rigid_transform,
    serialize_rigid_transform,
)

console_logger = logging.getLogger(__name__)


class ManipulandPlacementMixin:
    """Manipuland placement, movement, removal, and rescaling operations."""

    @log_scene_action
    def _place_manipuland_on_surface_impl(
        self,
        asset_id: str,
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for placing manipuland on support surface."""
        console_logger.info("Tool called: place_manipuland_on_surface")

        try:
            # Validate surface_id exists.
            if surface_id not in self.support_surfaces:
                available_ids = list(self.support_surfaces.keys())
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Invalid surface_id: {surface_id}. "
                        f"Available surfaces: {available_ids}"
                    ),
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                )

            # Get the target surface.
            target_surface = self.support_surfaces[surface_id]

            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(asset_id)
            except Exception:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=f"Invalid asset ID format: {asset_id}",
                    error_type=ManipulandErrorType.ASSET_NOT_FOUND,
                )

            # Get asset from registry.
            original_asset = self.asset_manager.get_asset_by_id(unique_id)
            if not original_asset:
                # Get all assets and filter for manipulands.
                all_assets = self.asset_manager.list_available_assets()
                available_assets = [
                    asset
                    for asset in all_assets
                    if asset.object_type == ObjectType.MANIPULAND
                ]
                available_ids = [str(a.object_id) for a in available_assets]
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Asset {asset_id} not found. Available manipulands: "
                        f"{available_ids}"
                    ),
                    error_type=ManipulandErrorType.ASSET_NOT_FOUND,
                )

            furniture = self.scene.get_object(self.current_furniture_id)
            if furniture is not None:
                allowed, semantic_error = self._support_semantics_allow(
                    furniture, original_asset
                )
                if not allowed:
                    console_logger.warning("Support-zone rejection: %s", semantic_error)
                    return self._create_placement_failure_result(
                        asset_id=asset_id,
                        message=semantic_error or "Asset is incompatible with surface",
                        error_type=ManipulandErrorType.INVALID_OPERATION,
                    )

            # Validate position is within surface bounds (convex hull).
            position_2d = np.array([position_x, position_z])
            try:
                if not target_surface.contains_point_2d(position_2d):
                    return self._create_placement_failure_result(
                        asset_id=asset_id,
                        message=(
                            f"Position ({position_x:.3f}, {position_z:.3f}) is outside "
                            f"the convex hull of surface {surface_id}. "
                            f"Use list_support_surfaces() to see available surfaces."
                        ),
                        error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                    )
            except ValueError as e:
                # Surface has no mesh - this shouldn't happen with HSM extraction.
                console_logger.error(f"Surface {surface_id} has no mesh: {e}")
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Surface {surface_id} has no mesh geometry for "
                        f"placement validation."
                    ),
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                )

            # Validate object convex hull fits within surface boundary.
            # Top surfaces allow configurable overlap tolerance for natural overhang.
            # Non-top surfaces (shelves) require strict containment (0% overlap).
            overlap_ratio = (
                self.top_surface_overlap_tolerance
                if self._is_top_surface(surface_id)
                else 0.0
            )
            is_valid, error_msg = self._validate_convex_hull_footprint(
                target_surface=target_surface,
                geometry_path=original_asset.geometry_path,
                position_2d=position_2d,
                rotation_degrees=rotation_degrees,
                allow_overlap_ratio=overlap_ratio,
                scale_factor=original_asset.scale_factor,
            )
            if not is_valid:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=error_msg,
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            # Validate object height fits within surface clearance.
            object_height = float(
                original_asset.bbox_max[2] - original_asset.bbox_min[2]
            )
            surface_clearance = float(
                target_surface.bounding_box_max[2] - target_surface.bounding_box_min[2]
            )

            console_logger.info(
                f"Clearance check: object_height={object_height:.3f}m, "
                f"surface_clearance={surface_clearance:.3f}m"
            )

            if object_height > surface_clearance:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    message=(
                        f"Object height {object_height:.3f}m exceeds surface "
                        f"clearance {surface_clearance:.3f}m. Make sure you are "
                        f"placing on the correct surface that you planned to use. "
                        f"If this is the intended surface, choose a shorter object "
                        f"or find a surface with more clearance."
                    ),
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            console_logger.info(
                f"Placing manipuland {asset_id} ({original_asset.name}) at surface "
                f"position ({position_x:.3f}, {position_z:.3f}), "
                f"rotation {rotation_degrees:.1f}°"
            )

            # Thin coverings have no collision geometry and are welded, so they don't
            # fall during physics simulation. Compensate for the surface gravity offset
            # by placing them directly on the physical surface.
            is_thin_covering = (
                original_asset.metadata.get("asset_source") == "thin_covering"
            )
            z_offset = 0.0
            if is_thin_covering:
                z_offset = -self.cfg.support_surface_extraction.height.surface_offset_m
                console_logger.debug(
                    f"Thin covering detected: applying z_offset={z_offset:.3f}m"
                )

            # Convert SE(2) on surface to SE(3) in world.
            rotation_radians = math.radians(rotation_degrees)
            world_transform = target_surface.to_world_pose(
                position_2d=position_2d, rotation_2d=rotation_radians, z_offset=z_offset
            )

            # Apply placement noise for realistic variation.
            world_transform = apply_placement_noise(
                transform=world_transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )

            # Create new scene object with unique ID.
            object_id = self.scene.generate_unique_id(original_asset.name)
            scene_object = SceneObject(
                object_id=object_id,
                object_type=ObjectType.MANIPULAND,
                name=original_asset.name,
                description=original_asset.description,
                transform=world_transform,
                geometry_path=original_asset.geometry_path,
                sdf_path=original_asset.sdf_path,
                image_path=original_asset.image_path,
                metadata=original_asset.metadata.copy(),
                bbox_min=original_asset.bbox_min,
                bbox_max=original_asset.bbox_max,
                scale_factor=original_asset.scale_factor,
                placement_info=PlacementInfo(
                    parent_surface_id=target_surface.surface_id,
                    position_2d=position_2d.copy(),
                    rotation_2d=rotation_radians,
                    placement_method="surface_placement",
                ),
            )

            # Add to scene.
            self.scene.add_object(scene_object)

            # Extract world pose for response.
            world_position = world_transform.translation()
            world_rpy = RollPitchYaw(world_transform.rotation())

            console_logger.info(
                f"Successfully placed manipuland '{original_asset.name}' as "
                f"{object_id} on surface {surface_id}"
            )

            # Create success result.
            result = ManipulandPlacementResult(
                success=True,
                message=(
                    f"Successfully placed '{original_asset.name}' on surface at "
                    f"({position_x:.3f}, {position_z:.3f})"
                ),
                asset_id=asset_id,
                object_id=str(object_id),
                world_position=Position3D(
                    x=float(world_position[0]),
                    y=float(world_position[1]),
                    z=float(world_position[2]),
                ),
                world_rotation=Rotation3D(
                    roll=math.degrees(world_rpy.roll_angle()),
                    pitch=math.degrees(world_rpy.pitch_angle()),
                    yaw=math.degrees(world_rpy.yaw_angle()),
                ),
                surface_position=Position2D(x=position_x, y=position_z),
                surface_rotation_deg=rotation_degrees,
                parent_surface_id=surface_id,
                has_geometry=scene_object.geometry_path is not None,
            )

            return result.to_json()

        except Exception as e:
            console_logger.error(f"Error placing manipuland: {e}", exc_info=True)
            return self._create_placement_failure_result(
                asset_id=asset_id,
                message=f"Unexpected error: {str(e)}",
                error_type=None,
            )

    @log_scene_action
    def _move_manipuland_impl(
        self,
        object_id: str,
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for moving manipuland to new surface position."""
        console_logger.info("Tool called: move_manipuland")

        try:
            # Validate surface_id exists.
            if surface_id not in self.support_surfaces:
                available_ids = list(self.support_surfaces.keys())
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"Invalid surface_id: {surface_id}. "
                        f"Available surfaces: {available_ids}"
                    ),
                    object_id=object_id,
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                ).to_json()

            # Get the target surface.
            target_surface = self.support_surfaces[surface_id]

            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(object_id)
            except Exception:
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=f"Invalid object ID format: {object_id}",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                )

            # Check if object exists.
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Object with ID '{object_id}' not found in scene",
                    object_id=object_id,
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Validate position is within surface bounds (convex hull).
            position_2d = np.array([position_x, position_z])
            try:
                if not target_surface.contains_point_2d(position_2d):
                    return self._create_placement_failure_result(
                        asset_id=object_id,
                        message=(
                            f"Position ({position_x:.3f}, {position_z:.3f}) is outside "
                            f"the convex hull of surface {surface_id}."
                        ),
                        error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                    )
            except ValueError as e:
                # Surface has no mesh.
                console_logger.error(f"Surface {surface_id} has no mesh: {e}")
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=(
                        f"Surface {surface_id} has no mesh geometry for "
                        f"placement validation."
                    ),
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                )

            # Validate object convex hull fits within surface boundary.
            # Top surfaces allow configurable overlap tolerance for natural overhang.
            # Non-top surfaces (shelves) require strict containment (0% overlap).
            overlap_ratio = (
                self.top_surface_overlap_tolerance
                if self._is_top_surface(surface_id)
                else 0.0
            )

            # For composite objects, use reference member's geometry for footprint validation.
            geometry_path = scene_obj.geometry_path
            composite_type = scene_obj.metadata.get("composite_type")
            if geometry_path is None and composite_type == "stack":
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    bottom_geometry = member_assets[0].get("geometry_path")
                    if bottom_geometry:
                        geometry_path = Path(bottom_geometry)
            elif geometry_path is None and composite_type == "filled_container":
                container_asset = scene_obj.metadata.get("container_asset")
                if container_asset:
                    container_geometry = container_asset.get("geometry_path")
                    if container_geometry:
                        geometry_path = Path(container_geometry)
            elif geometry_path is None and composite_type == "pile":
                # Use first member's geometry for pile footprint validation.
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    first_geometry = member_assets[0].get("geometry_path")
                    if first_geometry:
                        geometry_path = Path(first_geometry)

            if geometry_path is None:
                return ManipulandOperationResult(
                    success=False,
                    message="Cannot validate placement: no geometry available",
                    object_id=object_id,
                    error_type=ManipulandErrorType.INVALID_OPERATION,
                ).to_json()

            is_valid, error_msg = self._validate_convex_hull_footprint(
                target_surface=target_surface,
                geometry_path=geometry_path,
                position_2d=position_2d,
                rotation_degrees=rotation_degrees,
                allow_overlap_ratio=overlap_ratio,
                scale_factor=scene_obj.scale_factor,
            )
            if not is_valid:
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=error_msg,
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            # Validate object height fits within surface clearance.
            object_height = float(scene_obj.bbox_max[2] - scene_obj.bbox_min[2])
            surface_clearance = float(
                target_surface.bounding_box_max[2] - target_surface.bounding_box_min[2]
            )

            console_logger.info(
                f"Clearance check (move): object_height={object_height:.3f}m, "
                f"surface_clearance={surface_clearance:.3f}m"
            )

            if object_height > surface_clearance:
                return self._create_placement_failure_result(
                    asset_id=object_id,
                    message=(
                        f"Object height {object_height:.3f}m exceeds surface "
                        f"clearance {surface_clearance:.3f}m. Make sure you are "
                        f"placing on the correct surface that you planned to use. "
                        f"If this is the intended surface, choose a shorter object "
                        f"or find a surface with more clearance."
                    ),
                    error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
                )

            # Get current surface-relative pose from placement info.
            if scene_obj.placement_info is None:
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"Object '{object_id}' has no placement info - "
                        "cannot determine current position"
                    ),
                    object_id=object_id,
                    error_type=ManipulandErrorType.INVALID_OPERATION,
                ).to_json()

            # Note: We allow moving objects between surfaces (no validation of current
            # surface).
            current_position_2d = scene_obj.placement_info.position_2d
            current_rotation_2d = scene_obj.placement_info.rotation_2d

            # Check if both position and rotation are unchanged.
            rotation_radians = math.radians(rotation_degrees)
            position_unchanged = np.allclose(
                current_position_2d, position_2d, atol=1e-6
            )
            rotation_unchanged = np.allclose(
                current_rotation_2d, rotation_radians, atol=1e-6
            )

            if position_unchanged and rotation_unchanged:
                console_logger.info(
                    f"Manipuland '{scene_obj.name}'/'{object_id}' is already at "
                    f"position ({position_x:.3f}, {position_z:.3f}) and rotation "
                    f"{rotation_degrees:.1f}° - no movement needed"
                )
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"{scene_obj.name} is already at the target position and "
                        "rotation - no movement needed"
                    ),
                    object_id=object_id,
                    error_type=ManipulandErrorType.NO_MOVEMENT,
                ).to_json()

            console_logger.info(
                f"Moving manipuland {object_id} ({scene_obj.name}) to surface "
                f"position ({position_x:.3f}, {position_z:.3f}), "
                f"rotation {rotation_degrees:.1f}°"
            )

            # Convert SE(2) on surface to SE(3) in world.
            world_transform = target_surface.to_world_pose(
                position_2d=position_2d, rotation_2d=rotation_radians
            )

            # Apply placement noise for realistic variation.
            world_transform = apply_placement_noise(
                transform=world_transform,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=self.active_noise_profile.rotation_yaw_std_degrees,
            )

            # For stacks, capture old transform before moving to compute delta.
            old_stack_transform = scene_obj.transform

            # Update object to new pose.
            self.scene.move_object(object_id=unique_id, new_transform=world_transform)

            # For composite objects, also update member transforms to match new position.
            composite_type = scene_obj.metadata.get("composite_type")
            if composite_type == "stack":
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    t_delta = world_transform @ old_stack_transform.inverse()

                    for member in member_assets:
                        old_member_transform = deserialize_rigid_transform(
                            member["transform"]
                        )
                        new_member_transform = t_delta @ old_member_transform
                        member["transform"] = serialize_rigid_transform(
                            new_member_transform
                        )
            elif composite_type == "filled_container":
                # Filled container: update container_asset + all fill_assets transforms.
                t_delta = world_transform @ old_stack_transform.inverse()

                # Update container transform.
                container_asset = scene_obj.metadata.get("container_asset")
                if container_asset:
                    old_container_transform = deserialize_rigid_transform(
                        container_asset["transform"]
                    )
                    new_container_transform = t_delta @ old_container_transform
                    container_asset["transform"] = serialize_rigid_transform(
                        new_container_transform
                    )

                # Update all fill asset transforms.
                fill_assets = scene_obj.metadata.get("fill_assets", [])
                for fill_asset in fill_assets:
                    old_fill_transform = deserialize_rigid_transform(
                        fill_asset["transform"]
                    )
                    new_fill_transform = t_delta @ old_fill_transform
                    fill_asset["transform"] = serialize_rigid_transform(
                        new_fill_transform
                    )
            elif composite_type == "pile":
                # Pile: update all member_assets transforms (same structure as stack).
                member_assets = scene_obj.metadata.get("member_assets", [])
                if member_assets:
                    t_delta = world_transform @ old_stack_transform.inverse()

                    for member in member_assets:
                        old_member_transform = deserialize_rigid_transform(
                            member["transform"]
                        )
                        new_member_transform = t_delta @ old_member_transform
                        member["transform"] = serialize_rigid_transform(
                            new_member_transform
                        )

            # Update placement info (including new surface_id if moved between surfaces).
            scene_obj.placement_info.parent_surface_id = target_surface.surface_id
            scene_obj.placement_info.position_2d = position_2d.copy()
            scene_obj.placement_info.rotation_2d = rotation_radians

            # Extract world pose for response.
            world_position = world_transform.translation()
            world_rpy = RollPitchYaw(world_transform.rotation())

            console_logger.info(
                f"Successfully moved manipuland '{scene_obj.name}' ({object_id}) "
                f"to surface {surface_id}"
            )

            # Create success result.
            # For move operations, asset_id uses object_id since no new asset is placed.
            result = ManipulandPlacementResult(
                success=True,
                message=(
                    f"Successfully moved '{scene_obj.name}' to surface position "
                    f"({position_x:.3f}, {position_z:.3f})"
                ),
                asset_id=object_id,
                object_id=str(unique_id),
                world_position=Position3D(
                    x=float(world_position[0]),
                    y=float(world_position[1]),
                    z=float(world_position[2]),
                ),
                world_rotation=Rotation3D(
                    roll=float(math.degrees(world_rpy.roll_angle())),
                    pitch=float(math.degrees(world_rpy.pitch_angle())),
                    yaw=float(math.degrees(world_rpy.yaw_angle())),
                ),
                surface_position=Position2D(x=float(position_x), y=float(position_z)),
                surface_rotation_deg=float(rotation_degrees),
                parent_surface_id=surface_id,
                has_geometry=scene_obj.geometry_path is not None,
            )
            return result.to_json()

        except Exception as e:
            console_logger.error(f"Error moving manipuland: {e}", exc_info=True)
            return self._create_placement_failure_result(
                asset_id=object_id,
                message=f"Unexpected error: {str(e)}",
                error_type=None,
            )

    @log_scene_action
    def _remove_manipuland_impl(self, object_id: str, **kwargs) -> str:
        """Implementation for removing manipuland from scene."""
        console_logger.info(f"Tool called: remove_manipuland({object_id})")

        try:
            # Convert string to UniqueID.
            try:
                unique_id = UniqueID(object_id)
            except Exception:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Invalid object ID format: {object_id}",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

            # Get object from scene.
            obj = self.scene.get_object(unique_id)
            if not obj:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

            # Verify it's a manipuland.
            if obj.object_type != ObjectType.MANIPULAND:
                return ManipulandOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a manipuland "
                        f"(type: {obj.object_type.value})"
                    ),
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

            # Remove from scene.
            success = self.scene.remove_object(unique_id)

            if success:
                console_logger.info(f"Successfully removed manipuland {object_id}")
                return ManipulandOperationResult(
                    success=True,
                    message=f"Successfully removed '{obj.name}' ({object_id})",
                    object_id=object_id,
                ).to_json()
            else:
                return ManipulandOperationResult(
                    success=False,
                    message=f"Failed to remove {object_id}",
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                    object_id=object_id,
                ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing manipuland: {e}", exc_info=True)
            return ManipulandOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
            ).to_json()

    @log_scene_action
    def _rescale_manipuland_impl(
        self, object_id: str, scale_factor: float, **kwargs
    ) -> str:
        """Implementation for rescaling manipuland."""
        console_logger.info(
            f"Tool called: rescale_manipuland (id={object_id}, scale={scale_factor})"
        )
        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="manipuland",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()
