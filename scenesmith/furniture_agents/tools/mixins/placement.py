import copy
import logging
import math

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.design.placement_noise import apply_placement_noise
from scenesmith.agent_utils.geometry.rescale_helpers import rescale_object_common
from scenesmith.agent_utils.geometry.rescale_result import (
    RescaleErrorType,
    RescaleResult,
)
from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject, UniqueID
from scenesmith.agent_utils.scene.room_parts.room_support import (
    copy_scene_object_with_new_pose,
)
from scenesmith.furniture_agents.tools.response_dataclasses import (
    FurnitureErrorType,
    FurnitureOperationResult,
    FurniturePlacementResult,
    Position3D,
    Rotation3D,
)

console_logger = logging.getLogger(__name__)


class FurniturePlacementMixin:
    """Furniture add, move, remove, and rescale operations."""

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
            is_valid, error_msg = self._check_floor_bounds(x=x, y=y, reference_z=z)
            if not is_valid:
                return self._create_failure_result(
                    asset_id=asset_id,
                    message=error_msg,
                    error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                )

            surface_pose = self._surface_aligned_pose(x, y, yaw, reference_z=z)
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

            placement_collisions, structural_collisions = (
                self._placement_collisions_for(scene_object.object_id)
            )
            if placement_collisions or structural_collisions:
                self.scene.remove_object(scene_object.object_id)
                if structural_collisions:
                    message = (
                        "Placement rejected because it intersects room structure: "
                        + "; ".join(structural_collisions)
                        + ". Choose a clear supported pose and retry."
                    )
                else:
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
            is_valid, error_msg = self._check_floor_bounds(x=x, y=y, reference_z=z)
            if not is_valid:
                return FurnitureOperationResult(
                    success=False,
                    message=error_msg,
                    object_id=object_id,
                    error_type=FurnitureErrorType.POSITION_OUT_OF_BOUNDS,
                ).to_json()

            surface_pose = self._surface_aligned_pose(x, y, yaw, reference_z=z)
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

            placement_collisions, structural_collisions = (
                self._placement_collisions_for(unique_id)
            )
            if placement_collisions or structural_collisions:
                self.scene.move_object(
                    object_id=unique_id, new_transform=current_transform
                )
                if structural_collisions:
                    message = (
                        "Move rejected because it intersects room structure: "
                        + "; ".join(structural_collisions)
                        + ". Try a clear supported position."
                    )
                else:
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
