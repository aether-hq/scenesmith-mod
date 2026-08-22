import logging

from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.physics.feasibility.surface_projection import (
    apply_surface_projection,
)
from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject, UniqueID
from scenesmith.manipuland_agents.tools.response_dataclasses import (
    ManipulandErrorType,
    ManipulandPlacementResult,
    PenetrationResolutionResult,
    Position2D,
    Position3D,
    Rotation3D,
)

console_logger = logging.getLogger(__name__)


class ManipulandResolutionMixin:
    """Penetration-resolution operation and failure-result construction."""

    @log_scene_action
    def _resolve_penetrations_impl(self, object_ids: list[str], **kwargs) -> str:
        """Implementation for resolving penetrations.

        All rotations (roll, pitch, yaw) are fixed. Only XY translation is solved.
        Surface is inferred from the objects.
        """
        console_logger.info(
            f"Tool called: resolve_penetrations with object IDs: {object_ids}"
        )

        # Empty list is a no-op.
        if len(object_ids) == 0:
            console_logger.warning("resolve_penetrations: No objects to resolve")
            return PenetrationResolutionResult(
                success=True,
                message="No objects to resolve",
                num_objects_considered=0,
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
            ).to_json()

        # Convert string IDs to UniqueID and validate they exist.
        unique_ids: list[UniqueID] = []
        scene_objects: list[SceneObject] = []
        for obj_id in object_ids:
            unique_id = UniqueID(obj_id)
            scene_obj = self.scene.get_object(unique_id)
            if scene_obj is None:
                console_logger.warning(
                    f"resolve_penetrations: Object {obj_id} not found in scene"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=f"Object {obj_id} not found in scene",
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.OBJECT_NOT_FOUND,
                ).to_json()
            unique_ids.append(unique_id)
            scene_objects.append(scene_obj)

        # Infer surface from objects - all must be on same surface.
        # Also validate all objects are on surfaces of the current furniture.
        surface_ids = set()
        valid_surface_ids = set(self.support_surfaces.keys())
        for scene_obj in scene_objects:
            if scene_obj.placement_info is None:
                console_logger.warning(
                    f"resolve_penetrations: Object {scene_obj.object_id} has no placement info"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=f"Object {scene_obj.object_id} has no placement info",
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                ).to_json()

            parent_surface = str(scene_obj.placement_info.parent_surface_id)
            if parent_surface not in valid_surface_ids:
                console_logger.warning(
                    f"resolve_penetrations: Object {scene_obj.object_id} is on surface "
                    f"{parent_surface} which is not on furniture {self.current_furniture_id}"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=(
                        f"Object {scene_obj.object_id} is on surface {parent_surface} "
                        f"which is not on the current furniture {self.current_furniture_id}"
                    ),
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.INVALID_SURFACE,
                ).to_json()

            surface_ids.add(parent_surface)

        # Check for unsupported composite types (piles).
        # Piles have scattered members whose footprints cannot be accurately computed.
        for scene_obj in scene_objects:
            if scene_obj.metadata.get("composite_type") == "pile":
                console_logger.warning(
                    f"resolve_penetrations: Cannot include pile '{scene_obj.object_id}' - "
                    "piles have scattered members with complex footprints"
                )
                return PenetrationResolutionResult(
                    success=False,
                    message=(
                        f"Cannot resolve penetrations for pile '{scene_obj.object_id}'. "
                        "Piles have scattered members whose combined footprint cannot be "
                        "accurately computed for collision resolution.\n\n"
                        "Instead:\n"
                        "1. Move other objects away from the pile, OR\n"
                        "2. Remove and recreate the pile at a different location"
                    ),
                    num_objects_considered=len(object_ids),
                    num_objects_moved=0,
                    moved_object_ids=[],
                    max_displacement_m=0.0,
                    error_type=ManipulandErrorType.UNSUPPORTED_COMPOSITE_TYPE,
                ).to_json()

        if len(surface_ids) > 1:
            console_logger.warning(
                f"resolve_penetrations: Objects are on {len(surface_ids)} different "
                f"surfaces: {sorted(surface_ids)}"
            )
            return PenetrationResolutionResult(
                success=False,
                message=(
                    f"Objects are on {len(surface_ids)} different surfaces: "
                    f"{sorted(surface_ids)}. All objects must be on the same surface."
                ),
                num_objects_considered=len(object_ids),
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.OBJECTS_ON_DIFFERENT_SURFACES,
            ).to_json()

        surface_id = surface_ids.pop()
        surface = self.support_surfaces[surface_id]

        # Call projection (rotations always fixed, only XY translation solved).
        try:
            _, success, moved_ids, max_displacement = apply_surface_projection(
                scene=self.scene,
                surface=surface,
                object_ids=unique_ids,
                influence_distance=self.cfg.penetration_resolution.influence_distance,
                solver_name=self.cfg.penetration_resolution.solver_name,
                iteration_limit=self.cfg.penetration_resolution.iteration_limit,
                time_limit_s=self.cfg.penetration_resolution.time_limit_s,
            )
        except Exception as e:
            console_logger.error(f"Surface projection failed: {e}", exc_info=True)
            return PenetrationResolutionResult(
                success=False,
                message=f"Physics resolution failed with error: {e}",
                num_objects_considered=len(object_ids),
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.PHYSICS_RESOLUTION_FAILED,
            ).to_json()

        # Build result.
        if success:
            moved_id_strs = [str(uid) for uid in moved_ids]
            return PenetrationResolutionResult(
                success=True,
                message=f"Resolved penetrations. Moved {len(moved_ids)} objects.",
                num_objects_considered=len(object_ids),
                num_objects_moved=len(moved_ids),
                moved_object_ids=moved_id_strs,
                max_displacement_m=max_displacement,
            ).to_json()
        else:
            return PenetrationResolutionResult(
                success=False,
                message=(
                    "Failed to resolve penetrations - objects cannot fit on surface.\n\n"
                    "Recovery options:\n"
                    "1. Remove some objects from the surface\n"
                    "2. Use a larger surface\n"
                    "3. Re-orient objects before calling (e.g., rotate knife parallel to edge)"
                ),
                num_objects_considered=len(object_ids),
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.PHYSICS_RESOLUTION_FAILED,
            ).to_json()

    def _create_placement_failure_result(
        self, asset_id: str, message: str, error_type: ManipulandErrorType | None
    ) -> str:
        """Create failure result for placement operations."""
        result = ManipulandPlacementResult(
            success=False,
            message=message,
            asset_id=asset_id,
            object_id="",
            world_position=Position3D(x=0.0, y=0.0, z=0.0),
            world_rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=0.0),
            surface_position=Position2D(x=0.0, y=0.0),
            surface_rotation_deg=0.0,
            parent_surface_id="",  # No surface association for error results.
            has_geometry=False,
            error_type=error_type,
        )
        return result.to_json()
