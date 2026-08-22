"""Selection and error handling for furniture snapping algorithms."""

import logging
import time

import numpy as np

from omegaconf import DictConfig

from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject
from scenesmith.furniture_agents.tools.response_dataclasses import (
    FurnitureErrorType,
    SnapToObjectResult,
)
from scenesmith.furniture_agents.tools.snapping_helpers import (
    compute_snap_direction_mesh_to_mesh,
    snap_mesh_to_aabb,
    snap_mesh_to_aabb_along_axis,
    snap_with_iterative_collision_check,
)

console_logger = logging.getLogger(__name__)


def select_and_execute_snap_algorithm(
    obj: SceneObject,
    target: SceneObject,
    orientation: str,
    orientation_applied: bool,
    object_id: str,
    target_id: str,
    cfg: DictConfig,
) -> tuple[np.ndarray, float] | str:
    """Select and execute the appropriate snapping algorithm.

    Args:
        obj: Object to snap.
        target: Target to snap to.
        orientation: Orientation mode ("toward", "away", or "none").
        orientation_applied: Whether orientation was applied.
        object_id: Object ID string (for error messages).
        target_id: Target ID string (for error messages).
        cfg: Configuration object.

    Returns:
        Tuple of (movement_vector, distance) if successful,
        or error JSON string if failed.
    """
    start_time = time.time()
    try:
        if orientation_applied:
            # Axis-constrained snapping: move only along facing direction.
            # This preserves the facing relationship established.

            # Compute axis direction from object's current rotation.
            # For "toward": +Y axis (forward).
            # For "away": -Y axis (backward).
            local_axis = np.array([0.0, 1.0 if orientation == "toward" else -1.0, 0.0])
            axis_world = obj.transform.rotation() @ local_axis

            if target.geometry_path and target.sdf_path:
                # Target has mesh geometry: use iterative collision checking.
                movement_vector, distance = snap_with_iterative_collision_check(
                    obj=obj, target=target, direction=axis_world, cfg=cfg
                )
                algorithm = "iterative-axis-constrained"
                console_logger.info(
                    f"Using iterative axis-constrained snapping (mesh target, "
                    f"orientation='{orientation}')"
                )
            else:
                # Target is AABB (wall): use fast single-step approach.
                movement_vector, distance = snap_mesh_to_aabb_along_axis(
                    obj=obj, target=target, axis_world=axis_world, cfg=cfg
                )
                algorithm = "axis-constrained"
                console_logger.info(
                    f"Using axis-constrained snapping (AABB target, "
                    f"orientation='{orientation}')"
                )
        else:
            # Closest-point snapping.
            if obj.geometry_path and target.geometry_path:
                # Use iterative mesh-to-mesh algorithm.
                # Compute direction from closest points on visual geometry.
                direction = compute_snap_direction_mesh_to_mesh(
                    obj=obj, target=target, cfg=cfg
                )
                # Use iterative collision checking to find safe snap distance.
                movement_vector, distance = snap_with_iterative_collision_check(
                    obj=obj, target=target, direction=direction, cfg=cfg
                )
                algorithm = "iterative-mesh-to-mesh"
            elif obj.geometry_path:
                # Use mesh-to-AABB algorithm.
                movement_vector, distance = snap_mesh_to_aabb(obj, target, cfg)
                algorithm = "mesh-to-AABB"
            else:
                # Object missing geometry.
                return SnapToObjectResult(
                    success=False,
                    message=f"{obj.name} missing geometry_path - cannot snap",
                    object_id=object_id,
                    target_id=target_id,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                    suggested_action=(
                        "Snap requires 3D geometry. Use a different object that has "
                        "geometry"
                    ),
                ).to_json()

            console_logger.info(f"Using closest-point snapping (orientation='none')")
    except Exception as e:
        console_logger.error(f"Error computing snap: {e}")
        return SnapToObjectResult(
            success=False,
            message=f"Failed to compute snap: {str(e)}",
            object_id=object_id,
            target_id=target_id,
            suggested_action="Check object and target IDs are valid and have geometry",
        ).to_json()

    console_logger.info(
        f"Snap computation ({algorithm}): distance={distance:.3f}m, "
        f"computed in {time.time() - start_time:.2f}s"
    )

    return (movement_vector, distance)
