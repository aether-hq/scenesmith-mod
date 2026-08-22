"""Physical feasibility post-processing for scene collision resolution.

This module provides two-stage post-processing adapted from scene_gen repository:
1. Projection - IK-based collision resolution with configurable DOF constraints
2. Simulation - Physics settling to static equilibrium (always full 6DOF)

See: https://github.com/nepfaff/steerable-scene-generation/blob/main/steerable_scene_generation/algorithms/scene_diffusion/postprocessing.py
"""

import logging

from pathlib import Path

import numpy as np

from omegaconf import DictConfig
from pydrake.all import RigidTransform

from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.feasibility.ik import compute_tilt_angle_degrees
from scenesmith.agent_utils.physics.feasibility.projection import (
    _apply_floor_penetration_fallback,
    apply_non_penetration_projection,
)
from scenesmith.agent_utils.physics.feasibility.simulation import (
    apply_forward_simulation,
)


def apply_physical_feasibility_postprocessing(
    scene: RoomScene,
    weld_furniture: bool,
    projection_enabled: bool = True,
    projection_influence_distance: float = 0.02,
    projection_solver_name: str = "snopt",
    projection_iteration_limit: int = 5000,
    projection_time_limit_s: float = 360.0,
    projection_xy_only: bool = True,
    projection_fix_rotation: bool = True,
    large_scene_optimization_threshold: int = 100,
    collision_penetration_threshold_m: float = 0.001,
    simulation_enabled: bool = True,
    simulation_time_s: float = 5.0,
    simulation_time_step_s: float = 1e-3,
    simulation_timeout_s: float = 300.0,
    simulation_html_path: Path | None = None,
    remove_fallen_furniture: bool = False,
    fallen_tilt_threshold_degrees: float = 45.0,
    remove_fallen_manipulands: bool = False,
    fallen_manipuland_floor_z: float = -0.5,
    fallen_manipuland_near_floor_z: float = 0.02,
    fallen_manipuland_z_displacement: float = 0.3,
    validation_object_penetration_threshold_m: float = 0.001,
    validation_floor_penetration_tolerance_m: float = 0.05,
    fallback_max_translation_m: float = 0.05,
    fallback_max_tilt_delta_degrees: float = 5.0,
) -> tuple[RoomScene, bool, list[UniqueID]]:
    """Apply complete physical feasibility post-processing pipeline.

    Combines projection (Stage 1) and simulation (Stage 2) with graceful
    error handling. On any failure, returns original scene unchanged.

    Args:
        scene: RoomScene to process.
        weld_furniture: If True, weld furniture (process manipulands only).
        projection_enabled: Whether to run projection stage.
        projection_influence_distance: Collision influence distance.
        projection_solver_name: NLP solver name.
        projection_iteration_limit: Max solver iterations.
        projection_time_limit_s: Max solver time in seconds.
        projection_xy_only: If True, only optimize XY translation.
        projection_fix_rotation: If True, fix rotations during projection.
        large_scene_optimization_threshold: When scene has more objects than
            this threshold, only colliding objects are made free in IK.
        collision_penetration_threshold_m: Minimum penetration depth (meters)
            to consider an object as colliding for large scene optimization.
        simulation_enabled: Whether to run simulation stage.
        simulation_time_s: Simulation duration.
        simulation_time_step_s: Physics time step.
        simulation_timeout_s: Max wall-clock time for simulation.
        simulation_html_path: If provided, save meshcat visualization to this HTML file.
        remove_fallen_furniture: If True, remove furniture that falls over during
            simulation.
        fallen_tilt_threshold_degrees: Tilt angle threshold for fallen detection.
        remove_fallen_manipulands: If True, remove manipuland objects that fall
            off furniture surfaces during simulation.
        fallen_manipuland_floor_z: Absolute Z threshold for floor penetration.
        fallen_manipuland_near_floor_z: Object bottom below this Z is on floor.
        fallen_manipuland_z_displacement: Z drop threshold for detecting falling.
        validation_object_penetration_threshold_m: Collision threshold used to
            validate a simulated fallback after a projection-solver failure.
        validation_floor_penetration_tolerance_m: Allowed support settling depth
            during simulated-fallback validation.
        fallback_max_translation_m: Maximum furniture displacement allowed when
            recovering from a projection-solver failure.
        fallback_max_tilt_delta_degrees: Maximum furniture tilt change allowed
            when recovering from a projection-solver failure.

    Returns:
        Tuple of (processed_scene, projection_success, removed_ids).
    """
    # Furniture placement is already a valid deterministic checkpoint. Preserve
    # it transactionally: a closed room-shell collision mesh can make Drake
    # interpret the whole room interior as solid and "resolve" every object by
    # ejecting it through a wall or the floor. Those invalid results must never
    # replace the last valid layout.
    original_furniture_transforms: dict[UniqueID, RigidTransform] = {}
    if not weld_furniture:
        original_furniture_transforms = {
            obj.object_id: RigidTransform(
                obj.transform.rotation(), obj.transform.translation().copy()
            )
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
        }

    owner_bound_relative_transforms: dict[UniqueID, tuple[UniqueID, RigidTransform]] = (
        {}
    )
    for obj in scene.objects.values():
        owner_value = (obj.metadata or {}).get("dense_library_owner_bound")
        if not owner_value:
            continue
        owner_id = UniqueID(str(owner_value))
        owner = scene.get_object(owner_id)
        if owner is None:
            continue
        owner_bound_relative_transforms[obj.object_id] = (
            owner_id,
            owner.transform.inverse() @ obj.transform,
        )

    def restore_owner_bound_relative_transforms() -> None:
        for object_id, (
            owner_id,
            relative_transform,
        ) in owner_bound_relative_transforms.items():
            obj = scene.get_object(object_id)
            owner = scene.get_object(owner_id)
            if obj is None or owner is None:
                continue
            obj.transform = owner.transform @ relative_transform

    def find_ejected_furniture() -> list[UniqueID]:
        if not original_furniture_transforms:
            return []
        floor_bounds = (
            scene.room_geometry.floor.compute_world_bounds()
            if scene.room_geometry is not None and scene.room_geometry.floor is not None
            else None
        )
        if floor_bounds is None:
            return []
        floor_min, floor_max = floor_bounds
        floor_z = float(floor_max[2])
        ceiling_z = floor_z + float(scene.room_geometry.wall_height)
        ejected: list[UniqueID] = []
        for object_id, original_transform in original_furniture_transforms.items():
            obj = scene.get_object(object_id)
            if obj is None:
                continue
            world_bounds = obj.compute_world_bounds()
            if world_bounds is None:
                continue
            bounds_min, bounds_max = world_bounds
            object_height = max(float(bounds_max[2] - bounds_min[2]), 0.1)
            position = obj.transform.translation()
            original_position = original_transform.translation()
            excessive_drop = float(original_position[2] - position[2]) > max(
                0.5, object_height
            )
            outside_vertical = (
                float(position[2]) < floor_z - max(0.25, object_height * 0.5)
                or float(bounds_min[2]) > ceiling_z + 0.25
            )
            outside_horizontal = (
                float(position[0]) < float(floor_min[0]) - 0.25
                or float(position[0]) > float(floor_max[0]) + 0.25
                or float(position[1]) < float(floor_min[1]) - 0.25
                or float(position[1]) > float(floor_max[1]) + 0.25
            )
            if not (excessive_drop or outside_vertical or outside_horizontal):
                continue
            ejected.append(object_id)
        return ejected

    def restore_ejected_furniture(object_ids: list[UniqueID], operation: str) -> None:
        for object_id in object_ids:
            obj = scene.get_object(object_id)
            if obj is None:
                continue
            position = obj.transform.translation().copy()
            original_transform = original_furniture_transforms[object_id]
            obj.transform = RigidTransform(
                original_transform.rotation(), original_transform.translation().copy()
            )
            console_logger.error(
                "%s ejected furniture %s to (%.3f, %.3f, %.3f); restored its "
                "last valid deterministic pose",
                operation,
                object_id,
                position[0],
                position[1],
                position[2],
            )

    projection_success = True
    projection_attempt_failed = False
    removed_ids: list[UniqueID] = []
    # Stage 1: Projection.
    if projection_enabled:
        scene, projection_success = apply_non_penetration_projection(
            scene=scene,
            influence_distance=projection_influence_distance,
            solver_name=projection_solver_name,
            iteration_limit=projection_iteration_limit,
            time_limit_s=projection_time_limit_s,
            weld_furniture=weld_furniture,
            xy_only=projection_xy_only,
            fix_rotation=projection_fix_rotation,
            large_scene_optimization_threshold=large_scene_optimization_threshold,
            collision_penetration_threshold_m=collision_penetration_threshold_m,
        )
        projection_attempt_failed = not projection_success
        restore_owner_bound_relative_transforms()

        if not projection_success and not weld_furniture:
            # Only apply floor fallback when furniture is free to move.
            # When weld_furniture=True, furniture is fixed and can't penetrate floor.
            console_logger.warning(
                "Projection failed, applying floor penetration fallback"
            )
            # Lift furniture above floor to prevent tipping during simulation.
            scene, lifted_count = _apply_floor_penetration_fallback(
                scene=scene, margin_m=0.001
            )
            if lifted_count > 0:
                console_logger.info(
                    f"Floor fallback: lifted {lifted_count} furniture piece(s)"
                )
            else:
                console_logger.warning(
                    "Floor fallback: no floor penetrations found, "
                    "projection may have failed for other reasons"
                )
        elif not projection_success:
            console_logger.warning(
                "Projection failed (furniture welded, skipping floor fallback)"
            )

    projection_ejected = find_ejected_furniture()
    projection_restored = 0
    initial_furniture_count = len(original_furniture_transforms)
    isolated_projection_ejection = (
        projection_success
        and len(projection_ejected) == 1
        and initial_furniture_count > 0
        and len(projection_ejected) / initial_furniture_count <= 0.1
    )
    if isolated_projection_ejection:
        object_id = projection_ejected[0]
        scene.remove_object(object_id)
        removed_ids.append(object_id)
        console_logger.warning(
            "Removed isolated furniture item %s after projection ejected it; "
            "%d of %d initial furniture items were invalid",
            object_id,
            len(projection_ejected),
            initial_furniture_count,
        )
    elif projection_ejected:
        restore_ejected_furniture(projection_ejected, "Projection")
        projection_restored = len(projection_ejected)
        projection_success = False
        console_logger.error(
            "Rejected invalid projection output for %d furniture item(s); "
            "skipping forward simulation for this checkpoint",
            projection_restored,
        )
    restore_owner_bound_relative_transforms()

    pre_simulation_furniture_transforms: dict[UniqueID, RigidTransform] = {}
    if not weld_furniture and not projection_restored:
        pre_simulation_furniture_transforms = {
            obj.object_id: RigidTransform(
                obj.transform.rotation(), obj.transform.translation().copy()
            )
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
        }

    # Stage 2: Simulation (runs regardless of projection result).
    if simulation_enabled and not projection_restored:
        scene, simulation_removed_ids = apply_forward_simulation(
            scene=scene,
            simulation_time_s=simulation_time_s,
            time_step_s=simulation_time_step_s,
            timeout_s=simulation_timeout_s,
            weld_furniture=weld_furniture,
            output_html_path=simulation_html_path,
            remove_fallen_furniture=remove_fallen_furniture,
            fallen_tilt_threshold_degrees=fallen_tilt_threshold_degrees,
            remove_fallen_manipulands=remove_fallen_manipulands,
            fallen_manipuland_floor_z=fallen_manipuland_floor_z,
            fallen_manipuland_near_floor_z=fallen_manipuland_near_floor_z,
            fallen_manipuland_z_displacement=fallen_manipuland_z_displacement,
        )
        removed_ids.extend(simulation_removed_ids)
        restore_owner_bound_relative_transforms()

    simulation_ejected = find_ejected_furniture()
    restore_ejected_furniture(simulation_ejected, "Simulation")
    restore_owner_bound_relative_transforms()
    simulation_restored = len(simulation_ejected)
    if simulation_restored:
        projection_success = False
        console_logger.error(
            "Rejected invalid simulation output for %d furniture item(s)",
            simulation_restored,
        )

    if simulation_enabled and pre_simulation_furniture_transforms:
        post_simulation_collisions = compute_scene_collisions(
            scene=scene,
            penetration_threshold=validation_object_penetration_threshold_m,
            floor_penetration_tolerance=validation_floor_penetration_tolerance_m,
            current_furniture_id=None,
        )
        if post_simulation_collisions:
            colliding_furniture_ids: set[UniqueID] = set()
            for collision in post_simulation_collisions:
                for object_id in (
                    getattr(collision, "object_a_id", None),
                    getattr(collision, "object_b_id", None),
                ):
                    if object_id is None:
                        continue
                    unique_id = UniqueID(object_id)
                    obj = scene.get_object(unique_id)
                    if (
                        obj is not None
                        and obj.object_type == ObjectType.FURNITURE
                        and unique_id in pre_simulation_furniture_transforms
                    ):
                        colliding_furniture_ids.add(unique_id)
            for object_id in colliding_furniture_ids:
                obj = scene.get_object(object_id)
                if obj is None:
                    continue
                transform = pre_simulation_furniture_transforms[object_id]
                obj.transform = RigidTransform(
                    transform.rotation(), transform.translation().copy()
                )
            restore_owner_bound_relative_transforms()
            repaired_collisions = compute_scene_collisions(
                scene=scene,
                penetration_threshold=validation_object_penetration_threshold_m,
                floor_penetration_tolerance=validation_floor_penetration_tolerance_m,
                current_furniture_id=None,
            )
            if repaired_collisions:
                projection_success = False
                console_logger.error(
                    "Post-simulation furniture collision repair remains invalid: "
                    "%d collision(s) after restoring %d implicated item(s)",
                    len(repaired_collisions),
                    len(colliding_furniture_ids),
                )
            else:
                console_logger.warning(
                    "Restored %d furniture item(s) after simulation introduced "
                    "%d collision(s); authoritative full-scene recheck passed",
                    len(colliding_furniture_ids),
                    len(post_simulation_collisions),
                )

    if (
        projection_attempt_failed
        and not weld_furniture
        and simulation_enabled
        and not projection_restored
    ):
        post_simulation_collisions = compute_scene_collisions(
            scene=scene,
            penetration_threshold=validation_object_penetration_threshold_m,
            floor_penetration_tolerance=validation_floor_penetration_tolerance_m,
            current_furniture_id=None,
        )
        max_translation = 0.0
        max_tilt_delta = 0.0
        for object_id, original_transform in original_furniture_transforms.items():
            obj = scene.get_object(object_id)
            if obj is None:
                continue
            translation = float(
                np.linalg.norm(
                    obj.transform.translation() - original_transform.translation()
                )
            )
            tilt_delta = abs(
                compute_tilt_angle_degrees(obj.transform)
                - compute_tilt_angle_degrees(original_transform)
            )
            max_translation = max(max_translation, translation)
            max_tilt_delta = max(max_tilt_delta, tilt_delta)

        clean_bounded_simulation = (
            not post_simulation_collisions
            and not removed_ids
            and not simulation_ejected
            and max_translation <= fallback_max_translation_m
            and max_tilt_delta <= fallback_max_tilt_delta_degrees
        )
        if clean_bounded_simulation:
            projection_success = True
            console_logger.warning(
                "Projection solver failed, but clean bounded simulation recovered "
                "the furniture checkpoint (max translation %.3fm, max tilt %.2f°)",
                max_translation,
                max_tilt_delta,
            )
        else:
            projection_success = False
            console_logger.error(
                "Projection solver failure remains unrecovered after simulation: "
                "%d collision(s), %d removed, %d ejected, max translation %.3fm, "
                "max tilt %.2f°",
                len(post_simulation_collisions),
                len(removed_ids),
                len(simulation_ejected),
                max_translation,
                max_tilt_delta,
            )

    if projection_attempt_failed and weld_furniture and simulation_enabled:
        # The nonlinear projection solver is a repair mechanism, not the
        # physical-validity oracle. Manipulands are intentionally allowed to
        # settle during forward simulation. If the authoritative post-simulation
        # collision check is clean and nothing fell or was ejected, the scene has
        # positive physics evidence even when the optimizer itself timed out.
        post_simulation_collisions = compute_scene_collisions(
            scene=scene,
            penetration_threshold=validation_object_penetration_threshold_m,
            floor_penetration_tolerance=validation_floor_penetration_tolerance_m,
            current_furniture_id=None,
        )
        clean_simulated_scene = (
            not post_simulation_collisions
            and not removed_ids
            and not simulation_ejected
        )
        if clean_simulated_scene:
            projection_success = True
            console_logger.warning(
                "Projection solver failed, but authoritative collision validation "
                "accepted the clean post-simulation scene"
            )
        else:
            projection_success = False
            console_logger.error(
                "Projection solver failure remains unrecovered after welded-"
                "furniture simulation: %d collision(s), %d removed, %d ejected",
                len(post_simulation_collisions),
                len(removed_ids),
                len(simulation_ejected),
            )

    restore_owner_bound_relative_transforms()
    return scene, projection_success, removed_ids


def apply_per_furniture_postprocessing(
    full_scene: RoomScene,
    furniture_id: UniqueID,
    config: DictConfig,
    simulation_html_path: Path | None = None,
) -> RoomScene:
    """Run post-processing for a single furniture piece and its manipulands.

    Creates a subset scene containing only the target furniture, its manipulands,
    and room structure (walls/floor), then runs the full post-processing pipeline.
    The manipuland poses are then merged back into the full scene.

    This enables solving smaller, more tractable subproblems before the final
    combined post-processing pass.

    Args:
        full_scene: The complete scene with all objects.
        furniture_id: ID of the furniture piece to process.
        config: Post-processing configuration with projection and simulation settings.
        simulation_html_path: If provided, save meshcat visualization to this HTML file.

    Returns:
        The full scene with updated manipuland poses.
    """
    from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject

    # Build subset scene: walls + floor + this furniture + its manipulands.
    subset_objects: dict[UniqueID, SceneObject] = {}

    # Add walls and floor.
    for obj in full_scene.objects.values():
        if obj.object_type in [ObjectType.WALL, ObjectType.FLOOR]:
            subset_objects[obj.object_id] = obj

    # Add target furniture.
    furniture = full_scene.get_object(furniture_id)
    if furniture is None:
        console_logger.warning(
            f"Furniture {furniture_id} not found, skipping per-furniture post-processing"
        )
        return full_scene

    subset_objects[furniture.object_id] = furniture

    # Add manipulands on this furniture's surfaces.
    manipuland_ids: list[UniqueID] = []
    for surface in furniture.support_surfaces:
        for manip in full_scene.get_objects_on_surface(surface.surface_id):
            subset_objects[manip.object_id] = manip
            manipuland_ids.append(manip.object_id)

    # Skip if no manipulands to process.
    if not manipuland_ids:
        console_logger.info(
            f"No manipulands on furniture {furniture_id}, skipping post-processing"
        )
        return full_scene

    console_logger.info(
        f"Running per-furniture post-processing for {furniture_id} "
        f"with {len(manipuland_ids)} manipuland(s)"
    )

    # Create subset scene.
    subset_scene = RoomScene(
        room_geometry=full_scene.room_geometry,
        scene_dir=full_scene.scene_dir,
        objects=subset_objects,
        text_description=full_scene.text_description,
    )

    # Run post-processing on subset (furniture welded).
    # Fallen furniture removal not needed: furniture is welded here.
    projection_cfg = config.projection
    simulation_cfg = config.simulation
    processed_subset, success, _ = apply_physical_feasibility_postprocessing(
        scene=subset_scene,
        weld_furniture=True,
        projection_enabled=projection_cfg.enabled,
        projection_influence_distance=projection_cfg.influence_distance,
        projection_solver_name=projection_cfg.solver_name,
        projection_iteration_limit=projection_cfg.iteration_limit,
        projection_time_limit_s=projection_cfg.time_limit_s,
        projection_xy_only=projection_cfg.xy_only,
        projection_fix_rotation=projection_cfg.fix_rotation,
        simulation_enabled=simulation_cfg.enabled,
        simulation_time_s=simulation_cfg.simulation_time_s,
        simulation_time_step_s=simulation_cfg.time_step_s,
        simulation_timeout_s=simulation_cfg.timeout_s,
        simulation_html_path=simulation_html_path,
        remove_fallen_furniture=False,
        remove_fallen_manipulands=simulation_cfg.remove_fallen_manipulands,
        fallen_manipuland_floor_z=simulation_cfg.fallen_manipuland_floor_z,
        fallen_manipuland_near_floor_z=simulation_cfg.fallen_manipuland_near_floor_z,
        fallen_manipuland_z_displacement=simulation_cfg.fallen_manipuland_z_displacement,
    )

    if not success:
        console_logger.warning(
            f"Per-furniture post-processing for {furniture_id} had issues"
        )

    # Merge manipuland poses back to full scene.
    for manip_id in manipuland_ids:
        processed_manip = processed_subset.get_object(manip_id)
        if processed_manip:
            full_scene.move_object(
                object_id=manip_id, new_transform=processed_manip.transform
            )
        else:
            # Object was removed during post-processing (e.g., fell off furniture).
            full_scene.remove_object(manip_id)

    return full_scene
