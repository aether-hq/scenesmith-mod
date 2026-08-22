"""Fill container utilities using physics simulation.

This module provides functionality for:
- Computing container interior bounds using top rim heuristic.
- Computing fill object spawn transforms.
- Resolving initial fill object collisions using NLP projection.
- Simulating fill objects dropping into containers using Drake physics.
"""

import logging

import numpy as np
import trimesh

from pydrake.all import RigidTransform

from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject

console_logger = logging.getLogger(__name__)

from scenesmith.manipuland_agents.tools.fill.bounds import ContainerInteriorBounds
from scenesmith.manipuland_agents.tools.fill.packing import (
    project_fill_objects_non_penetrating,
)
from scenesmith.manipuland_agents.tools.fill.simulation import simulate_fill_physics
from scenesmith.manipuland_agents.tools.fill.transforms import (
    compute_fill_spawn_transforms,
)


def compute_bbox_corners(
    bbox_min: np.ndarray, bbox_max: np.ndarray
) -> list[np.ndarray]:
    """Compute the 8 corners of an axis-aligned bounding box."""
    return [
        np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
    ]


def run_fill_simulation_loop(
    container_scene_obj: SceneObject,
    container_transform: RigidTransform,
    container_interior: ContainerInteriorBounds,
    fill_scene_objects: list[SceneObject],
    fill_collision_meshes: list[list[trimesh.Trimesh]],
    max_iterations: int,
    spawn_height_above_rim: float,
    height_stagger_fraction: float,
    min_height_stagger: float,
    nlp_influence_distance: float,
    nlp_solver_name: str,
    catch_floor_z: float,
    inside_z_threshold: float,
    simulation_time: float,
    simulation_time_step: float,
    max_nan_retries: int = 3,
) -> tuple[list[int], list[RigidTransform]]:
    """Run iterative fill simulation loop with retry for objects that fall out.

    Runs physics simulation iteratively, respawning objects that fall outside
    the container until all objects are settled or max iterations reached.

    Args:
        container_scene_obj: Temporary SceneObject for the container.
        container_transform: Container's world transform.
        container_interior: Interior bounds from compute_container_interior_bounds.
        fill_scene_objects: List of temporary SceneObjects for fill items.
        fill_collision_meshes: Collision meshes for each fill item.
        max_iterations: Maximum retry iterations.
        spawn_height_above_rim: Height above rim to spawn fill objects.
        height_stagger_fraction: Fraction of bbox diagonal for Z spacing.
        min_height_stagger: Minimum stagger between objects (meters).
        nlp_influence_distance: Distance threshold for NLP collision influence.
        nlp_solver_name: NLP solver name ("snopt" or "ipopt").
        catch_floor_z: Z position of catch floor.
        inside_z_threshold: Z threshold for inside detection.
        simulation_time: Simulation duration in seconds.
        simulation_time_step: Simulation time step in seconds.
        max_nan_retries: Max retries on NaN simulation errors (different seeds).

    Returns:
        Tuple of (inside_indices, final_fill_transforms) where inside_indices
        are indices of objects that ended up inside the container.
    """
    console_logger.info(
        f"Running fill simulation loop with {max_iterations} iterations"
    )
    inside_indices: list[int] = []
    final_fill_transforms: list[RigidTransform] = [RigidTransform()] * len(
        fill_scene_objects
    )
    remaining_indices = list(range(len(fill_scene_objects)))

    # Track settled objects for subsequent iterations.
    settled_objects: list[SceneObject] = []
    settled_transforms: list[RigidTransform] = []
    settled_indices: list[int] = []  # Original indices for settled objects.

    rng = np.random.default_rng()

    for iteration in range(max_iterations):
        if not remaining_indices:
            break

        console_logger.info(
            f"Fill iteration {iteration + 1}/{max_iterations}: "
            f"{len(remaining_indices)} objects to place, "
            f"{len(settled_objects)} already settled"
        )

        # Get remaining fill objects and their meshes.
        remaining_objects = [fill_scene_objects[i] for i in remaining_indices]
        remaining_meshes = [fill_collision_meshes[i] for i in remaining_indices]

        # Retry loop for NaN simulation errors (different spawn positions).
        sim_result = None
        for nan_retry in range(max_nan_retries):
            # Compute spawn transforms for remaining objects.
            spawn_transforms = compute_fill_spawn_transforms(
                fill_collision_meshes=remaining_meshes,
                container_interior=container_interior,
                container_transform=container_transform,
                spawn_height_above_rim=spawn_height_above_rim,
                height_stagger_fraction=height_stagger_fraction,
                min_height_stagger=min_height_stagger,
                rng=rng,
            )

            # Run NLP projection to resolve overlaps.
            projected_transforms, _ = project_fill_objects_non_penetrating(
                fill_scene_objects=remaining_objects,
                fill_initial_transforms=spawn_transforms,
                influence_distance=nlp_influence_distance,
                solver_name=nlp_solver_name,
            )

            # Run physics simulation with settled objects from previous iterations.
            sim_result = simulate_fill_physics(
                container_scene_object=container_scene_obj,
                container_transform=container_transform,
                new_fill_objects=remaining_objects,
                new_fill_transforms=projected_transforms,
                settled_fill_objects=settled_objects if settled_objects else None,
                settled_fill_transforms=(
                    settled_transforms if settled_transforms else None
                ),
                catch_floor_z=catch_floor_z,
                inside_z_threshold=inside_z_threshold,
                simulation_time=simulation_time,
                simulation_time_step=simulation_time_step,
            )

            # Check for NaN error and retry with different spawn positions.
            if sim_result.error_message and "nan" in sim_result.error_message.lower():
                console_logger.warning(
                    f"Simulation NaN error (attempt {nan_retry + 1}/{max_nan_retries})"
                )
                if nan_retry < max_nan_retries - 1:
                    console_logger.info("Retrying with different spawn positions...")
                    continue
            break  # Success or non-NaN error.

        if sim_result.error_message:
            console_logger.error(f"Fill simulation error: {sim_result.error_message}")
            # Continue to next iteration anyway.
            continue

        # Update settled object transforms (they may have shifted due to new objects).
        if sim_result.settled_final_transforms:
            for i, updated_transform in enumerate(sim_result.settled_final_transforms):
                settled_transforms[i] = updated_transform
                # Also update the final transforms dict.
                original_idx = settled_indices[i]
                final_fill_transforms[original_idx] = updated_transform

        # Map simulation results back to original indices.
        new_inside = []
        new_outside = []

        # Handle settled objects that were pushed out of container.
        if sim_result.settled_fell_out_indices:
            console_logger.info(
                f"{len(sim_result.settled_fell_out_indices)} settled objects "
                "were pushed out by new objects"
            )
            # Remove from inside_indices and add back to remaining for retry.
            # Process in reverse to avoid index shifting issues.
            for settled_idx in sorted(
                sim_result.settled_fell_out_indices, reverse=True
            ):
                original_idx = settled_indices[settled_idx]
                if original_idx in inside_indices:
                    inside_indices.remove(original_idx)
                # Add to new_outside for retry in next iteration.
                new_outside.append(original_idx)
                # Remove from settled tracking lists.
                del settled_objects[settled_idx]
                del settled_transforms[settled_idx]
                del settled_indices[settled_idx]

        for local_idx, final_transform in enumerate(sim_result.final_transforms):
            original_idx = remaining_indices[local_idx]
            if local_idx in sim_result.inside_indices:
                inside_indices.append(original_idx)
                final_fill_transforms[original_idx] = final_transform
                new_inside.append(original_idx)
                # Add to settled objects for next iteration.
                settled_objects.append(fill_scene_objects[original_idx])
                settled_transforms.append(final_transform)
                settled_indices.append(original_idx)
            else:
                new_outside.append(original_idx)

        # Update remaining for next iteration.
        remaining_indices = new_outside

        console_logger.info(
            f"Iteration {iteration + 1}: {len(new_inside)} inside, "
            f"{len(new_outside)} outside"
        )

    return inside_indices, final_fill_transforms


def compute_composite_bbox_in_local_frame(
    container_asset: SceneObject,
    container_transform: RigidTransform,
    fill_assets: list[SceneObject],
    final_fill_transforms: list[RigidTransform],
    inside_indices: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute composite bounding box from container and fill objects in local frame.

    Args:
        container_asset: Container SceneObject with bbox.
        container_transform: Container's world transform.
        fill_assets: List of all fill asset SceneObjects.
        final_fill_transforms: Final transforms for each fill asset.
        inside_indices: Indices of fill assets that are inside the container.

    Returns:
        Tuple of (bbox_min, bbox_max) in container's local frame, or (None, None)
        if no valid bounding boxes.
    """
    all_bbox_min = np.array([np.inf, np.inf, np.inf])
    all_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    bbox_count = 0

    # Add container bbox.
    if container_asset.bbox_min is not None and container_asset.bbox_max is not None:
        bbox_count += 1
        corners = compute_bbox_corners(
            bbox_min=container_asset.bbox_min, bbox_max=container_asset.bbox_max
        )
        for corner in corners:
            world_corner = container_transform.multiply(corner)
            all_bbox_min = np.minimum(all_bbox_min, world_corner)
            all_bbox_max = np.maximum(all_bbox_max, world_corner)

    # Add fill objects bboxes.
    for idx in inside_indices:
        fill_asset = fill_assets[idx]
        fill_transform = final_fill_transforms[idx]
        if fill_asset.bbox_min is not None and fill_asset.bbox_max is not None:
            bbox_count += 1
            corners = compute_bbox_corners(
                bbox_min=fill_asset.bbox_min, bbox_max=fill_asset.bbox_max
            )
            for corner in corners:
                world_corner = fill_transform.multiply(corner)
                all_bbox_min = np.minimum(all_bbox_min, world_corner)
                all_bbox_max = np.maximum(all_bbox_max, world_corner)

    # Convert to local frame relative to container transform.
    if bbox_count == 0:
        return None, None

    inverse_transform = container_transform.inverse()
    world_corners = compute_bbox_corners(bbox_min=all_bbox_min, bbox_max=all_bbox_max)
    local_bbox_min = np.array([np.inf, np.inf, np.inf])
    local_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    for corner in world_corners:
        local_corner = inverse_transform.multiply(corner)
        local_bbox_min = np.minimum(local_bbox_min, local_corner)
        local_bbox_max = np.maximum(local_bbox_max, local_corner)

    return local_bbox_min, local_bbox_max
