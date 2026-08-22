"""Physical feasibility post-processing for scene collision resolution.

This module provides two-stage post-processing adapted from scene_gen repository:
1. Projection - IK-based collision resolution with configurable DOF constraints
2. Simulation - Physics settling to static equilibrium (always full 6DOF)

See: https://github.com/nepfaff/steerable-scene-generation/blob/main/steerable_scene_generation/algorithms/scene_diffusion/postprocessing.py
"""

import logging
import time

import numpy as np
import trimesh

from pydrake.all import BodyIndex, DiagramBuilder, RotationMatrix
from pydrake.geometry.optimization import HPolyhedron, VPolytope

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SupportSurface,
    UniqueID,
)
from scenesmith.utils.geometry.geometry_utils import safe_convex_hull_2d

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.feasibility.ik import (
    _create_drake_plant_for_ik,
    _find_surface_owner,
    _update_scene_from_plant,
    solve_non_penetration_ik,
)


def get_object_xy_footprint(
    mesh: "trimesh.Trimesh", rotation: RotationMatrix
) -> VPolytope:
    """Get object's 2D XY footprint after applying rotation.

    Computes the convex hull of the object's mesh vertices projected onto
    the XY plane, after applying the given rotation. Used with Pontryagin
    difference to compute feasible placement regions.

    Args:
        mesh: Object's trimesh mesh.
        rotation: Fixed rotation to apply.

    Returns:
        VPolytope representing the 2D footprint of the rotated object.

    Raises:
        ValueError: If mesh has no vertices.
    """
    if mesh.vertices.shape[0] == 0:
        raise ValueError("Cannot compute footprint for mesh with no vertices")

    # Apply rotation to mesh vertices.
    rotated_vertices: np.ndarray = (rotation.matrix() @ mesh.vertices.T).T

    # Project to XY and compute convex hull.
    xy_vertices = rotated_vertices[:, :2]

    hull, processed_vertices = safe_convex_hull_2d(xy_vertices)
    if hull is None:
        # Degenerate hull (collinear vertices, e.g., very thin knife).
        # Fall back to AABB of XY vertices.
        lb = xy_vertices.min(axis=0)
        ub = xy_vertices.max(axis=0)
        # Ensure non-zero dimensions (add small epsilon if needed).
        epsilon = 1e-4
        if ub[0] - lb[0] < epsilon:
            lb[0] -= epsilon / 2
            ub[0] += epsilon / 2
        if ub[1] - lb[1] < epsilon:
            lb[1] -= epsilon / 2
            ub[1] += epsilon / 2
        return VPolytope.MakeBox(lb=lb, ub=ub)

    hull_vertices = processed_vertices[hull.vertices]
    # VPolytope expects 2xN array (dim x num_vertices).
    return VPolytope(vertices=hull_vertices.T)


def apply_surface_projection(
    scene: RoomScene,
    surface: "SupportSurface",
    object_ids: list[UniqueID],
    influence_distance: float = 0.02,
    solver_name: str = "snopt",
    iteration_limit: int = 5000,
    time_limit_s: float = 120.0,
) -> tuple[RoomScene, bool, list[UniqueID], float]:
    """Project specified objects on a surface to resolve penetrations.

    Solves for XY translations only - all rotations (roll, pitch, yaw) are
    fixed. Objects are constrained to stay within the surface boundary using
    Pontryagin difference (surface hull - object footprint).

    Uses adaptive scope for performance optimization:
    - For furniture surfaces: includes only parent furniture + its manipulands
    - For floor surfaces: includes all furniture + floor manipulands
    This prevents loading the entire scene into Drake when only a subset of
    objects are relevant for collision checking.

    Args:
        scene: RoomScene to project.
        surface: SupportSurface defining boundary for projection.
        object_ids: Objects to project. Can be 1+ objects. These are the
            movable objects (free bodies in IK).
        influence_distance: Collision influence distance.
        solver_name: NLP solver ("snopt" or "ipopt").
        iteration_limit: Max solver iterations.
        time_limit_s: Max solver time.

    Returns:
        Tuple of (projected_scene, success, moved_object_ids, max_displacement).
    """
    start_time = time.time()
    console_logger.debug(
        f"Starting surface projection for {len(object_ids)} objects on "
        f"surface {surface.surface_id}"
    )

    if not object_ids:
        console_logger.debug("No objects to project.")
        return scene, True, [], 0.0

    # Fail early if any object is a pile - piles have complex footprints that
    # cannot be accurately represented by a single member's mesh.
    for obj_id in object_ids:
        obj = scene.get_object(obj_id)
        if obj is not None and obj.metadata.get("composite_type") == "pile":
            raise ValueError(
                f"Cannot resolve penetrations for pile object {obj_id}. "
                "Piles have scattered members whose combined footprint cannot be "
                "accurately computed. Instead, move other objects or recreate "
                "the pile at a different location."
            )

    # Record original positions for displacement tracking.
    original_positions: dict[UniqueID, np.ndarray] = {}
    for obj_id in object_ids:
        obj = scene.get_object(obj_id)
        if obj is not None:
            original_positions[obj_id] = obj.transform.translation().copy()

    # Get surface boundary as HPolyhedron.
    surface_vpoly = surface.get_xy_convex_hull()
    surface_hpoly = HPolyhedron(vpoly=surface_vpoly)

    # Build per-object feasible regions using Pontryagin difference.
    # This accounts for object footprint - long objects (knives) can be
    # placed closer to edges when oriented parallel.
    object_feasible_regions: dict[UniqueID, HPolyhedron] = {}
    for obj_id in object_ids:
        obj = scene.get_object(obj_id)
        if obj is None:
            continue

        # Get mesh for the object. For composites, use the reference member's mesh.
        obj_mesh = None
        if obj.geometry_path is not None:
            try:
                obj_mesh = trimesh.load(obj.geometry_path, force="mesh")
            except Exception as e:
                console_logger.warning(f"Failed to load mesh for {obj_id}: {e}")

        composite_type = obj.metadata.get("composite_type")

        if obj_mesh is None and composite_type == "stack":
            # Stack: use bottom member's mesh.
            member_assets = obj.metadata.get("member_assets", [])
            if member_assets:
                bottom_geometry = member_assets[0].get("geometry_path")
                if bottom_geometry:
                    try:
                        obj_mesh = trimesh.load(bottom_geometry, force="mesh")
                    except Exception as e:
                        console_logger.warning(f"Failed to load stack bottom mesh: {e}")

        if obj_mesh is None and composite_type == "filled_container":
            # Filled container: use container's mesh.
            container_asset = obj.metadata.get("container_asset")
            if container_asset:
                container_geometry = container_asset.get("geometry_path")
                if container_geometry:
                    try:
                        obj_mesh = trimesh.load(container_geometry, force="mesh")
                    except Exception as e:
                        console_logger.warning(f"Failed to load container mesh: {e}")

        if obj_mesh is None:
            # No mesh available - use surface bounds as fallback.
            console_logger.warning(
                f"No mesh for {obj_id}, using surface bounds as feasible region"
            )
            object_feasible_regions[obj_id] = surface_hpoly
            continue

        # Apply scale_factor to mesh for correct footprint dimensions.
        if obj.scale_factor != 1.0:
            obj_mesh.vertices *= obj.scale_factor

        # Get object's current rotation (fixed during projection).
        obj_rotation = obj.transform.rotation()

        # Compute object's XY footprint after rotation.
        object_vpoly = get_object_xy_footprint(obj_mesh, obj_rotation)
        object_hpoly = HPolyhedron(vpoly=object_vpoly)

        # Pontryagin difference: set of all center positions where entire
        # object footprint stays within surface boundary.
        try:
            feasible_region = surface_hpoly.PontryaginDifference(object_hpoly)
            if feasible_region.IsEmpty():
                console_logger.warning(
                    f"Object {obj_id} is too large for surface {surface.surface_id}"
                )
                # Use surface bounds as fallback.
                feasible_region = surface_hpoly
            object_feasible_regions[obj_id] = feasible_region
        except Exception as e:
            console_logger.warning(
                f"Failed to compute feasible region for {obj_id}: {e}. "
                "Using surface bounds."
            )
            object_feasible_regions[obj_id] = surface_hpoly

    # Compute adaptive scope based on surface type.
    # For furniture surfaces: include parent furniture + its manipulands.
    # For floor surfaces: include all furniture + floor manipulands.
    owner_id, is_floor = _find_surface_owner(scene=scene, surface_id=surface.surface_id)

    include_objects: list[UniqueID] = []
    if is_floor:
        # Floor surface: include all furniture as obstacles.
        for obj in scene.objects.values():
            if obj.object_type == ObjectType.FURNITURE:
                include_objects.append(obj.object_id)
        # Include all manipulands on the floor.
        if owner_id is not None:
            floor_obj = scene.get_object(owner_id)
            if floor_obj is not None:
                for surf in floor_obj.support_surfaces:
                    for manip in scene.get_objects_on_surface(surf.surface_id):
                        if manip.object_id not in include_objects:
                            include_objects.append(manip.object_id)
        console_logger.debug(
            f"Floor surface mode: including {len(include_objects)} objects "
            f"(all furniture + floor manipulands)"
        )
    elif owner_id is not None:
        # Furniture surface: include parent furniture + its manipulands.
        include_objects.append(owner_id)
        furniture = scene.get_object(owner_id)
        if furniture is not None:
            for surf in furniture.support_surfaces:
                for manip in scene.get_objects_on_surface(surf.surface_id):
                    if manip.object_id not in include_objects:
                        include_objects.append(manip.object_id)
        console_logger.debug(
            f"Furniture surface mode: including {len(include_objects)} objects "
            f"(furniture {owner_id} + its manipulands)"
        )
    else:
        # Unknown surface owner - fall back to all objects.
        console_logger.warning(
            f"Could not find owner for surface {surface.surface_id}, "
            "using all objects for collision checking"
        )
        include_objects = None  # type: ignore[assignment]

    try:
        # Create Drake plant using shared helper (handles composite objects properly).
        builder = DiagramBuilder()
        plant, scene_graph, object_indices, composite_info = _create_drake_plant_for_ik(
            scene=scene,
            builder=builder,
            weld_furniture=True,
            time_step=0.0,
            free_objects=list(object_ids),
            include_objects=include_objects,
        )

        if not object_indices:
            console_logger.warning("No free bodies found for projection.")
            return scene, True, [], 0.0

        # Filter object_indices to only include requested object_ids.
        filtered_indices = {k: v for k, v in object_indices.items() if k in object_ids}

        if not filtered_indices:
            console_logger.warning("No matching free bodies found for projection.")
            return scene, True, [], 0.0

        # Build xy_regions mapping (BodyIndex -> HPolyhedron).
        xy_regions: dict[BodyIndex, HPolyhedron] = {}
        for obj_id, (_, body_idx) in filtered_indices.items():
            if obj_id in object_feasible_regions:
                xy_regions[body_idx] = object_feasible_regions[obj_id]

        # Solve using shared utility.
        plant_context, success = solve_non_penetration_ik(
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
            influence_distance=influence_distance,
            fix_rotation=True,
            fix_z=True,
            solver_name=solver_name,
            iteration_limit=iteration_limit,
            time_limit_s=time_limit_s,
            xy_regions=xy_regions,
        )

        if not success or plant_context is None:
            elapsed = time.time() - start_time
            console_logger.warning(f"Surface projection failed after {elapsed:.2f}s")
            return scene, False, [], 0.0

        # Update scene using shared helper (handles composite member transforms).
        # Filter to exclude obstacle composites - only update movable objects.
        # (composite_info contains all composites in plant, including welded obstacles)
        filtered_composite_info = {
            k: v for k, v in composite_info.items() if k in object_ids
        }
        _update_scene_from_plant(
            scene=scene,
            plant=plant,
            plant_context=plant_context,
            object_indices=filtered_indices,
            composite_info=filtered_composite_info,
            operation_name="Surface projection",
        )

        # Compute displacements for moved objects.
        moved_ids: list[UniqueID] = []
        max_displacement = 0.0
        for obj_id in object_ids:
            obj = scene.get_object(obj_id)
            if obj is None:
                continue

            if obj_id in original_positions:
                old_pos = original_positions[obj_id]
                new_pos = obj.transform.translation()
                displacement = np.linalg.norm(new_pos - old_pos)

                if displacement > 1e-6:  # Non-trivial movement.
                    moved_ids.append(obj_id)
                    max_displacement = max(max_displacement, displacement)
                    console_logger.info(f"  {obj_id}: moved {displacement:.4f}m")

        # Update placement_info for moved objects (world → surface conversion).
        # This keeps surface-relative coordinates in sync after physics resolution.
        for obj_id in moved_ids:
            obj = scene.get_object(obj_id)
            if obj is None or obj.placement_info is None:
                continue

            # Convert new world position back to surface-relative coordinates.
            new_pos_2d, new_rot_2d = surface.from_world_pose(obj.transform)
            obj.placement_info.position_2d = new_pos_2d.copy()
            # Note: rotation_2d should be unchanged since fix_rotation=True,
            # but update it anyway for correctness and future-proofing.
            obj.placement_info.rotation_2d = new_rot_2d

        elapsed = time.time() - start_time
        console_logger.info(
            f"Surface projection succeeded in {elapsed:.2f}s. "
            f"Moved {len(moved_ids)} objects, max displacement: {max_displacement:.4f}m"
        )

        return scene, True, moved_ids, max_displacement

    except Exception as e:
        console_logger.error(f"Surface projection failed with exception: {e}")
        return scene, False, [], 0.0
