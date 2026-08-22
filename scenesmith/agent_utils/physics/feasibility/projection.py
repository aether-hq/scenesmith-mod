"""Physical feasibility post-processing for scene collision resolution.

This module provides two-stage post-processing adapted from scene_gen repository:
1. Projection - IK-based collision resolution with configurable DOF constraints
2. Simulation - Physics settling to static equilibrium (always full 6DOF)

See: https://github.com/nepfaff/steerable-scene-generation/blob/main/steerable_scene_generation/algorithms/scene_diffusion/postprocessing.py
"""

import logging
import time

from pydrake.all import DiagramBuilder, RigidTransform
from pydrake.geometry import QueryObject

from scenesmith.agent_utils.physics.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.feasibility.ik import (
    _create_drake_plant_for_ik,
    _update_scene_from_plant,
    solve_non_penetration_ik,
)


def _get_colliding_object_ids(
    scene: RoomScene, penetration_threshold: float = 1e-5
) -> set[UniqueID]:
    """Identify objects that are currently in collision.

    Uses broadphase collision detection to efficiently find penetrating pairs.
    This is used to reduce DOFs in large scene projection by only making
    colliding objects free bodies.

    Note: Only returns scene object IDs, not room geometry IDs (floor, walls).
    Objects colliding with room geometry are included, but "floor"/"wall" IDs
    are filtered out since room geometry is always welded.

    Args:
        scene: RoomScene to check for collisions.
        penetration_threshold: Minimum penetration depth to consider (meters).

    Returns:
        Set of UniqueIDs for scene objects that are in collision.
    """

    def is_room_geometry_id(object_id: str) -> bool:
        """Check if an object ID refers to room geometry (floor, wall, etc.)."""
        # Room geometry IDs can be:
        # - Simple: "floor", "wall"
        # - Prefixed: "room_geometry::floor", "room_geometry::wall_collision"
        room_geometry_patterns = ("floor", "wall", "room_geometry")
        object_id_lower = object_id.lower()
        return any(pattern in object_id_lower for pattern in room_geometry_patterns)

    collisions = compute_scene_collisions(
        scene=scene,
        penetration_threshold=penetration_threshold,
        floor_penetration_tolerance=0.01,  # 1cm tolerance for floor/surface resting.
        current_furniture_id=None,
    )

    colliding_ids: set[UniqueID] = set()
    for collision in collisions:
        allowed_owner_contact = False
        for row_id, other_id in (
            (collision.object_a_id, collision.object_b_id),
            (collision.object_b_id, collision.object_a_id),
        ):
            try:
                row = scene.get_object(UniqueID(row_id))
            except (TypeError, ValueError):
                row = None
            if (
                row is not None
                and str(
                    (getattr(row, "metadata", None) or {}).get(
                        "dense_library_owner_bound",
                        "",
                    )
                )
                == other_id
            ):
                allowed_owner_contact = True
                break
        if allowed_owner_contact:
            continue
        # Only add scene object IDs, skip room geometry.
        if not is_room_geometry_id(collision.object_a_id):
            colliding_ids.add(UniqueID(collision.object_a_id))
        if not is_room_geometry_id(collision.object_b_id):
            colliding_ids.add(UniqueID(collision.object_b_id))

    return colliding_ids


def apply_non_penetration_projection(
    scene: RoomScene,
    influence_distance: float = 0.02,
    solver_name: str = "snopt",
    iteration_limit: int = 5000,
    time_limit_s: float = 360.0,
    weld_furniture: bool = False,
    xy_only: bool = True,
    fix_rotation: bool = True,
    large_scene_optimization_threshold: int = 100,
    collision_penetration_threshold_m: float = 0.001,
) -> tuple[RoomScene, bool]:
    """Apply IK-based non-penetration projection to resolve collisions.

    This is Stage 1 of physical feasibility post-processing. Uses Drake's
    InverseKinematics with minimum distance constraints to resolve penetrations.

    For large scenes (above threshold), uses collision pre-check optimization
    to reduce DOFs by only making colliding objects free bodies.

    Args:
        scene: RoomScene to project.
        influence_distance: Collision influence distance in meters.
        solver_name: NLP solver ("snopt" or "ipopt").
        iteration_limit: Max solver iterations.
        time_limit_s: Max solver time in seconds.
        weld_furniture: If True, weld furniture (project manipulands only).
                        If False, all objects are free for optimization.
        xy_only: If True, only allow XY translation (fix Z).
                 If False, allow XYZ translation.
        fix_rotation: If True, fix rotations during projection (default).
                      If False, allow rotation optimization.
        large_scene_optimization_threshold: When scene has more objects than
            this threshold, only colliding objects are made free in IK.
            Set to 0 to always use optimization, very high to disable.
        collision_penetration_threshold_m: Minimum penetration depth (meters)
            to consider an object as colliding. Objects with penetration below
            this are treated as surface contacts and excluded from optimization.

    Returns:
        Tuple of (projected_scene, success_flag).
        On failure: returns (original_scene, False).
    """
    start_time = time.time()
    total_objects = len(scene.objects)
    console_logger.info(
        f"Starting non-penetration projection (weld_furniture={weld_furniture}, "
        f"xy_only={xy_only}, fix_rotation={fix_rotation}, objects={total_objects})"
    )

    free_object_ids: list[UniqueID] | None = None
    if weld_furniture:
        movable_objects = [
            obj
            for obj in scene.objects.values()
            if obj.object_type != ObjectType.FURNITURE
            and obj.object_type
            not in {
                ObjectType.WALL,
                ObjectType.FLOOR,
                ObjectType.THIN_COVERING,
                ObjectType.WALL_MOUNTED,
                ObjectType.CEILING_MOUNTED,
            }
            and (obj.metadata or {}).get("asset_source") != "thin_covering"
            and not (obj.metadata or {}).get("dense_library_owner_bound")
        ]
        colliding_ids = _get_colliding_object_ids(
            scene,
            penetration_threshold=collision_penetration_threshold_m,
        )
        if not colliding_ids:
            elapsed = time.time() - start_time
            console_logger.info(
                "No unresolved collisions; welded-furniture projection is a "
                "successful no-op (%.2fs)",
                elapsed,
            )
            return scene, True

        movable_ids = {obj.object_id for obj in movable_objects}
        free_object_ids = [
            object_id for object_id in colliding_ids if object_id in movable_ids
        ]
        if not free_object_ids:
            console_logger.warning(
                "No movable bodies can resolve %d colliding scene object(s)",
                len(colliding_ids),
            )
            return scene, False
        console_logger.info(
            "Welded-furniture collision optimization: %d/%d movable objects "
            "participate in unresolved collisions",
            len(free_object_ids),
            len(movable_objects),
        )

    # For large scenes, use collision pre-check to reduce DOFs.
    # Small scenes (per-furniture projection) use existing fast path.
    if free_object_ids is None and total_objects > large_scene_optimization_threshold:
        colliding_ids = _get_colliding_object_ids(
            scene, penetration_threshold=collision_penetration_threshold_m
        )

        if not colliding_ids:
            elapsed = time.time() - start_time
            console_logger.info(
                f"No collisions detected, skipping projection ({elapsed:.2f}s)"
            )
            return scene, True

        console_logger.info(
            f"Large scene optimization: {len(colliding_ids)}/{total_objects} "
            f"objects colliding (DOF: {total_objects * 7} -> {len(colliding_ids) * 7})"
        )
        free_object_ids = list(colliding_ids)

    try:
        # Create Drake plant for IK.
        builder = DiagramBuilder()
        plant, scene_graph, object_indices, composite_info = _create_drake_plant_for_ik(
            scene=scene,
            builder=builder,
            weld_furniture=weld_furniture,
            time_step=0.0,
            free_objects=free_object_ids,
        )

        if not object_indices:
            console_logger.warning("No free bodies for projection. Skipping.")
            return scene, True

        # Solve using shared utility.
        plant_context, success = solve_non_penetration_ik(
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
            influence_distance=influence_distance,
            fix_rotation=fix_rotation,
            fix_z=xy_only,
            solver_name=solver_name,
            iteration_limit=iteration_limit,
            time_limit_s=time_limit_s,
        )

        if success and plant_context is not None:
            # Update scene poses from plant.
            _update_scene_from_plant(
                scene=scene,
                plant=plant,
                plant_context=plant_context,
                object_indices=object_indices,
                composite_info=composite_info,
            )
            elapsed = time.time() - start_time
            console_logger.info(f"Projection succeeded in {elapsed:.2f}s")
        else:
            elapsed = time.time() - start_time
            console_logger.warning(f"Projection failed after {elapsed:.2f}s")

        return scene, success

    except Exception as e:
        console_logger.error(f"Projection failed with exception: {e}")
        return scene, False


def _apply_floor_penetration_fallback(
    scene: RoomScene, margin_m: float = 0.001
) -> tuple[RoomScene, int]:
    """Lift furniture above floor when NLP projection fails.

    Uses Drake's signed distance query to find floor penetrations and
    lifts each penetrating furniture piece by the exact penetration
    depth plus a small margin.

    Args:
        scene: RoomScene with potentially penetrating furniture.
        margin_m: Safety margin above floor (default 1mm).

    Returns:
        Tuple of (updated scene, number of objects lifted).
    """
    # Create Drake scene to query collisions.
    builder = DiagramBuilder()
    _, scene_graph = create_drake_plant_and_scene_graph_from_scene(
        scene=scene, builder=builder, weld_furniture=False
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Get query object for collision detection.
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    # Find all floor penetrations.
    all_pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
        max_distance=0.0  # Only penetrating pairs.
    )

    # Track max penetration per furniture piece.
    furniture_penetrations: dict[UniqueID, float] = {}
    inspector = query_object.inspector()

    for pair in all_pairs:
        # Get object names from geometry IDs.
        name_a = inspector.GetName(pair.id_A)
        name_b = inspector.GetName(pair.id_B)

        # Check if this is a floor collision (floor may be named "floor" or "ground").
        name_a_lower = name_a.lower()
        name_b_lower = name_b.lower()
        is_floor_a = "floor" in name_a_lower or "ground" in name_a_lower
        is_floor_b = "floor" in name_b_lower or "ground" in name_b_lower

        if not (is_floor_a or is_floor_b):
            continue  # Not a floor collision.

        # Get the non-floor object name.
        other_name = name_b if is_floor_a else name_a
        penetration_depth = abs(pair.distance)

        # Find furniture ID from geometry name.
        for obj in scene.objects.values():
            if obj.object_type != ObjectType.FURNITURE:
                continue
            # Match by checking if object ID is in the geometry name.
            obj_id_str = str(obj.object_id)
            if (
                obj_id_str in other_name
                or obj.name.lower().replace(" ", "_") in other_name.lower()
            ):
                furn_id = obj.object_id
                # Track max penetration for this furniture.
                current_max = furniture_penetrations.get(furn_id, 0.0)
                furniture_penetrations[furn_id] = max(current_max, penetration_depth)
                break

    # Lift each penetrating furniture piece.
    lifted_count = 0
    for furn_id, penetration in furniture_penetrations.items():
        if penetration > 0:
            lift_amount = penetration + margin_m
            obj = scene.get_object(furn_id)
            if obj is None:
                continue

            # Create new transform with lifted Z.
            old_transform = obj.transform
            new_translation = old_transform.translation().copy()
            new_translation[2] += lift_amount

            new_transform = RigidTransform(old_transform.rotation(), new_translation)
            obj.transform = new_transform

            console_logger.info(
                f"Floor fallback: lifted {obj.name} ({obj.object_id}) by "
                f"{lift_amount*1000:.1f}mm (penetration was {penetration*1000:.1f}mm)"
            )
            lifted_count += 1

    return scene, lifted_count
