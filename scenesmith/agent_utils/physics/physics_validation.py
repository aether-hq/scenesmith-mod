"""Physics validation for scenes using Drake."""

import logging
import time

from pydrake.all import DiagramBuilder, GeometryId, QueryObject, SceneGraphInspector

from scenesmith.agent_utils.physics.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenesmith.agent_utils.physics.validation.collision_filtering import (
    _get_furniture_id_for_manipuland,
    _get_object_info_from_geometry_id,
)
from scenesmith.agent_utils.physics.validation.models import CollisionPair
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID

console_logger = logging.getLogger(__name__)


def _compute_relevant_objects_for_collision(
    scene: RoomScene, current_furniture_id: UniqueID | None
) -> list[UniqueID] | None:
    """
    Determine which objects are relevant for collision checking.

    When current_furniture_id is provided (manipuland agent workflow):
    - Include all furniture (manipulands might extend beyond current surface)
    - Include only manipulands on current furniture/floor
    - For floor placement: floor manipulands must be checked against all furniture

    Returns:
        List of object IDs to include, or None for all objects.
    """
    if current_furniture_id is None:
        return None  # Full scene collision check.

    relevant_objects = []

    # Include all furniture and wall-mounted objects (current + surrounding).
    for obj in scene.objects.values():
        if obj.object_type in (ObjectType.FURNITURE, ObjectType.WALL_MOUNTED):
            relevant_objects.append(obj.object_id)

    # Check if current target is the floor.
    current_obj = scene.objects.get(current_furniture_id)
    is_floor_placement = current_obj and current_obj.object_type == ObjectType.FLOOR

    # Include only manipulands on current furniture/floor.
    for obj_id, obj in scene.objects.items():
        if obj.object_type == ObjectType.MANIPULAND:
            parent_id = _get_furniture_id_for_manipuland(str(obj_id), scene)
            if parent_id == str(current_furniture_id):
                relevant_objects.append(obj_id)

    if is_floor_placement:
        num_furniture = len(
            [
                o
                for o in relevant_objects
                if scene.objects.get(o)
                and scene.objects.get(o).object_type == ObjectType.FURNITURE
            ]
        )
        console_logger.info(
            f"Floor placement mode: including floor manipulands for collision check "
            f"against {num_furniture} furniture items"
        )

    console_logger.debug(
        f"Early filtering: {len(relevant_objects)} objects for collision check "
        f"(out of {len(scene.objects)} total)"
    )

    return relevant_objects


def compute_scene_collisions(
    scene: RoomScene,
    penetration_threshold: float = 0.001,
    floor_penetration_tolerance: float = 0.05,
    current_furniture_id: UniqueID | None = None,
    manipuland_furniture_tolerance_m: float = 0.02,
) -> list[CollisionPair]:
    """
    Compute collision violations. Also checks for collisions between welded bodies.

    Args:
        scene: RoomScene to check for collisions.
        penetration_threshold: Minimum penetration depth to report (meters).
            Only penetrations deeper than this threshold are considered
            collisions.
        floor_penetration_tolerance: Tolerance for furniture-floor penetration
            (meters). Floor collisions with penetration less than this amount
            are ignored.
        current_furniture_id: Optional ID of furniture currently being populated
            by manipuland agent. When provided, filters out collisions involving
            manipulands from other furniture (unless they collide with current
            furniture's manipulands).
        manipuland_furniture_tolerance_m: Tolerance for current manipuland-current
            furniture surface contact (meters). Mild collisions within this threshold
            are filtered as expected contact. Default 0.02 (2cm).

    Returns:
        List of CollisionPair objects representing detected collisions.
        penetration_depth values are positive (positive = penetration,
        zero = touching).

    Raises:
        RuntimeError: If Drake physics validation fails.
    """
    collision_start_time = time.time()

    # Compute relevant objects for early filtering.
    # When current_furniture_id is provided, only load relevant objects into Drake.
    include_objects = _compute_relevant_objects_for_collision(
        scene=scene, current_furniture_id=current_furniture_id
    )

    # Create Drake scene graph with all objects as free bodies (not welded).
    # This allows broadphase query to detect all collisions.
    # weld_furniture=False makes furniture free, and free_mounted_objects_for_collision=True
    # also makes wall-mounted and ceiling-mounted objects free for collision detection.
    builder = DiagramBuilder()
    _, scene_graph = create_drake_plant_and_scene_graph_from_scene(
        scene=scene,
        builder=builder,
        include_objects=include_objects,
        weld_furniture=False,
        free_mounted_objects_for_collision=True,
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Get query object for collision detection.
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    # Use broadphase collision detection (all objects are free bodies now).
    # This uses Drake's internal broadphase (BVH) for efficient filtering.
    try:
        inspector = query_object.inspector()
        all_signed_distance_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(
                max_distance=0.0  # Only penetrating pairs (distance < 0).
            )
        )

        # Filter by penetration threshold.
        signed_distance_pairs = [
            pair
            for pair in all_signed_distance_pairs
            if pair.distance <= -penetration_threshold
        ]

        console_logger.debug(
            f"Broadphase found {len(all_signed_distance_pairs)} penetrating pairs, "
            f"{len(signed_distance_pairs)} above threshold"
        )
    except Exception as e:
        error_msg = f"Physics validation failed: {str(e)}"
        console_logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Convert to CollisionPair objects with filtering and deduplication.
    collisions = []
    seen_pairs = set()  # Track unique collision pairs to avoid duplicates
    for pair in signed_distance_pairs:
        # Map geometry IDs to object names and IDs.
        object_a_info = _get_object_info_from_geometry_id(
            geometry_id=pair.id_A, scene=scene, query_object=query_object
        )
        object_b_info = _get_object_info_from_geometry_id(
            geometry_id=pair.id_B, scene=scene, query_object=query_object
        )

        # Check if this is a floor collision and apply tolerance.
        is_floor_collision = (
            object_a_info["name"] == "floor" or object_b_info["name"] == "floor"
        )
        if is_floor_collision:
            # Skip floor collisions that are within tolerance.
            penetration_depth = abs(pair.distance)
            if penetration_depth <= floor_penetration_tolerance:
                continue

        # Create a unique identifier for this collision pair to avoid duplicates.
        # Sort the IDs to ensure A->B and B->A are treated as the same collision.
        pair_id = tuple(
            sorted(
                [
                    f"{object_a_info['name']}[{object_a_info['id']}]",
                    f"{object_b_info['name']}[{object_b_info['id']}]",
                ]
            )
        )

        # Skip if we've already seen this collision pair.
        if pair_id in seen_pairs:
            continue
        seen_pairs.add(pair_id)

        collision = CollisionPair(
            object_a_name=object_a_info["name"],
            object_a_id=object_a_info["id"],
            object_b_name=object_b_info["name"],
            object_b_id=object_b_info["id"],
            penetration_depth=abs(pair.distance),
        )

        # Post-computation filtering: Apply distance-based filtering.
        # ONLY filters current manipuland × current furniture with ≤2cm penetration.
        # All other collision types use strict no-tolerance policy.
        should_skip, skip_reason = _should_skip_collision_pair(
            gid_a=pair.id_A,
            gid_b=pair.id_B,
            inspector=inspector,
            scene=scene,
            query_object=query_object,
            current_furniture_id=current_furniture_id,
            collision=collision,
            manipuland_furniture_tolerance_m=manipuland_furniture_tolerance_m,
        )
        if should_skip:
            console_logger.debug(f"Skipping collision: {skip_reason}")
            continue

        collisions.append(collision)

    collision_end_time = time.time()
    console_logger.info(
        "Computed scene collisions in "
        f"{collision_end_time - collision_start_time:.2f} seconds. "
        f"Found {len(collisions)} collisions."
    )

    # Log detailed collision information.
    if collisions:
        console_logger.info(f"=== Collision Details ({len(collisions)} total) ===")
        for i, collision in enumerate(collisions, 1):
            console_logger.info(
                f"Collision {i}: {collision.object_a_name} [{collision.object_a_id}] "
                f"<-> {collision.object_b_name} [{collision.object_b_id}] | "
                f"Penetration: {collision.penetration_depth * 100:.2f}cm"
            )
        console_logger.info("=" * 60)

    return collisions


def _should_skip_collision_pair(
    gid_a: GeometryId,
    gid_b: GeometryId,
    inspector: SceneGraphInspector,
    scene: RoomScene,
    query_object: QueryObject,
    current_furniture_id: UniqueID | None,
    collision: CollisionPair | None = None,
    manipuland_furniture_tolerance_m: float = 0.02,
) -> tuple[bool, str]:
    """
    Determine if a collision pair should be skipped.

    Returns True for self-collisions, wall-to-wall collisions,
    (when filtering is enabled) collisions involving non-current manipulands/furniture,
    and (when collision is provided) mild current manipuland-furniture contact.

    This function is called in two stages:
    1. Pre-computation (collision=None): Filters based on geometry relationships
    2. Post-computation (collision provided): Applies distance-based filtering

    Args:
        gid_a: First geometry ID.
        gid_b: Second geometry ID.
        inspector: Drake geometry inspector.
        scene: RoomScene containing objects.
        query_object: Drake query object for geometry inspection.
        current_furniture_id: Optional ID of furniture currently being populated.
            When provided, filters out collisions involving manipulands/furniture from
            other furniture pieces.
        collision: Optional CollisionPair for post-computation distance-based filtering.
            When None, only geometry-based filtering is applied.
        manipuland_furniture_tolerance_m: Tolerance for current manipuland-current
            furniture contact (meters). Only used when collision is provided.

    Returns:
        Tuple of (should_skip: bool, reason: str).
    """
    try:
        # Get frame IDs for both geometries.
        frame_a = inspector.GetFrameId(gid_a)
        frame_b = inspector.GetFrameId(gid_b)

        # Skip self-collisions (same frame).
        if frame_a == frame_b:
            return True, "self-collision (same frame)"

        # Get geometry names to check for wall-to-wall collisions.
        name_a = inspector.GetName(gid_a).lower()
        name_b = inspector.GetName(gid_b).lower()

        # Skip wall-to-wall collisions.
        if "wall" in name_a and "wall" in name_b:
            return True, "wall-to-wall collision"

        # Get object info for both geometries.
        obj_a_info = _get_object_info_from_geometry_id(
            geometry_id=gid_a, scene=scene, query_object=query_object
        )
        obj_b_info = _get_object_info_from_geometry_id(
            geometry_id=gid_b, scene=scene, query_object=query_object
        )

        obj_a_id = obj_a_info["id"]
        obj_b_id = obj_b_info["id"]

        # Skip intra-object collisions (e.g., articulated links, stack members).
        # Both geometries resolve to the same parent object ID.
        if obj_a_id == obj_b_id:
            return True, f"intra-object collision (same parent: {obj_a_id})"

        # Apply manipuland filtering if current_furniture_id is provided.
        if current_furniture_id is not None:
            # Check if objects are manipulands or furniture.
            obj_a_scene_obj = scene.objects.get(UniqueID(obj_a_id))
            obj_b_scene_obj = scene.objects.get(UniqueID(obj_b_id))

            obj_a_is_manipuland = (
                obj_a_scene_obj and obj_a_scene_obj.object_type == ObjectType.MANIPULAND
            )
            obj_b_is_manipuland = (
                obj_b_scene_obj and obj_b_scene_obj.object_type == ObjectType.MANIPULAND
            )

            obj_a_is_furniture = (
                obj_a_scene_obj and obj_a_scene_obj.object_type == ObjectType.FURNITURE
            )
            obj_b_is_furniture = (
                obj_b_scene_obj and obj_b_scene_obj.object_type == ObjectType.FURNITURE
            )

            # Determine if manipulands belong to current furniture.
            is_current_a = False
            is_current_b = False

            if obj_a_is_manipuland:
                furniture_a = _get_furniture_id_for_manipuland(
                    manipuland_id=obj_a_id, scene=scene
                )
                is_current_a = furniture_a == str(current_furniture_id)

            if obj_b_is_manipuland:
                furniture_b = _get_furniture_id_for_manipuland(
                    manipuland_id=obj_b_id, scene=scene
                )
                is_current_b = furniture_b == str(current_furniture_id)

            # Skip if one is non-current manipuland and other is non-current furniture.
            # We have no control over objects from other furniture pieces.
            if obj_a_is_manipuland and not is_current_a and obj_b_is_furniture:
                if str(obj_b_id) != str(current_furniture_id):
                    return True, "non-current manipuland with non-current furniture"
            if obj_b_is_manipuland and not is_current_b and obj_a_is_furniture:
                if str(obj_a_id) != str(current_furniture_id):
                    return True, "non-current manipuland with non-current furniture"

            # Skip if both are non-current manipulands.
            if obj_a_is_manipuland and obj_b_is_manipuland:
                if not is_current_a and not is_current_b:
                    return True, "both non-current manipulands"

            # Skip if one is non-current manipuland and other is floor/wall.
            # (Floor/wall has obj_id == "room_geometry").
            if obj_a_is_manipuland and not is_current_a and obj_b_id == "room_geometry":
                return True, "non-current manipuland with floor/wall"
            if obj_b_is_manipuland and not is_current_b and obj_a_id == "room_geometry":
                return True, "non-current manipuland with floor/wall"

            # Skip if both are non-current furniture.
            # We have no control over furniture we're not currently working with.
            if obj_a_is_furniture and obj_b_is_furniture:
                is_current_furniture_a = str(obj_a_id) == str(current_furniture_id)
                is_current_furniture_b = str(obj_b_id) == str(current_furniture_id)
                if not is_current_furniture_a and not is_current_furniture_b:
                    return True, "both non-current furniture"

            # Skip if one is non-current furniture and other is floor/wall.
            # Other furniture's floor contact is irrelevant to current work.
            if obj_a_is_furniture and obj_b_id == "room_geometry":
                if str(obj_a_id) != str(current_furniture_id):
                    return True, "non-current furniture with floor/wall"
            if obj_b_is_furniture and obj_a_id == "room_geometry":
                if str(obj_b_id) != str(current_furniture_id):
                    return True, "non-current furniture with floor/wall"

        # Wall-mounted object filtering.
        # Wall objects use Drake collision for wall↔wall and wall↔furniture checks.
        # Skip wall↔room_geometry (attached to walls).
        obj_a_scene_obj = scene.objects.get(UniqueID(obj_a_id))
        obj_b_scene_obj = scene.objects.get(UniqueID(obj_b_id))
        obj_a_is_wall_mounted = (
            obj_a_scene_obj and obj_a_scene_obj.object_type == ObjectType.WALL_MOUNTED
        )
        obj_b_is_wall_mounted = (
            obj_b_scene_obj and obj_b_scene_obj.object_type == ObjectType.WALL_MOUNTED
        )
        if obj_a_is_wall_mounted or obj_b_is_wall_mounted:
            # Get the other object's type.
            other_id = obj_b_id if obj_a_is_wall_mounted else obj_a_id

            # Skip wall object ↔ room geometry (wall objects are attached to walls).
            # Room geometry IDs can be "room_geometry" or "room_geometry::north_wall" etc.
            if other_id == "room_geometry" or other_id.startswith("room_geometry::"):
                return True, "wall-mounted object with room geometry"

        # Apply distance-based filtering if collision is provided.
        # This is the post-computation stage after penetration depth is known.
        if collision is not None and current_furniture_id is not None:
            obj_a_id = collision.object_a_id
            obj_b_id = collision.object_b_id

            # Get object types.
            obj_a = scene.objects.get(UniqueID(obj_a_id))
            obj_b = scene.objects.get(UniqueID(obj_b_id))

            obj_a_is_manipuland = obj_a and obj_a.object_type == ObjectType.MANIPULAND
            obj_b_is_manipuland = obj_b and obj_b.object_type == ObjectType.MANIPULAND

            obj_a_is_furniture = obj_a and obj_a.object_type == ObjectType.FURNITURE
            obj_b_is_furniture = obj_b and obj_b.object_type == ObjectType.FURNITURE

            # Check if we have a manipuland-furniture collision.
            if (obj_a_is_manipuland and obj_b_is_furniture) or (
                obj_b_is_manipuland and obj_a_is_furniture
            ):
                # Determine which is manipuland and which is furniture.
                manipuland_id = obj_a_id if obj_a_is_manipuland else obj_b_id
                furniture_id = obj_a_id if obj_a_is_furniture else obj_b_id

                # Check if manipuland belongs to current furniture.
                manipuland_furniture_id = _get_furniture_id_for_manipuland(
                    manipuland_id=manipuland_id, scene=scene
                )
                is_current_manipuland = manipuland_furniture_id == str(
                    current_furniture_id
                )

                # Check if furniture is current furniture.
                is_current_furniture = furniture_id == str(current_furniture_id)

                # Only filter if BOTH are current (manipuland on current furniture ×
                # current furniture). This represents expected surface contact.
                # Do NOT filter if furniture is non-current (we need to know if our
                # manipuland collides with nearby furniture).
                if is_current_manipuland and is_current_furniture:
                    # Apply distance threshold.
                    if collision.penetration_depth <= manipuland_furniture_tolerance_m:
                        penetration_cm = collision.penetration_depth * 100
                        return (
                            True,
                            "mild current manipuland-furniture contact "
                            f"({penetration_cm:.1f}cm ≤ 2cm)",
                        )

        return False, ""

    except Exception as e:
        # If we can't determine the relationship, don't skip.
        console_logger.error(f"Could not determine collision pair relationship: {e}")
        return False, "unknown"
