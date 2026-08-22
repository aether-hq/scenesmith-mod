"""Physics validation for scenes using Drake."""

import logging

from pydrake.all import GeometryId, QueryObject

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    AgentType,
    ObjectType,
    UniqueID,
)

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.validation.models import CollisionPair


def _get_furniture_id_for_manipuland(
    manipuland_id: str, scene: RoomScene
) -> str | None:
    """
    Get the furniture or floor ID that owns the surface this manipuland is placed on.

    Args:
        manipuland_id: String ID of the manipuland object.
        scene: RoomScene containing objects.

    Returns:
        String ID of the furniture/floor that owns the surface, or None if not found.
    """
    manipuland = scene.objects.get(UniqueID(manipuland_id))
    if not manipuland or not manipuland.placement_info:
        return None

    surface_id = manipuland.placement_info.parent_surface_id

    # Find furniture, floor, or wall-mounted object that owns this surface.
    for obj_id, obj in scene.objects.items():
        if obj.object_type in (
            ObjectType.FURNITURE,
            ObjectType.FLOOR,
            ObjectType.WALL_MOUNTED,
        ):
            for surface in obj.support_surfaces:
                if surface.surface_id == surface_id:
                    return str(obj_id)

    # Also check room_geometry.floor (not always in scene.objects).
    if scene.room_geometry and scene.room_geometry.floor:
        floor = scene.room_geometry.floor
        for surface in floor.support_surfaces:
            if surface.surface_id == surface_id:
                return str(floor.object_id)

    return None


def _find_composite_by_model_name(
    frame_name: str, scene: RoomScene
) -> dict[str, str] | None:
    """Find parent composite (stack or filled_container) by direct model name lookup.

    Uses member_model_names stored in composite metadata during to_drake_directive()
    for reliable O(1) lookup without regex parsing.

    Args:
        frame_name: Drake frame name (e.g., "plate_abc12345_s0001_2::base_link").
        scene: RoomScene containing composite objects.

    Returns:
        Dict with 'name' and 'id' keys if match found, None otherwise.
    """
    # Extract model name from frame (strip ::link_name suffix).
    model_name = frame_name.split("::")[0]

    # Direct lookup in composite metadata (stacks, filled containers, and piles).
    for object_id, scene_object in scene.objects.items():
        composite_type = scene_object.metadata.get("composite_type")
        if composite_type not in ("stack", "filled_container", "pile"):
            continue

        member_model_names = scene_object.metadata.get("member_model_names", [])
        if model_name in member_model_names:
            console_logger.debug(
                f"Composite lookup: model_name={model_name} -> "
                f"composite_type={composite_type}, id={object_id}, "
                f"name={scene_object.name}"
            )
            return {"name": scene_object.name, "id": str(object_id)}

    # Log when no match found - this helps debug association issues.
    console_logger.warning(
        f"Composite lookup FAILED for model_name={model_name}. "
        f"No composite has this in member_model_names."
    )
    return None


# Alias for backwards compatibility.
_find_stack_by_model_name = _find_composite_by_model_name


def _get_object_info_from_geometry_id(
    geometry_id: GeometryId, scene: RoomScene, query_object: QueryObject
) -> dict[str, str]:
    """
    Map a Drake geometry ID to scene object name and ID.

    Args:
        geometry_id: Drake geometry ID.
        scene: RoomScene containing objects.
        query_object: Drake query object for geometry inspection.

    Returns:
        Dictionary with 'name' and 'id' keys.
    """
    inspector = query_object.inspector()

    try:
        # Get frame ID from geometry.
        frame_id = inspector.GetFrameId(geometry_id)
        frame_name = inspector.GetName(frame_id)

        # Special handling for room geometry elements (walls, floor).
        if "room_geometry" in frame_name:
            # Try to extract specific wall/floor name from geometry.
            geometry_name = inspector.GetName(geometry_id)
            geometry_name_lower = geometry_name.lower()
            geometry_basename = geometry_name_lower.rsplit("::", 1)[-1]
            is_primary_additional_support = (
                "_additional_support_" in frame_name
                and geometry_basename == "structure_collision"
            )
            if is_primary_additional_support:
                return {"name": "floor", "id": "room_geometry"}
            if "wall" in geometry_name_lower:
                # Extract wall ID from geometry name (e.g., "west_wall_collision" -> "west_wall").
                wall_id = geometry_name_lower.rsplit("_collision", 1)[0]
                return {"name": geometry_name, "id": wall_id}
            elif "floor" in geometry_name_lower or "ground" in geometry_name_lower:
                # Floor or ground element.
                return {"name": "floor", "id": "room_geometry"}
            else:
                # Generic room geometry element (fallback).
                return {"name": "wall", "id": "room_geometry"}

        # Extract object ID from frame name for regular scene objects.
        for object_id, scene_object in scene.objects.items():
            # Reconstruct expected model name using same logic as to_drake_directive().
            # This ensures exact matching and avoids suffix collisions.
            base_name = scene_object.name.lower().replace(" ", "_")
            id_suffix = str(object_id).split("_")[-1][:8]
            expected_model_name = f"{base_name}_{id_suffix}"

            # Extract model name from frame (strip ::link_name suffix).
            model_name = frame_name.split("::")[0]

            # Check for exact model name match.
            if model_name == expected_model_name:
                return {"name": scene_object.name, "id": str(object_id)}

        # Try matching to stack objects via direct model name lookup.
        # Uses member_model_names stored in stack metadata.
        stack_match = _find_stack_by_model_name(frame_name=frame_name, scene=scene)
        if stack_match:
            return stack_match

        raise RuntimeError(
            f"Could not map geometry ID {geometry_id} with frame name '{frame_name}' "
            f"to any scene object. This indicates a mismatch between Drake's internal "
            f"naming and our scene object IDs."
        )

    except Exception as e:
        raise RuntimeError(
            f"Error mapping geometry ID {geometry_id} to object: {e}"
        ) from e


def _get_object_type_for_collision_id(
    object_id: str, scene: RoomScene
) -> ObjectType | None:
    """Get ObjectType for a collision object ID.

    Handles special cases like room_geometry (walls/floor).

    Args:
        object_id: Object ID from collision pair.
        scene: RoomScene for looking up object types.

    Returns:
        ObjectType or None if unknown.
    """
    # Handle room geometry special cases.
    if object_id == "room_geometry" or object_id.startswith("room_geometry::"):
        return None  # Room geometry is not modifiable by any agent.

    # Handle specific wall/floor IDs.
    if "_wall" in object_id.lower():
        return None  # Walls are not modifiable.

    scene_obj = scene.objects.get(UniqueID(object_id))
    return scene_obj.object_type if scene_obj else None


def _is_collision_relevant_to_agent(
    collision: CollisionPair,
    scene: RoomScene,
    agent_type: AgentType,
    current_furniture_id: UniqueID | None = None,
) -> bool:
    """Determine if a collision is relevant to the specified agent.

    A collision is relevant if at least one object in the pair is of a type
    that the agent can modify.

    Args:
        collision: CollisionPair to check.
        scene: RoomScene for object type lookups.
        agent_type: Type of agent checking collisions.
        current_furniture_id: For ManipulandAgent, the furniture being populated.

    Returns:
        True if collision is relevant to the agent.
    """
    type_a = _get_object_type_for_collision_id(
        object_id=collision.object_a_id, scene=scene
    )
    type_b = _get_object_type_for_collision_id(
        object_id=collision.object_b_id, scene=scene
    )

    target_type = agent_type.to_object_type()
    if target_type is None:
        # FLOOR_PLAN agent has no object type - can't modify any objects.
        return False

    if agent_type == AgentType.MANIPULAND:
        # Special case: manipuland must belong to current furniture.
        if current_furniture_id is None:
            return False

        def is_current_manipuland(obj_id: str) -> bool:
            scene_obj = scene.objects.get(UniqueID(obj_id))
            if not scene_obj or scene_obj.object_type != ObjectType.MANIPULAND:
                return False
            parent_id = _get_furniture_id_for_manipuland(
                manipuland_id=obj_id, scene=scene
            )
            return parent_id == str(current_furniture_id)

        return is_current_manipuland(collision.object_a_id) or is_current_manipuland(
            collision.object_b_id
        )

    # Standard case: at least one object must match agent's target type.
    return type_a == target_type or type_b == target_type


def filter_collisions_by_agent(
    collisions: list[CollisionPair],
    scene: RoomScene,
    agent_type: AgentType,
    current_furniture_id: UniqueID | None = None,
) -> list[CollisionPair]:
    """Filter collisions to show only those relevant to the specified agent.

    Each agent type can only modify certain object types:
    - FurnitureAgent: FURNITURE objects
    - ManipulandAgent: MANIPULAND objects on current furniture
    - WallAgent: WALL_MOUNTED objects
    - CeilingAgent: CEILING_MOUNTED objects

    A collision is relevant if at least one object in the pair is of a type
    the agent can modify.

    Args:
        collisions: List of collision pairs to filter.
        scene: RoomScene for looking up object types.
        agent_type: Type of agent requesting collisions.
        current_furniture_id: For ManipulandAgent, the furniture being populated.

    Returns:
        Filtered list of collision pairs.
    """
    filtered = []
    for collision in collisions:
        if _is_collision_relevant_to_agent(
            collision=collision,
            scene=scene,
            agent_type=agent_type,
            current_furniture_id=current_furniture_id,
        ):
            filtered.append(collision)

    if len(filtered) < len(collisions):
        console_logger.debug(
            f"Filtered collisions by agent type {agent_type.value}: "
            f"{len(collisions)} -> {len(filtered)}"
        )

    return filtered
