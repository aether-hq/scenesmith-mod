"""Physics validation for scenes using Drake."""

import logging

from scenesmith.agent_utils.scene.clearance_zones import (
    DoorClearanceViolation,
    OpenConnectionBlockedViolation,
    WallHeightExceededViolation,
    WindowClearanceViolation,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.validation.collision_filtering import (
    _get_object_type_for_collision_id,
)


def filter_window_violations_by_agent(
    violations: list[WindowClearanceViolation], scene: RoomScene, agent_type: AgentType
) -> list[WindowClearanceViolation]:
    """Filter window clearance violations by agent type.

    Only shows violations where the blocking object is of a type the agent can modify.

    Args:
        violations: List of window clearance violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        # Floor plan agent doesn't place objects that can block windows.
        return []

    filtered = []
    for v in violations:
        obj_type = _get_object_type_for_collision_id(v.furniture_id, scene)
        if obj_type == target_object_type:
            filtered.append(v)

    return filtered


def filter_wall_height_violations_by_agent(
    violations: list[WallHeightExceededViolation],
    scene: RoomScene,
    agent_type: AgentType,
) -> list[WallHeightExceededViolation]:
    """Filter wall height violations by agent type.

    Only shows violations where the object exceeding wall height is of a type
    the agent can modify.

    Args:
        violations: List of wall height violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        # Floor plan agent doesn't place objects that can exceed wall height.
        return []

    filtered = []
    for v in violations:
        obj_type = _get_object_type_for_collision_id(v.object_id, scene)
        if obj_type == target_object_type:
            filtered.append(v)

    return filtered


def filter_door_violations_by_agent(
    violations: list[DoorClearanceViolation],
    scene: RoomScene,
    agent_type: AgentType,
) -> list[DoorClearanceViolation]:
    """Filter door clearance violations by agent type.

    Only shows violations where the blocking object is of a type the agent can modify.

    Args:
        violations: List of door clearance violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        return []

    filtered = []
    for v in violations:
        obj_type = _get_object_type_for_collision_id(v.furniture_id, scene)
        if obj_type == target_object_type:
            filtered.append(v)

    return filtered


def filter_open_connection_violations_by_agent(
    violations: list[OpenConnectionBlockedViolation],
    scene: RoomScene,
    agent_type: AgentType,
) -> list[OpenConnectionBlockedViolation]:
    """Filter open connection violations by agent type.

    Only shows violations where at least one blocking object is of a type the agent
    can modify.

    Args:
        violations: List of open connection violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        return []

    filtered = []
    for v in violations:
        # Check if any blocking furniture is of the agent's type.
        for furniture_id in v.blocking_furniture_ids:
            obj_type = _get_object_type_for_collision_id(furniture_id, scene)
            if obj_type == target_object_type:
                filtered.append(v)
                break  # Only add once per violation.

    return filtered
