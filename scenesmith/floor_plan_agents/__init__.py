"""Floor plan agents for designing and generating house layouts."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scenesmith.floor_plan_agents.base_floor_plan_agent import BaseFloorPlanAgent
    from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
        StatefulFloorPlanAgent,
    )

__all__ = [
    "BaseFloorPlanAgent",
    "StatefulFloorPlanAgent",
]


def __getattr__(name: str) -> Any:
    """Load heavy agent implementations only when explicitly requested."""
    if name == "BaseFloorPlanAgent":
        from scenesmith.floor_plan_agents.base_floor_plan_agent import (
            BaseFloorPlanAgent,
        )

        return BaseFloorPlanAgent
    if name == "StatefulFloorPlanAgent":
        from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
            StatefulFloorPlanAgent,
        )

        return StatefulFloorPlanAgent
    raise AttributeError(name)
