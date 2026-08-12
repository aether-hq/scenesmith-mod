"""Tools for floor plan agents."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
    from scenesmith.floor_plan_agents.tools.vision_tools import FloorPlanVisionTools

__all__ = ["FloorPlanTools", "FloorPlanVisionTools"]


def __getattr__(name: str) -> Any:
    """Keep geometry utilities importable without agent/render dependencies."""
    if name == "FloorPlanTools":
        from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools

        return FloorPlanTools
    if name == "FloorPlanVisionTools":
        from scenesmith.floor_plan_agents.tools.vision_tools import FloorPlanVisionTools

        return FloorPlanVisionTools
    raise AttributeError(name)
