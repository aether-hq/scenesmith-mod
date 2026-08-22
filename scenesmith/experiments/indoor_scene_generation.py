import logging

from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.ceiling_agents.stateful_ceiling_agent import StatefulCeilingAgent
from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent

console_logger = logging.getLogger(__name__)

# Pipeline stages in execution order (derived from AgentType enum).
PIPELINE_STAGES = [agent.value for agent in AgentType]

# Stage dependencies for resume from checkpoint.
# Maps start_stage to the checkpoint it needs from the previous stage.
STAGE_CHECKPOINTS = {
    "floor_plan": None,
    "furniture": None,
    "wall_mounted": "scene_after_furniture",
    "ceiling_mounted": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
}

# Maps start_stage to the asset directories it needs from previous stages.
STAGE_ASSET_DIRS = {
    "floor_plan": [],
    "furniture": [],
    "wall_mounted": ["furniture"],
    "ceiling_mounted": ["furniture", "wall_mounted"],
    "manipuland": ["furniture", "wall_mounted", "ceiling_mounted"],
}

from scenesmith.experiments.indoor.mixins.generation import IndoorGenerationMixin
from scenesmith.experiments.indoor.mixins.service_lifecycle import (
    IndoorServiceLifecycleMixin,
)


class IndoorSceneGenerationExperiment(
    IndoorServiceLifecycleMixin,
    IndoorGenerationMixin,
    BaseExperiment,
):
    """Generate complete indoor scenes with resumable staged execution."""

    compatible_floor_plan_agents = {
        "stateful_floor_plan_agent": StatefulFloorPlanAgent,
    }
    compatible_furniture_agents = {
        "stateful_furniture_agent": StatefulFurnitureAgent,
    }
    compatible_manipuland_agents = {
        "stateful_manipuland_agent": StatefulManipulandAgent,
    }
    compatible_wall_agents = {"stateful_wall_agent": StatefulWallAgent}
    compatible_ceiling_agents = {"stateful_ceiling_agent": StatefulCeilingAgent}
