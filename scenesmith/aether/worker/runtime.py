"""Resource-bounded native SceneSmith completion runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..runtime import SceneSmithCompletionRuntime


@dataclass(frozen=True, slots=True)
class OperationResources:
    asset_manager: Any
    adapter: Any
    close: Callable[[], None]


OperationFactory = Callable[[str], OperationResources]


class SwitchingCompletionRuntime(SceneSmithCompletionRuntime):
    """Keep no more than one operation category's native servers alive."""

    def __init__(self, *, operation_factory: OperationFactory, **kwargs: Any) -> None:
        super().__init__(asset_managers={}, placement_adapters={}, **kwargs)
        self.operation_factory = operation_factory
        self._active_kind: str | None = None
        self._active: OperationResources | None = None

    def place_asset_brief(self, operation, asset_brief, *, round_index):
        self._activate(str(operation["operation"]))
        return super().place_asset_brief(
            operation, asset_brief, round_index=round_index
        )

    def close(self) -> None:
        if self._active is not None:
            self._active.close()
        self._active = None
        self._active_kind = None
        self.asset_managers.clear()
        self.placement_adapters.clear()

    def _activate(self, kind: str) -> None:
        if kind == self._active_kind:
            return
        self.close()
        resources = self.operation_factory(kind)
        self.asset_managers[kind] = resources.asset_manager
        self.placement_adapters[kind] = resources.adapter
        self._active = resources
        self._active_kind = kind


class NativeOperationFactory:
    """Construct native tools and servers lazily for one completion category."""

    def __init__(
        self,
        *,
        scene: Any,
        stage_input: dict[str, Any],
        cfg_dict: dict[str, Any],
        logger: Any,
        house_layout: Any,
        ceiling_height: float,
        render_gpu_id: int | None,
    ) -> None:
        self.scene = scene
        self.stage_input = stage_input
        self.cfg_dict = cfg_dict
        self.logger = logger
        self.house_layout = house_layout
        self.ceiling_height = ceiling_height
        self.render_gpu_id = render_gpu_id

    def __call__(self, kind: str) -> OperationResources:
        from scenesmith.experiments.base_experiment import BaseExperiment
        from scenesmith.experiments.indoor_scene_generation import (
            IndoorSceneGenerationExperiment,
        )

        experiment = IndoorSceneGenerationExperiment
        if kind == "place-floor-group":
            from scenesmith.furniture_agents.tools.furniture_tools import FurnitureTools
            from scenesmith.aether.native_placement import FloorPlacementAdapter

            agent = BaseExperiment.build_furniture_agent(
                self.cfg_dict,
                experiment.compatible_furniture_agents,
                self.logger,
                self.render_gpu_id,
            )
            tool = FurnitureTools(self.scene, agent.asset_manager, agent.cfg)
            adapter = FloorPlacementAdapter(tool, self.stage_input)
        elif kind == "populate-surfaces":
            from scenesmith.manipuland_agents.tools.manipuland_tools import (
                ManipulandTools,
            )
            from scenesmith.aether.native_placement import SurfacePlacementAdapter

            agent = BaseExperiment.build_manipuland_agent(
                self.cfg_dict,
                experiment.compatible_manipuland_agents,
                self.logger,
                self.render_gpu_id,
            )

            def tool_factory(owner_id, surfaces):
                return ManipulandTools(
                    self.scene, agent.asset_manager, agent.cfg, owner_id, surfaces
                )

            adapter = SurfacePlacementAdapter(tool_factory, self.stage_input)
        elif kind == "place-wall-group":
            if self.house_layout is None:
                raise RuntimeError("wall completion requires the accepted house layout")
            from scenesmith.wall_agents.tools.wall_tools import WallTools
            from scenesmith.aether.native_placement import WallPlacementAdapter

            agent = BaseExperiment.build_wall_agent(
                self.cfg_dict,
                experiment.compatible_wall_agents,
                self.logger,
                self.house_layout,
                self.ceiling_height,
                render_gpu_id=self.render_gpu_id,
            )
            surfaces = agent._extract_wall_surfaces(self.scene.room_id)
            tool = WallTools(self.scene, surfaces, agent.asset_manager, agent.cfg)
            adapter = WallPlacementAdapter(tool, self.stage_input)
        elif kind == "place-ceiling-group":
            from scenesmith.ceiling_agents.tools.ceiling_tools import CeilingTools
            from scenesmith.aether.native_placement import CeilingPlacementAdapter

            agent = BaseExperiment.build_ceiling_agent(
                self.cfg_dict,
                experiment.compatible_ceiling_agents,
                self.logger,
                self.ceiling_height,
                self.render_gpu_id,
            )
            bounds = agent._extract_room_bounds(self.scene)
            tool = CeilingTools(
                self.scene,
                bounds,
                self.ceiling_height,
                agent.asset_manager,
                agent.cfg,
            )
            adapter = CeilingPlacementAdapter(tool, self.stage_input)
        else:
            raise RuntimeError(f"unsupported SceneSmith completion operation {kind}")
        return OperationResources(agent.asset_manager, adapter, agent.cleanup)
