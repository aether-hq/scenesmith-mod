import faulthandler
import json
import logging
import os
import platform
import time
import uuid

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from agents import trace
from omegaconf import OmegaConf

from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.scene.house import HouseLayout, HouseScene
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType, ObjectType
from scenesmith.ceiling_agents.stateful_ceiling_agent import StatefulCeilingAgent
from scenesmith.experiments.indoor.checkpoint_io import _copy_checkpoint_for_stage
from scenesmith.experiments.indoor.runtime_support import (
    RenderAllocationAllocator,
    _load_prompts_from_csv,
    _reset_inherited_sdk_state,
)
from scenesmith.experiments.indoor.workers import (
    _generate_floor_plan_worker,
    _run_parallel_room_generation,
    _run_sequential_room_generation,
)
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.utils.logging import ConsoleLogger, FileLoggingContext
from scenesmith.utils.parallel import run_parallel_isolated
from scenesmith.utils.print_utils import bold_green, yellow
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


class IndoorGenerationMixin:
    """An experiment that generates indoor scenes."""

    compatible_floor_plan_agents = {
        "stateful_floor_plan_agent": StatefulFloorPlanAgent,
    }
    compatible_furniture_agents = {
        "stateful_furniture_agent": StatefulFurnitureAgent,
    }
    compatible_manipuland_agents = {
        "stateful_manipuland_agent": StatefulManipulandAgent,
    }
    compatible_wall_agents = {
        "stateful_wall_agent": StatefulWallAgent,
    }
    compatible_ceiling_agents = {
        "stateful_ceiling_agent": StatefulCeilingAgent,
    }

    @staticmethod
    def _generate_single_scene(
        prompt: str,
        scene_id: int,
        output_dir: Path,
        cfg_dict: dict,
        capture_logs: bool = False,
        experiment_run_id: str | None = None,
        render_allocation: RenderAllocation | None = None,
    ) -> None:
        """Generate a single scene (static method for parallel execution).

        Pipeline stages run in order:
        floor_plan → furniture → wall_mounted → ceiling_mounted → manipulands
        Use config pipeline.start_stage and pipeline.stop_stage to control execution.

        Args:
            prompt: Scene description.
            scene_id: Scene identifier.
            output_dir: Base output directory for the experiment.
            cfg_dict: Configuration as dictionary.
            capture_logs: If True, suppress stdout and only write to file.
            experiment_run_id: Unique ID for this experiment run.
            render_allocation: Provider-owned Blender render slot.
        """
        # Reset any SDK state inherited via fork (defense in depth).
        _reset_inherited_sdk_state()

        faulthandler.enable()

        scene_generation_start_time = time.time()

        # Create scene directory.
        scene_dir = output_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        # Always create log file.
        log_path = scene_dir / "scene.log"

        # Log start message before potential suppression.
        if capture_logs:
            console_logger.info(
                f"Scene {scene_id:03d} started (logs → {log_path})\n"
                f"Prompt: {prompt}"
            )
        else:
            console_logger.info(
                f"Scene {scene_id:03d} started (debug mode)\nPrompt: {prompt}"
            )

        # Create a logger for this scene.
        logger = ConsoleLogger(output_dir=scene_dir)

        # Get pipeline stage configuration.
        pipeline_cfg = cfg_dict["experiment"]["pipeline"]
        start_stage = pipeline_cfg["start_stage"]
        stop_stage = pipeline_cfg["stop_stage"]

        # Validate stages.
        if start_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid start_stage '{start_stage}'. "
                f"Valid options: {PIPELINE_STAGES}"
            )
        if stop_stage not in PIPELINE_STAGES:
            raise ValueError(
                f"Invalid stop_stage '{stop_stage}'. "
                f"Valid options: {PIPELINE_STAGES}"
            )

        start_idx = PIPELINE_STAGES.index(start_stage)
        stop_idx = PIPELINE_STAGES.index(stop_stage)
        if start_idx > stop_idx:
            raise ValueError(
                f"start_stage '{start_stage}' cannot be after stop_stage '{stop_stage}'"
            )

        console_logger.info(
            f"Pipeline: start_stage='{start_stage}', stop_stage='{stop_stage}'"
        )

        # Handle resume from checkpoint if resume_from_path is specified.
        resume_from_path = pipeline_cfg.get("resume_from_path")
        if resume_from_path and start_stage != "floor_plan":
            source_experiment_dir = Path(resume_from_path)
            if not source_experiment_dir.exists():
                raise FileNotFoundError(
                    f"resume_from_path does not exist: {resume_from_path}"
                )
            _copy_checkpoint_for_stage(
                source_scene_dir=source_experiment_dir / f"scene_{scene_id:03d}",
                target_scene_dir=scene_dir,
                start_stage=start_stage,
            )

        with FileLoggingContext(log_file_path=log_path, suppress_stdout=capture_logs):
            try:
                # Create trace metadata for this scene.
                trace_metadata = {
                    "scene_id": f"scene_{scene_id:03d}",
                    "experiment_name": cfg_dict["name"],
                    "scene_dir": str(scene_dir),
                    "prompt": prompt,
                }
                if experiment_run_id:
                    trace_metadata["experiment_run_id"] = experiment_run_id

                console_logger.info(f"Generating scene for prompt: {prompt}")

                # Single trace wraps entire scene generation (floor plan + rooms).
                with trace(
                    workflow_name=f"scene_{scene_id:03d}_generation",
                    metadata=trace_metadata,
                ):
                    # Stage 1: Floor plan generation (or load from saved state).
                    if start_stage == "floor_plan":
                        # Run floor plan in subprocess to isolate fork-unsafe SDK
                        # state (SQLiteSession locks, tracing threads). The subprocess
                        # saves results to disk and exits cleanly before we fork room
                        # workers.
                        console_logger.info("Generating house layout")
                        layout_start_time = time.time()

                        floor_plan_kwargs = {
                            "prompt": prompt,
                            "scene_dir": str(scene_dir),
                            "cfg_dict": cfg_dict,
                            "experiment_run_id": experiment_run_id,
                            "render_allocation": render_allocation,
                        }
                        if stop_stage == "floor_plan" or platform.system() == "Darwin":
                            # macOS multiprocessing uses spawn, which re-imports the
                            # Hydra/Blender application entrypoint and can terminate
                            # before the floor-plan worker starts. Sequential room
                            # generation does not need this isolation, so keep the
                            # whole local Mac pipeline in the parent process.
                            console_logger.info(
                                "Running floor plan inline for this pipeline"
                            )
                            _generate_floor_plan_worker(
                                **floor_plan_kwargs, reset_sdk_state=False
                            )
                        else:
                            # Run floor plan generation in isolated subprocess.
                            results = run_parallel_isolated(
                                tasks=[
                                    (
                                        "floor_plan",
                                        _generate_floor_plan_worker,
                                        floor_plan_kwargs,
                                    )
                                ],
                                max_workers=1,
                            )

                            # Check for failure.
                            success, error = results["floor_plan"]
                            if not success:
                                raise RuntimeError(
                                    f"Floor plan generation failed: {error}"
                                )

                        # Load result from disk (subprocess saved it).
                        house_layout_path = scene_dir / "house_layout.json"
                        with open(house_layout_path) as f:
                            house_layout_dict = json.load(f)
                        house_layout = HouseLayout.from_dict(
                            house_layout_dict, house_dir=scene_dir
                        )
                        layout_end_time = time.time()
                        console_logger.info(
                            f"House layout generated in "
                            f"{timedelta(seconds=layout_end_time - layout_start_time)}"
                        )
                    else:
                        # Load house layout from saved state.
                        house_layout_path = scene_dir / "house_layout.json"
                        if not house_layout_path.exists():
                            raise FileNotFoundError(
                                f"Cannot start from '{start_stage}' stage: "
                                f"house_layout.json not found at {house_layout_path}. "
                                "Run with start_stage='floor_plan' first."
                            )
                        console_logger.info(
                            f"Loading house layout from {house_layout_path}"
                        )
                        with open(house_layout_path) as f:
                            house_layout_dict = json.load(f)
                        house_layout = HouseLayout.from_dict(
                            house_layout_dict, house_dir=scene_dir
                        )

                    revision_feedback = os.environ.get("SCENESMITH_REVISION_FEEDBACK")
                    if revision_feedback:
                        console_logger.info(
                            "Applying revision feedback from restored %s checkpoint",
                            start_stage,
                        )
                        if (
                            f"Revision request: {revision_feedback}"
                            not in house_layout.house_prompt
                        ):
                            house_layout.house_prompt = (
                                f"{house_layout.house_prompt}\n"
                                f"Revision request: {revision_feedback}"
                            )
                        for room_spec in house_layout.room_specs:
                            if (
                                f"Revision request: {revision_feedback}"
                                not in room_spec.prompt
                            ):
                                room_spec.prompt = (
                                    f"{room_spec.prompt}\n"
                                    f"Revision request: {revision_feedback}"
                                )

                    # Check if we should stop after floor_plan stage.
                    if stop_stage == "floor_plan":
                        console_logger.info(
                            "Stopping after floor_plan stage as configured"
                        )
                        console_logger.info(
                            "Scene generation completed successfully in "
                            f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                        )
                        return

                    # Stages 2-4: Furniture, wall objects, and manipulands (per-room).
                    # Determine room-level start/stop stages.
                    room_start_stage = (
                        "furniture" if start_stage == "floor_plan" else start_stage
                    )
                    room_stop_stage = stop_stage

                    # Generate rooms (parallel or sequential based on config).
                    parallel_rooms = pipeline_cfg["parallel_rooms"]
                    max_parallel_rooms = pipeline_cfg["max_parallel_rooms"]
                    num_rooms = len(house_layout.room_ids)

                    # Only use parallel if enabled, max_workers > 1, and multiple rooms.
                    use_parallel = (
                        parallel_rooms and max_parallel_rooms > 1 and num_rooms > 1
                    )

                    if use_parallel:
                        rooms = _run_parallel_room_generation(
                            house_layout=house_layout,
                            output_dir=scene_dir,
                            cfg_dict=cfg_dict,
                            start_stage=room_start_stage,
                            stop_stage=room_stop_stage,
                            max_workers=max_parallel_rooms,
                            scene_id=scene_id,
                            experiment_run_id=experiment_run_id,
                            render_allocation=render_allocation,
                        )
                    else:
                        rooms = _run_sequential_room_generation(
                            house_layout=house_layout,
                            logger=logger,
                            cfg_dict=cfg_dict,
                            start_stage=room_start_stage,
                            stop_stage=room_stop_stage,
                            render_allocation=render_allocation,
                        )

                    # Build HouseScene from generated rooms.
                    house_scene = HouseScene(layout=house_layout, rooms=rooms)

                    # Assemble house with intermediate snapshots filtered by object type.
                    # Each snapshot includes objects from completed stages only.
                    # Note: Thin coverings keep their agent's object_type (FURNITURE,
                    # WALL_MOUNTED, MANIPULAND) so they're included automatically.
                    snapshots = [
                        ("combined_house_after_furniture", [ObjectType.FURNITURE]),
                        (
                            "combined_house_after_wall_objects",
                            [ObjectType.FURNITURE, ObjectType.WALL_MOUNTED],
                        ),
                        (
                            "combined_house_after_ceiling",
                            [
                                ObjectType.FURNITURE,
                                ObjectType.WALL_MOUNTED,
                                ObjectType.CEILING_MOUNTED,
                            ],
                        ),
                        ("combined_house", None),  # Final: all objects.
                    ]

                    # Map stop_stage to number of snapshots to create.
                    stage_to_count = {
                        "furniture": 1,
                        AgentType.WALL_MOUNTED.value: 2,
                        AgentType.CEILING_MOUNTED.value: 3,
                    }
                    snapshot_count = stage_to_count.get(stop_stage, len(snapshots))

                    for name, types in snapshots[:snapshot_count]:
                        house_scene.assemble(
                            cfg=cfg_dict, output_name=name, include_object_types=types
                        )

                    console_logger.info(
                        "Scene generation completed successfully in "
                        f"{timedelta(seconds=time.time() - scene_generation_start_time)}"
                    )

            except Exception as e:
                console_logger.error(f"Scene generation failed: {e}")
                raise

    def _run_serial_generation(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        experiment_run_id: str,
    ) -> None:
        """Run scene generation in serial."""
        console_logger.info("Running scene generation serially in main thread")

        # GPU distribution is useful for parallel rooms within each scene.
        allocation_allocator = RenderAllocationAllocator(
            self.provider_selection.render,
            self.provider_selection.render_process,
        )

        for scene_id, prompt in prompts_with_ids:
            render_allocation = allocation_allocator.allocate()
            self._generate_single_scene(
                prompt=prompt,
                scene_id=scene_id,
                output_dir=self.output_dir,
                cfg_dict=cfg_dict,
                capture_logs=False,
                experiment_run_id=experiment_run_id,
                render_allocation=render_allocation,
            )
            console_logger.info(f"Completed scene {scene_id:03d}")

    def _run_parallel_generation(
        self,
        prompts_with_ids: list[tuple[int, str]],
        cfg_dict: dict,
        experiment_run_id: str,
        num_workers: int,
    ) -> None:
        """Run scene generation in parallel with fault tolerance.

        Uses isolated processes per scene instead of a shared executor pool.
        This ensures that if one scene crashes (e.g., GPU OOM), other scenes
        continue running unaffected.

        Raises:
            RuntimeError: If any scene generation fails.
        """
        console_logger.info(f"Running in parallel with {num_workers} workers")

        # Create GPU allocator for distributing Blender rendering.
        allocation_allocator = RenderAllocationAllocator(
            self.provider_selection.render,
            self.provider_selection.render_process,
        )

        # Build task list.
        tasks: list[tuple[str, Callable, dict]] = []
        for scene_id, prompt in prompts_with_ids:
            render_allocation = allocation_allocator.allocate()
            task_id = f"scene_{scene_id:03d}"
            kwargs = {
                "prompt": prompt,
                "scene_id": scene_id,
                "output_dir": self.output_dir,
                "cfg_dict": cfg_dict,
                "capture_logs": True,
                "experiment_run_id": experiment_run_id,
                "render_allocation": render_allocation,
            }
            tasks.append(
                (
                    task_id,
                    type(self)._generate_single_scene,
                    kwargs,
                )
            )
            console_logger.info(
                "Queued %s on %s: %s",
                task_id,
                render_allocation.target_label,
                prompt,
            )

        # Run with fault tolerance - one crash doesn't affect others.
        results = run_parallel_isolated(tasks=tasks, max_workers=num_workers)

        # Report failures.
        failed_scenes = [
            (task_id, error)
            for task_id, (success, error) in results.items()
            if not success
        ]
        if failed_scenes:
            failure_details = "\n".join(
                f"  - {task_id}: {error}" for task_id, error in failed_scenes
            )
            raise RuntimeError(
                f"{len(failed_scenes)}/{len(tasks)} scene(s) failed:\n{failure_details}"
            )

    def generate_scenes(self) -> None:
        """Generate scenes with parallel support."""
        # Load prompts from CSV or YAML config.
        csv_path = self.cfg.experiment.csv_path
        if csv_path:
            prompts_with_ids = _load_prompts_from_csv(csv_path)
            console_logger.info(
                f"Loaded {len(prompts_with_ids)} prompts from CSV: {csv_path}"
            )
        else:
            prompts = self.cfg.experiment.prompts
            prompts_with_ids = list(enumerate(prompts))

        num_workers = min(self.cfg.experiment.num_workers, len(prompts_with_ids))

        # Get pipeline stage configuration.
        pipeline_cfg = self.cfg.experiment.pipeline
        start_stage = pipeline_cfg.start_stage
        stop_stage = pipeline_cfg.stop_stage
        parallel_rooms = pipeline_cfg.parallel_rooms

        # Validate mutual exclusion: parallel scenes vs parallel rooms.
        if parallel_rooms and num_workers > 1:
            raise ValueError(
                "Cannot use both parallel rooms and parallel scenes. "
                "Set num_workers=1 to use parallel_rooms, or set parallel_rooms=false."
            )

        # Generate experiment run ID for trace filtering.
        experiment_run_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        console_logger.info(f"Starting scene generation with {num_workers} workers")
        console_logger.info(f"Processing {len(prompts_with_ids)} scenes")
        console_logger.info(f"Experiment run ID: {experiment_run_id}")
        console_logger.info(
            f"Pipeline stages: start='{start_stage}', stop='{stop_stage}'"
        )

        # Convert config to dictionary for static method.
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        requires_room_asset_servers = stop_stage != "floor_plan"
        floor_plan_uses_material_server = (
            start_stage == "floor_plan"
            and self.cfg.floor_plan_agent.materials.use_retrieval_server
        )

        try:
            # Floor-plan generation does not retrieve or generate room assets.
            # Avoid initializing those later-stage services so a layout-only run
            # can work without SAM3D checkpoints or optional asset datasets.
            if requires_room_asset_servers:
                self._start_geometry_server()
                self._start_hssd_server()
                self._start_objaverse_server()
                self._start_polyhaven_server()
                self._start_articulated_server()
            else:
                console_logger.info(
                    "Skipping room asset services for floor-plan-only pipeline"
                )

            # Room agents use semantic material lookup. The floor-plan agent can
            # instead use its two local defaults for a lightweight layout build.
            if requires_room_asset_servers or floor_plan_uses_material_server:
                self._start_materials_server()
            else:
                console_logger.info(
                    "Using local default materials; retrieval server not required"
                )

            if num_workers == 1:
                self._run_serial_generation(
                    prompts_with_ids=prompts_with_ids,
                    cfg_dict=cfg_dict,
                    experiment_run_id=experiment_run_id,
                )
            else:
                self._run_parallel_generation(
                    prompts_with_ids=prompts_with_ids,
                    cfg_dict=cfg_dict,
                    experiment_run_id=experiment_run_id,
                    num_workers=num_workers,
                )

            console_logger.info("All scenes completed")

            # Log clear completion message.
            console_logger.info("=" * 60)
            console_logger.info(bold_green("ALL SCENES COMPLETED!"))
            console_logger.info("=" * 60)
            console_logger.info(yellow("Press Ctrl+C to exit the script."))
            console_logger.info("=" * 60)

        finally:
            # Stop GPU servers.
            self._stop_materials_server()
            self._stop_articulated_server()
            self._stop_polyhaven_server()
            self._stop_objaverse_server()
            self._stop_hssd_server()
            self._stop_geometry_server()

    def evaluate_scenes(self) -> None:
        """
        Evaluate previously generated scenes.
        """
        raise NotImplementedError
