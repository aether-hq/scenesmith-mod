import asyncio
import faulthandler
import json
import logging
import os

from pathlib import Path
from typing import Callable

from agents import custom_span, trace

from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.experiments.indoor.room_generation import _generate_room
from scenesmith.experiments.indoor.runtime_support import _reset_inherited_sdk_state
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.utils.logging import ConsoleLogger, FileLoggingContext
from scenesmith.utils.parallel import run_parallel_isolated

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


def _run_sequential_room_generation(
    house_layout: HouseLayout,
    logger: ConsoleLogger,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    render_allocation: RenderAllocation | None = None,
) -> dict[str, RoomScene]:
    """Generate rooms sequentially (existing behavior).

    Args:
        house_layout: HouseLayout containing room specs and geometries.
        logger: Logger for output routing.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        render_allocation: Provider-owned Blender render slot.

    Returns:
        Dictionary mapping room_id to RoomScene.
    """
    rooms: dict[str, RoomScene] = {}
    for room_id in house_layout.room_ids:
        room_spec = house_layout.get_room_spec(room_id)
        room_geometry = house_layout.get_room_geometry(room_id)
        if room_geometry is None:
            raise RuntimeError(f"Room geometry not generated for room '{room_id}'")

        with custom_span(f"room_{room_id}_generation"):
            with logger.room_context(room_id) as room_dir:
                console_logger.info(f"Generating room '{room_id}': {room_spec.prompt}")
                room_scene = _generate_room(
                    room_id=room_id,
                    room_prompt=room_spec.prompt,
                    room_geometry=room_geometry,
                    room_dir=room_dir,
                    logger=logger,
                    cfg_dict=cfg_dict,
                    start_stage=start_stage,
                    stop_stage=stop_stage,
                    house_layout=house_layout,
                    render_allocation=render_allocation,
                )
                rooms[room_id] = room_scene
    return rooms


def _generate_floor_plan_worker(
    prompt: str,
    scene_dir: str,
    cfg_dict: dict,
    experiment_run_id: str | None,
    render_allocation: RenderAllocation | None = None,
    reset_sdk_state: bool = True,
) -> None:
    """Run floor plan generation in isolated subprocess.

    This function runs in a separate process to ensure all fork-unsafe state
    (SQLiteSession locks, tracing threads) is destroyed when the subprocess
    exits, before we fork room workers.

    Args:
        prompt: Scene description prompt.
        scene_dir: Path to scene output directory (as string).
        cfg_dict: Configuration dictionary.
        experiment_run_id: Unique ID for this experiment run.
        render_allocation: Provider-owned Blender render slot.
    """
    # Reset any SDK state inherited via fork (defense in depth). Layout-only
    # runs execute this worker inline and must preserve the parent trace.
    if reset_sdk_state:
        _reset_inherited_sdk_state()

    faulthandler.enable()

    scene_path = Path(scene_dir)
    logger = ConsoleLogger(output_dir=scene_path)

    # Use FileLoggingContext to capture floor plan logs to scene.log.
    log_path = scene_path / "scene.log"
    with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
        console_logger.info(f"Floor plan worker started for scene: {scene_dir}")

        # Create trace metadata for this floor plan generation.
        trace_metadata = {"scene_dir": scene_dir, "prompt": prompt}
        if experiment_run_id:
            trace_metadata["experiment_run_id"] = experiment_run_id

        with trace(workflow_name="floor_plan_generation", metadata=trace_metadata):
            with custom_span("floor_plan_generation"):
                floor_plan_agent = BaseExperiment.build_floor_plan_agent(
                    cfg_dict=cfg_dict,
                    compatible_agents=(
                        {
                            "stateful_floor_plan_agent": StatefulFloorPlanAgent,
                        }
                    ),
                    logger=logger,
                    render_allocation=render_allocation,
                )
                try:
                    revision_source = os.environ.get(
                        "SCENESMITH_REVISION_SOURCE_LAYOUT"
                    )
                    revision_feedback = os.environ.get("SCENESMITH_REVISION_FEEDBACK")
                    if revision_source and revision_feedback:
                        source_layout_path = Path(revision_source)
                        if not source_layout_path.exists():
                            raise FileNotFoundError(
                                "Revision source layout does not exist: "
                                f"{source_layout_path}"
                            )
                        with source_layout_path.open() as file:
                            source_layout = HouseLayout.from_dict(
                                json.load(file), house_dir=source_layout_path.parent
                            )
                        try:
                            locks = tuple(
                                str(value)
                                for value in json.loads(
                                    os.environ.get("SCENESMITH_REVISION_LOCKS", "[]")
                                )
                            )
                        except (TypeError, json.JSONDecodeError):
                            locks = ()
                        console_logger.info(
                            "Restoring floor-plan checkpoint for architectural revision"
                        )
                        house_layout = asyncio.run(
                            floor_plan_agent.revise_house_layout(
                                existing_layout=source_layout,
                                feedback=revision_feedback,
                                output_dir=scene_path / "floor_plans",
                                locks=locks,
                            )
                        )
                    else:
                        house_layout = asyncio.run(
                            floor_plan_agent.generate_house_layout(
                                prompt=prompt,
                                output_dir=scene_path / "floor_plans",
                            )
                        )
                finally:
                    floor_plan_agent.cleanup()

                # Save to disk for parent to load.
                house_layout_path = scene_path / "house_layout.json"
                with open(house_layout_path, "w") as f:
                    json.dump(house_layout.to_dict(scene_dir=scene_path), f, indent=2)
                console_logger.info(f"Saved house layout to {house_layout_path}")


def _generate_room_worker(
    room_id: str,
    room_prompt: str,
    room_geometry_dict: dict,
    room_dir: str,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    scene_id: int,
    experiment_run_id: str | None = None,
    house_layout_dict: dict | None = None,
    render_allocation: RenderAllocation | None = None,
) -> dict:
    """Worker function for parallel room generation.

    Runs in a subprocess. All args must be picklable (no Path, no complex objects).

    Note on tracing: Room traces are INDEPENDENT from parent scene trace because
    ProcessPoolExecutor creates separate processes. We include scene_id in metadata
    to enable correlation via trace queries.

    Args:
        room_id: Unique identifier for the room.
        room_prompt: Text description for the room.
        room_geometry_dict: Serialized RoomGeometry dictionary.
        room_dir: Path to room output directory (as string).
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        scene_id: Parent scene ID for trace correlation.
        experiment_run_id: Unique ID for this experiment run.
        house_layout_dict: Optional serialized HouseLayout for door/window export.
        render_allocation: Provider-owned Blender render slot.

    Returns:
        Dict containing scene_state and metadata for reconstruction.
    """
    # Reset any SDK state inherited via fork (defense in depth).
    _reset_inherited_sdk_state()

    room_dir_path = Path(room_dir)

    faulthandler.enable()

    log_path = room_dir_path / "room.log"

    # Create logger for this room (logs to file, not stdout).
    room_logger = ConsoleLogger(output_dir=room_dir_path)

    # Reconstruct RoomGeometry from serialized dict.
    room_geometry = RoomGeometry.from_dict(room_geometry_dict, scene_dir=room_dir_path)

    # Reconstruct HouseLayout from serialized dict (if provided).
    house_layout = None
    if house_layout_dict:
        house_layout = HouseLayout.from_dict(
            house_layout_dict, house_dir=room_dir_path.parent
        )

    # Use FileLoggingContext to capture logs to room.log.
    with FileLoggingContext(log_file_path=log_path, suppress_stdout=True):
        console_logger.info(
            f"Worker started for room '{room_id}' with room prompt '{room_prompt}'"
        )

        # Create trace metadata for this room.
        trace_metadata = {
            "room_id": room_id,
            "parent_scene_id": f"scene_{scene_id:03d}",
            "experiment_name": cfg_dict["name"],
            "room_dir": str(room_dir_path),
            "room_prompt": room_prompt,
        }
        if experiment_run_id:
            trace_metadata["experiment_run_id"] = experiment_run_id

        with trace(
            workflow_name=f"scene_{scene_id:03d}_room_{room_id}",
            metadata=trace_metadata,
        ):
            room_scene = _generate_room(
                room_id=room_id,
                room_prompt=room_prompt,
                room_geometry=room_geometry,
                room_dir=room_dir_path,
                logger=room_logger,
                cfg_dict=cfg_dict,
                start_stage=start_stage,
                stop_stage=stop_stage,
                house_layout=house_layout,
                render_allocation=render_allocation,
            )

        console_logger.info(f"Worker completed for room '{room_id}'")

    # Return serializable result for cross-process transfer.
    return {
        "scene_state": room_scene.to_state_dict(),
        "room_id": room_scene.room_id,
        "text_description": room_scene.text_description,
    }


def _reconstruct_room_scene(worker_result: dict, scene_dir: Path) -> RoomScene:
    """Reconstruct RoomScene from worker result dict.

    Args:
        worker_result: Dict containing scene_state from worker.
        scene_dir: Path to room directory for path resolution.

    Returns:
        Reconstructed RoomScene.
    """
    scene_state = worker_result["scene_state"]

    # Reconstruct RoomGeometry first (needed for RoomScene constructor).
    room_geometry = RoomGeometry.from_dict(
        scene_state["room_geometry"], scene_dir=scene_dir
    )

    # Create RoomScene with required fields.
    room_scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=scene_dir,
        room_id=worker_result["room_id"],
        text_description=worker_result.get("text_description", ""),
        action_log_path=scene_dir / "action_log.json",
    )

    # Restore objects and other state.
    room_scene.restore_from_state_dict(scene_state)

    return room_scene


def _run_parallel_room_generation(
    house_layout: HouseLayout,
    output_dir: Path,
    cfg_dict: dict,
    start_stage: str,
    stop_stage: str,
    max_workers: int,
    scene_id: int,
    experiment_run_id: str | None = None,
    render_allocation: RenderAllocation | None = None,
) -> dict[str, RoomScene]:
    """Generate rooms in parallel with fault tolerance.

    Uses isolated processes per room instead of a shared executor pool.
    This ensures that if one room crashes, other rooms continue running.

    Args:
        house_layout: HouseLayout containing room specs and geometries.
        output_dir: Base output directory for the scene.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from.
        stop_stage: Stage to stop after.
        max_workers: Maximum number of concurrent room processes.
        scene_id: Scene identifier for trace correlation.
        experiment_run_id: Unique ID for this experiment run.
        render_allocation: Provider-owned Blender render slot.

    Returns:
        Dictionary mapping room_id to RoomScene.

    Raises:
        RuntimeError: If any room generation fails.
    """
    console_logger.info("Running room generation in parallel")

    # Build task list.
    tasks: list[tuple[str, Callable, dict]] = []
    room_dirs: dict[str, Path] = {}
    for room_id in house_layout.room_ids:
        room_spec = house_layout.get_room_spec(room_id)
        room_geometry = house_layout.get_room_geometry(room_id)
        if room_geometry is None:
            raise RuntimeError(f"Room geometry not generated for room '{room_id}'")

        # Create room directory (must exist before worker starts).
        room_dir = output_dir / f"room_{room_id}"
        room_dir.mkdir(parents=True, exist_ok=True)
        room_dirs[room_id] = room_dir

        console_logger.info(f"Queued room '{room_id}' (logs → {room_dir / 'room.log'})")

        kwargs = {
            "room_id": room_id,
            "room_prompt": room_spec.prompt,
            "room_geometry_dict": room_geometry.to_dict(scene_dir=room_dir),
            "room_dir": str(room_dir),
            "cfg_dict": cfg_dict,
            "start_stage": start_stage,
            "stop_stage": stop_stage,
            "scene_id": scene_id,
            "experiment_run_id": experiment_run_id,
            "house_layout_dict": house_layout.to_dict(scene_dir=output_dir),
            "render_allocation": render_allocation,
        }
        tasks.append((room_id, _generate_room_worker, kwargs))

    # Run with fault tolerance and get return values.
    results = run_parallel_isolated(
        tasks=tasks, max_workers=max_workers, return_values=True
    )

    # Reconstruct RoomScenes from worker results.
    rooms: dict[str, RoomScene] = {}
    failures: list[tuple[str, str]] = []
    for room_id, (success, result_or_error) in results.items():
        room_dir = room_dirs[room_id]
        if success:
            rooms[room_id] = _reconstruct_room_scene(
                worker_result=result_or_error, scene_dir=room_dir
            )
            console_logger.info(f"Room '{room_id}' completed successfully")
        else:
            console_logger.error(f"Room '{room_id}' failed: {result_or_error}")
            failures.append((room_id, result_or_error))

    if failures:
        error_msg = "; ".join(f"Room '{rid}': {err}" for rid, err in failures)
        raise RuntimeError(f"Room generation failures: {error_msg}")

    return rooms
