import asyncio
import json
import logging
import time

from datetime import timedelta
from pathlib import Path

from agents import custom_span

from scenesmith.agent_utils.blender.process_provider import RenderAllocation
from scenesmith.agent_utils.design.design_system import (
    apply_style_bible,
    compile_style_bible,
    load_design_system_from_env,
    persist_design_contract,
)
from scenesmith.agent_utils.design.room_kits import select_room_kit
from scenesmith.agent_utils.physics.physical_feasibility import (
    apply_physical_feasibility_postprocessing,
)
from scenesmith.agent_utils.rendering.sceneeval_exporter import (
    SceneEvalExportConfig,
    SceneEvalExporter,
)
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.agent_utils.semantics.environment.semantic_group_materializer import (
    load_and_materialize_locked_semantic_groups,
)
from scenesmith.ceiling_agents.stateful_ceiling_agent import StatefulCeilingAgent
from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.experiments.indoor.publication import certify_room_publication
from scenesmith.experiments.indoor.runtime_support import (
    _export_scene_blend_file,
    _require_projection_success,
    _validate_final_dense_library_book_rows,
)
from scenesmith.furniture_agents.room_kit.validation import (
    _validate_room_kit_completion,
)
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.utils.logging import ConsoleLogger
from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent

console_logger = logging.getLogger(__name__)

COMPATIBLE_FURNITURE_AGENTS = {
    "stateful_furniture_agent": StatefulFurnitureAgent,
}
COMPATIBLE_MANIPULAND_AGENTS = {
    "stateful_manipuland_agent": StatefulManipulandAgent,
}
COMPATIBLE_WALL_AGENTS = {"stateful_wall_agent": StatefulWallAgent}
COMPATIBLE_CEILING_AGENTS = {"stateful_ceiling_agent": StatefulCeilingAgent}

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


def _generate_room(
    room_id: str,
    room_prompt: str,
    room_geometry: RoomGeometry,
    room_dir: Path,
    logger: ConsoleLogger,
    cfg_dict: dict,
    start_stage: str = "furniture",
    stop_stage: str = "manipuland",
    house_layout: HouseLayout | None = None,
    render_allocation: RenderAllocation | None = None,
) -> RoomScene:
    """Generate a single room with furniture, wall/ceiling objects, and manipulands.

    This is the core room generation function used by both single-room and
    multi-room (house) modes. It receives a pre-generated RoomGeometry from the
    HouseLayout and handles furniture, wall object, ceiling object, and
    manipuland placement.

    The room geometry is generated at the house level (by the floor plan generator)
    and passed in here. This ensures consistent handling for both single-room
    and multi-room modes.

    Pipeline stages run in order: furniture → wall_mounted → ceiling_mounted → manipuland
    (floor_plan stage is handled at house level before calling this function)

    State is always saved after each stage for resumability:
    - After furniture: scene_after_furniture.json
    - After wall_mounted: scene_after_wall_objects.json
    - After ceiling_mounted: scene_after_ceiling_objects.json
    - After manipuland: scene_after_manipulands.json (via final_scene logging)

    Args:
        room_id: Unique identifier for the room (e.g., "main", "living_room").
        room_prompt: Text description for the room.
        room_geometry: Pre-generated RoomGeometry from HouseLayout.
        room_dir: Directory for room outputs (e.g., scene_000/room_main/).
        logger: Logger instance for saving outputs.
        cfg_dict: Configuration dictionary.
        start_stage: Stage to start from ("furniture", "wall_mounted",
            "ceiling_mounted", or "manipuland").
        stop_stage: Stage to stop after ("furniture", "wall_mounted",
            "ceiling_mounted", or "manipuland").
        house_layout: Optional HouseLayout for door/window export in SceneEval.
        render_allocation: Provider-owned Blender render slot.

    Returns:
        RoomScene with furniture, wall/ceiling objects, and (optionally) manipulands.
    """
    room_start_time = time.time()

    # Create scene and add walls and floor from room geometry.
    design_system = load_design_system_from_env()
    if design_system is not None:
        style_bible = compile_style_bible(design_system)
        persist_design_contract(design_system, style_bible, room_dir)
        room_prompt = apply_style_bible(room_prompt, style_bible)

    scene = RoomScene(
        room_geometry=room_geometry,
        scene_dir=room_dir,
        room_id=room_id,
        text_description=room_prompt,
        action_log_path=room_dir / "action_log.json",
    )
    for wall in room_geometry.walls:
        scene.add_object(wall)
    # Note: Floor is NOT added to scene.objects to avoid duplicate
    # collision geometry (room_geometry.sdf already contains floor).
    # Floor remains accessible via scene.room_geometry.floor for
    # manipuland placement queries.

    # Get stage index for comparison (room stages exclude floor_plan).
    # ["furniture", "wall_mounted", "ceiling_mounted", "manipuland"]
    room_stages = PIPELINE_STAGES[1:]
    start_idx = room_stages.index(start_stage) if start_stage in room_stages else 0
    if start_idx == 0:
        load_and_materialize_locked_semantic_groups(scene, room_dir.parent)

    # Load projection config (needed for furniture and final post-processing).
    projection_cfg = cfg_dict["experiment"]["projection"]

    # Furniture stage.
    if start_idx <= 0:  # Run furniture if starting from furniture or earlier.
        with custom_span("furniture_placement"):
            console_logger.info("Adding furniture to scene")
            start_time = time.time()
            furniture_agent = BaseExperiment.build_furniture_agent(
                cfg_dict=cfg_dict,
                compatible_agents=(COMPATIBLE_FURNITURE_AGENTS),
                logger=logger,
                render_allocation=render_allocation,
            )
            try:
                asyncio.run(furniture_agent.add_furniture(scene=scene))
            finally:
                # Always cleanup server subprocesses.
                furniture_agent.cleanup()
            end_time = time.time()
            console_logger.info(
                f"Furniture added to room {room_id} in "
                f"{timedelta(seconds=end_time - start_time)}"
            )

        # Furniture post-processing (projection + simulation).
        if projection_cfg["enabled"] and projection_cfg["furniture"]["enabled"]:
            furniture_cfg = projection_cfg["furniture"]
            sim_cfg = projection_cfg["simulation"]

            # Log pre-projection state for debugging.
            logger.log_scene(scene=scene, name="furniture_only_pre_projection")

            console_logger.info(
                "Running furniture post-processing (projection + simulation)"
            )
            start_time = time.time()

            # Determine HTML output path for simulation.
            furniture_sim_html_path = None
            if sim_cfg.get("save_html", False):
                furniture_sim_html_path = (
                    logger.output_dir / "simulation" / "furniture_simulation.html"
                )

            # Get fallen furniture config from physics_validation.
            physics_val_cfg = cfg_dict["furniture_agent"]["physics_validation"]
            scene, projection_success, removed_ids = (
                apply_physical_feasibility_postprocessing(
                    scene=scene,
                    weld_furniture=False,
                    projection_enabled=True,
                    projection_influence_distance=furniture_cfg["influence_distance"],
                    projection_solver_name=furniture_cfg["solver_name"],
                    projection_iteration_limit=furniture_cfg["iteration_limit"],
                    projection_time_limit_s=furniture_cfg["time_limit_s"],
                    projection_xy_only=furniture_cfg["xy_only"],
                    projection_fix_rotation=furniture_cfg["fix_rotation"],
                    simulation_enabled=sim_cfg["enabled"],
                    simulation_time_s=sim_cfg["simulation_time_s"],
                    simulation_time_step_s=sim_cfg["time_step_s"],
                    simulation_timeout_s=sim_cfg["timeout_s"],
                    simulation_html_path=furniture_sim_html_path,
                    remove_fallen_furniture=physics_val_cfg["remove_fallen_furniture"],
                    fallen_tilt_threshold_degrees=physics_val_cfg[
                        "fallen_tilt_threshold_degrees"
                    ],
                    validation_object_penetration_threshold_m=physics_val_cfg[
                        "object_penetration_threshold_m"
                    ],
                    validation_floor_penetration_tolerance_m=physics_val_cfg[
                        "floor_penetration_tolerance_m"
                    ],
                )
            )
            end_time = time.time()
            if removed_ids:
                console_logger.info(
                    f"Removed {len(removed_ids)} fallen furniture item(s) during "
                    f"simulation: {removed_ids}"
                )
            if not projection_success:
                console_logger.error(
                    "Furniture projection failed; rejecting checkpoint"
                )
            else:
                console_logger.info(
                    f"Furniture post-processing completed for room {room_id} in "
                    f"{end_time - start_time:.2f} seconds"
                )
            _require_projection_success("furniture", projection_success)

        room_area_m2 = float(scene.room_geometry.width) * float(
            scene.room_geometry.length
        )
        checkpoint_room_kit = select_room_kit(
            scene.text_description, room_area_m2=room_area_m2
        )
        _validate_room_kit_completion(scene, checkpoint_room_kit)

        # Save only after physical and semantic publication gates pass.
        logger.log_scene(scene=scene, name="scene_after_furniture")
        _export_scene_blend_file(
            scene=scene,
            scene_dir=room_dir,
            cfg_dict=cfg_dict,
            name="scene_after_furniture",
        )
        console_logger.info("Saved furniture checkpoint (scene_after_furniture)")
    elif start_idx == 1:
        # Starting from wall_objects - load scene from saved furniture state.
        console_logger.info("Loading scene from saved furniture state for wall_objects")
        furniture_state_path = (
            room_dir / "scene_states" / "scene_after_furniture" / "scene_state.json"
        )
        if not furniture_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'wall_objects' stage: furniture state not found at "
                f"{furniture_state_path}. Run with start_stage='furniture' first."
            )
        with open(furniture_state_path) as f:
            furniture_state = json.load(f)
        scene.restore_from_state_dict(furniture_state)
        load_and_materialize_locked_semantic_groups(scene, room_dir.parent)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from furniture checkpoint"
        )

    # Check if we should stop after furniture stage.
    if stop_stage == "furniture":
        console_logger.info("Stopping after furniture stage as configured")
        return scene

    # Wall objects stage.
    if start_idx <= 1:  # Run wall_objects if starting from wall_objects or earlier.
        with custom_span("wall_object_placement"):
            console_logger.info("Adding wall-mounted objects to scene")
            start_time = time.time()

            # Load house_layout from parent directory (saved during floor plan stage).
            house_layout_path = room_dir.parent / "house_layout.json"
            if not house_layout_path.exists():
                raise FileNotFoundError(
                    f"Cannot run wall_objects stage: house_layout.json not found at "
                    f"{house_layout_path}. This should have been saved during floor "
                    f"plan generation."
                )
            with open(house_layout_path) as f:
                house_layout_dict = json.load(f)
            house_layout = HouseLayout.from_dict(
                house_layout_dict, house_dir=room_dir.parent
            )

            wall_agent = BaseExperiment.build_wall_agent(
                cfg_dict=cfg_dict,
                compatible_agents=COMPATIBLE_WALL_AGENTS,
                logger=logger,
                house_layout=house_layout,
                ceiling_height=room_geometry.wall_height,
                wall_thickness=room_geometry.wall_thickness,
                render_allocation=render_allocation,
            )
            try:
                asyncio.run(wall_agent.add_wall_objects(scene=scene))
            finally:
                # Always cleanup server subprocesses.
                wall_agent.cleanup()
            end_time = time.time()
            console_logger.info(
                f"Wall objects added to room {room_id} in "
                f"{timedelta(seconds=end_time - start_time)}"
            )

        # Always save state after wall_objects stage (unconditional for resumability).
        logger.log_scene(scene=scene, name="scene_after_wall_objects")
        _export_scene_blend_file(
            scene=scene,
            scene_dir=room_dir,
            cfg_dict=cfg_dict,
            name="scene_after_wall_objects",
        )
        console_logger.info("Saved wall_objects checkpoint (scene_after_wall_objects)")
    elif start_idx == 2:
        # Starting from ceiling_mounted - load scene from saved wall_objects state.
        console_logger.info("Loading scene from saved wall_objects state for ceiling")
        wall_objects_state_path = (
            room_dir / "scene_states" / "scene_after_wall_objects" / "scene_state.json"
        )
        if not wall_objects_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'ceiling_mounted' stage: wall_objects state not "
                f"found at {wall_objects_state_path}. Run with "
                f"start_stage='wall_mounted' first."
            )
        with open(wall_objects_state_path) as f:
            wall_objects_state = json.load(f)
        scene.restore_from_state_dict(wall_objects_state)
        load_and_materialize_locked_semantic_groups(scene, room_dir.parent)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from wall_objects checkpoint"
        )

    # Check if we should stop after wall_mounted stage.
    if stop_stage == AgentType.WALL_MOUNTED.value:
        console_logger.info("Stopping after wall_mounted stage as configured")
        return scene

    # Ceiling objects stage.
    if start_idx <= 2:  # Run ceiling if starting from ceiling or earlier.
        with custom_span("ceiling_object_placement"):
            console_logger.info("Adding ceiling-mounted objects to scene")
            start_time = time.time()

            ceiling_agent = BaseExperiment.build_ceiling_agent(
                cfg_dict=cfg_dict,
                compatible_agents=(COMPATIBLE_CEILING_AGENTS),
                logger=logger,
                ceiling_height=room_geometry.wall_height,
                render_allocation=render_allocation,
            )
            try:
                asyncio.run(ceiling_agent.add_ceiling_objects(scene=scene))
            finally:
                # Always cleanup server subprocesses.
                ceiling_agent.cleanup()
            end_time = time.time()
            console_logger.info(
                f"Ceiling objects added to room {room_id} in "
                f"{timedelta(seconds=end_time - start_time)}"
            )

        # Always save state after ceiling stage (unconditional for resumability).
        logger.log_scene(scene=scene, name="scene_after_ceiling_objects")
        _export_scene_blend_file(
            scene=scene,
            scene_dir=room_dir,
            cfg_dict=cfg_dict,
            name="scene_after_ceiling_objects",
        )
        console_logger.info(
            "Saved ceiling_objects checkpoint (scene_after_ceiling_objects)"
        )
    else:
        # Starting from manipulands - load scene from saved ceiling_objects state.
        console_logger.info("Loading scene from saved ceiling_objects state")
        ceiling_objects_state_path = (
            room_dir
            / "scene_states"
            / "scene_after_ceiling_objects"
            / "scene_state.json"
        )
        if not ceiling_objects_state_path.exists():
            raise FileNotFoundError(
                f"Cannot start from 'manipuland' stage: ceiling_objects state not "
                f"found at {ceiling_objects_state_path}. Run with "
                f"start_stage='ceiling_mounted' first."
            )
        with open(ceiling_objects_state_path) as f:
            ceiling_objects_state = json.load(f)
        scene.restore_from_state_dict(ceiling_objects_state)
        load_and_materialize_locked_semantic_groups(scene, room_dir.parent)
        console_logger.info(
            f"Loaded {len(scene.objects)} objects from ceiling_objects checkpoint"
        )

    # Check if we should stop after ceiling_mounted stage.
    if stop_stage == AgentType.CEILING_MOUNTED.value:
        console_logger.info("Stopping after ceiling_mounted stage as configured")
        return scene

    # Add manipulands.
    with custom_span("manipuland_placement"):
        console_logger.info("Adding manipulands to scene")
        start_time = time.time()
        manipuland_agent = BaseExperiment.build_manipuland_agent(
            cfg_dict=cfg_dict,
            compatible_agents=(COMPATIBLE_MANIPULAND_AGENTS),
            logger=logger,
            render_allocation=render_allocation,
        )
        try:
            asyncio.run(manipuland_agent.add_manipulands(scene=scene))
        finally:
            # The final semantic verifier runs after manipuland placement and
            # does not need the geometry helper processes. Closing them here
            # also prevents a later publication exception from orphaning
            # servers that keep the parent job pipe open.
            manipuland_agent.cleanup()
        end_time = time.time()
        console_logger.info(
            f"Manipulands added to room {room_id} in "
            f"{timedelta(seconds=end_time - start_time)}"
        )

    # Every preceding stage has already run its own physics gate. The final
    # projection/simulation result becomes the publication certificate's
    # independent physics boundary when enabled.
    final_physics_verified = True
    final_physics_evidence_refs = ("physics:all-stage-gates-passed",)

    # Final post-processing (projection + simulation).
    if projection_cfg["enabled"] and projection_cfg["final"]["enabled"]:
        final_cfg = projection_cfg["final"]
        sim_cfg = projection_cfg["simulation"]

        # Log pre-projection state for debugging.
        logger.log_scene(scene=scene, name="final_scene_pre_projection")

        console_logger.info("Running final post-processing (projection + simulation)")
        start_time = time.time()

        # Determine HTML output path for simulation.
        final_sim_html_path = None
        if sim_cfg.get("save_html", False):
            final_sim_html_path = (
                logger.output_dir / "simulation" / "final_simulation.html"
            )

        # Final post-processing: weld_furniture=True means only manipulands move.
        # Fallen furniture removal is not needed here (furniture is welded).
        # Get fallen manipuland config from manipuland_agent physics_validation.
        manipuland_physics_cfg = cfg_dict["manipuland_agent"]["physics_validation"]
        scene, projection_success, removed_ids = (
            apply_physical_feasibility_postprocessing(
                scene=scene,
                weld_furniture=True,
                projection_enabled=True,
                projection_influence_distance=final_cfg["influence_distance"],
                projection_solver_name=final_cfg["solver_name"],
                projection_iteration_limit=final_cfg["iteration_limit"],
                projection_time_limit_s=final_cfg["time_limit_s"],
                projection_xy_only=final_cfg["xy_only"],
                projection_fix_rotation=final_cfg["fix_rotation"],
                simulation_enabled=sim_cfg["enabled"],
                simulation_time_s=sim_cfg["simulation_time_s"],
                simulation_time_step_s=sim_cfg["time_step_s"],
                simulation_timeout_s=sim_cfg["timeout_s"],
                simulation_html_path=final_sim_html_path,
                remove_fallen_furniture=False,
                remove_fallen_manipulands=manipuland_physics_cfg[
                    "remove_fallen_manipulands"
                ],
                fallen_manipuland_floor_z=manipuland_physics_cfg[
                    "fallen_manipuland_floor_z"
                ],
                fallen_manipuland_near_floor_z=manipuland_physics_cfg[
                    "fallen_manipuland_near_floor_z"
                ],
                fallen_manipuland_z_displacement=manipuland_physics_cfg[
                    "fallen_manipuland_z_displacement"
                ],
            )
        )
        end_time = time.time()
        if removed_ids:
            console_logger.info(
                f"Removed {len(removed_ids)} fallen manipuland(s) during "
                f"final simulation: {removed_ids}"
            )
        if not projection_success:
            console_logger.error("Final projection failed, keeping original positions")
        else:
            console_logger.info(
                f"Final post-processing completed for room {room_id} in "
                f"{end_time - start_time:.2f} seconds"
            )
        final_physics_verified = bool(projection_success)
        final_physics_evidence_refs = (
            "physics:all-stage-gates-passed",
            "physics:final-projection-and-simulation",
        )

    _validate_final_dense_library_book_rows(scene, manipuland_agent)

    certify_room_publication(
        scene=scene,
        room_dir=room_dir,
        house_layout=house_layout,
        cfg_dict=cfg_dict,
        manipuland_agent=manipuland_agent,
        final_physics_verified=final_physics_verified,
        final_physics_evidence_refs=final_physics_evidence_refs,
    )

    # Log and export final scene.
    logger.log_scene(scene=scene, name="final_scene")
    _export_scene_blend_file(
        scene=scene, scene_dir=room_dir, cfg_dict=cfg_dict, name="final_scene"
    )

    # Export to SceneEval format if enabled.
    sceneeval_cfg = cfg_dict["experiment"]["sceneeval_export"]
    if sceneeval_cfg["enabled"]:
        export_config = SceneEvalExportConfig(
            asset_id_prefix=sceneeval_cfg["asset_id_prefix"]
        )
        exporter = SceneEvalExporter(
            scene=scene,
            scene_dir=room_dir,
            config=export_config,
            house_layout=house_layout,
        )
        exporter.export()

    console_logger.info(
        f"Room {room_id} generation completed successfully in "
        f"{timedelta(seconds=time.time() - room_start_time)}"
    )

    return scene
