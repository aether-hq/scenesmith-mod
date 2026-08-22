"""Physical feasibility post-processing for scene collision resolution.

This module provides two-stage post-processing adapted from scene_gen repository:
1. Projection - IK-based collision resolution with configurable DOF constraints
2. Simulation - Physics settling to static equilibrium (always full 6DOF)

See: https://github.com/nepfaff/steerable-scene-generation/blob/main/steerable_scene_generation/algorithms/scene_diffusion/postprocessing.py
"""

import logging
import time

from pathlib import Path

from pydrake.all import (
    Context,
    DiagramBuilder,
    EventStatus,
    MeshcatVisualizer,
    Simulator,
    StartMeshcat,
)

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.feasibility.ik import (
    _create_drake_plant_for_ik,
    _update_scene_from_plant,
    compute_tilt_angle_degrees,
)


def apply_forward_simulation(
    scene: RoomScene,
    simulation_time_s: float = 5.0,
    time_step_s: float = 1e-3,
    timeout_s: float = 300.0,
    weld_furniture: bool = True,
    output_html_path: Path | None = None,
    remove_fallen_furniture: bool = False,
    fallen_tilt_threshold_degrees: float = 45.0,
    remove_fallen_manipulands: bool = False,
    fallen_manipuland_floor_z: float = -0.5,
    fallen_manipuland_near_floor_z: float = 0.02,
    fallen_manipuland_z_displacement: float = 0.3,
) -> tuple[RoomScene, list[UniqueID]]:
    """Apply forward simulation to settle scene to static equilibrium.

    This is Stage 2 of physical feasibility post-processing. Objects settle
    under gravity after projection resolves penetrations.

    Always allows full 6DOF motion (xyz + roll/pitch/yaw) for free bodies.

    Args:
        scene: RoomScene to simulate.
        simulation_time_s: Simulation duration in seconds.
        time_step_s: Physics time step in seconds.
        timeout_s: Maximum wall-clock time for simulation.
        weld_furniture: If True, weld furniture (simulate manipulands only).
                        If False, all objects are simulated.
        output_html_path: If provided, save meshcat visualization to this HTML file.
        remove_fallen_furniture: If True, remove furniture objects that fall over
            (tilt beyond threshold) during simulation.
        fallen_tilt_threshold_degrees: Tilt angle threshold in degrees to consider
            furniture "fallen" (only used if remove_fallen_furniture is True).
        remove_fallen_manipulands: If True, remove manipuland objects that fall
            off furniture surfaces during simulation.
        fallen_manipuland_floor_z: Absolute Z threshold for floor penetration.
            Manipulands below this Z are removed (physics bug).
        fallen_manipuland_near_floor_z: Object bottom below this Z is considered
            on floor (used with z_displacement check).
        fallen_manipuland_z_displacement: Z drop threshold. Manipulands that drop
            more than this AND end up on floor are removed.

    Returns:
        Tuple of (scene, removed_ids) where:
        - scene: Scene with updated poses after simulation
        - removed_ids: List of UniqueIDs of fallen objects (furniture and manipulands)
    """
    start_time = time.time()
    console_logger.info(
        f"Starting forward simulation (weld_furniture={weld_furniture}, "
        f"time={simulation_time_s}s)"
    )

    meshcat = None
    visualizer = None

    try:
        # Create Drake plant for simulation.
        builder = DiagramBuilder()
        plant, scene_graph, object_indices, composite_info = _create_drake_plant_for_ik(
            scene=scene,
            builder=builder,
            weld_furniture=weld_furniture,
            time_step=time_step_s,
        )

        if not object_indices:
            console_logger.warning("No free bodies for simulation. Skipping.")
            return scene, []

        # Store pre-simulation Z positions for fallen manipuland detection.
        pre_sim_z: dict[UniqueID, float] = {}
        if remove_fallen_manipulands:
            for obj in scene.objects.values():
                if obj.object_type == ObjectType.MANIPULAND:
                    pre_sim_z[obj.object_id] = obj.transform.translation()[2]

        # Set up visualization if HTML output is requested.
        if output_html_path is not None:
            meshcat = StartMeshcat()
            visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

        # Build diagram.
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        # Set up timeout monitor.
        sim_start = time.time()

        def timeout_monitor(_: Context) -> EventStatus:
            if time.time() - sim_start > timeout_s:
                return EventStatus.ReachedTermination(None, "timeout")
            return EventStatus.DidNothing()

        # Run simulation.
        simulator = Simulator(diagram, context)
        simulator.set_monitor(timeout_monitor)

        # Start recording if visualizing.
        if visualizer is not None:
            visualizer.StartRecording()

        simulator.AdvanceTo(simulation_time_s)

        # Stop recording and export HTML if visualizing.
        if visualizer is not None and meshcat is not None:
            visualizer.StopRecording()
            visualizer.PublishRecording()

            html = meshcat.StaticHtml()
            output_html_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_html_path, "w") as f:
                f.write(html)
            console_logger.info(f"Saved simulation HTML to {output_html_path}")

        # Update scene poses from plant.
        _update_scene_from_plant(
            scene=scene,
            plant=plant,
            plant_context=plant_context,
            object_indices=object_indices,
            composite_info=composite_info,
            operation_name="Simulation",
        )

        # Detect and remove fallen furniture if enabled.
        removed_ids: list[UniqueID] = []
        if remove_fallen_furniture:
            # Only check furniture objects (not manipulands, walls, etc.).
            furniture_ids = [
                obj.object_id
                for obj in scene.objects.values()
                if obj.object_type == ObjectType.FURNITURE
            ]
            for obj_id in furniture_ids:
                obj = scene.get_object(obj_id)
                if obj is None:
                    continue
                tilt_angle = compute_tilt_angle_degrees(obj.transform)
                if tilt_angle > fallen_tilt_threshold_degrees:
                    console_logger.warning(
                        f"Removing fallen furniture {obj_id}: "
                        f"tilt={tilt_angle:.1f}° > threshold={fallen_tilt_threshold_degrees}°"
                    )
                    scene.remove_object(obj_id)
                    removed_ids.append(obj_id)

            if removed_ids:
                console_logger.info(
                    f"Removed {len(removed_ids)} fallen furniture item(s): {removed_ids}"
                )

        # Detect and remove fallen manipulands if enabled.
        if remove_fallen_manipulands:
            furniture_removed_count = len(removed_ids)
            manipuland_ids = [
                obj.object_id
                for obj in scene.objects.values()
                if obj.object_type == ObjectType.MANIPULAND
            ]
            for obj_id in manipuland_ids:
                obj = scene.get_object(obj_id)
                if obj is None:
                    continue

                current_z = obj.transform.translation()[2]

                # Check 1: Floor penetration (physics bug).
                if current_z < fallen_manipuland_floor_z:
                    console_logger.warning(
                        f"Removing fallen manipuland {obj_id}: "
                        f"z={current_z:.4f}m < floor_z={fallen_manipuland_floor_z}m"
                    )
                    scene.remove_object(obj_id)
                    removed_ids.append(obj_id)
                    continue

                # Check 2: On floor + significant Z drop (fell off furniture).
                # Use world-frame bounds (handles rotation).
                world_bounds = obj.compute_world_bounds()
                if world_bounds is None:
                    continue
                world_bbox_min, _ = world_bounds
                bottom_z = world_bbox_min[2]
                is_on_floor = bottom_z < fallen_manipuland_near_floor_z

                if obj_id in pre_sim_z and is_on_floor:
                    z_delta = current_z - pre_sim_z[obj_id]
                    if z_delta < -fallen_manipuland_z_displacement:
                        console_logger.warning(
                            f"Removing fallen manipuland {obj_id}: "
                            f"bottom_z={bottom_z:.4f}m, z_delta={z_delta:.4f}m"
                        )
                        scene.remove_object(obj_id)
                        removed_ids.append(obj_id)

            fallen_manipuland_count = len(removed_ids) - furniture_removed_count
            if fallen_manipuland_count > 0:
                console_logger.info(
                    f"Removed {fallen_manipuland_count} fallen manipuland(s)"
                )

        elapsed = time.time() - start_time
        console_logger.info(f"Simulation completed in {elapsed:.2f}s")

        return scene, removed_ids

    except Exception as e:
        console_logger.error(f"Simulation failed with exception: {e}")
        return scene, []

    finally:
        # Explicitly delete Meshcat on the main thread to avoid threading issues.
        # Drake's Meshcat destructor asserts it must be called from the thread
        # that created it. Without explicit deletion, Python's GC might destroy
        # the Meshcat from a ThreadPoolExecutor worker thread, causing a crash.
        if meshcat is not None:
            del meshcat
