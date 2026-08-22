"""Fill container utilities using physics simulation.

This module provides functionality for:
- Computing container interior bounds using top rim heuristic.
- Computing fill object spawn transforms.
- Resolving initial fill object collisions using NLP projection.
- Simulating fill objects dropping into containers using Drake physics.
"""

import logging
import tempfile

from pathlib import Path

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    ProcessModelDirectives,
    RigidTransform,
    Simulator,
    StartMeshcat,
)

from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject
from scenesmith.utils.geometry.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)

from scenesmith.manipuland_agents.tools.fill.bounds import FillSimulationResult


def simulate_fill_physics(
    container_scene_object: SceneObject,
    container_transform: RigidTransform,
    new_fill_objects: list[SceneObject],
    new_fill_transforms: list[RigidTransform],
    settled_fill_objects: list[SceneObject] | None = None,
    settled_fill_transforms: list[RigidTransform] | None = None,
    catch_floor_z: float = -5.0,
    inside_z_threshold: float = -2.0,
    simulation_time: float = 5.0,
    simulation_time_step: float = 0.001,
    output_html_path: Path | None = None,
) -> FillSimulationResult:
    """Simulate fill objects dropping into a container.

    Creates a Drake simulation with:
    - Container welded in the air at an elevated position.
    - Previously settled fill objects welded at their positions.
    - New fill objects as free bodies spawned above container.
    - Catch floor below to detect objects that fell out.

    Args:
        container_scene_object: Container SceneObject (must have sdf_path).
        container_transform: World transform for container.
        new_fill_objects: List of NEW fill SceneObjects to simulate (free bodies).
        new_fill_transforms: Initial transforms for new fill objects.
        settled_fill_objects: List of previously settled SceneObjects (welded).
        settled_fill_transforms: Transforms for previously settled fill objects.
        catch_floor_z: Z position of catch floor.
        inside_z_threshold: Z threshold for inside/outside classification.
        simulation_time: Duration to simulate.
        simulation_time_step: Simulation time step.
        output_html_path: If provided, record simulation as HTML.

    Returns:
        FillSimulationResult with inside/outside classification and final transforms
        for the NEW fill objects only.
    """
    if len(new_fill_objects) != len(new_fill_transforms):
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message="Mismatch between new fill objects and transforms count",
        )

    # Validate settled lists are consistent.
    if settled_fill_objects is None:
        settled_fill_objects = []
    if settled_fill_transforms is None:
        settled_fill_transforms = []
    if len(settled_fill_objects) != len(settled_fill_transforms):
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message="Mismatch between settled fill objects and transforms count",
        )

    # Validate container has SDF.
    if (
        not container_scene_object.sdf_path
        or not container_scene_object.sdf_path.exists()
    ):
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message="Container has no SDF path",
        )

    try:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=simulation_time_step
        )

        # Set up visualization if recording.
        meshcat = None
        visualizer = None
        if output_html_path is not None:
            meshcat = StartMeshcat()
            console_logger.info(f"Meshcat URL: {meshcat.web_url()}")

        # Create catch floor SDF.
        catch_floor_sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="catch_floor">
    <static>true</static>
    <pose>0 0 {catch_floor_z} 0 0 0</pose>
    <link name="catch_link">
      <collision name="catch_collision">
        <geometry>
          <box><size>20 20 0.1</size></box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False
        ) as catch_file:
            catch_file.write(catch_floor_sdf)
            catch_floor_path = catch_file.name

        # Build directive.
        directive_parts = ["directives:"]

        # Add catch floor.
        directive_parts.append(
            f"""
- add_model:
    name: catch_floor
    file: file://{catch_floor_path}"""
        )

        # Add container (welded).
        container_translation = container_transform.translation()
        container_angle_axis = container_transform.rotation().ToAngleAxis()
        container_angle_deg = container_angle_axis.angle() * 180 / np.pi
        container_axis = container_angle_axis.axis()

        try:
            container_base_link = extract_base_link_name_from_sdf(
                container_scene_object.sdf_path
            )
        except ValueError:
            container_base_link = "base_link"

        directive_parts.append(
            f"""
- add_model:
    name: container
    file: file://{container_scene_object.sdf_path.absolute()}
- add_weld:
    parent: world
    child: container::{container_base_link}
    X_PC:
      translation: [{container_translation[0]}, {container_translation[1]}, {container_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {container_angle_deg}
        axis: [{container_axis[0]}, {container_axis[1]}, {container_axis[2]}]"""
        )

        # Add settled fill objects as free bodies (not welded) so new objects can
        # push them realistically. They start at their settled positions.
        settled_model_names: list[str] = []
        for i, (obj, transform) in enumerate(
            zip(settled_fill_objects, settled_fill_transforms)
        ):
            if not obj.sdf_path or not obj.sdf_path.exists():
                continue

            translation = transform.translation()
            angle_axis = transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            try:
                base_link = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError:
                base_link = "base_link"

            model_name = f"settled_fill_{i}"
            settled_model_names.append(model_name)
            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      {base_link}:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        if settled_model_names:
            console_logger.info(
                f"Added {len(settled_model_names)} settled fill objects (free)"
            )

        # Add new fill objects as free bodies.
        free_model_names = []
        for i, (obj, transform) in enumerate(
            zip(new_fill_objects, new_fill_transforms)
        ):
            if not obj.sdf_path or not obj.sdf_path.exists():
                continue

            model_name = f"fill_obj_{i}"
            free_model_names.append((i, model_name))

            translation = transform.translation()
            angle_axis = transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            try:
                base_link = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError:
                base_link = "base_link"

            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      {base_link}:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        directive_yaml = "\n".join(directive_parts)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as directive_file:
            directive_file.write(directive_yaml)
            directive_path = directive_file.name

        try:
            # Load directives.
            directives = LoadModelDirectives(str(directive_path))
            ProcessModelDirectives(directives, plant, parser=None)
            plant.Finalize()

            # Add visualizer after finalize.
            if meshcat is not None:
                visualizer = MeshcatVisualizer.AddToBuilder(
                    builder=builder, scene_graph=scene_graph, meshcat=meshcat
                )

            # Build and simulate.
            diagram = builder.Build()
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()

            if visualizer is not None:
                visualizer.StartRecording()

            simulator.AdvanceTo(simulation_time)

            if visualizer is not None and meshcat is not None:
                visualizer.StopRecording()
                visualizer.PublishRecording()
                html = meshcat.StaticHtml()
                output_html_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_html_path, "w") as f:
                    f.write(html)
                console_logger.info(f"Saved fill simulation HTML to {output_html_path}")

            # Get final positions and classify.
            plant_context = plant.GetMyContextFromRoot(context)
            final_transforms = []
            inside_indices = []
            outside_indices = []

            for i, (obj_idx, model_name) in enumerate(free_model_names):
                model_instance = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_instance)

                if body_indices:
                    body = plant.get_body(body_indices[0])
                    final_pose = plant.EvalBodyPoseInWorld(plant_context, body)
                    final_transforms.append(final_pose)

                    # Classify by Z position.
                    final_z = final_pose.translation()[2]
                    if final_z > inside_z_threshold:
                        inside_indices.append(obj_idx)
                        console_logger.debug(
                            f"Fill object {obj_idx} INSIDE: z={final_z:.3f}"
                        )
                    else:
                        outside_indices.append(obj_idx)
                        console_logger.debug(
                            f"Fill object {obj_idx} OUTSIDE: z={final_z:.3f}"
                        )
                else:
                    final_transforms.append(new_fill_transforms[i])
                    outside_indices.append(obj_idx)

            console_logger.info(
                f"Fill simulation: {len(inside_indices)} inside, "
                f"{len(outside_indices)} outside"
            )

            # Extract updated transforms for settled objects and check if any fell out.
            settled_final_transforms: list[RigidTransform] | None = None
            settled_fell_out_indices: list[int] | None = None
            if settled_model_names:
                settled_final_transforms = []
                settled_fell_out_indices = []
                for i, model_name in enumerate(settled_model_names):
                    model_instance = plant.GetModelInstanceByName(model_name)
                    body_indices = plant.GetBodyIndices(model_instance)
                    if body_indices:
                        body = plant.get_body(body_indices[0])
                        final_pose = plant.EvalBodyPoseInWorld(plant_context, body)
                        settled_final_transforms.append(final_pose)
                        # Check if this settled object was pushed out.
                        final_z = final_pose.translation()[2]
                        if final_z <= inside_z_threshold:
                            settled_fell_out_indices.append(i)
                            console_logger.warning(
                                f"Settled object {i} was pushed out: z={final_z:.3f}"
                            )

            return FillSimulationResult(
                inside_indices=inside_indices,
                outside_indices=outside_indices,
                final_transforms=final_transforms,
                settled_final_transforms=settled_final_transforms,
                settled_fell_out_indices=settled_fell_out_indices,
            )

        finally:
            Path(directive_path).unlink(missing_ok=True)
            Path(catch_floor_path).unlink(missing_ok=True)
            if meshcat is not None:
                del meshcat

    except Exception as e:
        console_logger.error(f"Fill simulation failed: {e}")
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message=str(e),
        )
