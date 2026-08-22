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
    ProcessModelDirectives,
    RigidTransform,
)

from scenesmith.agent_utils.physics.feasibility.ik import solve_non_penetration_ik
from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject
from scenesmith.utils.geometry.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if a 2D point is inside a convex polygon using cross product.

    Args:
        point: 2D point [x, y].
        polygon: Nx2 array of polygon vertices (ordered).

    Returns:
        True if point is inside polygon.
    """
    n = len(polygon)
    sign = None

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Compute cross product of (p2-p1) and (point-p1).
        d = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

        if sign is None:
            sign = d >= 0
        elif (d >= 0) != sign:
            return False

    return True


def project_fill_objects_non_penetrating(
    fill_scene_objects: list[SceneObject],
    fill_initial_transforms: list[RigidTransform],
    influence_distance: float = 0.02,
    solver_name: str = "snopt",
    iteration_limit: int = 1000,
    time_limit_s: float = 30.0,
) -> tuple[list[RigidTransform], bool]:
    """Resolve penetrations between fill objects using IK projection.

    Creates a temporary Drake plant with only the fill objects and uses
    shared IK projection utility to push apart any overlapping objects.
    This prevents explosive contact forces when physics simulation starts.

    Uses the same projection logic as scene-level non-penetration projection,
    but with fix_rotation=False and fix_z=False to allow full 3D movement
    (objects will fall and rotate during physics simulation anyway).

    Args:
        fill_scene_objects: List of fill SceneObjects (must have sdf_path).
        fill_initial_transforms: Initial transforms for each fill object.
        influence_distance: Distance threshold for collision influence.
        solver_name: NLP solver name ("snopt" or "ipopt").
        iteration_limit: Maximum solver iterations.
        time_limit_s: Maximum solver time in seconds.

    Returns:
        Tuple of (projected_transforms, success_flag).
        On failure: returns (original_transforms, False).
    """
    if not fill_scene_objects:
        return fill_initial_transforms, True

    console_logger.info(
        f"Starting NLP projection for {len(fill_scene_objects)} fill objects"
    )

    try:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        # Build directive for fill objects as free bodies.
        directive_parts = ["directives:"]
        model_names = []
        for i, (obj, transform) in enumerate(
            zip(fill_scene_objects, fill_initial_transforms)
        ):
            if not obj.sdf_path or not obj.sdf_path.exists():
                continue

            model_name = f"fill_obj_{i}"
            model_names.append(model_name)

            translation = transform.translation()
            angle_axis = transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            # Extract base link name.
            try:
                base_link_name = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError:
                base_link_name = "base_link"

            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      {base_link_name}:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        if not model_names:
            console_logger.warning("No valid SDF paths for fill objects")
            return fill_initial_transforms, False

        directive_yaml = "\n".join(directive_parts)

        # Write directive to temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as directive_file:
            directive_file.write(directive_yaml)
            directive_path = directive_file.name

        try:
            # Load directives into plant.
            directives = LoadModelDirectives(str(directive_path))
            ProcessModelDirectives(directives, plant, parser=None)
            plant.Finalize()

            # Use shared IK projection utility.
            # fix_rotation=False, fix_z=False: allow full 3D movement since
            # fill objects will fall and rotate during physics simulation anyway.
            plant_context, success = solve_non_penetration_ik(
                builder=builder,
                plant=plant,
                scene_graph=scene_graph,
                influence_distance=influence_distance,
                fix_rotation=False,
                fix_z=False,
                solver_name=solver_name,
                iteration_limit=iteration_limit,
                time_limit_s=time_limit_s,
            )

            if not success or plant_context is None:
                console_logger.warning(
                    "NLP projection failed. Proceeding with original positions."
                )
                return fill_initial_transforms, False

            # Extract projected transforms.
            projected_transforms = []
            for i, model_name in enumerate(model_names):
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)

                if body_indices:
                    body = plant.get_body(body_indices[0])
                    pose = plant.EvalBodyPoseInWorld(plant_context, body)
                    projected_transforms.append(pose)
                else:
                    projected_transforms.append(fill_initial_transforms[i])

            console_logger.info(
                f"NLP projection completed successfully for {len(projected_transforms)} "
                "fill objects"
            )
            return projected_transforms, True

        finally:
            Path(directive_path).unlink(missing_ok=True)

    except Exception as e:
        console_logger.error(f"NLP projection failed with exception: {e}")
        return fill_initial_transforms, False
