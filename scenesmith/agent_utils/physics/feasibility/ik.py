"""Physical feasibility post-processing for scene collision resolution.

This module provides two-stage post-processing adapted from scene_gen repository:
1. Projection - IK-based collision resolution with configurable DOF constraints
2. Simulation - Physics settling to static equilibrium (always full 6DOF)

See: https://github.com/nepfaff/steerable-scene-generation/blob/main/steerable_scene_generation/algorithms/scene_diffusion/postprocessing.py
"""

import logging
import signal
import tempfile

from pathlib import Path

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BodyIndex,
    Context,
    DiagramBuilder,
    DiscreteContactApproximation,
    InverseKinematics,
    IpoptSolver,
    LoadModelDirectives,
    ModelInstanceIndex,
    MultibodyPlant,
    ProcessModelDirectives,
    Quaternion,
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    SnoptSolver,
    SolverOptions,
)
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet, Role
from pydrake.geometry.optimization import HPolyhedron

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID
from scenesmith.utils.geometry.sdf_utils import (
    deserialize_rigid_transform,
    serialize_rigid_transform,
)

console_logger = logging.getLogger(__name__)


def _find_surface_owner(
    scene: RoomScene, surface_id: UniqueID
) -> tuple[UniqueID | None, bool]:
    """Find the furniture or floor that owns a support surface.

    Args:
        scene: RoomScene containing objects.
        surface_id: ID of the support surface to find owner for.

    Returns:
        Tuple of (owner_id, is_floor) where:
        - owner_id: UniqueID of the owning furniture/floor, or None if not found
        - is_floor: True if owner is a floor object, False otherwise
    """
    for obj in scene.objects.values():
        if obj.object_type in (ObjectType.FURNITURE, ObjectType.FLOOR):
            for surface in obj.support_surfaces:
                if surface.surface_id == surface_id:
                    is_floor = obj.object_type == ObjectType.FLOOR
                    return obj.object_id, is_floor
    return None, False


def compute_tilt_angle_degrees(transform: RigidTransform) -> float:
    """Compute tilt angle (deviation from upright) in degrees.

    Measures how much the object's local up-vector (Z-axis) deviates from the
    world up-vector. This captures combined roll+pitch rotation while ignoring
    yaw (turning in place).

    Args:
        transform: Object's world-frame pose.

    Returns:
        Tilt angle in degrees (0 = perfectly upright, 90 = horizontal, 180 = inverted).
    """
    # Get object's local Z-axis (up) in world frame.
    object_up = transform.rotation().matrix() @ np.array([0.0, 0.0, 1.0])
    world_up = np.array([0.0, 0.0, 1.0])

    # Compute angle between vectors.
    cos_tilt = np.clip(np.dot(object_up, world_up), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))


def _create_drake_plant_for_ik(
    scene: RoomScene,
    builder: DiagramBuilder,
    weld_furniture: bool = False,
    time_step: float = 0.0,
    free_objects: list[UniqueID] | None = None,
    include_objects: list[UniqueID] | None = None,
) -> tuple[
    MultibodyPlant,
    SceneGraph,
    dict[UniqueID, tuple[ModelInstanceIndex, BodyIndex]],
    dict[UniqueID, dict],
]:
    """Create Drake plant configured for IK optimization.

    Args:
        scene: RoomScene to load into the plant.
        builder: DiagramBuilder to use.
        weld_furniture: If True, weld furniture (only manipulands are free).
                        If False, all objects are free bodies.
        time_step: Physics time step (0.0 for kinematics-only).
        free_objects: If provided, only these specific objects will be free
            bodies. Overrides weld_furniture for these objects.
        include_objects: If provided, only include these objects in the Drake
            plant. Objects not in this list are excluded entirely (not just
            welded). Useful for performance optimization when only a subset of
            objects are relevant for collision checking.

    Returns:
        Tuple of (plant, scene_graph, object_indices, composite_info) where:
        - object_indices maps object ID to (model_instance_index, body_index)
        - composite_info maps composite ID to original transforms for delta computation
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    if time_step > 0.0:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)

    # Generate Drake directive with composite members welded for rigid unit behavior.
    directive_yaml = scene.to_drake_directive(
        include_objects=include_objects,
        weld_furniture=weld_furniture,
        free_objects=free_objects,
        exclude_room_geometry=False,
        weld_stack_members=True,
    )

    # Write directive to temporary file and load it.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(directive_yaml)
        temp_directive_path = f.name

    try:
        directives = LoadModelDirectives(temp_directive_path)
        ProcessModelDirectives(directives, plant, parser=None)
    finally:
        Path(temp_directive_path).unlink(missing_ok=True)

    plant.Finalize()

    # Build mapping from object ID to Drake indices.
    # Model names follow pattern: {name}_{id_suffix} from to_drake_directive().
    object_indices: dict[UniqueID, tuple[ModelInstanceIndex, BodyIndex]] = {}
    composite_info: dict[UniqueID, dict] = {}

    for obj in scene.objects.values():
        # Handle composite objects (stacks) by tracking bottom member.
        if obj.metadata.get("composite_type") == "stack":
            member_assets = obj.metadata.get("member_assets", [])
            if not member_assets:
                continue

            # Reconstruct bottom member model name (same pattern as to_drake_directive).
            bottom_member = member_assets[0]
            member_name = bottom_member.get("name", "stack_member")
            member_id = bottom_member.get("asset_id", "unknown")
            id_suffix = member_id.split("_")[-1][:8]
            stack_suffix = str(obj.object_id).split("_")[-1][:4]
            model_name = (
                f"{member_name.lower().replace(' ', '_')}_{id_suffix}_s{stack_suffix}_0"
            )

            try:
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)
                if len(body_indices) > 0:
                    body_idx = body_indices[0]
                    object_indices[obj.object_id] = (model_idx, body_idx)

                    # Store original bottom transform for delta computation.
                    composite_info[obj.object_id] = {
                        "original_bottom_transform": deserialize_rigid_transform(
                            bottom_member.get("transform", {})
                        ),
                        "member_assets": member_assets,
                    }
            except RuntimeError:
                console_logger.warning(f"Model {model_name} not found in plant")
            continue

        # Handle filled containers by tracking container as reference member.
        if obj.metadata.get("composite_type") == "filled_container":
            container_asset = obj.metadata.get("container_asset")
            fill_assets = obj.metadata.get("fill_assets", [])
            if not container_asset:
                continue

            # Reconstruct container model name (same pattern as to_drake_directive).
            container_name = container_asset.get("name", "container")
            container_id = container_asset.get("asset_id", "unknown")
            id_suffix = container_id.split("_")[-1][:8]
            fill_suffix = str(obj.object_id).split("_")[-1][:4]
            model_name = f"{container_name.lower().replace(' ', '_')}_{id_suffix}_f{fill_suffix}_c"

            try:
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)
                if len(body_indices) > 0:
                    body_idx = body_indices[0]
                    object_indices[obj.object_id] = (model_idx, body_idx)

                    # Store original container transform and assets for delta computation.
                    composite_info[obj.object_id] = {
                        "original_bottom_transform": deserialize_rigid_transform(
                            container_asset.get("transform", {})
                        ),
                        "container_asset": container_asset,
                        "fill_assets": fill_assets,
                        "composite_type": "filled_container",
                    }
            except RuntimeError:
                console_logger.warning(f"Model {model_name} not found in plant")
            continue

        # Handle piles by tracking first member (same structure as stack).
        if obj.metadata.get("composite_type") == "pile":
            member_assets = obj.metadata.get("member_assets", [])
            if not member_assets:
                continue

            # Reconstruct first member model name (same pattern as to_drake_directive).
            first_member = member_assets[0]
            member_name = first_member.get("name", "pile_member")
            member_id = first_member.get("asset_id", "unknown")
            id_suffix = member_id.split("_")[-1][:8]
            pile_suffix = str(obj.object_id).split("_")[-1][:4]
            model_name = (
                f"{member_name.lower().replace(' ', '_')}_{id_suffix}_p{pile_suffix}_0"
            )

            try:
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)
                if len(body_indices) > 0:
                    body_idx = body_indices[0]
                    object_indices[obj.object_id] = (model_idx, body_idx)

                    # Store original first member transform for delta computation.
                    composite_info[obj.object_id] = {
                        "original_bottom_transform": deserialize_rigid_transform(
                            first_member.get("transform", {})
                        ),
                        "member_assets": member_assets,
                        "composite_type": "pile",
                    }
            except RuntimeError:
                console_logger.warning(f"Model {model_name} not found in plant")
            continue

        if obj.sdf_path is None:
            continue

        # Reconstruct model name as done in to_drake_directive().
        id_suffix = str(obj.object_id).split("_")[-1][:8]
        model_name = f"{obj.name.lower().replace(' ', '_')}_{id_suffix}"

        try:
            model_idx = plant.GetModelInstanceByName(model_name)
            body_indices = plant.GetBodyIndices(model_idx)
            if len(body_indices) > 0:
                body_idx = body_indices[0]
                object_indices[obj.object_id] = (model_idx, body_idx)
        except RuntimeError:
            console_logger.warning(f"Model {model_name} not found in plant")

    return plant, scene_graph, object_indices, composite_info


def _update_scene_from_plant(
    scene: RoomScene,
    plant: MultibodyPlant,
    plant_context: Context,
    object_indices: dict[UniqueID, tuple[ModelInstanceIndex, BodyIndex]],
    composite_info: dict[UniqueID, dict] | None = None,
    operation_name: str = "Projection",
) -> None:
    """Update Scene object poses from Drake plant context.

    Args:
        scene: RoomScene to update (modified in place).
        plant: Drake MultibodyPlant.
        plant_context: Plant context with current poses.
        object_indices: Mapping from object ID to Drake indices.
        composite_info: Optional mapping from composite ID to original transforms.
            Used to update all composite member transforms based on reference member delta.
            Handles stacks, filled_containers, and piles.
        operation_name: Name of the operation for logging (e.g., "Projection" or
            "Simulation").
    """
    for obj_id, (model_idx, body_idx) in object_indices.items():
        obj = scene.get_object(obj_id)
        if obj is None:
            continue

        body = plant.get_body(body_idx)
        if not body.is_floating():
            # Welded body - pose is fixed.
            continue

        # Get updated pose from plant.
        positions = plant.GetPositions(plant_context, model_idx)
        if len(positions) < 7:
            # Not a floating body (quaternion + translation = 7 DOF).
            continue

        # Drake uses [qw, qx, qy, qz, x, y, z] for floating bodies.
        quaternion = positions[:4]
        translation = positions[4:7]

        # Normalize quaternion to handle numerical drift.
        quaternion = quaternion / np.linalg.norm(quaternion)

        # Create new RigidTransform.
        rotation = RotationMatrix(Quaternion(wxyz=quaternion))
        new_transform = RigidTransform(rotation, translation)

        # Log pose change if significant.
        old_translation = obj.transform.translation()
        delta_translation = new_transform.translation() - old_translation
        translation_change = np.linalg.norm(delta_translation)

        # Compute rotation change as angle between old and new orientations.
        old_rotation = obj.transform.rotation()
        delta_rotation = new_transform.rotation().multiply(old_rotation.inverse())
        rotation_angle_deg = np.degrees(delta_rotation.ToAngleAxis().angle())

        if (
            translation_change > 0.001 or rotation_angle_deg > 0.1
        ):  # 1mm or 0.1° threshold.
            console_logger.info(
                f"{operation_name} moved {obj.object_id}: "
                f"delta=({delta_translation[0]:.4f}, {delta_translation[1]:.4f}, "
                f"{delta_translation[2]:.4f}), rot={rotation_angle_deg:.2f}°"
            )

        # Update object transform.
        obj.transform = new_transform

    # Update composite member transforms based on reference member delta.
    if composite_info:
        for composite_id, info in composite_info.items():
            if composite_id not in object_indices:
                continue

            model_idx, body_idx = object_indices[composite_id]
            body = plant.get_body(body_idx)

            if not body.is_floating():
                continue

            # Get new bottom pose from plant.
            positions = plant.GetPositions(plant_context, model_idx)
            if len(positions) < 7:
                continue

            quaternion = positions[:4]
            translation = positions[4:7]
            quaternion = quaternion / np.linalg.norm(quaternion)
            rotation = RotationMatrix(Quaternion(wxyz=quaternion))
            new_bottom_transform = RigidTransform(rotation, translation)

            # Compute delta transform from original bottom position.
            orig_bottom = info["original_bottom_transform"]
            t_delta = new_bottom_transform @ orig_bottom.inverse()

            # Log pose change if significant.
            delta_translation = (
                new_bottom_transform.translation() - orig_bottom.translation()
            )
            translation_change = np.linalg.norm(delta_translation)

            # Compute rotation change for composite.
            delta_rotation = new_bottom_transform.rotation().multiply(
                orig_bottom.rotation().inverse()
            )
            rotation_angle_deg = np.degrees(delta_rotation.ToAngleAxis().angle())

            if (
                translation_change > 0.001 or rotation_angle_deg > 0.1
            ):  # 1mm or 0.1° threshold.
                console_logger.info(
                    f"{operation_name} moved composite {composite_id} reference member: "
                    f"delta=({delta_translation[0]:.4f}, {delta_translation[1]:.4f}, "
                    f"{delta_translation[2]:.4f}), rot={rotation_angle_deg:.2f}°"
                )

            # Apply delta to all member transforms.
            composite_obj = scene.get_object(composite_id)
            if composite_obj is None:
                continue

            # Check composite type to determine how to update transforms.
            composite_type = info.get("composite_type")

            if composite_type == "filled_container":
                # Filled container: update container_asset + fill_assets.
                container_asset = info.get("container_asset")
                if container_asset:
                    old_transform = deserialize_rigid_transform(
                        container_asset["transform"]
                    )
                    new_container_transform = t_delta @ old_transform
                    container_asset["transform"] = serialize_rigid_transform(
                        new_container_transform
                    )

                fill_assets = info.get("fill_assets", [])
                for fill_asset in fill_assets:
                    old_transform = deserialize_rigid_transform(fill_asset["transform"])
                    new_fill_transform = t_delta @ old_transform
                    fill_asset["transform"] = serialize_rigid_transform(
                        new_fill_transform
                    )

                # Update metadata with new transforms.
                composite_obj.metadata["container_asset"] = container_asset
                composite_obj.metadata["fill_assets"] = fill_assets
            else:
                # Stack or pile: update member_assets (same structure).
                updated_members = []
                for member in info["member_assets"]:
                    old_transform = deserialize_rigid_transform(member["transform"])
                    new_member_transform = t_delta @ old_transform
                    member["transform"] = serialize_rigid_transform(
                        new_member_transform
                    )
                    updated_members.append(member)

                # Update metadata with new transforms.
                composite_obj.metadata["member_assets"] = updated_members

            # Also update composite transform to match new reference member position.
            # This keeps bbox aligned with visual mesh positions.
            composite_obj.transform = new_bottom_transform


def _apply_self_collision_filtering(
    plant: MultibodyPlant, scene_graph: SceneGraph
) -> int:
    """Apply collision filtering to exclude self-collisions within each model.

    Articulated models (e.g., cabinets with doors/drawers) have internal collisions
    between their parts that are impossible to resolve without significant joint
    movement. These self-collisions cause MinimumDistanceConstraint to fail.

    This function filters out collisions between geometries within the same model,
    so the solver only needs to resolve inter-object and floor/wall penetrations.

    Args:
        plant: Finalized MultibodyPlant with models loaded.
        scene_graph: SceneGraph connected to the plant.

    Returns:
        Number of models that had self-collision filtering applied.
    """
    filter_manager = scene_graph.collision_filter_manager()
    inspector = scene_graph.model_inspector()
    models_filtered = 0

    # Iterate over all model instances (skip world model at index 0).
    world_model = plant.world_body().model_instance()
    for i in range(plant.num_model_instances()):
        model_idx = ModelInstanceIndex(i)
        if model_idx == world_model:
            continue

        # Collect all proximity geometries for this model.
        model_geometry_ids = []
        for body_idx in plant.GetBodyIndices(model_idx):
            frame_id = plant.GetBodyFrameIdOrThrow(body_idx)
            geom_ids = inspector.GetGeometries(frame_id, Role.kProximity)
            model_geometry_ids.extend(geom_ids)

        # Only apply filtering if model has multiple geometries.
        if len(model_geometry_ids) > 1:
            geometry_set = GeometrySet(model_geometry_ids)
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(geometry_set)
            )
            models_filtered += 1

    if models_filtered > 0:
        console_logger.debug(
            f"Applied self-collision filtering to {models_filtered} models"
        )

    return models_filtered


def solve_non_penetration_ik(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    influence_distance: float = 0.02,
    fix_rotation: bool = True,
    fix_z: bool = False,
    solver_name: str = "snopt",
    iteration_limit: int = 5000,
    time_limit_s: float = 360.0,
    xy_regions: dict[BodyIndex, HPolyhedron] | None = None,
) -> tuple[Context | None, bool]:
    """Solve IK for non-penetration projection of free bodies.

    Shared utility for projecting free-floating bodies to resolve penetrations.
    Builds diagram, sets up IK with proper quaternion/position costs, adds
    non-penetration constraint, and solves.

    Automatically applies self-collision filtering to exclude internal collisions
    within each model (e.g., cabinet doors vs body). This prevents the solver from
    trying to resolve impossible self-penetrations in articulated models.

    Args:
        builder: DiagramBuilder with plant and scene_graph added (not yet built).
        plant: Finalized MultibodyPlant with free bodies to project.
        scene_graph: SceneGraph connected to the plant (for collision filtering).
        influence_distance: Collision influence distance in meters.
        fix_rotation: If True, hard-constrain rotations to initial values.
        fix_z: If True, hard-constrain Z positions (XY-only projection).
        solver_name: NLP solver ("snopt" or "ipopt").
        iteration_limit: Max solver iterations.
        time_limit_s: Max solver time in seconds.
        xy_regions: Optional per-body 2D convex region constraints. Maps each
            BodyIndex to an HPolyhedron (Ax <= b) defining the allowed XY
            positions for that body's origin. Each body can have a different
            region based on its footprint.

            Computed via Pontryagin difference: for each object, the feasible
            region is surface_hpoly.PontryaginDifference(object_footprint_hpoly).
            This accounts for object shape - long objects (knives) can be
            placed closer to edges when oriented parallel.

    Returns:
        Tuple of (plant_context, success).
        On success, plant_context has the projected positions applied.
        Caller can extract body poses via plant.EvalBodyPoseInWorld().
        On failure, returns (None, False).
    """
    # Apply self-collision filtering before building diagram.
    # This excludes internal collisions within articulated models (e.g., cabinet
    # doors vs body) that are impossible to resolve without significant joint movement.
    _apply_self_collision_filtering(plant=plant, scene_graph=scene_graph)

    # Build diagram to connect plant and scene_graph.
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Set up IK.
    ik = InverseKinematics(plant, plant_context)
    q_vars = ik.q()
    prog = ik.prog()

    # Get initial positions.
    q0 = plant.GetPositions(plant_context)
    if len(q0) == 0:
        console_logger.warning("No DOFs found for plant. Skipping projection.")
        return None, False

    # Add costs and constraints for each free body.
    for body_idx in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_idx)
        q_start_idx = body.floating_positions_start()
        model_idx = cyclopean_get_model_instance_for_body(plant, body_idx)

        # Quaternion variables [qw, qx, qy, qz].
        model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]
        quat0 = q0[q_start_idx : q_start_idx + 4]

        # Add quadratic cost to stay close to initial orientation.
        # For quaternion q and q0, the cost approximates 1-cos(θ) = 2 - 2*(qᵀq₀)².
        prog.AddQuadraticCost(
            -4 * np.outer(quat0, quat0),
            np.zeros((4,)),
            2,
            model_quat_vars,
            is_convex=False,
        )

        # Position variables [x, y, z].
        model_pos_vars = q_vars[q_start_idx + 4 : q_start_idx + 7]
        pos0 = q0[q_start_idx + 4 : q_start_idx + 7]

        # Add quadratic cost to stay close to initial position.
        prog.AddQuadraticErrorCost(np.eye(3), pos0, model_pos_vars)

        # Fix rotation if requested.
        if fix_rotation:
            model_q = plant.GetPositions(plant_context, model_idx)
            model_quat = model_q[:4]
            prog.AddBoundingBoxConstraint(model_quat, model_quat, model_quat_vars)

        # Fix Z if requested.
        if fix_z:
            model_q = plant.GetPositions(plant_context, model_idx)
            model_z = model_q[6]  # Z is at index 6 (after quat + x + y).
            z_var = q_vars[q_start_idx + 6]
            prog.AddBoundingBoxConstraint(model_z, model_z, [z_var])

        # Add XY convex region constraints if provided.
        # Each region is an HPolyhedron from Pontryagin difference:
        # feasible_region = surface - object_footprint.
        if xy_regions and body_idx in xy_regions:
            region = xy_regions[body_idx]  # HPolyhedron
            x_var = q_vars[q_start_idx + 4]
            y_var = q_vars[q_start_idx + 5]
            xy_vars = np.array([x_var, y_var])
            # AddPointInSetConstraints adds linear constraints Ax <= b.
            region.AddPointInSetConstraints(prog, xy_vars)

    # Add minimum distance constraint (non-penetration).
    # LowerBound ensures objects are at least 1e-5m distance apart.
    ik.AddMinimumDistanceLowerBoundConstraint(1e-5, influence_distance)

    # Set initial guess.
    prog.SetInitialGuess(q_vars, q0)

    # Configure solver.
    options = SolverOptions()
    if solver_name == "snopt":
        solver = SnoptSolver()
        if not solver.available():
            raise ValueError("SNOPT solver not available")
        options.SetOption(solver.id(), "Major feasibility tolerance", 1e-3)
        options.SetOption(solver.id(), "Major optimality tolerance", 1e-3)
        options.SetOption(solver.id(), "Major iterations limit", iteration_limit)
        options.SetOption(solver.id(), "Time limit", time_limit_s)
        options.SetOption(solver.id(), "Timing level", 3)
    elif solver_name == "ipopt":
        solver = IpoptSolver()
        if not solver.available():
            raise ValueError("IPOPT solver not available")
        options.SetOption(solver.id(), "max_iter", iteration_limit)
    else:
        raise ValueError(f"Invalid solver: {solver_name}")

    # Solve with hard timeout enforcement.
    # SNOPT's internal "Time limit" option is unreliable in edge cases (can hang
    # indefinitely). Use SIGALRM as external enforcement with grace period.
    hard_timeout_s = int(time_limit_s) + 60

    def hard_timeout_handler(signum, frame):
        raise TimeoutError(
            f"Projection hard timeout: solver did not respect {time_limit_s}s limit"
        )

    old_handler = signal.signal(signal.SIGALRM, hard_timeout_handler)
    signal.alarm(hard_timeout_s)
    try:
        result = solver.Solve(prog, None, options)
        success = result.is_success()
    except TimeoutError as e:
        console_logger.error(f"Hard timeout triggered: {e}")
        return None, False
    except (SystemExit, RuntimeError) as e:
        console_logger.warning(f"Solver failed with error: {e}")
        return None, False
    finally:
        signal.alarm(0)  # Cancel alarm.
        signal.signal(signal.SIGALRM, old_handler)  # Restore handler.

    if not success:
        solution_result = result.get_solution_result()
        console_logger.warning(f"Projection failed: {solution_result.name}")
        infeasible = result.GetInfeasibleConstraintNames(prog)
        if infeasible:
            console_logger.warning(f"Infeasible constraints: {infeasible}")
        return None, False

    # Apply solution.
    solution = result.GetSolution(q_vars)
    if not np.all(np.isfinite(solution)):
        console_logger.warning("Solver returned non-finite values")
        return None, False

    plant.SetPositions(plant_context, solution)
    return plant_context, True


def cyclopean_get_model_instance_for_body(
    plant: MultibodyPlant, body_idx: BodyIndex
) -> ModelInstanceIndex:
    """Get model instance for a body (workaround for Drake API gap)."""
    body = plant.get_body(body_idx)
    return body.model_instance()
