"""Fix USD physics for Isaac Sim compatibility.

The mujoco-usd-converter (v0.1.0a3) generates PhysicsFixedJoint prims that
connect objects to the root Xform, but the root has no PhysicsRigidBodyAPI.
PhysX requires valid physics bodies on both sides of a joint, so the
constraint solver pulls everything to (0,0,0).

This script post-processes Physics.usda files to fix three object categories:

1. **Static objects** (walls, desks, beds): Remove all physics body APIs and
   joints, leaving only collision geometry. Isaac Sim treats these as static
   colliders.

2. **Dynamic objects** (mugs, books): Flatten nested rigid bodies by moving
   MassAPI from base_link to wrapper, removing inner RigidBodyAPI, and
   deleting the internal FixedJoint.

3. **Articulated objects** (wardrobes with doors, fridges): Promote invalid
   base-body Xforms to real rigid bodies, reparent articulated links as
   siblings when needed, preserve authored collision geometry by default, and
   recreate self-collision filters (mirroring MuJoCo's ``<contact><exclude>``
   pairs). Optionally, articulated collision can be regenerated from visual
   meshes using Isaac-compatible mesh approximations.

Usage:
    # Fix single scene USD directory.
    python scripts/fix_usd_isaac_sim.py /path/to/scene/mujoco/usd

    # Fix all scenes recursively with parallel workers.
    python scripts/fix_usd_isaac_sim.py /path/to/SceneAgent_Cleaned \\
        --recursive --workers 16
"""

import logging

from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

console_logger = logging.getLogger(__name__)

from scripts.usd_physics.primitives import (
    _reparent_rigid_body_children,
    copy_mass_to_prim,
    remove_mass_api,
    remove_rigid_body_api,
)


def _collect_articulation_base_prims(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> list[Usd.Prim]:
    """Collect direct-child rigid bodies that anchor the articulation."""
    wrapper_path = wrapper_prim.GetPath()
    bases: dict[str, Usd.Prim] = {}

    for descendant in Usd.PrimRange(wrapper_prim):
        if not descendant.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(descendant)
        for rel in [joint.GetBody0Rel(), joint.GetBody1Rel()]:
            if not rel:
                continue
            for target in rel.GetTargets():
                path = target
                while path and path != wrapper_path:
                    prim = stage.GetPrimAtPath(path)
                    if (
                        prim
                        and prim.IsValid()
                        and prim.GetPath().GetParentPath() == wrapper_path
                        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    ):
                        bases[str(path)] = prim
                        break
                    path = path.GetParentPath()

    if not bases:
        for child in wrapper_prim.GetChildren():
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                bases[str(child.GetPath())] = child

    return list(bases.values())


def _promote_base_rigid_body(
    wrapper_prim: Usd.Prim,
    base_prim: Usd.Prim,
) -> None:
    """Move the base rigid-body authoring from the wrapper onto the base prim."""
    if not base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
    if wrapper_prim.HasAPI(UsdPhysics.MassAPI):
        copy_mass_to_prim(source=wrapper_prim, target=base_prim)
        remove_mass_api(wrapper_prim)
    remove_rigid_body_api(wrapper_prim)


def _find_internal_fixed_joints(
    wrapper_prim: Usd.Prim,
    wrapper_path: Sdf.Path,
    base_paths: set[Sdf.Path],
) -> list[Sdf.Path]:
    """Find internal fixed joints that connect the wrapper to a promoted base."""
    joint_paths: list[Sdf.Path] = []
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetTypeName() != "PhysicsFixedJoint":
            continue
        targets: set[Sdf.Path] = set()
        for rel_name in ["physics:body0", "physics:body1"]:
            rel = descendant.GetRelationship(rel_name)
            if rel:
                targets.update(rel.GetTargets())
        if wrapper_path in targets and base_paths.intersection(targets):
            joint_paths.append(descendant.GetPath())
    return joint_paths


def _clear_wrapper_target_from_fixed_joint(
    stage: Usd.Stage,
    joint_path: Sdf.Path,
    wrapper_path: Sdf.Path,
) -> None:
    """Clear whichever side of a fixed joint still targets the wrapper."""
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim:
        return
    for rel_name in ["physics:body0", "physics:body1"]:
        rel = joint_prim.GetRelationship(rel_name)
        if not rel:
            continue
        targets = rel.GetTargets()
        if wrapper_path in targets:
            remaining = [target for target in targets if target != wrapper_path]
            if remaining:
                rel.SetTargets(remaining)
            else:
                rel.ClearTargets(True)


def _find_nearest_rigid_body_ancestor(
    stage: Usd.Stage,
    target_path: Sdf.Path,
    stop_path: Sdf.Path,
) -> Sdf.Path | None:
    """Return the nearest rigid-body ancestor within the wrapper subtree."""
    path = target_path
    while str(path).startswith(str(stop_path)):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return path
        if path == stop_path:
            break
        path = path.GetParentPath()
    return None


def _flatten_nested_articulated_links(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    all_layers: list[Sdf.Layer],
) -> int:
    """Reparent nested articulated rigid bodies so links become siblings."""
    total_moved = 0

    while True:
        moved_this_pass = 0
        wrapper_prim = stage.GetPrimAtPath(wrapper_prim.GetPath())
        for child in list(wrapper_prim.GetChildren()):
            if not child.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            moved_this_pass += _reparent_rigid_body_children(
                stage=stage,
                source_parent_prim=child,
                new_parent_prim=wrapper_prim,
                all_layers=all_layers,
            )
        if not moved_this_pass:
            break
        total_moved += moved_this_pass

    return total_moved


def _retag_articulated_collision_meshes(
    wrapper_prim: Usd.Prim,
    approximation: str,
) -> dict[str, int]:
    """Replace articulated collision meshes with visual-mesh collision."""
    deactivated = 0
    visual_tagged = 0

    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetTypeName() != "Mesh":
            continue

        name_lower = descendant.GetName().lower()
        if "collision" in name_lower:
            if descendant.IsActive():
                descendant.SetActive(False)
                deactivated += 1
            continue

        if "visual" not in name_lower:
            continue

        if not descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(descendant)
        if not descendant.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(descendant)

        mesh_collision = UsdPhysics.MeshCollisionAPI(descendant)
        mesh_collision.GetApproximationAttr().Set(approximation)
        visual_tagged += 1

    return {
        "collision_deactivated": deactivated,
        "visual_tagged": visual_tagged,
    }


def _find_composed_scene_path(physics_usda_path: Path) -> Path | None:
    """Find the top-level composed scene stage for a USD export directory."""
    usd_dir = physics_usda_path.parent.parent
    scene_files = sorted(usd_dir.glob("scene_*.usda"))
    if scene_files:
        return scene_files[0]

    generic_scene_files = sorted(
        path
        for path in usd_dir.glob("*.usda")
        if path.parent == usd_dir and path.name != "scene_for_usd.xml"
    )
    if generic_scene_files:
        return generic_scene_files[0]

    return None


def _collect_articulated_wrapper_paths_from_joints(
    scene_stage: Usd.Stage,
) -> set[Sdf.Path]:
    """Find top-level geometry wrappers that contain non-fixed joints."""
    root_prim = scene_stage.GetDefaultPrim()
    if not root_prim:
        return set()

    geometry_path = root_prim.GetPath().AppendChild("Geometry")
    wrapper_paths: set[Sdf.Path] = set()

    for prim in scene_stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        if prim.GetTypeName() == "PhysicsFixedJoint":
            continue

        path = prim.GetPath()
        while path and path != geometry_path:
            if path.GetParentPath() == geometry_path:
                wrapper_paths.add(path)
                break
            path = path.GetParentPath()

    return wrapper_paths


def _apply_articulated_collision_mode(
    physics_usda_path: Path,
    collision_mode: str,
) -> dict[str, int]:
    """Apply articulated collision regeneration on the composed scene stage."""
    if collision_mode == "preserve":
        return {"wrappers": 0, "collision_deactivated": 0, "visual_tagged": 0}

    if collision_mode == "convex-hull":
        approximation = str(UsdPhysics.Tokens.convexHull)
    elif collision_mode == "convex-decomposition":
        approximation = str(UsdPhysics.Tokens.convexDecomposition)
    else:
        raise ValueError(f"Unsupported articulated collision mode: {collision_mode}")

    scene_path = _find_composed_scene_path(physics_usda_path)
    if scene_path is None:
        console_logger.warning(
            f"Could not find composed scene root for {physics_usda_path}; "
            "skipping articulated collision regeneration"
        )
        return {"wrappers": 0, "collision_deactivated": 0, "visual_tagged": 0}

    scene_stage = Usd.Stage.Open(str(scene_path))
    if not scene_stage:
        raise RuntimeError(f"Could not open composed scene stage: {scene_path}")

    totals = {"wrappers": 0, "collision_deactivated": 0, "visual_tagged": 0}
    for wrapper_path in sorted(
        _collect_articulated_wrapper_paths_from_joints(scene_stage),
        key=str,
    ):
        wrapper_prim = scene_stage.GetPrimAtPath(wrapper_path)
        if not wrapper_prim or not wrapper_prim.IsValid():
            continue
        counts = _retag_articulated_collision_meshes(wrapper_prim, approximation)
        totals["wrappers"] += 1
        totals["collision_deactivated"] += counts["collision_deactivated"]
        totals["visual_tagged"] += counts["visual_tagged"]

    scene_stage.GetRootLayer().Save()
    return totals


def _add_self_collision_filter(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> None:
    """Add self-collision filtering for all rigid bodies in an articulated object.

    The MuJoCo source has ``<contact><exclude>`` pairs that prevent adjacent
    articulated links from colliding (e.g. wardrobe body vs. its doors).
    The mujoco_usd_converter does not convert these (``Tf.Warn("excludes
    are not supported")``), so we recreate them using a PhysicsCollisionGroup
    that includes all rigid bodies within the object and filters against
    itself.

    Without this, PhysX detects collisions between overlapping bodies at
    hinge points, which prevents joints from moving interactively.
    """
    # Collect all rigid body prims under the wrapper.
    rigid_bodies = []
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(descendant.GetPath())

    if len(rigid_bodies) < 2:
        return  # No self-collision possible with fewer than 2 bodies.

    # Create a PhysicsCollisionGroup under the wrapper.
    group_path = wrapper_prim.GetPath().AppendChild("selfCollisionFilter")
    group = UsdPhysics.CollisionGroup.Define(stage, group_path)

    # Add all rigid bodies to the group via CollectionAPI.
    collection = group.GetCollidersCollectionAPI()
    includes_rel = collection.CreateIncludesRel()
    for body_path in rigid_bodies:
        includes_rel.AddTarget(body_path)

    # Filter the group against itself → disables collision between members.
    filtered_rel = group.GetFilteredGroupsRel()
    filtered_rel.AddTarget(group_path)

    console_logger.debug(
        f"  {wrapper_prim.GetPath().name}: self-collision filter for "
        f"{len(rigid_bodies)} bodies"
    )
