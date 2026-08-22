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

from pxr import Sdf, Usd, UsdPhysics

console_logger = logging.getLogger(__name__)

from scripts.usd_physics.articulations import (
    _add_self_collision_filter,
    _clear_wrapper_target_from_fixed_joint,
    _collect_articulation_base_prims,
    _find_internal_fixed_joints,
    _find_nearest_rigid_body_ancestor,
    _flatten_nested_articulated_links,
    _promote_base_rigid_body,
)
from scripts.usd_physics.primitives import (
    _collect_invalid_direct_joint_targets,
    _has_articulated_joint_descendants,
    copy_mass_to_prim,
    delete_prims,
    find_fixed_joints_with_body0,
    remove_mass_api,
    remove_rigid_body_api,
)


def _has_nested_rigid_bodies(wrapper_prim: Usd.Prim) -> bool:
    """Check if any child rigid body has a child that is also a rigid body."""
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            for grandchild in child.GetChildren():
                if grandchild.HasAPI(UsdPhysics.RigidBodyAPI):
                    return True
    return False


def _has_dynamic_mass_or_rigid_body_authoring(wrapper_prim: Usd.Prim) -> bool:
    """Return True when the wrapper already looks like a movable rigid object."""
    if wrapper_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return True
    if wrapper_prim.HasAPI(UsdPhysics.MassAPI):
        return True
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(UsdPhysics.MassAPI):
            return True
    return False


def classify_object(
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> str:
    """Classify an object as 'static', 'dynamic', or 'articulated'.

    Classification logic:
    1. Check ArticulationRootAPI first — articulated objects may not have
       FixedJoints to root (e.g. when furniture uses freejoints in MuJoCo).
    2. Check for nested rigid bodies — this catches partially-fixed objects
       from prior runs where ArticulationRootAPI was already removed but
       bodies were not yet reparented as siblings.
    3. Check if wrapper has a FixedJoint descendant with body0 targeting root.
    4. If welded and no ArticulationRootAPI -> 'static'.
    5. If not welded and no ArticulationRootAPI -> 'dynamic'.
    """
    if _has_articulated_joint_descendants(wrapper_prim):
        return "articulated"
    if wrapper_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return "articulated"
    welded_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    if welded_joints:
        return "static"
    if _has_dynamic_mass_or_rigid_body_authoring(wrapper_prim):
        return "dynamic"
    return "static"


def fix_static_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
) -> None:
    """Fix a static object by removing all physics body APIs and joints.

    Leaves only PhysicsCollisionAPI on collision geometry, making the object
    a static collider in Isaac Sim.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Remove RigidBodyAPI from wrapper.
    remove_rigid_body_api(wrapper_prim)

    # Remove RigidBodyAPI + MassAPI from all descendants.
    for descendant in Usd.PrimRange(wrapper_prim):
        if descendant.GetPath() == wrapper_path:
            continue
        remove_rigid_body_api(descendant)
        remove_mass_api(descendant)

    # Delete FixedJoint from wrapper to root.
    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    delete_prims(stage, root_joints)

    # Delete FixedJoint from base_link/body_link to wrapper.
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_dynamic_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> None:
    """Fix a dynamic object by flattening to a single rigid body.

    Moves MassAPI from base_link to wrapper and removes the inner
    RigidBodyAPI and FixedJoint.
    """
    wrapper_path = wrapper_prim.GetPath()

    # Find the immediate child (base_link) that has MassAPI.
    base_link = None
    for child in wrapper_prim.GetChildren():
        if child.HasAPI(UsdPhysics.MassAPI):
            base_link = child
            break

    if base_link is None:
        console_logger.debug(
            f"Dynamic object {wrapper_path} has no child with MassAPI, "
            "skipping mass copy."
        )
    else:
        # Copy mass properties from base_link to wrapper.
        copy_mass_to_prim(source=base_link, target=wrapper_prim)
        # Remove MassAPI + RigidBodyAPI from base_link.
        remove_mass_api(base_link)
        remove_rigid_body_api(base_link)

    # Delete FixedJoint inside base_link (base_link→wrapper).
    inner_joints = find_fixed_joints_with_body0(wrapper_prim, wrapper_path)
    delete_prims(stage, inner_joints)


def fix_articulated_object(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
    root_path: Sdf.Path,
    all_layers: list[Sdf.Layer],
) -> None:
    """Repair articulated objects without replacing authored collision."""
    wrapper_path = wrapper_prim.GetPath()

    root_joints = find_fixed_joints_with_body0(wrapper_prim, root_path)
    is_welded = len(root_joints) > 0

    promoted_bases = _collect_invalid_direct_joint_targets(stage, wrapper_prim)
    for base_prim in promoted_bases:
        _promote_base_rigid_body(wrapper_prim, base_prim)

    moved_links = _flatten_nested_articulated_links(
        stage=stage,
        wrapper_prim=wrapper_prim,
        all_layers=all_layers,
    )

    base_paths = {
        prim.GetPath() for prim in _collect_articulation_base_prims(stage, wrapper_prim)
    }
    wrapper_to_base_joints = _find_internal_fixed_joints(
        wrapper_prim=wrapper_prim,
        wrapper_path=wrapper_path,
        base_paths=base_paths,
    )
    if is_welded:
        for joint_path in wrapper_to_base_joints:
            _clear_wrapper_target_from_fixed_joint(
                stage=stage,
                joint_path=joint_path,
                wrapper_path=wrapper_path,
            )
        console_logger.debug(f"  {wrapper_path.name}: fixed-base articulation")
    else:
        delete_prims(stage, wrapper_to_base_joints)
        console_logger.debug(f"  {wrapper_path.name}: free articulation")

    delete_prims(stage, root_joints)
    if base_paths:
        remove_rigid_body_api(wrapper_prim)

    for descendant in Usd.PrimRange(wrapper_prim):
        if not descendant.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(descendant)
        for rel in [joint.GetBody0Rel(), joint.GetBody1Rel()]:
            if not rel:
                continue
            targets = rel.GetTargets()
            if not targets:
                continue
            target_path = targets[0]
            target_prim = stage.GetPrimAtPath(target_path)
            if (
                target_prim
                and target_prim.IsValid()
                and target_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                continue
            replacement = _find_nearest_rigid_body_ancestor(
                stage=stage,
                target_path=target_path,
                stop_path=wrapper_path,
            )
            if replacement and replacement != target_path:
                rel.SetTargets([replacement])

    _add_self_collision_filter(stage, wrapper_prim)
    if promoted_bases or moved_links:
        console_logger.debug(
            f"  {wrapper_path.name}: promoted {len(promoted_bases)} base bodies, "
            f"reparented {moved_links} articulated links"
        )
