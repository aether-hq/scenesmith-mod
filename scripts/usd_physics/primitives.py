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


def remove_rigid_body_api(prim: Usd.Prim) -> bool:
    """Remove PhysicsRigidBodyAPI from a prim if present."""
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        return True
    return False


def remove_mass_api(prim: Usd.Prim) -> None:
    """Remove PhysicsMassAPI and all mass properties from a prim."""
    if not prim.HasAPI(UsdPhysics.MassAPI):
        return
    prim.RemoveAPI(UsdPhysics.MassAPI)
    for prop_name in [
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ]:
        prop = prim.GetProperty(prop_name)
        if prop:
            prim.RemoveProperty(prop_name)


def copy_mass_to_prim(source: Usd.Prim, target: Usd.Prim) -> None:
    """Copy PhysicsMassAPI and its properties from source to target prim."""
    if not source.HasAPI(UsdPhysics.MassAPI):
        return
    UsdPhysics.MassAPI.Apply(target)

    mass_props = [
        ("physics:mass", "float"),
        ("physics:centerOfMass", "point3f"),
        ("physics:diagonalInertia", "float3"),
        ("physics:principalAxes", "quatf"),
    ]
    for prop_name, _ in mass_props:
        src_attr = source.GetAttribute(prop_name)
        if src_attr and src_attr.HasValue():
            tgt_attr = target.GetAttribute(prop_name)
            if not tgt_attr:
                # Create with same type as source.
                tgt_attr = target.CreateAttribute(prop_name, src_attr.GetTypeName())
            tgt_attr.Set(src_attr.Get())


def find_fixed_joints_with_body0(
    root_prim: Usd.Prim, body0_path: Sdf.Path
) -> list[Sdf.Path]:
    """Find all PhysicsFixedJoint descendants whose body0 targets body0_path."""
    joint_paths = []
    for descendant in Usd.PrimRange(root_prim):
        if descendant.GetTypeName() == "PhysicsFixedJoint":
            body0_rel = descendant.GetRelationship("physics:body0")
            if body0_rel:
                targets = body0_rel.GetTargets()
                if targets and targets[0] == body0_path:
                    joint_paths.append(descendant.GetPath())
    return joint_paths


def delete_prims(stage: Usd.Stage, paths: list[Sdf.Path]) -> int:
    """Delete prims at the given paths. Returns count of deleted prims."""
    count = 0
    for path in paths:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
            count += 1
    return count


def _reparent_rigid_body_children(
    stage: Usd.Stage,
    source_parent_prim: Usd.Prim,
    new_parent_prim: Usd.Prim,
    all_layers: list[Sdf.Layer],
) -> int:
    """Move rigid body children from one parent to another across all layers."""
    source_path = source_parent_prim.GetPath()
    new_parent_path = new_parent_prim.GetPath()

    prims_to_move: list[Sdf.Path] = []
    for child in source_parent_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            prims_to_move.append(child.GetPath())

    if not prims_to_move:
        return 0

    path_mapping: dict[str, str] = {}
    for old_path in prims_to_move:
        new_path = new_parent_path.AppendChild(old_path.name)
        path_mapping[str(old_path)] = str(new_path)

    for layer in all_layers:
        edit = Sdf.BatchNamespaceEdit()
        has_edits = False
        for old_path in prims_to_move:
            if layer.GetPrimAtPath(old_path):
                new_path = new_parent_path.AppendChild(old_path.name)
                edit.Add(old_path, new_path)
                has_edits = True
        if has_edits:
            if not layer.Apply(edit):
                console_logger.warning(
                    f"Failed to reparent in layer {layer.identifier}"
                )

    for descendant in Usd.PrimRange(new_parent_prim):
        for rel in descendant.GetRelationships():
            targets = rel.GetTargets()
            new_targets = []
            changed = False
            for target in targets:
                target_str = str(target)
                for old_str, new_str in path_mapping.items():
                    if target_str == old_str or target_str.startswith(old_str + "/"):
                        target_str = new_str + target_str[len(old_str) :]
                        changed = True
                        break
                new_targets.append(Sdf.Path(target_str))
            if changed:
                rel.SetTargets(new_targets)

    console_logger.debug(
        f"  {new_parent_path.name}: reparented {len(prims_to_move)} rigid "
        f"children from {source_path.name}"
    )
    return len(prims_to_move)


def _has_articulated_joint_descendants(wrapper_prim: Usd.Prim) -> bool:
    """Return True if the wrapper contains a non-fixed physics joint."""
    for descendant in Usd.PrimRange(wrapper_prim):
        if (
            descendant.IsA(UsdPhysics.Joint)
            and descendant.GetTypeName() != "PhysicsFixedJoint"
        ):
            return True
    return False


def _collect_invalid_direct_joint_targets(
    stage: Usd.Stage,
    wrapper_prim: Usd.Prim,
) -> list[Usd.Prim]:
    """Collect invalid direct child body0 targets that should be base rigid bodies."""
    wrapper_path = wrapper_prim.GetPath()
    targets: dict[str, Usd.Prim] = {}

    for descendant in Usd.PrimRange(wrapper_prim):
        if not descendant.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(descendant)
        rel = joint.GetBody0Rel()
        if not rel:
            continue
        for target in rel.GetTargets():
            if target.GetParentPath() != wrapper_path:
                continue
            target_prim = stage.GetPrimAtPath(target)
            if (
                target_prim
                and target_prim.IsValid()
                and not target_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                targets[str(target)] = target_prim

    return list(targets.values())
