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

from pxr import Sdf, Usd

console_logger = logging.getLogger(__name__)

from scripts.usd_physics.articulations import _apply_articulated_collision_mode
from scripts.usd_physics.classification import (
    classify_object,
    fix_articulated_object,
    fix_dynamic_object,
    fix_static_object,
)


def fix_physics_layer(
    physics_usda_path: Path,
    articulated_collision_mode: str = "preserve",
) -> dict[str, int]:
    """Fix physics in a Physics.usda file for Isaac Sim compatibility.

    Opens the composed stage and fixes all objects. For articulated
    objects, reparenting is applied across ALL sublayers (Physics,
    Geometry, Materials) so mesh data and materials move with the prims.

    Args:
        physics_usda_path: Path to the Physics.usda file.

    Returns:
        Dict with counts of objects fixed per category.
    """
    stage = Usd.Stage.Open(str(physics_usda_path))
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise RuntimeError(f"No default prim in {physics_usda_path}")

    root_path = root_prim.GetPath()

    # Find the Geometry scope.
    geometry_path = root_path.AppendChild("Geometry")
    geometry_prim = stage.GetPrimAtPath(geometry_path)
    if not geometry_prim:
        raise RuntimeError(f"No Geometry scope found at {geometry_path}")

    # Collect ALL sublayers in the Payload directory for reparenting.
    # The Payload dir contains Physics.usda, Geometry.usda, Materials.usda.
    payload_dir = physics_usda_path.parent
    all_layers: list[Sdf.Layer] = []
    for usda_file in sorted(payload_dir.glob("*.usda")):
        layer = Sdf.Layer.FindOrOpen(str(usda_file))
        if layer:
            all_layers.append(layer)

    counts: dict[str, int] = {"static": 0, "dynamic": 0, "articulated": 0}

    for wrapper_prim in geometry_prim.GetChildren():
        category = classify_object(
            wrapper_prim=wrapper_prim,
            root_path=root_path,
        )
        counts[category] += 1

        if category == "static":
            fix_static_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
            )
        elif category == "dynamic":
            fix_dynamic_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
            )
        elif category == "articulated":
            fix_articulated_object(
                stage=stage,
                wrapper_prim=wrapper_prim,
                root_path=root_path,
                all_layers=all_layers,
            )

    # Save ALL modified layers (Physics + Geometry + Materials).
    stage.GetRootLayer().Save()
    for layer in all_layers:
        if layer.dirty:
            layer.Save()

    collision_counts = _apply_articulated_collision_mode(
        physics_usda_path=physics_usda_path,
        collision_mode=articulated_collision_mode,
    )

    console_logger.info(
        f"Fixed {physics_usda_path}: "
        f"{counts['static']} static, "
        f"{counts['dynamic']} dynamic, "
        f"{counts['articulated']} articulated, "
        f"{collision_counts['wrappers']} articulated wrapper(s) collision-regenerated"
    )
    return counts


def _fix_single_scene(
    usd_dir: Path,
    articulated_collision_mode: str = "preserve",
) -> tuple[Path, dict[str, int] | str]:
    """Fix a single scene's Physics.usda. Returns (path, counts_or_error)."""
    physics_path = usd_dir / "Payload" / "Physics.usda"
    if not physics_path.exists():
        return usd_dir, "no Physics.usda found"
    try:
        counts = fix_physics_layer(
            physics_path,
            articulated_collision_mode=articulated_collision_mode,
        )
        return usd_dir, counts
    except Exception as e:
        return usd_dir, f"error: {e}"


def find_usd_dirs(base_path: Path, recursive: bool) -> list[Path]:
    """Find USD directories (containing Payload/Physics.usda)."""
    if not recursive:
        # Single scene: base_path should be the usd directory itself.
        if (base_path / "Payload" / "Physics.usda").exists():
            return [base_path]
        return []

    # Recursive: find all Physics.usda files.
    usd_dirs = []
    for physics_file in base_path.rglob("Payload/Physics.usda"):
        usd_dirs.append(physics_file.parent.parent)
    return sorted(usd_dirs)
