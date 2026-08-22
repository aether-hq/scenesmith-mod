#!/usr/bin/env python3
"""Export an existing scene to self-contained MuJoCo MJCF format.

Takes a scene directory (e.g., outputs/2025-12-05/13-39-27/scene_039) and exports
it to a self-contained MuJoCo directory with the scene.xml and all referenced
mesh assets.

Can also export a single Drake SDF file to MuJoCo MJCF format.

Usage:
    python scripts/export_scene_to_mujoco.py <scene_path> [--output <output_path>]

Example:
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039 \
        --output /tmp/mujoco_scene
"""

import logging
import re

from pathlib import Path

import mujoco
import numpy as np
import trimesh

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def apply_scale_to_trimesh(mesh: trimesh.Trimesh, scale: list[float]) -> None:
    """Apply scale transformation to mesh vertices in-place.

    Args:
        mesh: Trimesh object to scale.
        scale: [sx, sy, sz] scale factors.
    """
    if scale == [1.0, 1.0, 1.0]:
        return
    scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    mesh.apply_transform(scale_matrix)


def build_mesh_asset_filename(
    mesh_path: Path,
    sdf_dir: Path,
    room_id: str,
    scale: list[float],
) -> str:
    """Build a collision/visual mesh filename that is stable and collision-free.

    Drake-generated articulated assets commonly store per-link collision pieces in
    sibling directories like:

      E_body_1_combined_coacd/convex_piece_000.obj
      E_door_1_16_combined_coacd/convex_piece_000.obj

    Flattening those into a shared MuJoCo meshes/ directory by basename alone
    aliases unrelated meshes together. Include the asset directory and relative
    subpath so per-link meshes stay distinct.
    """

    room_prefix = f"{room_id}_" if room_id else ""
    scale_suffix = ""
    if scale != [1.0, 1.0, 1.0]:
        scale_suffix = f"_s{'_'.join(f'{s:.3g}' for s in scale)}"

    try:
        relative_parts = mesh_path.relative_to(sdf_dir).parts[:-1]
        relative_prefix = "_".join(relative_parts)
    except ValueError:
        relative_prefix = mesh_path.parent.name

    parts = [room_prefix.rstrip("_"), sdf_dir.name, relative_prefix, mesh_path.stem]
    stem = "_".join(part for part in parts if part)
    return f"{stem}{scale_suffix}{mesh_path.suffix}"


def get_degenerate_trimesh_reason(mesh: trimesh.Trimesh) -> str | None:
    """Return a human-readable reason when a mesh is not a valid 3D collider."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return f"unexpected vertex shape {vertices.shape}"
    if vertices.shape[0] < 4:
        return f"only {vertices.shape[0]} vertices"
    if not np.isfinite(vertices).all():
        return "non-finite vertices"

    centered = vertices - vertices.mean(axis=0, keepdims=True)
    extents = np.ptp(vertices, axis=0)
    scale = max(float(np.max(extents)), 1.0)
    tol = max(scale * 1e-6, 1e-9)
    rank = np.linalg.matrix_rank(centered, tol=tol)
    if rank < 3:
        extents_str = ", ".join(f"{extent:.6g}" for extent in extents)
        return f"rank {rank} geometry with extents [{extents_str}]"

    return None


def maybe_drop_degenerate_collision_geom(
    spec: mujoco.MjSpec,
    geom: mujoco._specs.MjsGeom,
    geom_name: str,
    mesh: trimesh.Trimesh,
    mesh_path: Path,
) -> bool:
    """Delete one collision geom when its mesh is too degenerate for MuJoCo."""
    reason = get_degenerate_trimesh_reason(mesh)
    if reason is None:
        return False

    console_logger.warning(
        "Dropping degenerate collision geom '%s' from '%s': %s",
        geom_name,
        mesh_path,
        reason,
    )
    spec.delete(geom)
    return True


def drop_bad_collision_mesh_from_spec(
    spec: mujoco.MjSpec,
    bad_mesh: str,
    mesh_assets: dict[str, str],
) -> bool:
    """Remove a single collision geom+mesh pair when MuJoCo rejects it."""
    if not bad_mesh.endswith("_mesh"):
        return False

    geom_name = bad_mesh.removesuffix("_mesh")
    if not geom_name.endswith("_collision"):
        return False

    geom = next((geom for geom in spec.geoms if geom.name == geom_name), None)
    if geom is None:
        return False

    console_logger.warning(
        "Dropping collision geom '%s' after MuJoCo rejected mesh '%s'",
        geom_name,
        bad_mesh,
    )
    spec.delete(geom)
    mesh_assets.pop(bad_mesh, None)

    mesh = next((mesh for mesh in spec.meshes if mesh.name == bad_mesh), None)
    if mesh is not None:
        spec.delete(mesh)
    return True


def get_bad_mesh_name_from_compile_error(err_str: str) -> str | None:
    """Extract the offending mesh name from common MuJoCo compile errors."""
    volume_match = re.search(r"mesh volume is too small: (\S+)", err_str)
    if volume_match:
        return volume_match.group(1)

    if "qhull error" not in err_str:
        return None

    qhull_match = re.search(r"Element name '([^']+)'", err_str)
    if qhull_match:
        return qhull_match.group(1)

    return None
