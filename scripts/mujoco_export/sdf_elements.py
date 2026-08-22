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
import shutil
import xml.etree.ElementTree as ET

from pathlib import Path

import mujoco
import numpy as np
import trimesh

from scenesmith.utils.geometry.sdf_utils import parse_scale

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scripts.mujoco_export.mesh_conversion import (
    convert_gltf_to_obj,
    get_gltf_base_color,
)
from scripts.mujoco_export.mesh_utils import (
    apply_scale_to_trimesh,
    build_mesh_asset_filename,
    maybe_drop_degenerate_collision_geom,
)
from scripts.mujoco_export.scene_io import resolve_package_uri
from scripts.mujoco_export.transforms import (
    parse_pose,
    quat_conjugate,
    quat_rotate_vector,
    quat_to_rotation_matrix,
)

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def apply_inertial(body: mujoco._specs.MjsBody, inertial_elem: ET.Element) -> None:
    """Apply inertial properties from SDF to MuJoCo body.

    Preserves off-diagonal inertia terms using MuJoCo's fullinertia attribute
    when needed. Falls back to diagonal inertia when off-diagonal terms are zero
    and the inertial frame has identity rotation.
    """
    mass_elem = inertial_elem.find("mass")
    if mass_elem is not None and mass_elem.text:
        body.mass = float(mass_elem.text)

    pose_elem = inertial_elem.find("pose")
    pos = [0.0, 0.0, 0.0]
    quat = [1.0, 0.0, 0.0, 0.0]
    if pose_elem is not None:
        pos, quat = parse_pose(pose_elem)
    body.ipos = pos

    inertia_elem = inertial_elem.find("inertia")
    if inertia_elem is not None:
        ixx = get_float(inertia_elem, "ixx", 0.0)
        iyy = get_float(inertia_elem, "iyy", 0.0)
        izz = get_float(inertia_elem, "izz", 0.0)
        ixy = get_float(inertia_elem, "ixy", 0.0)
        ixz = get_float(inertia_elem, "ixz", 0.0)
        iyz = get_float(inertia_elem, "iyz", 0.0)

        has_off_diagonal = ixy != 0.0 or ixz != 0.0 or iyz != 0.0
        has_rotation = quat != [1.0, 0.0, 0.0, 0.0]

        if has_off_diagonal or has_rotation:
            # Build full 3x3 symmetric inertia tensor.
            I_local = np.array(
                [
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz],
                ]
            )

            if has_rotation:
                # Transform inertia tensor from inertial frame to body frame.
                R = quat_to_rotation_matrix(quat)
                I_body = R @ I_local @ R.T
            else:
                I_body = I_local

            # fullinertia expects [ixx, iyy, izz, ixy, ixz, iyz].
            body.fullinertia = [
                I_body[0, 0],
                I_body[1, 1],
                I_body[2, 2],
                I_body[0, 1],
                I_body[0, 2],
                I_body[1, 2],
            ]
            # Do not set iquat; MuJoCo compiler sets it from eigendecomposition.
        else:
            body.inertia = [ixx, iyy, izz]
            body.iquat = quat

    body.explicitinertial = True


def get_float(parent: ET.Element, tag: str, default: float) -> float:
    """Get float value from child element."""
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return float(elem.text)
    return default


def add_joint_from_sdf(
    body: mujoco._specs.MjsBody,
    joint_elem: ET.Element,
    child_abs_quat: list[float] | None = None,
    joint_pos: list[float] | None = None,
    name_prefix: str = "",
) -> None:
    """Add joint to body based on SDF joint element.

    Args:
        body: MuJoCo body to add joint to.
        joint_elem: SDF <joint> element.
        child_abs_quat: Child link's absolute quaternion in model frame [w, x, y, z].
            Used to transform axis from model frame to child link frame.
        name_prefix: Prefix for joint name to ensure uniqueness across rooms.
        joint_pos: Joint anchor position [x, y, z] from SDF joint pose.
    """
    raw_joint_name = joint_elem.get("name", "joint")
    # Prefix with name_prefix (model_name + link_name) to ensure uniqueness
    # across rooms that may have the same articulated furniture.
    joint_name = f"{name_prefix}_{raw_joint_name}" if name_prefix else raw_joint_name
    joint_type_str = joint_elem.get("type", "revolute")

    # Get MuJoCo joint type.
    mj_joint_type = SDF_TO_MJCF_JOINT_TYPE.get(joint_type_str)
    if mj_joint_type is None:
        console_logger.warning(f"Unsupported joint type: {joint_type_str}")
        return

    # Create joint.
    joint = body.add_joint(name=joint_name)
    joint.type = mj_joint_type

    # Apply joint position (anchor point) if provided.
    if joint_pos is not None:
        joint.pos = joint_pos

    # Parse axis.
    axis_elem = joint_elem.find("axis")
    if axis_elem is not None:
        xyz_elem = axis_elem.find("xyz")
        if xyz_elem is not None and xyz_elem.text:
            axis_values = [float(v) for v in xyz_elem.text.split()]

            # Check if axis is expressed in model frame.
            # If so, transform it to the child link frame.
            expressed_in = xyz_elem.get("expressed_in", "")
            if expressed_in == "__model__" and child_abs_quat is not None:
                # Transform axis from model frame to child link frame.
                # axis_in_child = q_child^-1 * axis_in_model.
                child_quat_inv = quat_conjugate(child_abs_quat)
                axis_values = quat_rotate_vector(child_quat_inv, axis_values)

            joint.axis = axis_values

        # Parse limits.
        limit_elem = axis_elem.find("limit")
        if limit_elem is not None:
            lower = get_float(limit_elem, "lower", -np.inf)
            upper = get_float(limit_elem, "upper", np.inf)
            if np.isfinite(lower) and np.isfinite(upper):
                joint.limited = True
                joint.range = [lower, upper]

        # Parse dynamics (damping).
        dynamics_elem = axis_elem.find("dynamics")
        if dynamics_elem is not None:
            damping = get_float(dynamics_elem, "damping", 0.0)
            if damping > 0:
                joint.damping = damping

            friction = get_float(dynamics_elem, "friction", 0.0)
            if friction > 0:
                joint.frictionloss = friction


def add_geom_from_sdf(
    spec: mujoco.MjSpec,
    body: mujoco._specs.MjsBody,
    geom_elem: ET.Element,
    sdf_dir: Path,
    meshes_dir: Path,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
    color_assets: dict[str, list[float]],
    is_collision: bool,
    name_prefix: str,
    room_id: str = "",
) -> None:
    """Add geometry to body from SDF visual or collision element."""
    base_name = geom_elem.get("name", "geom")
    geom_kind = "collision" if is_collision else "visual"
    geom_name = f"{name_prefix}_{base_name}_{geom_kind}"

    geometry_elem = geom_elem.find("geometry")
    if geometry_elem is None:
        return

    pos, quat = parse_pose(geom_elem.find("pose"))

    geom = body.add_geom(name=geom_name)
    geom.pos = pos
    geom.quat = quat

    if is_collision:
        geom.contype = 1
        geom.conaffinity = 1
        geom.group = 3  # Collision geoms in group 3 (toggle with key 3 in viewer)

        surface_elem = geom_elem.find("surface")
        if surface_elem is not None:
            friction_elem = surface_elem.find("friction")
            if friction_elem is not None:
                ode_elem = friction_elem.find("ode")
                if ode_elem is not None:
                    mu = get_float(ode_elem, "mu", 1.0)
                    geom.friction = [mu, 0.005, 0.0001]
    else:
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 0  # Visual geoms in group 0 (toggle with key 0 in viewer)

    # Handle geometry types.
    box_elem = geometry_elem.find("box")
    if box_elem is not None:
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        size_elem = box_elem.find("size")
        if size_elem is not None and size_elem.text:
            sizes = [float(v) for v in size_elem.text.split()]
            geom.size = [s / 2 for s in sizes]
        return

    sphere_elem = geometry_elem.find("sphere")
    if sphere_elem is not None:
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        radius_elem = sphere_elem.find("radius")
        if radius_elem is not None and radius_elem.text:
            geom.size = [float(radius_elem.text), 0, 0]
        return

    cylinder_elem = geometry_elem.find("cylinder")
    if cylinder_elem is not None:
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        radius_elem = cylinder_elem.find("radius")
        length_elem = cylinder_elem.find("length")
        if radius_elem is not None and length_elem is not None:
            r = float(radius_elem.text) if radius_elem.text else 0.5
            h = float(length_elem.text) / 2 if length_elem.text else 0.5
            geom.size = [r, h, 0]
        return

    mesh_elem = geometry_elem.find("mesh")
    if mesh_elem is not None:
        uri_elem = mesh_elem.find("uri")
        scale_elem = mesh_elem.find("scale")
        mesh_scale = [1.0, 1.0, 1.0]
        base_color_name = None
        if scale_elem is not None and scale_elem.text:
            mesh_scale = parse_scale(scale_elem.text)
        if uri_elem is not None and uri_elem.text:
            mesh_uri = uri_elem.text
            # Resolve package:// URIs or regular paths.
            mesh_path = resolve_package_uri(mesh_uri, sdf_dir)
            mesh_name = f"{geom_name}_mesh"
            texture_name = None
            base_color = None

            if mesh_path is not None and mesh_path.exists():
                # Convert GLTF to OBJ if necessary.
                if mesh_path.suffix.lower() in (".gltf", ".glb"):
                    # Include parent directory to avoid filename collisions.
                    # e.g., "north_wall/wall.gltf" → "north_wall_wall.obj"
                    # Include room_id prefix to avoid collisions across rooms.
                    parent_prefix = mesh_path.parent.name
                    room_prefix = f"{room_id}_" if room_id else ""
                    # Include scale in filename to cache scaled variants separately.
                    scale_suffix = ""
                    if mesh_scale != [1.0, 1.0, 1.0]:
                        scale_suffix = f"_s{'_'.join(f'{s:.3g}' for s in mesh_scale)}"
                    obj_filename = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}.obj"
                    obj_path = meshes_dir / obj_filename
                    # Use consistent names based on mesh file, not geom.
                    # Include scale suffix to differentiate scaled variants.
                    base_texture_name = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}_tex"
                    base_color_name = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}_color"
                    expected_texture_file = f"{room_prefix}{parent_prefix}_{mesh_path.stem}{scale_suffix}_texture.png"

                    if not obj_path.exists():
                        success, texture_path, base_color = convert_gltf_to_obj(
                            gltf_path=mesh_path,
                            obj_path=obj_path,
                            texture_dir=meshes_dir,
                            scale=mesh_scale,
                        )
                        if not success:
                            console_logger.warning(
                                f"Skipping mesh {mesh_name}: GLTF conversion failed"
                            )
                            if is_collision:
                                spec.delete(geom)
                            else:
                                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                                geom.size = [0.1, 0.1, 0.1]
                            return
                        # Track texture if extracted.
                        if texture_path and texture_path.exists():
                            texture_name = base_texture_name
                            texture_assets[texture_name] = texture_path.name
                        # Track base color if no texture.
                        elif base_color:
                            color_assets[base_color_name] = base_color
                    else:
                        # OBJ already exists - check if texture was previously extracted.
                        existing_texture = meshes_dir / expected_texture_file
                        if existing_texture.exists():
                            texture_name = base_texture_name
                            # Ensure texture is in assets (may already be there).
                            if texture_name not in texture_assets:
                                texture_assets[texture_name] = expected_texture_file
                        elif base_color_name not in color_assets:
                            # Try to get base color from GLTF.
                            base_color = get_gltf_base_color(mesh_path)
                            if base_color:
                                color_assets[base_color_name] = base_color

                    if is_collision:
                        try:
                            collision_mesh = trimesh.load(obj_path, force="mesh")
                        except Exception as e:
                            console_logger.warning(
                                f"Failed to validate converted collision mesh {obj_path}: {e}"
                            )
                            spec.delete(geom)
                            return
                        if maybe_drop_degenerate_collision_geom(
                            spec=spec,
                            geom=geom,
                            geom_name=geom_name,
                            mesh=collision_mesh,
                            mesh_path=obj_path,
                        ):
                            return

                    mesh_assets[mesh_name] = obj_filename
                else:
                    # OBJ/STL mesh - validate and optionally scale.
                    try:
                        existing_mesh = trimesh.load(mesh_path, force="mesh")
                    except Exception as e:
                        console_logger.warning(
                            f"Failed to validate mesh {mesh_path}: {e}"
                        )
                        if is_collision:
                            spec.delete(geom)
                        else:
                            geom.type = mujoco.mjtGeom.mjGEOM_BOX
                            geom.size = [0.05, 0.05, 0.05]
                        return

                    if mesh_scale != [1.0, 1.0, 1.0]:
                        apply_scale_to_trimesh(existing_mesh, mesh_scale)

                    if is_collision and maybe_drop_degenerate_collision_geom(
                        spec=spec,
                        geom=geom,
                        geom_name=geom_name,
                        mesh=existing_mesh,
                        mesh_path=mesh_path,
                    ):
                        return

                    dest_filename = build_mesh_asset_filename(
                        mesh_path=mesh_path,
                        sdf_dir=sdf_dir,
                        room_id=room_id,
                        scale=mesh_scale,
                    )
                    dest_path = meshes_dir / dest_filename
                    if not dest_path.exists():
                        if mesh_scale != [1.0, 1.0, 1.0]:
                            # The mesh already has the export scale applied in-memory.
                            existing_mesh.export(dest_path)
                            console_logger.info(
                                f"Scaled mesh {mesh_path.name} -> {dest_filename}"
                            )
                        else:
                            shutil.copy(mesh_path, dest_path)
                    mesh_assets[mesh_name] = dest_filename
            else:
                console_logger.warning(f"Mesh not found: {mesh_uri}")
                # Use default box as fallback.
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = [0.1, 0.1, 0.1]
                return

            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh_name

            # Apply material/color (visual geoms only).
            if not is_collision:
                if texture_name:
                    geom.material = texture_name
                elif base_color_name and base_color_name in color_assets:
                    # Apply base color directly to geom.
                    geom.rgba = color_assets[base_color_name]
        return

    # Default fallback.
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.1, 0.1, 0.1]
