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
import xml.etree.ElementTree as ET

from pathlib import Path

import mujoco

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scripts.mujoco_export.scene_io import parse_sdf_with_drake_namespace
from scripts.mujoco_export.sdf_elements import (
    add_geom_from_sdf,
    add_joint_from_sdf,
    apply_inertial,
)
from scripts.mujoco_export.transforms import compute_relative_pose, parse_pose

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def process_sdf_model(
    sdf_path: Path,
    sdf_dir: Path,
    model_name: str,
    transform_pos: list[float],
    transform_quat: list[float],
    is_static: bool,
    spec: mujoco.MjSpec,
    meshes_dir: Path,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
    color_assets: dict[str, list[float]],
    room_id: str = "",
) -> list[str]:
    """Process an SDF model and add bodies/joints to spec.

    Supports both single-link (rigid) and multi-link (articulated) models.

    Args:
        sdf_path: Path to SDF file.
        sdf_dir: Directory containing SDF (for mesh resolution).
        model_name: Unique name for the model in MuJoCo.
        transform_pos: [x, y, z] position for model root.
        transform_quat: [w, x, y, z] quaternion for model root.
        is_static: Whether model should be static (no freejoint).
        spec: MuJoCo spec to add bodies to.
        meshes_dir: Directory for mesh assets.
        mesh_assets: Dict to track mesh assets (name -> filename).
        texture_assets: Dict to track texture assets (name -> filename).
        color_assets: Dict to track base color assets (name -> [r,g,b,a]).
        room_id: Optional room identifier for unique mesh naming across rooms.

    Returns:
        List of body names created for this model. Used for self-collision
        filtering on articulated models.
    """
    # Parse SDF (handles Drake namespace extensions).
    tree = parse_sdf_with_drake_namespace(sdf_path)
    root = tree.getroot()
    model_elem = root.find(".//model")
    if model_elem is None:
        console_logger.warning(f"No model element in SDF: {sdf_path}")
        return []

    # Check if SDF model is marked static.
    static_elem = model_elem.find("static")
    sdf_is_static = static_elem is not None and static_elem.text.lower() == "true"

    # Build link→parent mapping from joints.
    link_parents: dict[str, tuple[str, ET.Element | None]] = {}
    for joint_elem in model_elem.findall(".//joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is not None and child_elem is not None:
            link_parents[child_elem.text] = (parent_elem.text, joint_elem)

    # Collect links and their absolute poses in model frame.
    links = model_elem.findall(".//link")
    link_absolute_poses: dict[str, tuple[list[float], list[float]]] = {}
    for link_elem in links:
        link_name = link_elem.get("name")
        pos, quat = parse_pose(link_elem.find("pose"))
        link_absolute_poses[link_name] = (pos, quat)

    # Track created bodies and their absolute poses.
    link_bodies: dict[str, mujoco._specs.MjsBody] = {}

    # Create model root body with transform.
    model_root = spec.worldbody.add_body(name=model_name)
    model_root.pos = transform_pos
    model_root.quat = transform_quat

    # Add freejoint for dynamic objects.
    if not is_static and not sdf_is_static:
        model_root.add_freejoint(name=f"{model_name}_freejoint")

    # Process links in topological order (parents before children).
    processed: set[str] = set()
    while len(processed) < len(links):
        progress_made = False
        for link_elem in links:
            link_name = link_elem.get("name")
            if link_name in processed:
                continue

            parent_link_name, joint_elem = link_parents.get(link_name, (None, None))

            # Get parent body and its absolute pose.
            if parent_link_name is None:
                # Root link - parent is model root at origin in model frame.
                parent_body = model_root
                parent_abs_pos = [0.0, 0.0, 0.0]
                parent_abs_quat = [1.0, 0.0, 0.0, 0.0]
            elif parent_link_name in link_bodies:
                parent_body = link_bodies[parent_link_name]
                parent_abs_pos, parent_abs_quat = link_absolute_poses[parent_link_name]
            else:
                # Parent not yet processed - skip for now.
                continue

            body_name = f"{model_name}_{link_name}"
            child_abs_pos, child_abs_quat = link_absolute_poses[link_name]

            # Compute child pose relative to parent.
            rel_pos, rel_quat = compute_relative_pose(
                parent_abs_pos, parent_abs_quat, child_abs_pos, child_abs_quat
            )

            body = parent_body.add_body(name=body_name)
            body.pos = rel_pos
            body.quat = rel_quat

            inertial_elem = link_elem.find("inertial")
            if inertial_elem is not None:
                apply_inertial(body, inertial_elem)

            if joint_elem is not None:
                joint_type = joint_elem.get("type", "fixed")
                if joint_type != "fixed":
                    # Parse joint pose (anchor position relative to child body).
                    joint_pose_elem = joint_elem.find("pose")
                    joint_pos, _ = parse_pose(joint_pose_elem)
                    # Pass the child's absolute quaternion for axis transformation.
                    add_joint_from_sdf(
                        body=body,
                        joint_elem=joint_elem,
                        child_abs_quat=child_abs_quat,
                        joint_pos=joint_pos,
                        name_prefix=body_name,
                    )

            for collision_elem in link_elem.findall("collision"):
                add_geom_from_sdf(
                    spec=spec,
                    body=body,
                    geom_elem=collision_elem,
                    sdf_dir=sdf_dir,
                    meshes_dir=meshes_dir,
                    mesh_assets=mesh_assets,
                    texture_assets=texture_assets,
                    color_assets=color_assets,
                    is_collision=True,
                    name_prefix=body_name,
                    room_id=room_id,
                )

            for visual_elem in link_elem.findall("visual"):
                add_geom_from_sdf(
                    spec=spec,
                    body=body,
                    geom_elem=visual_elem,
                    sdf_dir=sdf_dir,
                    meshes_dir=meshes_dir,
                    mesh_assets=mesh_assets,
                    texture_assets=texture_assets,
                    color_assets=color_assets,
                    is_collision=False,
                    name_prefix=body_name,
                    room_id=room_id,
                )

            link_bodies[link_name] = body
            processed.add(link_name)
            progress_made = True

        if not progress_made:
            unprocessed = [
                l.get("name") for l in links if l.get("name") not in processed
            ]
            console_logger.warning(
                f"Could not process all links for {model_name}. "
                f"Unprocessed: {unprocessed}"
            )
            break

    return [f"{model_name}_{l.get('name')}" for l in links]
