"""URDF to SDF converter for articulated objects.

This module converts URDF files (specifically from PartNet-Mobility dataset) to
Drake-compatible SDF format. It handles the quirks of PartNet-Mobility URDFs:
- Missing <inertial> elements (adds defaults)
- Empty 'base' link as root
- Relative mesh paths

Key transformations:
- <robot name> → <sdf><model name>
- <link> → <link> with <pose>
- <joint origin xyz rpy> → joint pose computation
- <joint axis> → <axis><xyz>
- <joint limit> → <axis><limit>
"""

import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import trimesh

from scipy.spatial.transform import Rotation

console_logger = logging.getLogger(__name__)

# Default joint properties.
DEFAULT_JOINT_DAMPING = 0.05  # Nm/(rad/s) for revolute, N/(m/s) for prismatic
DEFAULT_JOINT_FRICTION = 0.05  # Nm for revolute, N for prismatic

SUPPORTED_MESH_EXTENSIONS = {".obj", ".gltf", ".glb"}

from scenesmith.agent_utils.geometry.urdf.link_physics import _parse_geom_origin
from scenesmith.agent_utils.geometry.urdf.models import URDFParseResult
from scenesmith.agent_utils.geometry.urdf.parsing import parse_origin, parse_urdf


def compute_forward_kinematics(
    urdf_result: URDFParseResult,
    joint_positions: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute link transforms at given joint positions.

    Args:
        urdf_result: Parsed URDF result.
        joint_positions: Dict mapping joint names to positions (default: all zeros).

    Returns:
        Dict mapping link names to 4x4 homogeneous transform matrices.
    """
    if joint_positions is None:
        joint_positions = {}

    # Build joint info for quick lookup.
    joint_info = {}  # child_link -> (joint_elem, parent_link)
    for joint_elem in urdf_result.joints.values():
        child_elem = joint_elem.find("child")
        parent_elem = joint_elem.find("parent")
        if child_elem is not None and parent_elem is not None:
            child_link = child_elem.get("link")
            parent_link = parent_elem.get("link")
            if child_link and parent_link:
                joint_info[child_link] = (joint_elem, parent_link)

    # Initialize transforms.
    transforms = {}

    def compute_transform(link_name: str) -> np.ndarray:
        """Recursively compute transform for a link."""
        if link_name in transforms:
            return transforms[link_name]

        if link_name not in joint_info:
            # Root link, identity transform.
            transforms[link_name] = np.eye(4)
            return transforms[link_name]

        joint_elem, parent_link = joint_info[link_name]

        # Get parent transform.
        parent_transform = compute_transform(parent_link)

        # Get joint origin transform.
        origin = joint_elem.find("origin")
        xyz, rpy = parse_origin(origin)

        # Build transform from origin.
        R = Rotation.from_euler("xyz", rpy).as_matrix()
        T_origin = np.eye(4)
        T_origin[:3, :3] = R
        T_origin[:3, 3] = xyz

        # Apply joint motion if applicable.
        joint_type = joint_elem.get("type", "fixed")
        joint_name = joint_elem.get("name", "")
        q = joint_positions.get(joint_name, 0.0)

        T_joint = np.eye(4)
        if joint_type in ("revolute", "continuous") and q != 0.0:
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_xyz = axis_elem.get("xyz", "0 0 1")
                axis = np.array([float(x) for x in axis_xyz.split()])
                axis = axis / np.linalg.norm(axis)
                T_joint[:3, :3] = Rotation.from_rotvec(q * axis).as_matrix()
        elif joint_type == "prismatic" and q != 0.0:
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_xyz = axis_elem.get("xyz", "0 0 1")
                axis = np.array([float(x) for x in axis_xyz.split()])
                axis = axis / np.linalg.norm(axis)
                T_joint[:3, 3] = q * axis

        # Combine transforms.
        transforms[link_name] = parent_transform @ T_origin @ T_joint
        return transforms[link_name]

    # Compute transforms for all links.
    for link_name in urdf_result.links:
        compute_transform(link_name)

    return transforms


def get_link_meshes(
    urdf_path: Path, urdf_result: URDFParseResult
) -> dict[str, list[Path]]:
    """Get mesh file paths for each link.

    Args:
        urdf_path: Path to URDF file.
        urdf_result: Parsed URDF result.

    Returns:
        Dict mapping link names to lists of mesh file paths.
    """
    urdf_dir = urdf_path.parent
    link_meshes = {}

    for link_name, link_elem in urdf_result.links.items():
        meshes = []

        for visual_or_collision in link_elem.findall("visual") + link_elem.findall(
            "collision"
        ):
            geometry = visual_or_collision.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename:
                        mesh_path = urdf_dir / filename
                        if mesh_path.exists() and mesh_path not in meshes:
                            meshes.append(mesh_path)

        if meshes:
            link_meshes[link_name] = meshes

    return link_meshes


def compute_articulated_bounding_box(
    urdf_path: Path, joint_positions: dict[str, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bounding box of articulated model at given joint positions.

    This computes the axis-aligned bounding box of all mesh vertices
    transformed to world space using forward kinematics.

    Args:
        urdf_path: Path to URDF file.
        joint_positions: Dict mapping joint names to positions (default: all zeros).

    Returns:
        Tuple of (min_xyz, max_xyz, center) arrays.
    """
    # Parse URDF.
    urdf_result = parse_urdf(urdf_path)
    urdf_dir = urdf_path.parent

    # Compute link transforms.
    link_transforms = compute_forward_kinematics(urdf_result, joint_positions)

    # Transform all mesh vertices to world space.
    all_verts_world = []
    for link_name, link_elem in urdf_result.links.items():
        if link_name not in link_transforms:
            continue

        T = link_transforms[link_name]

        # Get all visual/collision meshes with their origin transforms.
        for geom_type in ["visual", "collision"]:
            for geom in link_elem.findall(geom_type):
                geometry = geom.find("geometry")
                if geometry is not None:
                    mesh_elem = geometry.find("mesh")
                    if mesh_elem is not None:
                        filename = mesh_elem.get("filename")
                        if filename:
                            mesh_path = urdf_dir / filename
                            if not mesh_path.exists():
                                continue

                            try:
                                mesh = trimesh.load(mesh_path, force="mesh")
                                if isinstance(mesh, trimesh.Scene):
                                    meshes = [
                                        g
                                        for g in mesh.geometry.values()
                                        if isinstance(g, trimesh.Trimesh)
                                    ]
                                    if meshes:
                                        mesh = trimesh.util.concatenate(meshes)
                                    else:
                                        continue

                                # Apply visual/collision origin transform.
                                geom_xyz, geom_rot = _parse_geom_origin(geom)
                                verts = mesh.vertices @ geom_rot.T + geom_xyz

                                # Then apply link transform.
                                verts_homogeneous = np.hstack(
                                    [verts, np.ones((len(verts), 1))]
                                )
                                verts_world = (T @ verts_homogeneous.T).T[:, :3]
                                all_verts_world.extend(verts_world)

                            except Exception as e:
                                console_logger.warning(
                                    f"Failed to load mesh {mesh_path}: {e}"
                                )
                                continue

    if not all_verts_world:
        console_logger.warning("No mesh vertices found for bounding box computation")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    all_verts_world = np.array(all_verts_world)
    min_xyz = np.min(all_verts_world, axis=0)
    max_xyz = np.max(all_verts_world, axis=0)
    center = (min_xyz + max_xyz) / 2

    return min_xyz, max_xyz, center


def compute_sdf_bounding_box(
    sdf_path: Path,
    scale_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bounding box from SDF visual meshes with transforms applied.

    This computes the bounding box by loading GLTF meshes from the SDF's visual
    directory and applying the link pose transforms defined in the SDF. This
    accounts for coordinate frame changes introduced by the GLTF Y-up export.

    Note: The bounding box is computed in model frame (without model-level pose),
    suitable for computing the canonicalization pose.

    Args:
        sdf_path: Path to SDF file.
        scale_factor: Scale factor applied to mesh geometry (from <scale> element).

    Returns:
        Tuple of (min_xyz, max_xyz, center) arrays in model frame.
    """
    sdf_dir = sdf_path.parent
    visual_dir = sdf_dir / "visual"

    if not visual_dir.exists():
        console_logger.warning(f"Visual directory not found: {visual_dir}")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    # Parse SDF to get link poses.
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model = root.find("model")
    if model is None:
        console_logger.warning("No model element found in SDF")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    # First pass: collect raw poses and relative_to info.
    # SDF uses relative_to for pose inheritance, so we need to resolve the chain.
    raw_poses: dict[str, tuple[np.ndarray, np.ndarray, str | None]] = {}

    for link in model.findall("link"):
        link_name = link.get("name", "")
        pose_elem = link.find("pose")

        if pose_elem is not None and pose_elem.text:
            values = [float(v) for v in pose_elem.text.strip().split()]
            xyz = np.array(values[:3])
            rpy = values[3:6]
            rot = Rotation.from_euler("xyz", rpy).as_matrix()
            relative_to = pose_elem.get("relative_to", None)
        else:
            xyz = np.zeros(3)
            rot = np.eye(3)
            relative_to = None

        raw_poses[link_name] = (xyz, rot, relative_to)

    # Second pass: resolve relative_to chain to get model-frame poses.
    link_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def resolve_pose(
        name: str, visited: set | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recursively resolve pose chain to get model-frame pose."""
        if visited is None:
            visited = set()

        if name in link_poses:
            return link_poses[name]

        if name in visited:
            console_logger.warning(f"Circular reference in pose chain: {name}")
            return np.zeros(3), np.eye(3)
        visited.add(name)

        if name not in raw_poses:
            # Unknown link or "base" - return identity (model frame origin).
            return np.zeros(3), np.eye(3)

        xyz, rot, relative_to = raw_poses[name]

        if relative_to is None or relative_to == "__model__" or relative_to == "base":
            # Directly relative to model frame.
            model_xyz, model_rot = xyz, rot
        else:
            # Relative to another link - compose transforms.
            parent_xyz, parent_rot = resolve_pose(relative_to, visited)
            # T_model_link = T_model_parent * T_parent_link
            model_xyz = parent_rot @ xyz + parent_xyz
            model_rot = parent_rot @ rot

        link_poses[name] = (model_xyz, model_rot)
        return model_xyz, model_rot

    for name in raw_poses:
        resolve_pose(name)

    # Load GLTF meshes and apply transforms.
    all_verts_world = []

    for gltf_path in visual_dir.glob("*.gltf"):
        # Extract link name from filename (e.g., "link_0_visual.gltf" -> "link_0").
        link_name = gltf_path.stem.replace("_visual", "")

        if link_name not in link_poses:
            continue

        try:
            mesh = trimesh.load(gltf_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                meshes = [
                    g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    continue

            # R_GF converts: file Y → geometry Z, file Z → geometry -Y
            # This matches Drake's MakeFromOrthonormalColumns(UnitX, UnitZ, -UnitY).
            R_GF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

            # Transform chain: R_GF (Y-up to Z-up) → scale → link pose
            verts = mesh.vertices @ R_GF.T  # Apply Y-up to Z-up first
            verts = verts * scale_factor  # Then scale

            # Apply link pose transform.
            # Note: Link translations from URDF are in original (unscaled) coordinates,
            # so we must scale them to match the scaled mesh geometry.
            xyz, rot = link_poses[link_name]
            xyz_scaled = xyz * scale_factor
            verts_world = verts @ rot.T + xyz_scaled
            all_verts_world.extend(verts_world)

        except Exception as e:
            console_logger.warning(f"Failed to load GLTF {gltf_path}: {e}")
            continue

    if not all_verts_world:
        console_logger.warning("No mesh vertices found for SDF bounding box")
        return np.zeros(3), np.zeros(3), np.zeros(3)

    all_verts_world = np.array(all_verts_world)
    min_xyz = np.min(all_verts_world, axis=0)
    max_xyz = np.max(all_verts_world, axis=0)
    center = (min_xyz + max_xyz) / 2

    return min_xyz, max_xyz, center


def update_sdf_model_pose(
    sdf_path: Path,
    model_pose: tuple[float, float, float, float, float, float],
) -> None:
    """Update the model-level pose in an existing SDF file.

    Args:
        sdf_path: Path to SDF file to update.
        model_pose: New model pose (x, y, z, roll, pitch, yaw).
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model = root.find("model")

    if model is None:
        raise ValueError(f"No model element found in SDF: {sdf_path}")

    # Find or create pose element (should be first child after model name).
    pose_elem = model.find("pose")
    if pose_elem is None:
        # Insert pose as first child.
        pose_elem = ET.Element("pose")
        model.insert(0, pose_elem)

    # Update pose text.
    pose_elem.text = (
        f"{model_pose[0]:.8f} {model_pose[1]:.8f} {model_pose[2]:.8f} "
        f"{model_pose[3]:.8f} {model_pose[4]:.8f} {model_pose[5]:.8f}"
    )

    # Re-indent and write.
    ET.indent(root, space="  ", level=0)
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
    console_logger.info(f"Updated model pose in {sdf_path.name}")
