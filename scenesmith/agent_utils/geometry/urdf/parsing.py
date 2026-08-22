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

console_logger = logging.getLogger(__name__)

# Default joint properties.
DEFAULT_JOINT_DAMPING = 0.05  # Nm/(rad/s) for revolute, N/(m/s) for prismatic
DEFAULT_JOINT_FRICTION = 0.05  # Nm for revolute, N for prismatic

SUPPORTED_MESH_EXTENSIONS = {".obj", ".gltf", ".glb"}

from scenesmith.agent_utils.geometry.urdf.models import (
    URDFLinkMeshInfo,
    URDFParseResult,
)


def parse_origin(origin_elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Parse URDF origin element into position and RPY.

    Args:
        origin_elem: URDF <origin> element or None.

    Returns:
        Tuple of (xyz position array, rpy rotation array in radians).
    """
    if origin_elem is None:
        return np.zeros(3), np.zeros(3)

    xyz_str = origin_elem.get("xyz", "0 0 0")
    rpy_str = origin_elem.get("rpy", "0 0 0")

    xyz = np.array([float(x) for x in xyz_str.split()])
    rpy = np.array([float(x) for x in rpy_str.split()])

    return xyz, rpy


def parse_urdf(urdf_path: Path) -> URDFParseResult:
    """Parse URDF file and extract structure.

    Args:
        urdf_path: Path to URDF file.

    Returns:
        URDFParseResult with parsed data.

    Raises:
        ValueError: If URDF structure is invalid.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if root.tag != "robot":
        raise ValueError(f"Expected <robot> root element, got <{root.tag}>")

    robot_name = root.get("name", "unnamed_robot")

    # Collect links.
    links = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        if link_name:
            links[link_name] = link_elem

    # Collect joints and build parent map.
    joints = {}
    parent_map = {}
    child_links = set()

    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.get("name")
        if joint_name:
            joints[joint_name] = joint_elem

            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")

            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.get("link")
                child_link = child_elem.get("link")

                if parent_link and child_link:
                    parent_map[child_link] = parent_link
                    child_links.add(child_link)

    # Find root link (link with no parent).
    root_link = None
    for link_name in links:
        if link_name not in child_links:
            root_link = link_name
            break

    return URDFParseResult(
        robot_name=robot_name,
        links=links,
        joints=joints,
        parent_map=parent_map,
        root_link=root_link,
    )


def extract_link_meshes(urdf_path: Path) -> list[URDFLinkMeshInfo]:
    """Extract link-to-mesh mappings from URDF with position offsets.

    Parses a URDF file and extracts the visual mesh files for each link,
    along with their origin offsets and position offsets from the joint chain.
    Supports OBJ, GLTF, and GLB formats.

    Args:
        urdf_path: Path to URDF file.

    Returns:
        List of URDFLinkMeshInfo with link names, mesh paths, and position offsets.
    """
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    urdf_dir = urdf_path.parent

    # Build link-to-joint mapping for kinematic chain traversal.
    link_to_parent_joint: dict[str, ET.Element] = {}
    for joint in robot.findall("joint"):
        child = joint.find("child")
        if child is not None:
            child_link = child.get("link")
            if child_link:
                link_to_parent_joint[child_link] = joint

    def get_link_visual_position(link_name: str) -> np.ndarray:
        """Compute position offset for VLM visualization.

        For VLM visualization, we only accumulate position offsets (xyz) from
        joints, NOT coordinate system rotations (rpy). The rotations are for
        simulation coordinate transforms, not visual assembly.
        """
        # Build chain from link to root.
        chain = []
        current_link = link_name
        while current_link in link_to_parent_joint:
            joint = link_to_parent_joint[current_link]
            parent = joint.find("parent")
            if parent is None:
                break
            chain.append(joint)
            current_link = parent.get("link", "")

        # Accumulate position offsets only (ignore rotations for visualization).
        world_pos = np.zeros(3)
        for joint in reversed(chain):
            origin = joint.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                joint_xyz = np.array([float(v) for v in xyz_str.split()])
                world_pos = world_pos + joint_xyz

        return world_pos

    link_meshes = []

    for link in robot.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue

        mesh_paths = []
        origins = []

        for visual in link.findall("visual"):
            mesh = visual.find(".//mesh")
            if mesh is None:
                continue

            filename = mesh.get("filename")
            if not filename:
                continue

            # Check if file extension is supported.
            file_ext = Path(filename).suffix.lower()
            if file_ext not in SUPPORTED_MESH_EXTENSIONS:
                continue

            mesh_path = urdf_dir / filename
            if not mesh_path.exists():
                continue

            # Parse origin offset.
            origin = visual.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                xyz = tuple(float(v) for v in xyz_str.split())
            else:
                xyz = (0.0, 0.0, 0.0)

            mesh_paths.append(mesh_path)
            origins.append(xyz)

        # Only include links with visual geometry.
        if mesh_paths:
            # Compute position offset from joint chain (no rotation for VLM).
            world_pos = get_link_visual_position(link_name)

            link_meshes.append(
                URDFLinkMeshInfo(
                    link_name=link_name,
                    mesh_paths=mesh_paths,
                    origins=origins,
                    world_position=tuple(world_pos),
                    world_rotation=None,  # No rotation for VLM visualization.
                )
            )

    return link_meshes


def validate_urdf_meshes(
    urdf_path: Path, urdf_result: URDFParseResult
) -> tuple[list[str], list[str]]:
    """Validate that all mesh files referenced in URDF exist.

    Args:
        urdf_path: Path to URDF file.
        urdf_result: Parsed URDF result.

    Returns:
        Tuple of (list of valid mesh paths, list of missing mesh paths).
    """
    urdf_dir = urdf_path.parent
    valid_meshes = []
    missing_meshes = []

    for link_elem in urdf_result.links.values():
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
                        if mesh_path.exists():
                            if filename not in valid_meshes:
                                valid_meshes.append(filename)
                        else:
                            if filename not in missing_meshes:
                                missing_meshes.append(filename)

    return valid_meshes, missing_meshes


def repair_urdf_missing_meshes(
    urdf_path: Path, output_path: Path
) -> tuple[Path, list[str]]:
    """Repair URDF by removing references to missing mesh files.

    Args:
        urdf_path: Path to input URDF file.
        output_path: Path to write repaired URDF.

    Returns:
        Tuple of (path to repaired URDF, list of removed mesh references).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    removed_meshes = []

    for link_elem in root.findall("link"):
        elements_to_remove = []

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
                        if not mesh_path.exists():
                            elements_to_remove.append(visual_or_collision)
                            if filename not in removed_meshes:
                                removed_meshes.append(filename)

        for elem in elements_to_remove:
            link_elem.remove(elem)

    # Write repaired URDF.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return output_path, removed_meshes
