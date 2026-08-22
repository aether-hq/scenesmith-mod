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
import tempfile
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import trimesh

from scipy.spatial.transform import Rotation

from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionClient
from scenesmith.agent_utils.geometry.mesh_utils import (
    load_mesh_as_trimesh,
    merge_objs_to_gltf,
)
from scenesmith.utils.geometry.sdf_utils import pose_to_string

console_logger = logging.getLogger(__name__)

# Default joint properties.
DEFAULT_JOINT_DAMPING = 0.05  # Nm/(rad/s) for revolute, N/(m/s) for prismatic
DEFAULT_JOINT_FRICTION = 0.05  # Nm for revolute, N for prismatic

SUPPORTED_MESH_EXTENSIONS = {".obj", ".gltf", ".glb"}

from scenesmith.agent_utils.geometry.urdf.parsing import parse_origin


def convert_urdf_geometry_to_sdf(
    urdf_geometry: ET.Element,
    urdf_dir: Path,
    use_gltf: bool = False,
    scale_factor: float = 1.0,
) -> ET.Element | None:
    """Convert URDF geometry element to SDF format.

    Args:
        urdf_geometry: URDF <geometry> element.
        urdf_dir: Directory containing URDF (for resolving relative paths).
        use_gltf: If True, use .gltf extension for mesh files (for visuals).
            If False, use original .obj extension (for collisions).
        scale_factor: Uniform scale factor to apply to geometry.

    Returns:
        SDF <geometry> element or None if geometry is invalid.
    """
    sdf_geometry = ET.Element("geometry")

    mesh = urdf_geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename")
        if filename:
            # Verify mesh exists in URDF directory.
            mesh_path = urdf_dir / filename
            if mesh_path.exists():
                # Use original URDF relative path.
                # Assumes mesh directories (visual/) are copied to SDF location.
                sdf_mesh = ET.SubElement(sdf_geometry, "mesh")
                uri = ET.SubElement(sdf_mesh, "uri")
                # Use GLTF for visuals (Drake Meshcat textures), OBJ for collisions.
                if use_gltf and filename.endswith(".obj"):
                    uri.text = filename[:-4] + ".gltf"
                else:
                    uri.text = filename
                # Add scale element if not 1.0.
                if scale_factor != 1.0:
                    scale_elem = ET.SubElement(sdf_mesh, "scale")
                    scale_elem.text = f"{scale_factor} {scale_factor} {scale_factor}"
                return sdf_geometry

    # Handle primitive shapes (scale dimensions directly).
    box = urdf_geometry.find("box")
    if box is not None:
        size_str = box.get("size", "1 1 1")
        sizes = [float(s) * scale_factor for s in size_str.split()]
        sdf_box = ET.SubElement(sdf_geometry, "box")
        sdf_size = ET.SubElement(sdf_box, "size")
        sdf_size.text = f"{sizes[0]} {sizes[1]} {sizes[2]}"
        return sdf_geometry

    cylinder = urdf_geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.get("radius", "1")) * scale_factor
        length = float(cylinder.get("length", "1")) * scale_factor
        sdf_cylinder = ET.SubElement(sdf_geometry, "cylinder")
        ET.SubElement(sdf_cylinder, "radius").text = str(radius)
        ET.SubElement(sdf_cylinder, "length").text = str(length)
        return sdf_geometry

    sphere = urdf_geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius", "1")) * scale_factor
        sdf_sphere = ET.SubElement(sdf_geometry, "sphere")
        ET.SubElement(sdf_sphere, "radius").text = str(radius)
        return sdf_geometry

    return None


def convert_urdf_visual_to_sdf(
    urdf_visual: ET.Element,
    urdf_dir: Path,
    visual_index: int,
    link_name: str,
    scale_factor: float = 1.0,
) -> ET.Element | None:
    """Convert URDF visual element to SDF format.

    Args:
        urdf_visual: URDF <visual> element.
        urdf_dir: Directory containing URDF.
        visual_index: Index for unique naming.
        link_name: Name of parent link (for unique naming).
        scale_factor: Uniform scale factor to apply to geometry.

    Returns:
        SDF <visual> element or None if invalid.
    """
    # Always use unique name to avoid Drake conflicts with duplicate URDF names.
    name = f"{link_name}_visual_{visual_index}"
    sdf_visual = ET.Element("visual", name=name)

    # Convert origin to pose (scale the position).
    origin = urdf_visual.find("origin")
    xyz, rpy = parse_origin(origin)
    scaled_xyz = [v * scale_factor for v in xyz]
    pose = ET.SubElement(sdf_visual, "pose")
    pose.text = pose_to_string(scaled_xyz, rpy)

    # Convert geometry (use GLTF for visual meshes to support textures in Drake).
    geometry = urdf_visual.find("geometry")
    if geometry is not None:
        sdf_geometry = convert_urdf_geometry_to_sdf(
            geometry, urdf_dir, use_gltf=True, scale_factor=scale_factor
        )
        if sdf_geometry is not None:
            sdf_visual.append(sdf_geometry)
        else:
            return None

    return sdf_visual


def convert_urdf_collision_to_sdf(
    urdf_collision: ET.Element,
    urdf_dir: Path,
    collision_index: int,
    link_name: str,
    friction: float = 0.5,
) -> ET.Element | None:
    """Convert URDF collision element to SDF format.

    Args:
        urdf_collision: URDF <collision> element.
        urdf_dir: Directory containing URDF.
        collision_index: Index for unique naming.
        link_name: Name of parent link (for unique naming).
        friction: Friction coefficient for surface.

    Returns:
        SDF <collision> element or None if invalid.
    """
    # Always use unique name to avoid Drake conflicts with duplicate URDF names.
    name = f"{link_name}_collision_{collision_index}"
    sdf_collision = ET.Element("collision", name=name)

    # Convert origin to pose.
    origin = urdf_collision.find("origin")
    xyz, rpy = parse_origin(origin)
    pose = ET.SubElement(sdf_collision, "pose")
    pose.text = pose_to_string(xyz, rpy)

    # Convert geometry.
    geometry = urdf_collision.find("geometry")
    if geometry is not None:
        sdf_geometry = convert_urdf_geometry_to_sdf(geometry, urdf_dir)
        if sdf_geometry is not None:
            sdf_collision.append(sdf_geometry)
        else:
            return None

    # Add surface friction.
    surface = ET.SubElement(sdf_collision, "surface")
    friction_elem = ET.SubElement(surface, "friction")
    ode = ET.SubElement(friction_elem, "ode")
    ET.SubElement(ode, "mu").text = f"{friction:.3f}"
    ET.SubElement(ode, "mu2").text = f"{friction:.3f}"

    return sdf_collision


def merge_link_visual_meshes_for_sdf(
    urdf_link: ET.Element, urdf_dir: Path, sdf_dir: Path, link_name: str
) -> Path | None:
    """Merge all visual meshes for a link into a single GLTF file.

    Args:
        urdf_link: URDF <link> element.
        urdf_dir: Directory containing URDF (for resolving mesh paths).
        sdf_dir: Directory where SDF will be written (for saving merged GLTF).
        link_name: Name of the link.

    Returns:
        Path to merged GLTF file, or None if no valid meshes.
    """
    # Collect (OBJ path, origin offset) pairs from visual elements.
    obj_paths_with_offsets: list[tuple[Path, tuple[float, float, float]]] = []
    for visual in urdf_link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is None:
            continue

        mesh = geometry.find("mesh")
        if mesh is None:
            continue

        filename = mesh.get("filename")
        if not filename or not filename.endswith(".obj"):
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

        obj_paths_with_offsets.append((mesh_path, xyz))

    if not obj_paths_with_offsets:
        return None

    # Output path for merged GLTF.
    output_dir = sdf_dir / "visual"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_gltf = output_dir / f"{link_name}_visual.gltf"

    try:
        merge_objs_to_gltf(
            obj_paths_with_offsets=obj_paths_with_offsets,
            output_path=output_gltf,
        )
        console_logger.info(
            f"Merged {len(obj_paths_with_offsets)} visual meshes for {link_name}"
        )
        return output_gltf
    except Exception as e:
        console_logger.warning(f"Failed to merge visual meshes for {link_name}: {e}")
        return None


def generate_link_collision_geometry_for_sdf(
    urdf_link: ET.Element,
    urdf_dir: Path,
    sdf_dir: Path,
    link_name: str,
    collision_client: ConvexDecompositionClient,
    collision_threshold: float = 0.05,
) -> list[Path]:
    """Generate convex collision geometry for a link using CoACD.

    Combines all visual meshes for the link, runs CoACD convex decomposition
    via the convex decomposition server, and saves the resulting convex pieces
    as OBJ files. Note: This function always uses CoACD since articulated assets
    benefit from simpler collision geometry.

    Args:
        urdf_link: URDF <link> element.
        urdf_dir: Directory containing URDF (for resolving mesh paths).
        sdf_dir: Directory where SDF will be written (for saving collision meshes).
        link_name: Name of the link.
        collision_client: Convex decomposition client for collision geometry.
        collision_threshold: CoACD approximation threshold (0.01-0.1 typical).

    Returns:
        List of paths to generated collision OBJ files.
    """
    # Collect visual mesh files and their origins.
    mesh_infos: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for visual in urdf_link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is not None:
            mesh = geometry.find("mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if filename:
                    mesh_path = urdf_dir / filename
                    if mesh_path.exists():
                        origin = visual.find("origin")
                        xyz, rpy = parse_origin(origin)
                        rot = Rotation.from_euler("xyz", rpy).as_matrix()
                        mesh_infos.append((mesh_path, xyz, rot))

    if not mesh_infos:
        return []

    # Load and combine meshes, applying visual origins.
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0

    for mesh_path, visual_xyz, visual_rot in mesh_infos:
        try:
            mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)
            if mesh is not None:
                # Apply visual origin transform to vertices.
                vertices = mesh.vertices @ visual_rot.T + visual_xyz
                combined_vertices.append(vertices)
                combined_faces.append(mesh.faces + vertex_offset)
                vertex_offset += len(mesh.vertices)
        except Exception as e:
            console_logger.warning(f"Failed to load mesh {mesh_path}: {e}")

    if not combined_vertices:
        return []

    # Create combined mesh.
    all_vertices = np.vstack(combined_vertices)
    all_faces = np.vstack(combined_faces)
    combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    # Generate convex decomposition using convex decomposition server.
    # Save mesh to temp file since client requires a file path.
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            temp_mesh_path = Path(f.name)
        combined_mesh.export(temp_mesh_path)

        try:
            convex_pieces = collision_client.generate_collision_geometry(
                mesh_path=temp_mesh_path,
                method="coacd",
                threshold=collision_threshold,
            )
        finally:
            # Clean up temp file.
            temp_mesh_path.unlink(missing_ok=True)
    except Exception as e:
        console_logger.warning(
            f"Convex decomposition failed for {link_name}, falling back to convex "
            f"hull: {e}"
        )
        # Fallback to single convex hull.
        convex_pieces = [combined_mesh.convex_hull]

    # Save collision pieces as OBJ files.
    collision_dir = sdf_dir / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)

    collision_paths = []
    for i, piece in enumerate(convex_pieces):
        collision_path = collision_dir / f"{link_name}_collision_{i}.obj"
        piece.export(collision_path)
        collision_paths.append(collision_path)

    console_logger.debug(
        f"Generated {len(collision_paths)} collision pieces for {link_name}"
    )
    return collision_paths


def create_sdf_collision_elements_from_paths(
    collision_paths: list[Path],
    sdf_dir: Path,
    link_name: str,
    friction: float = 0.5,
    scale_factor: float = 1.0,
) -> list[ET.Element]:
    """Create SDF collision elements from generated collision mesh paths.

    Args:
        collision_paths: Paths to collision OBJ files.
        sdf_dir: Directory where SDF will be written (for relative paths).
        link_name: Name of parent link (for unique naming).
        friction: Friction coefficient for surfaces.
        scale_factor: Uniform scale factor to apply to collision geometry.

    Returns:
        List of SDF <collision> elements.
    """
    collision_elements = []
    for i, collision_path in enumerate(collision_paths):
        name = f"{link_name}_collision_{i}"
        sdf_collision = ET.Element("collision", name=name)

        # Identity pose (collision geometry already in link frame).
        pose = ET.SubElement(sdf_collision, "pose")
        pose.text = "0 0 0 0 0 0"

        # Geometry with mesh reference.
        sdf_geometry = ET.SubElement(sdf_collision, "geometry")
        sdf_mesh = ET.SubElement(sdf_geometry, "mesh")
        uri = ET.SubElement(sdf_mesh, "uri")
        rel_path = collision_path.relative_to(sdf_dir)
        uri.text = str(rel_path)
        # Add scale element if not 1.0.
        if scale_factor != 1.0:
            scale_elem = ET.SubElement(sdf_mesh, "scale")
            scale_elem.text = f"{scale_factor} {scale_factor} {scale_factor}"

        # Add surface friction.
        surface = ET.SubElement(sdf_collision, "surface")
        friction_elem = ET.SubElement(surface, "friction")
        ode = ET.SubElement(friction_elem, "ode")
        ET.SubElement(ode, "mu").text = f"{friction:.3f}"
        ET.SubElement(ode, "mu2").text = f"{friction:.3f}"

        collision_elements.append(sdf_collision)

    return collision_elements
