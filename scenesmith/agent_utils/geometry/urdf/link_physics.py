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

from scenesmith.agent_utils.geometry.mesh_utils import load_mesh_as_trimesh
from scenesmith.agent_utils.physics.physics_tools import compute_inertia_from_mesh

console_logger = logging.getLogger(__name__)

# Default joint properties.
DEFAULT_JOINT_DAMPING = 0.05  # Nm/(rad/s) for revolute, N/(m/s) for prismatic
DEFAULT_JOINT_FRICTION = 0.05  # Nm for revolute, N for prismatic

SUPPORTED_MESH_EXTENSIONS = {".obj", ".gltf", ".glb"}

from scenesmith.agent_utils.geometry.urdf.models import LinkPhysics
from scenesmith.agent_utils.geometry.urdf.parsing import parse_urdf


def _parse_geom_origin(geom_elem: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """Parse geometry origin into translation and rotation matrix."""
    origin = geom_elem.find("origin")
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz_str = origin.get("xyz", "0 0 0")
    xyz = np.array([float(v) for v in xyz_str.split()])
    rpy_str = origin.get("rpy", "0 0 0")
    rpy = np.array([float(v) for v in rpy_str.split()])
    rot = Rotation.from_euler("xyz", rpy).as_matrix()
    return xyz, rot


def compute_link_physics_from_meshes(
    urdf_path: Path, link_masses: dict[str, float]
) -> dict[str, LinkPhysics]:
    """Compute physics properties for each link from mesh geometry and given masses.

    This function loads all meshes for each link, combines them into a single mesh,
    and computes inertial properties using the specified mass.

    Args:
        urdf_path: Path to URDF file.
        link_masses: Dict mapping link names to masses in kg.

    Returns:
        Dict mapping link names to LinkPhysics objects.
    """
    urdf_result = parse_urdf(urdf_path)
    urdf_dir = urdf_path.parent

    physics_dict = {}
    for link_name, link_elem in urdf_result.links.items():
        if link_name not in link_masses:
            # Skip links without mass (e.g., 'base' link with no geometry).
            continue
        mass = link_masses[link_name]

        try:
            # Collect mesh files and their geometry origins.
            mesh_infos: list[tuple[Path, np.ndarray, np.ndarray]] = []
            for geom_type in ["visual", "collision"]:
                for geom in link_elem.findall(geom_type):
                    geometry = geom.find("geometry")
                    if geometry is not None:
                        mesh = geometry.find("mesh")
                        if mesh is not None:
                            filename = mesh.get("filename")
                            if filename:
                                mesh_path = urdf_dir / filename
                                if mesh_path.exists():
                                    xyz, rot = _parse_geom_origin(geom)
                                    # Avoid duplicates (same mesh in visual and collision).
                                    if not any(
                                        p == mesh_path for p, _, _ in mesh_infos
                                    ):
                                        mesh_infos.append((mesh_path, xyz, rot))

            if not mesh_infos:
                continue

            # Load and combine all meshes, applying geometry origins.
            combined_vertices = []
            combined_faces = []
            vertex_offset = 0

            for mesh_path, geom_xyz, geom_rot in mesh_infos:
                mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)
                if mesh is not None:
                    # Apply geometry origin transform.
                    vertices = mesh.vertices @ geom_rot.T + geom_xyz
                    combined_vertices.append(vertices)
                    combined_faces.append(mesh.faces + vertex_offset)
                    vertex_offset += len(vertices)

            if not combined_vertices:
                continue

            # Create combined mesh.
            all_vertices = np.vstack(combined_vertices)
            all_faces = np.vstack(combined_faces)
            combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

            # Compute inertia.
            inertial = compute_inertia_from_mesh(combined_mesh, mass)

            # Convert to LinkPhysics.
            if inertial.inertia_tensor is None:
                raise ValueError(
                    f"Inertia computation failed for link '{link_name}'. "
                    f"Mesh may have invalid geometry."
                )

            physics_dict[link_name] = LinkPhysics(
                mass=inertial.mass,
                inertia_ixx=inertial.inertia_tensor[0, 0],
                inertia_iyy=inertial.inertia_tensor[1, 1],
                inertia_izz=inertial.inertia_tensor[2, 2],
                inertia_ixy=inertial.inertia_tensor[0, 1],
                inertia_ixz=inertial.inertia_tensor[0, 2],
                inertia_iyz=inertial.inertia_tensor[1, 2],
                center_of_mass=tuple(inertial.center_of_mass),
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute physics for link '{link_name}': {e}"
            ) from e

    return physics_dict
