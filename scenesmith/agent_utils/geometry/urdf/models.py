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

from dataclasses import dataclass
from pathlib import Path

console_logger = logging.getLogger(__name__)

# Default joint properties.
DEFAULT_JOINT_DAMPING = 0.05  # Nm/(rad/s) for revolute, N/(m/s) for prismatic
DEFAULT_JOINT_FRICTION = 0.05  # Nm for revolute, N for prismatic

SUPPORTED_MESH_EXTENSIONS = {".obj", ".gltf", ".glb"}


class LinkPhysics:
    """Physics properties for a link."""

    mass: float
    inertia_ixx: float
    inertia_iyy: float
    inertia_izz: float
    inertia_ixy: float = 0.0
    inertia_ixz: float = 0.0
    inertia_iyz: float = 0.0
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class URDFParseResult:
    """Result of parsing a URDF file."""

    robot_name: str
    """Name of the robot from the URDF."""

    links: dict[str, ET.Element]
    """Mapping of link name to link element."""

    joints: dict[str, ET.Element]
    """Mapping of joint name to joint element."""

    parent_map: dict[str, str]
    """Mapping of child link to parent link."""

    root_link: str | None
    """Name of root link (no parent)."""


@dataclass
class URDFLinkMeshInfo:
    """Mesh information for a single URDF link."""

    link_name: str
    """Name of the link."""

    mesh_paths: list[Path]
    """Paths to mesh files (OBJ, GLTF, GLB) for visual geometry."""

    origins: list[tuple[float, float, float]]
    """Origin offsets for each mesh file (xyz in meters)."""

    world_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World position of the link from joint chain (xyz in meters)."""

    world_rotation: tuple[tuple[float, ...], ...] | None = None
    """World rotation matrix of the link from joint chain (3x3)."""
