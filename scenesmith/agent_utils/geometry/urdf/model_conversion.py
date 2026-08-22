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

from scenesmith.agent_utils.convex_decomposition_server import ConvexDecompositionClient
from scenesmith.utils.geometry.inertia_utils import fix_sdf_file_inertia
from scenesmith.utils.geometry.sdf_utils import pose_to_string

console_logger = logging.getLogger(__name__)

# Default joint properties.
DEFAULT_JOINT_DAMPING = 0.05  # Nm/(rad/s) for revolute, N/(m/s) for prismatic
DEFAULT_JOINT_FRICTION = 0.05  # Nm for revolute, N for prismatic

SUPPORTED_MESH_EXTENSIONS = {".obj", ".gltf", ".glb"}

from scenesmith.agent_utils.geometry.urdf.geometry_conversion import (
    convert_urdf_collision_to_sdf,
    convert_urdf_visual_to_sdf,
    create_sdf_collision_elements_from_paths,
    generate_link_collision_geometry_for_sdf,
    merge_link_visual_meshes_for_sdf,
)
from scenesmith.agent_utils.geometry.urdf.models import LinkPhysics
from scenesmith.agent_utils.geometry.urdf.parsing import (
    parse_origin,
    parse_urdf,
    repair_urdf_missing_meshes,
    validate_urdf_meshes,
)


def convert_urdf_link_to_sdf(
    urdf_link: ET.Element,
    urdf_dir: Path,
    sdf_dir: Path,
    physics: LinkPhysics | None = None,
    friction: float = 0.5,
    link_pose: tuple[list[float], list[float], str] | None = None,
    generated_collision_paths: list[Path] | None = None,
    merged_visual_path: Path | None = None,
    scale_factor: float = 1.0,
) -> ET.Element:
    """Convert URDF link element to SDF format.

    Args:
        urdf_link: URDF <link> element.
        urdf_dir: Directory containing URDF.
        sdf_dir: Directory where SDF will be written.
        physics: Physics properties for the link.
        friction: Friction coefficient for collisions.
        link_pose: Optional tuple of (xyz, rpy, relative_to_frame) for link pose.
            In URDF, child links are positioned by joint origins. In SDF, we need
            to explicitly set link poses relative to parent links.
        generated_collision_paths: Optional list of paths to pre-generated convex
            decomposition collision meshes. If provided, these are used instead
            of converting URDF collision elements.
        merged_visual_path: Optional path to pre-merged GLTF visual mesh. If
            provided, a single visual element is created using this mesh instead
            of converting individual URDF visual elements.
        scale_factor: Uniform scale factor to apply to geometry and positions.

    Returns:
        SDF <link> element.
    """
    link_name = urdf_link.get("name", "unnamed_link")
    sdf_link = ET.Element("link", name=link_name)

    # Set link pose (from joint origin in URDF semantics). Scale positions.
    if link_pose is not None:
        xyz, rpy, relative_to = link_pose
        scaled_xyz = [v * scale_factor for v in xyz]
        pose = ET.SubElement(sdf_link, "pose")
        pose.set("relative_to", relative_to)
        pose.text = pose_to_string(scaled_xyz, rpy)

    # Check if link has any geometry (visual or collision).
    has_geometry = bool(urdf_link.findall("visual") or urdf_link.findall("collision"))

    # Add inertial properties.
    # Mass is from VLM (already appropriate for target scale).
    # Center of mass and inertia are computed from mesh geometry (need scaling).
    if physics is not None:
        inertial = ET.SubElement(sdf_link, "inertial")
        ET.SubElement(inertial, "mass").text = f"{physics.mass:.6f}"

        # Center of mass pose (scale position, keep orientation).
        com_pose = ET.SubElement(inertial, "pose")
        com = physics.center_of_mass
        scaled_com = [v * scale_factor for v in com]
        com_pose.text = (
            f"{scaled_com[0]:.6f} {scaled_com[1]:.6f} {scaled_com[2]:.6f} 0 0 0"
        )

        # Inertia tensor (scales as scale_factor^2 for same mass at smaller distances).
        inertia_scale = scale_factor * scale_factor
        inertia = ET.SubElement(inertial, "inertia")
        ET.SubElement(inertia, "ixx").text = (
            f"{physics.inertia_ixx * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "iyy").text = (
            f"{physics.inertia_iyy * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "izz").text = (
            f"{physics.inertia_izz * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "ixy").text = (
            f"{physics.inertia_ixy * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "ixz").text = (
            f"{physics.inertia_ixz * inertia_scale:.6e}"
        )
        ET.SubElement(inertia, "iyz").text = (
            f"{physics.inertia_iyz * inertia_scale:.6e}"
        )
    elif has_geometry:
        raise ValueError(
            f"Link '{link_name}' has geometry but no physics properties provided. "
            f"VLM analysis should provide physics for all links with geometry."
        )
    else:
        # Add explicit zero inertia for massless links (e.g., 'base' anchor frame).
        # Without this, libsdformat assigns default mass=1kg which affects physics.
        inertial = ET.SubElement(sdf_link, "inertial")
        ET.SubElement(inertial, "mass").text = "0.0"
        inertia = ET.SubElement(inertial, "inertia")
        for comp in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]:
            ET.SubElement(inertia, comp).text = "0.0"

    # Convert visual elements.
    # Use merged visual path if provided, else convert from URDF.
    if merged_visual_path:
        # Create single visual element for merged GLTF.
        sdf_visual = ET.Element("visual", name=f"{link_name}_visual")
        pose = ET.SubElement(sdf_visual, "pose")
        pose.text = "0 0 0 0 0 0"  # Identity pose (offsets baked into GLTF).
        sdf_geometry = ET.SubElement(sdf_visual, "geometry")
        sdf_mesh = ET.SubElement(sdf_geometry, "mesh")
        uri = ET.SubElement(sdf_mesh, "uri")
        rel_path = merged_visual_path.relative_to(sdf_dir)
        uri.text = str(rel_path)
        # Add scale element if not 1.0.
        if scale_factor != 1.0:
            scale_elem = ET.SubElement(sdf_mesh, "scale")
            scale_elem.text = f"{scale_factor} {scale_factor} {scale_factor}"
        sdf_link.append(sdf_visual)
    else:
        for i, urdf_visual in enumerate(urdf_link.findall("visual")):
            sdf_visual = convert_urdf_visual_to_sdf(
                urdf_visual, urdf_dir, i, link_name, scale_factor=scale_factor
            )
            if sdf_visual is not None:
                sdf_link.append(sdf_visual)

    # Convert collision elements.
    # Use generated collision paths if provided, else convert from URDF.
    if generated_collision_paths:
        collision_elements = create_sdf_collision_elements_from_paths(
            collision_paths=generated_collision_paths,
            sdf_dir=sdf_dir,
            link_name=link_name,
            friction=friction,
            scale_factor=scale_factor,
        )
        for sdf_collision in collision_elements:
            sdf_link.append(sdf_collision)
    else:
        for i, urdf_collision in enumerate(urdf_link.findall("collision")):
            sdf_collision = convert_urdf_collision_to_sdf(
                urdf_collision, urdf_dir, i, link_name, friction
            )
            if sdf_collision is not None:
                sdf_link.append(sdf_collision)

    return sdf_link


def convert_urdf_joint_to_sdf(
    urdf_joint: ET.Element, include_dynamics: bool = True
) -> ET.Element:
    """Convert URDF joint element to SDF format.

    In URDF, joint origin positions the child link relative to parent. In SDF,
    we handle this by setting link poses (relative_to parent link) in
    convert_urdf_link_to_sdf. The joint pose defaults to identity relative to
    child, which is correct since in URDF the child link frame coincides with
    the joint frame at q=0.

    Args:
        urdf_joint: URDF <joint> element.
        include_dynamics: Whether to include damping/friction dynamics.

    Returns:
        SDF <joint> element.
    """
    joint_name = urdf_joint.get("name", "unnamed_joint")
    joint_type = urdf_joint.get("type", "fixed")

    # Map URDF joint types to SDF.
    type_map = {
        "revolute": "revolute",
        "continuous": "revolute",  # SDF uses revolute with no limits.
        "prismatic": "prismatic",
        "fixed": "fixed",
        "floating": "ball",  # Approximate.
        "planar": "universal",  # Approximate.
    }
    sdf_type = type_map.get(joint_type, "fixed")

    sdf_joint = ET.Element("joint", name=joint_name, type=sdf_type)

    # Parent and child.
    parent_elem = urdf_joint.find("parent")
    child_elem = urdf_joint.find("child")

    if parent_elem is not None:
        parent = ET.SubElement(sdf_joint, "parent")
        parent.text = parent_elem.get("link", "")

    if child_elem is not None:
        child = ET.SubElement(sdf_joint, "child")
        child.text = child_elem.get("link", "")

    # Axis (for revolute/prismatic).
    if sdf_type in ("revolute", "prismatic"):
        axis_elem = urdf_joint.find("axis")
        sdf_axis = ET.SubElement(sdf_joint, "axis")

        if axis_elem is not None:
            axis_xyz = axis_elem.get("xyz", "0 0 1")
        else:
            axis_xyz = "0 0 1"  # Default Z-axis.

        xyz_elem = ET.SubElement(sdf_axis, "xyz")
        xyz_elem.text = axis_xyz

        # Limits.
        limit_elem = urdf_joint.find("limit")
        if limit_elem is not None:
            sdf_limit = ET.SubElement(sdf_axis, "limit")

            lower = limit_elem.get("lower")
            upper = limit_elem.get("upper")
            effort = limit_elem.get("effort")
            velocity = limit_elem.get("velocity")

            if lower is not None:
                ET.SubElement(sdf_limit, "lower").text = lower
            if upper is not None:
                ET.SubElement(sdf_limit, "upper").text = upper
            if effort is not None:
                ET.SubElement(sdf_limit, "effort").text = effort
            if velocity is not None:
                ET.SubElement(sdf_limit, "velocity").text = velocity
        elif joint_type == "continuous":
            # No limits for continuous joints.
            pass

        # Dynamics (damping, friction).
        if include_dynamics:
            dynamics = ET.SubElement(sdf_axis, "dynamics")
            ET.SubElement(dynamics, "damping").text = f"{DEFAULT_JOINT_DAMPING}"
            ET.SubElement(dynamics, "friction").text = f"{DEFAULT_JOINT_FRICTION}"

    return sdf_joint


def convert_urdf_to_sdf(
    urdf_path: Path,
    output_path: Path,
    link_physics: dict[str, LinkPhysics] | None = None,
    link_friction: dict[str, float] | None = None,
    model_name: str | None = None,
    repair_missing_meshes: bool = True,
    model_pose: tuple[float, float, float, float, float, float] | None = None,
    generate_collision: bool = False,
    collision_client: ConvexDecompositionClient | None = None,
    collision_threshold: float = 0.05,
    merge_visuals: bool = False,
    scale_factor: float = 1.0,
) -> Path:
    """Convert URDF file to Drake-compatible SDF format.

    Args:
        urdf_path: Path to input URDF file.
        output_path: Path to output SDF file.
        link_physics: Optional dict mapping link names to physics properties.
        link_friction: Optional dict mapping link names to friction coefficients.
        model_name: Optional model name (defaults to URDF robot name).
        repair_missing_meshes: Whether to remove references to missing mesh files.
        model_pose: Optional model-level pose (x, y, z, roll, pitch, yaw) for
            canonicalization. Applied to the entire model as a single transform.
            Use this instead of transforming individual mesh poses.
        generate_collision: Whether to generate convex collision geometry using
            CoACD. If True, collision meshes are generated from visual geometry
            and saved to the output directory. If False, existing URDF collision
            elements are converted as-is.
        collision_client: Convex decomposition client for collision geometry
            generation. Required when generate_collision is True.
        collision_threshold: CoACD approximation threshold (0.01-0.1 typical).
            Lower values produce more convex pieces with higher fidelity.
        merge_visuals: Whether to merge all visual meshes per link into a single
            GLTF file. Reduces draw calls and simplifies file structure.
        scale_factor: Uniform scale factor to apply to all geometry and positions.
            Used to correct unit conversion issues (e.g., 0.1 to convert from
            decimeters to meters).

    Returns:
        Path to generated SDF file.

    Raises:
        FileNotFoundError: If URDF file doesn't exist.
        ValueError: If URDF structure is invalid or generate_collision is True
            but collision_client is None.
    """
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    if generate_collision and collision_client is None:
        raise ValueError(
            "collision_client is required when generate_collision is True. "
            "Start a convex decomposition server and pass the client."
        )

    urdf_dir = urdf_path.parent
    sdf_dir = output_path.parent

    # Parse URDF.
    urdf_result = parse_urdf(urdf_path)

    # Validate and optionally repair missing meshes.
    _, missing_meshes = validate_urdf_meshes(urdf_path, urdf_result)

    if missing_meshes:
        console_logger.warning(
            f"URDF '{urdf_path.name}' references {len(missing_meshes)} missing mesh "
            f"files: {missing_meshes[:5]}{'...' if len(missing_meshes) > 5 else ''}"
        )

        if repair_missing_meshes:
            # Create repaired URDF in temp location and re-parse.
            repaired_path = output_path.parent / f"{urdf_path.stem}_repaired.urdf"
            repaired_path, removed = repair_urdf_missing_meshes(
                urdf_path, repaired_path
            )
            console_logger.info(
                f"Repaired URDF by removing {len(removed)} missing mesh references"
            )
            urdf_result = parse_urdf(repaired_path)

    # Create SDF structure.
    sdf = ET.Element("sdf", version="1.7")
    model = ET.SubElement(sdf, "model", name=model_name or urdf_result.robot_name)

    # Add model-level pose for canonicalization (applied to entire model).
    # Scale position components but not rotation.
    if model_pose is not None:
        scaled_pose = (
            model_pose[0] * scale_factor,
            model_pose[1] * scale_factor,
            model_pose[2] * scale_factor,
            model_pose[3],
            model_pose[4],
            model_pose[5],
        )
        pose_elem = ET.SubElement(model, "pose")
        pose_elem.text = (
            f"{scaled_pose[0]:.8f} {scaled_pose[1]:.8f} {scaled_pose[2]:.8f} "
            f"{scaled_pose[3]:.8f} {scaled_pose[4]:.8f} {scaled_pose[5]:.8f}"
        )

    # Build mapping of child_link -> (parent_link, joint_origin).
    # In URDF, joint origin positions the child relative to parent.
    # In SDF, we express this as link pose relative to parent.
    link_poses: dict[str, tuple[list[float], list[float], str]] = {}
    for urdf_joint in urdf_result.joints.values():
        child_elem = urdf_joint.find("child")
        parent_elem = urdf_joint.find("parent")
        if child_elem is not None and parent_elem is not None:
            child_link = child_elem.get("link")
            parent_link = parent_elem.get("link")
            if child_link and parent_link:
                origin = urdf_joint.find("origin")
                xyz, rpy = parse_origin(origin)
                link_poses[child_link] = (xyz, rpy, parent_link)

    # Generate collision geometry for all links if requested.
    # This is done before link conversion to have collision paths available.
    link_collision_paths: dict[str, list[Path]] = {}
    if generate_collision:
        console_logger.info(
            f"Generating collision geometry (threshold={collision_threshold})"
        )
        for link_name, urdf_link in urdf_result.links.items():
            # Only generate for links with visual geometry.
            if urdf_link.findall("visual"):
                collision_paths = generate_link_collision_geometry_for_sdf(
                    urdf_link=urdf_link,
                    urdf_dir=urdf_dir,
                    sdf_dir=sdf_dir,
                    link_name=link_name,
                    collision_client=collision_client,  # type: ignore[arg-type]
                    collision_threshold=collision_threshold,
                )
                if collision_paths:
                    link_collision_paths[link_name] = collision_paths
                    console_logger.info(
                        f"Generated {len(collision_paths)} collision meshes "
                        f"for link '{link_name}'"
                    )

    # Merge visual meshes for all links if requested.
    # This is done before link conversion to have merged paths available.
    link_merged_visuals: dict[str, Path] = {}
    if merge_visuals:
        console_logger.info("Merging visual meshes per link")
        for link_name, urdf_link in urdf_result.links.items():
            # Only merge for links with visual geometry.
            if urdf_link.findall("visual"):
                merged_path = merge_link_visual_meshes_for_sdf(
                    urdf_link=urdf_link,
                    urdf_dir=urdf_dir,
                    sdf_dir=sdf_dir,
                    link_name=link_name,
                )
                if merged_path:
                    link_merged_visuals[link_name] = merged_path

    # Convert links.
    for link_name, urdf_link in urdf_result.links.items():
        physics = link_physics.get(link_name) if link_physics else None
        friction = link_friction.get(link_name, 0.5) if link_friction else 0.5
        pose = link_poses.get(link_name)
        collision_paths = link_collision_paths.get(link_name)
        merged_visual = link_merged_visuals.get(link_name)

        sdf_link = convert_urdf_link_to_sdf(
            urdf_link=urdf_link,
            urdf_dir=urdf_dir,
            sdf_dir=sdf_dir,
            physics=physics,
            friction=friction,
            link_pose=pose,
            generated_collision_paths=collision_paths,
            merged_visual_path=merged_visual,
            scale_factor=scale_factor,
        )
        model.append(sdf_link)

    # Convert joints.
    for urdf_joint in urdf_result.joints.values():
        sdf_joint = convert_urdf_joint_to_sdf(urdf_joint)
        model.append(sdf_joint)

    # Format XML with indentation.
    ET.indent(sdf, space="  ", level=0)

    # Write SDF file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(sdf)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    # Fix any inertia tensors that violate the triangle inequality.
    fix_sdf_file_inertia(output_path)

    console_logger.info(
        f"Converted URDF to SDF: {urdf_path.name} -> {output_path.name}"
    )

    return output_path
