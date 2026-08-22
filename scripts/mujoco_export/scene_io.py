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
import numpy as np
import yaml

from pydrake.all import Quaternion, RigidTransform, RotationMatrix

from scenesmith.agent_utils.scene.room_parts.room_models import SceneObject

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scripts.mujoco_export.transforms import parse_pose

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def parse_transform_dict(transform_data: dict) -> RigidTransform:
    """Parse a serialized transform dict into a RigidTransform.

    Args:
        transform_data: Dict with 'translation' and 'rotation_wxyz' keys.

    Returns:
        RigidTransform constructed from the data.
    """
    translation = np.array(transform_data.get("translation", [0, 0, 0]))
    rotation_wxyz = transform_data.get("rotation_wxyz", [1, 0, 0, 0])
    quaternion = Quaternion(wxyz=rotation_wxyz)
    rotation_matrix = RotationMatrix(quaternion)
    return RigidTransform(rotation_matrix, translation)


def expand_composite_to_members(
    obj: SceneObject, room_offset: np.ndarray
) -> list[tuple[Path, RigidTransform, str]]:
    """Expand composite object into (sdf_path, transform, name) tuples.

    Composites (stack, pile, filled_container) contain multiple physical objects
    that are grouped together. This function extracts each member so they can be
    individually exported.

    Adapted from scenesmith/robot_eval/dmd_scene.py::_expand_composite_members().

    Args:
        obj: SceneObject with composite_type in metadata.
        room_offset: [x, y, z] offset to apply to member transforms.

    Returns:
        List of (sdf_path, transform, name) for each member object.
    """
    metadata = obj.metadata
    composite_type = metadata.get("composite_type")
    members = []

    if composite_type in ("stack", "pile"):
        for asset in metadata.get("member_assets", []):
            sdf_path_str = asset.get("sdf_path")
            if not sdf_path_str:
                continue
            sdf_path = Path(sdf_path_str)
            transform = parse_transform_dict(asset.get("transform", {}))
            # Apply room offset.
            new_pos = transform.translation() + room_offset
            new_transform = RigidTransform(R=transform.rotation(), p=new_pos)
            members.append((sdf_path, new_transform, asset.get("name", "member")))

    elif composite_type == "filled_container":
        # Container.
        container = metadata.get("container_asset", {})
        if container.get("sdf_path"):
            sdf_path = Path(container["sdf_path"])
            transform = parse_transform_dict(container.get("transform", {}))
            new_pos = transform.translation() + room_offset
            new_transform = RigidTransform(R=transform.rotation(), p=new_pos)
            members.append(
                (sdf_path, new_transform, container.get("name", "container"))
            )
        # Fill items.
        for asset in metadata.get("fill_assets", []):
            sdf_path_str = asset.get("sdf_path")
            if not sdf_path_str:
                continue
            sdf_path = Path(sdf_path_str)
            transform = parse_transform_dict(asset.get("transform", {}))
            new_pos = transform.translation() + room_offset
            new_transform = RigidTransform(R=transform.rotation(), p=new_pos)
            members.append((sdf_path, new_transform, asset.get("name", "fill")))

    return members


def parse_dmd_yaml(dmd_path: Path) -> list[dict]:
    """Parse Drake Model Directive YAML file.

    Args:
        dmd_path: Path to the DMD YAML file.

    Returns:
        List of directives from the DMD file.
    """

    def angle_axis_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> dict:
        return {"!AngleAxis": loader.construct_mapping(node)}

    yaml.SafeLoader.add_constructor("!AngleAxis", angle_axis_constructor)

    with open(dmd_path) as f:
        data = yaml.safe_load(f)

    return data.get("directives", []) if data else []


def get_model_directives(directives: list[dict]) -> list[dict]:
    """Extract add_model directives in order."""
    return [d["add_model"] for d in directives if "add_model" in d]


def get_weld_directives_by_model(directives: list[dict]) -> dict[str, dict]:
    """Index add_weld directives by child model name."""
    welds: dict[str, dict] = {}
    for directive in directives:
        if "add_weld" not in directive:
            continue
        child = directive["add_weld"].get("child", "")
        if "::" not in child:
            continue
        model_name = child.split("::", 1)[0]
        welds[model_name] = directive["add_weld"]
    return welds


def get_welded_models(directives: list[dict]) -> set[str]:
    """Extract model names that are welded (to any parent) from DMD directives.

    Any model in an add_weld directive is static, regardless of the parent
    frame (world, room_*_frame, etc.).

    Args:
        directives: List of DMD directives.

    Returns:
        Set of model names that are welded (should be static).
    """
    welded = set()
    for d in directives:
        if "add_weld" not in d:
            continue
        child = d["add_weld"].get("child", "")
        # child format is "model_name::link_name", extract model_name.
        model_name = child.split("::")[0]
        welded.add(model_name)
    return welded


def resolve_package_uri(uri: str, sdf_dir: Path) -> Path | None:
    """Resolve package:// URI to filesystem path.

    Tries common resolution strategies for Drake and ROS package URIs.

    Args:
        uri: URI string, potentially starting with 'package://'.
        sdf_dir: Directory containing the SDF file.

    Returns:
        Resolved filesystem path, or None if resolution fails.
    """
    if not uri.startswith("package://"):
        return sdf_dir / uri

    # Strip package:// prefix.
    package_path = uri[len("package://") :]

    # Strategy 1: Look relative to SDF directory (common for packaged models).
    # package://pkg_name/path/... -> try sdf_dir/../path/...
    parts = package_path.split("/", 1)
    if len(parts) == 2:
        pkg_name, rel_path = parts

        # Try common parent directories.
        for parent in [
            sdf_dir.parent,
            sdf_dir.parent.parent,
            sdf_dir.parent.parent.parent,
        ]:
            candidate = parent / rel_path
            if candidate.exists():
                return candidate

            # Also try with package name prefix (pkg_name/rel_path).
            candidate = parent / pkg_name / rel_path
            if candidate.exists():
                return candidate

    # Strategy 2: Try relative to SDF dir directly.
    candidate = sdf_dir / package_path
    if candidate.exists():
        return candidate

    return None


def resolve_scene_file_uri(uri: str, scene_dir: Path, dmd_dir: Path) -> Path | None:
    """Resolve a DMD add_model file URI to a filesystem path."""
    if uri.startswith("package://scene/"):
        rel_path = uri[len("package://scene/") :]
        candidate = scene_dir / rel_path
        if candidate.exists():
            return candidate
        return None

    if uri.startswith("file://"):
        candidate = Path(uri[len("file://") :])
        return candidate if candidate.exists() else None

    candidate = Path(uri)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    candidate = dmd_dir / uri
    return candidate if candidate.exists() else None


def get_sdf_link_poses_and_roots(
    sdf_path: Path,
) -> tuple[dict[str, tuple[list[float], list[float]]], list[str]]:
    """Return model-frame link poses and root links for an SDF model."""
    tree = parse_sdf_with_drake_namespace(sdf_path)
    root = tree.getroot()
    model_elem = root.find(".//model")
    if model_elem is None:
        return {}, []

    link_absolute_poses: dict[str, tuple[list[float], list[float]]] = {}
    links = model_elem.findall("./link")
    for link_elem in links:
        link_name = link_elem.get("name")
        if not link_name:
            continue
        link_absolute_poses[link_name] = parse_pose(link_elem.find("pose"))

    child_links = set()
    for joint_elem in model_elem.findall("./joint"):
        child_elem = joint_elem.find("child")
        if child_elem is not None and child_elem.text:
            child_links.add(child_elem.text)

    root_links = [
        link_elem.get("name")
        for link_elem in links
        if link_elem.get("name") and link_elem.get("name") not in child_links
    ]
    return link_absolute_poses, root_links


def get_dmd_reference_link_name(
    add_model_data: dict,
    weld_data: dict | None,
    link_absolute_poses: dict[str, tuple[list[float], list[float]]],
    root_links: list[str],
) -> str | None:
    """Choose the link whose DMD-authored world pose should anchor the model."""
    free_body_pose = add_model_data.get("default_free_body_pose", {})
    if free_body_pose:
        link_name = next(iter(free_body_pose))
        if link_name in link_absolute_poses:
            return link_name

    if weld_data:
        child = weld_data.get("child", "")
        if "::" in child:
            link_name = child.split("::", 1)[1]
            if link_name in link_absolute_poses:
                return link_name

    if len(root_links) == 1:
        return root_links[0]

    if root_links:
        return root_links[0]

    if link_absolute_poses:
        return next(iter(link_absolute_poses))

    return None


def infer_is_furniture_from_sdf_path(sdf_path: Path) -> bool:
    """Infer furniture-ness from the packaged scene asset path."""
    path_str = sdf_path.as_posix()
    return "/generated_assets/furniture/" in path_str


def infer_room_id_from_scene_asset_path(scene_dir: Path, sdf_path: Path) -> str:
    """Infer room_id from a packaged scene asset path.

    Examples:
    - scene_dir/room_bedroom/generated_assets/... -> bedroom
    - scene_dir/room_geometry/room_geometry_bathroom.sdf -> bathroom
    """
    try:
        rel_path = sdf_path.relative_to(scene_dir)
    except ValueError:
        rel_path = sdf_path

    if rel_path.parts:
        first = rel_path.parts[0]
        if first.startswith("room_") and first != "room_geometry":
            return first[len("room_") :]

    stem = sdf_path.stem
    if stem.startswith("room_geometry_"):
        return stem[len("room_geometry_") :]

    return ""


def parse_sdf_with_drake_namespace(sdf_path: Path) -> ET.ElementTree:
    """Parse SDF file that may contain Drake-specific namespace extensions.

    Drake SDFs often use the 'drake:' prefix for custom elements without
    declaring the namespace. This causes ET.parse() to fail with
    'unbound prefix' error. We handle this by adding namespace declarations.

    Args:
        sdf_path: Path to SDF file.

    Returns:
        Parsed ElementTree.
    """
    # Read file content.
    with open(sdf_path, "r") as f:
        content = f.read()

    # Check if file uses drake: prefix without namespace declaration.
    if "drake:" in content and "xmlns:drake" not in content:
        # Add namespace declaration to sdf root element.
        content = content.replace(
            '<sdf version="1.7">',
            '<sdf version="1.7" xmlns:drake="http://drake.mit.edu">',
        )
        content = content.replace(
            '<sdf version="1.8">',
            '<sdf version="1.8" xmlns:drake="http://drake.mit.edu">',
        )
        content = content.replace(
            '<sdf version="1.9">',
            '<sdf version="1.9" xmlns:drake="http://drake.mit.edu">',
        )

    return ET.ElementTree(ET.fromstring(content))
