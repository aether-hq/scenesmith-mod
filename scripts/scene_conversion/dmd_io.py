#!/usr/bin/env python3
"""Convert DMD file welding configurations.

This script converts Drake Model Directive (DMD) files between different welding
configurations. It supports three modes:

- nothing: Only wall/ceiling-mounted objects welded (furniture FREE, composites FREE)
- furniture: Furniture welded, composites FREE
- all: Everything welded (furniture + manipulands)

The script requires house_state.json metadata to determine object types. It will
fail with an error if the metadata is not found or if a model is not found in
the metadata (no fallback heuristics).

Example usage:
    python scripts/convert_dmd_welding.py combined_house/house.dmd.yaml -m furniture
    python scripts/convert_dmd_welding.py house.dmd.yaml -m nothing -o house_free.dmd.yaml
"""

import json
import logging
import math

from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
console_logger = logging.getLogger(__name__)

# Object types that are always welded (regardless of mode).
ALWAYS_WELDED_TYPES = {"wall_mounted", "ceiling_mounted"}

# Object types that are free in all modes.
ALWAYS_FREE_TYPES = {"manipuland"}

# Asset sources that are always welded (regardless of mode).
# Thin coverings (rugs, carpets, tablecloths) have no collision geometry,
# so they must remain welded to avoid unrealistic physics behavior.
ALWAYS_WELDED_ASSET_SOURCES = {"thin_covering"}


def _angle_axis_to_matrix(angle_deg: float, axis: list[float]) -> np.ndarray:
    """Convert angle-axis rotation to a 3x3 rotation matrix."""
    angle_rad = math.radians(angle_deg)
    ax = np.array(axis, dtype=float)
    norm = np.linalg.norm(ax)
    if norm < 1e-12:
        return np.eye(3)
    ax = ax / norm
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def _matrix_to_angle_axis(
    r: np.ndarray,
) -> tuple[float, list[float]]:
    """Convert a 3x3 rotation matrix to angle-axis (degrees, unit axis)."""
    trace_val = float(np.trace(r))
    cos_angle = max(-1.0, min(1.0, (trace_val - 1.0) / 2.0))
    angle_rad = math.acos(cos_angle)
    if abs(angle_rad) < 1e-10:
        return 0.0, [0.0, 0.0, 1.0]
    sin_angle = math.sin(angle_rad)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (
        2.0 * sin_angle
    )
    return math.degrees(angle_rad), axis.tolist()


def _extract_translation_and_rotation(
    pose: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract translation vector and rotation matrix from a DMD pose dict."""
    t = np.array(pose.get("translation", [0, 0, 0]), dtype=float)
    rot_data = pose.get("rotation")
    if rot_data and "!AngleAxis" in rot_data:
        aa = rot_data["!AngleAxis"]
        r = _angle_axis_to_matrix(aa["angle_deg"], aa["axis"])
    else:
        r = np.eye(3)
    return t, r


def _compose_poses(parent_pose: dict, child_pose: dict) -> dict:
    """Compose two DMD poses: T_world_child = T_world_parent * T_parent_child."""
    pt, pr = _extract_translation_and_rotation(parent_pose)
    ct, cr = _extract_translation_and_rotation(child_pose)
    world_t = pt + pr @ ct
    world_r = pr @ cr
    angle_deg, axis = _matrix_to_angle_axis(world_r)
    return {
        "translation": [float(world_t[0]), float(world_t[1]), float(world_t[2])],
        "rotation": {
            "!AngleAxis": {
                "angle_deg": angle_deg,
                "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
            }
        },
    }


def load_house_state(state_path: Path) -> dict:
    """Load and parse house_state.json.

    Args:
        state_path: Path to house_state.json.

    Returns:
        Parsed house state dictionary.

    Raises:
        FileNotFoundError: If house_state.json does not exist.
    """
    if not state_path.exists():
        raise FileNotFoundError(
            f"house_state.json not found at {state_path}. "
            "This script requires scenesmith metadata to determine object types."
        )
    with open(state_path) as f:
        return json.load(f)


def build_object_registry(house_state: dict) -> dict[str, dict]:
    """Build a registry mapping model names to object metadata.

    Model names in DMD files follow the pattern: {room_id}_{object_id}
    For example: hallway_console_table_0, bedroom_2_bed_0

    Args:
        house_state: Parsed house_state.json dictionary.

    Returns:
        Dictionary mapping model names to object metadata including object_type,
        room_id, object_id, and whether it's a composite member.
    """
    registry = {}

    rooms = house_state.get("rooms", {})
    for room_id, room_data in rooms.items():
        objects = room_data.get("objects", {})
        for object_id, obj in objects.items():
            # Skip wall objects (they're not in DMD, they're in room_geometry SDF).
            if obj.get("object_type") == "wall":
                continue

            # Main object model name.
            model_name = f"{room_id}_{object_id}"
            metadata = obj.get("metadata", {})
            registry[model_name] = {
                "object_type": obj.get("object_type"),
                "asset_source": metadata.get("asset_source"),
                "room_id": room_id,
                "object_id": object_id,
                "is_composite_member": False,
                "parent_model_name": None,
            }

            # If object has member_assets (composite), register those too.
            member_assets = obj.get("member_assets", [])
            for i, member in enumerate(member_assets):
                member_model_name = f"{room_id}_{object_id}_member_{i}"
                registry[member_model_name] = {
                    "object_type": obj.get("object_type"),
                    "asset_source": metadata.get("asset_source"),
                    "room_id": room_id,
                    "object_id": object_id,
                    "is_composite_member": True,
                    "parent_model_name": model_name,
                    "member_index": i,
                }

    return registry


def parse_dmd_yaml(dmd_path: Path) -> list[dict]:
    """Parse DMD YAML file into a list of directives.

    Args:
        dmd_path: Path to the DMD YAML file.

    Returns:
        List of directive dictionaries.

    Raises:
        FileNotFoundError: If the DMD file does not exist.
    """
    if not dmd_path.exists():
        raise FileNotFoundError(f"DMD file not found: {dmd_path}")

    with open(dmd_path) as f:
        content = f.read()

    # Parse YAML with custom tag handling for !AngleAxis.
    def angle_axis_constructor(loader, node):
        return {"!AngleAxis": loader.construct_mapping(node)}

    yaml.SafeLoader.add_constructor("!AngleAxis", angle_axis_constructor)
    data = yaml.safe_load(content)

    return data.get("directives", [])


def write_dmd_yaml(directives: list[dict], output_path: Path) -> None:
    """Write directives to DMD YAML file.

    Args:
        directives: List of directive dictionaries.
        output_path: Path to write the output file.
    """

    # Custom representer for AngleAxis.
    def angle_axis_representer(dumper, data):
        if "!AngleAxis" in data:
            return dumper.represent_mapping(
                "!AngleAxis", data["!AngleAxis"], flow_style=False
            )
        return dumper.represent_dict(data)

    yaml.SafeDumper.add_representer(dict, angle_axis_representer)

    output = {"directives": directives}

    with open(output_path, "w") as f:
        yaml.safe_dump(
            output,
            f,
            default_flow_style=None,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        )
