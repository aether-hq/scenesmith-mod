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

import copy
import logging

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


def should_be_welded(
    model_name: str, object_registry: dict[str, dict], mode: str
) -> bool:
    """Determine if a model should be welded based on mode and object type.

    Args:
        model_name: The model name from the DMD file.
        object_registry: Registry mapping model names to object metadata.
        mode: Welding mode ('nothing', 'furniture', or 'all').

    Returns:
        True if the model should be welded to world.

    Raises:
        ValueError: If model_name is not found in registry.
    """
    # Room geometry and frames are handled separately (always welded to frame).
    if model_name.startswith("room_geometry_") or model_name.endswith("_frame"):
        return True

    if model_name not in object_registry:
        raise ValueError(
            f"Model '{model_name}' not found in house_state.json metadata. "
            "This indicates a mismatch between the DMD file and metadata."
        )

    obj_info = object_registry[model_name]
    obj_type = obj_info["object_type"]
    asset_source = obj_info.get("asset_source")

    # Always-welded types (wall_mounted, ceiling_mounted).
    if obj_type in ALWAYS_WELDED_TYPES:
        return True

    # Always-welded asset sources (thin coverings have no collision geometry).
    if asset_source in ALWAYS_WELDED_ASSET_SOURCES:
        return True

    # Manipulands are free except in 'all' mode where everything is welded.
    if obj_type in ALWAYS_FREE_TYPES:
        if mode == "all":
            return True
        return False

    # Furniture.
    if obj_type == "furniture":
        if mode == "nothing":
            return False
        # modes 'furniture' and 'all' weld furniture.
        return True

    # Unknown type - fail fast.
    raise ValueError(f"Unknown object type '{obj_type}' for model '{model_name}'.")


def extract_weld_pose(weld_directive: dict) -> dict:
    """Extract pose from an add_weld directive.

    Args:
        weld_directive: The add_weld directive dictionary.

    Returns:
        Pose dictionary with translation and rotation.

    Raises:
        ValueError: If the weld directive has unexpected structure.
    """
    if "add_weld" not in weld_directive:
        raise ValueError("Expected add_weld directive")

    x_pc = weld_directive["add_weld"].get("X_PC")
    if not x_pc:
        raise ValueError("add_weld missing X_PC pose")

    return x_pc


def _resolve_link_name(add_model_directive: dict, link_name: str | None) -> str | None:
    """Resolve the link name from a free body pose.

    If link_name is provided, uses it directly. Otherwise discovers the
    first (and typically only) link from default_free_body_pose.
    """
    if link_name is not None:
        return link_name
    model_data = add_model_directive.get("add_model", {})
    free_body_pose = model_data.get("default_free_body_pose", {})
    if not free_body_pose:
        return None
    return next(iter(free_body_pose))


def extract_free_body_pose(add_model_directive: dict, link_name: str | None = None):
    """Extract pose from a free body's default_free_body_pose.

    Strips the ``base_frame`` key so the returned dict contains only pose
    data (translation/rotation) suitable for use as ``X_PC``.

    Args:
        add_model_directive: The add_model directive dictionary.
        link_name: The link name to extract pose for. If None, uses the
            first link found.

    Returns:
        Pose dictionary with translation and rotation, or None if not found.
    """
    resolved = _resolve_link_name(add_model_directive, link_name)
    if resolved is None:
        return None
    model_data = add_model_directive.get("add_model", {})
    free_body_pose = model_data.get("default_free_body_pose", {})
    pose = free_body_pose.get(resolved)
    if pose is None:
        return None
    # Strip base_frame so it doesn't leak into X_PC.
    pose = {k: v for k, v in pose.items() if k != "base_frame"}
    return pose


def extract_base_frame(
    add_model_directive: dict, link_name: str | None = None
) -> str | None:
    """Extract base_frame from a free body's default_free_body_pose.

    Args:
        add_model_directive: The add_model directive dictionary.
        link_name: The link name to look up. If None, uses the first link.

    Returns:
        The base_frame string, or None if not present.
    """
    resolved = _resolve_link_name(add_model_directive, link_name)
    if resolved is None:
        return None
    model_data = add_model_directive.get("add_model", {})
    free_body_pose = model_data.get("default_free_body_pose", {})
    pose = free_body_pose.get(resolved)
    if pose is None:
        return None
    return pose.get("base_frame")


def convert_welded_to_free(
    add_model: dict, weld: dict, link_name: str = "base_link"
) -> dict:
    """Convert a welded model to a free body.

    Preserves the weld's parent frame as ``base_frame`` in the free body pose
    when the parent is not ``"world"``.

    Args:
        add_model: The add_model directive.
        weld: The corresponding add_weld directive.
        link_name: The link name for the free body pose.

    Returns:
        New add_model directive with default_free_body_pose.
    """
    new_model = copy.deepcopy(add_model)
    pose = extract_weld_pose(weld)
    parent_frame = get_weld_parent_frame(weld)

    free_pose = dict(pose)
    if parent_frame != "world":
        free_pose["base_frame"] = parent_frame

    new_model["add_model"]["default_free_body_pose"] = {link_name: free_pose}

    return new_model


def convert_free_to_welded(
    add_model: dict, link_name: str | None = None
) -> tuple[dict, dict]:
    """Convert a free body to a welded model.

    Uses the ``base_frame`` from the free body pose as the weld parent
    (falls back to ``"world"`` if absent).

    Args:
        add_model: The add_model directive with default_free_body_pose.
        link_name: The link name to extract pose from. If None,
            auto-discovers from the free body pose.

    Returns:
        Tuple of (new add_model without pose, new add_weld directive).

    Raises:
        ValueError: If no free body pose found.
    """
    resolved_link = _resolve_link_name(add_model, link_name)
    base_frame = extract_base_frame(add_model, resolved_link)
    pose = extract_free_body_pose(add_model, resolved_link)
    if pose is None:
        raise ValueError(
            f"No default_free_body_pose for {resolved_link} in model "
            f"{add_model.get('add_model', {}).get('name', 'unknown')}"
        )

    parent = base_frame if base_frame else "world"

    new_model = copy.deepcopy(add_model)
    del new_model["add_model"]["default_free_body_pose"]

    model_name = add_model["add_model"]["name"]
    weld = {
        "add_weld": {
            "parent": parent,
            "child": f"{model_name}::{resolved_link}",
            "X_PC": pose,
        }
    }

    return new_model, weld


def is_frame_weld(weld: dict) -> bool:
    """Check if a weld is to a frame (world or named room frame).

    Frame welds have parents like "world" or "room_bedroom_frame".
    Object welds have parents like "model_name::link_name".
    The distinguisher is whether the parent contains "::".
    """
    parent = weld.get("add_weld", {}).get("parent", "")
    return "::" not in parent


def get_weld_parent_frame(weld: dict) -> str:
    """Extract the parent frame name from a frame weld."""
    return weld.get("add_weld", {}).get("parent", "world")


def get_weld_model_name(weld: dict) -> str | None:
    """Extract model name from a weld's child field."""
    child = weld.get("add_weld", {}).get("child", "")
    if "::" in child:
        return child.split("::")[0]
    return None


def get_weld_child_link_name(weld: dict) -> str:
    """Extract link name from a weld's child field."""
    child = weld.get("add_weld", {}).get("child", "")
    if "::" in child:
        return child.split("::")[1]
    return "base_link"


def _should_free_composites(mode: str) -> bool:
    """Return True if composite members should be free bodies in this mode."""
    return mode in ("nothing", "furniture")


def _get_parent_model_name(weld: dict) -> str | None:
    """Extract parent model name from a non-world weld's parent field.

    Parent field looks like 'model_name::link_name'.
    """
    parent = weld.get("add_weld", {}).get("parent", "")
    if "::" in parent:
        return parent.split("::")[0]
    return None


def _build_non_world_weld_index(
    directives: list[dict],
) -> dict[str, dict]:
    """Build index of non-world welds keyed by child model name."""
    index: dict[str, dict] = {}
    for directive in directives:
        if "add_weld" in directive and not is_frame_weld(directive):
            child_name = get_weld_model_name(directive)
            if child_name:
                index[child_name] = directive
    return index
