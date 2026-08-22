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

from scripts.scene_conversion.dmd_io import _compose_poses
from scripts.scene_conversion.welding import (
    _build_non_world_weld_index,
    _get_parent_model_name,
    _should_free_composites,
    convert_free_to_welded,
    convert_welded_to_free,
    extract_base_frame,
    extract_free_body_pose,
    extract_weld_pose,
    get_weld_child_link_name,
    get_weld_model_name,
    get_weld_parent_frame,
    is_frame_weld,
    should_be_welded,
)

# Object types that are always welded (regardless of mode).
ALWAYS_WELDED_TYPES = {"wall_mounted", "ceiling_mounted"}

# Object types that are free in all modes.
ALWAYS_FREE_TYPES = {"manipuland"}

# Asset sources that are always welded (regardless of mode).
# Thin coverings (rugs, carpets, tablecloths) have no collision geometry,
# so they must remain welded to avoid unrealistic physics behavior.
ALWAYS_WELDED_ASSET_SOURCES = {"thin_covering"}


def convert_dmd(
    directives: list[dict], object_registry: dict[str, dict], mode: str
) -> list[dict]:
    """Convert DMD directives to the specified welding mode.

    Args:
        directives: List of DMD directive dictionaries.
        object_registry: Registry mapping model names to object metadata.
        mode: Target welding mode ('nothing', 'furniture', or 'all').

    Returns:
        New list of directives with converted welding.
    """
    result = []
    free_composites = _should_free_composites(mode)

    # Build index of non-world welds (composite member welds) by child name.
    non_world_welds = _build_non_world_weld_index(directives)

    # Set of child model names that have non-world welds (composite members).
    composite_children: set[str] = set(non_world_welds.keys())

    # Build index of world welds by model name for easy lookup.
    weld_by_model: dict[str, dict] = {}
    for directive in directives:
        if "add_weld" in directive and is_frame_weld(directive):
            model_name = get_weld_model_name(directive)
            if model_name:
                weld_by_model[model_name] = directive

    # Track which models we've seen to detect duplicates.
    processed_models: set[str] = set()

    # Track world poses for models (needed for composing composite poses).
    model_world_poses: dict[str, dict] = {}

    # Track parent frame for each model (for round-tripping frame welds).
    model_parent_frames: dict[str, str] = {}

    i = 0
    while i < len(directives):
        directive = directives[i]

        if "add_model" in directive:
            model_name = directive["add_model"]["name"]

            # Skip room geometry and frames (pass through unchanged).
            if model_name.startswith("room_geometry_"):
                result.append(directive)
                i += 1
                continue

            # Handle composite child models not in registry.
            if model_name not in object_registry:
                if free_composites and model_name in composite_children:
                    # This model is a composite member that should be free.
                    # Check if next directive is a non-world weld for it.
                    has_nw_weld = (
                        i + 1 < len(directives)
                        and "add_weld" in directives[i + 1]
                        and not is_frame_weld(directives[i + 1])
                        and get_weld_model_name(directives[i + 1]) == model_name
                    )
                    if has_nw_weld:
                        weld = directives[i + 1]
                        link = get_weld_child_link_name(weld)
                        parent_name = _get_parent_model_name(weld)
                        parent_pose = model_world_poses.get(parent_name)
                        if parent_pose is not None:
                            rel_pose = extract_weld_pose(weld)
                            world_pose = _compose_poses(parent_pose, rel_pose)
                            # Inherit base_frame from parent so Drake
                            # interprets the composed pose in the correct
                            # coordinate frame (e.g. room frame).
                            parent_frame = model_parent_frames.get(parent_name)
                            if parent_frame and parent_frame != "world":
                                world_pose["base_frame"] = parent_frame
                            new_model = copy.deepcopy(directive)
                            new_model["add_model"]["default_free_body_pose"] = {
                                link: world_pose
                            }
                            result.append(new_model)
                            model_world_poses[model_name] = {
                                k: v for k, v in world_pose.items() if k != "base_frame"
                            }
                            if parent_frame:
                                model_parent_frames[model_name] = parent_frame
                            i += 2  # Skip the weld.
                            continue
                        else:
                            console_logger.warning(
                                f"Could not find parent pose for "
                                f"'{parent_name}' (child: '{model_name}'). "
                                f"Keeping weld unchanged."
                            )

                # In "all" mode, weld unregistered models too.
                if mode == "all":
                    has_free_pose = "default_free_body_pose" in directive["add_model"]
                    has_next_world_weld = (
                        i + 1 < len(directives)
                        and "add_weld" in directives[i + 1]
                        and is_frame_weld(directives[i + 1])
                        and get_weld_model_name(directives[i + 1]) == model_name
                    )
                    has_next_nw_weld = (
                        i + 1 < len(directives)
                        and "add_weld" in directives[i + 1]
                        and not is_frame_weld(directives[i + 1])
                        and get_weld_model_name(directives[i + 1]) == model_name
                    )

                    if has_next_world_weld:
                        # Already welded to world, keep as-is.
                        result.append(directive)
                        weld_pose = extract_weld_pose(directives[i + 1])
                        model_world_poses[model_name] = weld_pose
                        result.append(directives[i + 1])
                        i += 2
                        continue
                    elif has_free_pose:
                        # Convert free to welded.
                        new_model, new_weld = convert_free_to_welded(directive)
                        result.append(new_model)
                        result.append(new_weld)
                        model_world_poses[model_name] = extract_free_body_pose(
                            directive
                        )
                        i += 1
                        continue
                    elif has_next_nw_weld:
                        # Composite child with non-world weld. Compose
                        # pose with parent to create world weld.
                        weld = directives[i + 1]
                        link = get_weld_child_link_name(weld)
                        parent_name = _get_parent_model_name(weld)
                        parent_pose = model_world_poses.get(parent_name)
                        if parent_pose is not None:
                            rel_pose = extract_weld_pose(weld)
                            world_pose = _compose_poses(parent_pose, rel_pose)
                            new_model = copy.deepcopy(directive)
                            if "default_free_body_pose" in (new_model["add_model"]):
                                del new_model["add_model"]["default_free_body_pose"]
                            new_weld = {
                                "add_weld": {
                                    "parent": "world",
                                    "child": f"{model_name}::{link}",
                                    "X_PC": world_pose,
                                }
                            }
                            result.append(new_model)
                            result.append(new_weld)
                            model_world_poses[model_name] = world_pose
                            i += 2
                            continue
                        else:
                            console_logger.warning(
                                f"Could not find parent pose for "
                                f"'{parent_name}' "
                                f"(child: '{model_name}'). "
                                f"Keeping weld unchanged."
                            )

                # Pass through unchanged (not in registry, not a
                # composite child we need to modify).
                result.append(directive)
                # Track pose if it has one (composite base objects).
                free_pose = extract_free_body_pose(directive)
                if free_pose:
                    model_world_poses[model_name] = free_pose
                i += 1
                continue

            processed_models.add(model_name)
            target_welded = should_be_welded(model_name, object_registry, mode)
            has_free_pose = "default_free_body_pose" in directive["add_model"]

            # Check if next directive is a weld for this model.
            has_weld = (
                i + 1 < len(directives)
                and "add_weld" in directives[i + 1]
                and is_frame_weld(directives[i + 1])
                and get_weld_model_name(directives[i + 1]) == model_name
            )

            if target_welded:
                if has_weld:
                    # Already welded, keep as-is.
                    result.append(directive)
                    result.append(directives[i + 1])
                    weld_pose = extract_weld_pose(directives[i + 1])
                    model_world_poses[model_name] = weld_pose
                    parent_frame = get_weld_parent_frame(directives[i + 1])
                    model_parent_frames[model_name] = parent_frame
                    i += 2
                elif has_free_pose:
                    # Convert free to welded.
                    base_frame = extract_base_frame(directive)
                    new_model, new_weld = convert_free_to_welded(directive)
                    result.append(new_model)
                    result.append(new_weld)
                    model_world_poses[model_name] = extract_free_body_pose(directive)
                    if base_frame:
                        model_parent_frames[model_name] = base_frame
                    i += 1
                else:
                    raise ValueError(
                        f"Model '{model_name}' should be welded but has no "
                        "pose information (no weld and no "
                        "default_free_body_pose)."
                    )
            else:
                # Should be free.
                if has_free_pose:
                    # Already free, keep as-is.
                    result.append(directive)
                    model_world_poses[model_name] = extract_free_body_pose(directive)
                    base_frame = extract_base_frame(directive)
                    if base_frame:
                        model_parent_frames[model_name] = base_frame
                    i += 1
                elif has_weld:
                    # Convert welded to free.
                    link = get_weld_child_link_name(directives[i + 1])
                    new_model = convert_welded_to_free(
                        directive, directives[i + 1], link_name=link
                    )
                    result.append(new_model)
                    weld_pose = extract_weld_pose(directives[i + 1])
                    model_world_poses[model_name] = weld_pose
                    parent_frame = get_weld_parent_frame(directives[i + 1])
                    model_parent_frames[model_name] = parent_frame
                    i += 2  # Skip the weld directive.
                else:
                    raise ValueError(
                        f"Model '{model_name}' should be free but has no "
                        "pose information (no weld and no "
                        "default_free_body_pose)."
                    )

        elif "add_weld" in directive:
            # Welds to world are handled with their add_model above.
            if is_frame_weld(directive):
                model_name = get_weld_model_name(directive)
                # Skip if already processed with add_model.
                if model_name and model_name in processed_models:
                    i += 1
                    continue
            else:
                # Non-world weld (composite member).
                child_name = get_weld_model_name(directive)
                if free_composites and child_name in composite_children:
                    # Already handled when processing the child's add_model
                    # (converted to free body pose). But if the add_model
                    # wasn't adjacent, we might reach here. Skip it.
                    if child_name in model_world_poses:
                        i += 1
                        continue
            result.append(directive)
            i += 1

        else:
            # Pass through other directives (add_frame, etc.).
            result.append(directive)
            i += 1

    return result
