"""Validated scene mutation for procedural manipuland piles."""

import logging
import time

from typing import Callable

import numpy as np

from omegaconf import DictConfig

from scenesmith.agent_utils.assets.asset_manager import AssetManager
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    PlacementInfo,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.manipuland_agents.tools.response_dataclasses import (
    ManipulandErrorType,
    PileCreationResult,
)
from scenesmith.manipuland_agents.tools.stacking_tools.pile_tools import (
    _compute_pile_composite_bbox_in_local_frame,
    _load_bounding_box,
    compute_pile_spawn_transforms,
    simulate_pile_physics,
)
from scenesmith.utils.geometry.sdf_utils import serialize_rigid_transform

console_logger = logging.getLogger(__name__)


def create_pile_tool_impl(
    asset_ids: list[str],
    surface_id: str,
    position_x: float,
    position_z: float,
    scene: RoomScene,
    cfg: DictConfig,
    asset_manager: AssetManager,
    support_surfaces: dict[str, SupportSurface],
    generate_unique_id: Callable[[str], UniqueID],
) -> str:
    """Create a pile of objects on a support surface.

    This is the core implementation logic for pile creation, taking all
    dependencies as explicit parameters.

    Args:
        asset_ids: List of asset IDs to pile (at least 2 objects).
        surface_id: ID of the support surface to place the pile on.
        position_x: X coordinate on the surface (in surface's local frame).
        position_z: Z coordinate on the surface (in surface's local frame).
        scene: The RoomScene to add the pile to.
        cfg: Configuration with pile_simulation settings.
        asset_manager: AssetManager for retrieving asset objects.
        support_surfaces: Dictionary of available support surfaces.
        generate_unique_id: Function to generate unique IDs for new objects.

    Returns:
        JSON string with PileCreationResult.
    """
    console_logger.info(
        f"Tool called: create_pile(asset_ids={asset_ids}, surface_id={surface_id}, "
        f"position_x={position_x}, position_z={position_z})"
    )
    start_time = time.time()

    # Validate at least 2 assets for a pile.
    if len(asset_ids) < 2:
        console_logger.warning(
            f"Pile creation failed: requires at least 2 assets, got {len(asset_ids)}"
        )
        return PileCreationResult(
            success=False,
            message=(
                "Pile requires at least 2 assets. "
                "Use place_manipuland_on_surface for single items."
            ),
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=0,
            removed_count=0,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Validate surface exists.
    if surface_id not in support_surfaces:
        available_ids = list(support_surfaces.keys())
        console_logger.warning(
            f"Pile creation failed: invalid surface_id '{surface_id}'"
        )
        return PileCreationResult(
            success=False,
            message=(
                f"Invalid surface_id: {surface_id}. "
                f"Available surfaces: {available_ids}"
            ),
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=0,
            removed_count=0,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_SURFACE,
        ).to_json()

    target_surface = support_surfaces[surface_id]

    # Validate all assets exist and build asset list.
    assets: list[SceneObject] = []
    for asset_id in asset_ids:
        try:
            unique_id = UniqueID(asset_id)
        except Exception:
            console_logger.warning(
                f"Pile creation failed: invalid asset ID format '{asset_id}'"
            )
            return PileCreationResult(
                success=False,
                message=f"Invalid asset ID format: {asset_id}",
                pile_object_id=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        asset = asset_manager.get_asset_by_id(unique_id)
        if not asset:
            available = asset_manager.list_available_assets()
            manipuland_ids = [
                str(a.object_id)
                for a in available
                if a.object_type == ObjectType.MANIPULAND
            ]
            console_logger.warning(
                f"Pile creation failed: asset '{asset_id}' not found"
            )
            return PileCreationResult(
                success=False,
                message=(
                    f"Asset {asset_id} not found. "
                    f"Available manipulands: {manipuland_ids}"
                ),
                pile_object_id=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        # Validate SDF path exists for physics simulation.
        if not asset.sdf_path or not asset.sdf_path.exists():
            console_logger.warning(
                f"Pile creation failed: asset '{asset_id}' has no SDF file"
            )
            return PileCreationResult(
                success=False,
                message=(
                    f"Asset {asset_id} has no SDF file. "
                    "Cannot simulate pile physics."
                ),
                pile_object_id=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.INVALID_OPERATION,
            ).to_json()

        assets.append(asset)

    # Load bounding boxes for each asset.
    bounding_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for asset in assets:
        if asset.bbox_min is not None and asset.bbox_max is not None:
            bounding_boxes.append((asset.bbox_min.copy(), asset.bbox_max.copy()))
        else:
            # Try to load from geometry.
            try:
                bbox = _load_bounding_box(
                    asset.sdf_path, scale_factor=asset.scale_factor
                )
                bounding_boxes.append(bbox)
            except ValueError as e:
                console_logger.warning(
                    f"Pile creation failed: could not load bounding box for "
                    f"'{asset.name}': {e}"
                )
                return PileCreationResult(
                    success=False,
                    message=f"Failed to load bounding box for {asset.name}: {e}",
                    pile_object_id=None,
                    parent_surface_id=surface_id,
                    num_items=len(asset_ids),
                    pile_count=0,
                    removed_count=0,
                    inside_assets=[],
                    removed_assets=[],
                    error_type=ManipulandErrorType.INVALID_OPERATION,
                ).to_json()

    # Compute base transform on surface.
    position_2d = np.array([position_x, position_z])
    base_transform = target_surface.to_world_pose(
        position_2d=position_2d, rotation_2d=0.0
    )

    # Get surface dimensions for ground plane sizing.
    surface_min = target_surface.bounding_box_min[:2]
    surface_max = target_surface.bounding_box_max[:2]
    surface_width = float(surface_max[0] - surface_min[0])
    surface_depth = float(surface_max[1] - surface_min[1])

    # Get pile simulation config.
    pile_cfg = cfg.pile_simulation

    # Compute spawn transforms.
    surface_z = float(target_surface.transform.translation()[2])
    initial_transforms = compute_pile_spawn_transforms(
        bounding_boxes=bounding_boxes,
        base_transform=base_transform,
        surface_z=surface_z,
        cfg=pile_cfg,
    )

    # Create temporary SceneObjects for simulation.
    temp_scene_objects = []
    for i, asset in enumerate(assets):
        temp_obj = SceneObject(
            object_id=UniqueID(f"pile_temp_{i}"),
            object_type=ObjectType.MANIPULAND,
            name=asset.name,
            description=asset.description,
            transform=initial_transforms[i],
            sdf_path=asset.sdf_path,
            scale_factor=asset.scale_factor,
        )
        temp_scene_objects.append(temp_obj)

    # Ground plane position: centered at pile spawn position on surface.
    ground_xyz = (
        float(base_transform.translation()[0]),
        float(base_transform.translation()[1]),
        surface_z,
    )

    # Run physics simulation.
    sim_start_time = time.time()
    inside_indices, outside_indices, final_transforms, error_msg = (
        simulate_pile_physics(
            scene_objects=temp_scene_objects,
            initial_transforms=initial_transforms,
            ground_xyz=ground_xyz,
            ground_size=(surface_width, surface_depth),
            surface_z=surface_z,
            inside_z_threshold=pile_cfg.inside_z_threshold,
            simulation_time=pile_cfg.simulation_time,
            simulation_time_step=pile_cfg.simulation_time_step,
        )
    )
    sim_elapsed = time.time() - sim_start_time
    console_logger.info(f"Pile simulation completed in {sim_elapsed:.2f}s")

    if error_msg:
        console_logger.error(f"Pile creation failed: simulation error: {error_msg}")
        return PileCreationResult(
            success=False,
            message=f"Simulation failed: {error_msg}",
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=0,
            removed_count=len(asset_ids),
            inside_assets=[],
            removed_assets=[a.name for a in assets],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Check if we have at least 2 objects on surface.
    pile_count = len(inside_indices)
    removed_count = len(outside_indices)

    inside_asset_names = [assets[i].name for i in inside_indices]
    removed_asset_names = [assets[i].name for i in outside_indices]

    if pile_count < 2:
        console_logger.warning(
            f"Pile creation failed: only {pile_count} objects stayed on surface "
            f"(need at least 2)"
        )
        return PileCreationResult(
            success=False,
            message=(
                f"Only {pile_count} object(s) stayed on surface (need at least 2). "
                f"{removed_count} fell off. Try placing pile further from edges "
                f"or using fewer objects."
            ),
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=pile_count,
            removed_count=removed_count,
            inside_assets=inside_asset_names,
            removed_assets=removed_asset_names,
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Pile is valid - create composite SceneObject.
    pile_id = generate_unique_id("pile")

    # Use first inside object's final transform as the pile's reference transform.
    first_inside_idx = inside_indices[0]
    pile_transform = final_transforms[first_inside_idx]

    # Build member_assets metadata with sdf_path for each member that stayed.
    member_assets = []
    for i in inside_indices:
        asset = assets[i]
        final_transform = final_transforms[i]
        member_assets.append(
            {
                "asset_id": str(asset.object_id),
                "name": asset.name,
                "transform": serialize_rigid_transform(final_transform),
                "sdf_path": str(asset.sdf_path.absolute()),
                "geometry_path": (
                    str(asset.geometry_path.absolute()) if asset.geometry_path else None
                ),
            }
        )

    # Compute composite bounding box.
    all_bbox_min, all_bbox_max = _compute_pile_composite_bbox_in_local_frame(
        assets=assets,
        final_transforms=final_transforms,
        inside_indices=inside_indices,
        pile_transform=pile_transform,
    )

    composite_object = SceneObject(
        object_id=pile_id,
        object_type=ObjectType.MANIPULAND,
        name=f"pile_{pile_count}",
        description=f"Pile of {pile_count} objects: " + ", ".join(inside_asset_names),
        transform=pile_transform,
        geometry_path=None,
        sdf_path=None,
        metadata={
            "composite_type": "pile",
            "member_assets": member_assets,
            "num_members": pile_count,
        },
        bbox_min=all_bbox_min,
        bbox_max=all_bbox_max,
        placement_info=PlacementInfo(
            parent_surface_id=target_surface.surface_id,
            position_2d=position_2d.copy(),
            rotation_2d=0.0,
            placement_method="pile_placement",
        ),
    )

    scene.add_object(composite_object)

    elapsed_time = time.time() - start_time

    # Build result message.
    if removed_count > 0:
        message = (
            f"Created pile '{pile_id}' with {pile_count} objects. "
            f"{removed_count} object(s) fell off: {removed_asset_names}"
        )
    else:
        message = f"Created pile '{pile_id}' with {pile_count} objects"

    console_logger.info(
        f"Successfully created pile {pile_id} with {pile_count} items "
        f"on surface {surface_id} in {elapsed_time:.2f}s"
    )

    return PileCreationResult(
        success=True,
        message=message,
        pile_object_id=str(pile_id),
        parent_surface_id=surface_id,
        num_items=len(asset_ids),
        pile_count=pile_count,
        removed_count=removed_count,
        inside_assets=inside_asset_names,
        removed_assets=removed_asset_names,
    ).to_json()
