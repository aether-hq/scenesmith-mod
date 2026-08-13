"""Acquire and seed accepted Aether inventory into native SceneSmith stages.

The semantic compiler owns object identity, role, dimensions, and transform. SceneSmith
owns asset acquisition and all contextual additions around those immutable anchors.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from pydrake.all import RigidTransform, RotationMatrix

from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID

from .scene_census import CensusError

console_logger = logging.getLogger(__name__)

_STAGE_TYPES = {
    "furniture": ObjectType.FURNITURE,
    "wall-mounted": ObjectType.WALL_MOUNTED,
    "ceiling-mounted": ObjectType.CEILING_MOUNTED,
    "manipuland": ObjectType.MANIPULAND,
}
_CPU_SOURCES = {"hssd", "objaverse"}
_SOURCE_ALIASES = {"curated": "hssd"}
_LOCKED_DIMENSION_RELATIVE_TOLERANCE = 0.20


def load_accepted_stage_input(path: str | Path | None) -> dict[str, Any] | None:
    """Load one immutable stage input or return ``None`` for standalone runs."""
    if not path:
        return None
    with Path(path).open() as file:
        value = json.load(file)
    if value.get("realization_engine") != "scenesmith":
        raise CensusError("accepted inventory requires realization_engine=scenesmith")
    if value.get("pipeline_profile") != "full":
        raise CensusError("accepted inventory requires the full SceneSmith profile")
    if value.get("people_allowed") is not False:
        raise CensusError("accepted scenic inventory cannot include people")
    placements = value.get("locked_placements")
    if not isinstance(placements, list):
        raise CensusError("accepted stage input has no locked_placements list")
    return value


def placements_for_stage(
    stage_input: Mapping[str, Any] | None, stage: str
) -> tuple[Mapping[str, Any], ...]:
    """Return the accepted placements owned by one native SceneSmith stage."""
    if stage_input is None:
        return ()
    if stage not in _STAGE_TYPES:
        raise CensusError(f"unknown native SceneSmith placement stage: {stage}")
    return tuple(
        placement
        for placement in stage_input["locked_placements"]
        if placement.get("stage") == stage
    )


def _configured_source(asset_manager: Any, placement: Mapping[str, Any]) -> str:
    configured = str(asset_manager.general_asset_source)
    if configured not in _CPU_SOURCES:
        raise CensusError(
            f"CPU locked-inventory acquisition requires hssd or objaverse; got {configured}"
        )
    allowed = {
        _SOURCE_ALIASES.get(str(source), str(source))
        for source in placement["route"]["sources"]
    }
    if configured not in allowed:
        raise CensusError(
            f"accepted object {placement['object_id']} requires one of "
            f"{sorted(allowed)}, but this CPU worker provides {configured}"
        )
    return configured


def _asset_identity(asset: SceneObject) -> str:
    source = str(asset.metadata.get("asset_source") or "unknown")
    for key in ("hssd_mesh_id", "objaverse_mesh_id", "articulated_id"):
        value = asset.metadata.get(key)
        if value:
            return f"{source}:{value}"
    if asset.geometry_path and asset.geometry_path.is_file():
        import hashlib

        digest = hashlib.sha256(asset.geometry_path.read_bytes()).hexdigest()
        return f"geometry-sha256:{digest}"
    raise CensusError(f"acquired asset {asset.object_id} has no content identity")


def _validate_locked_dimensions(
    asset: SceneObject,
    placement: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Refuse an asset whose measured envelope cannot honor locked dimensions.

    Library retrieval uses authored dimensions as a ranking/scaling target, but a
    semantically similar asset can still have the wrong proportions.  A locked
    placement must not silently accept that mismatch and then claim the approved
    layout was preserved.
    """
    if asset.bbox_min is None or asset.bbox_max is None:
        raise CensusError(
            f"acquired asset for {placement['object_id']} has no measured bounds"
        )
    requested = np.asarray(placement["size_m"], dtype=float)
    measured = np.asarray(asset.bbox_max - asset.bbox_min, dtype=float)
    if requested.shape != (3,) or np.any(requested <= 0):
        raise CensusError(
            f"accepted object {placement['object_id']} has invalid locked dimensions"
        )
    relative_error = np.abs(measured - requested) / requested
    if np.any(relative_error > _LOCKED_DIMENSION_RELATIVE_TOLERANCE):
        formatted = ", ".join(f"{value:.1%}" for value in relative_error)
        raise CensusError(
            f"acquired asset for {placement['object_id']} cannot honor locked dimensions: "
            f"requested={requested.tolist()}, measured={measured.tolist()}, "
            f"axis relative error=[{formatted}], allowed="
            f"{_LOCKED_DIMENSION_RELATIVE_TOLERANCE:.0%}"
        )
    return measured.tolist(), relative_error.tolist()


def _place_locked_asset(
    scene: Any,
    asset: SceneObject,
    placement: Mapping[str, Any],
    *,
    stage: str,
) -> SceneObject:
    object_id = UniqueID(str(placement["object_id"]))
    if object_id in scene.objects:
        existing = scene.objects[object_id]
        if existing.metadata.get("aether_locked"):
            return existing
        raise CensusError(f"accepted object id collides with existing object: {object_id}")
    measured_size, dimension_error = _validate_locked_dimensions(asset, placement)

    yaw = float(placement["yaw_radians"])
    rotation = RotationMatrix.MakeZRotation(yaw)
    local_center = (asset.bbox_min + asset.bbox_max) / 2
    target_center = np.asarray(placement["centre_origin_m"], dtype=float)
    translation = target_center - rotation.multiply(local_center)
    metadata = deepcopy(asset.metadata)
    metadata.update(
        {
            "aether_instance_id": str(object_id),
            "aether_asset_id": _asset_identity(asset),
            "aether_role_id": str(placement["role_id"]),
            "aether_object_class": "scenic-object",
            "aether_functional_zone_ids": list(placement["functional_zone_ids"]),
            "aether_locked": True,
            "aether_locked_fields": list(placement["locked_fields"]),
            "aether_placement_kind": str(placement["placement_kind"]),
            "aether_placement_stage": stage,
            "aether_accepts_dressing": bool(placement["accepts_dressing"]),
            "aether_provenance": list(placement["provenance"]),
            "aether_requested_size_m": list(placement["size_m"]),
            "aether_measured_size_m": measured_size,
            "aether_dimension_relative_error": dimension_error,
            "aether_asset_route": deepcopy(placement["route"]),
        }
    )
    placed = SceneObject(
        object_id=object_id,
        object_type=_STAGE_TYPES[stage],
        name=asset.name,
        description=asset.description,
        transform=RigidTransform(R=rotation, p=translation),
        geometry_path=asset.geometry_path,
        sdf_path=asset.sdf_path,
        image_path=asset.image_path,
        support_surfaces=deepcopy(asset.support_surfaces),
        metadata=metadata,
        bbox_min=asset.bbox_min.copy(),
        bbox_max=asset.bbox_max.copy(),
        immutable=True,
        scale_factor=asset.scale_factor,
    )
    scene.add_object(placed)
    return placed


def seed_locked_inventory(
    *,
    scene: Any,
    asset_manager: Any,
    stage_input: Mapping[str, Any] | None,
    stage: str,
) -> tuple[str, ...]:
    """Acquire and place all accepted anchors for a native SceneSmith stage.

    The configured CPU library must appear in each placement's approved source
    chain. CUDA-only distinctive or articulated requirements therefore fail loudly.
    """
    placements = placements_for_stage(stage_input, stage)
    if not placements:
        return ()
    for placement in placements:
        _configured_source(asset_manager, placement)

    request = AssetGenerationRequest(
        object_descriptions=[str(item["description"]) for item in placements],
        short_names=[str(item["object_id"]) for item in placements],
        object_type=_STAGE_TYPES[stage],
        desired_dimensions=[list(item["size_m"]) for item in placements],
        style_context=str(stage_input["room_prompt"]),
        scene_id=str(stage_input["job_id"]),
    )
    result = asset_manager.generate_assets(request)
    if result.has_failures:
        failures = "; ".join(
            f"{item.description}: {item.error_message}" for item in result.failed_assets
        )
        raise CensusError(f"locked {stage} asset acquisition failed: {failures}")
    assets_by_name = {str(asset.name): asset for asset in result.successful_assets}
    expected = {str(item["object_id"]) for item in placements}
    if set(assets_by_name) != expected:
        raise CensusError(
            f"locked {stage} acquisition returned {sorted(assets_by_name)}; "
            f"expected {sorted(expected)}"
        )

    placed_ids = tuple(
        str(
            _place_locked_asset(
                scene,
                assets_by_name[str(placement["object_id"])],
                placement,
                stage=stage,
            ).object_id
        )
        for placement in placements
    )
    console_logger.info("Seeded %d accepted %s object(s): %s", len(placed_ids), stage, placed_ids)
    return placed_ids
