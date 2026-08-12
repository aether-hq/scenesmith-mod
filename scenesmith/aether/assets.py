"""Compile Aether completion asset briefs into SceneSmith typed acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .scene_census import CensusError

_OBJECT_TYPES = {
    "populate-surfaces": "manipuland",
    "place-floor-group": "furniture",
    "place-wall-group": "wall_mounted",
    "place-ceiling-group": "ceiling_mounted",
}
_STRATEGIES = {
    # ``curated`` is an Aether policy class.  HSSD is the curated conventional
    # corpus currently provided by the SceneSmith worker.
    "curated": "hssd",
    "hssd": "hssd",
    "objaverse": "objaverse",
    "artvip": "articulated",
    "sam3d": "sam3d",
}


@dataclass(frozen=True, slots=True)
class TypedAssetSpec:
    description: str
    short_name: str
    dimensions: tuple[float, float, float]
    object_type: str
    strategies: tuple[str, ...]


def compile_asset_spec(
    operation: dict[str, Any], asset_brief: dict[str, Any]
) -> TypedAssetSpec:
    """Translate one validated Genesis brief without another semantic model call."""
    try:
        object_type = _OBJECT_TYPES[operation["operation"]]
    except KeyError as exc:
        raise CensusError(f"unsupported completion operation: {operation.get('operation')}") from exc
    strategies: list[str] = []
    for source in asset_brief["source_order"]:
        try:
            strategy = _STRATEGIES[source]
        except KeyError as exc:
            raise CensusError(f"unsupported completion asset source: {source}") from exc
        if strategy not in strategies:
            strategies.append(strategy)
    if not strategies:
        raise CensusError(f"asset brief {asset_brief.get('variant_id')} has no usable source")
    dimensions = tuple(float(value) for value in asset_brief["dimensions_m"])
    if len(dimensions) != 3 or any(value <= 0 for value in dimensions):
        raise CensusError(f"asset brief {asset_brief.get('variant_id')} has invalid dimensions")
    return TypedAssetSpec(
        description=str(asset_brief["description"]),
        short_name=str(asset_brief["short_name"]),
        dimensions=dimensions,
        object_type=object_type,
        strategies=tuple(strategies),
    )


def acquire_completion_assets(
    asset_manager,
    operation: dict[str, Any],
    asset_brief: dict[str, Any],
    *,
    style_context: str,
    scene_id: str,
):
    """Run SceneSmith validation/routing for one already-authored typed brief.

    Heavy SceneSmith types are imported lazily so census and bridge validation
    remain runnable on development machines without Drake, Blender, or CUDA.
    """
    from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
    from scenesmith.agent_utils.asset_router.dataclasses import AssetItem
    from scenesmith.agent_utils.room import ObjectType

    spec = compile_asset_spec(operation, asset_brief)
    object_type = ObjectType(spec.object_type)
    request = AssetGenerationRequest(
        object_descriptions=[spec.description],
        short_names=[spec.short_name],
        object_type=object_type,
        desired_dimensions=[list(spec.dimensions)],
        style_context=style_context,
        scene_id=scene_id,
    )
    item = AssetItem(
        description=spec.description,
        short_name=spec.short_name,
        dimensions=list(spec.dimensions),
        object_type=object_type,
        strategies=list(spec.strategies),
    )
    return asset_manager.generate_assets_from_typed_items(request, [item])
