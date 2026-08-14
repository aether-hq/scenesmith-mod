"""Aether's typed, deterministic bridge over SceneSmith scene state.

The public facade is lazy so wire-contract helpers remain usable without
importing Drake and Blender. Native dependencies load only when their concrete
placement names are first requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CeilingPlacementAdapter": ("native_placement", "CeilingPlacementAdapter"),
    "CensusError": ("scene_census", "CensusError"),
    "CompletionPlacementRuntime": ("completion_bridge", "CompletionPlacementRuntime"),
    "DeterministicPlacementAdapter": ("runtime", "DeterministicPlacementAdapter"),
    "FloorPlacementAdapter": ("native_placement", "FloorPlacementAdapter"),
    "PhysicalEvidenceProvider": ("physical_evidence", "PhysicalEvidenceProvider"),
    "SceneSmithCompletionRuntime": ("runtime", "SceneSmithCompletionRuntime"),
    "SurfacePlacementAdapter": ("native_placement", "SurfacePlacementAdapter"),
    "TypedAssetSpec": ("assets", "TypedAssetSpec"),
    "WallPlacementAdapter": ("native_placement", "WallPlacementAdapter"),
    "acquire_completion_assets": ("assets", "acquire_completion_assets"),
    "annotate_room_scene_instances": ("room_runtime", "annotate_room_scene_instances"),
    "build_scene_census": ("scene_census", "build_scene_census"),
    "canonical_digest": ("scene_census", "canonical_digest"),
    "compile_asset_spec": ("assets", "compile_asset_spec"),
    "execute_completion_patch": ("completion_bridge", "execute_completion_patch"),
    "load_accepted_stage_input": ("locked_inventory", "load_accepted_stage_input"),
    "normalize_instance_id": ("scene_census", "normalize_instance_id"),
    "placements_for_stage": ("locked_inventory", "placements_for_stage"),
    "room_geometry_digest": ("physical_evidence", "room_geometry_digest"),
    "seed_locked_inventory": ("locked_inventory", "seed_locked_inventory"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
