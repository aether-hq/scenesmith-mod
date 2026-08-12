"""Aether's typed, deterministic bridge over SceneSmith scene state."""

from .assets import TypedAssetSpec, acquire_completion_assets, compile_asset_spec
from .completion_bridge import CompletionPlacementRuntime, execute_completion_patch
from .room_runtime import annotate_room_scene_instances
from .runtime import DeterministicPlacementAdapter, SceneSmithCompletionRuntime
from .scene_census import CensusError, build_scene_census, canonical_digest, normalize_instance_id

__all__ = [
    "CensusError",
    "CompletionPlacementRuntime",
    "DeterministicPlacementAdapter",
    "SceneSmithCompletionRuntime",
    "TypedAssetSpec",
    "acquire_completion_assets",
    "annotate_room_scene_instances",
    "build_scene_census",
    "canonical_digest",
    "compile_asset_spec",
    "execute_completion_patch",
    "normalize_instance_id",
]
