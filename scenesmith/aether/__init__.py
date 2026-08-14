"""Aether's typed, deterministic bridge over SceneSmith scene state."""

from .assets import TypedAssetSpec, acquire_completion_assets, compile_asset_spec
from .completion_bridge import CompletionPlacementRuntime, execute_completion_patch
from .locked_inventory import (
    load_accepted_stage_input,
    placements_for_stage,
    seed_locked_inventory,
)
from .native_placement import (
    CeilingPlacementAdapter,
    FloorPlacementAdapter,
    SurfacePlacementAdapter,
    WallPlacementAdapter,
)
from .physical_evidence import PhysicalEvidenceProvider, room_geometry_digest
from .room_runtime import annotate_room_scene_instances
from .runtime import DeterministicPlacementAdapter, SceneSmithCompletionRuntime
from .scene_census import (
    CensusError,
    build_scene_census,
    canonical_digest,
    normalize_instance_id,
)

__all__ = [
    "CeilingPlacementAdapter",
    "CensusError",
    "CompletionPlacementRuntime",
    "DeterministicPlacementAdapter",
    "FloorPlacementAdapter",
    "PhysicalEvidenceProvider",
    "SceneSmithCompletionRuntime",
    "SurfacePlacementAdapter",
    "TypedAssetSpec",
    "WallPlacementAdapter",
    "acquire_completion_assets",
    "annotate_room_scene_instances",
    "build_scene_census",
    "canonical_digest",
    "compile_asset_spec",
    "execute_completion_patch",
    "load_accepted_stage_input",
    "normalize_instance_id",
    "placements_for_stage",
    "room_geometry_digest",
    "seed_locked_inventory",
]
