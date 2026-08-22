import csv
import logging

from pathlib import Path
from threading import Lock

from omegaconf import DictConfig, OmegaConf

from scenesmith.agent_utils.blender.process_provider import (
    RenderAllocation,
    render_allocations,
)
from scenesmith.agent_utils.geometry_generation_server.pipelines.sam_provider import (
    sam_provider_config_from_mapping,
)
from scenesmith.agent_utils.runtime.execution_providers import (
    HardwareInventory,
    resolve_torch_device,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)

console_logger = logging.getLogger(__name__)

# Pipeline stages in execution order (derived from AgentType enum).
PIPELINE_STAGES = [agent.value for agent in AgentType]

# Stage dependencies for resume from checkpoint.
# Maps start_stage to the checkpoint it needs from the previous stage.
STAGE_CHECKPOINTS = {
    "floor_plan": None,
    "furniture": None,
    "wall_mounted": "scene_after_furniture",
    "ceiling_mounted": "scene_after_wall_objects",
    "manipuland": "scene_after_ceiling_objects",
}

# Maps start_stage to the asset directories it needs from previous stages.
STAGE_ASSET_DIRS = {
    "floor_plan": [],
    "furniture": [],
    "wall_mounted": ["furniture"],
    "ceiling_mounted": ["furniture", "wall_mounted"],
    "manipuland": ["furniture", "wall_mounted", "ceiling_mounted"],
}


def _require_projection_success(stage: str, success: bool) -> None:
    """Prevent a physically invalid scene from becoming a resumable checkpoint."""

    if not success:
        raise RuntimeError(
            f"{stage} physical projection failed; cannot publish checkpoint."
        )


def _require_semantic_publication_inputs(artifacts: dict[str, Path]) -> None:
    """Prevent legacy/shadow diagnostics from becoming a publication bypass."""

    missing = tuple(
        name for name, artifact_path in artifacts.items() if not artifact_path.is_file()
    )
    if missing:
        raise RuntimeError(
            "Semantic publication blocked: missing mandatory enforcement artifacts "
            + ", ".join(missing)
            + ". Any shadow audit emitted for this checkpoint is diagnostic only."
        )


def _validate_final_dense_library_book_rows(
    scene: RoomScene,
    manipuland_agent: StatefulManipulandAgent,
) -> int:
    """Revalidate owner-bound book rows after every final pose mutation."""

    manipuland_agent.scene = scene
    normalize_surplus = getattr(
        manipuland_agent,
        "_normalize_dense_library_book_row_surplus",
        None,
    )
    removed = normalize_surplus() if callable(normalize_surplus) else 0
    if removed:
        console_logger.info(
            "Pruned %d surplus dense book rows after final pose mutation",
            removed,
        )
    invalid_row_ids = StatefulManipulandAgent._physically_invalid_dense_book_row_ids(
        scene,
        manipuland_agent.cfg,
    )
    count = StatefulManipulandAgent._validate_dense_library_book_rows(
        scene,
        invalid_row_ids=invalid_row_ids,
    )
    if count:
        console_logger.info(
            "Final dense library book-row gate passed with %d surviving rows",
            count,
        )
    return count


def _asset_config_uses_generated_geometry(asset_config: dict) -> bool:
    """Return whether an asset configuration can reach text-to-3D generation."""

    source = asset_config.get("general_asset_source")
    if source == "generated":
        return True
    if source != "all":
        return False

    router = asset_config.get("router", {})
    generated_strategy = router.get("strategies", {}).get("generated", {})
    if not generated_strategy.get("enabled", True):
        return False
    source_order = asset_config.get("federated", {}).get(
        "source_order", ["polyhaven", "hssd", "objaverse", "generated"]
    )
    return "generated" in source_order


def _resolve_geometry_runtime_configuration(
    config: dict | DictConfig,
) -> tuple[str, dict | None]:
    """Derive one authoritative geometry runtime shared by generated agents."""

    config_dict = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else config
    )
    generated: list[tuple[str, dict]] = []
    for agent_name in (
        "furniture_agent",
        "wall_agent",
        "ceiling_agent",
        "manipuland_agent",
    ):
        if agent_name not in config_dict:
            continue
        asset_config = config_dict[agent_name]["asset_manager"]
        if _asset_config_uses_generated_geometry(asset_config):
            generated.append((agent_name, asset_config))
    if not generated:
        raise ValueError("No generated asset agent requires a geometry runtime")
    backends = {
        str(asset_config.get("backend", "hunyuan3d")) for _, asset_config in generated
    }
    if len(backends) != 1:
        details = ", ".join(
            f"{agent}={asset.get('backend', 'hunyuan3d')}" for agent, asset in generated
        )
        raise ValueError(
            "All generated asset agents must use the same geometry backend; " + details
        )
    backend = backends.pop()
    if backend != "sam3d":
        return backend, None
    resolved_configs = [
        (agent, sam_provider_config_from_mapping(asset.get("sam3d", {})))
        for agent, asset in generated
    ]
    reference = resolved_configs[0][1]
    for agent, resolved in resolved_configs[1:]:
        if resolved != reference:
            raise ValueError(
                "All generated asset agents must use the same SAM3D runtime "
                f"configuration; {resolved_configs[0][0]} and {agent} differ."
            )
    return backend, reference


def _get_retrieval_compute_device(
    *, requested: str = "auto", policy: str = "balanced"
) -> str:
    """Resolve a provider-backed Torch device for retrieval servers.

    If CUDA is selected, the last logical device is reserved to reduce
    contention with geometry generation. MPS and CPU are valid first-class
    targets rather than implicit fallbacks.

    Provider visibility and physical-to-logical device mapping are handled by
    the shared execution-provider registry.

    Returns:
        A concrete Torch device such as ``cuda:1``, ``mps``, or ``cpu``.
    """

    # Import only after geometry workers have forked; importing Torch in the
    # parent earlier can initialize an accelerator runtime inherited by workers.
    import torch

    inventory = HardwareInventory.detect(torch_module=torch)
    return resolve_torch_device(
        requested=requested,
        policy=policy,
        inventory=inventory,
        device_preference="last",
        torch_module=torch,
        environ={},
    )


class RenderAllocationAllocator:
    """Thread-safe round-robin allocator for provider-owned render slots."""

    def __init__(
        self,
        requested_render_provider: str = "auto",
        requested_process_provider: str = "auto",
    ) -> None:
        self._devices = self._detect_devices(
            requested_render_provider,
            requested_process_provider,
        )
        self._counter = 0
        self._lock = Lock()
        console_logger.info(
            "RenderAllocationAllocator initialized with isolation devices: %s",
            self._devices,
        )

    @staticmethod
    def _detect_devices(
        requested_render_provider: str,
        requested_process_provider: str,
    ) -> list[RenderAllocation]:
        """Return provider-owned render process slots."""

        return list(
            render_allocations(
                requested_render_provider,
                requested_process_provider=requested_process_provider,
            )
        )

    def allocate(self) -> RenderAllocation:
        """Get the next immutable render allocation."""
        with self._lock:
            device = self._devices[self._counter % len(self._devices)]
            self._counter += 1
            return device

    @property
    def available_allocations(self) -> list[RenderAllocation]:
        """Get a snapshot of all resolved render allocations."""

        return self._devices.copy()


def _reset_inherited_sdk_state() -> None:
    """Reset OpenAI Agents SDK state inherited via fork.

    After fork(), the child inherits corrupted SDK state:
    1. Active trace/span ContextVars - makes workers think they're in parent's trace
    2. BatchTraceProcessor with orphaned threading.Lock and dead background thread
    3. BackendSpanExporter with corrupted httpx.Client connections

    We clear all of these so workers start fresh. Workers can reinitialize
    tracing if needed.

    Must be called at the start of each worker function.
    """
    from agents.tracing import scope

    # Clear any inherited trace/span context so workers start fresh.
    scope._current_trace.set(None)
    scope._current_span.set(None)

    # Clear the corrupted processor from the provider's processor list.
    # After fork(), the BatchTraceProcessor has orphaned locks and dead background thread.
    # The provider holds a reference to it via _multi_processor._processors.
    # We clear that list so traces won't try to use the corrupted processor.
    # Traces will still work, just won't be exported (which is fine for subprocesses).
    try:
        from agents.tracing import setup as tracing_setup

        provider = tracing_setup.GLOBAL_TRACE_PROVIDER
        if provider and hasattr(provider, "_multi_processor"):
            provider._multi_processor.set_processors([])
    except Exception:
        pass  # Best effort - don't crash on reset failure.


def _load_prompts_from_csv(csv_path: str) -> list[tuple[int, str]]:
    """Load scene prompts from CSV file.

    Args:
        csv_path: Path to CSV file with columns: scene_index, prompt.

    Returns:
        List of (scene_id, prompt) tuples.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV has invalid format or data.
    """
    prompts_with_ids = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row.
        # Start at 2 (after header).
        for row_num, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"CSV row {row_num} has fewer than 2 columns: {row}")
            try:
                scene_id = int(row[0])
            except ValueError:
                raise ValueError(
                    f"CSV row {row_num}: scene_index '{row[0]}' is not a valid integer"
                )
            prompt = row[1]
            prompts_with_ids.append((scene_id, prompt))
    return prompts_with_ids


def _export_scene_blend_file(
    scene: RoomScene, scene_dir: Path, cfg_dict: dict, name: str = "final_scene"
) -> None:
    """Export scene to a .blend file.

    Args:
        scene: The scene to export.
        scene_dir: Base directory for scene outputs.
        cfg_dict: Configuration dictionary.
        name: Name for the scene state subdirectory.
    """
    from scenesmith.agent_utils.rendering.pipeline.blend_export import (
        save_scene_as_blend,
    )

    blend_output_path = scene_dir / "scene_states" / name / "scene.blend"
    try:
        rendering_cfg = cfg_dict.get("furniture_agent", {}).get("rendering", {})
        save_scene_as_blend(
            scene=scene,
            output_path=blend_output_path,
            blender_server_host=rendering_cfg.get("blender_server_host", "127.0.0.1"),
            blender_server_port_range=tuple(
                rendering_cfg.get("blender_server_port_range", [8000, 8050])
            ),
            server_startup_delay=rendering_cfg.get("server_startup_delay", 0.1),
            port_cleanup_delay=rendering_cfg.get("port_cleanup_delay", 0.1),
        )
    except Exception as e:
        console_logger.error(f"Failed to export .blend file: {e}")
