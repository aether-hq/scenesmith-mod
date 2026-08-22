import json
import logging
import shutil

from pathlib import Path

from scenesmith.agent_utils.scene.room_parts.room_models import AgentType

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


def _fix_paths_in_json_file(
    json_path: Path,
    new_room_dir: Path,
    new_scene_dir: Path | None = None,
    source_room_dir: Path | None = None,
    source_scene_dir: Path | None = None,
) -> None:
    """Fix absolute paths in a JSON file to point to new directories.

    Scans JSON for any string values containing absolute paths and rebases them:
    - Room-level paths (generated_assets/, scene_renders/) → new_room_dir
    - Scene-level paths (room_geometry/, floor_plans/) → new_scene_dir

    Args:
        json_path: Path to JSON file to fix.
        new_room_dir: New room directory for room-level paths.
        new_scene_dir: New scene directory for scene-level paths.
                       If None, defaults to parent of new_room_dir.
    """
    if not json_path.exists():
        return

    if new_scene_dir is None:
        new_scene_dir = new_room_dir.parent

    with open(json_path) as f:
        data = json.load(f)

    def fix_path(value: str) -> str:
        """Fix a single path string if it's an absolute path."""
        if not value.startswith("/"):
            return value  # Already relative, no fix needed.

        path = Path(value)
        if source_room_dir is not None:
            try:
                return str(new_room_dir / path.relative_to(source_room_dir))
            except ValueError:
                pass
            if source_scene_dir is not None:
                try:
                    return str(new_scene_dir / path.relative_to(source_scene_dir))
                except ValueError:
                    pass
            # References to shared assets or prior runs were not copied into the
            # target checkpoint and must retain their existing absolute paths.
            return value

        # Room-level paths (relative to room directory).
        room_markers = ["generated_assets/", "scene_renders/", "scene_states/"]
        for marker in room_markers:
            if marker in value:
                rel_path = value.split(marker, 1)[1]
                return str(new_room_dir / marker.rstrip("/") / rel_path)

        # Scene-level paths (relative to scene directory).
        scene_markers = ["room_geometry/", "floor_plans/"]
        for marker in scene_markers:
            if marker in value:
                rel_path = value.split(marker, 1)[1]
                return str(new_scene_dir / marker.rstrip("/") / rel_path)

        return value  # Unknown pattern, leave as-is.

    def fix_paths_recursive(obj):
        """Recursively fix paths in a nested structure."""
        if isinstance(obj, dict):
            return {k: fix_paths_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_paths_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return fix_path(obj)
        return obj

    fixed_data = fix_paths_recursive(data)

    with open(json_path, "w") as f:
        json.dump(fixed_data, f, indent=2)

    console_logger.debug(f"Fixed paths in {json_path}")


def _fix_paths_in_yaml_file(
    yaml_path: Path,
    new_room_dir: Path,
    new_scene_dir: Path | None = None,
    source_room_dir: Path | None = None,
    source_scene_dir: Path | None = None,
) -> None:
    """Fix absolute paths in YAML file (e.g., scene.dmd.yaml Drake directives).

    Handles file:// URIs used in Drake model directives.

    Args:
        yaml_path: Path to YAML file to fix.
        new_room_dir: New room directory for room-level paths.
        new_scene_dir: New scene directory for scene-level paths.
                       If None, defaults to parent of new_room_dir.
    """
    import re

    if not yaml_path.exists():
        return

    if new_scene_dir is None:
        new_scene_dir = new_room_dir.parent

    content = yaml_path.read_text()

    def replace_path(match: re.Match) -> str:
        """Replace a file:// URI with the correct new path."""
        old_path = match.group(1)
        path = Path(old_path)
        if source_room_dir is not None:
            try:
                return f"file://{new_room_dir / path.relative_to(source_room_dir)}"
            except ValueError:
                pass
            if source_scene_dir is not None:
                try:
                    return (
                        f"file://{new_scene_dir / path.relative_to(source_scene_dir)}"
                    )
                except ValueError:
                    pass
            return match.group(0)

        # Determine if room-level or scene-level path.
        if "/generated_assets/" in old_path or "/scene_renders/" in old_path:
            # Room-level: extract relative part after room_*/.
            rel_match = re.search(r"room_[^/]+/(.+)$", old_path)
            if rel_match:
                return f"file://{new_room_dir / rel_match.group(1)}"
        elif "/room_geometry/" in old_path or "/floor_plans/" in old_path:
            # Scene-level: extract relative part after scene_*/.
            rel_match = re.search(r"scene_\d+/(.+)$", old_path)
            if rel_match:
                return f"file://{new_scene_dir / rel_match.group(1)}"
        return match.group(0)

    new_content = re.sub(r"file://(/[^\s\"']+)", replace_path, content)
    yaml_path.write_text(new_content)
    console_logger.debug(f"Fixed paths in {yaml_path}")


def _copy_checkpoint_for_stage(
    source_scene_dir: Path, target_scene_dir: Path, start_stage: str
) -> None:
    """Copy only the checkpoint state needed to resume from start_stage.

    Unlike copytree of entire scene, this explicitly copies only required files:
    - Scene-level: floor_plans/, house_layout.json, scene_blueprint.json,
      scene_requirement_graph.json, semantic prompt artifacts,
      semantic_obligation_ledger.json, and legacy room_geometry/
    - Room-level: checkpoint directory + referenced assets + semantic room kit

    NOT copied (ensuring fresh start for resumed stage):
    - *.db (session files - agent starts fresh conversation)
    - scene_renders/ (render directories - counter starts at 0)
    - *.log (log files - clean logs for new run)
    - action_log.json (replay log - new run builds its own)

    Args:
        source_scene_dir: Path to source scene directory.
        target_scene_dir: Path to target scene directory.
        start_stage: Stage to resume from (determines what to copy).
    """
    if not source_scene_dir.exists():
        raise FileNotFoundError(
            f"Source scene directory not found: {source_scene_dir}. "
            f"Ensure resume_from_path points to an experiment with this scene."
        )

    console_logger.info(f"Copying checkpoint for {start_stage} from {source_scene_dir}")

    # Remove target if it exists (Hydra may have created it).
    if target_scene_dir.exists():
        shutil.rmtree(target_scene_dir)

    target_scene_dir.mkdir(parents=True, exist_ok=True)

    # Current checkpoints keep structural geometry under floor_plans/. Older
    # checkpoints also wrote a scene-root room_geometry/ compatibility tree.
    legacy_room_geometry = source_scene_dir / "room_geometry"
    if legacy_room_geometry.exists():
        shutil.copytree(
            legacy_room_geometry,
            target_scene_dir / "room_geometry",
        )
    shutil.copytree(
        source_scene_dir / "floor_plans",
        target_scene_dir / "floor_plans",
    )
    # Materials directory contains textures referenced by floor/wall GLTFs.
    materials_dir = source_scene_dir / "materials"
    if materials_dir.exists():
        shutil.copytree(materials_dir, target_scene_dir / "materials")
    shutil.copy(
        source_scene_dir / "house_layout.json",
        target_scene_dir / "house_layout.json",
    )
    # Downstream deterministic detail selection consumes the persisted semantic
    # blueprint. Preserve it when resuming a current checkpoint; older
    # checkpoints without one retain the existing prompt-derived fallback.
    scene_blueprint = source_scene_dir / "scene_blueprint.json"
    if scene_blueprint.exists():
        shutil.copy(scene_blueprint, target_scene_dir / "scene_blueprint.json")
    requirement_graph = source_scene_dir / "scene_requirement_graph.json"
    if requirement_graph.exists():
        shutil.copy(
            requirement_graph,
            target_scene_dir / "scene_requirement_graph.json",
        )
    for enrichment_name in (
        "semantic_scene_enrichment.json",
        "semantic_prompt_enrichment.json",
    ):
        enrichment_artifact = source_scene_dir / enrichment_name
        if enrichment_artifact.exists():
            shutil.copy(enrichment_artifact, target_scene_dir / enrichment_name)
    semantic_ledger = source_scene_dir / "semantic_obligation_ledger.json"
    if semantic_ledger.exists():
        shutil.copy(
            semantic_ledger,
            target_scene_dir / "semantic_obligation_ledger.json",
        )
    capability_manifest = source_scene_dir / "semantic_capability_manifest.json"
    if capability_manifest.exists():
        shutil.copy(
            capability_manifest,
            target_scene_dir / "semantic_capability_manifest.json",
        )
    strategy_journal = source_scene_dir / "semantic_strategy_journal.json"
    if strategy_journal.exists():
        shutil.copy(
            strategy_journal,
            target_scene_dir / "semantic_strategy_journal.json",
        )
    spatial_compilation = source_scene_dir / "semantic_spatial_compilation.json"
    if spatial_compilation.exists():
        shutil.copy(
            spatial_compilation,
            target_scene_dir / "semantic_spatial_compilation.json",
        )
    topology_manifest = source_scene_dir / "semantic_topology_manifest.json"
    if topology_manifest.exists():
        shutil.copy(
            topology_manifest,
            target_scene_dir / "semantic_topology_manifest.json",
        )

    checkpoint_name = STAGE_CHECKPOINTS[start_stage]
    asset_dirs = STAGE_ASSET_DIRS[start_stage]

    # Copy room-level checkpoint state and assets.
    for room_dir in source_scene_dir.iterdir():
        if not room_dir.is_dir() or not room_dir.name.startswith("room_"):
            continue

        target_room = target_scene_dir / room_dir.name
        target_room.mkdir(parents=True, exist_ok=True)

        # The selected kit is durable semantic provenance, not agent-session
        # state.  Late-stage resumes still need it for completion validation
        # and for regression/publication evidence.
        room_kit = room_dir / "room_kit.json"
        if room_kit.exists():
            shutil.copy(room_kit, target_room / "room_kit.json")

        # Copy entire checkpoint directory for self-containment.
        # Includes scene_state.json, scene.dmd.yaml, and scene.blend.
        if checkpoint_name:
            source_state = room_dir / "scene_states" / checkpoint_name
            if source_state.exists():
                target_state = target_room / "scene_states" / checkpoint_name
                shutil.copytree(source_state, target_state)

                # Fix absolute paths in scene_state.json.
                _fix_paths_in_json_file(
                    json_path=target_state / "scene_state.json",
                    new_room_dir=target_room,
                    new_scene_dir=target_scene_dir,
                    source_room_dir=room_dir,
                    source_scene_dir=source_scene_dir,
                )

                # Fix absolute paths in scene.dmd.yaml (Drake directives).
                _fix_paths_in_yaml_file(
                    yaml_path=target_state / "scene.dmd.yaml",
                    new_room_dir=target_room,
                    new_scene_dir=target_scene_dir,
                    source_room_dir=room_dir,
                    source_scene_dir=source_scene_dir,
                )

        # Copy required asset directories.
        for asset_subdir in asset_dirs:
            source_assets = room_dir / "generated_assets" / asset_subdir
            if source_assets.exists():
                target_assets = target_room / "generated_assets" / asset_subdir
                shutil.copytree(source_assets, target_assets)

                # Fix absolute paths in asset_registry.json.
                asset_registry = target_assets / "asset_registry.json"
                if asset_registry.exists():
                    _fix_paths_in_json_file(
                        json_path=asset_registry,
                        new_room_dir=target_room,
                        new_scene_dir=target_scene_dir,
                        source_room_dir=room_dir,
                        source_scene_dir=source_scene_dir,
                    )

    console_logger.info(
        f"Copied checkpoint for {start_stage}: "
        f"checkpoint={checkpoint_name}, assets={asset_dirs}"
    )
