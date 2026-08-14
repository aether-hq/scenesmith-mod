"""Resolve native SceneSmith artifact paths after worker-output relocation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_scene_path(value: str | Path, scene_root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if not path.is_absolute():
        return scene_root / path
    parts = path.parts
    if scene_root.name in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index(scene_root.name)
        return scene_root.joinpath(*parts[index + 1 :])
    return path


def composite_geometry_paths(
    item: dict[str, Any], scene_root: Path
) -> tuple[Path, ...]:
    metadata = item.get("metadata") or {}
    if not metadata.get("composite_type"):
        return ()
    members = [metadata.get("container_asset"), *metadata.get("fill_assets", ())]
    paths = []
    for member in members:
        if isinstance(member, dict) and isinstance(member.get("geometry_path"), str):
            paths.append(resolve_scene_path(member["geometry_path"], scene_root))
    return tuple(paths)
