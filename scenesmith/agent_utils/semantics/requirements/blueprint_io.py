"""Durable persistence for canonical scene blueprints."""

import os
import tempfile

from pathlib import Path

from scenesmith.agent_utils.semantics.requirements.scene_blueprint import SceneBlueprint


def persist_scene_blueprint(blueprint: SceneBlueprint, output_path: Path) -> None:
    """Atomically persist canonical scene intent for resume and revisions."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = blueprint.model_dump_json(indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
