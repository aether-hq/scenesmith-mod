"""Failure-safe publication helpers for generated directory products."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid

from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def rebuild_directory_atomically(target: Path, builder: Callable[[Path], T]) -> T:
    """Build a fresh tree and atomically publish it at ``target``.

    The builder never sees the currently published tree. If it fails, its
    staging tree is removed and the previous target remains untouched. The
    final directory rename stays on the target filesystem.
    """

    requested = Path(target).expanduser()
    if requested.is_symlink():
        raise ValueError("Atomic output target must not be a symbolic link")
    target_path = requested.absolute()
    if target_path == Path(target_path.anchor):
        raise ValueError("Atomic output target must not be a filesystem root")
    if target_path.exists() and not target_path.is_dir():
        raise ValueError("Atomic output target must be a directory")
    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target_path.name}.build-", dir=parent))
    backup: Path | None = None
    try:
        result = builder(staging)
        if not staging.is_dir():
            raise RuntimeError("Atomic output builder removed its staging directory")
        if target_path.exists():
            backup = parent / f".{target_path.name}.previous-{uuid.uuid4().hex}"
            os.replace(target_path, backup)
        try:
            os.replace(staging, target_path)
            _fsync_directory(parent)
        except BaseException:
            if backup is not None and backup.exists() and not target_path.exists():
                os.replace(backup, target_path)
                _fsync_directory(parent)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry renames when the platform supports it."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
