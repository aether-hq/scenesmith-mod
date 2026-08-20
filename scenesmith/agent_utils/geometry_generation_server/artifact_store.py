"""Bounded, atomic, content-addressed storage for geometry transport."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_ARTIFACT_ID = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    filename: str
    size_bytes: int


class ArtifactStore:
    """Store immutable artifacts under their SHA-256 digest.

    Bytes are written to a private staging file, fsynced, and atomically moved
    into the object directory only after their size and digest are known.
    """

    def __init__(self, root: Path, *, max_artifact_bytes: int = 512 * 1024 * 1024):
        if max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")
        self.root = root.expanduser().resolve()
        self.objects = self.root / "objects"
        self.staging = self.root / "staging"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.max_artifact_bytes = max_artifact_bytes

    def publish_path(
        self, source: Path, *, filename: str | None = None
    ) -> ArtifactRecord:
        source = source.expanduser().resolve(strict=True)
        with source.open("rb") as stream:
            return self.publish_stream(stream, filename=filename or source.name)

    def publish_stream(
        self, stream: BinaryIO, *, filename: str = "artifact.bin"
    ) -> ArtifactRecord:
        safe_filename = Path(filename).name or "artifact.bin"
        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.staging, prefix="upload-", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_artifact_bytes:
                        raise ValueError(
                            f"Artifact exceeds {self.max_artifact_bytes}-byte budget"
                        )
                    digest.update(chunk)
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())

            artifact_id = digest.hexdigest()
            destination = self.objects / artifact_id
            if destination.exists():
                temporary_path.unlink()
            else:
                os.replace(temporary_path, destination)
            return ArtifactRecord(
                artifact_id=artifact_id,
                filename=safe_filename,
                size_bytes=size,
            )
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def resolve(self, artifact_id: str) -> Path:
        normalized = str(artifact_id).strip().lower()
        if not _ARTIFACT_ID.fullmatch(normalized):
            raise ValueError("Invalid artifact identifier")
        path = self.objects / normalized
        if not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {normalized}")
        return path
