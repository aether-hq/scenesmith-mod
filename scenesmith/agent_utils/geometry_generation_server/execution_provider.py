"""Execution providers for local geometry-generation workers.

Model implementations remain backend-specific, but worker topology and process
environment are injected through this provider contract. This prevents the
server from assuming that every worker is an NVIDIA GPU.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Protocol

from scenesmith.agent_utils.execution_providers import (
    CudaVisibilityToken,
    ProviderUnavailableError,
    detect_cuda_device_ids,
)


@dataclass(frozen=True)
class GeometryWorkerTarget:
    """One isolated local worker target."""

    worker_id: str
    provider: str
    device_id: CudaVisibilityToken | str | None
    label: str
    environment: tuple[tuple[str, str | None], ...]


class GeometryExecutionProvider(Protocol):
    """Provider interface consumed by the geometry worker pool."""

    key: str

    def targets(self) -> tuple[GeometryWorkerTarget, ...]:
        """Return local workers or fail when the provider is unavailable."""

    def process_start_method(self) -> str:
        """Return a portable multiprocessing start strategy."""


class CudaGeometryExecutionProvider:
    """One isolated geometry worker per detected NVIDIA CUDA device."""

    key = "cuda"

    def __init__(
        self,
        *,
        detector: Callable[
            [], tuple[CudaVisibilityToken, ...]
        ] = detect_cuda_device_ids,
    ) -> None:
        self._detector = detector

    def targets(self) -> tuple[GeometryWorkerTarget, ...]:
        device_ids = self._detector()
        if not device_ids:
            raise ProviderUnavailableError(
                "CUDA geometry generation was selected, but no CUDA devices were "
                "detected. Select the MLX provider on Apple Silicon, use a remote "
                "geometry server, or configure an asset-retrieval source."
            )
        return tuple(
            GeometryWorkerTarget(
                worker_id=f"worker-{ordinal}",
                provider=self.key,
                device_id=device_id,
                label=f"cuda/{device_id}",
                environment=(("CUDA_VISIBLE_DEVICES", str(device_id)),),
            )
            for ordinal, device_id in enumerate(device_ids)
        )

    def process_start_method(self) -> str:
        """Spawn clean interpreters so no parent accelerator state is inherited."""

        return "spawn"


class MlxGeometryExecutionProvider:
    """One process-isolated SAM3D MLX/Metal worker."""

    key = "mlx"

    def targets(self) -> tuple[GeometryWorkerTarget, ...]:
        return (
            GeometryWorkerTarget(
                worker_id="worker-0",
                provider=self.key,
                device_id="metal",
                label="mlx/metal",
                environment=(("CUDA_VISIBLE_DEVICES", None),),
            ),
        )

    def process_start_method(self) -> str:
        return "spawn"


def resolve_geometry_execution_provider(
    *,
    backend: str,
    sam3d_config: dict | None,
    requested: str | None = None,
    cuda_detector: Callable[
        [], tuple[CudaVisibilityToken, ...]
    ] = detect_cuda_device_ids,
    environ: Mapping[str, str] | None = None,
) -> GeometryExecutionProvider:
    """Resolve the local worker provider for a model backend."""

    current_environ = os.environ if environ is None else environ
    requested = (
        current_environ.get("SCENESMITH_GEOMETRY_PROVIDER") or requested or "auto"
    ).lower()
    aliases = {"apple": "mlx", "metal": "mlx", "mps": "mlx", "nvidia": "cuda"}
    requested = aliases.get(requested, requested)
    normalized_backend = backend.strip().lower()
    if normalized_backend == "sam3d":
        if sam3d_config is None:
            raise ValueError("sam3d_config is required for the SAM3D backend")
        if requested == "auto":
            from scenesmith.agent_utils.geometry_generation_server.sam_provider import (
                resolve_sam_provider,
            )

            requested = resolve_sam_provider(
                sam3d_config,
                environ={},
                cuda_detector=cuda_detector,
            )
    elif normalized_backend == "hunyuan3d":
        if requested == "auto":
            requested = "cuda"
        if requested != "cuda":
            raise ProviderUnavailableError(
                f"Hunyuan3D has no local '{requested}' implementation in SceneSmith. "
                "Use CUDA, a remote geometry server, or SAM3D/MLX on Apple Silicon."
            )
    else:
        raise ValueError(
            f"Unknown geometry backend '{backend}'. Expected hunyuan3d or sam3d."
        )

    if requested == "cuda":
        return CudaGeometryExecutionProvider(detector=cuda_detector)
    if requested == "mlx" and normalized_backend == "sam3d":
        return MlxGeometryExecutionProvider()
    raise ProviderUnavailableError(
        f"Geometry execution provider '{requested}' is unavailable for "
        f"backend '{normalized_backend}'."
    )


def configure_geometry_worker_environment(
    target: GeometryWorkerTarget,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Apply provider-owned environment changes before backend imports."""

    current_environ = os.environ if environ is None else environ
    for name, value in target.environment:
        if value is None:
            current_environ.pop(name, None)
        else:
            current_environ[name] = value
