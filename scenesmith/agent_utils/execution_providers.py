"""Injectable, dependency-light execution provider selection.

This module owns *selection* of general compute devices. Backend-specific code
is allowed to mention CUDA, MPS, or CPU inside its provider implementation, but
callers should request a capability through :class:`ExecutionProviderRegistry`
instead of probing a vendor runtime themselves.

The module intentionally imports neither PyTorch nor any accelerator SDK at
module import time. This keeps it safe in parents that must fork workers before
an accelerator runtime is initialized.
"""

from __future__ import annotations

import os
import platform
import subprocess

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, TypeAlias

SelectionPolicy = Literal["balanced", "performance", "cost"]
DevicePreference = Literal["first", "last"]
CudaVisibilityToken: TypeAlias = int | str


class ProviderUnavailableError(RuntimeError):
    """Raised when an explicitly requested provider or device is unavailable."""


@dataclass(frozen=True)
class ProviderSelectionContext:
    """Immutable provider choices resolved once at the composition root."""

    compute: str
    policy: SelectionPolicy
    render: str
    render_process: str
    geometry: str
    geometry_service: str
    external_scheme: str
    external_auth_token: str | None

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | Any,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "ProviderSelectionContext":
        current_environ = os.environ if environ is None else environ
        providers = _mapping_get(config, "execution_providers", {})
        geometry_server = _mapping_get(config, "geometry_generation_server", {})
        policy = str(_mapping_get(providers, "policy", "balanced")).lower()
        if policy not in {"balanced", "performance", "cost"}:
            raise ValueError(
                f"Unknown execution provider policy '{policy}'. Expected balanced, "
                "performance, or cost."
            )
        auth_token_env = str(
            _mapping_get(geometry_server, "auth_token_env", "SCENESMITH_GEOMETRY_TOKEN")
        )
        return cls(
            compute=str(
                current_environ.get("SCENESMITH_COMPUTE_PROVIDER")
                or _mapping_get(providers, "compute", "auto")
            ).lower(),
            policy=policy,  # type: ignore[arg-type]
            render=str(
                current_environ.get("SCENESMITH_RENDER_PROVIDER")
                or _mapping_get(providers, "render", "auto")
            ).lower(),
            render_process=str(
                current_environ.get("SCENESMITH_RENDER_PROCESS_PROVIDER")
                or _mapping_get(providers, "render_process", "auto")
            ).lower(),
            geometry=str(
                current_environ.get("SCENESMITH_GEOMETRY_PROVIDER")
                or _mapping_get(providers, "geometry", "auto")
            ).lower(),
            geometry_service=str(
                current_environ.get("SCENESMITH_GEOMETRY_SERVICE_PROVIDER")
                or _mapping_get(geometry_server, "provider", "local")
            ).lower(),
            external_scheme=str(
                _mapping_get(geometry_server, "scheme", "https")
            ).lower(),
            external_auth_token=current_environ.get(auth_token_env),
        )


@dataclass(frozen=True)
class HardwareInventory:
    """A side-effect-free snapshot used by execution providers.

    ``cuda_device_ids`` are device indices addressable by the current process.
    Geometry worker isolation uses :func:`detect_cuda_device_ids` directly when
    it needs the physical IDs from ``CUDA_VISIBLE_DEVICES``.
    """

    system: str
    machine: str
    cuda_device_ids: tuple[int, ...]
    mps_available: bool
    cpu_count: int

    @classmethod
    def detect(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        run: Callable[..., Any] = subprocess.run,
        torch_module: Any | None = None,
        probe_mps: bool = True,
    ) -> "HardwareInventory":
        """Detect locally available providers without importing SDKs eagerly."""

        current_environ = os.environ if environ is None else environ
        physical_cuda_ids = detect_cuda_device_ids(
            environ=current_environ,
            run=run,
        )
        visible_value = current_environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if physical_cuda_ids:
            cuda_device_ids = (
                tuple(range(len(physical_cuda_ids)))
                if visible_value
                else physical_cuda_ids
            )
        else:
            # nvidia-smi is not universal (notably on Windows and some managed
            # runtimes). General Torch inference can use its own runtime probe;
            # geometry isolation still requires physical IDs from the provider.
            cuda_device_ids = _detect_torch_cuda_device_ids(torch_module=torch_module)
        mps_available = False
        if probe_mps:
            mps_available = _detect_mps_available(torch_module=torch_module)
        return cls(
            system=platform.system(),
            machine=platform.machine(),
            cuda_device_ids=cuda_device_ids,
            mps_available=mps_available,
            cpu_count=max(1, os.cpu_count() or 1),
        )


@dataclass(frozen=True)
class ExecutionTarget:
    """One concrete execution target offered by a provider."""

    provider: str
    device: str
    worker_id: int | str | None
    accelerated: bool
    cost_rank: int
    performance_rank: int


class ExecutionProvider(Protocol):
    """Provider interface accepted by :class:`ExecutionProviderRegistry`."""

    key: str

    def targets(self, inventory: HardwareInventory) -> tuple[ExecutionTarget, ...]:
        """Return concrete targets available in ``inventory``."""


class CudaExecutionProvider:
    """NVIDIA CUDA provider for general Torch inference."""

    key = "cuda"

    def targets(self, inventory: HardwareInventory) -> tuple[ExecutionTarget, ...]:
        return tuple(
            ExecutionTarget(
                provider=self.key,
                device=f"cuda:{device_id}",
                worker_id=device_id,
                accelerated=True,
                cost_rank=2,
                performance_rank=100,
            )
            for device_id in inventory.cuda_device_ids
        )


class MpsExecutionProvider:
    """Apple Metal Performance Shaders provider for general Torch inference."""

    key = "mps"

    def targets(self, inventory: HardwareInventory) -> tuple[ExecutionTarget, ...]:
        if not inventory.mps_available:
            return ()
        return (
            ExecutionTarget(
                provider=self.key,
                device="mps",
                worker_id=0,
                accelerated=True,
                cost_rank=0,
                performance_rank=80,
            ),
        )


class CpuExecutionProvider:
    """Portable CPU provider."""

    key = "cpu"

    def targets(self, inventory: HardwareInventory) -> tuple[ExecutionTarget, ...]:
        del inventory
        return (
            ExecutionTarget(
                provider=self.key,
                device="cpu",
                worker_id=None,
                accelerated=False,
                cost_rank=0,
                performance_rank=10,
            ),
        )


class ExecutionProviderRegistry:
    """Injectable registry that resolves providers using an explicit policy."""

    def __init__(self, providers: Sequence[ExecutionProvider]) -> None:
        provider_map = {provider.key: provider for provider in providers}
        if len(provider_map) != len(providers):
            raise ValueError("Execution provider keys must be unique")
        if not provider_map:
            raise ValueError("At least one execution provider is required")
        self._providers = provider_map

    @property
    def provider_keys(self) -> tuple[str, ...]:
        return tuple(self._providers)

    def targets_for(
        self, inventory: HardwareInventory, provider: str
    ) -> tuple[ExecutionTarget, ...]:
        normalized = _normalize_provider(provider)
        implementation = self._providers.get(normalized)
        if implementation is None:
            expected = ", ".join(sorted(self._providers))
            raise ValueError(
                f"Unknown execution provider '{provider}'. Expected auto or {expected}."
            )
        return implementation.targets(inventory)

    def select(
        self,
        inventory: HardwareInventory,
        *,
        requested: str = "auto",
        policy: SelectionPolicy = "balanced",
        device_preference: DevicePreference = "first",
    ) -> ExecutionTarget:
        """Select one available target or fail for an unavailable override."""

        normalized = _normalize_provider(requested)
        if policy not in {"balanced", "performance", "cost"}:
            raise ValueError(
                f"Unknown execution provider policy '{policy}'. Expected balanced, "
                "performance, or cost."
            )
        if device_preference not in {"first", "last"}:
            raise ValueError("device_preference must be 'first' or 'last'")

        if normalized != "auto":
            targets = self.targets_for(inventory, normalized)
            if not targets:
                raise ProviderUnavailableError(
                    f"Execution provider '{normalized}' was requested but is not "
                    "available on this host."
                )
            return targets[-1] if device_preference == "last" else targets[0]

        candidates = tuple(
            target
            for provider in self._providers.values()
            for target in provider.targets(inventory)
        )
        if not candidates:
            raise ProviderUnavailableError("No execution providers are available")
        if policy == "performance":
            rank = lambda target: (  # noqa: E731
                -target.performance_rank,
                target.cost_rank,
                target.provider,
                str(target.worker_id),
            )
        elif policy == "cost":
            rank = lambda target: (  # noqa: E731
                target.cost_rank,
                -target.performance_rank,
                target.provider,
                str(target.worker_id),
            )
        else:
            rank = lambda target: (  # noqa: E731
                -(target.performance_rank - target.cost_rank * 25),
                target.cost_rank,
                target.provider,
                str(target.worker_id),
            )
        return sorted(candidates, key=rank)[0]


def default_execution_provider_registry() -> ExecutionProviderRegistry:
    """Return SceneSmith's built-in general compute providers."""

    return ExecutionProviderRegistry(
        (
            CudaExecutionProvider(),
            MpsExecutionProvider(),
            CpuExecutionProvider(),
        )
    )


def resolve_torch_device(
    requested: str | None = None,
    *,
    policy: SelectionPolicy = "balanced",
    inventory: HardwareInventory | None = None,
    registry: ExecutionProviderRegistry | None = None,
    device_preference: DevicePreference = "first",
    torch_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a Torch device through the shared execution provider registry."""

    current_environ = os.environ if environ is None else environ
    effective = (
        current_environ.get("SCENESMITH_COMPUTE_PROVIDER") or requested or "auto"
    )
    current_inventory = inventory or HardwareInventory.detect(torch_module=torch_module)
    current_registry = registry or default_execution_provider_registry()
    normalized = str(effective).strip().lower()
    if normalized.startswith("cuda:"):
        try:
            requested_id = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device '{effective}'") from exc
        targets = current_registry.targets_for(current_inventory, "cuda")
        for target in targets:
            if target.worker_id == requested_id:
                return target.device
        raise ProviderUnavailableError(
            f"Execution device 'cuda:{requested_id}' was requested but is not visible."
        )
    return current_registry.select(
        current_inventory,
        requested=normalized,
        policy=policy,
        device_preference=device_preference,
    ).device


def detect_cuda_device_ids(
    *,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> tuple[CudaVisibilityToken, ...]:
    """Detect CUDA visibility tokens without initializing an accelerator runtime.

    An explicitly present ``CUDA_VISIBLE_DEVICES`` is authoritative, including
    its empty and standard disabled forms.  Integer indices, GPU UUIDs, and MIG
    identifiers are all valid visibility tokens.  The function name is kept for
    compatibility; callers must not assume every token is an integer device ID.
    """

    current_environ = os.environ if environ is None else environ
    visible = current_environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        normalized = visible.strip()
        if not normalized or normalized.lower() in {"-1", "nodevfiles"}:
            return ()
        raw_tokens = tuple(value.strip() for value in visible.split(","))
        if any(not token for token in raw_tokens):
            raise ValueError(
                f"Invalid CUDA_VISIBLE_DEVICES='{visible}'; empty device token."
            )
        tokens: tuple[CudaVisibilityToken, ...] = tuple(
            int(token) if token.isdecimal() else token for token in raw_tokens
        )
        if any(isinstance(token, int) and token < 0 for token in tokens) or len(
            set(tokens)
        ) != len(tokens):
            raise ValueError(
                f"Invalid CUDA_VISIBLE_DEVICES='{visible}'; tokens must be unique "
                "and non-negative."
            )
        for token in tokens:
            if isinstance(token, str) and not token.startswith(("GPU-", "MIG-")):
                raise ValueError(
                    f"Invalid CUDA_VISIBLE_DEVICES='{visible}'; expected integer, "
                    "GPU UUID, or MIG identifier tokens."
                )
        return tokens

    try:
        result = run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ()
    if result.returncode != 0:
        return ()
    try:
        return tuple(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )
    except ValueError:
        return ()


def release_torch_cache(device: str, *, torch_module: Any | None = None) -> None:
    """Release unused cache for the selected Torch provider only."""

    torch = torch_module
    if torch is None:
        import torch as imported_torch

        torch = imported_torch
    normalized = device.strip().lower()
    if normalized == "cuda" or normalized.startswith("cuda:"):
        torch.cuda.empty_cache()
    elif normalized == "mps":
        torch.mps.empty_cache()


def _detect_mps_available(*, torch_module: Any | None = None) -> bool:
    torch = torch_module
    try:
        if torch is None:
            import torch as imported_torch

            torch = imported_torch
        return bool(torch.backends.mps.is_available())
    except (AttributeError, ImportError, RuntimeError):
        return False


def _detect_torch_cuda_device_ids(
    *, torch_module: Any | None = None
) -> tuple[int, ...]:
    torch = torch_module
    try:
        if torch is None:
            import torch as imported_torch

            torch = imported_torch
        if not torch.cuda.is_available():
            return ()
        return tuple(range(int(torch.cuda.device_count())))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return ()


def _normalize_provider(provider: str) -> str:
    normalized = str(provider).strip().lower()
    aliases = {
        "apple": "mps",
        "metal": "mps",
        "mlx": "mps",
        "nvidia": "cuda",
    }
    return aliases.get(normalized, normalized)


def _mapping_get(config: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
