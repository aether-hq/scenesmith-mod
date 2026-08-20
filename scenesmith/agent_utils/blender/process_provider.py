"""Subprocess providers for Blender server isolation.

Rendering backend selection and process isolation are separate concerns. Metal,
HIP, oneAPI, and CPU renderers use a normal shared process. Linux/NVIDIA hosts
may inject bubblewrap isolation so parallel Blender processes see one device.
"""

from __future__ import annotations

import logging
import os
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from scenesmith.agent_utils.execution_providers import (
    ProviderUnavailableError,
    detect_cuda_device_ids,
)

console_logger = logging.getLogger(__name__)


class RenderProcessProvider(Protocol):
    """Prepare a Blender server command for its execution environment."""

    key: str

    def prepare(
        self, command: list[str], environment: Mapping[str, str]
    ) -> "PreparedProcess": ...


@dataclass(frozen=True)
class PreparedProcess:
    """Atomic child command and environment produced by one provider."""

    command: tuple[str, ...]
    environment: Mapping[str, str]


@dataclass(frozen=True)
class RenderAllocation:
    """Provider-neutral render slot handed to experiment orchestration."""

    slot_id: str
    target_label: str
    render_provider: str
    process_provider: RenderProcessProvider


class SharedRenderProcessProvider:
    """Portable process provider with no device-file isolation."""

    key = "shared"

    def prepare(
        self, command: list[str], environment: Mapping[str, str]
    ) -> PreparedProcess:
        return PreparedProcess(tuple(command), dict(environment))


class NvidiaEnvironmentProcessProvider:
    """Enforce one CUDA visibility token through the child environment."""

    key = "nvidia-environment"

    def __init__(self, visibility_token: int | str) -> None:
        token = str(visibility_token).strip()
        if not token:
            raise ValueError("visibility_token must not be empty")
        self.visibility_token = token

    def prepare(
        self, command: list[str], environment: Mapping[str, str]
    ) -> PreparedProcess:
        prepared_environment = dict(environment)
        prepared_environment["CUDA_VISIBLE_DEVICES"] = self.visibility_token
        return PreparedProcess(tuple(command), prepared_environment)


class NvidiaBwrapProcessProvider(NvidiaEnvironmentProcessProvider):
    """Linux bubblewrap isolation for one NVIDIA/Vulkan device."""

    key = "nvidia-bwrap"

    def prepare(
        self, command: list[str], environment: Mapping[str, str]
    ) -> PreparedProcess:
        prepared = super().prepare(command, environment)
        home_dir = Path.home()
        cwd = Path.cwd()
        wrapped = [
            "bwrap",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(home_dir),
            str(home_dir),
            "--bind",
            "/tmp",
            "/tmp",
            "--bind",
            "/dev/shm",
            "/dev/shm",
            "--proc",
            "/proc",
            "--dev-bind",
            "/dev/urandom",
            "/dev/urandom",
            "--dev-bind",
            "/dev/null",
            "/dev/null",
        ]
        if cwd != home_dir:
            wrapped.extend(["--bind", str(cwd), str(cwd)])
        _append_nested_mounts(wrapped, cwd)

        devices = [
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            f"/dev/nvidia{self.visibility_token}",
        ]
        for device in devices:
            if Path(device).exists():
                wrapped.extend(["--dev-bind", device, device])
        if Path("/dev/dri").exists():
            wrapped.extend(["--dev-bind", "/dev/dri", "/dev/dri"])
        return PreparedProcess(
            command=tuple([*wrapped, "--", *prepared.command]),
            environment=prepared.environment,
        )


def _append_nested_mounts(command: list[str], cwd: Path) -> None:
    """Re-bind container volume mounts nested beneath the working directory."""

    try:
        cwd_prefix = f"{cwd}/"
        with Path("/proc/mounts").open() as mounts:
            for line in mounts:
                parts = line.split()
                if len(parts) < 2:
                    continue
                mount_point = parts[1]
                if mount_point.startswith(cwd_prefix) and Path(mount_point).exists():
                    command.extend(["--bind", mount_point, mount_point])
    except OSError:
        return


def resolve_render_process_provider(
    visibility_token: int | str | None,
    *,
    requested: str | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> RenderProcessProvider:
    """Resolve portable shared execution or Linux/NVIDIA device isolation."""

    current_environ = os.environ if environ is None else environ
    requested = str(
        requested or current_environ.get("SCENESMITH_RENDER_PROCESS_PROVIDER") or "auto"
    ).lower()
    aliases = {"none": "shared", "nvidia": "nvidia-bwrap", "bwrap": "nvidia-bwrap"}
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "shared", "nvidia-bwrap"}:
        raise ValueError(
            f"Unknown render process provider '{requested}'. Expected auto, shared, "
            "or nvidia-bwrap."
        )
    if requested == "shared" or visibility_token is None:
        if requested == "nvidia-bwrap" and visibility_token is None:
            raise ProviderUnavailableError(
                "The nvidia-bwrap process provider requires a device ID."
            )
        return SharedRenderProcessProvider()
    if which("bwrap"):
        return NvidiaBwrapProcessProvider(visibility_token)
    if requested == "nvidia-bwrap":
        raise ProviderUnavailableError(
            "The nvidia-bwrap process provider requires bubblewrap ('bwrap')."
        )
    console_logger.info(
        "bubblewrap is unavailable; enforcing render allocation %s through the "
        "child visibility environment",
        visibility_token,
    )
    return NvidiaEnvironmentProcessProvider(visibility_token)


def render_allocations(
    requested_render_provider: str | None = None,
    *,
    requested_process_provider: str | None = None,
) -> tuple[RenderAllocation, ...]:
    """Resolve immutable, provider-neutral Blender render allocations."""

    requested = (
        requested_render_provider
        or os.environ.get("SCENESMITH_RENDER_PROVIDER")
        or "auto"
    ).lower()
    aliases = {"amd": "hip", "intel": "oneapi", "nvidia": "optix"}
    requested = aliases.get(requested, requested)
    if requested in {"cpu", "metal", "hip", "oneapi"}:
        return (
            RenderAllocation(
                slot_id="render-0",
                target_label=f"{requested}/shared",
                render_provider=requested,
                process_provider=SharedRenderProcessProvider(),
            ),
        )
    visibility_tokens = detect_cuda_device_ids()
    if not visibility_tokens:
        return (
            RenderAllocation(
                slot_id="render-0",
                target_label="auto/shared",
                render_provider=requested,
                process_provider=SharedRenderProcessProvider(),
            ),
        )
    return tuple(
        RenderAllocation(
            slot_id=f"render-{ordinal}",
            target_label=f"cuda/{token}",
            render_provider=requested,
            process_provider=resolve_render_process_provider(
                token,
                requested=requested_process_provider,
                environ={},
            ),
        )
        for ordinal, token in enumerate(visibility_tokens)
    )
