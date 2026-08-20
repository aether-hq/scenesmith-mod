"""Injectable Blender Cycles render provider selection."""

from __future__ import annotations

import os
import platform

from typing import Any, Sequence


class BlenderProviderUnavailableError(RuntimeError):
    """Raised when an explicitly requested Cycles provider is unavailable."""


_COMPUTE_TYPES = {
    "optix": "OPTIX",
    "cuda": "CUDA",
    "hip": "HIP",
    "oneapi": "ONEAPI",
    "metal": "METAL",
}


def configure_cycles_provider(
    *,
    preferences: Any,
    scene: Any,
    requested: str = "auto",
    system: str | None = None,
    machine: str | None = None,
    provider_order: Sequence[str] | None = None,
) -> str:
    """Configure and return the selected Cycles provider.

    Provider-specific probing stays behind this function. Explicit requests
    fail when unavailable; ``auto`` tries every supported vendor API and then
    writes an explicit CPU state.
    """

    del machine  # Kept as an injectable probe dimension for future backends.
    effective = (
        (os.environ.get("SCENESMITH_RENDER_PROVIDER") or requested or "auto")
        .strip()
        .lower()
    )
    aliases = {"nvidia": "optix", "amd": "hip", "intel": "oneapi"}
    effective = aliases.get(effective, effective)
    if effective == "cpu":
        _configure_cpu(scene=scene)
        return "cpu"
    if effective != "auto" and effective not in _COMPUTE_TYPES:
        expected = ", ".join((*_COMPUTE_TYPES, "cpu"))
        raise ValueError(
            f"Unknown Blender render provider '{effective}'. Expected auto or "
            f"one of: {expected}."
        )

    if provider_order is None:
        current_system = (system or platform.system()).lower()
        provider_order = (
            ("metal", "optix", "cuda", "hip", "oneapi")
            if current_system == "darwin"
            else ("optix", "cuda", "hip", "oneapi", "metal")
        )
    candidates = (effective,) if effective != "auto" else tuple(provider_order)
    for provider in candidates:
        if provider not in _COMPUTE_TYPES:
            raise ValueError(f"Unknown Blender render provider '{provider}'")
        if _try_enable_provider(
            preferences=preferences,
            scene=scene,
            provider=provider,
        ):
            return provider

    if effective != "auto":
        raise BlenderProviderUnavailableError(
            f"Blender render provider '{effective}' was requested but no compatible "
            "device is available."
        )
    _configure_cpu(scene=scene)
    return "cpu"


def _try_enable_provider(*, preferences: Any, scene: Any, provider: str) -> bool:
    try:
        preferences.compute_device_type = _COMPUTE_TYPES[provider]
        preferences.get_devices()
        enabled = False
        expected_type = _COMPUTE_TYPES[provider]
        for device in preferences.devices:
            is_provider_device = str(device.type).upper() == expected_type
            device.use = is_provider_device
            enabled = enabled or is_provider_device
        if not enabled:
            return False
        scene.cycles.device = "GPU"
        scene.render.use_persistent_data = True
        return True
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False


def _configure_cpu(*, scene: Any) -> None:
    scene.cycles.device = "CPU"
    scene.render.use_persistent_data = False
