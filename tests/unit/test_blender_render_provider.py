"""Tests for injectable Blender Cycles render provider selection."""

import os
import unittest

from dataclasses import dataclass
from unittest.mock import patch

from scenesmith.agent_utils.blender.render_provider import (
    BlenderProviderUnavailableError,
    configure_cycles_provider,
)


@dataclass
class _Device:
    type: str
    use: bool = False


class _Preferences:
    def __init__(self, devices_by_provider: dict[str, list[_Device]]) -> None:
        self.devices_by_provider = devices_by_provider
        self.devices: list[_Device] = []
        self.attempts: list[str] = []
        self._compute_device_type = "NONE"

    @property
    def compute_device_type(self) -> str:
        return self._compute_device_type

    @compute_device_type.setter
    def compute_device_type(self, value: str) -> None:
        self.attempts.append(value)
        if value not in self.devices_by_provider:
            raise TypeError(f"unsupported {value}")
        self._compute_device_type = value

    def get_devices(self) -> None:
        self.devices = self.devices_by_provider[self._compute_device_type]


class _Scene:
    class _Cycles:
        device = "CPU"

    class _Render:
        use_persistent_data = False

    def __init__(self) -> None:
        self.cycles = self._Cycles()
        self.render = self._Render()


class TestBlenderRenderProvider(unittest.TestCase):
    def test_auto_prefers_metal_on_apple_silicon(self) -> None:
        preferences = _Preferences({"METAL": [_Device("METAL"), _Device("CPU")]})
        scene = _Scene()

        selected = configure_cycles_provider(
            preferences=preferences,
            scene=scene,
            requested="auto",
            system="Darwin",
            machine="arm64",
        )

        self.assertEqual(selected, "metal")
        self.assertEqual(preferences.attempts[0], "METAL")
        self.assertEqual(scene.cycles.device, "GPU")
        self.assertTrue(scene.render.use_persistent_data)

    def test_auto_supports_amd_and_intel_before_cpu(self) -> None:
        amd_preferences = _Preferences({"HIP": [_Device("HIP")]})
        intel_preferences = _Preferences({"ONEAPI": [_Device("ONEAPI")]})

        self.assertEqual(
            configure_cycles_provider(
                preferences=amd_preferences,
                scene=_Scene(),
                requested="auto",
                system="Linux",
                machine="x86_64",
            ),
            "hip",
        )
        self.assertEqual(
            configure_cycles_provider(
                preferences=intel_preferences,
                scene=_Scene(),
                requested="auto",
                system="Linux",
                machine="x86_64",
            ),
            "oneapi",
        )

    def test_auto_falls_back_to_explicit_cpu_state(self) -> None:
        preferences = _Preferences({})
        scene = _Scene()

        selected = configure_cycles_provider(
            preferences=preferences,
            scene=scene,
            requested="auto",
            system="Linux",
            machine="aarch64",
        )

        self.assertEqual(selected, "cpu")
        self.assertEqual(scene.cycles.device, "CPU")
        self.assertFalse(scene.render.use_persistent_data)

    def test_explicit_unavailable_provider_fails(self) -> None:
        with self.assertRaisesRegex(BlenderProviderUnavailableError, "metal"):
            configure_cycles_provider(
                preferences=_Preferences({}),
                scene=_Scene(),
                requested="metal",
                system="Darwin",
                machine="arm64",
            )

    def test_provider_enables_only_matching_device_type(self) -> None:
        metal = _Device("METAL")
        unrelated = _Device("CUDA")
        cpu = _Device("CPU")
        preferences = _Preferences({"METAL": [metal, unrelated, cpu]})

        configure_cycles_provider(
            preferences=preferences,
            scene=_Scene(),
            requested="metal",
            system="Darwin",
            machine="arm64",
        )

        self.assertTrue(metal.use)
        self.assertFalse(unrelated.use)
        self.assertFalse(cpu.use)

    def test_environment_override_wins(self) -> None:
        scene = _Scene()
        with patch.dict(os.environ, {"SCENESMITH_RENDER_PROVIDER": "cpu"}):
            selected = configure_cycles_provider(
                preferences=_Preferences({"OPTIX": [_Device("OPTIX")]}),
                scene=scene,
                requested="auto",
                system="Linux",
                machine="x86_64",
            )

        self.assertEqual(selected, "cpu")
        self.assertEqual(scene.cycles.device, "CPU")


if __name__ == "__main__":
    unittest.main()
