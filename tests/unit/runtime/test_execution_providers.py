"""Tests for dependency-light hardware execution provider selection."""

import os
import subprocess
import unittest

from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.runtime.execution_providers import (
    CpuExecutionProvider,
    CudaExecutionProvider,
    ExecutionProviderRegistry,
    ExecutionTarget,
    HardwareInventory,
    MpsExecutionProvider,
    ProviderSelectionContext,
    ProviderUnavailableError,
    detect_cuda_device_ids,
    release_torch_cache,
    resolve_torch_device,
)


class _InjectedProvider:
    key = "injected"

    def targets(self, inventory: HardwareInventory) -> tuple[ExecutionTarget, ...]:
        del inventory
        return (
            ExecutionTarget(
                provider="injected",
                device="injected:cheap",
                worker_id="cheap",
                accelerated=True,
                cost_rank=0,
                performance_rank=50,
            ),
        )


class TestExecutionProviderSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ExecutionProviderRegistry(
            (
                CudaExecutionProvider(),
                MpsExecutionProvider(),
                CpuExecutionProvider(),
            )
        )

    def test_balanced_prefers_local_mps_over_cuda(self) -> None:
        inventory = HardwareInventory(
            system="Darwin",
            machine="arm64",
            cuda_device_ids=(0,),
            mps_available=True,
            cpu_count=12,
        )

        selected = self.registry.select(inventory, policy="balanced")

        self.assertEqual(selected.provider, "mps")
        self.assertEqual(selected.device, "mps")

    def test_performance_prefers_cuda(self) -> None:
        inventory = HardwareInventory(
            system="Linux",
            machine="x86_64",
            cuda_device_ids=(2, 5),
            mps_available=True,
            cpu_count=16,
        )

        selected = self.registry.select(inventory, policy="performance")

        self.assertEqual(selected.provider, "cuda")
        self.assertEqual(selected.device, "cuda:2")

    def test_cost_prefers_local_acceleration_then_cpu(self) -> None:
        accelerated = HardwareInventory(
            system="Darwin",
            machine="arm64",
            cuda_device_ids=(0,),
            mps_available=True,
            cpu_count=10,
        )
        portable = HardwareInventory(
            system="Linux",
            machine="aarch64",
            cuda_device_ids=(),
            mps_available=False,
            cpu_count=8,
        )

        self.assertEqual(
            self.registry.select(accelerated, policy="cost").provider, "mps"
        )
        self.assertEqual(self.registry.select(portable, policy="cost").provider, "cpu")

    def test_explicit_unavailable_provider_fails(self) -> None:
        inventory = HardwareInventory(
            system="Linux",
            machine="x86_64",
            cuda_device_ids=(),
            mps_available=False,
            cpu_count=4,
        )

        with self.assertRaisesRegex(ProviderUnavailableError, "cuda"):
            self.registry.select(inventory, requested="cuda")

    def test_custom_provider_can_be_injected(self) -> None:
        registry = ExecutionProviderRegistry(
            (_InjectedProvider(), CpuExecutionProvider())
        )
        inventory = HardwareInventory(
            system="Plan9",
            machine="mips",
            cuda_device_ids=(),
            mps_available=False,
            cpu_count=1,
        )

        selected = registry.select(inventory, requested="injected")

        self.assertEqual(selected.device, "injected:cheap")

    def test_environment_override_wins_for_torch_device(self) -> None:
        inventory = HardwareInventory(
            system="Darwin",
            machine="arm64",
            cuda_device_ids=(),
            mps_available=True,
            cpu_count=8,
        )

        with patch.dict(os.environ, {"SCENESMITH_COMPUTE_PROVIDER": "cpu"}):
            selected = resolve_torch_device(
                requested="auto", inventory=inventory, registry=self.registry
            )

        self.assertEqual(selected, "cpu")

    def test_injected_empty_environment_prevents_late_policy_override(self) -> None:
        inventory = HardwareInventory(
            system="Darwin",
            machine="arm64",
            cuda_device_ids=(),
            mps_available=True,
            cpu_count=8,
        )

        with patch.dict(os.environ, {"SCENESMITH_COMPUTE_PROVIDER": "cpu"}):
            selected = resolve_torch_device(
                requested="mps",
                environ={},
                inventory=inventory,
                registry=self.registry,
            )

        self.assertEqual(selected, "mps")

    def test_explicit_cuda_device_must_be_visible(self) -> None:
        inventory = HardwareInventory(
            system="Linux",
            machine="x86_64",
            cuda_device_ids=(1, 3),
            mps_available=False,
            cpu_count=8,
        )

        self.assertEqual(
            resolve_torch_device(
                requested="cuda:3", inventory=inventory, registry=self.registry
            ),
            "cuda:3",
        )
        with self.assertRaisesRegex(ProviderUnavailableError, "cuda:2"):
            resolve_torch_device(
                requested="cuda:2", inventory=inventory, registry=self.registry
            )

    def test_last_device_policy_reserves_last_cuda_device(self) -> None:
        inventory = HardwareInventory(
            system="Linux",
            machine="x86_64",
            cuda_device_ids=(0, 1, 2),
            mps_available=False,
            cpu_count=12,
        )

        selected = resolve_torch_device(
            requested="cuda",
            inventory=inventory,
            registry=self.registry,
            device_preference="last",
        )

        self.assertEqual(selected, "cuda:2")

    def test_selection_context_snapshots_all_environment_overrides_once(self) -> None:
        config = {
            "execution_providers": {
                "compute": "cpu",
                "policy": "cost",
                "render": "cpu",
                "geometry": "mlx",
            },
            "geometry_generation_server": {
                "provider": "local",
                "scheme": "https",
                "auth_token_env": "GEOMETRY_TOKEN",
            },
        }
        environment = {
            "SCENESMITH_COMPUTE_PROVIDER": "mps",
            "SCENESMITH_RENDER_PROVIDER": "metal",
            "SCENESMITH_RENDER_PROCESS_PROVIDER": "shared",
            "SCENESMITH_GEOMETRY_PROVIDER": "cuda",
            "SCENESMITH_GEOMETRY_SERVICE_PROVIDER": "external",
            "GEOMETRY_TOKEN": "secret",
        }

        context = ProviderSelectionContext.from_mapping(config, environ=environment)
        environment["SCENESMITH_GEOMETRY_PROVIDER"] = "mlx"

        self.assertEqual(context.compute, "mps")
        self.assertEqual(context.render, "metal")
        self.assertEqual(context.render_process, "shared")
        self.assertEqual(context.geometry, "cuda")
        self.assertEqual(context.geometry_service, "external")
        self.assertEqual(context.external_auth_token, "secret")
        self.assertEqual(context.policy, "cost")


class TestHardwareDetection(unittest.TestCase):
    def test_visible_devices_are_detected_without_torch(self) -> None:
        run = MagicMock()

        detected = detect_cuda_device_ids(
            environ={"CUDA_VISIBLE_DEVICES": "3, 7"}, run=run
        )

        self.assertEqual(detected, (3, 7))
        run.assert_not_called()

    def test_explicit_empty_visibility_disables_cuda_without_probing(self) -> None:
        run = MagicMock()

        detected = detect_cuda_device_ids(environ={"CUDA_VISIBLE_DEVICES": ""}, run=run)

        self.assertEqual(detected, ())
        run.assert_not_called()

    def test_standard_disabled_visibility_values_disable_cuda(self) -> None:
        for value in ("-1", "NoDevFiles", "  -1  "):
            with self.subTest(value=value):
                run = MagicMock()

                detected = detect_cuda_device_ids(
                    environ={"CUDA_VISIBLE_DEVICES": value}, run=run
                )

                self.assertEqual(detected, ())
                run.assert_not_called()

    def test_uuid_and_mig_visibility_tokens_are_preserved(self) -> None:
        run = MagicMock()

        detected = detect_cuda_device_ids(
            environ={"CUDA_VISIBLE_DEVICES": "GPU-deadbeef,MIG-GPU-cafe/1/0"},
            run=run,
        )

        self.assertEqual(detected, ("GPU-deadbeef", "MIG-GPU-cafe/1/0"))
        run.assert_not_called()

    def test_hardware_inventory_uses_logical_ordinals_for_visibility_tokens(
        self,
    ) -> None:
        inventory = HardwareInventory.detect(
            environ={"CUDA_VISIBLE_DEVICES": "GPU-deadbeef,MIG-GPU-cafe/1/0"},
            run=MagicMock(),
            probe_mps=False,
        )

        self.assertEqual(inventory.cuda_device_ids, (0, 1))

    def test_nvidia_smi_devices_are_detected(self) -> None:
        run = MagicMock(
            return_value=MagicMock(returncode=0, stdout="0\n2\n", stderr="")
        )

        detected = detect_cuda_device_ids(environ={}, run=run)

        self.assertEqual(detected, (0, 2))

    def test_missing_cuda_does_not_invent_device_zero(self) -> None:
        run = MagicMock(side_effect=FileNotFoundError("nvidia-smi"))

        detected = detect_cuda_device_ids(environ={}, run=run)

        self.assertEqual(detected, ())

    def test_malformed_visible_devices_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
            detect_cuda_device_ids(
                environ={"CUDA_VISIBLE_DEVICES": "0,banana"}, run=MagicMock()
            )

    def test_timeout_means_no_cuda_provider(self) -> None:
        run = MagicMock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 5))

        self.assertEqual(detect_cuda_device_ids(environ={}, run=run), ())

    def test_torch_runtime_fallback_supports_hosts_without_nvidia_smi(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        fake_torch.backends.mps.is_available.return_value = False

        inventory = HardwareInventory.detect(
            environ={},
            run=MagicMock(side_effect=FileNotFoundError("nvidia-smi")),
            torch_module=fake_torch,
        )

        self.assertEqual(inventory.cuda_device_ids, (0, 1))

    def test_torch_runtime_fallback_never_invents_cuda_when_unavailable(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = False

        inventory = HardwareInventory.detect(
            environ={},
            run=MagicMock(side_effect=FileNotFoundError("nvidia-smi")),
            torch_module=fake_torch,
        )

        self.assertEqual(inventory.cuda_device_ids, ())
        fake_torch.cuda.device_count.assert_not_called()


class TestTorchCacheRelease(unittest.TestCase):
    def test_releases_cuda_and_mps_without_cross_backend_access(self) -> None:
        fake_torch = MagicMock()

        release_torch_cache("cuda:0", torch_module=fake_torch)
        fake_torch.cuda.empty_cache.assert_called_once_with()
        fake_torch.mps.empty_cache.assert_not_called()

        fake_torch.reset_mock()
        release_torch_cache("mps", torch_module=fake_torch)
        fake_torch.mps.empty_cache.assert_called_once_with()
        fake_torch.cuda.empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
