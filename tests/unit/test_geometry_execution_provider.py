"""Tests for injectable geometry worker execution providers."""

import os
import unittest

from unittest.mock import patch

from scenesmith.agent_utils.execution_providers import ProviderUnavailableError
from scenesmith.agent_utils.geometry_generation_server.execution_provider import (
    CudaGeometryExecutionProvider,
    GeometryWorkerTarget,
    MlxGeometryExecutionProvider,
    configure_geometry_worker_environment,
    resolve_geometry_execution_provider,
)
from scenesmith.agent_utils.geometry_generation_server.worker_pool import (
    GeometryWorkerPool,
    GPUWorkerPool,
)


class _InjectedGeometryProvider:
    key = "injected"

    def targets(self) -> tuple[GeometryWorkerTarget, ...]:
        return (
            GeometryWorkerTarget(
                worker_id="injected-0",
                provider="injected",
                device_id="local",
                label="injected/local",
                environment=(),
            ),
        )

    def process_start_method(self) -> str:
        return "spawn"


class TestGeometryExecutionProviders(unittest.TestCase):
    def test_cuda_provider_uses_only_detected_devices(self) -> None:
        provider = CudaGeometryExecutionProvider(detector=lambda: (2, 7))

        targets = provider.targets()

        self.assertEqual(
            [target.worker_id for target in targets], ["worker-0", "worker-1"]
        )
        self.assertEqual(targets[1].device_id, 7)
        self.assertIn(("CUDA_VISIBLE_DEVICES", "7"), targets[1].environment)

    def test_cuda_provider_supports_uuid_visibility_without_leaking_device_ids(
        self,
    ) -> None:
        provider = CudaGeometryExecutionProvider(
            detector=lambda: ("GPU-deadbeef", "MIG-GPU-cafe/1/0")
        )

        targets = provider.targets()

        self.assertEqual(
            [target.worker_id for target in targets], ["worker-0", "worker-1"]
        )
        self.assertEqual(targets[0].device_id, "GPU-deadbeef")
        self.assertIn(
            ("CUDA_VISIBLE_DEVICES", "MIG-GPU-cafe/1/0"),
            targets[1].environment,
        )

    def test_cuda_provider_without_hardware_fails_instead_of_inventing_gpu_zero(
        self,
    ) -> None:
        provider = CudaGeometryExecutionProvider(detector=lambda: ())

        with self.assertRaisesRegex(ProviderUnavailableError, "CUDA"):
            provider.targets()

    def test_mlx_provider_exposes_one_isolated_metal_worker(self) -> None:
        target = MlxGeometryExecutionProvider().targets()[0]

        self.assertEqual(target.provider, "mlx")
        self.assertEqual(target.worker_id, "worker-0")
        self.assertEqual(target.label, "mlx/metal")
        self.assertIn(("CUDA_VISIBLE_DEVICES", None), target.environment)

    def test_worker_environment_is_provider_owned(self) -> None:
        environment = {"CUDA_VISIBLE_DEVICES": "9", "KEEP": "yes"}
        mlx_target = MlxGeometryExecutionProvider().targets()[0]

        configure_geometry_worker_environment(mlx_target, environ=environment)

        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["KEEP"], "yes")

    def test_sam_provider_selects_matching_worker_provider(self) -> None:
        self.assertEqual(
            resolve_geometry_execution_provider(
                backend="sam3d",
                sam3d_config={"provider": "mlx"},
            ).key,
            "mlx",
        )
        self.assertEqual(
            resolve_geometry_execution_provider(
                backend="sam3d",
                sam3d_config={"provider": "cuda"},
                cuda_detector=lambda: (0,),
            ).key,
            "cuda",
        )

    def test_hunyuan_rejects_non_cuda_local_provider(self) -> None:
        with patch.dict(os.environ, {"SCENESMITH_GEOMETRY_PROVIDER": "mlx"}):
            with self.assertRaisesRegex(ProviderUnavailableError, "Hunyuan"):
                resolve_geometry_execution_provider(
                    backend="hunyuan3d",
                    sam3d_config=None,
                    cuda_detector=lambda: (),
                )

    def test_explicit_requested_provider_is_used_without_global_state(self) -> None:
        provider = resolve_geometry_execution_provider(
            backend="sam3d",
            sam3d_config={"provider": "cuda"},
            requested="mlx",
        )

        self.assertEqual(provider.key, "mlx")

    def test_environment_override_wins_over_requested_provider(self) -> None:
        with patch.dict(os.environ, {"SCENESMITH_GEOMETRY_PROVIDER": "mlx"}):
            provider = resolve_geometry_execution_provider(
                backend="sam3d",
                sam3d_config={"provider": "cuda"},
                requested="cuda",
            )

        self.assertEqual(provider.key, "mlx")

    def test_geometry_provider_is_the_only_environment_authority(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SCENESMITH_GEOMETRY_PROVIDER": "mlx",
                "SCENESMITH_SAM_PROVIDER": "cuda",
            },
        ):
            provider = resolve_geometry_execution_provider(
                backend="sam3d",
                sam3d_config={"provider": "auto"},
                cuda_detector=lambda: (0,),
            )

        self.assertEqual(provider.key, "mlx")

    @patch(
        "scenesmith.agent_utils.geometry_generation_server.sam_provider.platform.machine",
        return_value="x86_64",
    )
    @patch(
        "scenesmith.agent_utils.geometry_generation_server.sam_provider.platform.system",
        return_value="Linux",
    )
    def test_auto_geometry_provider_requires_real_capability(
        self, _system, _machine
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderUnavailableError, "no supported"):
                resolve_geometry_execution_provider(
                    backend="sam3d",
                    sam3d_config={"provider": "auto"},
                    cuda_detector=lambda: (),
                )

    def test_worker_pool_accepts_injected_provider(self) -> None:
        pool = GeometryWorkerPool(execution_provider=_InjectedGeometryProvider())

        self.assertEqual(pool.num_workers, 1)
        self.assertEqual(pool.execution_provider, "injected")
        self.assertEqual(pool.worker_targets[0].worker_id, "injected-0")

    def test_gpu_worker_pool_name_is_a_compatibility_alias(self) -> None:
        self.assertIs(GPUWorkerPool, GeometryWorkerPool)


if __name__ == "__main__":
    unittest.main()
