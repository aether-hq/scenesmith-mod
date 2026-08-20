"""Tests for injectable Blender subprocess providers."""

import unittest

from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.blender.process_provider import (
    NvidiaBwrapProcessProvider,
    NvidiaEnvironmentProcessProvider,
    PreparedProcess,
    SharedRenderProcessProvider,
    render_allocations,
    resolve_render_process_provider,
)
from scenesmith.agent_utils.execution_providers import ProviderUnavailableError


class TestBlenderProcessProviders(unittest.TestCase):
    def test_no_device_uses_portable_shared_process(self) -> None:
        provider = resolve_render_process_provider(None, which=lambda _: None)

        self.assertIsInstance(provider, SharedRenderProcessProvider)
        prepared = provider.prepare(["python", "server.py"], {"PRESERVE": "yes"})
        self.assertEqual(prepared.command, ("python", "server.py"))
        self.assertEqual(prepared.environment["PRESERVE"], "yes")

    def test_nvidia_device_uses_bwrap_when_available(self) -> None:
        provider = resolve_render_process_provider(3, which=lambda _: "/usr/bin/bwrap")

        self.assertIsInstance(provider, NvidiaBwrapProcessProvider)
        self.assertEqual(provider.visibility_token, "3")

    def test_auto_without_bwrap_still_enforces_environment_isolation(self) -> None:
        provider = resolve_render_process_provider(2, which=lambda _: None)

        self.assertIsInstance(provider, NvidiaEnvironmentProcessProvider)
        prepared = provider.prepare(["blender"], {"CUDA_VISIBLE_DEVICES": "0,1,2"})
        self.assertEqual(prepared.command, ("blender",))
        self.assertEqual(prepared.environment["CUDA_VISIBLE_DEVICES"], "2")

    def test_explicit_unavailable_isolation_fails(self) -> None:
        with patch.dict(
            "os.environ",
            {"SCENESMITH_RENDER_PROCESS_PROVIDER": "nvidia-bwrap"},
        ):
            with self.assertRaisesRegex(ProviderUnavailableError, "bubblewrap"):
                resolve_render_process_provider(2, which=lambda _: None)

    def test_custom_process_provider_is_injectable(self) -> None:
        from scenesmith.agent_utils.blender.server_manager import BlenderServer

        provider = MagicMock()
        provider.key = "custom"
        provider.prepare.return_value = PreparedProcess(
            command=("wrapped",), environment={}
        )
        server = BlenderServer(process_provider=provider)

        self.assertIs(server._process_provider, provider)

    def test_render_provider_is_scoped_to_child_environment(self) -> None:
        from scenesmith.agent_utils.blender.server_manager import BlenderServer

        with patch.dict("os.environ", {}, clear=True):
            environment = BlenderServer(
                render_provider="metal"
            )._subprocess_environment()

        self.assertEqual(environment["SCENESMITH_RENDER_PROVIDER"], "metal")

    def test_resolved_render_allocation_is_authoritative_in_child(self) -> None:
        from scenesmith.agent_utils.blender.server_manager import BlenderServer

        with patch.dict(
            "os.environ", {"SCENESMITH_RENDER_PROVIDER": "cpu"}, clear=True
        ):
            environment = BlenderServer(
                render_provider="metal"
            )._subprocess_environment()

        self.assertEqual(environment["SCENESMITH_RENDER_PROVIDER"], "metal")

    def test_render_process_policy_is_injectable_without_global_environment(
        self,
    ) -> None:
        with patch.dict(
            "os.environ",
            {"SCENESMITH_RENDER_PROCESS_PROVIDER": "nvidia-bwrap"},
        ):
            provider = resolve_render_process_provider(
                2,
                requested="shared",
                environ={},
                which=lambda _: "/usr/bin/bwrap",
            )

        self.assertIsInstance(provider, SharedRenderProcessProvider)

    @patch(
        "scenesmith.agent_utils.blender.process_provider.detect_cuda_device_ids",
        return_value=(1, 4),
    )
    def test_cuda_render_isolation_uses_detected_devices(self, detector) -> None:
        allocations = render_allocations("cuda", requested_process_provider="shared")

        self.assertEqual(
            [allocation.slot_id for allocation in allocations], ["render-0", "render-1"]
        )
        self.assertEqual(
            [allocation.target_label for allocation in allocations],
            ["cuda/1", "cuda/4"],
        )
        detector.assert_called_once_with()

    @patch("scenesmith.agent_utils.blender.process_provider.detect_cuda_device_ids")
    def test_metal_render_does_not_probe_cuda(self, detector) -> None:
        allocations = render_allocations("metal")

        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].target_label, "metal/shared")
        detector.assert_not_called()

    def test_blender_server_uses_one_atomic_prepared_process_contract(self) -> None:
        from scenesmith.agent_utils.blender.server_manager import BlenderServer

        provider = MagicMock()
        provider.key = "custom"
        provider.prepare.return_value = PreparedProcess(
            command=("isolated", "server"),
            environment={"ONLY": "child"},
        )
        server = BlenderServer(process_provider=provider)

        with (
            patch.object(server, "_determine_port", return_value=8123),
            patch(
                "scenesmith.agent_utils.blender.server_manager.subprocess.Popen"
            ) as popen,
            patch("scenesmith.agent_utils.blender.server_manager.time.sleep"),
        ):
            popen.return_value.pid = 42
            server.start()

        popen.assert_called_once_with(
            ["isolated", "server"], text=True, env={"ONLY": "child"}
        )


if __name__ == "__main__":
    unittest.main()
