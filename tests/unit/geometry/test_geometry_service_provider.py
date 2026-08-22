"""Tests for local and external geometry service providers."""

import unittest

from unittest.mock import MagicMock, patch

from scenesmith.agent_utils.geometry_generation_server.execution_provider import (
    MlxGeometryExecutionProvider,
)
from scenesmith.agent_utils.geometry_generation_server.service_provider import (
    ExternalGeometryService,
    ExternalGeometryServiceProvider,
    LocalGeometryServiceProvider,
    resolve_geometry_service_provider,
)


class TestGeometryServiceProviders(unittest.TestCase):
    def test_local_provider_constructs_managed_server(self) -> None:
        server_factory = MagicMock(return_value=MagicMock())
        provider = LocalGeometryServiceProvider(server_factory=server_factory)

        result = provider.connect(
            host="127.0.0.1",
            port=7005,
            backend="sam3d",
            sam3d_config={"provider": "mlx"},
            execution_provider=MlxGeometryExecutionProvider(),
        )

        self.assertIs(result, server_factory.return_value)
        server_factory.assert_called_once_with(
            host="127.0.0.1",
            port=7005,
            backend="sam3d",
            sam3d_config={"provider": "mlx"},
            log_file=None,
            execution_provider=unittest.mock.ANY,
        )

    def test_external_provider_never_starts_or_stops_remote_process(self) -> None:
        client = MagicMock(unsafe=True)
        client.assert_compatible.return_value = {
            "api_version": "1",
            "ready": True,
            "backend": "hunyuan3d",
            "transports": ["artifact"],
        }
        client.health_check.return_value = True
        provider = ExternalGeometryServiceProvider(
            client_factory=MagicMock(return_value=client),
            scheme="https",
            auth_token="secret",
        )

        service = provider.connect(
            host="geometry.example",
            port=7443,
            backend="hunyuan3d",
            sam3d_config=None,
            execution_provider=None,
        )
        service.start()
        service.stop()

        self.assertTrue(service.is_running())
        self.assertIsInstance(service, ExternalGeometryService)
        client.assert_compatible.assert_called_once_with(backend="hunyuan3d")

    def test_external_provider_fails_fast_when_endpoint_is_unhealthy(self) -> None:
        client = MagicMock(unsafe=True)
        client.assert_compatible.side_effect = ConnectionError("unhealthy")
        service = ExternalGeometryServiceProvider(
            client_factory=MagicMock(return_value=client),
            scheme="https",
            auth_token="secret",
        ).connect(
            host="geometry.example",
            port=7443,
            backend="sam3d",
            sam3d_config=None,
            execution_provider=None,
        )

        with self.assertRaisesRegex(ConnectionError, "geometry.example:7443"):
            service.start()

    def test_external_provider_constructs_authenticated_artifact_client(self) -> None:
        client_factory = MagicMock(return_value=MagicMock())
        provider = ExternalGeometryServiceProvider(
            client_factory=client_factory,
            scheme="https",
            auth_token="secret",
        )

        provider.connect(
            host="geometry.example",
            port=7443,
            backend="sam3d",
            sam3d_config=None,
        )

        client_factory.assert_called_once_with(
            host="geometry.example",
            port=7443,
            scheme="https",
            transport="artifact",
            auth_token="secret",
        )

    def test_environment_override_selects_external_provider(self) -> None:
        with patch.dict(
            "os.environ",
            {"SCENESMITH_GEOMETRY_SERVICE_PROVIDER": "external"},
        ):
            provider = resolve_geometry_service_provider("local")

        self.assertEqual(provider.key, "external")

    def test_unknown_service_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "local or external"):
            resolve_geometry_service_provider("magic")


if __name__ == "__main__":
    unittest.main()
