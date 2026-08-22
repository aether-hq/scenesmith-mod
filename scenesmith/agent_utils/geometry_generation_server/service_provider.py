"""Injectable lifecycle providers for geometry generation services.

This seam is separate from the model execution provider:

* ``local`` owns a SceneSmith server and selects CUDA or MLX workers.
* ``external`` connects to an already-managed server and never starts or stops it.

The external provider lets inexpensive workstations use a shared accelerator host
without embedding cloud or vendor assumptions in the scene-generation pipeline.
"""

from __future__ import annotations

import logging
import os

from pathlib import Path
from typing import Callable, Mapping, Protocol

from .client import GeometryGenerationClient
from .execution_provider import GeometryExecutionProvider
from .server.server_manager import GeometryGenerationServer

console_logger = logging.getLogger(__name__)


class GeometryService(Protocol):
    """Lifecycle surface used by the experiment orchestrator."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def wait_until_ready(self, timeout_s: float = 30) -> None: ...

    def is_running(self) -> bool: ...


class GeometryServiceProvider(Protocol):
    """Creates a local or external geometry service handle."""

    key: str

    def connect(
        self,
        *,
        host: str,
        port: int,
        backend: str,
        sam3d_config: dict | None,
        log_file: Path | None = None,
        execution_provider: GeometryExecutionProvider | None = None,
    ) -> GeometryService: ...


class LocalGeometryServiceProvider:
    """Own the lifecycle of an in-process SceneSmith geometry server."""

    key = "local"

    def __init__(
        self,
        *,
        server_factory: Callable[..., GeometryService] = GeometryGenerationServer,
    ) -> None:
        self._server_factory = server_factory

    def connect(
        self,
        *,
        host: str,
        port: int,
        backend: str,
        sam3d_config: dict | None,
        log_file: Path | None = None,
        execution_provider: GeometryExecutionProvider | None = None,
    ) -> GeometryService:
        return self._server_factory(
            host=host,
            port=port,
            backend=backend,
            sam3d_config=sam3d_config,
            log_file=log_file,
            execution_provider=execution_provider,
        )


class ExternalGeometryService:
    """Non-owning handle for an externally managed geometry endpoint."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        backend: str,
        client: GeometryGenerationClient,
    ) -> None:
        self._host = host
        self._port = port
        self._backend = backend
        self._client = client

    def start(self) -> None:
        """Validate the endpoint; never create a remote process."""

        try:
            self._client.assert_compatible(backend=self._backend)
        except Exception as exc:
            raise ConnectionError(
                f"External geometry service is unavailable at "
                f"{self._host}:{self._port}"
            ) from exc
        console_logger.info(
            "Connected to external geometry service at %s:%s",
            self._host,
            self._port,
        )

    def stop(self) -> None:
        """Do nothing because SceneSmith does not own the remote service."""

        console_logger.info(
            "Leaving externally managed geometry service running at %s:%s",
            self._host,
            self._port,
        )

    def wait_until_ready(self, timeout_s: float = 30) -> None:
        """Validate readiness using the common client health contract."""

        del timeout_s
        self.start()

    def is_running(self) -> bool:
        return self._client.health_check()


class ExternalGeometryServiceProvider:
    """Connect to a shared or remotely hosted geometry service."""

    key = "external"

    def __init__(
        self,
        *,
        client_factory: Callable[
            ..., GeometryGenerationClient
        ] = GeometryGenerationClient,
        scheme: str = "https",
        auth_token: str | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._scheme = scheme
        self._auth_token = auth_token

    def connect(
        self,
        *,
        host: str,
        port: int,
        backend: str,
        sam3d_config: dict | None,
        log_file: Path | None = None,
        execution_provider: GeometryExecutionProvider | None = None,
    ) -> GeometryService:
        del sam3d_config, log_file, execution_provider
        return ExternalGeometryService(
            host=host,
            port=port,
            backend=backend,
            client=self._client_factory(
                host=host,
                port=port,
                scheme=self._scheme,
                transport="artifact",
                auth_token=self._auth_token,
            ),
        )


def resolve_geometry_service_provider(
    requested: str | None = None,
    *,
    external_scheme: str = "https",
    external_auth_token: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> GeometryServiceProvider:
    """Resolve a managed-local or non-owning external service provider."""

    current_environ = os.environ if environ is None else environ
    effective = (
        current_environ.get("SCENESMITH_GEOMETRY_SERVICE_PROVIDER")
        or requested
        or "local"
    )
    normalized = str(effective).strip().lower()
    aliases = {"remote": "external", "managed": "local"}
    normalized = aliases.get(normalized, normalized)
    if normalized == "local":
        return LocalGeometryServiceProvider()
    if normalized == "external":
        return ExternalGeometryServiceProvider(
            scheme=external_scheme,
            auth_token=external_auth_token,
        )
    raise ValueError(
        f"Unknown geometry service provider '{effective}'. Expected local or external."
    )
