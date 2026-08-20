import json
import logging
import os
import tempfile
import time

from pathlib import Path
from typing import Iterator

import requests

from .dataclasses import (
    GeometryGenerationError,
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
    StreamedResult,
)

console_logger = logging.getLogger(__name__)


class GeometryGenerationClient:
    """Client for making requests to the geometry generation server.

    Provides a high-level interface for generating 3D geometry from images
    using the geometry generation server. Handles HTTP communication, retries,
    error handling, and response parsing.

    The client maintains a persistent HTTP session for connection pooling
    and includes automatic retry logic with exponential backoff for
    transient failures.

    Example:
        >>> client = GeometryGenerationClient()
        >>> requests = [GeometryGenerationServerRequest(
        ...     image_path="/path/to/image.png",
        ...     output_dir="/path/to/output",
        ...     prompt="Modern wooden chair"
        ... )]
        >>> for index, response in client.generate_geometries(requests):
        ...     print(f"Generated geometry: {response.geometry_path}")
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7000,
        *,
        scheme: str = "http",
        transport: str = "local-path",
        auth_token: str | None = None,
        session: requests.Session | None = None,
        max_download_bytes: int = 512 * 1024 * 1024,
    ):
        """Initialize geometry generation client.

        Args:
            host: Server hostname or IP address. Should be accessible from
                the current network context. Defaults to localhost.
            port: Server port number. Must match the port where the geometry
                generation server is listening. Defaults to 7000.
        """
        normalized_scheme = scheme.strip().lower()
        if normalized_scheme not in {"http", "https"}:
            raise ValueError("scheme must be http or https")
        normalized_transport = transport.strip().lower()
        if normalized_transport not in {"local-path", "artifact"}:
            raise ValueError("transport must be local-path or artifact")
        is_loopback = host.strip().lower() in {"127.0.0.1", "localhost", "::1"}
        if normalized_transport == "artifact" and not is_loopback:
            if normalized_scheme != "https":
                raise ValueError("Remote artifact transport requires HTTPS")
            if not auth_token:
                raise ValueError("Remote artifact transport requires an auth token")
        if type(max_download_bytes) is not int or max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be a positive integer")
        self.base_url = f"{normalized_scheme}://{host}:{port}"
        self.session = session or requests.Session()
        self.transport = normalized_transport
        self._auth_token = auth_token
        self._max_download_bytes = max_download_bytes
        console_logger.debug(
            f"Geometry generation client initialized for {self.base_url}"
        )

    def generate_geometries(
        self,
        geometry_requests: list[GeometryGenerationServerRequest],
        max_retries: int = 1,
        timeout_s: float | None = None,
    ) -> Iterator[
        tuple[int, GeometryGenerationServerResponse | GeometryGenerationError]
    ]:
        """Send batch geometry generation requests and yield results as they complete.

        Submits a batch of geometry generation requests to the server and yields
        results as they stream back. This enables pipelining where the client can
        start processing earlier results while the server continues working on
        later requests.

        Individual request failures are yielded as GeometryGenerationError objects
        rather than raising exceptions, allowing the batch to continue processing.

        Args:
            geometry_requests: List of geometry generation requests to process as a batch.
            max_retries: Maximum number of retries for transient failures.
            timeout_s: Timeout in seconds for the entire batch. Should scale with
                batch size and expected server queue depth.

        Yields:
            Tuple of (index, result) where index corresponds to the request's
            position in the input list and result is either:
            - GeometryGenerationServerResponse: Contains the generated geometry path
            - GeometryGenerationError: Contains error details for failed requests

        Raises:
            ConnectionError: If unable to connect to server after max retries.
            RuntimeError: If server returns invalid data (e.g., malformed JSON).
            TimeoutError: If request exceeds timeout limit.
            ValueError: If the requests list is empty.
        """
        if not geometry_requests:
            raise ValueError("Requests list cannot be empty")

        if timeout_s is None:
            from scenesmith.agent_utils.retrieval_policy import (
                geometry_operation_timeout_seconds,
            )

            timeout_s = geometry_operation_timeout_seconds()
        if max_retries < 1:
            raise ValueError("max_retries must allow at least one attempt")

        batch_started = time.monotonic()

        for attempt in range(max_retries):
            try:
                console_logger.debug(
                    f"Sending batch request (attempt {attempt + 1}) with "
                    f"{len(geometry_requests)} requests"
                )

                request_data = self._prepare_request_data(
                    geometry_requests, timeout_s=timeout_s
                )

                # Send streaming request.
                endpoint = (
                    "/v1/generate_geometries"
                    if self.transport == "artifact"
                    else "/generate_geometries"
                )
                request_kwargs = {
                    "json": request_data,
                    "stream": True,
                    "timeout": (min(1.0, timeout_s), timeout_s),
                }
                if self.transport == "artifact":
                    request_kwargs["headers"] = self._headers()
                http_response = self.session.post(
                    f"{self.base_url}{endpoint}", **request_kwargs
                )
                http_response.raise_for_status()

                # Parse streaming NDJSON response.
                for line in http_response.iter_lines():
                    if line:
                        try:
                            result_data = json.loads(line.decode("utf-8"))
                            streamed_result = StreamedResult(**result_data)

                            if streamed_result.status == "error":
                                # Yield error and continue processing remaining results.
                                console_logger.warning(
                                    f"Geometry generation failed for request "
                                    f"{streamed_result.index}: {streamed_result.error}"
                                )
                                yield streamed_result.index, GeometryGenerationError(
                                    index=streamed_result.index,
                                    error_message=streamed_result.error
                                    or "Unknown error",
                                )
                                continue

                            # Convert to response object.
                            if self.transport == "artifact":
                                response = self._download_result(
                                    streamed_result.index,
                                    streamed_result.data or {},
                                    geometry_requests,
                                    timeout_s=timeout_s,
                                )
                            else:
                                response = GeometryGenerationServerResponse(
                                    **streamed_result.data
                                )
                            yield streamed_result.index, response

                        except json.JSONDecodeError as e:
                            raise RuntimeError(
                                f"Invalid JSON in streaming response: {e}"
                            ) from e

                elapsed = time.monotonic() - batch_started
                console_logger.info(
                    "Geometry batch completed in %.3fs (%d request(s))",
                    elapsed,
                    len(geometry_requests),
                )
                return  # Success, exit retry loop

            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    console_logger.warning(
                        f"Connection failed, retrying... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(min(2**attempt, 60))  # Exponential backoff with max 60s
                else:
                    console_logger.error("Asset server connection failed after retries")
                    raise ConnectionError(
                        f"Failed to connect to asset server at {self.base_url}"
                    ) from e

            except requests.exceptions.HTTPError as e:
                if e.response.status_code >= 500:
                    # Server error, might be temporary.
                    if attempt < max_retries - 1:
                        console_logger.warning(
                            f"Server error, retrying... ({attempt + 1}/{max_retries})"
                        )
                        time.sleep(2**attempt)
                        continue

                # Client error or persistent server error.
                try:
                    error_detail = e.response.json()["error"]
                except (KeyError, ValueError):
                    error_detail = str(e)
                console_logger.error(f"HTTP error from asset server: {error_detail}")
                raise RuntimeError(f"Asset server error: {error_detail}") from e

            except requests.exceptions.Timeout as e:
                elapsed = time.monotonic() - batch_started
                console_logger.error(
                    "Geometry batch exceeded %.3fs after %.3fs; failing batch",
                    timeout_s,
                    elapsed,
                )
                raise TimeoutError(
                    f"Geometry batch exceeded {timeout_s:g}s"
                ) from e

    def health_check(self) -> bool:
        """Check if the geometry generation server is healthy and responsive.

        Returns:
            True if server responds successfully to health check within
            5 seconds, False if server is unreachable, returns an error,
            or times out.
        """
        try:
            kwargs: dict = {"timeout": 5}
            if self.transport == "artifact":
                kwargs["headers"] = self._headers()
            response = self.session.get(f"{self.base_url}/health", **kwargs)
            response.raise_for_status()
            return True
        except Exception as e:
            console_logger.warning(f"Health check failed: {e}")
            return False

    def capabilities(self) -> dict:
        """Fetch the versioned server capability contract."""

        response = self.session.get(
            f"{self.base_url}/v1/capabilities",
            headers=self._headers(),
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Geometry capability response must be an object")
        return payload

    def assert_compatible(self, *, backend: str) -> dict:
        """Fail fast unless the remote API, transport, backend, and readiness match."""

        try:
            capabilities = self.capabilities()
        except Exception as exc:
            raise ConnectionError(
                f"Could not negotiate geometry capabilities at {self.base_url}: {exc}"
            ) from exc
        if str(capabilities.get("api_version")) != "1":
            raise ConnectionError(
                f"Unsupported geometry API version: {capabilities.get('api_version')}"
            )
        if capabilities.get("ready") is not True:
            raise ConnectionError("Remote geometry service is not ready")
        if str(capabilities.get("backend", "")).lower() != backend.lower():
            raise ConnectionError(
                f"Remote geometry backend '{capabilities.get('backend')}' does not "
                f"match requested backend '{backend}'"
            )
        transports = capabilities.get("transports", [])
        if self.transport not in transports:
            raise ConnectionError(
                f"Remote geometry service does not support '{self.transport}' transport"
            )
        return capabilities

    def _headers(self) -> dict[str, str]:
        if not self._auth_token:
            return {}
        return {"Authorization": f"Bearer {self._auth_token}"}

    def _prepare_request_data(
        self,
        geometry_requests: list[GeometryGenerationServerRequest],
        *,
        timeout_s: float,
    ) -> list[dict]:
        if self.transport == "local-path":
            return [request.to_dict() for request in geometry_requests]
        payloads: list[dict] = []
        for request in geometry_requests:
            image_path = Path(request.image_path).expanduser().resolve(strict=True)
            with image_path.open("rb") as image_stream:
                upload_response = self.session.post(
                    f"{self.base_url}/v1/artifacts",
                    files={
                        "file": (
                            image_path.name,
                            image_stream,
                            "application/octet-stream",
                        )
                    },
                    headers=self._headers(),
                    timeout=(min(1.0, timeout_s), timeout_s),
                )
            upload_response.raise_for_status()
            artifact_id = upload_response.json()["artifact_id"]
            payload = request.to_dict()
            payload.pop("image_path", None)
            payload.pop("output_dir", None)
            payload.pop("debug_folder", None)
            payload["input_artifact"] = artifact_id
            payloads.append(payload)
        return payloads

    def _download_result(
        self,
        index: int,
        result_data: dict,
        requests_batch: list[GeometryGenerationServerRequest],
        *,
        timeout_s: float,
    ) -> GeometryGenerationServerResponse:
        try:
            artifact_id = str(result_data["artifact_id"])
        except KeyError as exc:
            raise RuntimeError("Artifact response is missing artifact_id") from exc
        request = requests_batch[index]
        filename = request.output_filename or str(
            result_data.get("filename", f"geometry-{artifact_id[:12]}.glb")
        )
        filename = Path(filename).name
        output_dir = Path(request.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / filename
        response = self.session.get(
            f"{self.base_url}/v1/artifacts/{artifact_id}",
            headers=self._headers(),
            stream=True,
            timeout=(min(1.0, timeout_s), timeout_s),
        )
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Artifact response has an invalid Content-Length"
                ) from exc
            if declared_size > self._max_download_bytes:
                raise ValueError(
                    f"Artifact exceeds {self._max_download_bytes}-byte download budget"
                )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_dir, prefix=f".{filename}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                downloaded_bytes = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > self._max_download_bytes:
                            raise ValueError(
                                "Artifact exceeds "
                                f"{self._max_download_bytes}-byte download budget"
                            )
                        temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return GeometryGenerationServerResponse(geometry_path=str(destination))
