"""Flask application for provider-backed geometry generation.

This module provides the HTTP interface for geometry generation. It uses a
worker pool to distribute requests across provider-owned execution targets.

CRITICAL: This module must not import accelerator runtimes at module level.
Provider-specific imports are deferred to isolated worker processes.
"""

import logging
import secrets
import shutil
import time
import uuid

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

import flask

from scenesmith.agent_utils.assets.retrieval_policy import (
    geometry_operation_timeout_seconds,
    stream_local_results,
)
from scenesmith.agent_utils.core.scheduler import StrictRoundRobinScheduler
from scenesmith.agent_utils.geometry_generation_server.artifact_store import (
    ArtifactStore,
)
from scenesmith.agent_utils.geometry_generation_server.execution_provider import (
    GeometryExecutionProvider,
)
from scenesmith.agent_utils.geometry_generation_server.worker_pool import (
    GeometryWorkerPool,
)

from ..dataclasses import GeometryGenerationServerRequest, StreamedResult

console_logger = logging.getLogger(__name__)
GEOMETRY_API_VERSION = "1"


class GeometryGenerationApp(flask.Flask):
    """Flask application for provider-backed geometry generation.

    This application manages a pool of isolated worker processes and distributes
    geometry generation requests across them using fair round-robin scheduling.
    """

    def __init__(
        self,
        use_mini: bool = False,
        backend: str = "hunyuan3d",
        sam3d_config: dict | None = None,
        preload_pipeline: bool = True,
        log_file: Path | None = None,
        execution_provider: GeometryExecutionProvider | None = None,
        artifact_store: ArtifactStore | None = None,
        auth_token: str | None = None,
        max_batch_size: int = 64,
    ) -> None:
        """Initialize the Flask app with a provider-backed worker pool.

        Args:
            use_mini: Whether to use the mini model variant (0.6B parameters) instead
                of the full model. The mini model is faster with lower memory usage
                but may have reduced quality. Default: False (use full model).
            backend: 3D generation backend to use ("hunyuan3d" or "sam3d").
                Default: "hunyuan3d".
            sam3d_config: Configuration for SAM3D backend. Required if backend="sam3d".
                Should contain sam3_checkpoint and sam3d_checkpoint paths.
            preload_pipeline: Whether to preload pipelines in workers on start.
                Default: True.
            log_file: Optional path to log file for worker logging (e.g., experiment.log).
            execution_provider: Optional injected local execution provider.
        """
        super().__init__("geometry_generation_server")

        self._use_mini = use_mini
        self._backend = backend
        self._sam3d_config = sam3d_config
        self._preload_pipeline = preload_pipeline
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self._max_batch_size = max_batch_size
        self._artifact_store = artifact_store
        self._auth_token = auth_token

        # Fair scheduling across clients.
        self._scheduler = StrictRoundRobinScheduler()

        # Provider-backed worker pool (created but not started).
        self._worker_pool = GeometryWorkerPool(
            use_mini=use_mini,
            backend=backend,
            sam3d_config=sam3d_config,
            preload_pipeline=preload_pipeline,
            log_file=log_file,
            execution_provider=execution_provider,
        )

        # Coordinator thread dispatches from scheduler to worker pool.
        self._processing_thread: Thread | None = None
        self._processing_active = False

        # Setup routes.
        self.add_url_rule("/health", "health", self._health_endpoint, methods=["GET"])
        self.add_url_rule(
            "/v1/capabilities",
            "capabilities",
            self._capabilities_endpoint,
            methods=["GET"],
        )
        self.add_url_rule(
            "/v1/artifacts",
            "upload_artifact",
            self._upload_artifact_endpoint,
            methods=["POST"],
        )
        self.add_url_rule(
            "/v1/artifacts/<artifact_id>",
            "download_artifact",
            self._download_artifact_endpoint,
            methods=["GET"],
        )
        self.add_url_rule(
            "/shutdown", "shutdown", self._shutdown_endpoint, methods=["POST"]
        )
        self.add_url_rule(
            "/generate_geometries",
            "generate_geometries",
            self._generate_geometries_endpoint,
            methods=["POST"],
        )
        self.add_url_rule(
            "/v1/generate_geometries",
            "generate_artifact_geometries",
            self._generate_artifact_geometries_endpoint,
            methods=["POST"],
        )
        self.before_request(self._authorize_request)

    def start_processing(self) -> None:
        """Start the worker pool and coordinator thread."""
        if self._processing_active:
            console_logger.warning("Processing already active")
            return

        console_logger.info("Starting geometry generation processing...")

        # Start the worker pool first.
        self._worker_pool.start()
        num_workers = self._worker_pool.num_workers
        console_logger.info(
            "Started %s worker(s) with provider '%s'",
            num_workers,
            self._worker_pool.execution_provider,
        )

        # Start coordinator thread.
        self._processing_active = True
        self._processing_thread = Thread(target=self._process_queue, daemon=True)
        self._processing_thread.start()

        console_logger.info("Geometry generation processing started")

    def stop_processing(self) -> None:
        """Stop the coordinator thread and worker pool gracefully."""
        if not self._processing_active:
            return

        console_logger.info("Stopping geometry generation processing...")
        self._processing_active = False

        # Wait for coordinator thread to complete.
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
            if self._processing_thread.is_alive():
                console_logger.warning("Coordinator thread did not stop gracefully")

        # Stop the worker pool.
        self._worker_pool.stop()

        console_logger.info("Geometry generation processing stopped")

    def _process_queue(self) -> None:
        """Dispatch requests from scheduler to worker pool.

        This runs in a coordinator thread, pulling requests from the fair
        scheduler and dispatching them to the provider-backed worker pool. The dispatch
        blocks until a worker is available, preserving fair ordering.
        """
        try:
            console_logger.info("Coordinator thread started")

            while self._processing_active:
                # Get next request from fair scheduler.
                queued_request = self._scheduler.get_next_request()
                if queued_request:
                    console_logger.debug(
                        f"Dispatching request from {queued_request.client_id}: "
                        f"{queued_request.request.prompt}"
                    )

                    # Dispatch to worker pool (blocks until worker available).
                    # A malformed client request must fail that request, not
                    # permanently kill the coordinator for every client.
                    try:
                        self._worker_pool.submit_request(
                            request=queued_request.request,
                            callback=queued_request.callback,
                            request_index=queued_request.request_index,
                            received_timestamp=queued_request.received_timestamp,
                        )
                    except Exception as exc:
                        console_logger.exception("Geometry request dispatch failed")
                        queued_request.callback(
                            queued_request.request_index,
                            ("error", str(exc)),
                        )
                else:
                    # No requests available, sleep briefly.
                    time.sleep(0.1)

        except Exception as e:
            console_logger.error(f"Coordinator thread failed: {e}")

        finally:
            console_logger.info("Coordinator thread stopped")

    def _health_endpoint(self) -> flask.Response:
        """Health check endpoint with pool and scheduler details."""
        scheduler_queue_size = self._scheduler.get_queue_size()
        active_clients = self._scheduler.get_client_count()
        pool_stats = self._worker_pool.get_stats()

        ready = self._processing_active and self._worker_pool.is_ready()
        response = flask.jsonify(
            {
                "status": "ready" if ready else "unavailable",
                "api_version": GEOMETRY_API_VERSION,
                "backend": self._backend,
                "execution_provider": self._worker_pool.execution_provider,
                "num_workers": pool_stats.num_workers,
                "scheduler_queue_size": scheduler_queue_size,
                "active_clients": active_clients,
                "processing_active": self._processing_active,
                "total_requests": pool_stats.total_requests,
                "completed_requests": pool_stats.completed_requests,
                "failed_requests": pool_stats.failed_requests,
                "avg_processing_time_seconds": pool_stats.avg_processing_time_s,
                "avg_end_to_end_latency_seconds": pool_stats.avg_end_to_end_latency_s,
                "avg_queue_wait_seconds": pool_stats.avg_queue_wait_s,
                "max_queue_wait_seconds": pool_stats.max_queue_wait_s,
                "workers": pool_stats.worker_details,
                "startup": self._worker_pool.startup_diagnostics(),
            }
        )
        return response, 200 if ready else 503

    def _capabilities_endpoint(self) -> flask.Response:
        ready = self._processing_active and self._worker_pool.is_ready()
        return flask.jsonify(
            {
                "api_version": GEOMETRY_API_VERSION,
                "ready": ready,
                "backend": self._backend,
                "execution_provider": self._worker_pool.execution_provider,
                "transports": ["local-path"]
                + (["artifact"] if self._artifact_store else []),
                "max_batch_size": self._max_batch_size,
            }
        ), (200 if ready else 503)

    def _authorize_request(self) -> flask.Response | None:
        protected = (
            flask.request.path.startswith("/v1/") or flask.request.path == "/shutdown"
        )
        if not protected or self._auth_token is None:
            return None
        authorization = flask.request.headers.get("Authorization", "")
        expected = f"Bearer {self._auth_token}"
        if not secrets.compare_digest(authorization, expected):
            return flask.jsonify({"error": "Unauthorized"}), 401
        return None

    def _require_artifact_store(self) -> ArtifactStore:
        if self._artifact_store is None:
            raise RuntimeError("Artifact transport is not enabled on this server")
        return self._artifact_store

    def _upload_artifact_endpoint(self) -> flask.Response:
        try:
            upload = flask.request.files.get("file")
            if upload is None:
                return flask.jsonify({"error": "Missing multipart file field"}), 400
            record = self._require_artifact_store().publish_stream(
                upload.stream, filename=upload.filename or "artifact.bin"
            )
            return (
                flask.jsonify(
                    {
                        "artifact_id": record.artifact_id,
                        "filename": record.filename,
                        "size_bytes": record.size_bytes,
                    }
                ),
                201,
            )
        except (ValueError, RuntimeError) as exc:
            return flask.jsonify({"error": str(exc)}), 400

    def _download_artifact_endpoint(self, artifact_id: str) -> flask.Response:
        try:
            path = self._require_artifact_store().resolve(artifact_id)
            return flask.send_file(path, as_attachment=True)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return flask.jsonify({"error": str(exc)}), 404

    def _shutdown_endpoint(self) -> flask.Response:
        """Shutdown endpoint for graceful server termination."""
        console_logger.info("Shutdown endpoint called")

        # Get the shutdown function from werkzeug.
        shutdown_func = flask.request.environ.get("werkzeug.server.shutdown")

        if shutdown_func is None:
            console_logger.warning(
                "Not running with the Werkzeug Server, cannot shutdown"
            )
            return flask.jsonify({"status": "error", "message": "shutdown failed"}), 500

        # Shutdown the server.
        shutdown_func()
        return flask.jsonify({"status": "shutting down"}), 200

    def _generate_geometries_endpoint(self) -> flask.Response:
        """Handle batch geometry generation requests with streaming response."""
        try:
            if flask.request.remote_addr not in {"127.0.0.1", "::1", None}:
                return (
                    flask.jsonify(
                        {
                            "error": "The local-path endpoint is loopback-only; use "
                            "/v1/generate_geometries with artifact transport."
                        }
                    ),
                    403,
                )
            data = flask.request.json
            if not data:
                return flask.jsonify({"error": "No JSON data provided"}), 400

            if not isinstance(data, list):
                return flask.jsonify({"error": "Expected a list of requests"}), 400

            if len(data) == 0:
                return flask.jsonify({"error": "Empty request list"}), 400
            if len(data) > self._max_batch_size:
                return flask.jsonify({"error": "Batch size budget exceeded"}), 400

            # Validate each request in the batch.
            required_fields = ["image_path", "output_dir", "prompt"]
            for i, request_data in enumerate(data):
                if not isinstance(request_data, dict):
                    return (
                        flask.jsonify({"error": f"Request {i} is not an object"}),
                        400,
                    )

                for field in required_fields:
                    if field not in request_data:
                        return (
                            flask.jsonify(
                                {"error": f"Request {i} missing field: {field}"}
                            ),
                            400,
                        )

            # Create batch request.
            batch_requests = [
                GeometryGenerationServerRequest(**req_data) for req_data in data
            ]
            return self._stream_requests(batch_requests)

        except Exception as e:
            console_logger.error(f"Batch request handling failed: {e}")
            return flask.jsonify({"error": str(e)}), 500

    def _generate_artifact_geometries_endpoint(self) -> flask.Response:
        """Generate from uploaded artifacts and return output artifact IDs."""

        try:
            store = self._require_artifact_store()
            data = flask.request.get_json(silent=True)
            if not isinstance(data, list) or not data:
                return (
                    flask.jsonify({"error": "Expected a non-empty request list"}),
                    400,
                )
            if len(data) > self._max_batch_size:
                return flask.jsonify({"error": "Batch size budget exceeded"}), 400
            allowed = {
                "input_artifact",
                "prompt",
                "output_filename",
                "backend",
                "sam3d_config",
                "scene_id",
            }
            batch_requests: list[GeometryGenerationServerRequest] = []
            for index, payload in enumerate(data):
                if not isinstance(payload, dict):
                    raise ValueError(f"Request {index} is not an object")
                unknown = set(payload) - allowed
                if unknown:
                    raise ValueError(
                        f"Request {index} has unknown fields: {sorted(unknown)}"
                    )
                artifact_path = store.resolve(payload.get("input_artifact", ""))
                prompt = payload.get("prompt")
                if type(prompt) is not str or not prompt.strip():
                    raise ValueError(
                        f"Request {index} prompt must be a non-empty string"
                    )
                output_filename = payload.get("output_filename")
                if output_filename is not None:
                    if (
                        type(output_filename) is not str
                        or Path(output_filename).name != output_filename
                    ):
                        raise ValueError(
                            f"Request {index} output_filename must be a safe filename"
                        )
                job_dir = store.root / "jobs" / str(uuid.uuid4())
                job_dir.mkdir(parents=True, exist_ok=False)
                batch_requests.append(
                    GeometryGenerationServerRequest(
                        image_path=str(artifact_path),
                        output_dir=str(job_dir),
                        prompt=prompt,
                        output_filename=output_filename,
                        backend=payload.get("backend", self._backend),
                        sam3d_config=payload.get("sam3d_config"),
                        scene_id=payload.get("scene_id"),
                    )
                )

            def publish_result(_index: int, result: dict) -> dict:
                job_dir = Path(batch_requests[_index].output_dir).resolve(strict=True)
                try:
                    geometry_path = Path(result["geometry_path"]).resolve(strict=True)
                    if not geometry_path.is_relative_to(job_dir):
                        raise ValueError(
                            "Generated artifact is outside its server job directory"
                        )
                    record = store.publish_path(
                        geometry_path, filename=geometry_path.name
                    )
                    return {
                        "artifact_id": record.artifact_id,
                        "filename": record.filename,
                        "size_bytes": record.size_bytes,
                    }
                finally:
                    shutil.rmtree(job_dir, ignore_errors=True)

            return self._stream_requests(batch_requests, publish_result)
        except (TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
            return flask.jsonify({"error": str(exc)}), 400
        except Exception as exc:
            console_logger.exception("Artifact batch request handling failed")
            return flask.jsonify({"error": str(exc)}), 500

    def _stream_requests(
        self,
        batch_requests: list[GeometryGenerationServerRequest],
        success_transform: Callable[[int, dict], dict] | None = None,
    ) -> flask.Response:
        """Enqueue one validated batch and stream terminal results."""

        first_scene_id = batch_requests[0].scene_id
        batch_id = first_scene_id or str(uuid.uuid4())
        client_result_queue: Queue = Queue()
        batch_size = len(batch_requests)

        def result_callback(index: int, result: tuple[str, dict]) -> None:
            client_result_queue.put((index, result))

        self._scheduler.add_batch(
            client_id=batch_id,
            requests=batch_requests,
            callback=result_callback,
            received_timestamp=time.time(),
        )

        return flask.Response(
            stream_local_results(
                result_queue=client_result_queue,
                batch_size=batch_size,
                result_type=StreamedResult,
                catalog_name="Geometry generation",
                logger=console_logger,
                timeout_seconds=geometry_operation_timeout_seconds(),
                success_transform=success_transform,
            ),
            mimetype="application/x-ndjson",
        )
