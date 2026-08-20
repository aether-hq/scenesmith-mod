"""Provider-backed worker pool for geometry generation.

This module distributes geometry requests across worker targets supplied by an
execution provider (CUDA today, or one isolated MLX/Metal worker).

Key design principles:
1. Workers are spawned before backend runtime initialization in the parent
2. Each provider owns its worker process environment and device isolation
3. On-demand dispatch blocks until a worker is available (preserves fair scheduling)
4. Single code path works for both single-GPU and multi-GPU configurations
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
import uuid

from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty, Queue as ThreadQueue
from threading import Thread
from typing import Any, Callable

from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.geometry_generation_server.execution_provider import (
    GeometryExecutionProvider,
    GeometryWorkerTarget,
    resolve_geometry_execution_provider,
)
from scenesmith.agent_utils.geometry_generation_server.gpu_worker import (
    ShutdownRequest,
    WorkerReady,
    WorkerStartupFailed,
    WorkRequest,
    WorkResult,
    geometry_worker_main,
)

console_logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Information about a worker process."""

    worker_id: str
    """Provider-neutral worker ID."""

    target: GeometryWorkerTarget
    """Provider-owned execution target."""

    process: Process
    """The worker process."""

    work_queue: Queue
    """Queue to send work to this worker."""


@dataclass
class PoolStats:
    """Statistics from the worker pool for health reporting."""

    num_workers: int
    """Number of provider-backed workers in the pool."""

    total_requests: int
    """Total number of requests processed."""

    completed_requests: int
    """Number of successfully completed requests."""

    failed_requests: int
    """Number of failed requests."""

    avg_processing_time_s: float | None
    """Average provider execution time in seconds, or None if no data."""

    avg_end_to_end_latency_s: float | None
    """Average end-to-end latency in seconds, or None if no data."""

    avg_queue_wait_s: float | None
    """Average queue wait time (end-to-end - processing), or None if no data."""

    max_queue_wait_s: float | None
    """Maximum queue wait time observed over server lifecycle, or None if no data."""

    worker_details: list[dict]
    """Per-worker statistics."""


class GeometryWorkerPool:
    """Manages provider-backed worker processes with on-demand dispatch.

    Workers signal availability after completing each request.
    submit_request() blocks until a worker is free, ensuring:
    - Fair scheduler ordering is preserved (requests dispatched in order)
    - Natural load balancing (faster workers process more requests)
    - Works identically with one or many provider targets

    Example:
        >>> pool = GeometryWorkerPool(use_mini=False, backend="hunyuan3d")
        >>> pool.start()
        >>> print(f"Pool has {pool.num_workers} workers")
        >>>
        >>> def callback(index, result):
        ...     print(f"Request {index}: {result}")
        >>>
        >>> pool.submit_request(request, callback, request_index=0)
        >>> pool.stop()
    """

    def __init__(
        self,
        use_mini: bool = False,
        backend: str = "hunyuan3d",
        sam3d_config: dict | None = None,
        preload_pipeline: bool = True,
        log_file: Path | None = None,
        execution_provider: GeometryExecutionProvider | None = None,
        multiprocessing_context: Any | None = None,
        worker_entrypoint: Callable[..., None] = geometry_worker_main,
        startup_timeout_s: float = 300.0,
        restart_limit: int = 3,
        health_check_interval_s: float = 5.0,
    ) -> None:
        """Initialize the provider-backed worker pool.

        Args:
            use_mini: Whether to use mini model variant (Hunyuan3D only).
            backend: Generation backend ("hunyuan3d" or "sam3d").
            sam3d_config: Configuration for SAM3D backend.
            preload_pipeline: Whether to preload pipeline in workers on start.
            log_file: Optional path to log file for worker logging.
            execution_provider: Optional injected worker provider. When omitted,
                the backend and SAM configuration select a built-in provider.
        """
        self._use_mini = use_mini
        self._backend = backend
        self._sam3d_config = sam3d_config
        self._preload_pipeline = preload_pipeline
        self._log_file = str(log_file) if log_file else None
        if startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if restart_limit < 0:
            raise ValueError("restart_limit must be non-negative")
        if health_check_interval_s <= 0:
            raise ValueError("health_check_interval_s must be positive")
        self._startup_timeout_s = startup_timeout_s
        self._restart_limit = restart_limit
        self._health_check_interval_s = health_check_interval_s
        self._worker_entrypoint = worker_entrypoint

        self._execution_provider = execution_provider or (
            resolve_geometry_execution_provider(
                backend=backend,
                sam3d_config=sam3d_config,
            )
        )
        self._worker_targets = self._execution_provider.targets()
        self._targets_by_id = {
            target.worker_id: target for target in self._worker_targets
        }
        if len(self._targets_by_id) != len(self._worker_targets):
            raise ValueError(
                "Geometry execution provider returned duplicate worker IDs"
            )
        self._worker_ids = [target.worker_id for target in self._worker_targets]
        self._num_workers = len(self._worker_targets)
        console_logger.info(
            "Resolved %d geometry worker(s) through provider '%s': %s",
            self._num_workers,
            self._execution_provider.key,
            [target.label for target in self._worker_targets],
        )

        start_method = self._execution_provider.process_start_method()
        self._mp_ctx = multiprocessing_context or mp.get_context(start_method)

        # Lock for serializing pipeline initialization to avoid I/O contention.
        # SAM3D checkpoints are ~15GB total. Loading them on 8 workers simultaneously
        # causes severe disk I/O contention. This lock ensures only one worker loads
        # checkpoints at a time.
        self._init_lock = self._mp_ctx.Lock()

        # Worker tracking.
        self._workers: dict[str, WorkerInfo] = {}
        # Availability is exchanged only between the result collector and the
        # coordinator, which are threads in this process.  A multiprocessing
        # Queue adds a feeder thread and pipe that can lose the initial ready
        # token on macOS after spawn, leaving submit_request() blocked even
        # though startup diagnostics report the worker as ready.
        self._available_workers: ThreadQueue[str] = ThreadQueue()
        self._result_queue: Queue = self._mp_ctx.Queue()
        self._pending_callbacks: dict[str, tuple[Callable, int]] = {}
        self._pending_by_worker: dict[str, str] = {}
        self._pending_callbacks_lock = threading.Lock()

        # Result collector thread.
        self._result_thread: Thread | None = None
        self._health_monitor_thread: Thread | None = None
        self._running = False
        self._startup_status = "stopped"
        self._startup_failures: dict[str, str] = {}
        self._ready_worker_ids: set[str] = set()
        self._startup_condition = threading.Condition()
        self._restart_counts: dict[str, int] = {
            worker_id: 0 for worker_id in self._worker_ids
        }

        # Aggregate statistics.
        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._processing_times: list[float] = []
        self._end_to_end_latencies: list[float] = []
        self._max_queue_wait: float | None = None
        self._stats_lock = threading.Lock()

        # Per-worker statistics for utilization tracking.
        self._per_worker_completed: dict[str, int] = {}
        self._per_worker_failed: dict[str, int] = {}

    @property
    def num_workers(self) -> int:
        """Get the number of workers in the pool."""
        return self._num_workers

    @property
    def execution_provider(self) -> str:
        """Return the selected geometry execution provider key."""

        return self._execution_provider.key

    @property
    def worker_targets(self) -> tuple[GeometryWorkerTarget, ...]:
        """Return immutable provider-owned worker targets."""

        return self._worker_targets

    def is_running(self) -> bool:
        """Return whether the coordinator currently owns active workers."""

        return self._running

    def is_ready(self) -> bool:
        """Return true only after every configured worker reports readiness."""

        with self._startup_condition:
            return self._startup_status == "ready"

    def startup_diagnostics(self) -> dict[str, Any]:
        """Return a stable, JSON-safe worker startup snapshot."""

        with self._startup_condition:
            return {
                "status": self._startup_status,
                "provider": self.execution_provider,
                "ready_workers": sorted(self._ready_worker_ids),
                "expected_workers": list(self._worker_ids),
                "failures": dict(self._startup_failures),
            }

    def validate_request(self, request: GeometryGenerationServerRequest) -> None:
        """Reject request fields that attempt to replace the resolved runtime."""

        request_backend = request.backend.strip().lower()
        if request_backend != self._backend.strip().lower():
            raise ValueError(
                f"Request backend '{request.backend}' does not match the server "
                f"runtime '{self._backend}'."
            )
        if request_backend != "sam3d" or not request.sam3d_config:
            return
        requested_provider = str(request.sam3d_config.get("provider", "auto")).lower()
        aliases = {"apple": "mlx", "metal": "mlx", "mps": "mlx", "nvidia": "cuda"}
        requested_provider = aliases.get(requested_provider, requested_provider)
        if requested_provider not in {"auto", self.execution_provider}:
            raise ValueError(
                f"Request SAM provider '{requested_provider}' does not match the "
                f"resolved server provider '{self.execution_provider}'."
            )
        server_config = self._sam3d_config or {}
        authoring_keys = {"mode", "object_description", "threshold"}
        path_keys = {
            "sam3_checkpoint",
            "sam3d_checkpoint",
            "mlx_repo_path",
            "mlx_python_path",
            "mlx_checkpoint_dir",
        }
        for key, value in request.sam3d_config.items():
            if key in authoring_keys or key == "provider":
                continue
            server_value = server_config.get(key)
            if key in path_keys and server_value is not None:
                requested_path = Path(str(value)).expanduser().resolve()
                server_path = Path(str(server_value)).expanduser().resolve()
                matches_runtime = requested_path == server_path
            else:
                matches_runtime = key in server_config and server_value == value
            if not matches_runtime:
                raise ValueError(
                    f"Request SAM runtime field '{key}' cannot override the server "
                    "runtime configuration."
                )

    def _start_single_worker(self, worker_id: str) -> None:
        """Start a single provider-backed worker process.

        Args:
            worker_id: Provider-neutral worker identifier.
        """
        work_queue = self._mp_ctx.Queue()
        target = self._targets_by_id[worker_id]

        process = self._mp_ctx.Process(
            target=self._worker_entrypoint,
            kwargs={
                "worker_id": worker_id,
                "execution_target": target,
                "work_queue": work_queue,
                "result_queue": self._result_queue,
                "use_mini": self._use_mini,
                "backend": self._backend,
                "sam3d_config": self._sam3d_config,
                "preload_pipeline": self._preload_pipeline,
                "init_lock": self._init_lock,
                "log_file": self._log_file,
            },
        )
        process.start()

        self._workers[worker_id] = WorkerInfo(
            worker_id=worker_id,
            target=target,
            process=process,
            work_queue=work_queue,
        )

        console_logger.info(
            "Started geometry worker %s on %s (PID: %s)",
            worker_id,
            target.label,
            process.pid,
        )

    def _restart_worker(self, worker_id: str) -> None:
        """Restart a dead worker.

        Args:
            worker_id: Provider-neutral worker identifier.
        """
        old_worker = self._workers.get(worker_id)
        if old_worker:
            # Clean up old process if still somehow alive.
            if old_worker.process.is_alive():
                old_worker.process.terminate()
                old_worker.process.join(timeout=5.0)

        self._restart_counts[worker_id] += 1
        self._start_single_worker(worker_id)
        console_logger.info(f"Restarted worker {worker_id}")

    def _health_monitor_loop(self) -> None:
        """Monitor worker health and restart dead workers."""
        while self._running:
            time.sleep(self._health_check_interval_s)

            # Re-check after sleep to avoid restarting during shutdown.
            if not self._running:
                break

            for worker_id, worker in list(self._workers.items()):
                if not self._running:
                    break
                if not worker.process.is_alive():
                    console_logger.warning(
                        f"Worker {worker_id} (PID {worker.process.pid}) died, restarting..."
                    )
                    self._handle_dead_worker(worker_id)

    def _handle_dead_worker(self, worker_id: str) -> None:
        """Fail in-flight work and restart within the configured budget."""

        with self._pending_callbacks_lock:
            request_id = self._pending_by_worker.pop(worker_id, None)
            callback_info = (
                self._pending_callbacks.pop(request_id, None) if request_id else None
            )
        if callback_info is not None:
            callback, request_index = callback_info
            error = (
                f"Geometry worker '{worker_id}' exited while processing the request."
            )
            with self._stats_lock:
                self._failed_requests += 1
                self._per_worker_failed[worker_id] = (
                    self._per_worker_failed.get(worker_id, 0) + 1
                )
            try:
                callback(request_index, ("error", error))
            except Exception:
                console_logger.exception(
                    "Failed to deliver worker-crash result for request %s", request_id
                )
        if self._restart_counts[worker_id] >= self._restart_limit:
            with self._startup_condition:
                self._startup_status = "failed"
                self._startup_failures[worker_id] = (
                    f"restart limit ({self._restart_limit}) exhausted"
                )
                self._ready_worker_ids.discard(worker_id)
                self._startup_condition.notify_all()
            console_logger.error(
                "Worker %s exhausted its restart limit; pool is degraded", worker_id
            )
            return
        with self._startup_condition:
            self._ready_worker_ids.discard(worker_id)
            if self._startup_status == "ready":
                self._startup_status = "starting"
        self._restart_worker(worker_id)

    def start(self) -> None:
        """Start all provider-backed worker processes.

        The execution provider owns the multiprocessing start strategy. Each
        worker applies its provider environment before model-specific imports.

        Workers are staggered to avoid contention during pipeline loading.
        """
        if self._running:
            raise RuntimeError("Worker pool is already running")

        console_logger.info(
            f"Starting geometry worker pool with {self._num_workers} workers..."
        )
        self._running = True
        with self._startup_condition:
            self._startup_status = "starting"
            self._startup_failures.clear()
            self._ready_worker_ids.clear()

        # Collect lifecycle messages before workers can emit them.
        self._result_thread = Thread(target=self._collect_results, daemon=True)
        self._result_thread.start()

        # Start one process for each provider-owned target.
        for i, worker_id in enumerate(self._worker_ids):
            self._start_single_worker(worker_id)

            # Stagger worker starts to avoid contention during pipeline loading.
            if i < self._num_workers - 1 and self._preload_pipeline:
                time.sleep(1.0)

        deadline = time.monotonic() + self._startup_timeout_s
        with self._startup_condition:
            while (
                self._startup_status == "starting"
                and len(self._ready_worker_ids) < self._num_workers
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._startup_status = "failed"
                    self._startup_failures["pool"] = (
                        f"startup timed out after {self._startup_timeout_s:.1f}s"
                    )
                    break
                self._startup_condition.wait(timeout=remaining)
            if self._startup_failures:
                error = "; ".join(
                    f"{worker}: {message}"
                    for worker, message in sorted(self._startup_failures.items())
                )
            else:
                self._startup_status = "ready"
                error = None

        if error is not None:
            self.stop()
            raise RuntimeError(f"Geometry worker pool failed to start: {error}")

        # Start health monitor thread.
        self._health_monitor_thread = Thread(
            target=self._health_monitor_loop, daemon=True, name="WorkerHealthMonitor"
        )
        self._health_monitor_thread.start()

        console_logger.info("Geometry worker pool started successfully")

    def stop(self) -> None:
        """Stop all worker processes gracefully."""
        if not self._running:
            console_logger.warning("Worker pool is not running")
            return

        console_logger.info("Stopping geometry worker pool...")
        self._running = False

        # Send shutdown signal to all workers.
        for worker_id, worker in self._workers.items():
            try:
                worker.work_queue.put(ShutdownRequest())
                console_logger.debug(f"Sent shutdown signal to worker {worker_id}")
            except Exception as e:
                console_logger.warning(
                    f"Failed to send shutdown to worker {worker_id}: {e}"
                )

        # Wait for health monitor to stop (daemon, checks _running flag).
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=2.0)

        # Wait for result collector to finish.
        if self._result_thread and self._result_thread.is_alive():
            self._result_thread.join(timeout=5)
            if self._result_thread.is_alive():
                console_logger.warning(
                    "Result collector thread did not stop gracefully"
                )

        # Wait for workers to finish (with timeout).
        for worker_id, worker in self._workers.items():
            if worker.process.is_alive():
                worker.process.join(timeout=10)
                if worker.process.is_alive():
                    console_logger.warning(
                        f"Worker {worker_id} did not stop gracefully, terminating..."
                    )
                    worker.process.terminate()
                    worker.process.join(timeout=2)

        # Clean up.
        self._workers.clear()
        with self._startup_condition:
            if self._startup_status != "failed":
                self._startup_status = "stopped"
            self._ready_worker_ids.clear()
        console_logger.info("Geometry worker pool stopped")

    def submit_request(
        self,
        request: GeometryGenerationServerRequest,
        callback: Callable[[int, tuple[str, dict | str]], None],
        request_index: int,
        received_timestamp: float,
    ) -> None:
        """Submit a request to an available worker.

        This method blocks until a worker is available, preserving the fair
        ordering from the StrictRoundRobinScheduler.

        Args:
            request: The geometry generation request.
            callback: Function to call with (index, result) when complete.
            request_index: Index of this request in the batch.
            received_timestamp: Time when request was received by server.
        """
        if not self._running:
            raise RuntimeError("Worker pool is not running")

        self.validate_request(request)

        # Block until a worker is available, but wake periodically so shutdown
        # cannot strand the coordinator forever when a worker exits mid-job.
        while self._running:
            try:
                worker_id = self._available_workers.get(timeout=0.5)
                break
            except Empty:
                continue
        else:
            raise RuntimeError("Worker pool stopped while waiting for a worker")

        # Generate unique request ID.
        request_id = str(uuid.uuid4())

        # Store callback for later invocation.
        with self._pending_callbacks_lock:
            self._pending_callbacks[request_id] = (callback, request_index)
            self._pending_by_worker[worker_id] = request_id

        # Track request.
        with self._stats_lock:
            self._total_requests += 1

        # Submit work to the worker.
        worker = self._workers[worker_id]
        worker.work_queue.put(
            WorkRequest(
                request_id=request_id,
                request=request,
                received_timestamp=received_timestamp,
            )
        )

        console_logger.info(
            f"Submitted request {request_id} to worker {worker_id}: {request.prompt}"
        )

    def get_stats(self) -> PoolStats:
        """Get aggregate statistics from the pool.

        Returns:
            Pool statistics for health reporting.
        """
        with self._stats_lock:
            avg_time = None
            if self._processing_times:
                avg_time = sum(self._processing_times) / len(self._processing_times)

            avg_latency = None
            if self._end_to_end_latencies:
                avg_latency = sum(self._end_to_end_latencies) / len(
                    self._end_to_end_latencies
                )

            # Compute queue wait time as difference between latency and processing.
            avg_queue_wait = None
            if avg_latency is not None and avg_time is not None:
                avg_queue_wait = avg_latency - avg_time

            # Build per-worker details with utilization stats.
            total_processed = self._completed_requests + self._failed_requests
            worker_details = []
            for worker_id, worker in self._workers.items():
                completed = self._per_worker_completed.get(worker_id, 0)
                failed = self._per_worker_failed.get(worker_id, 0)
                worker_total = completed + failed
                proportion = (
                    worker_total / total_processed if total_processed > 0 else 0
                )

                worker_details.append(
                    {
                        "worker_id": worker_id,
                        "provider": worker.target.provider,
                        "device_id": worker.target.device_id,
                        "pid": worker.process.pid,
                        "alive": worker.process.is_alive(),
                        "completed_requests": completed,
                        "failed_requests": failed,
                        "total_requests": worker_total,
                        "proportion": round(proportion, 4),
                    }
                )

            return PoolStats(
                num_workers=self._num_workers,
                total_requests=self._total_requests,
                completed_requests=self._completed_requests,
                failed_requests=self._failed_requests,
                avg_processing_time_s=avg_time,
                avg_end_to_end_latency_s=avg_latency,
                avg_queue_wait_s=avg_queue_wait,
                max_queue_wait_s=self._max_queue_wait,
                worker_details=worker_details,
            )

    def _collect_results(self) -> None:
        """Collect results from workers and invoke callbacks.

        This runs in a separate thread, continuously collecting results from
        the shared result queue and routing them back to the appropriate
        callbacks. Also handles WorkerReady signals from workers that have
        finished initialization.
        """
        console_logger.debug("Result collector thread started")

        while self._running or not self._result_queue.empty():
            try:
                message = self._result_queue.get(timeout=0.5)
            except Empty:
                continue

            # Handle worker ready signal (worker finished initialization).
            if isinstance(message, WorkerReady):
                console_logger.info(
                    f"Worker {message.worker_id} initialized and ready for requests"
                )
                self._available_workers.put(message.worker_id)
                with self._startup_condition:
                    self._ready_worker_ids.add(message.worker_id)
                    if len(self._ready_worker_ids) == self._num_workers:
                        self._startup_status = "ready"
                    self._startup_condition.notify_all()
                continue

            if isinstance(message, WorkerStartupFailed):
                with self._startup_condition:
                    self._startup_status = "failed"
                    self._startup_failures[message.worker_id] = message.error
                    self._startup_condition.notify_all()
                continue

            # Handle work result.
            result: WorkResult = message

            # Look up and invoke callback.
            with self._pending_callbacks_lock:
                callback_info = self._pending_callbacks.pop(result.request_id, None)
                self._pending_by_worker.pop(result.worker_id, None)

            if callback_info is None:
                console_logger.warning(
                    f"No callback found for request {result.request_id}"
                )
                # Still return worker to available pool.
                self._available_workers.put(result.worker_id)
                continue

            callback, request_index = callback_info

            # Update statistics (aggregate and per-worker).
            with self._stats_lock:
                if result.status == "success":
                    self._completed_requests += 1
                    self._per_worker_completed[result.worker_id] = (
                        self._per_worker_completed.get(result.worker_id, 0) + 1
                    )
                else:
                    self._failed_requests += 1
                    self._per_worker_failed[result.worker_id] = (
                        self._per_worker_failed.get(result.worker_id, 0) + 1
                    )

                # Aggregate processing time from worker.
                if result.processing_time_seconds is not None:
                    self._processing_times.append(result.processing_time_seconds)
                    # Keep only last 10000 times to bound memory.
                    if len(self._processing_times) > 10000:
                        self._processing_times.pop(0)

                # Aggregate end-to-end latency from worker.
                if result.end_to_end_latency_seconds is not None:
                    self._end_to_end_latencies.append(result.end_to_end_latency_seconds)
                    # Keep only last 10000 latencies to bound memory.
                    if len(self._end_to_end_latencies) > 10000:
                        self._end_to_end_latencies.pop(0)

                # Track max queue wait time when both measurements available.
                if (
                    result.end_to_end_latency_seconds is not None
                    and result.processing_time_seconds is not None
                ):
                    queue_wait = (
                        result.end_to_end_latency_seconds
                        - result.processing_time_seconds
                    )
                    if (
                        self._max_queue_wait is None
                        or queue_wait > self._max_queue_wait
                    ):
                        self._max_queue_wait = queue_wait

            # Invoke callback.
            try:
                if result.status == "success":
                    callback(request_index, ("success", result.data))
                else:
                    callback(request_index, ("error", result.error))
            except Exception as e:
                console_logger.error(
                    f"Callback failed for request {result.request_id}: {e}"
                )

            # Return worker to available pool.
            self._available_workers.put(result.worker_id)

        console_logger.debug("Result collector thread stopped")


# Public compatibility alias. New code should use the provider-neutral name.
GPUWorkerPool = GeometryWorkerPool
