"""Unit tests for GPU worker pool."""

import os
import threading
import time
import unittest

from pathlib import Path
from queue import Queue as ThreadQueue

from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
)
from scenesmith.agent_utils.geometry_generation_server.execution_provider import (
    CudaGeometryExecutionProvider,
    GeometryWorkerTarget,
)
from scenesmith.agent_utils.geometry_generation_server.gpu_worker import (
    ShutdownRequest,
    WorkerReady,
    WorkerStartupFailed,
    WorkRequest,
    WorkResult,
)
from scenesmith.agent_utils.geometry_generation_server.worker_pool import (
    GeometryWorkerPool,
    GPUWorkerPool,
    PoolStats,
)


class _PortableProvider:
    key = "portable-test"

    def targets(self):
        return (
            GeometryWorkerTarget(
                worker_id="worker-0",
                provider=self.key,
                device_id=None,
                label="portable/test",
                environment=(),
            ),
        )

    def process_start_method(self) -> str:
        return "spawn"


def _ready_worker(**kwargs) -> None:
    worker_id = kwargs["execution_target"].worker_id
    result_queue = kwargs["result_queue"]
    work_queue = kwargs["work_queue"]
    result_queue.put(WorkerReady(worker_id=worker_id))
    while not isinstance(work_queue.get(), ShutdownRequest):
        pass


def _failed_worker(**kwargs) -> None:
    target = kwargs["execution_target"]
    kwargs["result_queue"].put(
        WorkerStartupFailed(
            worker_id=target.worker_id,
            error="fixture preload failure",
        )
    )


def _crashing_worker(**kwargs) -> None:
    target = kwargs["execution_target"]
    result_queue = kwargs["result_queue"]
    work_queue = kwargs["work_queue"]
    result_queue.put(WorkerReady(worker_id=target.worker_id))
    work_queue.get()
    os._exit(17)


class TestPoolStats(unittest.TestCase):
    """Test PoolStats dataclass."""

    def test_pool_stats_creation(self):
        """Test PoolStats creation with all fields."""
        stats = PoolStats(
            num_workers=4,
            total_requests=100,
            completed_requests=95,
            failed_requests=5,
            avg_processing_time_s=25.5,
            avg_end_to_end_latency_s=30.0,
            avg_queue_wait_s=4.5,
            max_queue_wait_s=10.0,
            worker_details=[
                {"gpu_id": 0, "pid": 1234, "alive": True},
                {"gpu_id": 1, "pid": 1235, "alive": True},
            ],
        )

        self.assertEqual(stats.num_workers, 4)
        self.assertEqual(stats.total_requests, 100)
        self.assertEqual(stats.completed_requests, 95)
        self.assertEqual(stats.failed_requests, 5)
        self.assertEqual(stats.avg_processing_time_s, 25.5)
        self.assertEqual(stats.avg_end_to_end_latency_s, 30.0)
        self.assertEqual(stats.avg_queue_wait_s, 4.5)
        self.assertEqual(stats.max_queue_wait_s, 10.0)
        self.assertEqual(len(stats.worker_details), 2)

    def test_pool_stats_none_avg_time(self):
        """Test PoolStats with no average processing time."""
        stats = PoolStats(
            num_workers=1,
            total_requests=0,
            completed_requests=0,
            failed_requests=0,
            avg_processing_time_s=None,
            avg_end_to_end_latency_s=None,
            avg_queue_wait_s=None,
            max_queue_wait_s=None,
            worker_details=[],
        )

        self.assertIsNone(stats.avg_processing_time_s)
        self.assertIsNone(stats.avg_end_to_end_latency_s)
        self.assertIsNone(stats.avg_queue_wait_s)
        self.assertIsNone(stats.max_queue_wait_s)


class TestWorkRequestResult(unittest.TestCase):
    """Test WorkRequest and WorkResult dataclasses."""

    def test_work_request_creation(self):
        """Test WorkRequest creation."""
        request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="A wooden chair",
        )

        work_request = WorkRequest(
            request_id="test-123",
            request=request,
            received_timestamp=1234567890.0,
        )

        self.assertEqual(work_request.request_id, "test-123")
        self.assertEqual(work_request.request.prompt, "A wooden chair")
        self.assertEqual(work_request.received_timestamp, 1234567890.0)

    def test_work_result_success(self):
        """Test WorkResult for successful request."""
        result = WorkResult(
            request_id="test-123",
            worker_id=0,
            status="success",
            data={"geometry_path": "/test/output/chair.glb"},
            error=None,
        )

        self.assertEqual(result.request_id, "test-123")
        self.assertEqual(result.worker_id, 0)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["geometry_path"], "/test/output/chair.glb")
        self.assertIsNone(result.error)

    def test_work_result_error(self):
        """Test WorkResult for failed request."""
        result = WorkResult(
            request_id="test-456",
            worker_id=1,
            status="error",
            data=None,
            error="Generation failed: out of memory",
        )

        self.assertEqual(result.status, "error")
        self.assertIsNone(result.data)
        self.assertEqual(result.error, "Generation failed: out of memory")


class TestShutdownRequest(unittest.TestCase):
    """Test ShutdownRequest sentinel class."""

    def test_shutdown_request_is_distinct_type(self):
        """Test that ShutdownRequest is distinguishable from other types."""
        shutdown = ShutdownRequest()
        work_request = WorkRequest(
            request_id="test",
            request=GeometryGenerationServerRequest(
                image_path="/test/image.png",
                output_dir="/test/output",
                prompt="test",
            ),
            received_timestamp=1234567890.0,
        )

        self.assertIsInstance(shutdown, ShutdownRequest)
        self.assertNotIsInstance(work_request, ShutdownRequest)
        self.assertNotIsInstance("string", ShutdownRequest)


class TestWorkerReady(unittest.TestCase):
    """Test WorkerReady signal class."""

    def test_worker_ready_creation(self):
        """Test WorkerReady creation with worker ID."""
        ready = WorkerReady(worker_id=3)

        self.assertEqual(ready.worker_id, 3)

    def test_worker_ready_is_distinct_type(self):
        """Test that WorkerReady is distinguishable from other message types."""
        ready = WorkerReady(worker_id=0)
        shutdown = ShutdownRequest()
        work_result = WorkResult(
            request_id="test",
            worker_id=0,
            status="success",
            data={"geometry_path": "/test/output.glb"},
            error=None,
        )

        self.assertIsInstance(ready, WorkerReady)
        self.assertNotIsInstance(shutdown, WorkerReady)
        self.assertNotIsInstance(work_result, WorkerReady)


class TestWorkerLifecycle(unittest.TestCase):
    def test_start_waits_until_every_worker_is_ready(self) -> None:
        pool = GeometryWorkerPool(
            execution_provider=_PortableProvider(),
            preload_pipeline=False,
            worker_entrypoint=_ready_worker,
            startup_timeout_s=5.0,
        )

        pool.start()
        try:
            self.assertTrue(pool.is_ready())
            self.assertEqual(pool.startup_diagnostics()["status"], "ready")
        finally:
            pool.stop()

    def test_startup_failure_is_reported_and_pool_is_stopped(self) -> None:
        pool = GeometryWorkerPool(
            execution_provider=_PortableProvider(),
            preload_pipeline=False,
            worker_entrypoint=_failed_worker,
            startup_timeout_s=5.0,
        )

        with self.assertRaisesRegex(RuntimeError, "fixture preload failure"):
            pool.start()

        self.assertFalse(pool.is_running())
        self.assertEqual(pool.startup_diagnostics()["status"], "failed")

    def test_request_backend_cannot_override_resolved_runtime(self) -> None:
        pool = GeometryWorkerPool(
            backend="sam3d",
            sam3d_config={"provider": "mlx"},
            execution_provider=_PortableProvider(),
            preload_pipeline=False,
        )
        request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="test",
            backend="hunyuan3d",
        )

        with self.assertRaisesRegex(ValueError, "server runtime"):
            pool.validate_request(request)

    def test_request_provider_cannot_override_resolved_runtime(self) -> None:
        pool = GeometryWorkerPool(
            backend="sam3d",
            sam3d_config={"provider": "mlx"},
            execution_provider=_PortableProvider(),
            preload_pipeline=False,
        )
        request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="test",
            backend="sam3d",
            sam3d_config={"provider": "cuda"},
        )

        with self.assertRaisesRegex(ValueError, "provider"):
            pool.validate_request(request)

    def test_equivalent_relative_and_absolute_runtime_paths_are_accepted(self) -> None:
        relative_checkpoint = "external/Sam3D-Objects-MLX/checkpoints/hf"
        pool = GeometryWorkerPool(
            backend="sam3d",
            sam3d_config={
                "provider": "auto",
                "mlx_checkpoint_dir": str(Path(relative_checkpoint).resolve()),
                "mlx_steps": 12,
            },
            execution_provider=_PortableProvider(),
            preload_pipeline=False,
        )
        request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="test",
            backend="sam3d",
            sam3d_config={
                "provider": "auto",
                "mlx_checkpoint_dir": relative_checkpoint,
                "mlx_steps": 12,
                "mode": "foreground",
            },
        )

        pool.validate_request(request)

    def test_worker_crash_completes_inflight_request_with_error(self) -> None:
        pool = GeometryWorkerPool(
            execution_provider=_PortableProvider(),
            preload_pipeline=False,
            worker_entrypoint=_crashing_worker,
            startup_timeout_s=5.0,
            restart_limit=0,
            health_check_interval_s=0.05,
        )
        completed = threading.Event()
        result_holder = []

        def callback(index, result):
            result_holder.append((index, result))
            completed.set()

        pool.start()
        try:
            pool.submit_request(
                GeometryGenerationServerRequest(
                    image_path="/test/image.png",
                    output_dir="/test/output",
                    prompt="test",
                ),
                callback,
                request_index=3,
                received_timestamp=time.time(),
            )
            self.assertTrue(completed.wait(timeout=3.0))
            self.assertEqual(result_holder[0][0], 3)
            self.assertEqual(result_holder[0][1][0], "error")
            self.assertIn("exited", result_holder[0][1][1])
        finally:
            pool.stop()


class TestWorkerPoolInitialization(unittest.TestCase):
    """Test GPUWorkerPool initialization (without starting)."""

    def test_pool_initialization_defaults(self):
        """Test pool initialization with default parameters."""
        pool = GPUWorkerPool(
            execution_provider=CudaGeometryExecutionProvider(
                detector=lambda: (0, 1, 2, 3)
            )
        )

        self.assertEqual(pool.num_workers, 4)
        self.assertEqual(pool._use_mini, False)
        self.assertEqual(pool._backend, "hunyuan3d")
        self.assertIsNone(pool._sam3d_config)
        self.assertTrue(pool._preload_pipeline)
        # Verify multiprocessing context is created.
        self.assertIsNotNone(pool._mp_ctx)
        # Worker availability is an in-process thread handoff. Keeping it out
        # of multiprocessing.Queue avoids a macOS spawn/feeder deadlock.
        self.assertIsInstance(pool._available_workers, ThreadQueue)

    def test_pool_initialization_custom_params(self):
        """Test pool initialization with custom parameters."""
        sam3d_config = {
            "provider": "cuda",
            "sam3_checkpoint": "/path/to/sam3.pt",
        }

        pool = GPUWorkerPool(
            use_mini=True,
            backend="sam3d",
            sam3d_config=sam3d_config,
            preload_pipeline=False,
            execution_provider=CudaGeometryExecutionProvider(detector=lambda: (0, 1)),
        )

        self.assertEqual(pool.num_workers, 2)
        self.assertTrue(pool._use_mini)
        self.assertEqual(pool._backend, "sam3d")
        self.assertEqual(pool._sam3d_config, sam3d_config)
        self.assertFalse(pool._preload_pipeline)

    def test_pool_stats_before_start(self):
        """Test getting stats before pool is started."""
        pool = GPUWorkerPool(
            execution_provider=CudaGeometryExecutionProvider(detector=lambda: (0,))
        )

        stats = pool.get_stats()

        self.assertEqual(stats.num_workers, 1)
        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.completed_requests, 0)
        self.assertEqual(stats.failed_requests, 0)
        self.assertIsNone(stats.avg_processing_time_s)
        self.assertEqual(stats.worker_details, [])


if __name__ == "__main__":
    unittest.main()
