"""Shared latency and retry policy for local semantic asset catalogs."""

import os
import time

from collections.abc import Iterator
from queue import Empty, Queue
from typing import Any, Callable

DEFAULT_LOCAL_RETRIEVAL_TIMEOUT_SECONDS = 2.0
DEFAULT_GEOMETRY_OPERATION_TIMEOUT_SECONDS = 30.0


def local_retrieval_timeout_seconds() -> float:
    """Return the hard wall-clock budget for one local catalog batch."""
    raw = os.environ.get(
        "SCENESMITH_ASSET_RETRIEVAL_TIMEOUT_SECONDS",
        str(DEFAULT_LOCAL_RETRIEVAL_TIMEOUT_SECONDS),
    )
    timeout = float(raw)
    if timeout <= 0:
        raise ValueError("SCENESMITH_ASSET_RETRIEVAL_TIMEOUT_SECONDS must be positive")
    return timeout


def geometry_operation_timeout_seconds() -> float:
    """Return the hard deadline for one generated-geometry batch."""
    raw = os.environ.get(
        "SCENESMITH_GEOMETRY_TIMEOUT_SECONDS",
        str(DEFAULT_GEOMETRY_OPERATION_TIMEOUT_SECONDS),
    )
    timeout = float(raw)
    if timeout <= 0:
        raise ValueError("SCENESMITH_GEOMETRY_TIMEOUT_SECONDS must be positive")
    return timeout


def stream_local_results(
    *,
    result_queue: Queue,
    batch_size: int,
    result_type: type,
    catalog_name: str,
    logger: Any,
    timeout_seconds: float | None = None,
    success_transform: Callable[[int, dict], dict] | None = None,
) -> Iterator[str]:
    """Stream a local batch, converting a missing callback into explicit errors."""
    timeout = timeout_seconds or local_retrieval_timeout_seconds()
    deadline = time.monotonic() + timeout
    received: set[int] = set()

    while len(received) < batch_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            index, (status, result_data) = result_queue.get(timeout=remaining)
        except Empty:
            break

        received.add(index)
        if status == "success":
            try:
                if success_transform is not None:
                    result_data = success_transform(index, result_data)
                result = result_type(index=index, status="success", data=result_data)
            except Exception as exc:
                result = result_type(index=index, status="error", error=str(exc))
        else:
            result = result_type(index=index, status="error", error=result_data)
        yield result.to_json() + "\n"

    missing = [index for index in range(batch_size) if index not in received]
    if not missing:
        return

    message = f"{catalog_name} local retrieval exceeded {timeout:g}s"
    logger.error(
        "%s; failing %d unfinished request(s): %s",
        message,
        len(missing),
        missing,
    )
    for index in missing:
        yield result_type(index=index, status="error", error=message).to_json() + "\n"
