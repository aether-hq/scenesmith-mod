"""Transient provider failure classification and compatibility retries."""

from __future__ import annotations

import asyncio

from typing import Callable, TypeVar


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, asyncio.TimeoutError))
        or "timeout" in name
        or "timed out" in message
        or "subscription_cli_stalled" in message
        or "produced no progress" in message
        or status == 504
    )


def is_transient_provider_error(exc: Exception) -> bool:
    """Classify retryable provider failures without retrying invalid model output."""
    if _is_timeout_error(exc):
        return False
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if "subscription_queue_busy" in message or "queue admission" in message:
        return False
    if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
        return True
    name = type(exc).__name__.lower()
    return any(
        marker in name or marker in message
        for marker in (
            "connectionerror",
            "connection error",
            "rate limit",
            "temporarily unavailable",
        )
    )


T = TypeVar("T")


def run_with_transient_retry(operation: Callable[[int], T], *, max_attempts: int) -> T:
    """Retry transient transport/provider failures for compatibility callers."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(attempt)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts or not is_transient_provider_error(exc):
                raise
    assert last_error is not None
    raise last_error
