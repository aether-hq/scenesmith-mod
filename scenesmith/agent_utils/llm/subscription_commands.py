"""Serialized subscription CLI process execution with bounded deadlines."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import threading
import time

from typing import Any

console_logger = logging.getLogger(__name__)

DEFAULT_SUBSCRIPTION_HARD_TIMEOUT_SECONDS = 300.0
SUBSCRIPTION_HEARTBEAT_SECONDS = 10.0


class SubscriptionCommandCancelled(RuntimeError):
    """Raised after an explicit caller cancellation stops the CLI process group."""


class SubscriptionCommandTimeout(subprocess.TimeoutExpired):
    """A typed response-start or rolling-inactivity subscription timeout."""

    def __init__(
        self,
        cmd: list[str],
        timeout: float,
        *,
        reason: str,
        output: str,
        stderr: str,
    ) -> None:
        super().__init__(cmd, timeout, output=output, stderr=stderr)
        self.reason = reason


class SubscriptionQueueBusy(RuntimeError):
    """Raised when a second request cannot promptly enter the serialized CLI."""


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Stop a CLI and any helper processes it launched."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait(timeout=2.0)


def _run_subscription_command(
    lock: threading.Lock,
    command: list[str],
    *,
    timeout_seconds: float,
    response_start_timeout_seconds: float | None = None,
    cancel_event: threading.Event | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run one serialized CLI turn with bounded queue and execution deadlines."""
    queue_timeout_seconds = float(
        os.environ.get("SCENESMITH_LLM_QUEUE_TIMEOUT_SECONDS", "1")
    )
    if queue_timeout_seconds <= 0:
        raise ValueError("SCENESMITH_LLM_QUEUE_TIMEOUT_SECONDS must be positive")
    if not lock.acquire(timeout=queue_timeout_seconds):
        raise SubscriptionQueueBusy(
            f"subscription_queue_busy: queue admission exceeded "
            f"{queue_timeout_seconds:g} seconds"
        )

    try:
        # The event is shared by the serialized executor. Clear cancellation from
        # the prior request only after this request owns the worker lock.
        if cancel_event is not None:
            cancel_event.clear()
        if timeout_seconds <= 0:
            raise ValueError("subscription inactivity timeout must be positive")
        response_start_timeout_seconds = float(
            response_start_timeout_seconds
            if response_start_timeout_seconds is not None
            else os.environ.get(
                "SCENESMITH_LLM_RESPONSE_START_TIMEOUT_SECONDS",
                "15",
            )
        )
        if response_start_timeout_seconds <= 0:
            raise ValueError("subscription response-start timeout must be positive")
        hard_timeout_seconds = float(
            os.environ.get(
                "SCENESMITH_LLM_HARD_TIMEOUT_SECONDS",
                str(DEFAULT_SUBSCRIPTION_HARD_TIMEOUT_SECONDS),
            )
        )
        if hard_timeout_seconds <= 0:
            raise ValueError("subscription hard timeout must be positive")
        if hard_timeout_seconds < response_start_timeout_seconds:
            raise ValueError(
                "subscription hard timeout must cover the response-start timeout"
            )

        input_value = kwargs.pop("input", None)
        text_mode = bool(kwargs.pop("text", False))
        capture_output = bool(kwargs.pop("capture_output", False))
        check = bool(kwargs.pop("check", False))
        if not text_mode or not capture_output:
            raise ValueError(
                "Subscription commands require text=True and capture_output=True"
            )

        with (
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdin_file,
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file,
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
        ):
            if input_value is not None:
                stdin_file.write(str(input_value))
            stdin_file.flush()
            stdin_file.seek(0)

            started = time.monotonic()
            last_activity = started
            last_heartbeat = started
            last_size = (0, 0)
            response_started = False
            process = subprocess.Popen(
                command,
                stdin=stdin_file,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
                **kwargs,
            )

            timed_out: tuple[str, float] | None = None
            cancelled = False
            while process.poll() is None:
                now = time.monotonic()
                current_size = (
                    os.fstat(stdout_file.fileno()).st_size,
                    os.fstat(stderr_file.fileno()).st_size,
                )
                if current_size != last_size:
                    last_size = current_size
                    last_activity = now
                    response_started = True
                if now - last_heartbeat >= SUBSCRIPTION_HEARTBEAT_SECONDS:
                    console_logger.info(
                        "Subscription CLI active for %.1fs; last output %.1fs ago",
                        now - started,
                        now - last_activity,
                    )
                    last_heartbeat = now
                if (
                    not response_started
                    and now - started >= response_start_timeout_seconds
                ):
                    timed_out = (
                        "response start",
                        response_start_timeout_seconds,
                    )
                    break
                if response_started and now - last_activity >= timeout_seconds:
                    timed_out = ("stream inactivity", timeout_seconds)
                    break
                if now - started >= hard_timeout_seconds:
                    timed_out = ("absolute deadline", hard_timeout_seconds)
                    break
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                time.sleep(0.1)

            if cancelled:
                _terminate_process_group(process)
                console_logger.warning(
                    "Subscription CLI request cancelled; process group stopped"
                )
                raise SubscriptionCommandCancelled(
                    "Subscription CLI request was cancelled"
                )

            if timed_out is not None:
                reason, limit = timed_out
                _terminate_process_group(process)
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read()
                stderr = stderr_file.read()
                console_logger.error(
                    "Subscription CLI %s reached after %.1fs; process group stopped",
                    reason,
                    limit,
                )
                raise SubscriptionCommandTimeout(
                    command,
                    limit,
                    reason=reason,
                    output=stdout,
                    stderr=stderr,
                )

            stdout_file.seek(0)
            stderr_file.seek(0)
            completed = subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout_file.read(),
                stderr_file.read(),
            )
            if check and completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    command,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )
            return completed
    finally:
        lock.release()
