"""Loopback OpenAI-compatible facade backed by a Claude Code subscription."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.cli_llm_proxy import (
    DEFAULT_SUBSCRIPTION_TURN_TIMEOUT_SECONDS,
    OpenAIChatProxy,
    _chat_completion_response,
    _render_messages,
    _requested_model,
    _run_subscription_command,
    _tool_output_schema,
    _tool_prompt,
)
from scenesmith.agent_utils.llm_harness import (
    LLMStructuredOutputError,
    extract_json_object,
    parse_json_value,
)


console_logger = logging.getLogger(__name__)


CLAUDE_BINARY_RECOVERY_SECONDS = 5.0


def _extract_terminal_result_event(stream_output: str) -> dict[str, Any]:
    """Return Claude's terminal result event from newline-delimited output."""
    result_event: dict[str, Any] | None = None
    for line in stream_output.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result_event = event
    if result_event is None:
        raise RuntimeError("Claude CLI stream ended without a result event")
    if result_event.get("is_error") or result_event.get("subtype") != "success":
        diagnostic = (
            result_event.get("result")
            or result_event.get("errors")
            or "unknown Claude CLI error"
        )
        raise RuntimeError(f"Claude CLI stream failed: {diagnostic}")
    return result_event


def _extract_stream_result(stream_output: str) -> str:
    """Return Claude's terminal text from newline-delimited stream events."""
    result_event = _extract_terminal_result_event(stream_output)
    result = result_event.get("result")
    if not isinstance(result, str) or not result.strip():
        structured = result_event.get("structured_output")
        if structured is not None:
            return json.dumps(structured, separators=(",", ":"))
        raise RuntimeError("Claude CLI returned an empty streamed result")
    return result.strip()


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def _extract_stream_usage(
    stream_output: str,
    *,
    requested_model: str,
) -> dict[str, Any]:
    """Normalize Claude's terminal usage into a durable per-request record."""
    result_event = _extract_terminal_result_event(stream_output)
    raw_usage = result_event.get("usage")
    model_usage = result_event.get("modelUsage") or result_event.get("model_usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    model_records = (
        [record for record in model_usage.values() if isinstance(record, dict)]
        if isinstance(model_usage, dict)
        else []
    )
    input_tokens = _non_negative_int(usage.get("input_tokens")) or sum(
        _non_negative_int(record.get("inputTokens", record.get("input_tokens")))
        for record in model_records
    )
    cache_creation_tokens = _non_negative_int(
        usage.get("cache_creation_input_tokens")
    ) or sum(
        _non_negative_int(
            record.get(
                "cacheCreationInputTokens",
                record.get("cache_creation_input_tokens"),
            )
        )
        for record in model_records
    )
    cache_read_tokens = _non_negative_int(usage.get("cache_read_input_tokens")) or sum(
        _non_negative_int(
            record.get("cacheReadInputTokens", record.get("cache_read_input_tokens"))
        )
        for record in model_records
    )
    output_tokens = _non_negative_int(usage.get("output_tokens")) or sum(
        _non_negative_int(record.get("outputTokens", record.get("output_tokens")))
        for record in model_records
    )
    total_tokens = (
        input_tokens + cache_creation_tokens + cache_read_tokens + output_tokens
    )
    raw_cost = result_event.get("total_cost_usd")
    if raw_cost is None:
        raw_cost = sum(
            _non_negative_float(record.get("costUSD", record.get("cost_usd")))
            for record in model_records
        )
    cost = _non_negative_float(raw_cost)
    reported = isinstance(raw_usage, dict) or bool(model_records)
    return {
        "schema_version": 1,
        "provider": "claude-cli",
        "model": requested_model,
        "requests": 1,
        "turns": max(1, _non_negative_int(result_event.get("num_turns"))),
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "api_equivalent_cost_usd": cost,
        "reported": reported,
    }


def _openai_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Map Claude token accounting onto Chat Completions usage fields."""
    prompt_tokens = (
        usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage["output_tokens"],
        "total_tokens": prompt_tokens + usage["output_tokens"],
        "prompt_tokens_details": {
            "cached_tokens": usage["cache_read_input_tokens"],
        },
    }


def _extract_json_object(raw_output: str) -> str:
    """Compatibility wrapper around the provider-neutral JSON repair boundary."""

    return extract_json_object(raw_output)


def _response_json_schema(body: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the final-output JSON schema from an OpenAI-compatible request."""
    response_format = body.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        return schema if isinstance(schema, dict) else None
    if response_format.get("type") == "json_object":
        return {"type": "object"}
    return None


def _normalize_structured_content(content: Any, schema: dict[str, Any]) -> str:
    """Normalize and validate a final response against its requested schema."""
    value = parse_json_value(content) if isinstance(content, str) else content
    validate_json_schema(instance=value, schema=schema)
    return json.dumps(value, separators=(",", ":"))


class _ClaudeExecutor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self.requested_binary = os.environ.get("SCENESMITH_CLAUDE_BINARY", "claude")
        self.binary = self._resolve_binary()
        self.model = os.environ.get("SCENESMITH_LLM_MODEL", "sonnet")
        turn_timeout = float(
            os.environ.get(
                "SCENESMITH_LLM_TIMEOUT_SECONDS",
                str(DEFAULT_SUBSCRIPTION_TURN_TIMEOUT_SECONDS),
            )
        )
        self.response_start_timeout = float(
            os.environ.get(
                "SCENESMITH_LLM_RESPONSE_START_TIMEOUT_SECONDS",
                str(min(15.0, turn_timeout)),
            )
        )
        self.timeout = float(
            os.environ.get(
                "SCENESMITH_LLM_CLI_INACTIVITY_TIMEOUT_SECONDS",
                "45",
            )
        )

    def _resolve_binary(self, *, wait_seconds: float = 0.0) -> str:
        """Resolve a runnable launcher, allowing a bounded updater replacement."""
        deadline = time.monotonic() + wait_seconds
        while True:
            resolved_binary = shutil.which(self.requested_binary)
            if resolved_binary and os.access(resolved_binary, os.X_OK):
                return str(Path(resolved_binary).resolve())
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Claude CLI is not executable: {self.requested_binary}"
                )
            time.sleep(0.1)

    def cancel_active(self) -> None:
        """Cancel the currently running serialized CLI request, if any."""
        self._cancel_event.set()

    def is_active(self) -> bool:
        """Return whether a serialized subscription turn owns the worker."""
        return self._lock.locked()

    def wait_until_idle(self, timeout_seconds: float = 4.0) -> bool:
        """Wait until cancellation has released the serialized CLI lock."""
        acquired = self._lock.acquire(timeout=timeout_seconds)
        if acquired:
            self._lock.release()
        return acquired

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages", [])
        tools = body.get("tools", [])
        tool_choice = body.get("tool_choice", "auto")
        model = _requested_model(body, self.model)
        response_schema = _response_json_schema(body)
        started = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="scenesmith-claude-") as temp_name:
            work_dir = Path(temp_name)
            image_paths: list[Path] = []
            conversation = _render_messages(messages, image_paths, work_dir)
            prompt = conversation + _tool_prompt(tools, tool_choice)
            if tools and response_schema:
                prompt += (
                    "\n\n<FINAL_OUTPUT_JSON_SCHEMA>\n"
                    + json.dumps(response_schema, indent=2)
                    + "\n</FINAL_OUTPUT_JSON_SCHEMA>\nWhen kind=final, content must be a "
                    "string containing only valid JSON matching this schema. Do not "
                    "put prose or Markdown outside that JSON string."
                )
            if image_paths:
                prompt += "\n\n<ATTACHED_IMAGES>\n"
                prompt += "\n".join(
                    f"attached image {index} = {image_path}"
                    for index, image_path in enumerate(image_paths, start=1)
                )
                prompt += (
                    "\n</ATTACHED_IMAGES>\nUse the Read tool to inspect each attached "
                    "image before answering."
                )

            command = [
                self.binary,
                "--print",
                "--safe-mode",
                "--no-session-persistence",
                "--permission-mode",
                "dontAsk",
                "--tools",
                "Read" if image_paths else "",
                "--model",
                model,
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--verbose",
                "--system-prompt",
                (
                    "You are the reasoning engine inside SceneSmith, a 3D indoor-"
                    "scene composer. Follow the supplied conversation and preserve "
                    "its system instructions. Do not modify files or execute shell "
                    "commands. When SceneSmith tools are supplied, select them but "
                    "do not execute them yourself."
                ),
            ]
            if tools:
                command.extend(
                    ["--json-schema", json.dumps(_tool_output_schema(tools))]
                )
            elif body.get("response_format", {}).get("type") == "json_object":
                prompt += (
                    "\n\nReturn only the requested valid JSON object, with no "
                    "markdown. Preserve the requested JSON value types exactly: "
                    "booleans and numbers must not be quoted."
                )

            child_env = os.environ.copy()
            # Claude Code otherwise prefers an inherited Anthropic API key over
            # the cached claude.ai/Max login. Remove both API keys deliberately.
            child_env.pop("ANTHROPIC_API_KEY", None)
            child_env.pop("OPENAI_API_KEY", None)
            # Anthropic's CLI checks for updates on startup and periodically. A
            # global npm update can briefly remove the executable between two
            # SceneSmith turns, so builds pin the resolved binary and disable
            # self-update for the lifetime of this engine process.
            child_env["DISABLE_AUTOUPDATER"] = "1"

            def run_command() -> subprocess.CompletedProcess[str]:
                return _run_subscription_command(
                    self._lock,
                    command,
                    timeout_seconds=self.timeout,
                    response_start_timeout_seconds=self.response_start_timeout,
                    cancel_event=self._cancel_event,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=work_dir,
                    env=child_env,
                    check=False,
                )

            try:
                process = run_command()
            except PermissionError as exc:
                if exc.errno != errno.EACCES:
                    raise
                replacement_binary = self._resolve_binary(
                    wait_seconds=CLAUDE_BINARY_RECOVERY_SECONDS
                )
                console_logger.warning(
                    "Claude CLI executable changed during the build; retrying once "
                    "with %s",
                    replacement_binary,
                )
                self.binary = replacement_binary
                command[0] = replacement_binary
                process = run_command()
            if process.returncode != 0:
                diagnostic = (process.stderr or process.stdout)[-4000:]
                raise RuntimeError(
                    f"Claude CLI exited {process.returncode}: {diagnostic.strip()}"
                )
            raw_output = _extract_stream_result(process.stdout)
            stream_usage = _extract_stream_usage(
                process.stdout,
                requested_model=model,
            )
            stream_usage["request_id"] = f"claude_{uuid.uuid4().hex}"
            if tools and response_schema:
                structured = json.loads(raw_output)
                if structured.get("kind") == "final":
                    try:
                        structured["content"] = _normalize_structured_content(
                            structured.get("content", ""), response_schema
                        )
                    except (
                        json.JSONDecodeError,
                        JsonSchemaValidationError,
                        LLMStructuredOutputError,
                        TypeError,
                    ) as exc:
                        raise RuntimeError(
                            "Claude returned invalid structured output after "
                            "deterministic repair; the common harness may retry or "
                            "fail over to its next configured model"
                        ) from exc
                    raw_output = json.dumps(structured, separators=(",", ":"))
            elif not tools and response_schema:
                raw_output = _extract_json_object(raw_output)
                raw_output = _normalize_structured_content(raw_output, response_schema)

        elapsed = time.monotonic() - started
        console_logger.info(
            "Claude subscription turn completed in %.1fs (%d tools, %d images)",
            elapsed,
            len(tools),
            len(image_paths),
        )
        console_logger.info(
            "SCENESMITH_LLM_USAGE %s",
            json.dumps(stream_usage, sort_keys=True, separators=(",", ":")),
        )
        response = _chat_completion_response(
            raw_output=raw_output,
            tools=tools,
            model=model,
            tool_choice=tool_choice,
        )
        response["usage"] = _openai_usage(stream_usage)
        return response


class ClaudeCliProxy(OpenAIChatProxy):
    """Expose the authenticated Claude Code CLI as Chat Completions."""

    def __init__(self) -> None:
        super().__init__(_ClaudeExecutor(), "claude-cli")


_proxy: ClaudeCliProxy | None = None


def start_claude_cli_proxy() -> str:
    """Start the process-wide Claude subscription proxy once."""
    global _proxy
    if _proxy is None:
        _proxy = ClaudeCliProxy()
        return _proxy.start()
    return _proxy.base_url
