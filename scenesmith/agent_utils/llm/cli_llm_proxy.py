"""Loopback OpenAI-compatible facade backed by the authenticated Codex CLI.

This lets local SceneSmith runs use a ChatGPT/Codex subscription without
pretending that a ChatGPT subscription is an OpenAI API credit balance.  The
facade implements the small Chat Completions subset used by the OpenAI Agents
SDK and delegates each model turn to ``codex exec``.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.llm.contracts.errors import LLMStructuredOutputError
from scenesmith.agent_utils.llm.contracts.response_normalization import (
    normalize_arguments_to_schema,
)
from scenesmith.agent_utils.llm.contracts.structured_output import (
    parse_json_object,
    parse_json_value,
)
from scenesmith.agent_utils.llm.subscription_commands import (
    SubscriptionCommandCancelled,
    SubscriptionQueueBusy,
    _run_subscription_command,
)

console_logger = logging.getLogger(__name__)


DEFAULT_SUBSCRIPTION_TURN_TIMEOUT_SECONDS = 30.0


def _requested_model(body: dict[str, Any], default: str) -> str:
    """Resolve a per-request model without permitting CLI argument injection."""
    model = str(body.get("model") or default).strip()
    if not model or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", model):
        raise ValueError(f"Invalid subscription model identifier: {model!r}")
    return model


def _text_content(content: Any, image_paths: list[Path], work_dir: Path) -> str:
    """Convert OpenAI message content to text and materialize data URL images."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, default=str)

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text"}:
            parts.append(str(item.get("text", "")))
            continue
        if item_type not in {"image_url", "input_image", "image"}:
            parts.append(json.dumps(item, default=str))
            continue

        image_value = item.get("image_url") or item.get("url")
        if isinstance(image_value, dict):
            image_value = image_value.get("url")
        if not isinstance(image_value, str):
            parts.append("[image omitted: unsupported payload]")
            continue
        if not image_value.startswith("data:image/"):
            parts.append(f"[remote image: {image_value}]")
            continue

        try:
            header, encoded = image_value.split(",", 1)
            mime_type = header.split(";", 1)[0].removeprefix("data:")
            suffix = mimetypes.guess_extension(mime_type) or ".png"
            image_path = work_dir / f"image-{len(image_paths) + 1}{suffix}"
            image_path.write_bytes(base64.b64decode(encoded))
            image_paths.append(image_path)
            parts.append(f"[attached image {len(image_paths)}]")
        except (ValueError, OSError) as exc:
            parts.append(f"[image omitted: {exc}]")
    return "\n".join(part for part in parts if part)


def _render_messages(
    messages: list[dict[str, Any]], image_paths: list[Path], work_dir: Path
) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = _text_content(message.get("content", ""), image_paths, work_dir)
        rendered.append(f"<{role}>\n{content}\n</{role}>")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            rendered.append(
                "<ASSISTANT_TOOL_CALLS>\n"
                + json.dumps(tool_calls, default=str)
                + "\n</ASSISTANT_TOOL_CALLS>"
            )
        if message.get("tool_call_id"):
            rendered.append(f"<TOOL_CALL_ID>{message['tool_call_id']}</TOOL_CALL_ID>")
    return "\n\n".join(rendered)


def _tool_prompt(tools: list[dict[str, Any]], tool_choice: Any) -> str:
    if not tools:
        return ""
    rendered_tools = []
    for tool in tools:
        function = tool.get("function", tool)
        rendered_tools.append(
            {
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )
    choice_instruction = ""
    if tool_choice == "required":
        choice_instruction = "You must return at least one tool call."
    elif isinstance(tool_choice, dict):
        forced_name = tool_choice.get("function", {}).get("name")
        if forced_name:
            choice_instruction = f"You must call {forced_name!r} first."
    elif isinstance(tool_choice, str) and tool_choice not in {"auto", "none"}:
        choice_instruction = f"You must call {tool_choice!r} first."

    return (
        "\n\n<AVAILABLE_SCENESMITH_TOOLS>\n"
        + json.dumps(rendered_tools, indent=2, default=str)
        + "\n</AVAILABLE_SCENESMITH_TOOLS>\n"
        + choice_instruction
        + "\nDo not execute these tools yourself. Select the next tool call(s) for "
        "the SceneSmith runtime. Put each tool's arguments directly in the "
        "arguments object. Use kind=final only when the task is complete."
    )


def _tool_output_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    names = list(
        dict.fromkeys(
            tool.get("function", tool).get("name")
            for tool in tools
            if tool.get("function", tool).get("name")
        )
    )
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["tool_calls", "final"]},
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["kind", "content", "tool_calls"],
        "additionalProperties": False,
    }


def _chat_completion_response(
    *,
    raw_output: str,
    tools: list[dict[str, Any]],
    model: str,
    tool_choice: Any = "auto",
) -> dict[str, Any]:
    """Translate a local CLI result into an OpenAI Chat Completion response."""
    message: dict[str, Any] = {"role": "assistant", "content": raw_output}
    finish_reason = "stop"
    if tools:
        structured = parse_json_object(raw_output)
        required_tool_name = None
        tool_call_required = tool_choice == "required"
        if isinstance(tool_choice, dict):
            required_tool_name = tool_choice.get("function", {}).get("name")
            tool_call_required = bool(required_tool_name)
        elif isinstance(tool_choice, str) and tool_choice not in {
            "auto",
            "none",
            "required",
        }:
            required_tool_name = tool_choice
            tool_call_required = True

        allowed_tools = {
            re.sub(
                r"[^a-z0-9]+",
                "",
                str(tool.get("function", tool).get("name", "")).lower(),
            ): tool.get("function", tool)
            for tool in tools
            if tool.get("function", tool).get("name")
        }
        # Accept the common envelopes used by OpenAI, Anthropic, Qwen, and
        # OpenAI-compatible servers. If a required single tool receives its
        # arguments directly, infer that tool rather than wasting a model turn.
        for envelope in ("response", "output", "result", "data"):
            nested = structured.get(envelope)
            if isinstance(nested, dict) and any(
                key in nested
                for key in (
                    "tool_calls",
                    "toolCalls",
                    "calls",
                    "actions",
                    "function_call",
                )
            ):
                structured = nested
                break
        raw_calls = (
            structured.get("tool_calls")
            or structured.get("toolCalls")
            or structured.get("calls")
            or structured.get("actions")
            or structured.get("function_call")
            or structured.get("functionCall")
        )
        if raw_calls is None:
            direct_name = structured.get("name") or structured.get("tool")
            if direct_name:
                raw_calls = [structured]
            else:
                matching_keys = [
                    key
                    for key in structured
                    if re.sub(r"[^a-z0-9]+", "", str(key).lower()) in allowed_tools
                ]
                if matching_keys:
                    raw_calls = [
                        {"name": key, "arguments": structured[key]}
                        for key in matching_keys
                    ]
                elif (
                    tool_call_required
                    and len(allowed_tools) == 1
                    and not any(
                        key in structured
                        for key in (
                            "kind",
                            "content",
                            "tool_calls",
                            "toolCalls",
                            "calls",
                            "actions",
                        )
                    )
                ):
                    raw_calls = [
                        {
                            "name": next(iter(allowed_tools.values()))["name"],
                            "arguments": dict(structured),
                        }
                    ]
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        if not isinstance(raw_calls, list):
            raw_calls = []
        if tool_call_required and not raw_calls and len(allowed_tools) == 1:
            selected_tool = next(iter(allowed_tools.values()))
            parameters = selected_tool.get("parameters", {})
            required_arguments = set(parameters.get("required", ()))
            permissive_arguments = parameters.get("additionalProperties") is True
            deterministic_empty_fallback = selected_tool.get("name") in {
                "submit_floor_plan",
            }
            content = structured.get("content")
            inferred_arguments: dict[str, Any] = {}
            if isinstance(content, dict):
                inferred_arguments = content
            elif isinstance(content, str) and content.strip():
                try:
                    inferred_arguments = parse_json_object(content)
                except (
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                    LLMStructuredOutputError,
                ):
                    inferred_arguments = {}
            if (
                deterministic_empty_fallback
                or permissive_arguments
                or required_arguments.issubset(inferred_arguments)
            ):
                raw_calls = [
                    {
                        "name": selected_tool["name"],
                        "arguments": inferred_arguments,
                    }
                ]
                structured["kind"] = "tool_calls"
        if raw_calls and not structured.get("kind"):
            structured["kind"] = "tool_calls"
        structured["tool_calls"] = raw_calls

        if structured.get("kind") == "tool_calls":
            calls = []
            for call in structured.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    call = {
                        **function,
                        **{
                            key: value
                            for key, value in call.items()
                            if key != "function"
                        },
                    }
                supplied_name = (
                    call.get("name") or call.get("tool") or call.get("function_name")
                )
                normalized_name = re.sub(
                    r"[^a-z0-9]+", "", str(supplied_name or "").lower()
                )
                selected_tool = allowed_tools.get(normalized_name)
                if selected_tool is None and len(allowed_tools) == 1:
                    selected_tool = next(iter(allowed_tools.values()))
                if selected_tool is None:
                    raise ValueError(f"Unknown tool selected: {supplied_name!r}")
                # Current proxy schemas make arguments a real JSON object, so
                # the CLI's outer schema validation also validates its syntax.
                # Accept the legacy string envelope for resumable sessions, but
                # never ask a model to double-encode JSON again.
                arguments_value = call.get("arguments")
                if arguments_value is None:
                    arguments_value = (
                        call.get("args")
                        if "args" in call
                        else call.get("input", call.get("parameters"))
                    )
                if arguments_value is None and "arguments_json" in call:
                    arguments_value = parse_json_value(call["arguments_json"])
                if arguments_value is None:
                    arguments_value = {}
                if isinstance(arguments_value, str):
                    arguments_value = parse_json_value(arguments_value)
                if not isinstance(arguments_value, dict):
                    properties = selected_tool.get("parameters", {}).get(
                        "properties", {}
                    )
                    if len(properties) == 1:
                        arguments_value = {next(iter(properties)): arguments_value}
                    else:
                        raise ValueError("Tool arguments must be a JSON object")
                arguments_value = normalize_arguments_to_schema(
                    arguments_value, selected_tool.get("parameters")
                )
                arguments = json.dumps(arguments_value, separators=(",", ":"))
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": selected_tool["name"],
                            "arguments": arguments,
                        },
                    }
                )
            if calls:
                if required_tool_name and not any(
                    call["function"]["name"] == required_tool_name for call in calls
                ):
                    raise ValueError(
                        f"Required tool {required_tool_name!r} was not selected"
                    )
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": calls,
                }
                finish_reason = "tool_calls"
            else:
                if tool_call_required:
                    raise ValueError(
                        "A tool call was required, but the subscription model "
                        "returned an empty tool_calls list"
                    )
                message = {
                    "role": "assistant",
                    "content": structured.get("content")
                    or "No further tool action is required.",
                }
                finish_reason = "stop"
        else:
            if tool_call_required:
                raise ValueError(
                    "A tool call was required, but the subscription model "
                    "returned a final text response"
                )
            message = {
                "role": "assistant",
                "content": structured.get("content", ""),
            }

    return {
        "id": f"chatcmpl_cli_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class _CodexExecutor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        requested_binary = os.environ.get("SCENESMITH_CODEX_BINARY", "codex")
        resolved_binary = shutil.which(requested_binary)
        if not resolved_binary:
            raise RuntimeError(f"Codex CLI is not installed: {requested_binary}")
        self.binary = str(Path(resolved_binary).resolve())
        self.model = os.environ.get("SCENESMITH_LLM_MODEL", "gpt-5.6-sol")
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
        started = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="scenesmith-codex-") as temp_name:
            work_dir = Path(temp_name)
            image_paths: list[Path] = []
            conversation = _render_messages(messages, image_paths, work_dir)
            prompt = (
                "You are the reasoning engine inside SceneSmith, a 3D indoor-scene "
                "composer. Follow the supplied conversation and preserve its system "
                "instructions. Do not inspect or modify the local filesystem.\n\n"
                + conversation
                + _tool_prompt(tools, tool_choice)
            )

            output_path = work_dir / "result.txt"
            command = [
                self.binary,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(work_dir),
                "--model",
                model,
                "--output-last-message",
                str(output_path),
            ]

            if tools:
                schema_path = work_dir / "tool-response.schema.json"
                schema_path.write_text(json.dumps(_tool_output_schema(tools)))
                command.extend(["--output-schema", str(schema_path)])
            elif body.get("response_format", {}).get("type") == "json_object":
                prompt += "\n\nReturn only the requested valid JSON object."

            for image_path in image_paths:
                command.extend(["--image", str(image_path)])
            command.append("-")

            child_env = os.environ.copy()
            # Force the Codex CLI to use its existing ChatGPT login rather than
            # accidentally consuming the exhausted API key inherited by SceneSmith.
            child_env.pop("OPENAI_API_KEY", None)
            child_env.pop("ANTHROPIC_API_KEY", None)

            process = _run_subscription_command(
                self._lock,
                command,
                timeout_seconds=self.timeout,
                response_start_timeout_seconds=self.response_start_timeout,
                cancel_event=self._cancel_event,
                input=prompt,
                text=True,
                capture_output=True,
                env=child_env,
                check=False,
            )
            if process.returncode != 0 or not output_path.exists():
                diagnostic = (process.stderr or process.stdout)[-4000:]
                raise RuntimeError(
                    f"Codex CLI exited {process.returncode}: {diagnostic.strip()}"
                )

            raw_output = output_path.read_text().strip()

        elapsed = time.monotonic() - started
        console_logger.info(
            "Codex subscription turn completed in %.1fs (%d tools, %d images)",
            elapsed,
            len(tools),
            len(image_paths),
        )
        return _chat_completion_response(
            raw_output=raw_output,
            tools=tools,
            model=model,
            tool_choice=tool_choice,
        )


class OpenAIChatProxy:
    """Own a loopback HTTP server for a local CLI completion executor."""

    def __init__(self, executor: Any, provider: str) -> None:
        self.provider = provider

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                console_logger.debug("Codex proxy: " + format, *args)

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    console_logger.info(
                        "%s proxy client disconnected before response", provider
                    )

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/") == "/v1/status":
                    self._json(
                        200,
                        {
                            "status": "active" if executor.is_active() else "idle",
                            "provider": provider,
                        },
                    )
                    return
                if self.path.rstrip("/") in {"", "/health", "/v1/models"}:
                    self._json(200, {"status": "ok", "provider": provider})
                else:
                    self._json(404, {"error": {"message": "Not found"}})

            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip("/") == "/v1/cancel":
                    executor.cancel_active()
                    idle = executor.wait_until_idle()
                    self._json(
                        200 if idle else 202,
                        {
                            "status": "idle" if idle else "cancelling",
                            "provider": provider,
                        },
                    )
                    return
                if self.path.rstrip("/") != "/v1/chat/completions":
                    self._json(404, {"error": {"message": "Not found"}})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length))
                    self._json(200, executor.complete(body))
                except subprocess.TimeoutExpired as exc:
                    timeout_seconds = float(exc.timeout)
                    reason = getattr(exc, "reason", "stream inactivity")
                    if reason == "response start":
                        message = (
                            f"{provider} CLI did not begin responding within "
                            f"{timeout_seconds:g} seconds; the process was stopped"
                        )
                        code = "subscription_response_start_timeout"
                    elif reason == "absolute deadline":
                        message = (
                            f"{provider} CLI response exceeded the "
                            f"{timeout_seconds:g}-second absolute ceiling; the "
                            "process was stopped"
                        )
                        code = "subscription_response_deadline"
                    else:
                        message = (
                            f"{provider} CLI stopped streaming for "
                            f"{timeout_seconds:g} seconds; the process was stopped"
                        )
                        code = "subscription_stream_stalled"
                    self._json(
                        504,
                        {
                            "error": {
                                "message": message,
                                "code": code,
                            }
                        },
                    )
                except SubscriptionCommandCancelled as exc:
                    self._json(
                        499,
                        {"error": {"message": str(exc), "code": "cancelled"}},
                    )
                except SubscriptionQueueBusy as exc:
                    self._json(
                        503,
                        {
                            "error": {
                                "message": str(exc),
                                "code": "subscription_queue_busy",
                            }
                        },
                    )
                except Exception as exc:
                    console_logger.exception("%s subscription turn failed", provider)
                    self._json(500, {"error": {"message": str(exc)}})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._server.block_on_close = False
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"scenesmith-{provider}-proxy",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1/"

    def start(self) -> str:
        self._thread.start()
        console_logger.info(
            "%s subscription proxy listening at %s", self.provider, self.base_url
        )
        return self.base_url

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class CodexCliProxy(OpenAIChatProxy):
    """Expose the authenticated Codex CLI as Chat Completions."""

    def __init__(self) -> None:
        super().__init__(_CodexExecutor(), "codex-cli")


_proxy: CodexCliProxy | None = None


def start_codex_cli_proxy() -> str:
    """Start the process-wide proxy once and return its OpenAI base URL."""
    global _proxy
    if _proxy is None:
        _proxy = CodexCliProxy()
        return _proxy.start()
    return _proxy.base_url
