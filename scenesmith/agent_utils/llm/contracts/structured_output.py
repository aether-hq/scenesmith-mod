"""Deterministic parsing and repair of JSON-like model output."""

from __future__ import annotations

import ast
import json
import re

from typing import Any

from scenesmith.agent_utils.llm.contracts.errors import LLMStructuredOutputError


def _json_fence_body(raw_output: str) -> str:
    """Remove a Markdown fence while retaining any surrounding explanation."""

    candidate = str(raw_output or "").strip()
    fenced = re.search(
        r"```(?:json|javascript|js|python)?\s*([\s\S]*?)```", candidate, re.I
    )
    return fenced.group(1).strip() if fenced else candidate


def _json_container_candidate(raw_output: str) -> str:
    """Extract the first object/array and close truncated delimiters safely."""

    candidate = _json_fence_body(raw_output)
    starts = [
        index for index in (candidate.find("{"), candidate.find("[")) if index >= 0
    ]
    if not starts:
        raise LLMStructuredOutputError(
            f"Model did not return JSON: {candidate[:300]!r}"
        )
    start = min(starts)
    stack: list[str] = []
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                continue
            stack.pop()
            if not stack:
                return candidate[start : index + 1]
    return candidate[start:] + "".join(reversed(stack))


def _strip_json_comments(candidate: str) -> str:
    """Remove JavaScript comments without touching comment markers in strings."""

    output: list[str] = []
    index = 0
    in_string = False
    quote = ""
    escaped = False
    while index < len(candidate):
        char = candidate[index]
        following = candidate[index + 1] if index + 1 < len(candidate) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            index += 1
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            index += 2
            while index < len(candidate) and candidate[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and following == "*":
            end = candidate.find("*/", index + 2)
            index = len(candidate) if end < 0 else end + 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _pythonize_json_literals(candidate: str) -> str:
    """Translate JSON literals outside strings for ``ast.literal_eval``."""

    output: list[str] = []
    token: list[str] = []
    in_string = False
    quote = ""
    escaped = False

    def flush() -> None:
        if not token:
            return
        raw = "".join(token)
        output.append({"true": "True", "false": "False", "null": "None"}.get(raw, raw))
        token.clear()

    for char in candidate:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            flush()
            in_string = True
            quote = char
            output.append(char)
        elif char.isalpha():
            token.append(char)
        else:
            flush()
            output.append(char)
    flush()
    return "".join(output)


def parse_json_value(raw_output: Any) -> Any:
    """Parse common model JSON variants without another paid model turn.

    Recovery is deliberately syntactic: prose/fences, single quotes, comments,
    unquoted identifier keys, trailing commas, JS literals, and truncated closing
    delimiters are accepted. Semantic defaults remain the owning tool's job.
    """

    if isinstance(raw_output, (dict, list, int, float, bool)) or raw_output is None:
        return raw_output
    candidate = _json_container_candidate(str(raw_output))
    attempts = [candidate]
    repaired = _strip_json_comments(candidate).translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    )
    repaired = re.sub(
        r"([,{]\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*:)", r'\1"\2"\3', repaired
    )
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if repaired not in attempts:
        attempts.append(repaired)

    failures: list[Exception] = []
    for value in attempts:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            failures.append(exc)
        try:
            return ast.literal_eval(_pythonize_json_literals(value))
        except (SyntaxError, ValueError) as exc:
            failures.append(exc)
    last = failures[-1] if failures else None
    raise LLMStructuredOutputError(
        f"Model returned unrecoverable JSON: {candidate[:300]!r}"
    ) from last


def parse_json_object(raw_output: Any) -> dict[str, Any]:
    """Parse a JSON-ish value and require a mapping at the provider boundary."""

    value = parse_json_value(raw_output)
    if not isinstance(value, dict):
        raise LLMStructuredOutputError(
            "Structured model response must be a JSON object"
        )
    return value
