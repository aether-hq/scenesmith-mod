"""Normalized Agents SDK tool arguments and model responses."""

from __future__ import annotations

import json
import re

from typing import Any

from scenesmith.agent_utils.llm.contracts.errors import LLMStructuredOutputError
from scenesmith.agent_utils.llm.contracts.structured_output import parse_json_object


def extract_json_object(raw_output: str) -> str:
    """Deterministically extract and repair one JSON object."""

    return json.dumps(parse_json_object(raw_output), separators=(",", ":"))


def _normalized_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def normalize_arguments_to_schema(value: Any, schema: dict[str, Any] | None) -> Any:
    """Coerce harmless provider differences to the declared tool schema.

    This intentionally fixes representation, not meaning: camel/snake/case key
    differences and primitive string encodings are normalized, but required
    semantic values are never invented.
    """

    if not isinstance(schema, dict):
        return value
    expected = schema.get("type")
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), None)
    if expected == "object":
        if not isinstance(value, dict):
            properties = schema.get("properties") or {}
            if len(properties) == 1:
                value = {next(iter(properties)): value}
            else:
                return value
        properties = schema.get("properties") or {}
        aliases = {_normalized_identifier(key): key for key in properties}
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            canonical = aliases.get(_normalized_identifier(key), str(key))
            if (
                canonical not in properties
                and schema.get("additionalProperties") is False
            ):
                continue
            normalized[canonical] = normalize_arguments_to_schema(
                item, properties.get(canonical)
            )
        return normalized
    if expected == "array":
        items = value if isinstance(value, (list, tuple)) else [value]
        return [
            normalize_arguments_to_schema(item, schema.get("items")) for item in items
        ]
    if expected in {"number", "integer"} and isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            number = float(match.group())
            return int(number) if expected == "integer" else number
    if expected == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    if expected == "string" and value is not None and not isinstance(value, str):
        return str(value)
    enum = schema.get("enum") or []
    if enum and value not in enum:
        matched = next(
            (
                item
                for item in enum
                if _normalized_identifier(item) == _normalized_identifier(value)
            ),
            None,
        )
        if matched is not None:
            return matched
    return value


def _agent_request_contract(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], Any | None]:
    tools = kwargs.get("tools")
    output_schema = kwargs.get("output_schema")
    if tools is None and len(args) >= 4:
        tools = args[3]
    if output_schema is None and len(args) >= 5:
        output_schema = args[4]
    return list(tools or []), output_schema


def normalize_agent_model_response(
    response: Any,
    *,
    tools: list[Any],
    output_schema: Any | None,
) -> Any:
    """Repair native Agents SDK tool arguments and structured final output."""

    tool_by_name = {
        _normalized_identifier(getattr(tool, "name", "")): tool
        for tool in tools
        if getattr(tool, "name", None)
    }
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call":
            supplied_name = getattr(item, "name", "")
            tool = tool_by_name.get(_normalized_identifier(supplied_name))
            if tool is None and len(tool_by_name) == 1:
                tool = next(iter(tool_by_name.values()))
            if tool is None:
                raise LLMStructuredOutputError(
                    f"Model selected unknown tool {supplied_name!r}"
                )
            arguments = parse_json_object(getattr(item, "arguments", "{}"))
            arguments = normalize_arguments_to_schema(
                arguments, getattr(tool, "params_json_schema", None)
            )
            item.name = tool.name
            item.arguments = json.dumps(arguments, separators=(",", ":"))

    if (
        output_schema is None
        or getattr(output_schema, "is_plain_text", lambda: False)()
    ):
        return response
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) != "output_text":
                continue
            repaired = extract_json_object(getattr(part, "text", ""))
            try:
                output_schema.validate_json(repaired)
            except Exception as exc:
                raise LLMStructuredOutputError(
                    f"Model output did not match {output_schema.name()!r} after "
                    "deterministic JSON repair"
                ) from exc
            part.text = repaired
    return response


def _with_vision_detail(
    messages: list[dict[str, Any]], vision_detail: str
) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": message["role"], "content": content})
            continue
        parts = []
        for item in content:
            if item.get("type") != "image_url":
                parts.append(item)
                continue
            image = item.get("image_url", {})
            url = image.get("url") if isinstance(image, dict) else image
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": vision_detail},
                }
            )
        normalized.append({"role": message["role"], "content": parts})
    return normalized


def _to_responses_messages(
    messages: list[dict[str, Any]], vision_detail: str
) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": content})
            continue
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image = item.get("image_url", {})
                url = image.get("url") if isinstance(image, dict) else image
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": url,
                        "detail": vision_detail,
                    }
                )
        converted.append({"role": message["role"], "content": parts})
    return converted


def _request_json_in_prompt(
    messages: list[dict[str, Any]], *, responses_format: bool = False
) -> list[dict[str, Any]]:
    updated = [
        {"role": message["role"], "content": message.get("content", "")}
        for message in messages
    ]
    if not updated:
        return updated
    part_type = "input_text" if responses_format else "text"
    instruction = "Return only one valid JSON object, with no Markdown fence."
    content = updated[-1]["content"]
    if isinstance(content, str):
        updated[-1]["content"] = f"{content}\n\n{instruction}"
    else:
        updated[-1]["content"] = [*content, {"type": part_type, "text": instruction}]
    return updated


def _has_images(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(message.get("content"), list)
        and any(
            item.get("type") in {"image_url", "input_image"}
            for item in message["content"]
        )
        for message in messages
    )
