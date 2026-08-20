import json
import subprocess

import pytest

from scenesmith.agent_utils.claude_cli_proxy import (
    _ClaudeExecutor,
    _extract_json_object,
    _extract_stream_result,
    _response_json_schema,
)


def _stream_result(value, *, is_error=False, subtype="success"):
    return json.dumps(
        {
            "type": "result",
            "subtype": subtype,
            "is_error": is_error,
            "result": value,
        }
    )


def test_extract_json_preserves_boolean_and_number_types():
    normalized = _extract_json_object(
        'Result:\n```json\n{"ok": true, "score": 0.75}\n```'
    )

    assert json.loads(normalized) == {"ok": True, "score": 0.75}


def test_claude_subscription_uses_start_and_rolling_inactivity_deadlines(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CLAUDE_BINARY", "true")
    monkeypatch.delenv("SCENESMITH_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv(
        "SCENESMITH_LLM_RESPONSE_START_TIMEOUT_SECONDS", raising=False
    )
    monkeypatch.delenv("SCENESMITH_LLM_CLI_INACTIVITY_TIMEOUT_SECONDS", raising=False)

    executor = _ClaudeExecutor()
    assert executor.response_start_timeout == 15.0
    assert executor.timeout == 45.0


def test_extract_stream_result_uses_terminal_success_event():
    stream = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps({"type": "stream_event", "event": {"type": "message_start"}}),
            _stream_result('{"ok":true}'),
        ]
    )

    assert _extract_stream_result(stream) == '{"ok":true}'


def test_claude_executor_uses_subscription_and_returns_tool_call(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CLAUDE_BINARY", "true")
    monkeypatch.setenv("SCENESMITH_LLM_MODEL", "sonnet")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key-that-must-not-be-used")

    def fake_run(_lock, command, **kwargs):
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["DISABLE_AUTOUPDATER"] == "1"
        assert "--safe-mode" in command
        assert "--json-schema" in command
        result = json.dumps(
            {
                "kind": "tool_calls",
                "content": "",
                "tool_calls": [
                    {
                        "name": "place_chair",
                        "arguments": {"x": 9, "y": 10},
                    }
                ],
            }
        )
        return subprocess.CompletedProcess(
            command,
            0,
            _stream_result(result),
            "",
        )

    monkeypatch.setattr(
        "scenesmith.agent_utils.claude_cli_proxy._run_subscription_command", fake_run
    )
    result = _ClaudeExecutor().complete(
        {
            "model": "haiku",
            "messages": [{"role": "user", "content": "Place a chair."}],
            "tools": [
                {"type": "function", "function": {"name": "place_chair"}}
            ],
        }
    )

    function = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert result["model"] == "haiku"
    assert function == {"name": "place_chair", "arguments": '{"x":9,"y":10}'}


def test_response_json_schema_extracts_agents_sdk_final_output_schema():
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }

    assert _response_json_schema(
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "final_output", "schema": schema},
            }
        }
    ) == schema


def test_claude_executor_surfaces_unrepairable_output_for_common_failover(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CLAUDE_BINARY", "true")
    monkeypatch.setenv("SCENESMITH_LLM_MODEL", "sonnet")
    calls = []

    def fake_run(_lock, command, **kwargs):
        calls.append((command, kwargs["input"]))
        if len(calls) == 1:
            assert "<FINAL_OUTPUT_JSON_SCHEMA>" in kwargs["input"]
            output = json.dumps({
                "kind": "final",
                "content": "**Evaluation Checklist**\nThe design scores nine.",
                "tool_calls": [],
            })
        return subprocess.CompletedProcess(command, 0, _stream_result(output), "")

    monkeypatch.setattr(
        "scenesmith.agent_utils.claude_cli_proxy._run_subscription_command", fake_run
    )
    schema = {
        "type": "object",
        "properties": {
            "critique": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["critique", "score"],
        "additionalProperties": False,
    }
    with pytest.raises(RuntimeError, match="common harness may retry or fail over"):
        _ClaudeExecutor().complete(
            {
                "messages": [{"role": "user", "content": "Critique the room."}],
                "tools": [
                    {"type": "function", "function": {"name": "observe_scene"}}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "final_output", "schema": schema},
                },
            }
        )

    assert len(calls) == 1
