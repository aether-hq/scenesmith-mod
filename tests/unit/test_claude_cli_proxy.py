import errno
import json
import subprocess

import pytest

from scenesmith.agent_utils.claude_cli_proxy import (
    _ClaudeExecutor,
    _extract_json_object,
    _extract_stream_result,
    _extract_stream_usage,
    _response_json_schema,
)


def _stream_result(value, *, is_error=False, subtype="success", **metadata):
    return json.dumps(
        {
            "type": "result",
            "subtype": subtype,
            "is_error": is_error,
            "result": value,
            **metadata,
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
    monkeypatch.delenv("SCENESMITH_LLM_RESPONSE_START_TIMEOUT_SECONDS", raising=False)
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


def test_extract_stream_usage_normalizes_tokens_cache_turns_and_cost():
    stream = _stream_result(
        '{"ok":true}',
        num_turns=3,
        total_cost_usd=0.012345,
        usage={
            "input_tokens": 120,
            "cache_creation_input_tokens": 80,
            "cache_read_input_tokens": 900,
            "output_tokens": 45,
        },
    )

    assert _extract_stream_usage(stream, requested_model="haiku") == {
        "schema_version": 1,
        "provider": "claude-cli",
        "model": "haiku",
        "requests": 1,
        "turns": 3,
        "input_tokens": 120,
        "cache_creation_input_tokens": 80,
        "cache_read_input_tokens": 900,
        "output_tokens": 45,
        "total_tokens": 1145,
        "api_equivalent_cost_usd": 0.012345,
        "reported": True,
    }


def test_extract_stream_usage_falls_back_to_per_model_totals():
    stream = _stream_result(
        "done",
        modelUsage={
            "claude-haiku": {
                "inputTokens": 12,
                "outputTokens": 4,
                "cacheReadInputTokens": 30,
                "cacheCreationInputTokens": 5,
                "costUSD": 0.0012,
            }
        },
    )

    usage = _extract_stream_usage(stream, requested_model="haiku")
    assert usage["total_tokens"] == 51
    assert usage["api_equivalent_cost_usd"] == 0.0012
    assert usage["reported"] is True


def test_claude_executor_uses_subscription_and_returns_tool_call(monkeypatch, caplog):
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
            _stream_result(
                result,
                num_turns=2,
                total_cost_usd=0.0042,
                usage={
                    "input_tokens": 20,
                    "cache_creation_input_tokens": 30,
                    "cache_read_input_tokens": 40,
                    "output_tokens": 10,
                },
            ),
            "",
        )

    monkeypatch.setattr(
        "scenesmith.agent_utils.claude_cli_proxy._run_subscription_command", fake_run
    )
    with caplog.at_level("INFO"):
        result = _ClaudeExecutor().complete(
            {
                "model": "haiku",
                "messages": [{"role": "user", "content": "Place a chair."}],
                "tools": [{"type": "function", "function": {"name": "place_chair"}}],
            }
        )

    function = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert result["model"] == "haiku"
    assert function == {"name": "place_chair", "arguments": '{"x":9,"y":10}'}
    assert result["usage"] == {
        "prompt_tokens": 90,
        "completion_tokens": 10,
        "total_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 40},
    }
    usage_line = next(
        record.message
        for record in caplog.records
        if record.message.startswith("SCENESMITH_LLM_USAGE ")
    )
    usage_record = json.loads(usage_line.split(" ", 1)[1])
    assert usage_record["request_id"].startswith("claude_")
    assert usage_record["total_tokens"] == 100
    assert usage_record["api_equivalent_cost_usd"] == 0.0042


def test_claude_executor_recovers_when_pinned_binary_is_replaced_mid_run(
    monkeypatch, tmp_path
):
    first_binary = tmp_path / "claude-v1.exe"
    replacement_binary = tmp_path / "claude-v2.exe"
    launcher = tmp_path / "claude"
    first_binary.write_text("first")
    first_binary.chmod(0o755)
    launcher.symlink_to(first_binary)
    monkeypatch.setenv("SCENESMITH_CLAUDE_BINARY", str(launcher))

    executor = _ClaudeExecutor()
    commands = []

    def fake_run(_lock, command, **_kwargs):
        commands.append(command[0])
        if len(commands) == 1:
            first_binary.unlink()
            launcher.unlink()
            replacement_binary.write_text("replacement")
            replacement_binary.chmod(0o755)
            launcher.symlink_to(replacement_binary)
            raise PermissionError(errno.EACCES, "Permission denied", command[0])
        return subprocess.CompletedProcess(
            command,
            0,
            _stream_result("replacement succeeded"),
            "",
        )

    monkeypatch.setattr(
        "scenesmith.agent_utils.claude_cli_proxy._run_subscription_command", fake_run
    )

    result = executor.complete(
        {"messages": [{"role": "user", "content": "Continue the build."}]}
    )

    assert result["choices"][0]["message"]["content"] == "replacement succeeded"
    assert commands == [str(first_binary.resolve()), str(replacement_binary.resolve())]


def test_response_json_schema_extracts_agents_sdk_final_output_schema():
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }

    assert (
        _response_json_schema(
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "final_output", "schema": schema},
                }
            }
        )
        == schema
    )


def test_claude_executor_surfaces_unrepairable_output_for_common_failover(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CLAUDE_BINARY", "true")
    monkeypatch.setenv("SCENESMITH_LLM_MODEL", "sonnet")
    calls = []

    def fake_run(_lock, command, **kwargs):
        calls.append((command, kwargs["input"]))
        if len(calls) == 1:
            assert "<FINAL_OUTPUT_JSON_SCHEMA>" in kwargs["input"]
            output = json.dumps(
                {
                    "kind": "final",
                    "content": "**Evaluation Checklist**\nThe design scores nine.",
                    "tool_calls": [],
                }
            )
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
                "tools": [{"type": "function", "function": {"name": "observe_scene"}}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "final_output", "schema": schema},
                },
            }
        )

    assert len(calls) == 1
