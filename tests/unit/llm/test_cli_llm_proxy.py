import base64
import json
import subprocess
import sys
import threading
import time

from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import pytest

from scenesmith.agent_utils.llm.cli_llm_proxy import (
    OpenAIChatProxy,
    _chat_completion_response,
    _CodexExecutor,
    _text_content,
    _tool_output_schema,
)
from scenesmith.agent_utils.llm.subscription_commands import (
    DEFAULT_SUBSCRIPTION_HARD_TIMEOUT_SECONDS,
    SubscriptionCommandCancelled,
    SubscriptionCommandTimeout,
    SubscriptionQueueBusy,
    _run_subscription_command,
)


def test_subscription_absolute_safety_ceiling_allows_long_active_streams():
    assert DEFAULT_SUBSCRIPTION_HARD_TIMEOUT_SECONDS == 300.0


def test_proxy_reports_active_subscription_turn_status():
    class FakeExecutor:
        def is_active(self):
            return True

    proxy = OpenAIChatProxy(FakeExecutor(), "test-cli")
    base_url = proxy.start()
    try:
        with urlopen(urljoin(base_url, "status"), timeout=1.0) as response:
            payload = json.loads(response.read())
    finally:
        proxy.stop()

    assert payload == {"status": "active", "provider": "test-cli"}


def test_data_url_images_are_materialized(tmp_path: Path):
    payload = base64.b64encode(b"png fixture").decode()
    images: list[Path] = []

    text = _text_content(
        [
            {"type": "text", "text": "Inspect this chair."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            },
        ],
        images,
        tmp_path,
    )

    assert text == "Inspect this chair.\n[attached image 1]"
    assert len(images) == 1
    assert images[0].read_bytes() == b"png fixture"


def test_codex_subscription_uses_start_and_rolling_inactivity_deadlines(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CODEX_BINARY", "true")
    monkeypatch.delenv("SCENESMITH_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SCENESMITH_LLM_RESPONSE_START_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SCENESMITH_LLM_CLI_INACTIVITY_TIMEOUT_SECONDS", raising=False)

    executor = _CodexExecutor()
    assert executor.response_start_timeout == 15.0
    assert executor.timeout == 45.0


def test_subscription_queue_admission_is_bounded(monkeypatch):
    monkeypatch.setenv("SCENESMITH_LLM_QUEUE_TIMEOUT_SECONDS", "0.05")
    lock = threading.Lock()
    lock.acquire()
    release = threading.Thread(target=lambda: (time.sleep(0.25), lock.release()))
    release.start()
    started = time.monotonic()
    with pytest.raises(SubscriptionQueueBusy, match="queue admission"):
        _run_subscription_command(
            lock,
            [sys.executable, "-c", "print('ok')"],
            timeout_seconds=0.2,
            text=True,
            capture_output=True,
        )
    release.join()

    assert time.monotonic() - started < 0.3


def test_subscription_response_start_stops_a_silent_process():
    with pytest.raises(SubscriptionCommandTimeout) as raised:
        _run_subscription_command(
            threading.Lock(),
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=1,
            response_start_timeout_seconds=0.05,
            text=True,
            capture_output=True,
        )
    assert raised.value.reason == "response start"


def test_explicit_cancellation_stops_active_process_group():
    cancel_event = threading.Event()
    canceller = threading.Thread(target=lambda: (time.sleep(0.05), cancel_event.set()))
    canceller.start()
    started = time.monotonic()

    with pytest.raises(SubscriptionCommandCancelled):
        _run_subscription_command(
            threading.Lock(),
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=5,
            cancel_event=cancel_event,
            text=True,
            capture_output=True,
        )

    canceller.join()
    assert time.monotonic() - started < 2


def test_subscription_progress_refreshes_rolling_inactivity_deadline():
    script = (
        "import time\n"
        "for index in range(6):\n"
        " print(index, flush=True)\n"
        " time.sleep(0.03)\n"
    )
    result = _run_subscription_command(
        threading.Lock(),
        [sys.executable, "-c", script],
        timeout_seconds=0.06,
        response_start_timeout_seconds=0.06,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["0", "1", "2", "3", "4", "5"]


def test_subscription_stops_after_stream_goes_quiet():
    script = "import time; print('started', flush=True); time.sleep(5)"
    with pytest.raises(SubscriptionCommandTimeout) as raised:
        _run_subscription_command(
            threading.Lock(),
            [sys.executable, "-c", script],
            timeout_seconds=0.05,
            response_start_timeout_seconds=0.2,
            text=True,
            capture_output=True,
        )
    assert raised.value.reason == "stream inactivity"


def test_subscription_absolute_deadline_bounds_continuous_output(monkeypatch):
    monkeypatch.setenv("SCENESMITH_LLM_HARD_TIMEOUT_SECONDS", "0.12")
    script = (
        "import time\n"
        "for index in range(100):\n"
        " print(index, flush=True)\n"
        " time.sleep(0.02)\n"
    )
    with pytest.raises(SubscriptionCommandTimeout) as raised:
        _run_subscription_command(
            threading.Lock(),
            [sys.executable, "-c", script],
            timeout_seconds=0.08,
            response_start_timeout_seconds=0.08,
            text=True,
            capture_output=True,
        )
    assert raised.value.reason == "absolute deadline"


def test_tool_schema_restricts_calls_to_supplied_tools():
    schema = _tool_output_schema(
        [{"type": "function", "function": {"name": "place_chair"}}]
    )

    name_schema = schema["properties"]["tool_calls"]["items"]["properties"]["name"]
    assert name_schema["enum"] == ["place_chair"]
    argument_schema = schema["properties"]["tool_calls"]["items"]["properties"]
    assert argument_schema["arguments"] == {"type": "object"}
    assert "arguments_json" not in argument_schema


def test_tool_schema_deduplicates_repeated_function_names():
    schema = _tool_output_schema(
        [
            {"type": "function", "function": {"name": "observe_scene"}},
            {"type": "function", "function": {"name": "observe_scene"}},
        ]
    )

    name_schema = schema["properties"]["tool_calls"]["items"]["properties"]["name"]
    assert name_schema["enum"] == ["observe_scene"]


def test_empty_tool_selection_becomes_clean_final_response():
    result = _chat_completion_response(
        raw_output=json.dumps(
            {"kind": "tool_calls", "content": "Nothing to add.", "tool_calls": []}
        ),
        tools=[{"type": "function", "function": {"name": "place_art"}}],
        model="haiku",
    )

    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "Nothing to add."}


def test_required_single_optional_tool_coerces_final_text_to_safe_empty_call():
    result = _chat_completion_response(
        raw_output=json.dumps(
            {"kind": "final", "content": "I would submit it.", "tool_calls": []}
        ),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "submit_scene",
                    "parameters": {
                        "type": "object",
                        "properties": {"plan": {}},
                        "required": ["plan"],
                        "additionalProperties": True,
                    },
                },
            }
        ],
        model="haiku",
        tool_choice="required",
    )

    function = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert function["name"] == "submit_scene"
    assert json.loads(function["arguments"]) == {}


def test_required_tool_choice_rejects_text_when_required_arguments_are_missing():
    with pytest.raises(ValueError, match="tool call was required"):
        _chat_completion_response(
            raw_output=json.dumps(
                {"kind": "final", "content": "I would submit it.", "tool_calls": []}
            ),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "place_scene",
                        "parameters": {
                            "type": "object",
                            "properties": {"scene_id": {"type": "string"}},
                            "required": ["scene_id"],
                        },
                    },
                }
            ],
            model="haiku",
            tool_choice="required",
        )


def test_required_tool_choice_accepts_any_supplied_tool():
    result = _chat_completion_response(
        raw_output=json.dumps(
            {
                "kind": "tool_calls",
                "content": "",
                "tool_calls": [{"name": "submit_scene", "arguments": {}}],
            }
        ),
        tools=[{"type": "function", "function": {"name": "submit_scene"}}],
        model="haiku",
        tool_choice="required",
    )

    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_boundary_accepts_anthropic_style_action_and_repairs_arguments():
    result = _chat_completion_response(
        raw_output="""```json
        {actions: [{tool: 'Place Chair', input: {chairCount: '2 chairs', blue: 'yes',},}],}
        ```""",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "place_chair",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chair_count": {"type": "integer"},
                            "blue": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
            }
        ],
        model="qwen",
        tool_choice="required",
    )

    function = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert function["name"] == "place_chair"
    assert json.loads(function["arguments"]) == {"chair_count": 2, "blue": True}


def test_required_single_tool_accepts_direct_arguments_object():
    result = _chat_completion_response(
        raw_output="{'roomSpecs': [], 'wallHeightMeters': '3.2m',}",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "submit_floor_plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "room_specs": {"type": "array"},
                            "wall_height_meters": {"type": "number"},
                        },
                    },
                },
            }
        ],
        model="haiku",
        tool_choice="required",
    )

    function = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(function["arguments"]) == {
        "room_specs": [],
        "wall_height_meters": 3.2,
    }


def test_legacy_arguments_json_is_repaired_without_model_repair_call():
    result = _chat_completion_response(
        raw_output=json.dumps(
            {
                "kind": "tool_calls",
                "content": "",
                "tool_calls": [
                    {
                        "name": "place_chair",
                        "arguments_json": "{'x': 1, 'y': 2,}",
                    }
                ],
            }
        ),
        tools=[{"type": "function", "function": {"name": "place_chair"}}],
        model="opus",
    )

    arguments = result["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert json.loads(arguments) == {"x": 1, "y": 2}


def test_codex_executor_returns_openai_tool_call_without_api_key(monkeypatch):
    monkeypatch.setenv("SCENESMITH_CODEX_BINARY", "true")
    monkeypatch.setenv("SCENESMITH_LLM_MODEL", "subscription-model")
    monkeypatch.setenv("OPENAI_API_KEY", "exhausted-key")

    def fake_run(_lock, command, **kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "kind": "tool_calls",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "place_chair",
                            "arguments": {"x": 1, "y": 2},
                        }
                    ],
                }
            )
        )
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "scenesmith.agent_utils.llm.cli_llm_proxy._run_subscription_command", fake_run
    )
    result = _CodexExecutor().complete(
        {
            "model": "fallback-model",
            "messages": [{"role": "user", "content": "Place a chair."}],
            "tools": [{"type": "function", "function": {"name": "place_chair"}}],
        }
    )

    choice = result["choices"][0]
    assert result["model"] == "fallback-model"
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "place_chair",
        "arguments": '{"x":1,"y":2}',
    }
