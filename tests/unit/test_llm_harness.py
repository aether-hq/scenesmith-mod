import json
import asyncio

import pytest

from agents import ModelSettings

from scenesmith.agent_utils.llm_harness import (
    LLMCircuitOpenError,
    LLMCapabilityError,
    LLMHarness,
    LLMHarnessError,
    LLMHarnessConfig,
    LLMRequest,
    LLMStructuredOutputError,
    LLMTimeoutError,
    HardenedModelProvider,
    extract_json_object,
    normalize_agent_model_response,
    parse_json_object,
)
from scenesmith.agent_utils import llm_harness as llm_harness_module
from scenesmith.agent_utils.scene_analyzer import deterministic_manipuland_assignment


class FakeAdapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TransientFailure(RuntimeError):
    status_code = 503


def cli_config(*, attempts=2, fallbacks=()):
    return LLMHarnessConfig(
        provider="claude-cli",
        model="haiku",
        timeout_seconds=30,
        max_attempts=attempts,
        fallback_models=tuple(fallbacks),
    )


@pytest.fixture(autouse=True)
def isolated_circuit(monkeypatch):
    monkeypatch.setattr(
        llm_harness_module,
        "_CIRCUIT",
        llm_harness_module._LLMCircuitBreaker(cooldown_seconds=60),
    )


def test_extract_json_object_repairs_fence_and_surrounding_prose():
    assert json.loads(extract_json_object('Answer:\n```json\n{"ok":true}\n```')) == {
        "ok": True
    }


def test_unrecoverable_json_retries_without_a_paid_repair_prompt():
    adapter = FakeAdapter(['{"ok":'] * 3)
    harness = LLMHarness(cli_config(attempts=3), adapter=adapter)

    with pytest.raises(LLMHarnessError, match="llm_routes_exhausted"):
        harness.complete(
            LLMRequest(
                model="ignored",
                messages=[{"role": "user", "content": "JSON"}],
                response_format={"type": "json_object"},
            )
        )

    assert adapter.calls == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Result: ```json\n{'ok': True, 'count': 2,}\n```", {"ok": True, "count": 2}),
        ('{ok: true, items: [1, 2,],}', {"ok": True, "items": [1, 2]}),
        ('preface {"ok": true', {"ok": True}),
        ('{/* generated */ "ok": true, // accepted\n}', {"ok": True}),
    ],
)
def test_provider_neutral_json_repair_handles_common_model_variants(raw, expected):
    assert parse_json_object(raw) == expected


def test_native_agent_response_repairs_tool_name_keys_and_primitive_types():
    class Tool:
        name = "place_chair"
        params_json_schema = {
            "type": "object",
            "properties": {
                "chair_count": {"type": "integer"},
                "is_blue": {"type": "boolean"},
            },
            "additionalProperties": False,
        }

    item = type(
        "ToolCall",
        (),
        {
            "type": "function_call",
            "name": "Place Chair",
            "arguments": "{'chairCount': '3 chairs', 'isBlue': 'yes',}",
        },
    )()
    response = type("Response", (), {"output": [item]})()

    normalize_agent_model_response(response, tools=[Tool()], output_schema=None)

    assert item.name == "place_chair"
    assert json.loads(item.arguments) == {"chair_count": 3, "is_blue": True}


def test_transient_provider_failure_retries_once_then_normalizes_json():
    adapter = FakeAdapter([TransientFailure("down"), '```json\n{"ok":true}\n```'])
    harness = LLMHarness(cli_config(attempts=2), adapter=adapter)

    output = harness.complete(
        LLMRequest(
            model="ignored",
            messages=[{"role": "user", "content": "JSON"}],
            response_format={"type": "json_object"},
        )
    )

    assert json.loads(output) == {"ok": True}
    assert adapter.calls == 2


def test_timeout_retries_then_uses_the_next_model_without_opening_circuit():
    adapter = FakeAdapter(
        [TimeoutError("deadline"), TimeoutError("deadline"), '{"ok":true}']
    )
    harness = LLMHarness(
        cli_config(attempts=2, fallbacks=("opus",)), adapter=adapter
    )

    output = harness.complete(
        LLMRequest(
            model="ignored",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert json.loads(output) == {"ok": True}
    assert adapter.calls == 3


def test_exhausting_every_timeout_route_opens_the_circuit():
    adapter = FakeAdapter([TimeoutError("deadline")] * 4)
    harness = LLMHarness(
        cli_config(attempts=2, fallbacks=("opus",)), adapter=adapter
    )

    with pytest.raises(LLMTimeoutError, match="llm_turn_exhausted"):
        harness.complete(
            LLMRequest(model="ignored", messages=[{"role": "user", "content": "hello"}])
        )

    with pytest.raises(LLMCircuitOpenError):
        harness.complete(
            LLMRequest(
                model="ignored",
                messages=[{"role": "user", "content": "second object"}],
            )
        )

    assert adapter.calls == 4


class FakeAgentModel:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    async def get_response(self, *_args, **_kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeAgentProvider:
    def __init__(self, outcomes):
        self.outcomes = {name: list(values) for name, values in outcomes.items()}
        self.requested = []

    def get_model(self, model_name):
        self.requested.append(model_name)
        return FakeAgentModel(self.outcomes[model_name])


def test_agents_sdk_path_retries_cancels_orphans_and_fails_over_models(monkeypatch):
    cancellations = []
    monkeypatch.setattr(
        llm_harness_module,
        "_cancel_subscription_turn",
        lambda: cancellations.append(True) or True,
    )
    raw = FakeAgentProvider(
        {
            "haiku": [TimeoutError("deadline"), TimeoutError("deadline")],
            "opus": ["usable response"],
        }
    )
    provider = HardenedModelProvider(
        raw, cli_config(attempts=2, fallbacks=("opus",))
    )

    response = asyncio.run(
        provider.get_model("haiku").get_response(model_settings=ModelSettings())
    )

    assert response == "usable response"
    assert raw.requested == ["haiku", "opus"]
    assert cancellations == [True, True]


def test_agents_sdk_does_not_apply_direct_api_deadline_to_cli_bridge():
    class DelayedAgentModel:
        async def get_response(self, *_args, **_kwargs):
            await asyncio.sleep(0.03)
            return "stream-managed response"

    class DelayedProvider:
        def get_model(self, _model_name):
            return DelayedAgentModel()

    config = LLMHarnessConfig(
        provider="claude-cli",
        model="sonnet",
        timeout_seconds=0.01,
        max_attempts=1,
    )
    provider = HardenedModelProvider(DelayedProvider(), config)

    response = asyncio.run(
        provider.get_model("sonnet").get_response(model_settings=ModelSettings())
    )

    assert response == "stream-managed response"


def test_agents_sdk_cli_request_forces_loopback_timeout(monkeypatch):
    monkeypatch.setenv("SCENESMITH_LLM_PROXY_HTTP_TIMEOUT_SECONDS", "150")

    class CapturingModel:
        async def get_response(self, *_args, **kwargs):
            return kwargs["model_settings"].extra_args["timeout"]

    class CapturingProvider:
        def get_model(self, _model_name):
            return CapturingModel()

    config = LLMHarnessConfig(
        provider="claude-cli",
        model="sonnet",
        timeout_seconds=30,
        max_attempts=1,
    )
    provider = HardenedModelProvider(CapturingProvider(), config)

    timeout = asyncio.run(
        provider.get_model("sonnet").get_response(model_settings=ModelSettings())
    )

    assert timeout == 150


def test_unknown_compatible_endpoint_fails_closed_on_multimodal_request(monkeypatch):
    for name in ("VISION", "TOOLS", "STRUCTURED_OUTPUT"):
        monkeypatch.delenv(f"SCENESMITH_LLM_CAP_{name}", raising=False)
    config = LLMHarnessConfig(
        provider="openai-compatible",
        model="unknown-model",
        timeout_seconds=30,
        max_attempts=1,
    )
    harness = LLMHarness(config, adapter=FakeAdapter(['{"ok":true}']))

    with pytest.raises(LLMCapabilityError, match="vision, structured_output"):
        harness.complete(
            LLMRequest(
                model="ignored",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "inspect"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
        )


def test_native_provider_model_is_namespaced_once():
    config = LLMHarnessConfig("anthropic", "claude-sonnet", 30, 1)
    assert config.routed_model == "anthropic/claude-sonnet"
    already_routed = LLMHarnessConfig("openrouter", "openrouter/model", 30, 1)
    assert already_routed.routed_model == "openrouter/model"


@pytest.mark.parametrize(
    ("name", "description", "expected"),
    [
        ("medical bed", "sci-fi treatment bed", True),
        ("storage cabinet", "sterile supplies", True),
        ("diagnostic monitor", "wall screen", False),
        ("chair", "metal cafe chair", False),
    ],
)
def test_deterministic_manipuland_surface_classification(name, description, expected):
    obj = type("Object", (), {"name": name, "description": description})()
    assert (deterministic_manipuland_assignment(obj) is not None) is expected


def test_medical_bed_assignment_excludes_hard_equipment():
    obj = type(
        "Object", (), {"name": "medical bed", "description": "treatment bed"}
    )()
    suggested_items, _priority = deterministic_manipuland_assignment(obj)
    assert suggested_items == "one pillow"
