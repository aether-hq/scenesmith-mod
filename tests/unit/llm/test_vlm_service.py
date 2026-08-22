from scenesmith.agent_utils.llm.llm_harness import LLMHarnessConfig
from scenesmith.agent_utils.llm.vlm_service import VLMService


class RecordingHarness:
    def __init__(self):
        self.config = LLMHarnessConfig("claude-cli", "haiku", 30, 1)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return '{"ok":true}'


def test_vlm_service_is_a_thin_provider_neutral_facade():
    harness = RecordingHarness()
    service = VLMService(harness=harness)

    output = service.create_completion(
        model="caller-model-is-not-authoritative",
        messages=[{"role": "user", "content": "return JSON"}],
        reasoning_effort="low",
        verbosity="low",
        response_format={"type": "json_object"},
        vision_detail="high",
    )

    assert output == '{"ok":true}'
    assert service.provider == "claude-cli"
    assert len(harness.requests) == 1
    request = harness.requests[0]
    assert request.response_format == {"type": "json_object"}
    assert request.vision_detail == "high"
