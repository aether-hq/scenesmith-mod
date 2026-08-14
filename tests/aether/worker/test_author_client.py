from __future__ import annotations

import json
import urllib.request

import pytest

from scenesmith.aether.worker.author_client import (
    AetherCompletionClient,
    CompletionAuthorError,
    HttpResponse,
)


def _brief() -> dict:
    return {
        "contract_version": 1,
        "job_id": "bar-one",
        "report": {
            "round_index": 2,
            "census_sha256": "a" * 64,
        },
    }


def test_author_patch_queues_once_and_polls_pending_result() -> None:
    requests: list[urllib.request.Request] = []
    responses = iter(
        (
            HttpResponse(200, {}, json.dumps({"inference_run_id": "run-1"}).encode()),
            HttpResponse(409, {"Retry-After": "0.2"}, b'{"code":"precommit_unmet"}'),
            HttpResponse(
                200, {}, b'{"inference_run_id":"run-1","patch":{"job_id":"bar-one"}}'
            ),
        )
    )
    clock = iter((0.0, 0.0, 0.0, 0.2, 0.2))
    sleeps: list[float] = []

    def request(value: urllib.request.Request, timeout: float) -> HttpResponse:
        requests.append(value)
        assert timeout == 30.0
        return next(responses)

    client = AetherCompletionClient(
        base_url="https://staging.aether.film",
        project_id="project-1",
        bearer_token="secret-token",
        workspace_id="workspace-1",
        poll_interval_seconds=0.1,
        request=request,
        monotonic=lambda: next(clock),
        sleep=sleeps.append,
    )
    assert client.author_patch(_brief()) == {"job_id": "bar-one"}
    assert [item.get_method() for item in requests] == ["POST", "GET", "GET"]
    body = json.loads(requests[0].data)
    assert body["idempotency_key"] == f"bar-one-completion-2-{'a' * 16}"
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    assert requests[0].headers["X-aether-workspace-id"] == "workspace-1"
    assert sleeps == [0.2]


def test_author_patch_reports_terminal_error_without_leaking_token() -> None:
    responses = iter(
        (
            HttpResponse(200, {}, b'{"inference_run_id":"run-1"}'),
            HttpResponse(
                409, {}, b'{"code":"version_conflict","detail":"agent failed"}'
            ),
        )
    )
    client = AetherCompletionClient(
        base_url="http://localhost:8000",
        project_id="project-1",
        bearer_token="do-not-leak",
        request=lambda _request, _timeout: next(responses),
    )
    with pytest.raises(CompletionAuthorError, match="version_conflict") as error:
        client.author_patch(_brief())
    assert "do-not-leak" not in str(error.value)


def test_author_patch_rejects_non_json_success() -> None:
    client = AetherCompletionClient(
        base_url="http://localhost:8000",
        project_id="project-1",
        bearer_token="token",
        request=lambda _request, _timeout: HttpResponse(200, {}, b"not json"),
    )
    with pytest.raises(CompletionAuthorError, match="non-JSON"):
        client.author_patch(_brief())
