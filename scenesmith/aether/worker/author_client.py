"""HTTP boundary to Aether's attributed contextual-completion agent."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


class CompletionAuthorError(RuntimeError):
    """The attributed completion author could not return a valid patch."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


Request = Callable[[urllib.request.Request, float], HttpResponse]


def _urlopen(request: urllib.request.Request, timeout: float) -> HttpResponse:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                response.status, dict(response.headers), response.read()
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, dict(exc.headers), exc.read())


class AetherCompletionClient:
    """Queue one seam-bound authoring operation and wait for its typed result.

    The bearer token is held only in request headers. It is never included in
    returned diagnostics or persisted completion artifacts.
    """

    def __init__(
        self,
        *,
        base_url: str,
        project_id: str,
        bearer_token: str,
        workspace_id: str | None = None,
        poll_interval_seconds: float = 1.0,
        timeout_seconds: float = 180.0,
        request: Request = _urlopen,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Aether API base URL must use HTTP or HTTPS")
        if not project_id or not bearer_token:
            raise ValueError("Aether project id and bearer token are required")
        if poll_interval_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError("poll interval and timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.bearer_token = bearer_token
        self.workspace_id = workspace_id
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.request = request
        self.monotonic = monotonic
        self.sleep = sleep

    def author_patch(self, authoring: dict[str, Any]) -> dict[str, Any]:
        round_index = int(authoring["report"]["round_index"])
        digest = str(authoring["report"]["census_sha256"])
        idempotency_key = (
            f"{authoring['job_id']}-completion-{round_index}-{digest[:16]}"
        )
        queued = self._json_request(
            "POST",
            f"/api/v1/projects/{self.project_id}/commands/requestContextualCompletion",
            {"idempotency_key": idempotency_key, "authoring": authoring},
        )
        run_id = queued.get("inference_run_id")
        if not isinstance(run_id, str) or not run_id:
            raise CompletionAuthorError(
                "Aether returned no contextual-completion run id"
            )
        result_path = (
            f"/api/v1/projects/{self.project_id}/scenic-design/"
            f"contextual-completions/{run_id}"
        )
        deadline = self.monotonic() + self.timeout_seconds
        while True:
            response = self._request("GET", result_path, None)
            payload = self._decode(response)
            if response.status == 200:
                patch = payload.get("patch")
                if not isinstance(patch, dict):
                    raise CompletionAuthorError(
                        "Aether completed contextual authoring without a typed patch"
                    )
                return patch
            if response.status != 409 or payload.get("code") != "precommit_unmet":
                self._raise_response(response.status, payload)
            if self.monotonic() >= deadline:
                raise CompletionAuthorError(
                    f"contextual completion run {run_id} exceeded "
                    f"{self.timeout_seconds:g} seconds"
                )
            retry_after = response.headers.get("Retry-After")
            delay = self.poll_interval_seconds
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            self.sleep(min(delay, max(0.0, deadline - self.monotonic())))

    def _json_request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        response = self._request(method, path, payload)
        decoded = self._decode(response)
        if not 200 <= response.status < 300:
            self._raise_response(response.status, decoded)
        return decoded

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> HttpResponse:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.bearer_token}",
        }
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, separators=(",", ":")).encode()
        if self.workspace_id is not None:
            headers["X-Aether-Workspace-Id"] = self.workspace_id
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            return self.request(request, min(self.timeout_seconds, 30.0))
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise CompletionAuthorError(
                f"Aether contextual-completion request failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _decode(response: HttpResponse) -> dict[str, Any]:
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompletionAuthorError(
                f"Aether returned non-JSON HTTP {response.status}"
            ) from exc
        if not isinstance(value, dict):
            raise CompletionAuthorError(
                f"Aether returned a non-object HTTP {response.status} response"
            )
        return value

    @staticmethod
    def _raise_response(status: int, payload: dict[str, Any]) -> None:
        code = str(
            payload.get("code") or payload.get("error", {}).get("code") or "unknown"
        )
        message = str(
            payload.get("detail")
            or payload.get("message")
            or payload.get("error", {}).get("message")
            or "no public diagnostic"
        )
        raise CompletionAuthorError(
            f"Aether contextual completion failed: HTTP {status} {code}: {message}"
        )
