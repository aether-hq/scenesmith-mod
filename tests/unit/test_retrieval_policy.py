import json
import logging
import time

from queue import Queue

from scenesmith.agent_utils.hssd_retrieval_server.dataclasses import StreamedResult
from scenesmith.agent_utils.retrieval_policy import stream_local_results


def test_missing_callback_becomes_logged_stream_error(monkeypatch, caplog):
    monkeypatch.setenv("SCENESMITH_ASSET_RETRIEVAL_TIMEOUT_SECONDS", "0.01")
    started = time.monotonic()

    with caplog.at_level(logging.ERROR):
        lines = list(
            stream_local_results(
                result_queue=Queue(),
                batch_size=1,
                result_type=StreamedResult,
                catalog_name="HSSD",
                logger=logging.getLogger("retrieval-policy-test"),
            )
        )

    assert time.monotonic() - started < 0.5
    payload = json.loads(lines[0])
    assert payload["status"] == "error"
    assert payload["error"] == "HSSD local retrieval exceeded 0.01s"
    assert "failing 1 unfinished request" in caplog.text
