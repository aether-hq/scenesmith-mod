from __future__ import annotations

import pytest

from scenesmith.aether.worker.pipeline import run_native_contextual_completion


def test_native_completion_refuses_missing_attributed_inference(
    monkeypatch, tmp_path
) -> None:
    for name in ("AETHER_API_URL", "AETHER_PROJECT_ID", "AETHER_BEARER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="AETHER_API_URL"):
        run_native_contextual_completion(
            scene=object(),
            stage_input={},
            cfg_dict={},
            logger=object(),
            house_layout=object(),
            ceiling_height=3,
            render_gpu_id=None,
            room_dir=tmp_path,
        )
