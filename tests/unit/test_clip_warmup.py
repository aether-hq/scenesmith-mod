"""Regression tests for retrieval-server encoder warmup."""

from unittest.mock import patch

import numpy as np
import pytest

from scenesmith.agent_utils.clip_embeddings import warm_clip_text_encoder
from scenesmith.agent_utils.objaverse_retrieval.clip_similarity import (
    warm_objaverse_text_encoder,
)


def test_shared_clip_warmup_executes_encoder() -> None:
    with patch(
        "scenesmith.agent_utils.clip_embeddings.get_text_embedding",
        return_value=np.zeros(1024, dtype=np.float32),
    ) as embed:
        warm_clip_text_encoder(device="cpu")

    embed.assert_called_once_with("retrieval service warmup", device="cpu")


def test_shared_clip_warmup_rejects_wrong_embedding_space() -> None:
    with patch(
        "scenesmith.agent_utils.clip_embeddings.get_text_embedding",
        return_value=np.zeros(768, dtype=np.float32),
    ), pytest.raises(RuntimeError, match="Unexpected CLIP warmup embedding shape"):
        warm_clip_text_encoder(device="cpu")


def test_objaverse_clip_warmup_executes_catalog_encoder() -> None:
    with patch(
        "scenesmith.agent_utils.objaverse_retrieval.clip_similarity.get_objaverse_text_embedding",
        return_value=np.zeros(768, dtype=np.float32),
    ) as embed:
        warm_objaverse_text_encoder(device="cpu")

    embed.assert_called_once_with("retrieval service warmup", device="cpu")
