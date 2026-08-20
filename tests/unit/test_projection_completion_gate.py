"""Regression tests for physically valid pipeline checkpoint publication."""

import pytest

from scenesmith.experiments.indoor_scene_generation import (
    _require_projection_success,
)


def test_furniture_checkpoint_rejects_failed_projection():
    with pytest.raises(
        RuntimeError,
        match=r"furniture.*physical projection failed.*cannot publish",
    ):
        _require_projection_success("furniture", False)


def test_checkpoint_accepts_successful_projection():
    _require_projection_success("furniture", True)
