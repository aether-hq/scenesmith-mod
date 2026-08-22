"""Regression tests for physically valid pipeline checkpoint publication."""

import pytest

from scenesmith.experiments.indoor_scene_generation import (
    _require_projection_success,
    _require_semantic_publication_inputs,
)


def test_furniture_checkpoint_rejects_failed_projection():
    with pytest.raises(
        RuntimeError,
        match=r"furniture.*physical projection failed.*cannot publish",
    ):
        _require_projection_success("furniture", False)


def test_checkpoint_accepts_successful_projection():
    _require_projection_success("furniture", True)


def test_shadow_artifacts_cannot_bypass_mandatory_publication(tmp_path):
    graph = tmp_path / "scene_requirement_graph.json"
    graph.write_text("{}")

    with pytest.raises(
        RuntimeError,
        match=(
            "Semantic publication blocked: missing mandatory enforcement artifacts "
            "semantic_spatial_compilation.json.*diagnostic only"
        ),
    ):
        _require_semantic_publication_inputs(
            {
                "scene_requirement_graph.json": graph,
                "semantic_spatial_compilation.json": (
                    tmp_path / "semantic_spatial_compilation.json"
                ),
            }
        )


def test_complete_semantic_publication_inputs_are_accepted(tmp_path):
    artifacts = {
        name: tmp_path / name
        for name in (
            "scene_requirement_graph.json",
            "scene_blueprint.json",
            "semantic_publication_certificate.json",
        )
    }
    for artifact in artifacts.values():
        artifact.write_text("{}")

    _require_semantic_publication_inputs(artifacts)
