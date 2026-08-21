"""Tests for the retained semantic and visual regression corpus."""

import json

from pathlib import Path

from scenesmith.agent_utils.semantic_regression import (
    BaselineObservation,
    ObjectCountExpectation,
    TopologyExpectation,
    compare_reference_build,
    load_regression_corpus,
    validate_reference_run,
    verify_visual_references,
)


CORPUS_PATH = (
    Path(__file__).parents[1] / "test_data" / "semantic_regression" / "corpus.json"
)


def test_corpus_covers_approved_failure_generic_and_adversarial_cases():
    corpus = load_regression_corpus(CORPUS_PATH)

    assert len(corpus.cases) >= 8
    assert corpus.case("renaissance-library").reference.semantic_status == "approved"
    dock = corpus.case("spaceship-repair-dock")
    assert dock.reference.semantic_status == "known_failure"
    assert "ten repair bays discarded" in dock.reference.known_semantic_gaps
    assert corpus.case("simple-bedroom").reference is None
    assert corpus.case("contradictory-counts").target_outcome == "explicit_failure"


def test_object_count_expectation_rejects_invalid_bounds():
    try:
        ObjectCountExpectation(pattern="fighter", minimum=2, maximum=1)
    except ValueError as exc:
        assert "maximum object count" in str(exc)
    else:
        raise AssertionError("invalid expectation bounds were accepted")


def test_reference_comparison_reports_precise_semantic_drift():
    case = load_regression_corpus(CORPUS_PATH).case("spaceship-repair-dock")
    observation = BaselineObservation(
        run_id=case.reference.run_id,
        job_status="completed",
        duration_seconds=10.0,
        topology=TopologyExpectation(levels=1, spaces=1, openings=1, connectors=0),
        final_object_count=17,
        object_counts={
            expectation.pattern: (1 if "fighter" in expectation.pattern else 0)
            for expectation in case.reference.object_counts
        },
        exports=("scene.glb",),
        usage=None,
    )

    result = compare_reference_build(case, observation)

    assert not result.matches_reference
    assert any("topology expected" in issue for issue in result.issues)
    assert any("final objects expected >= 18" in issue for issue in result.issues)
    assert any("objects /fighter/ expected <= 0" in issue for issue in result.issues)
    assert any(
        "missing exports: scene_textured.glb" in issue for issue in result.issues
    )
    assert any("usage record is required" in issue for issue in result.issues)


def test_synthetic_retained_run_is_reproducibly_validated(tmp_path):
    corpus = load_regression_corpus(CORPUS_PATH)
    case = corpus.case("spaceship-repair-dock")
    run_dir = tmp_path / case.reference.run_id
    scene_dir = run_dir / "scene_000"
    final_dir = scene_dir / "room_room" / "scene_states" / "final_scene"
    final_dir.mkdir(parents=True)
    (run_dir / "scene.glb").write_bytes(b"solid")
    (run_dir / "scene_textured.glb").write_bytes(b"textured")
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "id": case.reference.run_id,
                "status": "completed",
                "startedAt": "2026-08-21T00:00:00Z",
                "finishedAt": "2026-08-21T00:10:00Z",
                "usage": {
                    "requests": 24,
                    "turns": 72,
                    "totalTokens": 401814,
                    "apiEquivalentCostUsd": 0.8217156999999999,
                },
            }
        )
    )
    (scene_dir / "scene_blueprint.json").write_text(
        json.dumps(
            {
                "levels": [{}],
                "spaces": [{}],
                "openings": [],
                "connectors": [],
            }
        )
    )
    object_names = [
        "equipment_rack",
        "repair_workbench",
        "burgundy_book_row",
        "encyclopedia_book_row",
        "candlestick_brass",
    ] + [f"generic_{index}" for index in range(13)]
    (final_dir / "scene_state.json").write_text(
        json.dumps({"objects": {name: {"name": name} for name in object_names}})
    )

    result = validate_reference_run(
        CORPUS_PATH,
        case_id=case.case_id,
        run_root=tmp_path,
    )

    assert result.matches_reference, result.issues
    assert result.observation.duration_seconds == 600.0


def test_visual_reference_hashes_match_retained_native_captures_when_available():
    corpus = load_regression_corpus(CORPUS_PATH)
    available_cases = [
        case
        for case in corpus.cases
        if case.reference
        and case.reference.visual_references
        and all(
            Path(reference.source_path).is_file()
            for reference in case.reference.visual_references
        )
    ]

    assert available_cases, "expected at least one retained native capture"
    for case in available_cases:
        assert verify_visual_references(case) == ()
