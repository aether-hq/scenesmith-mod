"""Tests for the retained semantic and visual regression corpus."""

import json

from pathlib import Path

from scenesmith.agent_utils.semantics.publication.semantic_regression import (
    BaselineObservation,
    ObjectCountExpectation,
    TopologyExpectation,
    compare_candidate_build,
    compare_reference_build,
    load_regression_corpus,
    observe_reference_build,
    validate_reference_run,
    verify_visual_references,
)

CORPUS_PATH = (
    Path(__file__).parents[2] / "test_data" / "semantic_regression" / "corpus.json"
)


def test_corpus_covers_approved_failure_generic_and_adversarial_cases():
    corpus = load_regression_corpus(CORPUS_PATH)

    assert len(corpus.cases) >= 8
    assert corpus.case("renaissance-library").reference.semantic_status == "approved"
    dock = corpus.case("spaceship-repair-dock")
    assert dock.reference.semantic_status == "approved"
    assert dock.target_outcome == "complete"
    assert dock.artifact_contract.require_semantic_certificate
    assert dock.artifact_contract.require_closed_ledger
    assert dock.artifact_contract.require_physics_verified
    assert dock.reference.known_semantic_gaps == ()
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
        job_status="failed",
        error="generic room published",
        duration_seconds=10.0,
        topology=TopologyExpectation(levels=1, spaces=1, openings=1, connectors=0),
        final_object_count=1,
        object_counts={},
        exports=("scene.glb",),
        usage=None,
    )

    result = compare_reference_build(case, observation)

    assert not result.matches_reference
    assert any("job_status expected completed" in issue for issue in result.issues)
    assert any("duration_seconds expected" in issue for issue in result.issues)
    assert any("topology expected" in issue for issue in result.issues)
    assert any("final objects expected >= 90" in issue for issue in result.issues)
    assert any("objects /hero_space_fighter/" in issue for issue in result.issues)
    assert "missing exports: scene_textured.glb" in result.issues


def test_synthetic_retained_run_is_reproducibly_validated(tmp_path):
    corpus = load_regression_corpus(CORPUS_PATH)
    case = corpus.case("spaceship-repair-dock")
    run_dir = tmp_path / case.reference.run_id
    scene_dir = run_dir / "scene_000"
    room_dir = scene_dir / "room_hangar"
    final_dir = room_dir / "scene_states" / "final_scene"
    final_dir.mkdir(parents=True)
    (run_dir / "scene.glb").write_bytes(b"solid")
    (run_dir / "scene_textured.glb").write_bytes(b"textured")
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "id": case.reference.run_id,
                "status": "completed",
                "error": None,
                "startedAt": "2026-08-21T00:00:00.000Z",
                "finishedAt": "2026-08-21T00:05:48.393Z",
                "usage": {
                    "requests": 5,
                    "turns": 11,
                    "totalTokens": 117166,
                    "apiEquivalentCostUsd": 0.4160854000000001,
                },
            }
        )
    )
    (scene_dir / "scene_blueprint.json").write_text(
        json.dumps(
            {
                "levels": [{}],
                "spaces": [{}],
                "openings": [{} for _ in range(9)],
                "connectors": [],
            }
        )
    )
    names = (
        ["hero_space_fighter"]
        + [f"repair_bay_{index}" for index in range(10)]
        + [f"repair_machine_{index}" for index in range(10)]
        + [f"repair_parts_rack_{index}" for index in range(20)]
        + [f"misc_equipment_prop_{index}" for index in range(15)]
        + [f"generic_{index}" for index in range(41)]
    )
    (final_dir / "scene_state.json").write_text(
        json.dumps({"objects": {name: {"name": name} for name in names}})
    )
    (room_dir / "semantic_publication_certificate.json").write_text(
        json.dumps({"publishable": True, "physics_verified": True})
    )
    (room_dir / "semantic_obligation_summary.json").write_text(
        json.dumps({"closed": True, "publishable": True})
    )

    result = validate_reference_run(
        CORPUS_PATH,
        case_id=case.case_id,
        run_root=tmp_path,
    )

    assert result.matches_reference, result.issues
    assert result.observation.duration_seconds == 348.393
    assert result.observation.exports == ("scene.glb", "scene_textured.glb")
    assert result.observation.final_object_count == 97
    assert result.observation.physics_verified is True


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


def _approved_library_candidate(**updates):
    case = load_regression_corpus(CORPUS_PATH).case("renaissance-library")
    values = {
        "run_id": "candidate-library",
        "job_status": "completed",
        "duration_seconds": 1000.0,
        "topology": case.reference.topology,
        "final_object_count": 93,
        "object_counts": {
            expectation.pattern: expectation.minimum
            for expectation in case.reference.object_counts
        },
        "exports": case.reference.required_exports,
        "usage": {
            "requests": 40,
            "turns": 100,
            "totalTokens": 800_000,
            "apiEquivalentCostUsd": 1.5,
        },
        "semantic_certificate_count": 1,
        "semantic_publishable": True,
        "physics_verified": True,
        "ledger_closed": True,
        "ledger_publishable": True,
        "room_kit_ids": ("library-reading-hall-v1",),
    }
    values.update(updates)
    return case, BaselineObservation(**values)


def test_candidate_comparison_protects_artifacts_and_reports_bounded_operations():
    case, observation = _approved_library_candidate()

    result = compare_candidate_build(case, observation)

    assert result.passed, result.issues
    deltas = {item.metric: item for item in result.operational_deltas}
    assert deltas["duration_seconds"].baseline == 1165.123
    assert round(deltas["duration_seconds"].delta, 3) == -165.123
    assert deltas["duration_seconds"].within_bounds
    assert deltas["total_tokens"].baseline is None
    assert deltas["total_tokens"].maximum == 1_000_000
    assert deltas["total_tokens"].within_bounds
    assert deltas["failure_rate"].observed == 0.0


def test_resumed_candidate_accepts_explicit_full_pipeline_operational_evidence():
    case, candidate = _approved_library_candidate(
        run_id="resumed-library",
        duration_seconds=90.0,
        usage=None,
    )
    _, operational = _approved_library_candidate(run_id="full-library")

    result = compare_candidate_build(
        case,
        candidate,
        operational_observation=operational,
    )

    assert result.passed, result.issues
    assert result.candidate_run_id == "resumed-library"
    assert result.operational_run_id == "full-library"
    assert result.observation.usage is None
    assert {item.metric: item.observed for item in result.operational_deltas}[
        "requests"
    ] == 40


def test_candidate_comparison_fails_closed_on_missing_evidence_and_usage():
    case, observation = _approved_library_candidate(
        duration_seconds=1600.0,
        usage=None,
        semantic_certificate_count=0,
        semantic_publishable=None,
        physics_verified=None,
        ledger_closed=None,
        ledger_publishable=None,
        room_kit_ids=(),
    )

    result = compare_candidate_build(
        case,
        observation,
        attempts=2,
        failed_attempts=1,
    )

    assert not result.passed
    assert "semantic publication certificate is required but missing" in result.issues
    assert "semantic obligation ledger is not closed" in result.issues
    assert "missing required room kits: library-reading-hall-v1" in result.issues
    assert any("operational duration_seconds" in issue for issue in result.issues)
    assert any("operational total_tokens" in issue for issue in result.issues)
    assert any("operational failure_rate" in issue for issue in result.issues)


def test_candidate_observation_reads_publication_ledger_and_room_kit(tmp_path):
    case = load_regression_corpus(CORPUS_PATH).case("renaissance-library")
    run_dir = tmp_path / "candidate-library"
    room_dir = run_dir / "scene_000" / "room_room"
    final_dir = room_dir / "scene_states" / "final_scene"
    final_dir.mkdir(parents=True)
    (run_dir / "scene.glb").write_bytes(b"solid")
    (run_dir / "scene_textured.glb").write_bytes(b"textured")
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "id": run_dir.name,
                "status": "completed",
                "startedAt": "2026-08-21T00:00:00Z",
                "finishedAt": "2026-08-21T00:16:40Z",
                "usage": {
                    "requests": 40,
                    "turns": 100,
                    "totalTokens": 800_000,
                    "apiEquivalentCostUsd": 1.5,
                },
            }
        )
    )
    (run_dir / "scene_000" / "scene_blueprint.json").write_text(
        json.dumps(
            {
                "levels": [{} for _ in range(case.reference.topology.levels)],
                "spaces": [{} for _ in range(case.reference.topology.spaces)],
                "openings": [{} for _ in range(case.reference.topology.openings)],
                "connectors": [{} for _ in range(case.reference.topology.connectors)],
            }
        )
    )
    names = (
        [f"renaissance_bookcase_{index}" for index in range(15)]
        + [f"classical_statue_{index}" for index in range(3)]
        + [f"reading_table_{index}" for index in range(7)]
        + [f"reading_chair_{index}" for index in range(13)]
        + [f"burgundy_book_row_{index}" for index in range(36)]
        + ["chandelier"]
        + [f"generic_{index}" for index in range(18)]
    )
    (final_dir / "scene_state.json").write_text(
        json.dumps({"objects": {name: {"name": name} for name in names}})
    )
    (room_dir / "semantic_publication_certificate.json").write_text(
        json.dumps({"publishable": True, "physics_verified": True})
    )
    (room_dir / "semantic_obligation_summary.json").write_text(
        json.dumps({"closed": True, "publishable": True})
    )
    (room_dir / "room_kit.json").write_text(
        json.dumps({"kit_id": "library-reading-hall-v1"})
    )

    observation = observe_reference_build(run_dir, case)
    result = compare_candidate_build(case, observation)

    assert observation.semantic_certificate_count == 1
    assert observation.room_kit_ids == ("library-reading-hall-v1",)
    assert result.passed, result.issues
