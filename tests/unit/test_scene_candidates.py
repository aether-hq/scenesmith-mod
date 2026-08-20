"""Tests for deterministic proxy-scene generation and ranking."""

import json

from scenesmith.agent_utils.scene_blueprint import blueprint_from_prompt
from scenesmith.agent_utils.scene_candidates import (
    create_candidate_tournament,
    persist_candidate_tournament,
)


def test_tournament_builds_six_deterministic_candidates_and_one_winner():
    prompt = "A large library with four reading tables and sixteen chairs"
    blueprint = blueprint_from_prompt(prompt)

    first = create_candidate_tournament(blueprint, prompt=prompt)
    second = create_candidate_tournament(blueprint, prompt=prompt)

    assert first == second
    assert len(first.candidates) == 6
    assert (
        sum(candidate.candidate_id == first.winner_id for candidate in first.candidates)
        == 1
    )
    assert first.winner.viable
    assert not first.comparator_used


def test_candidate_scores_cover_every_quality_dimension():
    prompt = "A two-level jewel-toned library with spiral stairs"
    tournament = create_candidate_tournament(
        blueprint_from_prompt(prompt), prompt=prompt, candidate_count=4
    )

    score_payload = tournament.winner.scores.to_dict()

    assert set(score_payload) == {
        "hard_validity",
        "prompt_coverage",
        "requested_counts",
        "circulation",
        "proportions",
        "density",
        "sightlines",
        "connectivity",
        "kit_coherence",
        "focal_hierarchy",
        "style_fit",
        "diversity",
        "total",
    }
    assert score_payload["hard_validity"] == 10
    assert score_payload["connectivity"] == 8


def test_room_kit_counts_are_preserved_in_every_candidate():
    prompt = "A medical exam room with one patient bed and one medical device"
    tournament = create_candidate_tournament(
        blueprint_from_prompt(prompt), prompt=prompt
    )

    for candidate in tournament.candidates:
        roles = candidate.blueprint.furniture_groups[0].roles
        assert roles["patient_bed"] == 1
        assert roles["medical_device"] == 1


def test_candidate_count_is_clamped_to_four_through_eight():
    prompt = "A calm studio"
    blueprint = blueprint_from_prompt(prompt)

    assert (
        len(
            create_candidate_tournament(
                blueprint, prompt=prompt, candidate_count=1
            ).candidates
        )
        == 4
    )
    assert (
        len(
            create_candidate_tournament(
                blueprint, prompt=prompt, candidate_count=99
            ).candidates
        )
        == 8
    )


def test_tournament_persistence_includes_lineage_scores_and_winner(tmp_path):
    prompt = "A bar lounge"
    tournament = create_candidate_tournament(
        blueprint_from_prompt(prompt), prompt=prompt
    )
    output = tmp_path / "scene_candidates.json"

    persist_candidate_tournament(tournament, output)

    payload = json.loads(output.read_text())
    assert payload["winner_id"] == tournament.winner_id
    assert payload["candidate_count"] == 6
    assert all(candidate["parent_blueprint_id"] for candidate in payload["candidates"])
    assert all("total" in candidate["scores"] for candidate in payload["candidates"])
