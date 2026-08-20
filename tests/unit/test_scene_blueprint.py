"""Tests for the canonical provider-neutral SceneBlueprint contract."""

import json

import pytest

from pydantic import ValidationError

from scenesmith.agent_utils.scene_blueprint import (
    BlueprintDesignTokens,
    SceneBlueprint,
    blueprint_from_prompt,
    diff_blueprints,
    floor_plan_submission_from_blueprint,
    normalize_scene_blueprint,
    persist_scene_blueprint,
)


def test_plain_prompt_produces_stable_multilevel_spiral_blueprint():
    prompt = "A two-level library with a usable spiral staircase"

    first = blueprint_from_prompt(prompt)
    second = blueprint_from_prompt(prompt)

    assert first == second
    assert len(first.levels) == 2
    assert len(first.connectors) == 1
    assert first.connectors[0].kind == "stairs_spiral"
    assert first.connectors[0].start.level_id != first.connectors[0].end.level_id
    assert SceneBlueprint.model_validate_json(first.model_dump_json()) == first


def test_anthropic_style_envelope_and_aliases_normalize_to_canonical_schema():
    raw = {
        "arguments_json": json.dumps(
            {
                "schema_version": 0,
                "stories": [
                    {"id": "ground", "name": "Ground", "elevation": 0},
                    {"id": "upper", "name": "Upper", "elevation": 3.1},
                ],
                "rooms": [
                    {
                        "id": "reading-ground",
                        "type": "library",
                        "level": 0,
                        "dimensions": {"length": 9, "width": 7},
                    },
                    {
                        "id": "reading-upper",
                        "type": "library",
                        "level": 1,
                        "dimensions": [9, 7],
                    },
                ],
                "stairs": [{"type": "spiral", "width": 1.2}],
            }
        )
    }

    blueprint = normalize_scene_blueprint(
        raw, prompt="A two-level library with spiral stairs"
    )

    assert blueprint.schema_version == 1
    assert [space.level_id for space in blueprint.spaces] == ["ground", "upper"]
    assert blueprint.connectors[0].kind == "stairs_spiral"
    assert "unwrapped arguments_json envelope" in blueprint.repair_log
    assert "renamed rooms to spaces" in blueprint.repair_log


def test_openai_style_partial_payload_gets_structural_fallbacks():
    blueprint = normalize_scene_blueprint(
        {"plan": {"style": {"lighting_mood": "warm"}}},
        prompt="A compact radio studio",
    )

    assert len(blueprint.levels) == 1
    assert len(blueprint.spaces) == 1
    assert blueprint.design_tokens.lighting_mood == "warm"
    assert "synthesized missing levels" in blueprint.repair_log


def test_blueprint_rejects_stairs_to_nowhere():
    valid = blueprint_from_prompt("A two-level room with stairs")
    payload = valid.model_dump(mode="json")
    payload["connectors"] = []

    with pytest.raises(ValidationError, match="unreachable"):
        SceneBlueprint.model_validate(payload)


def test_blueprint_diff_invalidates_only_downstream_stages():
    before = blueprint_from_prompt("A dining room")
    after = before.model_copy(
        update={
            "design_tokens": before.design_tokens.model_copy(
                update={"lighting_mood": "dramatic"}
            )
        }
    )

    diff = diff_blueprints(before, after)

    assert diff.changed_paths == ("design_tokens",)
    assert diff.invalidated_stages == (
        "furniture",
        "walls",
        "ceiling",
        "details",
        "render",
    )


def test_persisted_blueprint_round_trips(tmp_path):
    blueprint = blueprint_from_prompt("A warm bar lounge")
    output = tmp_path / "scene_blueprint.json"

    persist_scene_blueprint(blueprint, output)

    assert SceneBlueprint.model_validate_json(output.read_text()) == blueprint


def test_blueprint_projects_winning_dimensions_and_materials_to_floor_plan():
    blueprint = blueprint_from_prompt("A quiet studio").model_copy(
        update={
            "design_tokens": BlueprintDesignTokens(
                style_keywords=("graphic",),
                palette=("#111", "#eee"),
                material_roles={"floor": "black rubber", "walls": "white paint"},
                lighting_mood="crisp",
                focal_hierarchy=(),
            )
        }
    )

    submission = floor_plan_submission_from_blueprint(blueprint)

    assert submission["room_specs"][0]["width"] == 7.0
    assert submission["room_specs"][0]["depth"] == 7.0
    assert submission["materials"]["floor"] == "black rubber"
