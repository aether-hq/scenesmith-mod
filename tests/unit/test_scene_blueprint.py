"""Tests for the canonical provider-neutral SceneBlueprint contract."""

import json

import pytest

from pydantic import ValidationError

from scenesmith.agent_utils.scene_blueprint import (
    BlueprintDesignTokens,
    ConnectorBlueprint,
    ConnectorEndpoint,
    LevelBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
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


def test_large_library_prompt_expands_canonical_footprint():
    prompt = (
        "a large, multi-level library with thousands of books and a bunch of tables "
        "and chairs for patrons. A spiral staircase connects the floors, and there "
        "are huge archted windows, statues, and so on, as it has a renaiissance , "
        "gorgeous decor."
    )

    blueprint = blueprint_from_prompt(prompt)

    assert all(min(space.dimensions_m) >= 12.0 for space in blueprint.spaces)
    assert len(blueprint.openings) >= 3
    assert all(opening.shape == "arched" for opening in blueprint.openings)
    assert all(opening.width_m >= 3.5 for opening in blueprint.openings)
    assert all(opening.height_m >= 3.0 for opening in blueprint.openings)
    submission = floor_plan_submission_from_blueprint(blueprint)
    assert submission["window_shape"] == "arched"
    assert submission["window_width_m"] >= 3.5


def test_renaissance_library_prompt_preserves_authored_design_tokens():
    prompt = (
        "a large, multi-level library with thousands of books and a bunch of tables "
        "and chairs for patrons. A spiral staircase connects the floors, and there "
        "are huge archted windows, statues, and so on, as it has a renaiissance , "
        "gorgeous decor."
    )

    tokens = blueprint_from_prompt(prompt).design_tokens

    assert "renaissance" in {keyword.casefold() for keyword in tokens.style_keywords}
    assert "walnut" in tokens.material_roles["floor"].casefold()
    assert "stone" in tokens.material_roles["walls"].casefold()
    assert any("statue" in focal.casefold() for focal in tokens.focal_hierarchy)
    assert any("book" in focal.casefold() for focal in tokens.focal_hierarchy)


def test_ordinary_prompt_keeps_default_footprint_and_large_prompt_respects_cap():
    ordinary = blueprint_from_prompt("a quiet library")
    capped = blueprint_from_prompt(
        "a large grand library with thousands of books",
        maximum_dimension_m=10.0,
    )

    assert ordinary.spaces[0].dimensions_m == (7.0, 7.0)
    assert capped.spaces[0].dimensions_m == (10.0, 10.0)


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


def test_one_staircase_can_serve_an_intermediate_level():
    levels = (
        LevelBlueprint(level_id="ground", name="Ground", elevation_m=0),
        LevelBlueprint(level_id="mezzanine", name="Mezzanine", elevation_m=4),
        LevelBlueprint(level_id="upper", name="Upper", elevation_m=8),
    )
    spaces = tuple(
        SpaceBlueprint(
            space_id=f"space-{level.level_id}",
            name=level.name,
            room_type="library",
            level_id=level.level_id,
            dimensions_m=(12, 12),
        )
        for level in levels
    )
    blueprint = SceneBlueprint(
        blueprint_id="three-level-library",
        source_prompt="One spiral staircase connects all three library floors.",
        levels=levels,
        spaces=spaces,
        connectors=(
            ConnectorBlueprint(
                connector_id="spiral-stair",
                kind="stairs_spiral",
                start=ConnectorEndpoint(
                    space_id="space-ground",
                    level_id="ground",
                    position_m=(6, 0, 6),
                ),
                end=ConnectorEndpoint(
                    space_id="space-upper",
                    level_id="upper",
                    position_m=(6, 8, 6),
                ),
                parameters={
                    "intermediate_landings": [
                        {
                            "space_id": "space-mezzanine",
                            "level_id": "mezzanine",
                            "position_m": [6, 4, 6],
                        }
                    ]
                },
            ),
        ),
    )

    assert len(blueprint.connectors) == 1


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
