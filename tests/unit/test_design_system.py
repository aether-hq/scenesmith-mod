"""Tests for editable design systems and compiled StyleBibles."""

import json

import pytest

from scenesmith.agent_utils.design_system import (
    BUILTIN_DESIGN_SYSTEMS,
    DesignSystem,
    apply_style_bible,
    compile_style_bible,
    load_design_system,
    persist_design_contract,
)


def test_three_visually_distinct_builtin_systems_compile():
    assert set(BUILTIN_DESIGN_SYSTEMS) == {
        "warm-modern",
        "jewel-maximalist",
        "calm-natural",
    }
    compiled = {
        name: compile_style_bible(system)
        for name, system in BUILTIN_DESIGN_SYSTEMS.items()
    }
    assert len({bible.palette_roles["accent"] for bible in compiled.values()}) == 3
    assert len({bible.ceiling_direction for bible in compiled.values()}) == 3
    assert len({bible.detail_direction for bible in compiled.values()}) == 3


def test_custom_json_design_system_round_trips(tmp_path):
    system = BUILTIN_DESIGN_SYSTEMS["jewel-maximalist"].model_copy(
        update={"design_system_id": "custom-stage", "name": "Custom stage"}
    )
    source = tmp_path / "custom.json"
    source.write_text(system.model_dump_json(indent=2))

    loaded = load_design_system(source)

    assert loaded == system
    assert DesignSystem.model_validate_json(loaded.model_dump_json()) == system


def test_custom_yaml_uses_same_strict_schema(tmp_path):
    source = tmp_path / "custom.yaml"
    source.write_text(
        """
schema_version: 1
design_system_id: graphic-lab
name: Graphic lab
palette: ['#111111', '#eeeeee', '#ff3300']
material_roles: {floor: black rubber, walls: white paint, accent: orange metal}
lighting: {mood: crisp, color_temperature_k: 4200, contrast: dramatic, practical_density: sparse}
shape_vocabulary: [orthogonal frames, circles]
style_keywords: [graphic, industrial]
era: contemporary
contrast: 0.9
saturation: 0.65
set_dressing: {density: sparse, motifs: [orange circles], forbidden_motifs: [rustic decor]}
"""
    )

    loaded = load_design_system(source)

    assert loaded.design_system_id == "graphic-lab"
    assert compile_style_bible(loaded).palette_roles["accent"] == "#111111"


def test_invalid_palette_is_rejected():
    payload = BUILTIN_DESIGN_SYSTEMS["warm-modern"].model_dump(mode="json")
    payload["palette"] = ["#ffffff"]

    with pytest.raises(ValueError, match="2-12"):
        DesignSystem.model_validate(payload)


def test_prompt_application_is_idempotent():
    bible = compile_style_bible(BUILTIN_DESIGN_SYSTEMS["calm-natural"])

    once = apply_style_bible("A reading room", bible)
    twice = apply_style_bible(once, bible)

    assert once == twice
    assert "forbidden_motifs" in once


def test_persisted_contract_contains_editable_and_compiled_forms(tmp_path):
    system = BUILTIN_DESIGN_SYSTEMS["warm-modern"]
    bible = compile_style_bible(system)

    persist_design_contract(system, bible, tmp_path)

    assert (
        json.loads((tmp_path / "design_system.json").read_text())["name"] == system.name
    )
    assert (
        json.loads((tmp_path / "style_bible.json").read_text())["design_system_id"]
        == system.design_system_id
    )
