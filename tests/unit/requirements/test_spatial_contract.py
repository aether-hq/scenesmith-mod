"""Tests for generic semantic fulfillment capability preflight."""

from scenesmith.agent_utils.semantics.requirements.compilation.expansion import (
    _project_endpoint,
)
from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    SpatialRequirementCompilationWire,
)
from scenesmith.agent_utils.semantics.requirements.requirement_blueprint_compiler import (
    SPATIAL_COMPILER_INSTRUCTIONS,
    _validate_expected_mode_spaces,
    spatial_compilation_output_schema,
)
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    ConnectorEndpoint,
    blueprint_from_prompt,
)


def test_spatial_compiler_schema_allows_typed_blueprint_maps():
    output_schema = spatial_compilation_output_schema()

    assert output_schema.output_type is SpatialRequirementCompilationWire
    assert output_schema.is_strict_json_schema()


def test_spatial_compiler_contract_is_bounded_and_allows_one_space_per_level():
    instructions = " ".join(SPATIAL_COMPILER_INSTRUCTIONS.split())
    blueprint = blueprint_from_prompt(
        "A multi-level library with one spiral staircase."
    )

    assert "at most 18 words" in instructions
    assert len(blueprint.levels) == len(blueprint.spaces) == 2
    _validate_expected_mode_spaces(blueprint, "room")


def test_spatial_connector_projection_clamps_xy_and_uses_level_elevation_for_z():
    endpoint = ConnectorEndpoint(
        space_id="library",
        level_id="mezzanine",
        position_m=(30.0, -30.0, 999.0),
    )

    projected = _project_endpoint(
        endpoint,
        level_elevations={"mezzanine": 4.0},
        maximum_dimension_m=20.0,
    )

    assert projected.position_m == (10.0, -10.0, 4.0)
