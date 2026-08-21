"""Focused ceiling-tool regressions."""

from types import SimpleNamespace
from unittest.mock import Mock

from scenesmith.agent_utils.asset_manager import AssetGenerationRequest
from scenesmith.agent_utils.room import ObjectType
from scenesmith.ceiling_agents.tools.ceiling_tools import (
    CeilingTools,
    _grand_atrium_chandelier_dimensions,
)


def test_grand_atrium_chandelier_keeps_focal_width_and_drop() -> None:
    dimensions = _grand_atrium_chandelier_dimensions(
        description=(
            "Grand crystal chandelier with multiple tiers, ornate brass arms, "
            "and hanging crystal prisms for a renaissance-style library"
        ),
        dimensions=[1.4, 1.4, 1.0],
        room_bounds=(-6.6, -6.6, 6.6, 6.6),
        ceiling_height=12.0,
    )

    assert dimensions == [1.4, 1.4, 1.8]


def test_chandelier_focal_minimum_is_limited_to_large_tall_atriums() -> None:
    ordinary = _grand_atrium_chandelier_dimensions(
        description="Grand chandelier",
        dimensions=[0.8, 0.8, 0.9],
        room_bounds=(-3.5, -3.5, 3.5, 3.5),
        ceiling_height=3.8,
    )
    pendant = _grand_atrium_chandelier_dimensions(
        description="Small brass pendant light",
        dimensions=[0.4, 0.4, 0.6],
        room_bounds=(-6.6, -6.6, 6.6, 6.6),
        ceiling_height=12.0,
    )

    assert ordinary == [0.8, 0.8, 0.9]
    assert pendant == [0.4, 0.4, 0.6]


def test_ceiling_generation_normalizes_grand_atrium_chandelier_request() -> None:
    asset_manager = Mock()
    asset_manager.generate_assets.return_value = SimpleNamespace(
        successful_assets=[],
        has_failures=False,
    )
    tools = CeilingTools.__new__(CeilingTools)
    tools.asset_manager = asset_manager
    tools.room_bounds = (-6.6, -6.6, 6.6, 6.6)
    tools.ceiling_height = 12.0
    original = AssetGenerationRequest(
        object_descriptions=["Grand crystal chandelier"],
        short_names=["grand_chandelier"],
        object_type=ObjectType.CEILING_MOUNTED,
        desired_dimensions=[[1.4, 1.4, 1.0]],
    )

    tools._generate_assets_impl(original)

    generated_request = asset_manager.generate_assets.call_args.kwargs["request"]
    assert generated_request.desired_dimensions == [[1.4, 1.4, 1.8]]
    assert original.desired_dimensions == [[1.4, 1.4, 1.0]]
