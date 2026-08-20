"""Provider-neutral floor-plan submission normalization and fallback tests."""

import asyncio
import json

from scenesmith.agent_utils.house import HouseLayout
from scenesmith.agent_utils.structural_compiler import (
    compile_connector,
    compile_platform,
)
from scenesmith.agent_utils.structural_geometry import ConnectorSpec, PlatformSpec
from scenesmith.floor_plan_agents.tools.floor_plan_submission import (
    normalize_floor_plan_submission,
    synthesize_structural_layout,
)
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools


PROMPT = (
    "A large, multi-level library filled with bookshelves, spiral staircases, "
    "and large research tables."
)


def normalize(payload):
    return normalize_floor_plan_submission(
        payload,
        prompt=PROMPT,
        mode="room",
        room_dim_min=1.5,
        room_dim_max=20.0,
        wall_height_min=2.0,
        wall_height_max=12.0,
    )


def test_open_air_cover_flag_survives_provider_neutral_normalization():
    submission = normalize(
        {
            "room_specs": [
                {
                    "type": "courtyard",
                    "width": 8,
                    "depth": 6,
                    "covered": False,
                }
            ]
        }
    )

    assert submission.room_specs[0]["has_overhead_cover"] is False


def test_haiku_numeric_level_ids_survive_normalization_and_execution():
    payload = {
        "room_specs": [
            {
                "type": "library",
                "width": 15,
                "depth": 20,
                "prompt": PROMPT,
            }
        ],
        "wall_height_meters": 4,
        "structural": {
            "levels": [
                {"level_id": 0, "elevation": 0},
                {"level_id": 1, "elevation": 4},
                {"level_id": 2, "elevation": 8},
            ],
            "rooms": [
                {"space_id": "library", "level_id": 0},
                {"space_id": "library", "level_id": 1},
                {"space_id": "library", "level_id": 2},
            ],
            "connectors": [
                {
                    "type": "stairs_spiral",
                    "start": {
                        "space_id": "library",
                        "level_id": 0,
                        "position": [3, 3, 0],
                    },
                    "end": {
                        "space_id": "library",
                        "level_id": 1,
                        "position": [3, 3, 4],
                    },
                    "parameters": {
                        "center": [3, 3],
                        "radius": 2,
                        "turns": 1,
                        "direction": "cw",
                        "riser_count": 15,
                    },
                }
            ],
        },
    }

    submission = normalize(payload)
    layout = HouseLayout(house_prompt=PROMPT)
    result = FloorPlanTools(layout=layout, mode="room")._submit_floor_plan_impl(
        **submission.tool_kwargs()
    )

    assert result.success, result.message
    assert [level.level_id for level in layout.levels] == ["0", "1", "2"]
    assert layout.room_specs[0].level_id == "0"
    assert layout.wall_height == 10.5


def test_openai_camel_case_design_envelope_and_room_map_are_normalized():
    submission = normalize(
        {
            "design": {
                "rooms": {
                    "grandLibrary": {
                        "dimensions": "15m x 20m",
                        "description": PROMPT,
                    }
                },
                "wallHeight": "4 meters",
                "windowCount": "3",
                "materials": {
                    "floor": "polished oak",
                    "walls": "warm plaster and wood paneling",
                    "facade": "brick",
                },
            }
        }
    )

    assert submission.room_specs == [
        {
            "type": "grand_library",
            "width": 15.0,
            "depth": 20.0,
            "prompt": PROMPT,
        }
    ]
    assert submission.wall_height_meters == 4.0
    assert submission.windows_per_room == 3
    assert submission.floor_material_description == "polished oak"
    assert submission.structural is not None
    assert "unwrapped design envelope" in submission.repairs


def test_qwen_jsonish_plan_and_aliases_are_normalized_without_model_repair():
    submission = normalize(
        {
            "plan": (
                "{'spaces': [{'name': 'Library Hall', 'size': [14, 18], "
                "'details': 'bookshelves and study tables'}], "
                "'storeyHeight': '3.8m', 'windows': '2'}"
            )
        }
    )

    assert submission.room_specs[0]["type"] == "library_hall"
    assert submission.room_specs[0]["width"] == 14.0
    assert submission.room_specs[0]["depth"] == 18.0
    assert submission.wall_height_meters == 3.8
    assert submission.windows_per_room == 2


def test_missing_tool_fields_synthesize_safe_room_and_multilevel_structure():
    submission = normalize({"unexpected": "shape"})

    assert submission.room_specs[0]["type"] == "library"
    assert submission.structural is not None
    assert len(submission.structural["levels"]) == 3
    assert submission.structural["connectors"][0]["type"] == "stairs_spiral"
    assert len(submission.structural["platforms"]) == 4
    assert any("synthesized missing room" in repair for repair in submission.repairs)


def test_multilevel_synthesis_fits_story_heights_and_emits_walkable_slabs():
    structural = synthesize_structural_layout(
        PROMPT,
        [{"type": "library", "width": 15, "depth": 20, "prompt": PROMPT}],
        4.5,
        max_total_height=12.0,
    )

    assert structural is not None
    assert len(structural["levels"]) == 3
    assert (
        max(
            level["elevation"] + level["nominal_height"]
            for level in structural["levels"]
        )
        == 12.0
    )
    assert len(structural["platforms"]) == 4
    assert all(platform["traversable"] for platform in structural["platforms"])
    slabs = [
        platform
        for platform in structural["platforms"]
        if platform["id"].endswith("walkable_slab")
    ]
    assert all(platform["footprint"]["holes"] for platform in slabs)
    slab_min_x = min(point[0] for point in slabs[0]["footprint"]["outer"])
    slab_max_x = max(point[0] for point in slabs[0]["footprint"]["outer"])
    slab_min_y = min(point[1] for point in slabs[0]["footprint"]["outer"])
    slab_max_y = max(point[1] for point in slabs[0]["footprint"]["outer"])
    assert abs(slab_min_x + slab_max_x) < 1e-9
    assert abs(slab_min_y + slab_max_y) < 1e-9
    assert (
        structural["connectors"][0]["end"]["position"][:2]
        == structural["connectors"][1]["start"]["position"][:2]
    )
    assert structural["connectors"][0]["end"]["level_id"] == "level_1"
    assert structural["connectors"][1]["end"]["level_id"] == "level_2"
    hole = slabs[0]["footprint"]["holes"][0]
    hole_min_x = min(point[0] for point in hole)
    hole_max_x = max(point[0] for point in hole)
    hole_min_y = min(point[1] for point in hole)
    hole_max_y = max(point[1] for point in hole)
    first_landing = next(
        platform
        for platform in structural["platforms"]
        if platform["id"] == "level_1_stair_landing"
    )
    landing_points = first_landing["footprint"]["outer"]
    landing_min_x = min(point[0] for point in landing_points)
    landing_max_x = max(point[0] for point in landing_points)
    landing_min_y = min(point[1] for point in landing_points)
    landing_max_y = max(point[1] for point in landing_points)
    supported_overlap = max(
        hole_min_x - landing_min_x,
        landing_max_x - hole_max_x,
        hole_min_y - landing_min_y,
        landing_max_y - hole_max_y,
    )
    assert supported_overlap >= 0.7
    assert "Reduced each storey" in structural["_diagnostics"][0]


def test_multilevel_synthesis_supports_each_structural_stair_family():
    cases = {
        "with a spiral staircase": "stairs_spiral",
        "with a normal straight staircase": "stairs_straight",
        "with an L-shaped staircase": "stairs_l",
        "with a U-shaped switchback staircase": "stairs_u",
    }

    for request, expected_type in cases.items():
        prompt = f"A large two-story library {request}."
        structural = synthesize_structural_layout(
            prompt,
            [{"type": "library", "width": 18, "depth": 14, "prompt": prompt}],
            3.6,
        )

        assert structural is not None
        connector = ConnectorSpec.from_dict(structural["connectors"][0])
        assert connector.connector_type.value == expected_type
        assert compile_connector(connector).visual_mesh.vertices

        platforms = [PlatformSpec.from_dict(item) for item in structural["platforms"]]
        assert platforms[0].footprint.holes
        assert all(
            compile_platform(platform).visual_mesh.vertices for platform in platforms
        )


def test_unsafe_stair_family_degradation_is_explicit():
    prompt = "A narrow two-story room with a normal straight staircase."
    structural = synthesize_structural_layout(
        prompt,
        [{"type": "room", "width": 3.2, "depth": 3.2, "prompt": prompt}],
        3.0,
    )

    assert structural is not None
    assert structural["connectors"][0]["type"] == "stairs_spiral"
    assert any(
        "Changed stairs_straight to stairs_spiral" in item
        for item in structural["_diagnostics"]
    )


def test_captured_multilevel_library_uses_structure_instead_of_flat_fallback():
    submission = normalize(
        {
            "room_specs": [
                {
                    "type": "library",
                    "width": 15,
                    "depth": 20,
                    "prompt": PROMPT,
                }
            ],
            "wall_height_meters": 4.5,
        }
    )
    layout = HouseLayout(house_prompt=PROMPT)

    result = FloorPlanTools(
        layout=layout, mode="room"
    )._submit_floor_plan_with_fallback(submission)

    assert result.success, result.message
    assert "safe flat structural fallback" not in result.message
    assert layout.wall_height == 12.0
    assert len(layout.levels) == 3
    assert len(layout.platforms) == 4
    assert len(layout.connectors) == 2
    assert layout.connectors[0].start.level_id == "level_0"
    assert layout.connectors[0].end.level_id == "level_1"
    assert layout.connectors[1].start.level_id == "level_1"
    assert layout.connectors[1].end.level_id == "level_2"


def test_raw_function_tool_accepts_plan_envelope_instead_of_exact_signature():
    layout = HouseLayout(house_prompt="A bright reading room")
    tools = FloorPlanTools(layout=layout, mode="room")

    result = asyncio.run(
        tools.submit_floor_plan_tool.on_invoke_tool(
            None,
            json.dumps(
                {
                    "plan": {
                        "spaces": {
                            "reading room": {
                                "size": {"x": "8m", "y": "6m"},
                                "description": "A bright reading room",
                            }
                        },
                        "windows": 2,
                    }
                }
            ),
        )
    )

    assert result.success, result.message
    assert layout.room_specs[0].room_id == "reading_room"
    assert len(layout.windows) == 2


def test_invalid_provider_structure_degrades_to_deterministic_structure():
    submission = normalize(
        {
            "rooms": [{"type": "library", "size": [15, 20]}],
            "structural": {"connectors": [{"type": "teleporter"}]},
        }
    )
    layout = HouseLayout(house_prompt=PROMPT)
    tools = FloorPlanTools(layout=layout, mode="room")

    result = tools._submit_floor_plan_with_fallback(submission)

    assert result.success, result.message
    assert layout.connectors
    assert "deterministic multi-level fallback" in result.message


def test_checkpoint_geometry_failure_can_degrade_to_safe_flat_layout():
    checkpoint_calls = 0

    def checkpoint():
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return checkpoint_calls >= 2

    submission = normalize(
        {
            "rooms": [{"type": "library", "size": [15, 20]}],
            "structural": {"connectors": [{"type": "teleporter"}]},
        }
    )
    layout = HouseLayout(house_prompt=PROMPT)
    tools = FloorPlanTools(
        layout=layout,
        mode="room",
        checkpoint_callback=checkpoint,
    )

    result = tools._submit_floor_plan_with_fallback(submission)

    assert result.success, result.message
    assert checkpoint_calls >= 2
    assert not layout.connectors
    assert "safe flat structural fallback" in result.message


def test_nested_scalar_collections_and_numeric_room_type_never_escape_tool():
    layout = HouseLayout(house_prompt=PROMPT)
    tools = FloorPlanTools(layout=layout, mode="room")

    result = asyncio.run(
        tools.submit_floor_plan_tool.on_invoke_tool(
            None,
            json.dumps(
                {
                    "room_specs": [
                        {"type": 0, "dimensions": [12, 10], "prompt": PROMPT}
                    ],
                    "structural": {
                        "levels": 3,
                        "rooms": 1,
                        "connectors": 2,
                    },
                }
            ),
        )
    )

    assert result.success, result.message
    assert layout.room_specs[0].room_id == "library"
    assert layout.connectors
    assert "deterministic multi-level fallback" in result.message
