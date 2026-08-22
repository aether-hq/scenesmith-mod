"""Provider-neutral floor-plan submission normalization and fallback tests."""

import asyncio
import json

from types import SimpleNamespace

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import WindowShape
from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
    LevelBlueprint,
    OpeningBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
    blueprint_from_prompt,
    floor_plan_submission_from_blueprint,
)
from scenesmith.agent_utils.structure.compiler.connector_dispatch import (
    compile_connector,
)
from scenesmith.agent_utils.structure.compiler.surfaces import compile_platform
from scenesmith.agent_utils.structure.geometry_models.surface_models import PlatformSpec
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorSpec,
)
from scenesmith.floor_plan_agents.stateful_floor_plan_agent import (
    StatefulFloorPlanAgent,
)
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.tools.submission.floor_plan_normalization import (
    normalize_floor_plan_submission,
)
from scenesmith.floor_plan_agents.tools.submission.floor_plan_submission import (
    opening_placements_from_blueprint,
)
from scenesmith.floor_plan_agents.tools.submission.structural_submission import (
    structural_submission_from_blueprint,
    synthesize_structural_layout,
)

PROMPT = (
    "A large, multi-level library filled with bookshelves, spiral staircases, "
    "and large research tables."
)


def _large_dock_blueprint() -> SceneBlueprint:
    level = LevelBlueprint(
        level_id="dock-level",
        name="Dock level",
        clear_height_m=30.0,
    )
    space = SpaceBlueprint(
        space_id="space-dock-hall",
        name="Space dock hall",
        room_type="space_dock",
        level_id=level.level_id,
        dimensions_m=(150.0, 100.0),
    )
    openings = (
        OpeningBlueprint(
            opening_id="opening-space",
            kind="open_connection",
            host_space_id=space.space_id,
            width_m=40.0,
            height_m=24.0,
        ),
        OpeningBlueprint(
            opening_id="opening-station",
            kind="door",
            host_space_id=space.space_id,
            width_m=20.0,
            height_m=18.0,
        ),
        *tuple(
            OpeningBlueprint(
                opening_id=f"opening-crew-{index}",
                kind="door",
                host_space_id=space.space_id,
                width_m=2.0,
                height_m=3.0,
            )
            for index in range(3)
        ),
    )
    return SceneBlueprint(
        blueprint_id="scene-large-dock",
        source_prompt="A huge space-fighter dock with ten repair bays.",
        levels=(level,),
        spaces=(space,),
        openings=openings,
        locked_ids=tuple(opening.opening_id for opening in openings),
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


def test_large_blueprint_openings_are_placed_without_metric_clamping():
    blueprint = _large_dock_blueprint()

    placements = opening_placements_from_blueprint(blueprint)

    assert len(placements) == 5
    by_id = {placement.opening_id: placement for placement in placements}
    assert by_id["opening-space"].width_m == 40.0
    assert by_id["opening-space"].height_m == 24.0
    assert by_id["opening-station"].width_m == 20.0
    assert all(placement.position_along_m > 0.0 for placement in placements)


def test_locked_blueprint_replaces_small_planner_layout_and_opening_defaults():
    blueprint = _large_dock_blueprint()
    layout = HouseLayout(house_prompt=blueprint.source_prompt)
    initial = FloorPlanTools(layout=layout, mode="room")._submit_floor_plan_impl(
        room_specs=[{"type": "hangar_bay", "width": 8.0, "depth": 6.0}],
        wall_height_meters=4.0,
    )
    assert initial.success, initial.message
    assert layout.room_specs[0].length == 8.0
    assert layout.doors[0].width == 0.9

    agent = StatefulFloorPlanAgent.__new__(StatefulFloorPlanAgent)
    agent.mode = "room"
    agent.blueprint = blueprint
    agent.layout = layout
    agent.cfg = SimpleNamespace(
        max_floor_plan_dim_m=20.0,
        min_floor_plan_dim_m=1.5,
        wall_height=SimpleNamespace(min=2.0, max=12.0),
    )

    agent._apply_locked_blueprint_topology()

    room = layout.room_specs[0]
    assert (room.length, room.width) == (150.0, 100.0)
    assert layout.wall_height == 30.0
    constructed = {
        opening.opening_id: opening
        for placed_room in layout.placed_rooms
        for wall in placed_room.walls
        for opening in wall.openings
    }
    assert set(constructed) == {
        "opening-space",
        "opening-station",
        "opening-crew-0",
        "opening-crew-1",
        "opening-crew-2",
    }
    assert constructed["opening-space"].width == 40.0
    assert constructed["opening-space"].height == 24.0
    assert constructed["opening-space"].opening_type.value == "open"
    assert constructed["opening-station"].width == 20.0
    assert len(layout.doors) == 4
    assert {door.id for door in layout.doors} == {
        "opening-station",
        "opening-crew-0",
        "opening-crew-1",
        "opening-crew-2",
    }
    assert layout.connectivity_valid


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


def test_exact_prompt_preserves_huge_arched_window_intent():
    prompt = (
        "a large, multi-level library with thousands of books and a bunch of tables "
        "and chairs for patrons. A spiral staircase connects the floors, and there "
        "are huge archted windows, statues, and so on, as it has a renaiissance , "
        "gorgeous decor."
    )

    submission = normalize_floor_plan_submission(
        {},
        prompt=prompt,
        mode="room",
        room_dim_min=1.5,
        room_dim_max=20.0,
        wall_height_min=2.0,
        wall_height_max=12.0,
    )

    assert submission.windows_per_room >= 3
    assert submission.window_shape == "arched"
    assert submission.window_width_m >= 3.5
    assert submission.window_height_m >= 3.0

    layout = HouseLayout(house_prompt=prompt)
    result = FloorPlanTools(layout=layout, mode="room")._submit_floor_plan_impl(
        **submission.tool_kwargs()
    )

    assert result.success, result.message
    assert len(layout.windows) >= 3
    assert all(window.shape == WindowShape.ARCHED for window in layout.windows)
    window_openings = [
        opening
        for room in layout.placed_rooms
        for wall in room.walls
        for opening in wall.openings
        if opening.opening_type.value == "window"
    ]
    assert len(window_openings) == len(layout.windows)
    assert all(opening.shape == WindowShape.ARCHED for opening in window_openings)


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


def test_accepted_blueprint_drives_constructed_levels_and_connector_contract():
    blueprint = blueprint_from_prompt(
        "A two-level library with a usable spiral staircase"
    )
    blueprint = blueprint.model_copy(
        update={
            "levels": (
                blueprint.levels[0],
                blueprint.levels[1].model_copy(
                    update={"elevation_m": 5.5, "clear_height_m": 5.0}
                ),
            )
        }
    )
    connector = blueprint.connectors[0].model_copy(update={"width_m": 3.2})
    blueprint = blueprint.model_copy(update={"connectors": (connector,)})
    payload = floor_plan_submission_from_blueprint(blueprint)

    structural = structural_submission_from_blueprint(
        blueprint,
        payload["room_specs"],
        max_total_height=12.0,
    )

    assert structural is not None
    assert [level["id"] for level in structural["levels"]] == [
        level.level_id for level in blueprint.levels
    ]
    assert len(structural["connectors"]) == 1
    authored_connector = structural["connectors"][0]
    assert authored_connector["id"] == connector.connector_id
    assert authored_connector["type"] == "stairs_spiral"
    assert authored_connector["width"] == 3.2
    assert authored_connector["parameters"]["turns"] == 1.0
    assert authored_connector["start"]["level_id"] == connector.start.level_id
    assert authored_connector["end"]["level_id"] == connector.end.level_id

    payload["structural"] = structural
    submission = normalize(payload)
    layout = HouseLayout(house_prompt=blueprint.source_prompt)
    result = FloorPlanTools(layout=layout, mode="room")._submit_floor_plan_impl(
        **submission.tool_kwargs()
    )

    assert result.success, result.message
    assert [level.level_id for level in layout.levels] == [
        level.level_id for level in blueprint.levels
    ]
    assert layout.connectors[0].connector_id == connector.connector_id
    assert layout.connectors[0].width == 3.2


def test_multistop_semantic_staircase_compiles_to_adjacent_structural_spans():
    blueprint = blueprint_from_prompt(
        "A three-story library with one usable spiral staircase"
    )
    first, second = blueprint.connectors
    connector = first.model_copy(
        update={
            "end": second.end,
            "width_m": 3.0,
            "parameters": {
                "intermediate_landings": [first.end.model_dump(mode="json")]
            },
        }
    )
    blueprint = SceneBlueprint.model_validate(
        blueprint.model_copy(update={"connectors": (connector,)}).model_dump()
    )
    payload = floor_plan_submission_from_blueprint(blueprint)

    structural = structural_submission_from_blueprint(
        blueprint,
        payload["room_specs"],
        max_total_height=12.0,
    )

    assert structural is not None
    assert [item["id"] for item in structural["connectors"]] == [
        connector.connector_id,
        f"{connector.connector_id}-segment-2",
    ]
    assert [
        (item["start"]["level_id"], item["end"]["level_id"])
        for item in structural["connectors"]
    ] == [
        (blueprint.levels[0].level_id, blueprint.levels[1].level_id),
        (blueprint.levels[1].level_id, blueprint.levels[2].level_id),
    ]
    assert all(item["width"] == 3.0 for item in structural["connectors"])
    assert all(item["parameters"]["turns"] == 1.0 for item in structural["connectors"])


def test_large_multilevel_library_synthesizes_gallery_atrium():
    structural = synthesize_structural_layout(
        PROMPT,
        [{"type": "library", "width": 13.8, "depth": 13.8, "prompt": PROMPT}],
        4.0,
        max_total_height=12.0,
    )

    assert structural is not None
    slabs = [
        platform
        for platform in structural["platforms"]
        if platform["id"].endswith("walkable_slab")
    ]
    assert slabs
    for slab in slabs:
        holes = slab["footprint"]["holes"]
        assert len(holes) >= 2
        assert slab["guarded_hole_indices"] == [1]
        gallery_hole = max(
            holes,
            key=lambda hole: (
                max(point[0] for point in hole) - min(point[0] for point in hole)
            )
            * (max(point[1] for point in hole) - min(point[1] for point in hole)),
        )
        assert (
            max(point[0] for point in gallery_hole)
            - min(point[0] for point in gallery_hole)
            >= 5.0
        )
        assert (
            max(point[1] for point in gallery_hole)
            - min(point[1] for point in gallery_hole)
            >= 5.0
        )
        compiled = compile_platform(PlatformSpec.from_dict(slab))
        assert compiled.visual_mesh.vertices
        assert compiled.visual_mesh.bounds[1][2] > slab["elevation"] + 1.0


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
