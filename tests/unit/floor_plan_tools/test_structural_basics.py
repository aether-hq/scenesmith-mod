"""Tests for floor plan tools - door/window preservation after room changes."""

import asyncio
import json
import math
import unittest

from unittest.mock import MagicMock

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.structure.compiler.connector_primitives import (
    compile_spiral_stairs,
)
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools


class TestStructuralLayoutAuthoring(unittest.TestCase):
    def _create_two_room_layout(self) -> tuple[HouseLayout, FloorPlanTools]:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")
        result = tools._generate_room_specs_impl(
            json.dumps(
                [
                    {
                        "type": "lower",
                        "prompt": "Lower hall",
                        "width": 4,
                        "depth": 5,
                    },
                    {
                        "type": "upper",
                        "prompt": "Upper gallery",
                        "width": 4,
                        "depth": 5,
                    },
                ]
            )
        )
        assert result.success
        return layout, tools

    def test_public_resize_tool_forwards_room_id(self) -> None:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="room")
        created = tools._generate_room_specs_impl(
            '[{"type":"studio","width":6.0,"depth":5.0}]'
        )
        self.assertTrue(created.success)

        result = asyncio.run(
            tools.tools["resize_room"].on_invoke_tool(
                None,
                '{"room_id":"studio","width":5.0,"depth":4.0}',
            )
        )

        self.assertIn("resized", str(result))
        # RoomSpec uses length for the X/"width" dimension and width for the
        # Y/"depth" dimension; the public tool intentionally hides that legacy
        # internal naming.
        self.assertEqual(layout.get_room_spec("studio").length, 5.0)
        self.assertEqual(layout.get_room_spec("studio").width, 4.0)

    def test_public_structural_tool_accepts_object_without_double_encoding(
        self,
    ) -> None:
        layout, tools = self._create_two_room_layout()

        result = asyncio.run(
            tools.tools["set_structural_layout"].on_invoke_tool(
                None,
                json.dumps(
                    {
                        "structural_json": {
                            "levels": [
                                {
                                    "id": "ground",
                                    "elevation": 0,
                                    "nominal_height": 3,
                                },
                                {
                                    "id": "upper_level",
                                    "elevation": 3,
                                    "nominal_height": 3,
                                },
                            ],
                            "rooms": [{"id": "upper", "level_id": "upper_level"}],
                        }
                    }
                ),
            )
        )

        self.assertIn("Applied structural layout", str(result))
        self.assertEqual(layout.get_room_elevation("upper"), 3.0)

    def test_double_height_room_accepts_multilevel_wall_height(self) -> None:
        layout, tools = self._create_two_room_layout()

        result = tools._set_wall_height_impl(7.5)

        self.assertTrue(result.success, result.message)
        self.assertEqual(layout.wall_height, 7.5)
        too_tall = tools._set_wall_height_impl(12.1)
        self.assertFalse(too_tall.success)

    def test_validating_and_repairing_layout_triggers_durable_checkpoint(self) -> None:
        layout = HouseLayout()
        checkpoint = MagicMock(return_value=True)
        tools = FloorPlanTools(
            layout=layout,
            mode="house",
            checkpoint_callback=checkpoint,
        )
        created = tools._generate_room_specs_impl(
            json.dumps(
                [
                    {
                        "type": "library",
                        "prompt": "A library",
                        "width": 8,
                        "depth": 6,
                    }
                ]
            )
        )
        self.assertTrue(created.success)
        authored = {
            "portals": [
                {
                    "id": "entry",
                    "type": "door",
                    "source_space_id": "library",
                }
            ]
        }
        self.assertTrue(tools._set_structural_layout_impl(authored).success)
        checkpoint.assert_not_called()

        validation = asyncio.run(tools.tools["validate"].on_invoke_tool(None, "{}"))
        self.assertIn("connectivity='ok'", str(validation))
        checkpoint.assert_called_once_with()

        checkpoint.reset_mock()
        self.assertTrue(tools._set_structural_layout_impl(authored).success)
        checkpoint.assert_called_once_with()

    def test_structural_tool_repairs_common_llm_aliases(self) -> None:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")
        created = tools._generate_room_specs_impl(
            json.dumps(
                [
                    {
                        "type": "library",
                        "prompt": "A grand two-level library",
                        "width": 12,
                        "depth": 10,
                    }
                ]
            )
        )
        self.assertTrue(created.success)
        authored = {
            "levels": [
                {"id": "ground", "elevation": 0, "name": "Reading floor"},
                {"id": "mezzanine", "elevation": 3.5, "name": "Stacks"},
            ],
            "rooms": [
                {
                    "id": "library",
                    "space_id": "library",
                    "level_id": "ground",
                    "position": [0, 0],
                }
            ],
            "platforms": [
                {
                    "id": "mezzanine_platform",
                    "room_id": "library",
                    "level_id": "mezzanine",
                    "elevation": 3.5,
                    "thickness": 0.2,
                    "footprint": {"polygon": [[0, 5.5], [12, 5.5], [12, 10], [0, 10]]},
                }
            ],
            "connectors": [
                {
                    "id": "spiral_stair",
                    "type": "stairs_spiral",
                    "room_id": "library",
                    "from_level": "ground",
                    "to_level": "mezzanine",
                    "center": [3, 5],
                    "turns": 1.5,
                    "direction": "cw",
                    "riser_count": 20,
                    "parameters": {"radius": 1.0},
                }
            ],
            "portals": [
                {
                    "id": "entry",
                    "type": "door",
                    "room_id": "library",
                    "width": 1.2,
                    "height": 2.2,
                }
            ],
        }

        result = tools._set_structural_layout_impl(authored)

        self.assertTrue(result.success, result.message)
        self.assertEqual(authored["platforms"][0]["room_id"], "library")
        self.assertEqual(layout.platforms[0].space_id, "library")
        self.assertEqual(layout.platforms[0].footprint.area, 54.0)
        connector = layout.connectors[0]
        self.assertEqual(connector.start.space_id, "library")
        self.assertEqual(connector.start.level_id, "ground")
        self.assertEqual(connector.start.position[2], 0.0)
        self.assertEqual(connector.end.position[2], 3.5)
        self.assertAlmostEqual(
            math.dist(connector.start.position[:2], connector.parameters["center"]),
            1.0,
        )
        self.assertGreater(
            len(compile_spiral_stairs(connector).visual_mesh.triangles), 0
        )
        self.assertEqual(layout.portals[0].source_space_id, "library")

    def test_structural_tool_repairs_haiku_spiral_direction_and_radius(self) -> None:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="room")
        created = tools._generate_room_specs_impl(
            '[{"type":"library","width":14,"depth":16}]'
        )
        self.assertTrue(created.success)

        result = tools._set_structural_layout_impl(
            {
                "levels": [
                    {
                        "level_id": "ground",
                        "elevation": 0,
                        "nominal_height": 4,
                        "rooms": [{"space_id": "library"}],
                    },
                    {
                        "level_id": "upper",
                        "elevation": 4,
                        "nominal_height": 3,
                        "rooms": [{"space_id": "library"}],
                    },
                ],
                "connectors": [
                    {
                        "name": "main_spiral_stairs",
                        "type": "stairs_spiral",
                        "start": {
                            "space_id": "library",
                            "level_id": "ground",
                            "position": [7, 8, 0],
                        },
                        "end": {
                            "space_id": "library",
                            "level_id": "upper",
                            "position": [7, 8, 4],
                        },
                        "width": 1.5,
                        "clearance": 2.2,
                        "parameters": {
                            "center": [7, 8],
                            "turns": 1.5,
                            "direction": "clockwise",
                            "riser_count": 28,
                        },
                    }
                ],
            }
        )

        self.assertTrue(result.success, result.message)
        connector = layout.connectors[0]
        self.assertEqual(connector.connector_id, "main_spiral_stairs")
        self.assertEqual(layout.get_room_spec("library").level_id, "ground")
        self.assertEqual(connector.parameters["direction"], "cw")
        self.assertEqual(connector.parameters["radius"], 1.5)
        self.assertGreater(
            len(compile_spiral_stairs(connector).visual_mesh.triangles), 0
        )

    def test_one_shot_submission_finishes_and_checkpoints_locally(self) -> None:
        layout = HouseLayout(house_prompt="A bright reading room")
        checkpoint = MagicMock(return_value=True)
        tools = FloorPlanTools(
            layout=layout,
            mode="room",
            checkpoint_callback=checkpoint,
        )

        result = tools._submit_floor_plan_impl(
            room_specs=[{"type": "reading_room", "width": 8, "depth": 6}],
            wall_height_meters=3.5,
            windows_per_room=2,
            floor_material_description="warm oak floor",
            wall_material_description="soft white plaster wall",
            exterior_material_description="neutral plaster exterior",
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(layout.wall_height, 3.5)
        self.assertEqual(len(layout.doors), 1)
        self.assertIsNone(layout.doors[0].room_b)
        self.assertEqual(len(layout.windows), 2)
        self.assertIn("reading_room", layout.room_materials)
        self.assertTrue(layout.connectivity_valid)
        checkpoint.assert_called_once_with()

    def test_authored_levels_without_room_override_use_lowest_level(self) -> None:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="room")
        self.assertTrue(
            tools._generate_room_specs_impl(
                '[{"type":"library","width":18,"depth":16}]'
            ).success
        )

        result = tools._set_structural_layout_impl(
            {
                "levels": [
                    {"level_id": 0, "elevation": 0, "height": 3.5},
                    {"level_id": 1, "elevation": 3.5, "height": 3.5},
                ],
                "rooms": [
                    {"space_id": "library", "level_id": 0},
                    {"space_id": "library", "level_id": 1},
                ],
                "connectors": [
                    {
                        "type": "stairs_spiral",
                        "start": {
                            "space_id": "library",
                            "level_id": 0,
                            "position": [2, 2, 0],
                        },
                        "end": {
                            "space_id": "library",
                            "level_id": 1,
                            "position": [2, 2, 3.5],
                        },
                        "parameters": {
                            "center": [2, 2],
                            "radius": 1.5,
                            "turns": 1.5,
                            "direction": "cw",
                            "riser_count": 21,
                        },
                    }
                ],
            }
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(layout.get_room_spec("library").level_id, "0")
        self.assertEqual([level.level_id for level in layout.levels], ["0", "1"])
        self.assertEqual(layout.levels[1].nominal_height, 3.5)
        self.assertEqual(layout.connectors[0].connector_id, "connector_1")

        submitted_layout = HouseLayout(house_prompt="A two-level library")
        submitted = FloorPlanTools(
            layout=submitted_layout, mode="room"
        )._submit_floor_plan_impl(
            room_specs=[{"type": "library", "width": 18, "depth": 16}],
            wall_height_meters=3.5,
            structural={
                "levels": [
                    {"id": "level_0", "elevation": 0, "height": 3.5},
                    {"id": "level_1", "elevation": 3.5, "height": 3.5},
                ],
                "connectors": [
                    {
                        "type": "stairs_spiral",
                        "start": {
                            "space_id": "library",
                            "level_id": "level_0",
                            "position": [2, 2, 0],
                        },
                        "end": {
                            "space_id": "library",
                            "level_id": "level_1",
                            "position": [2, 2, 3.5],
                        },
                        "parameters": {
                            "center": [2, 2],
                            "radius": 1.5,
                            "turns": 1.5,
                            "direction": "cw",
                            "riser_count": 21,
                        },
                    }
                ],
            },
            windows_per_room=0,
        )
        self.assertTrue(submitted.success, submitted.message)
        self.assertEqual(submitted_layout.wall_height, 7.0)
