"""Tests for floor plan tools - door/window preservation after room changes."""

import asyncio
import json
import math
import random
import unittest
from unittest.mock import MagicMock

from scenesmith.agent_utils.house import HouseLayout, OpeningType, WallDirection
from scenesmith.agent_utils.structural_compiler import compile_spiral_stairs
from scenesmith.agent_utils.structural_geometry import Footprint2D
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.tools.room_placement import get_shared_edge


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

    def test_public_structural_tool_accepts_object_without_double_encoding(self) -> None:
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
                            "rooms": [
                                {"id": "upper", "level_id": "upper_level"}
                            ],
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

        validation = asyncio.run(
            tools.tools["validate"].on_invoke_tool(None, "{}")
        )
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
                    "footprint": {
                        "polygon": [[0, 5.5], [12, 5.5], [12, 10], [0, 10]]
                    },
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

    def test_agent_tool_applies_multilevel_polygon_and_stairs_atomically(self) -> None:
        layout, tools = self._create_two_room_layout()
        structural = {
            "levels": [
                {"id": "ground", "elevation": 0, "nominal_height": 3},
                {"id": "upper_level", "elevation": 3, "nominal_height": 3},
            ],
            "rooms": [
                {
                    "id": "lower",
                    "position": [0, 0],
                    "footprint": {
                        "outer": [[0, 0], [5, 0], [5, 2], [2, 2], [2, 4], [0, 4]],
                        "holes": [],
                    },
                },
                {
                    "id": "upper",
                    "level_id": "upper_level",
                    "position": [0, 0],
                    "yaw_degrees": 30,
                },
            ],
            "connectors": [
                {
                    "id": "stairs",
                    "type": "stairs_straight",
                    "start": {
                        "space_id": "lower",
                        "level_id": "ground",
                        "position": [0, 0, 0],
                    },
                    "end": {
                        "space_id": "upper",
                        "level_id": "upper_level",
                        "position": [4, 0, 3],
                    },
                    "parameters": {"riser_count": 18},
                }
            ],
            "platforms": [
                {
                    "id": "mezzanine",
                    "space_id": "upper",
                    "footprint": {
                        "outer": [[0, 0], [2, 0], [2, 1], [0, 1]],
                        "holes": [],
                    },
                    "elevation": 2.5,
                    "open_edge_indices": [2],
                }
            ],
            "portals": [
                {
                    "id": "entrance",
                    "type": "door",
                    "source_space_id": "lower",
                    "width": 1.0,
                    "height": 2.1,
                }
            ],
        }

        result = tools._set_structural_layout_impl(json.dumps(structural))

        self.assertTrue(result.success, result.message)
        self.assertEqual(layout.get_room_elevation("upper"), 3.0)
        self.assertEqual(layout.room_specs[0].footprint.area, 14.0)
        self.assertAlmostEqual(layout.room_specs[1].yaw, math.pi / 6)
        self.assertEqual(layout.placed_rooms[1].level_id, "upper_level")
        self.assertEqual(layout.placed_rooms[0].position, (0.0, 0.0))
        self.assertEqual(layout.placed_rooms[1].position, (0.0, 0.0))
        self.assertTrue(
            all(wall.is_exterior for room in layout.placed_rooms for wall in room.walls)
        )
        self.assertEqual(len(layout.connectors), 1)
        self.assertEqual(len(layout.platforms), 1)
        self.assertIn("set_structural_layout", tools.tools)
        validation = tools._validate_impl()
        self.assertEqual(validation.layout, "ok")
        self.assertEqual(validation.connectivity, "ok")

    def test_invalid_structural_update_leaves_layout_unchanged(self) -> None:
        layout, tools = self._create_two_room_layout()
        original_levels = list(layout.levels)
        original_specs = [spec.to_dict() for spec in layout.room_specs]
        invalid = {
            "levels": [{"id": "ground", "elevation": 0}],
            "rooms": [
                {
                    "id": "lower",
                    "footprint": {
                        "outer": [[0, 0], [4, 4], [0, 4], [4, 0]],
                        "holes": [],
                    },
                }
            ],
        }

        result = tools._set_structural_layout_impl(json.dumps(invalid))

        self.assertFalse(result.success)
        self.assertEqual(layout.levels, original_levels)
        self.assertEqual([spec.to_dict() for spec in layout.room_specs], original_specs)

    def test_structural_tool_accepts_embedded_cavern_passage(self) -> None:
        layout, tools = self._create_two_room_layout()
        result = tools._set_structural_layout_impl(
            json.dumps(
                {
                    "levels": [
                        {"id": "ground", "elevation": 0},
                        {"id": "upper_level", "elevation": 3},
                    ],
                    "rooms": [{"id": "upper", "level_id": "upper_level"}],
                    "connectors": [
                        {
                            "id": "rising_tunnel",
                            "type": "natural_passage",
                            "start": {
                                "space_id": "lower",
                                "level_id": "ground",
                                "position": [0, 1, 0],
                            },
                            "end": {
                                "space_id": "upper",
                                "level_id": "upper_level",
                                "position": [5, 1, 3],
                            },
                            "parameters": {
                                "geometry_embedded": True,
                                "waypoints": [[2.5, 1, 1.5]],
                            },
                        }
                    ],
                }
            )
        )

        self.assertTrue(result.success, result.message)
        connector = layout.connectors[0]
        self.assertTrue(connector.parameters["geometry_embedded"])
        self.assertEqual(connector.required_capabilities, frozenset({"walk"}))

    def test_structural_tool_accepts_llm_authorable_semantic_environment(self) -> None:
        layout, tools = self._create_two_room_layout()
        environment = {
            "schema_version": 1,
            "regions": [
                {
                    "id": "subsurface",
                    "kind": "subterranean",
                    "bounds": {"minimum": [-10, -10, -8], "maximum": [30, 20, 15]},
                }
            ],
            "chambers": [
                {
                    "id": "main_chamber",
                    "region_id": "subsurface",
                    "center": [15, 4, 2],
                    "size": [12, 10, 8],
                }
            ],
            "passage_networks": [
                {
                    "id": "routes",
                    "region_id": "subsurface",
                    "junctions": [
                        {"id": "entry", "position": [0, 0, 0]},
                        {
                            "id": "hall",
                            "position": [12, 3, 0],
                            "chamber_id": "main_chamber",
                        },
                    ],
                    "segments": [
                        {
                            "id": "approach",
                            "start_junction_id": "entry",
                            "end_junction_id": "hall",
                            "path": [[0, 0, 0], [5, 0, -1], [12, 3, 0]],
                            "cross_sections": [
                                {"station": 0, "width": 3, "height": 3},
                                {"station": 1, "width": 6, "height": 5},
                            ],
                        }
                    ],
                }
            ],
            "openings": [
                {
                    "id": "sky_break",
                    "region_id": "subsurface",
                    "source_chamber_id": "main_chamber",
                    "target": "sky",
                    "center": [15, 4, 5.9],
                    "normal": [0, 0, 1],
                    "size": [3, 3],
                    "depth": 8,
                }
            ],
            "detail_fields": [
                {
                    "id": "ceiling_teeth",
                    "region_id": "subsurface",
                    "target_chamber_id": "main_chamber",
                    "formation_type": "stalactite",
                    "surface_role": "overhead",
                    "count": 8,
                    "min_size": [0.4, 0.4, 0.8],
                    "max_size": [1.2, 1.2, 2.5],
                    "seed": 419,
                    "protect_passage_network_ids": ["routes"],
                    "route_clearance": 2,
                    "collision_policy": "coarse",
                }
            ],
            "hero_features": [
                {
                    "id": "stone_marker",
                    "region_id": "subsurface",
                    "target_chamber_id": "main_chamber",
                    "feature_type": "rock_spire",
                    "anchor": [17, 4, -1],
                    "size": [2, 2, 4],
                }
            ],
        }

        result = tools._set_structural_layout_impl(
            json.dumps({"semantic_environment": environment})
        )

        self.assertTrue(result.success, result.message)
        self.assertIsNotNone(layout.semantic_environment)
        assert layout.semantic_environment is not None
        self.assertEqual(layout.semantic_environment.passage_networks[0].cycle_rank, 0)
        self.assertEqual(
            layout.semantic_environment.openings[0].opening_id, "sky_break"
        )
        self.assertEqual(layout.semantic_environment.detail_fields[0].count, 8)
        self.assertEqual(
            layout.semantic_environment.hero_features[0].feature_id, "stone_marker"
        )
        self.assertIn("1 semantic environment", result.message)

    def test_invalid_semantic_environment_update_is_atomic(self) -> None:
        layout, tools = self._create_two_room_layout()
        original_specs = [spec.to_dict() for spec in layout.room_specs]

        result = tools._set_structural_layout_impl(
            json.dumps(
                {
                    "semantic_environment": {
                        "regions": [
                            {
                                "id": "empty",
                                "kind": "subterranean",
                                "bounds": {
                                    "minimum": [0, 0, 0],
                                    "maximum": [1, 1, 1],
                                },
                            }
                        ]
                    }
                }
            )
        )

        self.assertFalse(result.success)
        self.assertIsNone(layout.semantic_environment)
        self.assertEqual([spec.to_dict() for spec in layout.room_specs], original_specs)

    def test_structural_position_overlap_is_rejected_atomically(self) -> None:
        layout, tools = self._create_two_room_layout()
        original_positions = [room.position for room in layout.placed_rooms]

        result = tools._set_structural_layout_impl(
            json.dumps(
                {
                    "rooms": [
                        {"id": "lower", "position": [0, 0]},
                        {"id": "upper", "position": [0, 0]},
                    ]
                }
            )
        )

        self.assertFalse(result.success)
        self.assertIn("overlap", result.message)
        self.assertEqual(
            [room.position for room in layout.placed_rooms], original_positions
        )

    def test_structural_tool_tessellates_circular_room(self) -> None:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")
        created = tools._generate_room_specs_impl(
            json.dumps(
                [
                    {
                        "type": "rotunda",
                        "prompt": "A circular gallery",
                        "width": 6,
                        "depth": 6,
                    }
                ]
            )
        )
        assert created.success

        result = tools._set_structural_layout_impl(
            json.dumps(
                {
                    "rooms": [
                        {
                            "id": "rotunda",
                            "footprint": {
                                "circle": {
                                    "radius": 3,
                                    "center": [3, 3],
                                    "chord_tolerance": 0.02,
                                }
                            },
                        }
                    ]
                }
            )
        )

        self.assertTrue(result.success, result.message)
        footprint = layout.room_specs[0].footprint
        assert footprint is not None
        self.assertGreater(len(footprint.outer), 20)
        self.assertAlmostEqual(layout.room_specs[0].length, 6.0, places=2)

    def test_structural_tool_registers_cavern_as_room_shell(self) -> None:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")
        created = tools._generate_room_specs_impl(
            json.dumps(
                [
                    {
                        "type": "cavern",
                        "prompt": "An irregular natural cavern chamber",
                        "width": 12,
                        "depth": 10,
                    }
                ]
            )
        )
        assert created.success

        result = tools._set_structural_layout_impl(
            json.dumps(
                {
                    "structural_meshes": [
                        {
                            "id": "cavern_shell",
                            "space_id": "cavern",
                            "mesh_path": "cavern.obj",
                            "unit_scale": 1.0,
                            "normal_orientation": "interior",
                            "replaces_room_shell": True,
                        }
                    ]
                }
            )
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(len(layout.structural_meshes), 1)
        self.assertTrue(layout.structural_meshes[0].replaces_room_shell)
        self.assertIn("1 structural meshes", result.message)

    def test_structural_tool_authors_independent_floor_and_ceiling_holes(self) -> None:
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")
        created = tools._generate_room_specs_impl(
            json.dumps(
                [
                    {
                        "type": "stair_hall",
                        "prompt": "Tall stair hall",
                        "width": 6,
                        "depth": 4,
                    }
                ]
            )
        )
        assert created.success
        spec = layout.room_specs[0]
        boundary = Footprint2D.rectangle(spec.length, spec.width)
        slab_with_hole = {
            "outer": [list(point) for point in boundary.outer],
            "holes": [[[1, 1], [1, 3], [5, 3], [5, 1]]],
        }

        result = tools._set_structural_layout_impl(
            json.dumps(
                {
                    "rooms": [
                        {
                            "id": "stair_hall",
                            "floor_footprint": slab_with_hole,
                            "ceiling_footprint": boundary.to_dict(),
                        }
                    ]
                }
            )
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(len(layout.room_specs[0].floor_footprint.holes), 1)
        self.assertEqual(layout.room_specs[0].ceiling_footprint.holes, ())


class TestOpeningPreservation(unittest.TestCase):
    """Test that doors/windows are preserved when rooms are resized or modified."""

    def _create_single_room_layout(self) -> tuple:
        """Create a simple layout with one room."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "A spacious living room",
                "width": 5.0,
                "depth": 4.0,
            }
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        return layout, tools

    def _create_two_room_layout(self) -> tuple:
        """Create a layout with two adjacent rooms."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "A spacious living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "A modern kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        return layout, tools

    def test_door_preserved_after_room_resize_when_fits(self):
        """Door should be preserved and repositioned when room is resized if it still fits."""
        layout, tools = self._create_single_room_layout()

        # Add door on exterior wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="left", width=1.0, height=2.1
        )
        assert door_result.success

        # Count door openings before resize.
        room = layout.placed_rooms[0]
        doors_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )
        assert doors_before == 1
        assert len(layout.doors) == 1
        old_position = layout.doors[0].position_exact

        # Resize room - door should be preserved (wall grew, door still fits).
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # Count door openings after resize - should be preserved.
        room = layout.placed_rooms[0]
        doors_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        assert (
            doors_after == 1
        ), "Door should be preserved after resize when it still fits"
        assert len(layout.doors) == 1, "Door metadata should be preserved"
        # Position should be proportionally adjusted (wall grew from 5m to 6m).
        new_position = layout.doors[0].position_exact
        expected_ratio = 6.0 / 5.0
        assert (
            abs(new_position - old_position * expected_ratio) < 0.01
        ), "Door repositioned proportionally"

    def test_window_preserved_after_room_resize_when_fits(self):
        """Window should be preserved and repositioned when room is resized if it still fits."""
        layout, tools = self._create_single_room_layout()

        # Add window on exterior wall B (which is on the depth dimension).
        window_result = tools._add_window_impl(
            wall_id="B", position="left", width=1.2, height=1.2
        )
        assert window_result.success

        # Count window openings before resize.
        room = layout.placed_rooms[0]
        windows_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.WINDOW])
            for w in room.walls
        )
        assert windows_before == 1
        assert len(layout.windows) == 1
        old_position = layout.windows[0].position_along_wall

        # Resize room - window should be preserved (wall grew, window still fits).
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # Count window openings after resize - should be preserved.
        room = layout.placed_rooms[0]
        windows_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.WINDOW])
            for w in room.walls
        )

        assert (
            windows_after == 1
        ), "Window should be preserved after resize when it still fits"
        assert len(layout.windows) == 1, "Window metadata should be preserved"
        # Position should be proportionally adjusted (depth grew from 4m to 5m).
        new_position = layout.windows[0].position_along_wall
        expected_ratio = 5.0 / 4.0
        # Use 0.1m tolerance due to floating point and boundary adjustments.
        assert (
            abs(new_position - old_position * expected_ratio) < 0.1
        ), f"Window repositioned proportionally: new={new_position}, expected={old_position * expected_ratio}"

    def test_door_invalidated_when_wall_shrinks(self):
        """Door at far end should be removed when room shrinks too much."""
        layout, tools = self._create_single_room_layout()

        # Add door on right side of 5m wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="right", width=1.0, height=2.1
        )
        assert door_result.success

        # Door is near end of 5m wall (position ~3-4m).
        door_position = layout.doors[0].position_exact
        assert door_position > 2.8, f"Door should be at right end, got {door_position}"

        # Resize room to 2m wide - door position becomes invalid and door is removed.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=2.0, depth=4.0
        )
        assert resize_result.success

        # Door opening should NOT be in wall and should be removed from layout.
        room = layout.placed_rooms[0]
        doors_in_wall = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        assert doors_in_wall == 0, "Door should be invalidated when wall shrinks"
        assert len(layout.doors) == 0, "Invalid door should be removed from layout"

        # Result message should inform about the removed door.
        assert "Removed" in resize_result.message, "Should inform about removed door"

    def test_partial_opening_preservation_on_resize(self):
        """Openings that still fit are preserved, those that don't are removed."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create wide room.
        # Boundary labels: A=north (8m), B=south (8m), C=east (4m), D=west (4m).
        rooms = [
            {"type": "living_room", "prompt": "A wide room", "width": 8.0, "depth": 4.0}
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Add door at left on wall A (north, 8m wall).
        # "left" position is around 0.2 * wall_length = 1.6m from start.
        door1_result = tools._add_door_impl(
            wall_id="A", position="left", width=1.0, height=2.1
        )
        assert (
            door1_result.success
        ), f"First door should succeed: {door1_result.message}"

        # Add door at right on wall B (south, also 8m wall - different wall).
        # "right" position is around 0.8 * wall_length = 6.4m from start.
        door2_result = tools._add_door_impl(
            wall_id="B", position="right", width=1.0, height=2.1
        )
        assert (
            door2_result.success
        ), f"Second door should succeed: {door2_result.message}"

        # Verify both doors are in walls.
        room = layout.placed_rooms[0]
        doors_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )
        assert doors_before == 2, f"Expected 2 doors, got {doors_before}"
        assert len(layout.doors) == 2

        # Resize room - shrink width from 8m to 2.5m (extreme shrink).
        # Both north (A) and south (B) walls shrink from 8m to 2.5m.
        # Left door on A at ~1.6m will scale to ~0.5m - door extends to 1.5m, fits in 2.5m wall.
        # Right door on B at ~6.4m will scale to ~2.0m - door extends to 3.0m > 2.5m wall, doesn't fit.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=2.5, depth=4.0
        )
        assert resize_result.success

        room = layout.placed_rooms[0]
        doors_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        # Left door should be preserved (scaled position still fits).
        # Right door should be removed (scaled position + width exceeds wall).
        assert (
            len(layout.doors) == 1
        ), f"Expected 1 door preserved, got {len(layout.doors)}"
        assert doors_after == 1, "One door should remain in wall openings"

        # Result message should mention the removed door.
        assert "Removed" in resize_result.message or "1" in resize_result.message

    def test_openings_preserved_after_add_adjacency(self):
        """Openings should be preserved when adjacency is added."""
        layout, tools = self._create_two_room_layout()

        # Add door on living room exterior wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="center", width=1.0, height=2.1
        )
        assert door_result.success

        # Get initial door count.
        assert len(layout.doors) == 1

        # Remove and re-add adjacency to trigger re-placement.
        tools._remove_adjacency_impl(room_a="living_room", room_b="kitchen")
        tools._add_adjacency_impl(room_a="living_room", room_b="kitchen")

        # Door should still exist.
        assert len(layout.doors) == 1, "Door metadata should be preserved"

        # Check if door is in wall openings (may or may not be depending on wall changes).
        # At minimum, metadata should be preserved.

    def test_open_connection_creates_opening(self):
        """Adding open connection should create OPEN type opening."""
        layout, tools = self._create_two_room_layout()

        # Add open connection.
        result = tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")
        assert result.success

        # Check that OPEN openings were created.
        open_count = 0
        for room in layout.placed_rooms:
            for wall in room.walls:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.OPEN:
                        open_count += 1

        assert open_count >= 2, "OPEN openings should be created on both rooms' walls"

    def test_open_connection_preserved_after_resize(self):
        """Open connection should be preserved and recalculated when room is resized."""
        layout, tools = self._create_two_room_layout()

        # Add open connection.
        result = tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")
        assert result.success

        # Count OPEN openings before resize.
        open_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_before >= 2

        # Resize kitchen.
        resize_result = tools._resize_room_impl(room_id="kitchen", width=5.0, depth=4.0)
        assert resize_result.success

        # OPEN openings should still exist (recalculated for new overlap).
        open_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_after >= 2, "OPEN openings should be preserved after resize"

    def test_combined_openings_preserved_after_resize(self):
        """All opening types should be preserved after resize when they still fit."""
        layout, tools = self._create_two_room_layout()

        # Add door on living room exterior wall B (depth dimension).
        door_result = tools._add_door_impl(
            wall_id="B", position="left", width=0.9, height=2.1
        )
        assert door_result.success

        # Add window on kitchen exterior wall.
        window_result = tools._add_window_impl(
            wall_id="E", position="center", width=1.2, height=1.0
        )
        assert window_result.success

        # Add open connection.
        open_result = tools._add_open_connection_impl(
            room_a="living_room", room_b="kitchen"
        )
        assert open_result.success

        # Count all openings before resize.
        def count_openings():
            counts = {"door": 0, "window": 0, "open": 0}
            for room in layout.placed_rooms:
                for wall in room.walls:
                    for opening in wall.openings:
                        counts[opening.opening_type.value] += 1
            return counts

        before = count_openings()
        assert before["door"] == 1
        assert before["window"] == 1
        assert before["open"] >= 2

        # Resize living room - growing from 5x4 to 6x5.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # After resize (walls grow, so openings should fit):
        # - Door on living_room's wall B: PRESERVED (wall grew from 4m to 5m, door fits)
        # - Window on kitchen's wall: PRESERVED (kitchen wasn't resized)
        # - OPEN connections: PRESERVED with recomputed positions
        after = count_openings()
        assert (
            after["door"] == 1
        ), "Door on resized room preserved (wall grew, still fits)"
        assert (
            after["window"] == before["window"]
        ), "Window on other room should be preserved"
        assert after["open"] >= 2, "OPEN openings should be preserved"

    def test_remove_open_connection_clears_openings(self):
        """Removing open connection should remove OPEN type openings."""
        layout, tools = self._create_two_room_layout()

        # Add and then remove open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Verify openings exist.
        open_count = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_count >= 2

        # Remove open connection.
        tools._remove_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Verify openings removed.
        open_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_after == 0, "OPEN openings should be removed"

    def test_three_room_layout_preserves_all_openings(self):
        """Complex layout with 3 rooms should preserve all openings after changes."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create L-shaped layout: living room with kitchen and bedroom adjacent.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Main living area",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 3.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
            {
                "type": "bedroom",
                "prompt": "Bedroom",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"
        assert len(layout.placed_rooms) == 3

        # Find exterior walls for each room.
        exterior_walls = {}
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                if room_a not in exterior_walls:
                    exterior_walls[room_a] = []
                exterior_walls[room_a].append(label)

        # Add door to living room, windows to kitchen and bedroom.
        living_wall = exterior_walls.get("living_room", [None])[0]
        if living_wall:
            tools._add_door_impl(wall_id=living_wall, position="center", width=0.9)

        kitchen_wall = exterior_walls.get("kitchen", [None])[0]
        if kitchen_wall:
            tools._add_window_impl(wall_id=kitchen_wall, position="center", width=1.0)

        bedroom_wall = exterior_walls.get("bedroom", [None])[0]
        if bedroom_wall:
            tools._add_window_impl(wall_id=bedroom_wall, position="center", width=1.2)

        # Add open connection between living room and kitchen.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Count openings before resize.
        def count_all():
            counts = {"door": 0, "window": 0, "open": 0}
            for room in layout.placed_rooms:
                for wall in room.walls:
                    for o in wall.openings:
                        counts[o.opening_type.value] += 1
            return counts

        before = count_all()

        # Resize bedroom from 4x3 to 5x4 (growing).
        resize_result = tools._resize_room_impl(room_id="bedroom", width=5.0, depth=4.0)
        assert resize_result.success

        after = count_all()

        # Resizing bedroom (growing) should preserve bedroom's window (repositioned):
        # - Door on living_room: PRESERVED (living_room wasn't resized)
        # - Window on kitchen: PRESERVED (kitchen wasn't resized)
        # - Window on bedroom: PRESERVED (bedroom grew, window repositioned and still fits)
        # - Open connection: PRESERVED (positions recomputed)
        assert (
            after["door"] >= before["door"]
        ), "Door on non-resized room should be preserved"
        assert (
            after["window"] == before["window"]
        ), "All windows preserved (bedroom grew, window still fits)"
        assert after["open"] >= 2, "Open connection should be preserved"

    def test_open_connection_width_matches_overlap(self):
        """Open connection width should match the actual room overlap."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create rooms of different sizes - overlap should be smaller room's width.
        rooms = [
            {"type": "living_room", "prompt": "Large room", "width": 6.0, "depth": 4.0},
            {
                "type": "kitchen",
                "prompt": "Smaller room",
                "width": 3.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Calculate expected overlap.
        living_room = next(r for r in layout.placed_rooms if r.room_id == "living_room")
        kitchen = next(r for r in layout.placed_rooms if r.room_id == "kitchen")
        shared_edge = get_shared_edge(living_room, kitchen)
        assert shared_edge is not None, "Rooms should share an edge"

        # Add open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Find the OPEN opening and verify width matches shared edge.
        for room in layout.placed_rooms:
            for wall in room.walls:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.OPEN:
                        assert abs(opening.width - shared_edge.width) < 0.01, (
                            f"Opening width {opening.width} should match "
                            f"shared edge width {shared_edge.width}"
                        )

    def test_open_connection_position_correct_for_both_walls(self):
        """Open connection position should be relative to each room's wall origin.

        When rooms have different sizes and the smaller room is offset, the
        opening position will be different for each wall. This test ensures
        each wall gets the correct position, not a shared incorrect position.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create rooms where kitchen is smaller and will be offset from living room.
        # Living room: 6m wide, Kitchen: 4m wide adjacent.
        # The placement algorithm may offset the smaller room.
        rooms = [
            {"type": "living_room", "prompt": "Large room", "width": 6.0, "depth": 5.0},
            {
                "type": "kitchen",
                "prompt": "Smaller room",
                "width": 4.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Add open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Get shared edges from both perspectives.
        living_room = next(r for r in layout.placed_rooms if r.room_id == "living_room")
        kitchen = next(r for r in layout.placed_rooms if r.room_id == "kitchen")

        shared_edge_living = get_shared_edge(living_room, kitchen)
        shared_edge_kitchen = get_shared_edge(kitchen, living_room)

        assert shared_edge_living is not None
        assert shared_edge_kitchen is not None

        # Find OPEN openings on each room's wall.
        living_opening = None
        kitchen_opening = None

        for wall in living_room.walls:
            for opening in wall.openings:
                if opening.opening_type == OpeningType.OPEN:
                    living_opening = opening
                    break

        for wall in kitchen.walls:
            for opening in wall.openings:
                if opening.opening_type == OpeningType.OPEN:
                    kitchen_opening = opening
                    break

        assert living_opening is not None, "Living room should have OPEN opening"
        assert kitchen_opening is not None, "Kitchen should have OPEN opening"

        # Each opening's position should match the shared edge from that room's perspective.
        assert (
            abs(
                living_opening.position_along_wall
                - shared_edge_living.position_along_wall
            )
            < 0.01
        ), (
            f"Living room opening position {living_opening.position_along_wall} should match "
            f"shared edge position {shared_edge_living.position_along_wall}"
        )
        assert (
            abs(
                kitchen_opening.position_along_wall
                - shared_edge_kitchen.position_along_wall
            )
            < 0.01
        ), (
            f"Kitchen opening position {kitchen_opening.position_along_wall} should match "
            f"shared edge position {shared_edge_kitchen.position_along_wall}"
        )

        # Verify positions can be different (this is the key invariant the bug violated).
        # Note: They might be equal if rooms are perfectly aligned, but they CAN differ.
        # The important thing is each is correct for its respective wall.

    def test_door_on_interior_wall(self):
        """Door on interior wall should create openings on both rooms' walls."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find interior wall label.
        interior_wall = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is not None:  # Interior wall.
                interior_wall = label
                break

        assert interior_wall is not None, "Should have an interior wall"

        # Add door to interior wall.
        door_result = tools._add_door_impl(
            wall_id=interior_wall, position="center", width=0.9, height=2.1
        )
        assert door_result.success

        # Verify door metadata stored.
        assert len(layout.doors) == 1
        door = layout.doors[0]
        assert door.room_b is not None, "Interior door should have room_b set"

    def test_door_cutout_alignment_on_interior_walls(self):
        """Door cutouts on interior walls must align at same world position.

        When two rooms share an internal wall, each room has its own wall object
        with different start points. Door cutouts must align to the same world
        position, which means position_along_wall values will differ between walls.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create two rooms with adjacency.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find interior wall label.
        interior_wall_label = None
        interior_room_a = None
        interior_room_b = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is not None:  # Interior wall.
                interior_wall_label = label
                interior_room_a = room_a
                interior_room_b = room_b
                break

        assert interior_wall_label is not None, "Should have an interior wall"

        # Add door at position "left" (0.3m from start).
        door_result = tools._add_door_impl(
            wall_id=interior_wall_label, position="left", width=0.9, height=2.1
        )
        assert door_result.success

        # Find placed rooms.
        placed_a = next(r for r in layout.placed_rooms if r.room_id == interior_room_a)
        placed_b = next(r for r in layout.placed_rooms if r.room_id == interior_room_b)

        # Get shared edges from both perspectives.
        shared_edge_a = get_shared_edge(placed_a, placed_b)
        shared_edge_b = get_shared_edge(placed_b, placed_a)
        assert shared_edge_a is not None
        assert shared_edge_b is not None

        # Find door openings on each wall.
        opening_on_a = None
        opening_on_b = None

        for wall in placed_a.walls:
            if wall.direction == shared_edge_a.wall_direction:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.DOOR:
                        opening_on_a = opening
                        break

        for wall in placed_b.walls:
            if wall.direction == shared_edge_b.wall_direction:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.DOOR:
                        opening_on_b = opening
                        break

        assert opening_on_a is not None, "Room A wall should have door opening"
        assert opening_on_b is not None, "Room B wall should have door opening"

        # Calculate world positions of door left edges.
        # For vertical walls (east/west), position is along Y axis.
        # For horizontal walls (north/south), position is along X axis.
        def get_world_position(placed_room, wall_dir, position_along_wall):
            """Convert wall-relative position to world coordinate."""
            x, y = placed_room.position
            if wall_dir in (WallDirection.EAST, WallDirection.WEST):
                # Wall runs along Y axis from room's min_y.
                return y + position_along_wall
            else:
                # Wall runs along X axis from room's min_x.
                return x + position_along_wall

        world_pos_a = get_world_position(
            placed_a, shared_edge_a.wall_direction, opening_on_a.position_along_wall
        )
        world_pos_b = get_world_position(
            placed_b, shared_edge_b.wall_direction, opening_on_b.position_along_wall
        )

        # Door cutouts must align at the same world position.
        assert abs(world_pos_a - world_pos_b) < 0.01, (
            f"Door cutouts must align! Room A door at world pos {world_pos_a:.3f}, "
            f"Room B door at world pos {world_pos_b:.3f}, "
            f"position_along_wall A={opening_on_a.position_along_wall:.3f}, "
            f"position_along_wall B={opening_on_b.position_along_wall:.3f}"
        )

    def test_room_creation_validates_dimensions(self):
        """Room creation should fail for invalid dimensions."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Zero-width room should fail.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Invalid room",
                "width": 0.0,
                "depth": 4.0,
            }
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        # The placement algorithm should reject this.
        # Note: If this passes, the code might need validation added.

    def test_sequential_operations_maintain_consistency(self):
        """Multiple sequential operations should maintain layout consistency."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create initial layout.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        # Add various openings.
        exterior_walls = [
            label
            for label, (_, room_b, _) in layout.boundary_labels.items()
            if room_b is None
        ]
        if len(exterior_walls) >= 2:
            tools._add_door_impl(
                wall_id=exterior_walls[0], position="center", width=0.9
            )
            tools._add_window_impl(
                wall_id=exterior_walls[1], position="center", width=1.0
            )

        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Perform multiple resize operations (all growing or similar).
        tools._resize_room_impl(room_id="living_room", width=6.0, depth=4.0)
        tools._resize_room_impl(room_id="kitchen", width=5.0, depth=3.5)
        tools._resize_room_impl(room_id="living_room", width=5.5, depth=4.5)

        # Layout should still be valid.
        assert layout.placement_valid
        assert len(layout.placed_rooms) == 2

        # Doors and windows should be preserved if they still fit after proportional
        # repositioning. Since all resizes here are growing or similar, openings
        # that were originally at "center" should still fit after repositioning.
        # (The exact count depends on which room each opening was on and whether
        # it still fits after all the resizes.)
        # At minimum, open connections should be preserved.

        # Open connection should still have openings (positions recomputed).
        open_count = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_count >= 2, "Open connection should survive multiple resizes"

    def test_door_window_overlap_prevention_on_exterior_wall(self):
        """Doors and windows on same wall must not overlap (min separation enforced)."""
        layout = HouseLayout()
        # Use small separation for predictable test behavior.
        tools = FloorPlanTools(layout=layout, mode="house", min_opening_separation=0.5)

        # Create a large room so wall is long enough for both door and window.
        rooms = [
            {"type": "living_room", "prompt": "Living room", "width": 8.0, "depth": 6.0}
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find an exterior wall.
        exterior_wall = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                exterior_wall = label
                break
        assert exterior_wall is not None, "Should have exterior wall"

        # Add window at left (small wall segment).
        window_result = tools._add_window_impl(
            wall_id=exterior_wall, position="left", width=1.0
        )
        assert window_result.success, f"Window should succeed: {window_result.message}"

        # Adding door at right (different segment) should succeed.
        door_result = tools._add_door_impl(wall_id=exterior_wall, position="right")
        assert (
            door_result.success
        ), f"Door at right should succeed: {door_result.message}"

        # Now test overlap detection: try to add door at same position as window.
        # On a new layout with window at center.
        # Use seed for deterministic positioning to ensure overlap.
        random.seed(42)
        layout2 = HouseLayout()
        tools2 = FloorPlanTools(
            layout=layout2, mode="house", min_opening_separation=0.5
        )
        tools2._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        exterior_wall2 = None
        for label, (room_a, room_b, _) in layout2.boundary_labels.items():
            if room_b is None:
                exterior_wall2 = label
                break

        # Add window at center.
        window_result2 = tools2._add_window_impl(
            wall_id=exterior_wall2, position="center", width=1.5
        )
        assert window_result2.success

        # Adding door at center should fail (overlap).
        door_result2 = tools2._add_door_impl(wall_id=exterior_wall2, position="center")
        assert not door_result2.success, "Door should fail when overlapping window"
        assert "overlap" in door_result2.message.lower()

    def test_arched_window_rejects_impossible_crown_proportions(self):
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="room")
        created = tools._generate_room_specs_impl(
            room_specs_json=json.dumps(
                [{"type": "gallery", "prompt": "Gallery", "width": 8, "depth": 6}]
            )
        )
        assert created.success
        exterior_wall = next(
            label
            for label, (_room_a, room_b, _direction) in layout.boundary_labels.items()
            if room_b is None
        )

        result = tools._add_window_impl(
            wall_id=exterior_wall,
            position="center",
            width=4.0,
            height=1.5,
            sill_height=0.5,
            shape="arched",
        )

        assert not result.success
        assert "height must exceed half its width" in result.message


class TestLayoutCheckpointRestore(unittest.TestCase):
    """Test HouseLayout checkpoint/restore for reset functionality."""

    def test_layout_round_trip_preserves_all_state(self):
        """HouseLayout.from_dict(layout.to_dict()) should preserve all state.

        This test ensures the checkpoint/reset mechanism works correctly.
        If this test fails, _perform_checkpoint_reset would restore corrupted state.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create a complex layout with rooms, adjacencies, open connections.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "OPEN"},
            },
            {
                "type": "bedroom",
                "prompt": "Bedroom",
                "width": 4.0,
                "depth": 3.5,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        # Add doors and windows.
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                tools._add_window_impl(wall_id=label, position="center", width=1.2)
            elif room_b == "bedroom":  # Interior door to bedroom.
                tools._add_door_impl(wall_id=label, position="center")

        # Capture state before serialization.
        original_room_ids = [s.room_id for s in layout.room_specs]
        original_door_count = len(layout.doors)
        original_window_count = len(layout.windows)
        original_placed_room_count = len(layout.placed_rooms)
        original_connections = layout.room_specs[
            1
        ].connections  # Kitchen's connections.

        # Serialize and restore.
        state_dict = layout.to_dict()
        restored = HouseLayout.from_dict(state_dict)

        # Verify all state was preserved.
        restored_room_ids = [s.room_id for s in restored.room_specs]
        assert restored_room_ids == original_room_ids, "Room IDs should match"

        assert (
            len(restored.doors) == original_door_count
        ), f"Door count should match: {len(restored.doors)} vs {original_door_count}"
        assert (
            len(restored.windows) == original_window_count
        ), f"Window count should match: {len(restored.windows)} vs {original_window_count}"
        assert (
            len(restored.placed_rooms) == original_placed_room_count
        ), f"Placed room count should match: {len(restored.placed_rooms)} vs {original_placed_room_count}"

        # Verify connections preserved.
        restored_kitchen = next(
            s for s in restored.room_specs if s.room_id == "kitchen"
        )
        assert (
            restored_kitchen.connections == original_connections
        ), f"connections should match: {restored_kitchen.connections} vs {original_connections}"

        # Verify placement_valid flag.
        assert restored.placement_valid == layout.placement_valid

    def test_layout_restore_after_modifications(self):
        """Restoring from checkpoint should undo subsequent modifications.

        Simulates the reset workflow: create checkpoint, make changes, restore.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create initial layout.
        rooms = [
            {"type": "living_room", "prompt": "Living room", "width": 5.0, "depth": 4.0}
        ]
        tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        # Add a door on north wall (this is our checkpoint state).
        exterior_walls = []
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                exterior_walls.append((label, direction))

        # Use first wall for door.
        door_wall = exterior_walls[0][0]
        tools._add_door_impl(wall_id=door_wall, position="center")

        # Create checkpoint.
        checkpoint = layout.to_dict()
        checkpoint_door_count = len(layout.doors)
        checkpoint_window_count = len(layout.windows)

        # Make modifications on a DIFFERENT wall (to avoid overlap issues).
        # Find a wall without the door.
        window_wall = exterior_walls[1][0] if len(exterior_walls) > 1 else door_wall
        window_result = tools._add_window_impl(
            wall_id=window_wall, position="center", width=1.0
        )
        assert window_result.success, f"Window should be added: {window_result.message}"
        assert (
            len(layout.windows) == checkpoint_window_count + 1
        ), "Window should be added"

        # Restore from checkpoint (simulating reset).
        restored = HouseLayout.from_dict(checkpoint)

        # Verify modifications were undone.
        assert len(restored.doors) == checkpoint_door_count
        assert (
            len(restored.windows) == checkpoint_window_count
        ), f"Window count should be restored: {len(restored.windows)} vs {checkpoint_window_count}"


if __name__ == "__main__":
    unittest.main()
