"""Tests for floor plan tools - door/window preservation after room changes."""

import json
import math
import unittest

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.structure.geometry_models.surface_models import Footprint2D
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
