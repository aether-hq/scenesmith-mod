"""Executable guards for deterministic native SceneSmith completion placement."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from scenesmith.aether.native_placement import (
    CeilingPlacementAdapter,
    FloorPlacementAdapter,
    SurfacePlacementAdapter,
    WallPlacementAdapter,
)


def _stage_input():
    return {
        "request": {
            "shell": {"dimensions_m": [10, 3, 8]},
            "functional_zones": [
                {
                    "zone_id": "guest-zone",
                    "polygon_xy_m": [[0, 0], [10, 0], [10, 8], [0, 8]],
                }
            ],
            "circulation_routes": [
                {
                    "points_xy_m": [[0, 4], [10, 4]],
                    "clear_width_m": 1,
                }
            ],
            "story_positions": [{"position_m": [5, 2, 0], "clear_radius_m": 0.5}],
        }
    }


def _asset(object_id="asset-template", size=(0.5, 0.5, 0.5)):
    return SimpleNamespace(
        object_id=object_id,
        bbox_min=np.zeros(3),
        bbox_max=np.asarray(size, dtype=float),
    )


def _operation(kind, *, supports=()):
    return {
        "operation_id": f"test-{kind}",
        "operation": kind,
        "functional_zone_ids": ["guest-zone"],
        "support_role_ids": list(supports),
        "arrangement": "distributed",
    }


class _Scene:
    def __init__(self):
        self.objects = {}

    def remove_object(self, object_id):
        return self.objects.pop(object_id, None) is not None


class _Tool:
    def __init__(self, scene):
        self.scene = scene
        self.calls = []

    def set_noise_profile(self, mode):
        self.mode = mode

    def _place(self, *args):
        self.calls.append(args)
        object_id = f"placed-{len(self.calls)}"
        self.scene.objects[object_id] = SimpleNamespace()
        return json.dumps({"success": True, "object_id": object_id})

    _add_furniture_to_scene_impl = _place
    _place_ceiling_object_impl = _place
    _place_wall_object_impl = _place
    _place_manipuland_on_surface_impl = _place


def test_floor_uses_native_tool_and_retries_a_colliding_candidate():
    scene = _Scene()
    tool = _Tool(scene)
    checks = 0

    def collisions(_scene):
        nonlocal checks
        checks += 1
        if checks == 1:
            return [SimpleNamespace(object_a_id="placed-1", object_b_id="existing")]
        return []

    adapter = FloorPlacementAdapter(tool, _stage_input(), collisions)
    result = adapter.place(
        scene,
        _asset(),
        _operation("place-floor-group"),
        {"variant_id": "chair"},
        instance_index=0,
        round_index=0,
    )
    assert result == "placed-2"
    assert "placed-1" not in scene.objects
    assert len(tool.calls) == 2


def test_surface_uses_only_the_requested_semantic_support_role():
    scene = _Scene()
    surface = SimpleNamespace(
        surface_id="surface-1",
        bounding_box_min=np.array([-1.0, -0.5, 0.0]),
        bounding_box_max=np.array([1.0, 0.5, 1.0]),
    )
    scene.objects["counter"] = SimpleNamespace(
        name="counter",
        metadata={"aether_role_id": "service-counter", "aether_accepts_dressing": True},
        support_surfaces=[surface],
    )
    scene.objects["table"] = SimpleNamespace(
        name="table",
        metadata={"aether_role_id": "guest-table", "aether_accepts_dressing": True},
        support_surfaces=[surface],
    )
    tools = []

    def factory(owner_id, surfaces):
        assert str(owner_id) == "counter"
        assert set(surfaces) == {"surface-1"}
        tool = _Tool(scene)
        tools.append(tool)
        return tool

    adapter = SurfacePlacementAdapter(factory, _stage_input(), lambda _: [])
    result = adapter.place(
        scene,
        _asset(size=(0.1, 0.1, 0.3)),
        _operation("populate-surfaces", supports=("service-counter",)),
        {"variant_id": "bottle"},
        instance_index=0,
        round_index=0,
    )
    assert result == "placed-1"
    assert len(tools) == 1


def test_wall_uses_native_surface_bounds_and_opening_checks():
    scene = _Scene()

    class Surface:
        length = 5.0
        height = 3.0

        @staticmethod
        def check_object_bounds(x, z, width, height):
            return (x > 1.0, None)

    tool = _Tool(scene)
    tool.surfaces_by_id = {"wall-1": Surface()}
    adapter = WallPlacementAdapter(tool, _stage_input(), lambda _: [])
    result = adapter.place(
        scene,
        _asset(size=(0.5, 0.1, 0.5)),
        _operation("place-wall-group"),
        {"variant_id": "sign"},
        instance_index=0,
        round_index=0,
    )
    assert result == "placed-1"
    assert tool.calls[0][1] == "wall-1"


def test_ceiling_uses_only_candidates_inside_the_accepted_zone():
    scene = _Scene()
    tool = _Tool(scene)
    tool.room_bounds = (-5.0, -4.0, 5.0, 4.0)
    adapter = CeilingPlacementAdapter(tool, _stage_input(), lambda _: [])
    result = adapter.place(
        scene,
        _asset(size=(1, 1, 0.3)),
        _operation("place-ceiling-group"),
        {"variant_id": "fixture"},
        instance_index=0,
        round_index=0,
    )
    assert result == "placed-1"
    assert -5 <= tool.calls[0][1] <= 5
    assert -4 <= tool.calls[0][2] <= 4
