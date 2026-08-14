"""Measured evidence guards for native SceneSmith completion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scenesmith.aether.physical_evidence import (
    PhysicalEvidenceProvider,
    room_geometry_digest,
)
from scenesmith.agent_utils.room import ObjectType


class Item:
    def __init__(self, kind, bounds, *, placement=None):
        self.object_type = kind
        self._bounds = bounds
        self.placement_info = placement

    def compute_world_bounds(self):
        return np.asarray(self._bounds[0]), np.asarray(self._bounds[1])


class Scene:
    def __init__(self, objects):
        self.objects = objects

    def to_state_dict(self):
        return {"room_geometry": {"width": 10, "depth": 8}, "objects": {}}


def _stage_input():
    return {
        "locked_architecture_sha256": "a" * 64,
        "request": {
            "shell": {"dimensions_m": [10, 3, 8]},
            "circulation_routes": [
                {
                    "route_id": "guest-route",
                    "points_xy_m": [[0, 4], [10, 4]],
                    "clear_width_m": 1,
                }
            ],
            "story_positions": [
                {
                    "position_id": "meeting-point",
                    "position_m": [5, 2, 0],
                    "clear_radius_m": 0.5,
                }
            ],
        },
    }


def test_provider_measures_blocked_routes_and_clear_story_positions(monkeypatch):
    scene = Scene(
        {
            "chair": Item(ObjectType.FURNITURE, ((-0.2, -0.2, 0), (0.2, 0.2, 1))),
            "bottle": Item(
                ObjectType.MANIPULAND,
                ((2, 2, 1), (2.1, 2.1, 1.3)),
                placement=SimpleNamespace(parent_surface_id="surface"),
            ),
        }
    )
    monkeypatch.setattr(
        "scenesmith.agent_utils.physics_validation.compute_scene_collisions",
        lambda _scene: [],
    )
    baseline = room_geometry_digest(scene)
    evidence = PhysicalEvidenceProvider(
        scene, _stage_input(), baseline_geometry_sha256=baseline
    )()
    assert evidence["clear_circulation_route_ids"] == []
    assert evidence["clear_story_position_ids"] == ["meeting-point"]
    assert evidence["supported_instance_ids"] == ["chair", "bottle"]
    assert evidence["current_room_geometry_sha256"] == baseline
    assert evidence["pbr_complete_instance_ids"] == []


def test_provider_reports_real_collision_instance_ids(monkeypatch):
    scene = Scene({"chair": Item(ObjectType.FURNITURE, ((2, 2, 0), (3, 3, 1)))})
    monkeypatch.setattr(
        "scenesmith.agent_utils.physics_validation.compute_scene_collisions",
        lambda _scene: [SimpleNamespace(object_a_id="chair", object_b_id="table")],
    )
    baseline = room_geometry_digest(scene)
    evidence = PhysicalEvidenceProvider(
        scene, _stage_input(), baseline_geometry_sha256=baseline
    )()
    assert evidence["collision_instance_ids"] == ["chair", "table"]
    assert evidence["clear_circulation_route_ids"] == ["guest-route"]
