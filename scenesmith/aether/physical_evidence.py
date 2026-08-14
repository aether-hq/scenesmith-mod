"""Measured postconditions for SceneSmith contextual completion rounds."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from typing import Any

from scenesmith.agent_utils.room import ObjectType


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def room_geometry_digest(scene: Any) -> str:
    """Hash only compiled architecture, never movable scenic objects."""
    value = scene.to_state_dict().get("room_geometry")
    if value is None:
        raise ValueError("SceneSmith scene has no compiled room geometry")
    return _digest(value)


def _bounds_xy(item: Any) -> tuple[float, float, float, float] | None:
    bounds = item.compute_world_bounds()
    if bounds is None:
        return None
    minimum, maximum = bounds
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])


def _point_aabb_distance(
    point: tuple[float, float], bounds: tuple[float, ...]
) -> float:
    x, y = point
    min_x, min_y, max_x, max_y = bounds
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return math.hypot(dx, dy)


def _segment_aabb_distance(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, ...],
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    samples = max(2, math.ceil(length / 0.1) + 1)
    return min(
        _point_aabb_distance(
            (
                start[0] + delta_x * index / (samples - 1),
                start[1] + delta_y * index / (samples - 1),
            ),
            bounds,
        )
        for index in range(samples)
    )


def _local_xy(
    point: list[float] | tuple[float, ...], width: float, depth: float
) -> tuple[float, float]:
    return float(point[0]) - width / 2, float(point[1]) - depth / 2


class PhysicalEvidenceProvider:
    """Recompute every physical census field from the current RoomScene."""

    def __init__(
        self, scene: Any, stage_input: dict[str, Any], *, baseline_geometry_sha256: str
    ):
        self.scene = scene
        self.stage_input = stage_input
        self.baseline_geometry_sha256 = baseline_geometry_sha256

    def _scenic_bounds(self) -> list[tuple[str, tuple[float, ...]]]:
        architecture = {ObjectType.WALL, ObjectType.FLOOR}
        values = []
        for object_id, item in self.scene.objects.items():
            if item.object_type in architecture:
                continue
            bounds = _bounds_xy(item)
            if bounds is not None:
                values.append((str(object_id), bounds))
        return values

    def _clear_routes(
        self, bounds_by_id: list[tuple[str, tuple[float, ...]]]
    ) -> list[str]:
        request = self.stage_input["request"]
        width, _, depth = request["shell"]["dimensions_m"]
        clear = []
        for route in request["circulation_routes"]:
            points = tuple(
                _local_xy(point, width, depth) for point in route["points_xy_m"]
            )
            radius = float(route["clear_width_m"]) / 2
            blocked = any(
                _segment_aabb_distance(start, end, bounds) < radius
                for _, bounds in bounds_by_id
                for start, end in pairwise(points)
            )
            if not blocked:
                clear.append(str(route["route_id"]))
        return clear

    def _clear_story_positions(
        self, bounds_by_id: list[tuple[str, tuple[float, ...]]]
    ) -> list[str]:
        request = self.stage_input["request"]
        width, _, depth = request["shell"]["dimensions_m"]
        clear = []
        for position in request["story_positions"]:
            point = _local_xy(position["position_m"], width, depth)
            radius = float(position["clear_radius_m"])
            if all(
                _point_aabb_distance(point, bounds) >= radius
                for _, bounds in bounds_by_id
            ):
                clear.append(str(position["position_id"]))
        return clear

    def _supported_ids(self) -> list[str]:
        supported = []
        for object_id, item in self.scene.objects.items():
            if item.object_type in {ObjectType.WALL, ObjectType.FLOOR}:
                continue
            if (
                item.object_type
                in {
                    ObjectType.FURNITURE,
                    ObjectType.CEILING_MOUNTED,
                }
                or item.placement_info is not None
            ):
                supported.append(str(object_id))
        return supported

    def __call__(self) -> dict[str, Any]:
        from scenesmith.agent_utils.physics_validation import compute_scene_collisions

        bounds_by_id = self._scenic_bounds()
        collisions = compute_scene_collisions(self.scene)
        collision_ids = sorted(
            {
                str(value)
                for item in collisions
                for value in (item.object_a_id, item.object_b_id)
            }
        )
        return {
            "architecture_sha256": self.stage_input["locked_architecture_sha256"],
            "baseline_room_geometry_sha256": self.baseline_geometry_sha256,
            "current_room_geometry_sha256": room_geometry_digest(self.scene),
            "clear_circulation_route_ids": self._clear_routes(bounds_by_id),
            "clear_story_position_ids": self._clear_story_positions(bounds_by_id),
            "collision_instance_ids": collision_ids,
            "supported_instance_ids": self._supported_ids(),
            # Materials and visibility are measured by later mandatory stages. Completion
            # must not claim them before Blender/inspection has actually produced evidence.
            "pbr_complete_instance_ids": [],
            "visible_view_ids_by_instance": {},
        }
