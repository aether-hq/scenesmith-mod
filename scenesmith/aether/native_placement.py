"""Deterministic candidate solvers over SceneSmith's native placement tools.

The completion author is allowed to choose semantic roles, arrangements, asset
briefs, and counts.  It never supplies coordinates.  These adapters enumerate
bounded candidates from the accepted room contract and pass each candidate
through the native SceneSmith tool and physics validator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from scenesmith.agent_utils.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.room import UniqueID

CollisionChecker = Callable[[Any], list[Any]]
ManipulandToolFactory = Callable[[UniqueID, dict[str, Any]], Any]


def _native_collisions(scene: Any) -> list[Any]:
    from scenesmith.agent_utils.physics_validation import compute_scene_collisions

    return compute_scene_collisions(scene)


def _dimensions(asset: Any) -> tuple[float, float, float]:
    if asset.bbox_min is None or asset.bbox_max is None:
        raise ValueError(f"asset {asset.object_id} has no measured bounds")
    size = np.asarray(asset.bbox_max, dtype=float) - np.asarray(
        asset.bbox_min, dtype=float
    )
    if size.shape != (3,) or np.any(size <= 0):
        raise ValueError(f"asset {asset.object_id} has invalid measured bounds")
    return float(size[0]), float(size[1]), float(size[2])


def _axis_points(low: float, high: float, spacing: float) -> tuple[float, ...]:
    if high < low:
        return ()
    count = max(1, math.floor((high - low) / max(spacing, 0.05)) + 1)
    if count == 1:
        return ((low + high) / 2,)
    return tuple(float(value) for value in np.linspace(low, high, count))


def _point_in_polygon(
    point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
        length = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if abs(cross) <= 1e-8 and -1e-8 <= dot <= length + 1e-8:
            return True
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing > x:
                inside = not inside
        previous = current
    return inside


def _segment_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    value = np.asarray(point, dtype=float)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    delta = b - a
    if float(delta @ delta) <= 1e-12:
        return float(np.linalg.norm(value - a))
    t = float(np.clip(((value - a) @ delta) / (delta @ delta), 0.0, 1.0))
    return float(np.linalg.norm(value - (a + t * delta)))


@dataclass(frozen=True)
class _AcceptedSpace:
    width: float
    depth: float
    zones: dict[str, tuple[tuple[float, float], ...]]
    routes: tuple[tuple[tuple[tuple[float, float], ...], float], ...]
    story_positions: tuple[tuple[tuple[float, float], float], ...]

    @classmethod
    def from_stage_input(cls, stage_input: dict[str, Any]) -> _AcceptedSpace:
        request = stage_input["request"]
        width, _, depth = (float(value) for value in request["shell"]["dimensions_m"])

        def local(point: Iterable[float]) -> tuple[float, float]:
            x, y = point
            return float(x) - width / 2, float(y) - depth / 2

        zones = {
            str(item["zone_id"]): tuple(local(point) for point in item["polygon_xy_m"])
            for item in request["functional_zones"]
        }
        routes = tuple(
            (
                tuple(local(point) for point in item["points_xy_m"]),
                float(item["clear_width_m"]),
            )
            for item in request["circulation_routes"]
        )
        story = tuple(
            (local(item["position_m"][:2]), float(item["clear_radius_m"]))
            for item in request["story_positions"]
        )
        return cls(
            width=width, depth=depth, zones=zones, routes=routes, story_positions=story
        )

    def permits(
        self,
        point: tuple[float, float],
        zone_ids: Iterable[str],
        footprint_radius: float,
        *,
        protect_routes: bool,
    ) -> bool:
        polygons = tuple(self.zones[value] for value in zone_ids if value in self.zones)
        if not polygons or not any(
            _point_in_polygon(point, polygon) for polygon in polygons
        ):
            return False
        if protect_routes:
            for points, clear_width in self.routes:
                for start, end in pairwise(points):
                    if (
                        _segment_distance(point, start, end)
                        < clear_width / 2 + footprint_radius
                    ):
                        return False
            for position, clear_radius in self.story_positions:
                if math.dist(point, position) < clear_radius + footprint_radius:
                    return False
        return True


def _ordered_candidates(
    candidates: list[tuple[Any, ...]],
    operation: dict[str, Any],
    brief: dict[str, Any],
    instance_index: int,
    round_index: int,
) -> list[tuple[Any, ...]]:
    if operation.get("arrangement") == "clustered":
        candidates.sort(key=lambda item: sum(float(value) ** 2 for value in item[-2:]))
    else:
        candidates.sort(key=repr)
    if not candidates:
        return candidates
    identity = ":".join(
        (
            str(operation.get("operation_id")),
            str(brief.get("variant_id")),
            str(instance_index),
            str(round_index),
        )
    )
    offset = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % len(
        candidates
    )
    return candidates[offset:] + candidates[:offset]


class _PlacementAdapter:
    def __init__(
        self,
        stage_input: dict[str, Any],
        collision_checker: CollisionChecker | None = None,
    ):
        self.space = _AcceptedSpace.from_stage_input(stage_input)
        self.collision_checker = collision_checker or _native_collisions

    @staticmethod
    def _result(raw: str) -> str | None:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        object_id = str(payload.get("object_id") or "")
        return object_id if payload.get("success") is True and object_id else None

    def _accept_or_remove(self, scene: Any, raw: str) -> str | None:
        object_id = self._result(raw)
        if object_id is None:
            return None
        collisions = self.collision_checker(scene)
        if any(
            object_id in {str(item.object_a_id), str(item.object_b_id)}
            for item in collisions
        ):
            scene.remove_object(UniqueID(object_id))
            return None
        return object_id


class FloorPlacementAdapter(_PlacementAdapter):
    """Place furniture through ``FurnitureTools`` inside accepted zones."""

    def __init__(self, tool: Any, stage_input: dict[str, Any], collision_checker=None):
        super().__init__(stage_input, collision_checker)
        self.tool = tool
        self.tool.set_noise_profile(PlacementNoiseMode.PERFECT)

    def place(self, scene, asset, operation, brief, *, instance_index, round_index):
        width, depth, _ = _dimensions(asset)
        half_x, half_y = width / 2, depth / 2
        radius = math.hypot(half_x, half_y)
        xs = _axis_points(
            -self.space.width / 2 + half_x, self.space.width / 2 - half_x, width * 1.2
        )
        ys = _axis_points(
            -self.space.depth / 2 + half_y, self.space.depth / 2 - half_y, depth * 1.2
        )
        candidates = [
            (x, y)
            for x in xs
            for y in ys
            if self.space.permits(
                (x, y), operation["functional_zone_ids"], radius, protect_routes=True
            )
        ]
        for x, y in _ordered_candidates(
            candidates, operation, brief, instance_index, round_index
        ):
            raw = self.tool._add_furniture_to_scene_impl(
                str(asset.object_id), x, y, 0.0
            )
            accepted = self._accept_or_remove(scene, raw)
            if accepted is not None:
                return accepted
        return None


class SurfacePlacementAdapter(_PlacementAdapter):
    """Place dressing through a furniture-specific ``ManipulandTools`` instance."""

    def __init__(
        self,
        tool_factory: ManipulandToolFactory,
        stage_input: dict[str, Any],
        collision_checker=None,
    ):
        super().__init__(stage_input, collision_checker)
        self.tool_factory = tool_factory

    def place(self, scene, asset, operation, brief, *, instance_index, round_index):
        width, depth, _ = _dimensions(asset)
        supports = set(operation.get("support_role_ids", ()))
        candidates: list[tuple[Any, ...]] = []
        for owner_id, owner in sorted(
            scene.objects.items(), key=lambda item: str(item[0])
        ):
            role = str(owner.metadata.get("aether_role_id") or owner.name)
            if supports and role not in supports:
                continue
            if owner.metadata.get("aether_accepts_dressing") is False:
                continue
            for surface in sorted(
                owner.support_surfaces, key=lambda item: str(item.surface_id)
            ):
                minimum = np.asarray(surface.bounding_box_min, dtype=float)
                maximum = np.asarray(surface.bounding_box_max, dtype=float)
                xs = _axis_points(
                    minimum[0] + width / 2, maximum[0] - width / 2, width * 1.25
                )
                ys = _axis_points(
                    minimum[1] + depth / 2, maximum[1] - depth / 2, depth * 1.25
                )
                candidates.extend((owner_id, surface, x, y) for x in xs for y in ys)
        ordered = _ordered_candidates(
            candidates, operation, brief, instance_index, round_index
        )
        for owner_id, surface, x, y in ordered:
            surfaces = {str(surface.surface_id): surface}
            tool = self.tool_factory(owner_id, surfaces)
            tool.set_noise_profile(PlacementNoiseMode.PERFECT)
            raw = tool._place_manipuland_on_surface_impl(
                str(asset.object_id), str(surface.surface_id), x, y, 0.0
            )
            accepted = self._accept_or_remove(scene, raw)
            if accepted is not None:
                return accepted
        return None


class WallPlacementAdapter(_PlacementAdapter):
    """Place wall objects through ``WallTools`` while respecting apertures."""

    def __init__(self, tool: Any, stage_input: dict[str, Any], collision_checker=None):
        super().__init__(stage_input, collision_checker)
        self.tool = tool
        self.tool.set_noise_profile(PlacementNoiseMode.PERFECT)

    def place(self, scene, asset, operation, brief, *, instance_index, round_index):
        width, _, height = _dimensions(asset)
        candidates: list[tuple[Any, ...]] = []
        for surface_id, surface in sorted(self.tool.surfaces_by_id.items()):
            xs = _axis_points(width / 2, surface.length - width / 2, width * 1.25)
            zs = _axis_points(height / 2, surface.height - height / 2, height * 1.25)
            for x in xs:
                for z in zs:
                    valid, _ = surface.check_object_bounds(x, z, width, height)
                    if valid:
                        candidates.append((surface_id, x, z))
        for surface_id, x, z in _ordered_candidates(
            candidates, operation, brief, instance_index, round_index
        ):
            raw = self.tool._place_wall_object_impl(
                str(asset.object_id), surface_id, x, z, 0.0
            )
            accepted = self._accept_or_remove(scene, raw)
            if accepted is not None:
                return accepted
        return None


class CeilingPlacementAdapter(_PlacementAdapter):
    """Place overhead objects through ``CeilingTools`` in accepted zones."""

    def __init__(self, tool: Any, stage_input: dict[str, Any], collision_checker=None):
        super().__init__(stage_input, collision_checker)
        self.tool = tool
        self.tool.set_noise_profile(PlacementNoiseMode.PERFECT)

    def place(self, scene, asset, operation, brief, *, instance_index, round_index):
        width, depth, _ = _dimensions(asset)
        radius = math.hypot(width / 2, depth / 2)
        min_x, min_y, max_x, max_y = self.tool.room_bounds
        xs = _axis_points(min_x + width / 2, max_x - width / 2, width * 1.5)
        ys = _axis_points(min_y + depth / 2, max_y - depth / 2, depth * 1.5)
        candidates = [
            (x, y)
            for x in xs
            for y in ys
            if self.space.permits(
                (x, y), operation["functional_zone_ids"], radius, protect_routes=False
            )
        ]
        for x, y in _ordered_candidates(
            candidates, operation, brief, instance_index, round_index
        ):
            raw = self.tool._place_ceiling_object_impl(str(asset.object_id), x, y, 0.0)
            accepted = self._accept_or_remove(scene, raw)
            if accepted is not None:
                return accepted
        return None
