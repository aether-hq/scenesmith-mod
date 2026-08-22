"""Validation errors and planar geometry guards."""

from __future__ import annotations

import math
import re

from typing import Any, Iterable, Sequence

from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Loop2,
    Point2,
    Point3,
)


class GeometryValidationError(ValueError):
    """Base error for structural-geometry validation failures."""

    def __init__(
        self, code: str, message: str, *, entity_id: str | None = None
    ) -> None:
        self.code = code
        self.entity_id = entity_id
        prefix = f"[{code}]"
        if entity_id is not None:
            prefix += f" {entity_id}:"
        super().__init__(f"{prefix} {message}")


class UnknownLevelError(GeometryValidationError):
    """A space or connector references a level that does not exist."""

    def __init__(self, level_id: str, *, entity_id: str | None = None) -> None:
        super().__init__(
            "unknown_level",
            f"level '{level_id}' is not defined",
            entity_id=entity_id,
        )


class UnknownConnectorEndpointError(GeometryValidationError):
    """A connector references a space that does not exist."""

    def __init__(self, space_id: str, *, connector_id: str) -> None:
        super().__init__(
            "unknown_connector_endpoint",
            f"space '{space_id}' is not defined",
            entity_id=connector_id,
        )


class InvalidFootprintError(GeometryValidationError):
    """A footprint loop is degenerate, intersects itself, or has invalid holes."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("invalid_footprint", message, entity_id=entity_id)


class InvalidTransformError(GeometryValidationError):
    """A structural transform contains non-finite coordinates."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("invalid_transform", message, entity_id=entity_id)


class UnsafeConnectorError(GeometryValidationError):
    """A connector violates its declared geometric or ergonomic constraints."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("unsafe_connector", message, entity_id=entity_id)


class UnsupportedGeometryError(GeometryValidationError):
    """The semantic input is valid but its compiler is not implemented yet."""

    def __init__(self, message: str, *, entity_id: str | None = None) -> None:
        super().__init__("unsupported_geometry", message, entity_id=entity_id)


def require_safe_identifier(value: str, label: str) -> str:
    """Normalize one globally usable file/model/semantic identifier."""

    if type(value) is not str:
        raise GeometryValidationError(
            "invalid_identifier", f"{label} must be a JSON string"
        )
    normalized = value.strip()
    if not normalized:
        raise GeometryValidationError("missing_id", f"{label} must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", normalized):
        raise GeometryValidationError(
            "invalid_identifier",
            f"{label} must use only letters, numbers, '.', '_', or '-' and must "
            "start with a letter or number",
            entity_id=normalized,
        )
    return normalized


_require_id = require_safe_identifier


def validate_global_identifiers(identifiers: Iterable[tuple[str, str]]) -> None:
    """Reject cross-category collisions in one authoritative scene namespace."""

    uses: dict[str, list[str]] = {}
    for identifier, category in identifiers:
        normalized = require_safe_identifier(identifier, f"{category}_id")
        uses.setdefault(normalized, []).append(category)
    collisions = {
        identifier: categories
        for identifier, categories in uses.items()
        if len(categories) > 1
    }
    if collisions:
        details = ", ".join(
            f"{identifier} ({'/'.join(categories)})"
            for identifier, categories in sorted(collisions.items())
        )
        raise GeometryValidationError(
            "duplicate_scene_id",
            "scene IDs must be globally unique; duplicates: " + details,
        )


def _finite(value: Any, label: str, *, entity_id: str | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidTransformError(
            f"{label} must be numeric; got {value!r}", entity_id=entity_id
        ) from exc
    if not math.isfinite(number):
        raise InvalidTransformError(
            f"{label} must be finite; got {value!r}", entity_id=entity_id
        )
    return number


def _point2(value: Sequence[Any], label: str) -> Point2:
    if len(value) != 2:
        raise InvalidFootprintError(f"{label} must have exactly 2 coordinates")
    return (_finite(value[0], f"{label}.x"), _finite(value[1], f"{label}.y"))


def _point3(
    value: Sequence[Any], label: str, *, entity_id: str | None = None
) -> Point3:
    if len(value) != 3:
        raise InvalidTransformError(
            f"{label} must have exactly 3 coordinates", entity_id=entity_id
        )
    return (
        _finite(value[0], f"{label}.x", entity_id=entity_id),
        _finite(value[1], f"{label}.y", entity_id=entity_id),
        _finite(value[2], f"{label}.z", entity_id=entity_id),
    )


def _signed_area(loop: Loop2) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(loop, loop[1:] + loop[:1])
    )


def _orientation(a: Point2, b: Point2, c: Point2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point: Point2, start: Point2, end: Point2) -> bool:
    if abs(_orientation(start, end, point)) > GEOMETRY_TOLERANCE:
        return False
    return (
        min(start[0], end[0]) - GEOMETRY_TOLERANCE
        <= point[0]
        <= max(start[0], end[0]) + GEOMETRY_TOLERANCE
        and min(start[1], end[1]) - GEOMETRY_TOLERANCE
        <= point[1]
        <= max(start[1], end[1]) + GEOMETRY_TOLERANCE
    )


def _segments_intersect(a: Point2, b: Point2, c: Point2, d: Point2) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if (
        (o1 > GEOMETRY_TOLERANCE and o2 < -GEOMETRY_TOLERANCE)
        or (o1 < -GEOMETRY_TOLERANCE and o2 > GEOMETRY_TOLERANCE)
    ) and (
        (o3 > GEOMETRY_TOLERANCE and o4 < -GEOMETRY_TOLERANCE)
        or (o3 < -GEOMETRY_TOLERANCE and o4 > GEOMETRY_TOLERANCE)
    ):
        return True

    return (
        (abs(o1) <= GEOMETRY_TOLERANCE and _point_on_segment(c, a, b))
        or (abs(o2) <= GEOMETRY_TOLERANCE and _point_on_segment(d, a, b))
        or (abs(o3) <= GEOMETRY_TOLERANCE and _point_on_segment(a, c, d))
        or (abs(o4) <= GEOMETRY_TOLERANCE and _point_on_segment(b, c, d))
    )


def _loop_edges(loop: Loop2) -> Iterable[tuple[Point2, Point2]]:
    return zip(loop, loop[1:] + loop[:1])


def _validate_simple_loop(loop: Loop2, label: str) -> None:
    if len(loop) < 3:
        raise InvalidFootprintError(f"{label} needs at least 3 vertices")

    for index, (start, end) in enumerate(_loop_edges(loop)):
        if math.dist(start, end) <= GEOMETRY_TOLERANCE:
            raise InvalidFootprintError(
                f"{label} edge {index} has zero length or repeated vertices"
            )

    edge_count = len(loop)
    edges = list(_loop_edges(loop))
    for first_index, (a, b) in enumerate(edges):
        for second_index in range(first_index + 1, edge_count):
            # Adjacent edges meet at a vertex by definition and are valid.
            if second_index in {
                first_index,
                (first_index + 1) % edge_count,
                (first_index - 1) % edge_count,
            }:
                continue
            c, d = edges[second_index]
            if _segments_intersect(a, b, c, d):
                raise InvalidFootprintError(
                    f"{label} self-intersects at edges {first_index} and "
                    f"{second_index}"
                )

    if abs(_signed_area(loop)) <= GEOMETRY_TOLERANCE:
        raise InvalidFootprintError(f"{label} has zero area")


def _loops_intersect(first: Loop2, second: Loop2) -> bool:
    return any(
        _segments_intersect(a, b, c, d)
        for a, b in _loop_edges(first)
        for c, d in _loop_edges(second)
    )


def _point_in_loop(point: Point2, loop: Loop2, *, include_boundary: bool) -> bool:
    for start, end in _loop_edges(loop):
        if _point_on_segment(point, start, end):
            return include_boundary

    inside = False
    x, y = point
    previous = loop[-1]
    for current in loop:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            x_crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_crossing > x:
                inside = not inside
        previous = current
    return inside


def _normalize_loop(
    points: Sequence[Sequence[Any]], *, counter_clockwise: bool, label: str
) -> Loop2:
    loop = tuple(
        _point2(point, f"{label}[{index}]") for index, point in enumerate(points)
    )
    # A repeated closing point is common in interchange formats. Normalize it
    # before strict repeated-edge validation.
    if len(loop) >= 2 and math.dist(loop[0], loop[-1]) <= GEOMETRY_TOLERANCE:
        loop = loop[:-1]
    _validate_simple_loop(loop, label)
    is_counter_clockwise = _signed_area(loop) > 0
    if is_counter_clockwise != counter_clockwise:
        loop = tuple(reversed(loop))
    return loop
