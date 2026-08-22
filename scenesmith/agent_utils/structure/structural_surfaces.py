"""Surface-native placement queries for parametric and freeform structures."""

from __future__ import annotations

import json
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scenesmith.agent_utils.structure.compiler.models import (
    CompiledSurfacePatch,
    TriangleMesh,
)
from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point2,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    StructuralSurface,
    SurfaceRole,
    Transform3D,
)


def _add(a: Point3, b: Point3) -> Point3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _subtract(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(vector: Point3, factor: float) -> Point3:
    return tuple(value * factor for value in vector)  # type: ignore[return-value]


def _dot(a: Point3, b: Point3) -> float:
    return sum(first * second for first, second in zip(a, b))


def _cross(a: Point3, b: Point3) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(vector: Point3) -> Point3:
    length = math.sqrt(_dot(vector, vector))
    if length <= GEOMETRY_TOLERANCE:
        raise ValueError("cannot normalize a zero-length vector")
    return _scale(vector, 1.0 / length)


def _rotate_rpy(vector: Point3, rotation_rpy: Point3) -> Point3:
    """Apply intrinsic roll/pitch/yaw using the compiler's XYZ convention."""

    roll, pitch, yaw = rotation_rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y, z = vector
    roll_point = (x, cr * y - sr * z, sr * y + cr * z)
    pitch_point = (
        cp * roll_point[0] + sp * roll_point[2],
        roll_point[1],
        -sp * roll_point[0] + cp * roll_point[2],
    )
    return (
        cy * pitch_point[0] - sy * pitch_point[1],
        sy * pitch_point[0] + cy * pitch_point[1],
        pitch_point[2],
    )


def _point_on_segment(point: Point2, start: Point2, end: Point2) -> bool:
    cross = (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (
        end[0] - start[0]
    )
    if abs(cross) > 1e-8:
        return False
    dot = (point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (
        end[1] - start[1]
    )
    squared_length = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
    return -1e-8 <= dot <= squared_length + 1e-8


def _contains_loop(point: Point2, loop: Sequence[Point2]) -> bool:
    for start, end in zip(loop, loop[1:] + loop[:1]):
        if _point_on_segment(point, start, end):
            return True
    inside = False
    previous = loop[-1]
    for current in loop:
        if (previous[1] > point[1]) != (current[1] > point[1]):
            crossing_x = previous[0] + (point[1] - previous[1]) * (
                current[0] - previous[0]
            ) / (current[1] - previous[1])
            if crossing_x > point[0]:
                inside = not inside
        previous = current
    return inside


def _segment_distance(point: Point2, start: Point2, end: Point2) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    squared_length = dx * dx + dy * dy
    if squared_length <= GEOMETRY_TOLERANCE:
        return math.dist(point, start)
    parameter = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squared_length,
        ),
    )
    closest = (start[0] + parameter * dx, start[1] + parameter * dy)
    return math.dist(point, closest)


@dataclass(frozen=True)
class SurfacePose:
    """Position and orthonormal tangent frame for placing an object."""

    surface_id: str
    position: Point3
    normal: Point3
    tangent_x: Point3
    tangent_y: Point3
    clearance_to_edge: float


@dataclass(frozen=True)
class ClearanceResult:
    """Agent-sized vertical/edge clearance at one structural XY location.

    This is intentionally a conservative local predicate.  Route validation
    samples it along a centerline; it does not claim to replace full swept-
    volume collision checking in the simulator.
    """

    fits: bool
    support: SurfacePose | None
    overhead: SurfacePose | None
    vertical_clearance: float | None
    edge_clearance: float
    reasons: tuple[str, ...]


class SurfaceQuery:
    """Plane-local containment, pose, and edge-clearance queries for one patch."""

    def __init__(self, patch: CompiledSurfacePatch) -> None:
        self.patch = patch
        self.origin = patch.boundary[0]
        self.normal = _normalize(patch.normal)
        # Project world +X into the plane so yaw remains globally intuitive.
        # A near-X-facing wall falls back to projected world +Y.
        tangent = _subtract((1.0, 0.0, 0.0), _scale(self.normal, self.normal[0]))
        if _dot(tangent, tangent) <= GEOMETRY_TOLERANCE:
            tangent = _subtract((0.0, 1.0, 0.0), _scale(self.normal, self.normal[1]))
        self.tangent_u = _normalize(tangent)
        self.tangent_v = _normalize(_cross(self.normal, self.tangent_u))
        self.outer = tuple(self.project(point) for point in patch.boundary)
        self.holes = tuple(
            self._project_xy_loop(hole)
            for hole in patch.surface.metadata.get("holes", [])
        )

    def _project_xy_loop(self, loop: Sequence[Sequence[float]]) -> tuple[Point2, ...]:
        result: list[Point2] = []
        for x, y in loop:
            if abs(self.normal[2]) <= GEOMETRY_TOLERANCE:
                raise ValueError("XY-authored holes require a non-vertical surface")
            z = (
                self.origin[2]
                - (
                    self.normal[0] * (float(x) - self.origin[0])
                    + self.normal[1] * (float(y) - self.origin[1])
                )
                / self.normal[2]
            )
            result.append(self.project((float(x), float(y), z)))
        return tuple(result)

    def project(self, point: Point3) -> Point2:
        relative = _subtract(point, self.origin)
        return (_dot(relative, self.tangent_u), _dot(relative, self.tangent_v))

    def unproject(self, point: Point2) -> Point3:
        return _add(
            self.origin,
            _add(_scale(self.tangent_u, point[0]), _scale(self.tangent_v, point[1])),
        )

    def contains_uv(self, point: Point2) -> bool:
        return _contains_loop(point, self.outer) and not any(
            _contains_loop(point, hole) for hole in self.holes
        )

    def contains_point(self, point: Point3, *, plane_tolerance: float = 1e-6) -> bool:
        if abs(_dot(_subtract(point, self.origin), self.normal)) > plane_tolerance:
            return False
        return self.contains_uv(self.project(point))

    def clearance_at_uv(self, point: Point2) -> float:
        if not self.contains_uv(point):
            return 0.0
        loops = (self.outer, *self.holes)
        return min(
            _segment_distance(point, start, end)
            for loop in loops
            for start, end in zip(loop, loop[1:] + loop[:1])
        )

    def pose_at_uv(self, point: Point2, *, yaw: float = 0.0) -> SurfacePose:
        if not self.contains_uv(point):
            raise ValueError(
                f"point {point} is outside surface '{self.patch.surface.surface_id}'"
            )
        cosine, sine = math.cos(yaw), math.sin(yaw)
        tangent_x = _add(_scale(self.tangent_u, cosine), _scale(self.tangent_v, sine))
        tangent_y = _add(_scale(self.tangent_u, -sine), _scale(self.tangent_v, cosine))
        return SurfacePose(
            surface_id=self.patch.surface.surface_id,
            position=self.unproject(point),
            normal=self.normal,
            tangent_x=tangent_x,
            tangent_y=tangent_y,
            clearance_to_edge=self.clearance_at_uv(point),
        )


class StructuralSurfaceIndex:
    """Queries all structural patches by role and 3D location."""

    def __init__(self, patches: Iterable[CompiledSurfacePatch]) -> None:
        self.queries = tuple(SurfaceQuery(patch) for patch in patches)

    def by_role(self, role: SurfaceRole) -> tuple[SurfaceQuery, ...]:
        return tuple(
            query for query in self.queries if role in query.patch.surface.roles
        )

    def support_pose(
        self,
        x: float,
        y: float,
        *,
        reference_z: float | None = None,
        max_drop: float | None = None,
        max_slope_degrees: float = 35.0,
        yaw: float = 0.0,
    ) -> SurfacePose | None:
        """Find the highest compatible support under an XY/reference-Z point."""

        candidates: list[SurfacePose] = []
        minimum_normal_z = math.cos(math.radians(max_slope_degrees))
        for query in self.by_role(SurfaceRole.SUPPORT):
            if query.normal[2] < minimum_normal_z:
                continue
            if abs(query.normal[2]) <= GEOMETRY_TOLERANCE:
                continue
            z = (
                query.origin[2]
                - (
                    query.normal[0] * (x - query.origin[0])
                    + query.normal[1] * (y - query.origin[1])
                )
                / query.normal[2]
            )
            if reference_z is not None:
                drop = reference_z - z
                if drop < -1e-6 or (max_drop is not None and drop > max_drop):
                    continue
            point = (x, y, z)
            if not query.contains_point(point):
                continue
            candidates.append(query.pose_at_uv(query.project(point), yaw=yaw))
        return max(candidates, key=lambda pose: pose.position[2], default=None)

    def overhead_pose(
        self,
        x: float,
        y: float,
        *,
        reference_z: float | None = None,
        max_rise: float | None = None,
        yaw: float = 0.0,
    ) -> SurfacePose | None:
        """Find the lowest overhead patch above an XY/reference-Z point."""

        candidates: list[SurfacePose] = []
        for query in self.by_role(SurfaceRole.OVERHEAD):
            if abs(query.normal[2]) <= GEOMETRY_TOLERANCE:
                continue
            z = (
                query.origin[2]
                - (
                    query.normal[0] * (x - query.origin[0])
                    + query.normal[1] * (y - query.origin[1])
                )
                / query.normal[2]
            )
            if reference_z is not None:
                rise = z - reference_z
                if rise < -1e-6 or (max_rise is not None and rise > max_rise):
                    continue
            point = (x, y, z)
            if not query.contains_point(point):
                continue
            candidates.append(query.pose_at_uv(query.project(point), yaw=yaw))
        return min(candidates, key=lambda pose: pose.position[2], default=None)

    def clearance_at(
        self,
        x: float,
        y: float,
        *,
        agent_height: float,
        agent_radius: float = 0.0,
        reference_z: float | None = None,
        max_drop: float | None = None,
        max_support_slope_degrees: float = 35.0,
        require_overhead: bool = False,
    ) -> ClearanceResult:
        """Evaluate conservative local support, headroom, and edge clearance."""

        if not math.isfinite(agent_height) or agent_height <= 0:
            raise ValueError("agent_height must be finite and positive")
        if not math.isfinite(agent_radius) or agent_radius < 0:
            raise ValueError("agent_radius must be finite and non-negative")

        support = self.support_pose(
            x,
            y,
            reference_z=reference_z,
            max_drop=max_drop,
            max_slope_degrees=max_support_slope_degrees,
        )
        if support is None:
            return ClearanceResult(
                fits=False,
                support=None,
                overhead=None,
                vertical_clearance=None,
                edge_clearance=0.0,
                reasons=("no_compatible_support",),
            )

        overhead = self.overhead_pose(x, y, reference_z=support.position[2])
        vertical_clearance = (
            overhead.position[2] - support.position[2] if overhead is not None else None
        )
        reasons: list[str] = []
        if support.clearance_to_edge + GEOMETRY_TOLERANCE < agent_radius:
            reasons.append("insufficient_edge_clearance")
        if overhead is None:
            if require_overhead:
                reasons.append("no_overhead_surface")
        elif vertical_clearance + GEOMETRY_TOLERANCE < agent_height:
            reasons.append("insufficient_headroom")

        return ClearanceResult(
            fits=not reasons,
            support=support,
            overhead=overhead,
            vertical_clearance=vertical_clearance,
            edge_clearance=support.clearance_to_edge,
            reasons=tuple(reasons),
        )


def transform_surface_patches(
    patches: Iterable[CompiledSurfacePatch], transform: Transform3D
) -> tuple[CompiledSurfacePatch, ...]:
    """Transform explicit patch geometry into a parent structural frame."""

    transformed: list[CompiledSurfacePatch] = []
    for patch in patches:
        boundary = tuple(
            _add(
                _rotate_rpy(point, transform.rotation_rpy),
                transform.translation,
            )
            for point in patch.boundary
        )
        metadata = dict(patch.surface.metadata)
        if metadata.get("holes"):
            if abs(patch.normal[2]) <= GEOMETRY_TOLERANCE:
                raise ValueError("XY-authored holes require a non-vertical surface")
            local_origin = patch.boundary[0]
            transformed_holes = []
            for hole in metadata["holes"]:
                transformed_hole = []
                for x, y in hole:
                    z = (
                        local_origin[2]
                        - (
                            patch.normal[0] * (float(x) - local_origin[0])
                            + patch.normal[1] * (float(y) - local_origin[1])
                        )
                        / patch.normal[2]
                    )
                    world_point = _add(
                        _rotate_rpy((float(x), float(y), z), transform.rotation_rpy),
                        transform.translation,
                    )
                    transformed_hole.append([world_point[0], world_point[1]])
                transformed_holes.append(transformed_hole)
            metadata["holes"] = transformed_holes
        surface = StructuralSurface(
            surface_id=patch.surface.surface_id,
            roles=patch.surface.roles,
            source_id=patch.surface.source_id,
            transform=patch.surface.transform,
            geometry_ref=patch.surface.geometry_ref,
            metadata=metadata,
        )
        transformed.append(
            CompiledSurfacePatch(
                surface=surface,
                boundary=boundary,
                normal=_rotate_rpy(patch.normal, transform.rotation_rpy),
            )
        )
    return tuple(transformed)


def load_surface_patches(path: Path | str) -> tuple[CompiledSurfacePatch, ...]:
    """Load explicit patch boundaries/normals from a compiler sidecar."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("surface_encoding") == "triangle_mesh_v1":
        mesh_data = data["surface_mesh"]
        mesh = TriangleMesh(
            vertices=tuple(tuple(point) for point in mesh_data["vertices"]),
            triangles=tuple(tuple(triangle) for triangle in mesh_data["triangles"]),
        )
        group_by_triangle = {
            triangle_index: group_name
            for group_name, indices in data["triangle_groups"].items()
            for triangle_index in indices
        }
        metadata_by_triangle: list[dict] = [{} for _ in range(len(mesh.triangles))]
        for run in data.get("surface_metadata_runs", []):
            for triangle_index in range(run["start"], run["end"]):
                metadata_by_triangle[triangle_index] = dict(run["metadata"])
        structure_id = data["structure_id"]
        return tuple(
            CompiledSurfacePatch(
                surface=StructuralSurface(
                    surface_id=(
                        f"{structure_id}_{group_by_triangle[triangle_index]}_"
                        f"{triangle_index:06d}"
                    ),
                    roles=frozenset(
                        SurfaceRole(role)
                        for role in data["surface_roles"][
                            group_by_triangle[triangle_index]
                        ]
                    ),
                    source_id=structure_id,
                    geometry_ref=f"triangle:{triangle_index}",
                    metadata=metadata_by_triangle[triangle_index],
                ),
                boundary=tuple(mesh.vertices[index] for index in triangle),
                normal=mesh.triangle_normal(triangle_index),
            )
            for triangle_index, triangle in enumerate(mesh.triangles)
        )
    return tuple(
        CompiledSurfacePatch(
            surface=StructuralSurface.from_dict(surface),
            boundary=tuple(tuple(point) for point in surface["boundary"]),
            normal=tuple(surface["normal"]),
        )
        for surface in data["surfaces"]
    )
