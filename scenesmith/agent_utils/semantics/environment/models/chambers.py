"""Regions, spatial bounds, and cavern chamber declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scenesmith.agent_utils.semantics.environment.models.common import (
    _REFERENCE_TOLERANCE,
    CavernShape,
    EnvironmentKind,
    SemanticTransform3D,
    _identifier,
    _point,
    _positive_point,
    _reject_unknown_fields,
    _rotate_rpy,
    _strict_int,
)
from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)


@dataclass(frozen=True)
class Bounds3D:
    """Conservative axis-aligned compilation bounds in scene coordinates."""

    minimum: Point3
    maximum: Point3

    def __post_init__(self) -> None:
        minimum = _point(self.minimum, "minimum")
        maximum = _point(self.maximum, "maximum")
        if any(
            maximum[axis] - minimum[axis] <= GEOMETRY_TOLERANCE for axis in range(3)
        ):
            raise GeometryValidationError(
                "invalid_bounds", "bounds maximum must exceed minimum on every axis"
            )
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(
        self, point: Point3, *, tolerance: float = _REFERENCE_TOLERANCE
    ) -> bool:
        return all(
            self.minimum[axis] - tolerance
            <= point[axis]
            <= self.maximum[axis] + tolerance
            for axis in range(3)
        )

    def contains_box(
        self,
        minimum: Point3,
        maximum: Point3,
        *,
        tolerance: float = _REFERENCE_TOLERANCE,
    ) -> bool:
        return self.contains(minimum, tolerance=tolerance) and self.contains(
            maximum, tolerance=tolerance
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"minimum": list(self.minimum), "maximum": list(self.maximum)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Bounds3D":
        _reject_unknown_fields(data, {"minimum", "maximum", "min", "max"})
        minimum = data.get("minimum", data.get("min"))
        maximum = data.get("maximum", data.get("max"))
        if minimum is None or maximum is None:
            raise GeometryValidationError(
                "invalid_bounds", "bounds require minimum/min and maximum/max"
            )
        return cls(tuple(minimum), tuple(maximum))


@dataclass(frozen=True)
class EnvironmentRegionSpec:
    """Owning region and compilation policy for related semantic geometry."""

    region_id: str
    kind: EnvironmentKind
    bounds: Bounds3D
    transform: SemanticTransform3D = field(default_factory=SemanticTransform3D)
    material_context: Mapping[str, str] = field(default_factory=dict)
    detail_seed: int = 0
    chunk_policy: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(self, "kind", EnvironmentKind(self.kind))
        object.__setattr__(self, "material_context", dict(self.material_context))
        object.__setattr__(self, "chunk_policy", dict(self.chunk_policy))
        object.__setattr__(
            self,
            "detail_seed",
            _strict_int(self.detail_seed, "detail_seed", entity_id=self.region_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "kind": self.kind.value,
            "bounds": self.bounds.to_dict(),
            "transform": self.transform.to_dict(),
            "material_context": dict(sorted(self.material_context.items())),
            "detail_seed": self.detail_seed,
            "chunk_policy": dict(sorted(self.chunk_policy.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentRegionSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "kind",
                "bounds",
                "transform",
                "material_context",
                "detail_seed",
                "chunk_policy",
            },
            entity_id=data.get("id"),
        )
        return cls(
            region_id=data["id"],
            kind=EnvironmentKind(data["kind"]),
            bounds=Bounds3D.from_dict(data["bounds"]),
            transform=SemanticTransform3D.from_dict(data.get("transform")),
            material_context=data.get("material_context", {}),
            detail_seed=data.get("detail_seed", 0),
            chunk_policy=data.get("chunk_policy", {}),
        )


@dataclass(frozen=True)
class CavernChamberSpec:
    """One authored open volume within a surrounding substrate."""

    chamber_id: str
    region_id: str
    center: Point3
    size: Point3
    shape: CavernShape = CavernShape.ELLIPSOID
    orientation_rpy: Point3 = (0.0, 0.0, 0.0)
    substrate_id: str | None = None
    semantic_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        chamber_id = _identifier(self.chamber_id, "chamber_id")
        object.__setattr__(self, "chamber_id", chamber_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self, "center", _point(self.center, "center", entity_id=chamber_id)
        )
        object.__setattr__(
            self, "size", _positive_point(self.size, "size", entity_id=chamber_id)
        )
        object.__setattr__(self, "shape", CavernShape(self.shape))
        object.__setattr__(
            self,
            "orientation_rpy",
            _point(self.orientation_rpy, "orientation_rpy", entity_id=chamber_id),
        )
        object.__setattr__(
            self,
            "semantic_tags",
            frozenset(
                str(tag).strip() for tag in self.semantic_tags if str(tag).strip()
            ),
        )
        if self.substrate_id is not None:
            object.__setattr__(
                self, "substrate_id", _identifier(self.substrate_id, "substrate_id")
            )

    @property
    def bounds(self) -> Bounds3D:
        half = tuple(component / 2.0 for component in self.size)
        axes = tuple(
            _rotate_rpy(axis, self.orientation_rpy)
            for axis in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        extents = tuple(
            sum(
                abs(axes[local_axis][world_axis]) * half[local_axis]
                for local_axis in range(3)
            )
            for world_axis in range(3)
        )
        return Bounds3D(
            tuple(self.center[axis] - extents[axis] for axis in range(3)),
            tuple(self.center[axis] + extents[axis] for axis in range(3)),
        )

    def contains(self, point: Point3, *, tolerance: float = 1e-6) -> bool:
        """Return whether a region-local point lies inside the chamber volume."""

        offset = tuple(point[axis] - self.center[axis] for axis in range(3))
        axes = tuple(
            _rotate_rpy(axis, self.orientation_rpy)
            for axis in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        local = tuple(
            sum(offset[axis] * basis[axis] for axis in range(3)) for basis in axes
        )
        exponent = 2 if self.shape == CavernShape.ELLIPSOID else 4
        value = sum(
            abs(local[axis] / (self.size[axis] / 2.0)) ** exponent for axis in range(3)
        )
        return value <= 1.0 + tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chamber_id,
            "region_id": self.region_id,
            "center": list(self.center),
            "size": list(self.size),
            "shape": self.shape.value,
            "orientation_rpy": list(self.orientation_rpy),
            "substrate_id": self.substrate_id,
            "semantic_tags": sorted(self.semantic_tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CavernChamberSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "center",
                "size",
                "shape",
                "orientation_rpy",
                "substrate_id",
                "semantic_tags",
            },
            entity_id=data.get("id"),
        )
        return cls(
            chamber_id=data["id"],
            region_id=data["region_id"],
            center=tuple(data["center"]),
            size=tuple(data["size"]),
            shape=CavernShape(data.get("shape", "ellipsoid")),
            orientation_rpy=tuple(data.get("orientation_rpy", (0.0, 0.0, 0.0))),
            substrate_id=data.get("substrate_id"),
            semantic_tags=frozenset(data.get("semantic_tags", [])),
        )
