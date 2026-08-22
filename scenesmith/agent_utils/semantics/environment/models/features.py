"""Openings, repeated detail fields, and hero landmarks."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any, Mapping

from scenesmith.agent_utils.semantics.environment.models.common import (
    FORMATION_SURFACE_ROLES,
    DetailCollisionPolicy,
    DetailSurfaceRole,
    FormationType,
    HeroFeatureType,
    OpeningShape,
    OpeningTarget,
    _finite,
    _identifier,
    _point,
    _positive_point,
    _reject_unknown_fields,
    _strict_bool,
    _strict_int,
)
from scenesmith.agent_utils.structure.geometry_models.common import (
    GEOMETRY_TOLERANCE,
    Point2,
    Point3,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)


@dataclass(frozen=True)
class EnvironmentOpeningSpec:
    """A physical aperture connecting a chamber to sky or exterior."""

    opening_id: str
    region_id: str
    source_chamber_id: str
    target: OpeningTarget
    center: Point3
    normal: Point3
    size: Point2
    depth: float
    shape: OpeningShape = OpeningShape.ELLIPSE
    passable: bool = False
    visible: bool = True
    weather_exposed: bool = False

    def __post_init__(self) -> None:
        opening_id = _identifier(self.opening_id, "opening_id")
        object.__setattr__(self, "opening_id", opening_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self,
            "source_chamber_id",
            _identifier(self.source_chamber_id, "source_chamber_id"),
        )
        object.__setattr__(self, "target", OpeningTarget(self.target))
        object.__setattr__(
            self, "center", _point(self.center, "center", entity_id=opening_id)
        )
        normal = _point(self.normal, "normal", entity_id=opening_id)
        normal_length = math.sqrt(sum(component * component for component in normal))
        if normal_length <= GEOMETRY_TOLERANCE:
            raise GeometryValidationError(
                "invalid_opening_normal",
                "normal must have non-zero length",
                entity_id=opening_id,
            )
        object.__setattr__(
            self, "normal", tuple(component / normal_length for component in normal)
        )
        if len(self.size) != 2:
            raise GeometryValidationError(
                "invalid_opening_size",
                "size must contain width and height",
                entity_id=opening_id,
            )
        size = tuple(
            _finite(value, f"size[{axis}]", entity_id=opening_id)
            for axis, value in enumerate(self.size)
        )
        if any(value <= 0.0 for value in size):
            raise GeometryValidationError(
                "invalid_opening_size",
                "opening size must be positive",
                entity_id=opening_id,
            )
        object.__setattr__(self, "size", size)
        depth = _finite(self.depth, "depth", entity_id=opening_id)
        if depth <= 0.0:
            raise GeometryValidationError(
                "invalid_opening_depth",
                "opening depth must be positive",
                entity_id=opening_id,
            )
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "shape", OpeningShape(self.shape))
        object.__setattr__(
            self,
            "passable",
            _strict_bool(self.passable, "passable", entity_id=opening_id),
        )
        object.__setattr__(
            self,
            "visible",
            _strict_bool(self.visible, "visible", entity_id=opening_id),
        )
        object.__setattr__(
            self,
            "weather_exposed",
            _strict_bool(self.weather_exposed, "weather_exposed", entity_id=opening_id),
        )

    @property
    def sky_exposed(self) -> bool:
        return self.target == OpeningTarget.SKY

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.opening_id,
            "region_id": self.region_id,
            "source_chamber_id": self.source_chamber_id,
            "target": self.target.value,
            "center": list(self.center),
            "normal": list(self.normal),
            "size": list(self.size),
            "depth": self.depth,
            "shape": self.shape.value,
            "passable": self.passable,
            "visible": self.visible,
            "weather_exposed": self.weather_exposed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentOpeningSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "source_chamber_id",
                "target",
                "center",
                "normal",
                "size",
                "depth",
                "shape",
                "passable",
                "visible",
                "weather_exposed",
            },
            entity_id=data.get("id"),
        )
        return cls(
            opening_id=data["id"],
            region_id=data["region_id"],
            source_chamber_id=data["source_chamber_id"],
            target=OpeningTarget(data["target"]),
            center=tuple(data["center"]),
            normal=tuple(data["normal"]),
            size=tuple(data["size"]),
            depth=data["depth"],
            shape=OpeningShape(data.get("shape", "ellipse")),
            passable=data.get("passable", False),
            visible=data.get("visible", True),
            weather_exposed=data.get("weather_exposed", False),
        )


@dataclass(frozen=True)
class DetailFieldSpec:
    """A deterministic repeated geological-detail recipe."""

    field_id: str
    region_id: str
    target_chamber_id: str
    formation_type: FormationType
    surface_role: DetailSurfaceRole
    count: int
    min_size: Point3
    max_size: Point3
    seed: int
    protect_passage_network_ids: tuple[str, ...] = ()
    route_clearance: float = 1.0
    collision_policy: DetailCollisionPolicy = DetailCollisionPolicy.COARSE

    def __post_init__(self) -> None:
        field_id = _identifier(self.field_id, "field_id")
        object.__setattr__(self, "field_id", field_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self,
            "target_chamber_id",
            _identifier(self.target_chamber_id, "target_chamber_id"),
        )
        object.__setattr__(self, "formation_type", FormationType(self.formation_type))
        object.__setattr__(self, "surface_role", DetailSurfaceRole(self.surface_role))
        if self.surface_role not in FORMATION_SURFACE_ROLES[self.formation_type]:
            allowed = ", ".join(
                sorted(
                    role.value for role in FORMATION_SURFACE_ROLES[self.formation_type]
                )
            )
            raise GeometryValidationError(
                "unsupported_formation_surface",
                f"formation '{self.formation_type.value}' supports surface roles: "
                + allowed,
                entity_id=field_id,
            )
        count = _strict_int(self.count, "count", entity_id=field_id)
        if count <= 0:
            raise GeometryValidationError(
                "invalid_detail_count", "count must be positive", entity_id=field_id
            )
        object.__setattr__(self, "count", count)
        minimum = _positive_point(self.min_size, "min_size", entity_id=field_id)
        maximum = _positive_point(self.max_size, "max_size", entity_id=field_id)
        if any(minimum[axis] > maximum[axis] for axis in range(3)):
            raise GeometryValidationError(
                "invalid_detail_size",
                "min_size must not exceed max_size",
                entity_id=field_id,
            )
        object.__setattr__(self, "min_size", minimum)
        object.__setattr__(self, "max_size", maximum)
        object.__setattr__(
            self, "seed", _strict_int(self.seed, "seed", entity_id=field_id)
        )
        protected = tuple(
            sorted(
                _identifier(identifier, "protect_passage_network_id")
                for identifier in self.protect_passage_network_ids
            )
        )
        if len(protected) != len(set(protected)):
            raise GeometryValidationError(
                "duplicate_protected_network",
                "protected passage network IDs must be unique",
                entity_id=field_id,
            )
        object.__setattr__(self, "protect_passage_network_ids", protected)
        clearance = _finite(self.route_clearance, "route_clearance", entity_id=field_id)
        if clearance < 0.0:
            raise GeometryValidationError(
                "invalid_route_clearance",
                "route_clearance must not be negative",
                entity_id=field_id,
            )
        object.__setattr__(self, "route_clearance", clearance)
        object.__setattr__(
            self, "collision_policy", DetailCollisionPolicy(self.collision_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.field_id,
            "region_id": self.region_id,
            "target_chamber_id": self.target_chamber_id,
            "formation_type": self.formation_type.value,
            "surface_role": self.surface_role.value,
            "count": self.count,
            "min_size": list(self.min_size),
            "max_size": list(self.max_size),
            "seed": self.seed,
            "protect_passage_network_ids": list(self.protect_passage_network_ids),
            "route_clearance": self.route_clearance,
            "collision_policy": self.collision_policy.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetailFieldSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "target_chamber_id",
                "formation_type",
                "surface_role",
                "count",
                "min_size",
                "max_size",
                "seed",
                "protect_passage_network_ids",
                "route_clearance",
                "collision_policy",
            },
            entity_id=data.get("id"),
        )
        return cls(
            field_id=data["id"],
            region_id=data["region_id"],
            target_chamber_id=data["target_chamber_id"],
            formation_type=FormationType(data["formation_type"]),
            surface_role=DetailSurfaceRole(data["surface_role"]),
            count=data["count"],
            min_size=tuple(data["min_size"]),
            max_size=tuple(data["max_size"]),
            seed=data["seed"],
            protect_passage_network_ids=tuple(
                data.get("protect_passage_network_ids", [])
            ),
            route_clearance=data.get("route_clearance", 1.0),
            collision_policy=DetailCollisionPolicy(
                data.get("collision_policy", "coarse")
            ),
        )


@dataclass(frozen=True)
class HeroFeatureSpec:
    """One intentionally placed geological landmark."""

    feature_id: str
    region_id: str
    target_chamber_id: str
    feature_type: HeroFeatureType
    anchor: Point3
    size: Point3
    collision_policy: DetailCollisionPolicy = DetailCollisionPolicy.FULL
    semantic_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        feature_id = _identifier(self.feature_id, "feature_id")
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        object.__setattr__(
            self,
            "target_chamber_id",
            _identifier(self.target_chamber_id, "target_chamber_id"),
        )
        object.__setattr__(self, "feature_type", HeroFeatureType(self.feature_type))
        object.__setattr__(
            self, "anchor", _point(self.anchor, "anchor", entity_id=feature_id)
        )
        object.__setattr__(
            self, "size", _positive_point(self.size, "size", entity_id=feature_id)
        )
        object.__setattr__(
            self, "collision_policy", DetailCollisionPolicy(self.collision_policy)
        )
        object.__setattr__(
            self,
            "semantic_tags",
            frozenset(
                str(tag).strip() for tag in self.semantic_tags if str(tag).strip()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.feature_id,
            "region_id": self.region_id,
            "target_chamber_id": self.target_chamber_id,
            "feature_type": self.feature_type.value,
            "anchor": list(self.anchor),
            "size": list(self.size),
            "collision_policy": self.collision_policy.value,
            "semantic_tags": sorted(self.semantic_tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeroFeatureSpec":
        _reject_unknown_fields(
            data,
            {
                "id",
                "region_id",
                "target_chamber_id",
                "feature_type",
                "anchor",
                "size",
                "collision_policy",
                "semantic_tags",
            },
            entity_id=data.get("id"),
        )
        return cls(
            feature_id=data["id"],
            region_id=data["region_id"],
            target_chamber_id=data["target_chamber_id"],
            feature_type=HeroFeatureType(data["feature_type"]),
            anchor=tuple(data["anchor"]),
            size=tuple(data["size"]),
            collision_policy=DetailCollisionPolicy(
                data.get("collision_policy", "full")
            ),
            semantic_tags=frozenset(data.get("semantic_tags", [])),
        )
