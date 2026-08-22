"""Canonical semantic environment aggregate and reference validation."""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scenesmith.agent_utils.semantics.environment.models.chambers import (
    CavernChamberSpec,
    EnvironmentRegionSpec,
)
from scenesmith.agent_utils.semantics.environment.models.common import (
    MAX_DETAIL_INSTANCES_PER_FIELD,
    MAX_DETAIL_INSTANCES_PER_SCENE,
    MAX_PASSAGE_SEGMENTS_PER_SCENE,
    MAX_PATH_POINTS_PER_SEGMENT,
    SEMANTIC_ENVIRONMENT_SCHEMA_VERSION,
    CavernShape,
    EnvironmentKind,
    PassageFloorMode,
    _reject_unknown_fields,
    _strict_int,
    _unique_ids,
)
from scenesmith.agent_utils.semantics.environment.models.features import (
    DetailFieldSpec,
    EnvironmentOpeningSpec,
    HeroFeatureSpec,
)
from scenesmith.agent_utils.semantics.environment.models.passages import (
    PassageNetworkSpec,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    validate_global_identifiers,
)


@dataclass(frozen=True)
class SemanticEnvironmentSpec:
    """Canonical semantic graph for one or more environment regions."""

    regions: tuple[EnvironmentRegionSpec, ...]
    chambers: tuple[CavernChamberSpec, ...] = ()
    passage_networks: tuple[PassageNetworkSpec, ...] = ()
    openings: tuple[EnvironmentOpeningSpec, ...] = ()
    detail_fields: tuple[DetailFieldSpec, ...] = ()
    hero_features: tuple[HeroFeatureSpec, ...] = ()
    schema_version: int = SEMANTIC_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _strict_int(self.schema_version, "schema_version")
        if schema_version != SEMANTIC_ENVIRONMENT_SCHEMA_VERSION:
            raise GeometryValidationError(
                "unsupported_environment_schema",
                f"schema_version must be {SEMANTIC_ENVIRONMENT_SCHEMA_VERSION}",
            )
        regions = tuple(sorted(self.regions, key=lambda item: item.region_id))
        chambers = tuple(sorted(self.chambers, key=lambda item: item.chamber_id))
        networks = tuple(
            sorted(self.passage_networks, key=lambda item: item.network_id)
        )
        openings = tuple(sorted(self.openings, key=lambda item: item.opening_id))
        detail_fields = tuple(
            sorted(self.detail_fields, key=lambda item: item.field_id)
        )
        hero_features = tuple(
            sorted(self.hero_features, key=lambda item: item.feature_id)
        )
        if not regions:
            raise GeometryValidationError(
                "empty_environment", "an environment requires at least one region"
            )
        _unique_ids(regions, "region_id", "environment_region")
        _unique_ids(chambers, "chamber_id", "cavern_chamber")
        _unique_ids(networks, "network_id", "passage_network")
        _unique_ids(openings, "opening_id", "environment_opening")
        _unique_ids(detail_fields, "field_id", "detail_field")
        _unique_ids(hero_features, "feature_id", "hero_feature")
        identifiers: list[tuple[str, str]] = [
            *((item.region_id, "environment_region") for item in regions),
            *((item.chamber_id, "cavern_chamber") for item in chambers),
            *((item.network_id, "passage_network") for item in networks),
            *((item.opening_id, "environment_opening") for item in openings),
            *((item.field_id, "detail_field") for item in detail_fields),
            *((item.feature_id, "hero_feature") for item in hero_features),
            *(
                (item.junction_id, "passage_junction")
                for network in networks
                for item in network.junctions
            ),
            *(
                (item.segment_id, "passage_segment")
                for network in networks
                for item in network.segments
            ),
            *(
                (f"{item.field_id}_{index:04d}", "detail_instance")
                for item in detail_fields
                for index in range(item.count)
            ),
        ]
        try:
            validate_global_identifiers(identifiers)
        except GeometryValidationError as exc:
            if exc.code != "duplicate_scene_id":
                raise
            raise GeometryValidationError(
                "duplicate_semantic_id",
                str(exc).split(": ", 1)[-1],
            ) from exc
        segment_count = sum(len(network.segments) for network in networks)
        if segment_count > MAX_PASSAGE_SEGMENTS_PER_SCENE:
            raise GeometryValidationError(
                "semantic_budget_exceeded",
                f"scene has {segment_count} passage segments; budget is "
                f"{MAX_PASSAGE_SEGMENTS_PER_SCENE}",
            )
        for network in networks:
            for segment in network.segments:
                if len(segment.path) > MAX_PATH_POINTS_PER_SEGMENT:
                    raise GeometryValidationError(
                        "semantic_budget_exceeded",
                        f"passage path has {len(segment.path)} points; budget is "
                        f"{MAX_PATH_POINTS_PER_SEGMENT}",
                        entity_id=segment.segment_id,
                    )
        for field_spec in detail_fields:
            if field_spec.count > MAX_DETAIL_INSTANCES_PER_FIELD:
                raise GeometryValidationError(
                    "semantic_budget_exceeded",
                    f"detail count {field_spec.count} exceeds per-field budget "
                    f"{MAX_DETAIL_INSTANCES_PER_FIELD}",
                    entity_id=field_spec.field_id,
                )
        total_details = sum(field_spec.count for field_spec in detail_fields)
        if total_details > MAX_DETAIL_INSTANCES_PER_SCENE:
            raise GeometryValidationError(
                "semantic_budget_exceeded",
                f"scene requests {total_details} detail instances; budget is "
                f"{MAX_DETAIL_INSTANCES_PER_SCENE}",
            )
        region_by_id = {item.region_id: item for item in regions}
        chamber_by_id = {item.chamber_id: item for item in chambers}
        network_by_id = {item.network_id: item for item in networks}
        opening_by_id = {item.opening_id: item for item in openings}
        for chamber in chambers:
            if chamber.shape not in {
                CavernShape.ELLIPSOID,
                CavernShape.SUPERELLIPSOID,
            }:
                raise GeometryValidationError(
                    "unsupported_chamber_shape",
                    f"chamber shape '{chamber.shape.value}' is not supported by the "
                    "semantic compiler",
                    entity_id=chamber.chamber_id,
                )
            region = region_by_id.get(chamber.region_id)
            if region is None:
                raise GeometryValidationError(
                    "unknown_environment_region",
                    f"unknown region '{chamber.region_id}'",
                    entity_id=chamber.chamber_id,
                )
            chamber_bounds = chamber.bounds
            if not region.bounds.contains_box(
                chamber_bounds.minimum, chamber_bounds.maximum
            ):
                raise GeometryValidationError(
                    "environment_bounds_exceeded",
                    "chamber exceeds its region's conservative bounds",
                    entity_id=chamber.chamber_id,
                )
        for network in networks:
            region = region_by_id.get(network.region_id)
            if region is None:
                raise GeometryValidationError(
                    "unknown_environment_region",
                    f"unknown region '{network.region_id}'",
                    entity_id=network.network_id,
                )
            for junction in network.junctions:
                if not region.bounds.contains(junction.position):
                    raise GeometryValidationError(
                        "environment_bounds_exceeded",
                        "junction is outside its region bounds",
                        entity_id=junction.junction_id,
                    )
                if junction.chamber_id is not None:
                    chamber = chamber_by_id.get(junction.chamber_id)
                    if chamber is None:
                        raise GeometryValidationError(
                            "unknown_cavern_chamber",
                            f"unknown chamber '{junction.chamber_id}'",
                            entity_id=junction.junction_id,
                        )
                    if chamber.region_id != network.region_id:
                        raise GeometryValidationError(
                            "cross_region_junction",
                            "junction and referenced chamber must share a region",
                            entity_id=junction.junction_id,
                        )
                    if chamber.shape in {
                        CavernShape.ELLIPSOID,
                        CavernShape.SUPERELLIPSOID,
                    } and not chamber.contains(junction.position):
                        raise GeometryValidationError(
                            "disjoint_chamber_junction",
                            "junction position does not overlap its referenced chamber",
                            entity_id=junction.junction_id,
                        )
                if junction.opening_id is not None:
                    opening = opening_by_id.get(junction.opening_id)
                    if opening is None:
                        raise GeometryValidationError(
                            "unknown_environment_opening",
                            f"unknown opening '{junction.opening_id}'",
                            entity_id=junction.junction_id,
                        )
                    if opening.region_id != network.region_id:
                        raise GeometryValidationError(
                            "cross_region_junction",
                            "junction and referenced opening must share a region",
                            entity_id=junction.junction_id,
                        )
            for segment in network.segments:
                if segment.floor_mode == PassageFloorMode.STEPS:
                    raise GeometryValidationError(
                        "unsupported_passage_floor_mode",
                        "stepped passage floors are not supported by the semantic compiler",
                        entity_id=segment.segment_id,
                    )
                if any(not region.bounds.contains(point) for point in segment.path):
                    raise GeometryValidationError(
                        "environment_bounds_exceeded",
                        "passage path is outside its region bounds",
                        entity_id=segment.segment_id,
                    )

        for opening in openings:
            chamber = chamber_by_id.get(opening.source_chamber_id)
            if opening.region_id not in region_by_id:
                raise GeometryValidationError(
                    "unknown_environment_region",
                    f"unknown region '{opening.region_id}'",
                    entity_id=opening.opening_id,
                )
            if chamber is None:
                raise GeometryValidationError(
                    "unknown_cavern_chamber",
                    f"unknown chamber '{opening.source_chamber_id}'",
                    entity_id=opening.opening_id,
                )
            if chamber.region_id != opening.region_id:
                raise GeometryValidationError(
                    "cross_region_opening",
                    "opening and source chamber must share a region",
                    entity_id=opening.opening_id,
                )
            if chamber.shape in {
                CavernShape.ELLIPSOID,
                CavernShape.SUPERELLIPSOID,
            } and not chamber.contains(opening.center, tolerance=0.1):
                raise GeometryValidationError(
                    "disjoint_chamber_opening",
                    "opening center does not overlap its source chamber",
                    entity_id=opening.opening_id,
                )
        for field_spec in detail_fields:
            chamber = chamber_by_id.get(field_spec.target_chamber_id)
            if chamber is None or chamber.region_id != field_spec.region_id:
                raise GeometryValidationError(
                    "unknown_detail_target",
                    "detail target chamber is missing or belongs to another region",
                    entity_id=field_spec.field_id,
                )
            for network_id in field_spec.protect_passage_network_ids:
                network = network_by_id.get(network_id)
                if network is None or network.region_id != field_spec.region_id:
                    raise GeometryValidationError(
                        "unknown_protected_network",
                        f"protected network '{network_id}' is missing or cross-region",
                        entity_id=field_spec.field_id,
                    )
        for feature in hero_features:
            chamber = chamber_by_id.get(feature.target_chamber_id)
            if chamber is None or chamber.region_id != feature.region_id:
                raise GeometryValidationError(
                    "unknown_hero_target",
                    "hero target chamber is missing or belongs to another region",
                    entity_id=feature.feature_id,
                )
            half_size = tuple(value / 2.0 for value in feature.size)
            envelope_points = tuple(
                tuple(
                    feature.anchor[axis] + signs[axis] * half_size[axis]
                    for axis in range(3)
                )
                for signs in (
                    (-1, -1, 0),
                    (-1, 1, 0),
                    (1, -1, 0),
                    (1, 1, 0),
                    (-1, -1, 1),
                    (-1, 1, 1),
                    (1, -1, 1),
                    (1, 1, 1),
                )
            )
            if not all(chamber.contains(point) for point in envelope_points):
                raise GeometryValidationError(
                    "hero_outside_chamber",
                    "the complete hero envelope must lie inside its target chamber",
                    entity_id=feature.feature_id,
                )

        for region in regions:
            if region.kind == EnvironmentKind.SUBTERRANEAN and not (
                any(item.region_id == region.region_id for item in chambers)
                or any(item.region_id == region.region_id for item in networks)
            ):
                raise GeometryValidationError(
                    "empty_subterranean_region",
                    "a subterranean region requires a chamber or passage network",
                    entity_id=region.region_id,
                )
        object.__setattr__(
            self,
            "schema_version",
            schema_version,
        )
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "chambers", chambers)
        object.__setattr__(self, "passage_networks", networks)
        object.__setattr__(self, "openings", openings)
        object.__setattr__(self, "detail_fields", detail_fields)
        object.__setattr__(self, "hero_features", hero_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "regions": [item.to_dict() for item in self.regions],
            "chambers": [item.to_dict() for item in self.chambers],
            "passage_networks": [item.to_dict() for item in self.passage_networks],
            "openings": [item.to_dict() for item in self.openings],
            "detail_fields": [item.to_dict() for item in self.detail_fields],
            "hero_features": [item.to_dict() for item in self.hero_features],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticEnvironmentSpec":
        _reject_unknown_fields(
            data,
            {
                "schema_version",
                "regions",
                "chambers",
                "passage_networks",
                "openings",
                "detail_fields",
                "hero_features",
            },
        )
        return cls(
            schema_version=data.get(
                "schema_version", SEMANTIC_ENVIRONMENT_SCHEMA_VERSION
            ),
            regions=tuple(
                EnvironmentRegionSpec.from_dict(item)
                for item in data.get("regions", [])
            ),
            chambers=tuple(
                CavernChamberSpec.from_dict(item) for item in data.get("chambers", [])
            ),
            passage_networks=tuple(
                PassageNetworkSpec.from_dict(item)
                for item in data.get("passage_networks", [])
            ),
            openings=tuple(
                EnvironmentOpeningSpec.from_dict(item)
                for item in data.get("openings", [])
            ),
            detail_fields=tuple(
                DetailFieldSpec.from_dict(item)
                for item in data.get("detail_fields", [])
            ),
            hero_features=tuple(
                HeroFeatureSpec.from_dict(item)
                for item in data.get("hero_features", [])
            ),
        )

    def content_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate_layout_bindings(
        self, *, space_level_ids: Mapping[str, str], level_ids: Sequence[str]
    ) -> None:
        """Validate optional passage junction bindings against a HouseLayout."""

        known_levels = set(level_ids)
        for network in self.passage_networks:
            for junction in network.junctions:
                if junction.space_id is None:
                    continue
                assert junction.level_id is not None
                if junction.space_id not in space_level_ids:
                    raise GeometryValidationError(
                        "unknown_junction_space",
                        f"space '{junction.space_id}' is not defined",
                        entity_id=junction.junction_id,
                    )
                if junction.level_id not in known_levels:
                    raise GeometryValidationError(
                        "unknown_junction_level",
                        f"level '{junction.level_id}' is not defined",
                        entity_id=junction.junction_id,
                    )
                expected_level = space_level_ids[junction.space_id]
                if junction.level_id != expected_level:
                    raise GeometryValidationError(
                        "junction_level_mismatch",
                        f"space '{junction.space_id}' belongs to level "
                        f"'{expected_level}', not '{junction.level_id}'",
                        entity_id=junction.junction_id,
                    )
