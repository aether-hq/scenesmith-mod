"""Capability preflight for semantic obligation fulfillment strategies."""

from __future__ import annotations

import hashlib
import os
import tempfile

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from scenesmith.agent_utils.scene_requirements import (
    FulfillmentStrategy,
    RequirementKind,
    SceneRequirement,
    SceneRequirementGraph,
    assert_requirement_graph_consistent,
)
from scenesmith.agent_utils.semantic_ledger import (
    SemanticObligationLedger,
    transition_requirement,
    validate_ledger_against_graph,
)


class StrategyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticCapabilityProfile(StrategyModel):
    catalog_available: bool
    generated_geometry_available: bool
    reusable_composition_available: bool
    structural_compiler_available: bool
    catalog_provider: str = "global_asset_catalog"
    generated_geometry_provider: str = "text_to_3d_geometry"
    composition_provider: str = "reusable_assembly_compiler"
    structural_provider: str = "spatial_topology_compiler"
    source_notes: tuple[str, ...] = ()


class StrategyAvailability(StrategyModel):
    strategy: FulfillmentStrategy
    provider_id: str
    status: Literal["available", "unavailable"]
    rationale: str


class RequirementStrategyPlan(StrategyModel):
    requirement_id: str
    subject: str
    kind: RequirementKind
    ordered_strategies: tuple[StrategyAvailability, ...]
    selected_strategy: FulfillmentStrategy | None
    selected_provider: str | None
    planned_instances: int
    composition_brief: str
    failure_reason: str = ""


class SemanticCapabilityManifest(StrategyModel):
    schema_version: Literal[1] = 1
    graph_id: str
    graph_hash: str
    profile: SemanticCapabilityProfile
    plans: tuple[RequirementStrategyPlan, ...]
    preflight_passed: bool
    blocking_failures: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class CapabilityPreflightError(RuntimeError):
    """No configured strategy can fulfill one or more blocking requirements."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


_STRUCTURAL_KINDS: frozenset[RequirementKind] = frozenset(
    {
        "scene_type",
        "level",
        "repeated_zone",
        "opening",
        "connector",
        "spatial_constraint",
    }
)
_CATALOG_KINDS: frozenset[RequirementKind] = frozenset(
    {"hero_object", "object_group", "opening", "connector", "style"}
)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _generated_enabled(asset_manager: Mapping[str, object]) -> bool:
    source = str(asset_manager.get("general_asset_source", ""))
    if source == "generated":
        return True
    if source != "all":
        return False
    router = _mapping(asset_manager.get("router"))
    strategies = _mapping(router.get("strategies"))
    generated = _mapping(strategies.get("generated"))
    if generated.get("enabled", True) is False:
        return False
    federated = _mapping(asset_manager.get("federated"))
    order = federated.get(
        "source_order", ("polyhaven", "hssd", "objaverse", "generated")
    )
    return "generated" in tuple(order) if isinstance(order, (list, tuple)) else False


def capability_profile_from_config(
    config: Mapping[str, object],
) -> SemanticCapabilityProfile:
    """Derive configured capabilities without assuming any scene-domain nouns."""

    managers: list[Mapping[str, object]] = []
    notes: list[str] = []
    for agent_key in (
        "furniture_agent",
        "wall_agent",
        "ceiling_agent",
        "manipuland_agent",
    ):
        manager = _mapping(_mapping(config.get(agent_key)).get("asset_manager"))
        if manager:
            managers.append(manager)
            notes.append(
                f"{agent_key}:{manager.get('general_asset_source', 'unconfigured')}"
            )
    catalog = any(
        str(manager.get("general_asset_source", ""))
        in {"hssd", "objaverse", "polyhaven", "all"}
        for manager in managers
    )
    generated = any(_generated_enabled(manager) for manager in managers)
    experiment = _mapping(config.get("experiment"))
    semantic = _mapping(experiment.get("semantic_obligations"))
    composition = bool(semantic.get("composition_enabled", True)) and (
        catalog or generated
    )
    structural = bool(semantic.get("structural_compiler_enabled", True))
    return SemanticCapabilityProfile(
        catalog_available=catalog,
        generated_geometry_available=generated,
        reusable_composition_available=composition,
        structural_compiler_available=structural,
        source_notes=tuple(notes),
    )


def _strategy_availability(
    requirement: SceneRequirement,
    strategy: FulfillmentStrategy,
    profile: SemanticCapabilityProfile,
) -> StrategyAvailability:
    if requirement.kind == "unclassified":
        return StrategyAvailability(
            strategy=strategy,
            provider_id="none",
            status="unavailable",
            rationale="Semantic classification is required before strategy assignment.",
        )
    if strategy == "catalog":
        supported = requirement.kind in _CATALOG_KINDS
        available = profile.catalog_available and supported
        return StrategyAvailability(
            strategy=strategy,
            provider_id=profile.catalog_provider,
            status="available" if available else "unavailable",
            rationale=(
                "A configured catalog can retrieve this semantic artifact class."
                if available
                else "No configured catalog supports this artifact class."
            ),
        )
    if strategy == "composed":
        structural = (
            requirement.kind in _STRUCTURAL_KINDS
            and profile.structural_compiler_available
        )
        available = profile.reusable_composition_available or structural
        provider = (
            profile.structural_provider if structural else profile.composition_provider
        )
        return StrategyAvailability(
            strategy=strategy,
            provider_id=provider,
            status="available" if available else "unavailable",
            rationale=(
                "Reusable or parameterized parts can compose the requested whole."
                if available
                else "No reusable-part source or structural compiler is configured."
            ),
        )
    structural = (
        requirement.kind in _STRUCTURAL_KINDS and profile.structural_compiler_available
    )
    available = structural or profile.generated_geometry_available
    provider = (
        profile.structural_provider
        if structural
        else profile.generated_geometry_provider
    )
    return StrategyAvailability(
        strategy=strategy,
        provider_id=provider,
        status="available" if available else "unavailable",
        rationale=(
            "A configured procedural provider can generate requirement-specific geometry."
            if available
            else "No procedural provider is configured for this artifact class."
        ),
    )


def _planned_instances(requirement: SceneRequirement) -> int:
    if requirement.quantity.mode in {"exact", "minimum"}:
        return int(requirement.quantity.value or 1)
    return int(requirement.quantity.interpreted_minimum or 1)


def capability_preflight(
    graph: SceneRequirementGraph,
    profile: SemanticCapabilityProfile,
) -> SemanticCapabilityManifest:
    """Assign viable ordered providers before expensive construction starts."""

    assert_requirement_graph_consistent(graph)
    plans: list[RequirementStrategyPlan] = []
    blocking_failures: list[str] = []
    for requirement in graph.requirements:
        if requirement.strength != "hard":
            continue
        strategy_order = (
            requirement.composition.strategy_order
            if requirement.composition is not None
            else ("catalog", "composed", "procedural")
        )
        availabilities = tuple(
            _strategy_availability(requirement, strategy, profile)
            for strategy in strategy_order
        )
        selected = next(
            (item for item in availabilities if item.status == "available"), None
        )
        failure_reason = ""
        if selected is None:
            failure_reason = (
                f"No catalog, composition, or procedural provider can fulfill "
                f"{requirement.subject!r} ({requirement.requirement_id})."
            )
            if requirement.enforcement in {"blocking", "unresolved_blocking"}:
                blocking_failures.append(requirement.requirement_id)
        plans.append(
            RequirementStrategyPlan(
                requirement_id=requirement.requirement_id,
                subject=requirement.subject,
                kind=requirement.kind,
                ordered_strategies=availabilities,
                selected_strategy=(selected.strategy if selected else None),
                selected_provider=(selected.provider_id if selected else None),
                planned_instances=_planned_instances(requirement),
                composition_brief=(
                    requirement.composition.arrangement
                    if requirement.composition is not None
                    else "Semantic interpretation is required before composition."
                ),
                failure_reason=failure_reason,
            )
        )
    return SemanticCapabilityManifest(
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        profile=profile,
        plans=tuple(plans),
        preflight_passed=not blocking_failures,
        blocking_failures=tuple(blocking_failures),
    )


def apply_capability_manifest_to_ledger(
    ledger: SemanticObligationLedger,
    graph: SceneRequirementGraph,
    manifest: SemanticCapabilityManifest,
    *,
    manifest_ref: str,
    clock: Callable[[], datetime] = _utc_now,
) -> SemanticObligationLedger:
    """Append planning/strategy or specific capability-failure transitions."""

    validate_ledger_against_graph(ledger, graph)
    if manifest.graph_id != graph.graph_id or manifest.graph_hash != graph.content_hash:
        raise CapabilityPreflightError("capability manifest does not match graph")
    for plan in manifest.plans:
        prefix = f"capability:{manifest.content_hash}:{plan.requirement_id}"
        ledger = transition_requirement(
            ledger,
            plan.requirement_id,
            "planned",
            event_key=f"{prefix}:planned",
            actor="semantic_capability_preflight",
            stage="planning",
            evidence_refs=(f"graph:{graph.graph_id}",),
            clock=clock,
        )
        if plan.selected_strategy is None:
            ledger = transition_requirement(
                ledger,
                plan.requirement_id,
                "failed",
                event_key=f"{prefix}:failed",
                actor="semantic_capability_preflight",
                stage="capability",
                evidence_refs=(f"{manifest_ref}#{plan.requirement_id}",),
                failure_reason=plan.failure_reason,
                clock=clock,
            )
        else:
            ledger = transition_requirement(
                ledger,
                plan.requirement_id,
                "strategy_assigned",
                event_key=f"{prefix}:strategy:{plan.selected_strategy}",
                actor="semantic_capability_preflight",
                stage="capability",
                evidence_refs=(
                    f"{manifest_ref}#{plan.requirement_id}",
                    f"provider:{plan.selected_provider}",
                ),
                clock=clock,
            )
    return ledger


def assert_capability_preflight_passed(
    manifest: SemanticCapabilityManifest,
) -> None:
    if manifest.blocking_failures:
        failed = [
            plan.failure_reason
            for plan in manifest.plans
            if plan.requirement_id in manifest.blocking_failures
        ]
        raise CapabilityPreflightError(" ".join(failed))


def persist_capability_manifest(
    manifest: SemanticCapabilityManifest, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(manifest.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_capability_manifest(path: Path) -> SemanticCapabilityManifest:
    return SemanticCapabilityManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
