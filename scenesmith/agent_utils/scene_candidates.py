"""Deterministic proxy-scene tournament before expensive construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenesmith.agent_utils.contextual_solver import validate_blueprint_topology
from scenesmith.agent_utils.room_kits import RoomKitSelection, select_room_kit
from scenesmith.agent_utils.scene_blueprint import (
    FurnitureGroupBlueprint,
    SceneBlueprint,
    stable_blueprint_id,
)


@dataclass(frozen=True)
class CandidateStrategy:
    name: str
    scale: float
    layout: str
    furniture: str
    target_density: float
    sightline_quality: float
    focal_quality: float


@dataclass(frozen=True)
class CandidateScores:
    hard_validity: float
    prompt_coverage: float
    requested_counts: float
    circulation: float
    proportions: float
    density: float
    sightlines: float
    connectivity: float
    kit_coherence: float
    focal_hierarchy: float
    style_fit: float
    diversity: float

    @property
    def total(self) -> float:
        return round(sum(self.__dict__.values()), 3)

    def to_dict(self) -> dict[str, float]:
        return {**self.__dict__, "total": self.total}


@dataclass(frozen=True)
class SceneCandidate:
    candidate_id: str
    parent_blueprint_id: str
    ordinal: int
    seed: int
    strategy: CandidateStrategy
    blueprint: SceneBlueprint
    scores: CandidateScores
    viable: bool
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_blueprint_id": self.parent_blueprint_id,
            "ordinal": self.ordinal,
            "seed": self.seed,
            "strategy": self.strategy.__dict__,
            "blueprint": self.blueprint.model_dump(mode="json"),
            "scores": self.scores.to_dict(),
            "viable": self.viable,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class CandidateTournament:
    tournament_id: str
    candidates: tuple[SceneCandidate, ...]
    winner_id: str
    comparator_used: bool = False

    @property
    def winner(self) -> SceneCandidate:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.winner_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tournament_id": self.tournament_id,
            "winner_id": self.winner_id,
            "comparator_used": self.comparator_used,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


STRATEGIES: tuple[CandidateStrategy, ...] = (
    CandidateStrategy(
        "balanced", 1.00, "central-zones", "functional-groups", 0.22, 0.82, 0.82
    ),
    CandidateStrategy(
        "intimate", 0.90, "nested-zones", "close-conversation", 0.28, 0.72, 0.78
    ),
    CandidateStrategy(
        "spacious", 1.15, "open-center", "perimeter-anchors", 0.16, 0.90, 0.74
    ),
    CandidateStrategy(
        "axial", 1.05, "strong-primary-axis", "paired-symmetry", 0.20, 0.94, 0.92
    ),
    CandidateStrategy(
        "perimeter", 1.00, "clear-circulation-core", "wall-anchors", 0.18, 0.88, 0.80
    ),
    CandidateStrategy(
        "layered",
        1.10,
        "foreground-midground-focal",
        "asymmetric-clusters",
        0.26,
        0.78,
        0.94,
    ),
    CandidateStrategy(
        "compact-grid", 0.95, "modular-bays", "repeatable-modules", 0.25, 0.84, 0.76
    ),
    CandidateStrategy(
        "gallery", 1.12, "focal-vistas", "sparse-hero-pieces", 0.14, 0.98, 0.90
    ),
)


def _seed_for(prompt: str, ordinal: int) -> int:
    digest = hashlib.sha256(f"{prompt}:{ordinal}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _kit_footprint(selection: RoomKitSelection | None) -> float:
    if selection is None:
        return 4.0
    return sum(
        selection.slot_counts[slot.role]
        * slot.nominal_dimensions_m[0]
        * slot.nominal_dimensions_m[1]
        for slot in selection.slots
    )


def _roles_for(selection: RoomKitSelection | None) -> dict[str, int]:
    return dict(selection.slot_counts) if selection is not None else {}


def _build_candidate_blueprint(
    base: SceneBlueprint,
    strategy: CandidateStrategy,
    selection: RoomKitSelection | None,
    ordinal: int,
) -> SceneBlueprint:
    spaces = tuple(
        space.model_copy(
            update={
                "dimensions_m": (
                    round(space.dimensions_m[0] * strategy.scale, 3),
                    round(space.dimensions_m[1] * strategy.scale, 3),
                )
            }
        )
        for space in base.spaces
    )
    groups = base.furniture_groups
    if selection is not None and not groups:
        groups = (
            FurnitureGroupBlueprint(
                group_id=stable_blueprint_id(
                    "group", f"{selection.kit_id}-{strategy.name}", ordinal
                ),
                name=f"{selection.label} — {strategy.furniture}",
                space_id=spaces[0].space_id,
                roles=_roles_for(selection),
                focal_target=strategy.layout,
                density=(
                    "sparse"
                    if strategy.target_density < 0.18
                    else "layered" if strategy.target_density > 0.24 else "balanced"
                ),
            ),
        )
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "blueprint_id": stable_blueprint_id(
                "scene", f"{base.blueprint_id}-{strategy.name}", ordinal
            ),
            "spaces": [space.model_dump(mode="json") for space in spaces],
            "furniture_groups": [group.model_dump(mode="json") for group in groups],
        }
    )
    return SceneBlueprint.model_validate(payload)


def _score_candidate(
    candidate: SceneBlueprint,
    *,
    prompt: str,
    strategy: CandidateStrategy,
    selection: RoomKitSelection | None,
    prior_strategies: tuple[CandidateStrategy, ...],
) -> tuple[CandidateScores, bool, tuple[str, ...]]:
    topology = validate_blueprint_topology(candidate)
    diagnostics = tuple(violation.message for violation in topology.violations)
    area = sum(
        space.dimensions_m[0] * space.dimensions_m[1] for space in candidate.spaces
    )
    footprint = _kit_footprint(selection)
    occupancy_ratio = footprint / max(area, 1.0)
    requested_multilevel = any(
        token in prompt.casefold()
        for token in ("multi-level", "multilevel", "two-level", "mezzanine", "stairs")
    )
    prompt_coverage = 10.0
    if requested_multilevel:
        prompt_coverage += 5.0 if candidate.connectors else 0.0
    else:
        prompt_coverage += 5.0
    requested_counts = 15.0
    if selection is not None:
        authored_roles = (
            candidate.furniture_groups[0].roles if candidate.furniture_groups else {}
        )
        matching = sum(
            authored_roles.get(role) == count
            for role, count in selection.slot_counts.items()
        )
        requested_counts = 15.0 * matching / max(len(selection.slot_counts), 1)
    circulation = max(0.0, 15.0 - abs(occupancy_ratio - strategy.target_density) * 45.0)
    aspect_scores = []
    for space in candidate.spaces:
        ratio = max(space.dimensions_m) / max(min(space.dimensions_m), 0.1)
        aspect_scores.append(max(0.0, 10.0 - max(0.0, ratio - 1.6) * 5.0))
    proportions = sum(aspect_scores) / len(aspect_scores)
    density = max(0.0, 8.0 - abs(occupancy_ratio - strategy.target_density) * 24.0)
    connectivity = 8.0 if topology.valid else 0.0
    kit_coherence = 7.0 if selection is None or candidate.furniture_groups else 0.0
    style_fit = 5.0 if candidate.design_tokens.style_keywords else 3.0
    if not prior_strategies:
        diversity = 3.0
    else:
        nearest = min(
            abs(strategy.scale - prior.scale)
            + (strategy.layout != prior.layout) * 0.25
            + (strategy.furniture != prior.furniture) * 0.25
            for prior in prior_strategies
        )
        diversity = min(5.0, 2.0 + nearest * 5.0)
    scores = CandidateScores(
        hard_validity=10.0 if topology.valid else 0.0,
        prompt_coverage=round(prompt_coverage, 3),
        requested_counts=round(requested_counts, 3),
        circulation=round(circulation, 3),
        proportions=round(proportions, 3),
        density=round(density, 3),
        sightlines=round(strategy.sightline_quality * 8.0, 3),
        connectivity=connectivity,
        kit_coherence=kit_coherence,
        focal_hierarchy=round(strategy.focal_quality * 7.0, 3),
        style_fit=style_fit,
        diversity=round(diversity, 3),
    )
    return scores, topology.valid, diagnostics


def create_candidate_tournament(
    base_blueprint: SceneBlueprint,
    *,
    prompt: str,
    candidate_count: int = 6,
) -> CandidateTournament:
    """Create 4-8 cheap candidates and deterministically select the winner."""

    count = max(4, min(8, int(candidate_count)))
    area = sum(
        space.dimensions_m[0] * space.dimensions_m[1] for space in base_blueprint.spaces
    )
    selection = select_room_kit(prompt, room_area_m2=area)
    candidates: list[SceneCandidate] = []
    for ordinal, strategy in enumerate(STRATEGIES[:count]):
        blueprint = _build_candidate_blueprint(
            base_blueprint, strategy, selection, ordinal
        )
        scores, viable, diagnostics = _score_candidate(
            blueprint,
            prompt=prompt,
            strategy=strategy,
            selection=selection,
            prior_strategies=tuple(item.strategy for item in candidates),
        )
        candidates.append(
            SceneCandidate(
                candidate_id=stable_blueprint_id(
                    "candidate",
                    f"{base_blueprint.blueprint_id}-{strategy.name}",
                    ordinal,
                ),
                parent_blueprint_id=base_blueprint.blueprint_id,
                ordinal=ordinal,
                seed=_seed_for(prompt, ordinal),
                strategy=strategy,
                blueprint=blueprint,
                scores=scores,
                viable=viable,
                diagnostics=diagnostics,
            )
        )
    viable_candidates = [candidate for candidate in candidates if candidate.viable]
    if not viable_candidates:
        raise ValueError("all proxy scene candidates failed hard structural validation")
    if re.search(r"\b(?:hundreds|thousands) of books\b", prompt.casefold()):
        layered_candidates = [
            candidate
            for candidate in viable_candidates
            if candidate.strategy.name == "layered"
        ]
        dense_candidates = [
            candidate
            for candidate in viable_candidates
            if candidate.strategy.target_density >= 0.24
        ]
        viable_candidates = layered_candidates or dense_candidates or viable_candidates
    winner = max(
        viable_candidates,
        key=lambda candidate: (candidate.scores.total, -candidate.ordinal),
    )
    return CandidateTournament(
        tournament_id=stable_blueprint_id("tournament", base_blueprint.blueprint_id),
        candidates=tuple(candidates),
        winner_id=winner.candidate_id,
        comparator_used=False,
    )


def persist_candidate_tournament(
    tournament: CandidateTournament, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(tournament.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
