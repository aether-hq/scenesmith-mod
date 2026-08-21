"""Machine-readable regression baselines for end-to-end SceneSmith builds."""

from __future__ import annotations

import hashlib
import json
import re

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegressionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObjectCountExpectation(RegressionModel):
    """Expected number of final-scene objects whose IDs or names match a regex."""

    pattern: str
    minimum: int = 0
    maximum: int | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "ObjectCountExpectation":
        if self.minimum < 0:
            raise ValueError("minimum object count cannot be negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum object count cannot be below minimum")
        re.compile(self.pattern)
        return self


class TopologyExpectation(RegressionModel):
    levels: int
    spaces: int
    openings: int
    connectors: int


class UsageExpectation(RegressionModel):
    required: bool = False
    requests: int | None = None
    turns: int | None = None
    total_tokens: int | None = None
    api_equivalent_cost_usd: float | None = None


class VisualReference(RegressionModel):
    mode: str
    sha256: str
    width: int
    height: int
    source_path: str
    criteria: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_digest(self) -> "VisualReference":
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("visual reference sha256 must be a lowercase hex digest")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("visual reference dimensions must be positive")
        return self


class ReferenceBuild(RegressionModel):
    run_id: str
    job_status: Literal["completed", "failed", "cancelled"]
    semantic_status: Literal["approved", "known_failure"]
    topology: TopologyExpectation
    final_object_minimum: int
    final_object_maximum: int | None = None
    object_counts: tuple[ObjectCountExpectation, ...] = ()
    required_exports: tuple[str, ...] = ()
    usage: UsageExpectation = Field(default_factory=UsageExpectation)
    visual_references: tuple[VisualReference, ...] = ()
    known_semantic_gaps: tuple[str, ...] = ()


class RegressionCase(RegressionModel):
    case_id: str
    category: str
    prompt: str
    target_outcome: Literal["complete", "explicit_failure"]
    target_obligations: tuple[str, ...] = ()
    reference: ReferenceBuild | None = None


class RegressionCorpus(RegressionModel):
    schema_version: Literal[1] = 1
    cases: tuple[RegressionCase, ...]

    @model_validator(mode="after")
    def validate_case_ids(self) -> "RegressionCorpus":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("regression case IDs must be unique")
        return self

    def case(self, case_id: str) -> RegressionCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"unknown regression case {case_id!r}")


class BaselineObservation(RegressionModel):
    run_id: str
    job_status: str
    duration_seconds: float | None
    topology: TopologyExpectation
    final_object_count: int
    object_counts: dict[str, int]
    exports: tuple[str, ...]
    usage: dict[str, int | float | list[str]] | None


class BaselineResult(RegressionModel):
    case_id: str
    matches_reference: bool
    issues: tuple[str, ...]
    observation: BaselineObservation


def load_regression_corpus(path: Path) -> RegressionCorpus:
    """Load and validate a semantic regression corpus."""

    return RegressionCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _final_scene_path(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.glob("scene_*/room_*/scene_states/final_scene/scene_state.json")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one final scene in {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def observe_reference_build(run_dir: Path, case: RegressionCase) -> BaselineObservation:
    """Collect stable structured evidence from one retained build directory."""

    if case.reference is None:
        raise ValueError(f"case {case.case_id} has no retained reference build")
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    blueprint = json.loads(
        next(run_dir.glob("scene_*/scene_blueprint.json")).read_text(encoding="utf-8")
    )
    final_scene = json.loads(_final_scene_path(run_dir).read_text(encoding="utf-8"))
    objects = final_scene.get("objects") or {}
    if not isinstance(objects, dict):
        raise ValueError("final scene objects must be a mapping")

    searchable_names = [
        f"{object_id} {payload.get('name', '')}".casefold()
        for object_id, payload in objects.items()
        if isinstance(payload, dict)
    ]
    observed_counts = {
        expectation.pattern: sum(
            bool(re.search(expectation.pattern, name, flags=re.IGNORECASE))
            for name in searchable_names
        )
        for expectation in case.reference.object_counts
    }
    started = _parse_timestamp(job.get("startedAt"))
    finished = _parse_timestamp(job.get("finishedAt"))
    duration = (finished - started).total_seconds() if started and finished else None
    exports = tuple(
        export_name
        for export_name in case.reference.required_exports
        if (run_dir / export_name).is_file()
    )
    return BaselineObservation(
        run_id=str(job.get("id") or run_dir.name),
        job_status=str(job.get("status") or "unknown"),
        duration_seconds=duration,
        topology=TopologyExpectation(
            levels=len(blueprint.get("levels") or []),
            spaces=len(blueprint.get("spaces") or []),
            openings=len(blueprint.get("openings") or []),
            connectors=len(blueprint.get("connectors") or []),
        ),
        final_object_count=len(objects),
        object_counts=observed_counts,
        exports=exports,
        usage=job.get("usage"),
    )


def compare_reference_build(
    case: RegressionCase, observation: BaselineObservation
) -> BaselineResult:
    """Compare a structured build observation with its retained baseline."""

    reference = case.reference
    if reference is None:
        raise ValueError(f"case {case.case_id} has no retained reference build")
    issues: list[str] = []
    if observation.run_id != reference.run_id:
        issues.append(
            f"run_id expected {reference.run_id}, observed {observation.run_id}"
        )
    if observation.job_status != reference.job_status:
        issues.append(
            f"job_status expected {reference.job_status}, "
            f"observed {observation.job_status}"
        )
    if observation.topology != reference.topology:
        issues.append(
            "topology expected "
            f"{reference.topology.model_dump()}, observed "
            f"{observation.topology.model_dump()}"
        )
    if observation.final_object_count < reference.final_object_minimum:
        issues.append(
            f"final objects expected >= {reference.final_object_minimum}, "
            f"observed {observation.final_object_count}"
        )
    if (
        reference.final_object_maximum is not None
        and observation.final_object_count > reference.final_object_maximum
    ):
        issues.append(
            f"final objects expected <= {reference.final_object_maximum}, "
            f"observed {observation.final_object_count}"
        )
    for expectation in reference.object_counts:
        count = observation.object_counts.get(expectation.pattern, 0)
        if count < expectation.minimum:
            issues.append(
                f"objects /{expectation.pattern}/ expected >= "
                f"{expectation.minimum}, observed {count}"
            )
        if expectation.maximum is not None and count > expectation.maximum:
            issues.append(
                f"objects /{expectation.pattern}/ expected <= "
                f"{expectation.maximum}, observed {count}"
            )
    missing_exports = sorted(set(reference.required_exports) - set(observation.exports))
    if missing_exports:
        issues.append("missing exports: " + ", ".join(missing_exports))

    expected_usage = reference.usage
    if expected_usage.required and observation.usage is None:
        issues.append("usage record is required but missing")
    if observation.usage is not None:
        usage_fields = {
            "requests": expected_usage.requests,
            "turns": expected_usage.turns,
            "totalTokens": expected_usage.total_tokens,
            "apiEquivalentCostUsd": expected_usage.api_equivalent_cost_usd,
        }
        for field, expected in usage_fields.items():
            if expected is not None and observation.usage.get(field) != expected:
                issues.append(
                    f"usage.{field} expected {expected}, "
                    f"observed {observation.usage.get(field)}"
                )
    return BaselineResult(
        case_id=case.case_id,
        matches_reference=not issues,
        issues=tuple(issues),
        observation=observation,
    )


def verify_visual_references(case: RegressionCase) -> tuple[str, ...]:
    """Verify retained native captures by dimensions and content digest."""

    if case.reference is None:
        return ()
    issues: list[str] = []
    for reference in case.reference.visual_references:
        path = Path(reference.source_path)
        if not path.is_file():
            issues.append(f"visual reference missing: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != reference.sha256:
            issues.append(
                f"visual reference {reference.mode} digest expected "
                f"{reference.sha256}, observed {digest}"
            )
    return tuple(issues)


def validate_reference_run(
    corpus_path: Path,
    *,
    case_id: str,
    run_root: Path,
    verify_visuals: bool = False,
) -> BaselineResult:
    """Validate one retained reference run selected from a corpus."""

    case = load_regression_corpus(corpus_path).case(case_id)
    if case.reference is None:
        raise ValueError(f"case {case.case_id} has no retained reference build")
    observation = observe_reference_build(run_root / case.reference.run_id, case)
    result = compare_reference_build(case, observation)
    if not verify_visuals:
        return result
    visual_issues = verify_visual_references(case)
    issues = result.issues + visual_issues
    return result.model_copy(update={"matches_reference": not issues, "issues": issues})
