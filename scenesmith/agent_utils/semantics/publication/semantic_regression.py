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


class OperationalGuardrails(RegressionModel):
    """Fail-closed budgets for a post-change candidate build."""

    baseline_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    max_duration_increase_ratio: float | None = None
    max_requests: int | None = None
    max_turns: int | None = None
    max_total_tokens: int | None = None
    max_api_equivalent_cost_usd: float | None = None
    max_failure_rate: float = 0.0

    @model_validator(mode="after")
    def validate_limits(self) -> "OperationalGuardrails":
        numeric_limits = {
            "baseline_duration_seconds": self.baseline_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "max_duration_increase_ratio": self.max_duration_increase_ratio,
            "max_requests": self.max_requests,
            "max_turns": self.max_turns,
            "max_total_tokens": self.max_total_tokens,
            "max_api_equivalent_cost_usd": self.max_api_equivalent_cost_usd,
        }
        invalid = [
            name
            for name, value in numeric_limits.items()
            if value is not None and value < 0
        ]
        if invalid:
            raise ValueError(
                "operational guardrails cannot be negative: " + ", ".join(invalid)
            )
        if not 0.0 <= self.max_failure_rate <= 1.0:
            raise ValueError("max_failure_rate must be between zero and one")
        return self


class ArtifactContract(RegressionModel):
    """Durable publication evidence required from a protected candidate."""

    require_semantic_certificate: bool = False
    require_closed_ledger: bool = False
    require_physics_verified: bool = False
    required_room_kit_ids: tuple[str, ...] = ()


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
    duration_seconds: float | None = None
    topology: TopologyExpectation
    final_object_minimum: int
    final_object_maximum: int | None = None
    object_counts: tuple[ObjectCountExpectation, ...] = ()
    required_exports: tuple[str, ...] = ()
    expected_error_patterns: tuple[str, ...] = ()
    usage: UsageExpectation = Field(default_factory=UsageExpectation)
    visual_references: tuple[VisualReference, ...] = ()
    known_semantic_gaps: tuple[str, ...] = ()


class RegressionCase(RegressionModel):
    case_id: str
    category: str
    prompt: str
    target_outcome: Literal["complete", "explicit_failure"]
    target_obligations: tuple[str, ...] = ()
    artifact_contract: ArtifactContract = Field(default_factory=ArtifactContract)
    operational_guardrails: OperationalGuardrails | None = None
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
    error: str | None = None
    duration_seconds: float | None
    topology: TopologyExpectation
    final_object_count: int
    object_counts: dict[str, int]
    exports: tuple[str, ...]
    usage: dict[str, int | float | list[str]] | None
    semantic_certificate_count: int = 0
    semantic_publishable: bool | None = None
    physics_verified: bool | None = None
    ledger_closed: bool | None = None
    ledger_publishable: bool | None = None
    room_kit_ids: tuple[str, ...] = ()


class BaselineResult(RegressionModel):
    case_id: str
    matches_reference: bool
    issues: tuple[str, ...]
    observation: BaselineObservation


OperationalMetric = Literal[
    "duration_seconds",
    "requests",
    "turns",
    "total_tokens",
    "api_equivalent_cost_usd",
    "failure_rate",
]


class OperationalDelta(RegressionModel):
    metric: OperationalMetric
    baseline: float | None
    observed: float | None
    delta: float | None
    delta_ratio: float | None
    maximum: float | None
    within_bounds: bool


class CandidateResult(RegressionModel):
    case_id: str
    candidate_run_id: str
    operational_run_id: str
    passed: bool
    issues: tuple[str, ...]
    operational_deltas: tuple[OperationalDelta, ...]
    observation: BaselineObservation


def load_regression_corpus(path: Path) -> RegressionCorpus:
    """Load and validate a semantic regression corpus."""

    return RegressionCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _room_artifact_paths(run_dir: Path, filename: str) -> list[Path]:
    """Find canonical room artifacts with fallback for older retained runs."""

    paths: list[Path] = []
    for room_dir in sorted(run_dir.glob("scene_*/room_*")):
        canonical = room_dir / filename
        legacy = room_dir / "scene_states" / "final_scene" / filename
        if canonical.is_file():
            paths.append(canonical)
        elif legacy.is_file():
            paths.append(legacy)
    return paths


def observe_reference_build(run_dir: Path, case: RegressionCase) -> BaselineObservation:
    """Collect stable structured evidence from one retained build directory."""

    if case.reference is None:
        raise ValueError(f"case {case.case_id} has no retained reference build")
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    blueprint = json.loads(
        next(run_dir.glob("scene_*/scene_blueprint.json")).read_text(encoding="utf-8")
    )
    final_scene_paths = sorted(
        run_dir.glob("scene_*/room_*/scene_states/final_scene/scene_state.json")
    )
    if len(final_scene_paths) == 1:
        final_scene = json.loads(final_scene_paths[0].read_text(encoding="utf-8"))
        objects = final_scene.get("objects") or {}
    elif not final_scene_paths and case.target_outcome == "explicit_failure":
        objects = {}
    else:
        raise ValueError(
            f"expected exactly one final scene in {run_dir}, "
            f"found {len(final_scene_paths)}"
        )
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
    export_names = set(case.reference.required_exports) | {
        "scene.glb",
        "scene_textured.glb",
    }
    exports = tuple(
        export_name
        for export_name in sorted(export_names)
        if (run_dir / export_name).is_file()
    )
    certificate_paths = _room_artifact_paths(
        run_dir,
        "semantic_publication_certificate.json",
    )
    certificates = [
        json.loads(path.read_text(encoding="utf-8")) for path in certificate_paths
    ]
    ledger_summary_paths = _room_artifact_paths(
        run_dir,
        "semantic_obligation_summary.json",
    )
    ledger_summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in ledger_summary_paths
    ]
    room_kit_ids = tuple(
        sorted(
            {
                str(payload["kit_id"])
                for path in run_dir.glob("scene_*/room_*/room_kit.json")
                if isinstance(
                    payload := json.loads(path.read_text(encoding="utf-8")), dict
                )
                and payload.get("kit_id")
            }
        )
    )
    return BaselineObservation(
        run_id=str(job.get("id") or run_dir.name),
        job_status=str(job.get("status") or "unknown"),
        error=(str(job["error"]) if job.get("error") else None),
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
        semantic_certificate_count=len(certificates),
        semantic_publishable=(
            all(payload.get("publishable") is True for payload in certificates)
            if certificates
            else None
        ),
        physics_verified=(
            all(payload.get("physics_verified") is True for payload in certificates)
            if certificates
            else None
        ),
        ledger_closed=(
            all(payload.get("closed") is True for payload in ledger_summaries)
            if ledger_summaries
            else None
        ),
        ledger_publishable=(
            all(payload.get("publishable") is True for payload in ledger_summaries)
            if ledger_summaries
            else None
        ),
        room_kit_ids=room_kit_ids,
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
    for pattern in reference.expected_error_patterns:
        if observation.error is None or re.search(pattern, observation.error) is None:
            issues.append(
                f"error expected to match /{pattern}/, observed {observation.error!r}"
            )
    if case.target_outcome == "explicit_failure" and observation.exports:
        issues.append(
            "explicit failure unexpectedly published exports: "
            + ", ".join(observation.exports)
        )
    if (
        reference.duration_seconds is not None
        and observation.duration_seconds != reference.duration_seconds
    ):
        issues.append(
            f"duration_seconds expected {reference.duration_seconds}, "
            f"observed {observation.duration_seconds}"
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


def _usage_number(observation: BaselineObservation, field: str) -> float | None:
    if observation.usage is None:
        return None
    value = observation.usage.get(field)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _delta(
    metric: OperationalMetric,
    *,
    baseline: float | None,
    observed: float | None,
    maximum: float | None,
) -> OperationalDelta:
    difference = (
        observed - baseline if baseline is not None and observed is not None else None
    )
    ratio = (
        difference / baseline
        if difference is not None and baseline not in {None, 0.0}
        else None
    )
    return OperationalDelta(
        metric=metric,
        baseline=baseline,
        observed=observed,
        delta=difference,
        delta_ratio=ratio,
        maximum=maximum,
        within_bounds=(
            observed is not None and (maximum is None or observed <= maximum)
        ),
    )


def compare_candidate_build(
    case: RegressionCase,
    observation: BaselineObservation,
    *,
    operational_observation: BaselineObservation | None = None,
    attempts: int = 1,
    failed_attempts: int = 0,
) -> CandidateResult:
    """Validate one protected candidate and report bounded operational deltas.

    A late-stage resume may contain no LLM calls of its own. Callers can supply
    an explicit completed full-pipeline observation for operational evidence
    while retaining the resumed candidate as the semantic/artifact subject.
    """

    reference = case.reference
    guardrails = case.operational_guardrails
    if reference is None:
        raise ValueError(f"case {case.case_id} has no retained reference build")
    if reference.semantic_status != "approved":
        raise ValueError(f"case {case.case_id} is not an approved success reference")
    if guardrails is None:
        raise ValueError(f"case {case.case_id} has no operational guardrails")
    if attempts <= 0 or failed_attempts < 0 or failed_attempts > attempts:
        raise ValueError("attempt counts must satisfy 0 <= failed <= attempts")

    operational = operational_observation or observation

    issues: list[str] = []
    if observation.job_status != "completed":
        issues.append(
            f"candidate job_status expected completed, observed {observation.job_status}"
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

    contract = case.artifact_contract
    if (
        contract.require_semantic_certificate
        and not observation.semantic_certificate_count
    ):
        issues.append("semantic publication certificate is required but missing")
    if (
        contract.require_semantic_certificate
        and observation.semantic_publishable is not True
    ):
        issues.append("semantic publication certificate is not publishable")
    if contract.require_physics_verified and observation.physics_verified is not True:
        issues.append("semantic certificate lacks clean physics evidence")
    if contract.require_closed_ledger and observation.ledger_closed is not True:
        issues.append("semantic obligation ledger is not closed")
    if contract.require_closed_ledger and observation.ledger_publishable is not True:
        issues.append("semantic obligation ledger is not publishable")
    missing_room_kits = sorted(
        set(contract.required_room_kit_ids) - set(observation.room_kit_ids)
    )
    if missing_room_kits:
        issues.append("missing required room kits: " + ", ".join(missing_room_kits))
    if operational.job_status != "completed":
        issues.append(
            "operational evidence job_status expected completed, observed "
            f"{operational.job_status}"
        )

    baseline_usage = reference.usage
    duration_baseline = (
        guardrails.baseline_duration_seconds
        if guardrails.baseline_duration_seconds is not None
        else reference.duration_seconds
    )
    duration_limit = guardrails.max_duration_seconds
    if (
        duration_baseline is not None
        and guardrails.max_duration_increase_ratio is not None
    ):
        relative_limit = duration_baseline * (
            1.0 + guardrails.max_duration_increase_ratio
        )
        duration_limit = (
            min(duration_limit, relative_limit)
            if duration_limit is not None
            else relative_limit
        )
    failure_rate = failed_attempts / attempts
    deltas = (
        _delta(
            "duration_seconds",
            baseline=duration_baseline,
            observed=operational.duration_seconds,
            maximum=duration_limit,
        ),
        _delta(
            "requests",
            baseline=(
                float(baseline_usage.requests)
                if baseline_usage.requests is not None
                else None
            ),
            observed=_usage_number(operational, "requests"),
            maximum=(
                float(guardrails.max_requests)
                if guardrails.max_requests is not None
                else None
            ),
        ),
        _delta(
            "turns",
            baseline=(
                float(baseline_usage.turns)
                if baseline_usage.turns is not None
                else None
            ),
            observed=_usage_number(operational, "turns"),
            maximum=(
                float(guardrails.max_turns)
                if guardrails.max_turns is not None
                else None
            ),
        ),
        _delta(
            "total_tokens",
            baseline=(
                float(baseline_usage.total_tokens)
                if baseline_usage.total_tokens is not None
                else None
            ),
            observed=_usage_number(operational, "totalTokens"),
            maximum=(
                float(guardrails.max_total_tokens)
                if guardrails.max_total_tokens is not None
                else None
            ),
        ),
        _delta(
            "api_equivalent_cost_usd",
            baseline=baseline_usage.api_equivalent_cost_usd,
            observed=_usage_number(operational, "apiEquivalentCostUsd"),
            maximum=guardrails.max_api_equivalent_cost_usd,
        ),
        _delta(
            "failure_rate",
            baseline=0.0,
            observed=failure_rate,
            maximum=guardrails.max_failure_rate,
        ),
    )
    for metric in deltas:
        if not metric.within_bounds:
            issues.append(
                f"operational {metric.metric} expected <= {metric.maximum}, "
                f"observed {metric.observed}"
            )
    return CandidateResult(
        case_id=case.case_id,
        candidate_run_id=observation.run_id,
        operational_run_id=operational.run_id,
        passed=not issues,
        issues=tuple(issues),
        operational_deltas=deltas,
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


def validate_candidate_run(
    corpus_path: Path,
    *,
    case_id: str,
    run_root: Path,
    candidate_run_id: str,
    operational_run_id: str | None = None,
    attempts: int = 1,
    failed_attempts: int = 0,
) -> CandidateResult:
    """Validate a post-change candidate against its protected reference."""

    case = load_regression_corpus(corpus_path).case(case_id)
    observation = observe_reference_build(run_root / candidate_run_id, case)
    operational_observation = (
        observe_reference_build(run_root / operational_run_id, case)
        if operational_run_id is not None
        else None
    )
    return compare_candidate_build(
        case,
        observation,
        operational_observation=operational_observation,
        attempts=attempts,
        failed_attempts=failed_attempts,
    )
