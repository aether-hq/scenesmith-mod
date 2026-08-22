"""Append-only lifecycle ledger for source-bound semantic obligations."""

from __future__ import annotations

import hashlib
import os
import tempfile

from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from scenesmith.agent_utils.semantics.requirements.requirement_graph.models import (
    EnforcementDisposition,
    SceneRequirement,
    SceneRequirementGraph,
)

CURRENT_LEDGER_SCHEMA_VERSION = 1
LedgerStatus = Literal[
    "extracted",
    "planned",
    "strategy_assigned",
    "constructed",
    "verified",
    "fulfilled",
    "failed",
]
TerminalLedgerStatus = Literal["fulfilled", "failed"]

_TERMINAL_STATUSES = frozenset({"fulfilled", "failed"})
_ALLOWED_FORWARD_TRANSITION: dict[LedgerStatus, LedgerStatus] = {
    "extracted": "planned",
    "planned": "strategy_assigned",
    "strategy_assigned": "constructed",
    "constructed": "verified",
    "verified": "fulfilled",
}


class LedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LedgerEvent(LedgerModel):
    transition_id: str
    event_key: str
    from_status: LedgerStatus | None
    to_status: LedgerStatus
    actor: str
    stage: str
    timestamp_utc: datetime
    evidence_refs: tuple[str, ...] = ()
    failure_reason: str = ""

    @model_validator(mode="after")
    def validate_failure(self) -> "LedgerEvent":
        if self.to_status == "failed" and not self.failure_reason:
            raise ValueError("failed transition requires a specific failure reason")
        if self.to_status != "failed" and self.failure_reason:
            raise ValueError("failure reason is only valid for a failed transition")
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("ledger event timestamp must be timezone-aware")
        return self


class ObligationLedgerEntry(LedgerModel):
    requirement_id: str
    source_candidate_id: str
    requirement_hash: str
    enforcement: EnforcementDisposition
    current_status: LedgerStatus
    events: tuple[LedgerEvent, ...]

    @model_validator(mode="after")
    def validate_events(self) -> "ObligationLedgerEntry":
        if not self.events:
            raise ValueError("ledger entry requires its extraction event")
        if self.events[0].from_status is not None:
            raise ValueError("first ledger event must originate outside the ledger")
        if self.events[0].to_status != "extracted":
            raise ValueError("first ledger event must record extraction")
        previous: LedgerStatus | None = None
        transition_ids: set[str] = set()
        for event in self.events:
            if event.transition_id in transition_ids:
                raise ValueError("ledger transition IDs must be unique")
            transition_ids.add(event.transition_id)
            if event.from_status != previous:
                raise ValueError("ledger event chain is discontinuous")
            previous = event.to_status
        if self.current_status != self.events[-1].to_status:
            raise ValueError("ledger current status must equal its last event")
        return self


class SemanticObligationLedger(LedgerModel):
    schema_version: Literal[1] = CURRENT_LEDGER_SCHEMA_VERSION
    ledger_id: str
    graph_id: str
    graph_hash: str
    revision: int
    created_at_utc: datetime
    updated_at_utc: datetime
    entries: tuple[ObligationLedgerEntry, ...]

    @model_validator(mode="after")
    def validate_ledger(self) -> "SemanticObligationLedger":
        if self.revision < 0:
            raise ValueError("ledger revision cannot be negative")
        if self.created_at_utc.tzinfo is None or self.updated_at_utc.tzinfo is None:
            raise ValueError("ledger timestamps must be timezone-aware")
        ids = [entry.requirement_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("ledger requirement IDs must be unique")
        return self

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class LedgerStatusCount(LedgerModel):
    status: LedgerStatus
    count: int


class SemanticLedgerSummary(LedgerModel):
    schema_version: Literal[1] = 1
    ledger_id: str
    ledger_hash: str
    graph_id: str
    graph_hash: str
    revision: int
    total_requirements: int
    status_counts: tuple[LedgerStatusCount, ...]
    closed: bool
    publishable: bool
    blocking_failed: tuple[str, ...]
    blocking_unresolved: tuple[str, ...]
    advisory_failed: tuple[str, ...]


class LedgerTransitionError(ValueError):
    """A requested ledger transition violates the append-only lifecycle."""


class LedgerIntegrityError(ValueError):
    """The ledger no longer represents the immutable requirement graph."""


class StaleLedgerRevisionError(LedgerTransitionError):
    """A caller attempted to append to a stale ledger revision."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _requirement_hash(requirement: SceneRequirement) -> str:
    return hashlib.sha256(requirement.model_dump_json().encode("utf-8")).hexdigest()


def _transition_id(ledger_id: str, requirement_id: str, event_key: str) -> str:
    digest = hashlib.sha256(
        f"{ledger_id}:{requirement_id}:{event_key}".encode("utf-8")
    ).hexdigest()[:20]
    return f"transition-{digest}"


def initialize_semantic_ledger(
    graph: SceneRequirementGraph,
    *,
    clock: Callable[[], datetime] = _utc_now,
) -> SemanticObligationLedger:
    """Create extraction entries for every durable hard user obligation."""

    timestamp = clock()
    ledger_id = f"ledger-{graph.graph_id.removeprefix('requirements-')}"
    entries: list[ObligationLedgerEntry] = []
    for requirement in graph.requirements:
        if requirement.strength != "hard":
            continue
        event_key = f"extract:{requirement.requirement_id}"
        event = LedgerEvent(
            transition_id=_transition_id(
                ledger_id, requirement.requirement_id, event_key
            ),
            event_key=event_key,
            from_status=None,
            to_status="extracted",
            actor="requirement_graph",
            stage="extraction",
            timestamp_utc=timestamp,
            evidence_refs=(
                f"graph:{graph.graph_id}",
                f"candidate:{requirement.source_candidate_id}",
            ),
        )
        entries.append(
            ObligationLedgerEntry(
                requirement_id=requirement.requirement_id,
                source_candidate_id=requirement.source_candidate_id,
                requirement_hash=_requirement_hash(requirement),
                enforcement=requirement.enforcement,
                current_status="extracted",
                events=(event,),
            )
        )
    return SemanticObligationLedger(
        ledger_id=ledger_id,
        graph_id=graph.graph_id,
        graph_hash=graph.content_hash,
        revision=0,
        created_at_utc=timestamp,
        updated_at_utc=timestamp,
        entries=tuple(entries),
    )


def validate_ledger_against_graph(
    ledger: SemanticObligationLedger,
    graph: SceneRequirementGraph,
) -> None:
    """Reject missing, added, mutated, or weakened obligations."""

    if ledger.graph_id != graph.graph_id or ledger.graph_hash != graph.content_hash:
        raise LedgerIntegrityError("ledger graph identity/hash does not match graph")
    expected = {
        requirement.requirement_id: requirement
        for requirement in graph.requirements
        if requirement.strength == "hard"
    }
    observed = {entry.requirement_id: entry for entry in ledger.entries}
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise LedgerIntegrityError(
            f"ledger requirement set changed; missing={missing} extra={extra}"
        )
    for requirement_id, requirement in expected.items():
        entry = observed[requirement_id]
        if entry.requirement_hash != _requirement_hash(requirement):
            raise LedgerIntegrityError(
                f"ledger requirement hash changed for {requirement_id}"
            )
        if entry.enforcement != requirement.enforcement:
            raise LedgerIntegrityError(
                f"ledger enforcement changed for {requirement_id}"
            )
        if entry.source_candidate_id != requirement.source_candidate_id:
            raise LedgerIntegrityError(
                f"ledger source candidate changed for {requirement_id}"
            )


def _matching_event(
    ledger: SemanticObligationLedger,
    transition_id: str,
) -> LedgerEvent | None:
    return next(
        (
            event
            for entry in ledger.entries
            for event in entry.events
            if event.transition_id == transition_id
        ),
        None,
    )


def transition_requirement(
    ledger: SemanticObligationLedger,
    requirement_id: str,
    to_status: LedgerStatus,
    *,
    event_key: str,
    actor: str,
    stage: str,
    evidence_refs: Iterable[str] = (),
    failure_reason: str = "",
    expected_revision: int | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> SemanticObligationLedger:
    """Append one validated, idempotent requirement transition."""

    if not event_key.strip():
        raise LedgerTransitionError("event_key must be non-empty")
    transition_id = _transition_id(ledger.ledger_id, requirement_id, event_key)
    prior_event = _matching_event(ledger, transition_id)
    normalized_evidence = tuple(dict.fromkeys(str(item) for item in evidence_refs))
    if prior_event is not None:
        if (
            prior_event.to_status != to_status
            or prior_event.actor != actor
            or prior_event.stage != stage
            or prior_event.evidence_refs != normalized_evidence
            or prior_event.failure_reason != failure_reason
        ):
            raise LedgerTransitionError(
                f"event key {event_key!r} was reused with different transition data"
            )
        return ledger
    if expected_revision is not None and expected_revision != ledger.revision:
        raise StaleLedgerRevisionError(
            f"expected revision {expected_revision}, observed {ledger.revision}"
        )

    entry_index = next(
        (
            index
            for index, entry in enumerate(ledger.entries)
            if entry.requirement_id == requirement_id
        ),
        None,
    )
    if entry_index is None:
        raise LedgerTransitionError(f"unknown requirement {requirement_id}")
    entry = ledger.entries[entry_index]
    current = entry.current_status
    if current in _TERMINAL_STATUSES:
        raise LedgerTransitionError(
            f"terminal requirement {requirement_id} cannot transition from {current}"
        )
    expected_forward = _ALLOWED_FORWARD_TRANSITION[current]
    if to_status != "failed" and to_status != expected_forward:
        raise LedgerTransitionError(
            f"illegal transition {current}->{to_status}; expected {expected_forward} "
            "or failed"
        )
    if (
        to_status in {"constructed", "verified", "fulfilled"}
        and not normalized_evidence
    ):
        raise LedgerTransitionError(
            f"{to_status} transition requires artifact evidence"
        )

    timestamp = clock()
    event = LedgerEvent(
        transition_id=transition_id,
        event_key=event_key,
        from_status=current,
        to_status=to_status,
        actor=actor,
        stage=stage,
        timestamp_utc=timestamp,
        evidence_refs=normalized_evidence,
        failure_reason=failure_reason,
    )
    updated_entry = ObligationLedgerEntry(
        **entry.model_dump(exclude={"current_status", "events"}),
        current_status=to_status,
        events=entry.events + (event,),
    )
    entries = list(ledger.entries)
    entries[entry_index] = updated_entry
    return SemanticObligationLedger(
        **ledger.model_dump(exclude={"revision", "updated_at_utc", "entries"}),
        revision=ledger.revision + 1,
        updated_at_utc=timestamp,
        entries=tuple(entries),
    )


def semantic_ledger_summary(
    ledger: SemanticObligationLedger,
) -> SemanticLedgerSummary:
    counts = Counter(entry.current_status for entry in ledger.entries)
    blocking_failed = tuple(
        entry.requirement_id
        for entry in ledger.entries
        if entry.enforcement in {"blocking", "unresolved_blocking"}
        and entry.current_status == "failed"
    )
    blocking_unresolved = tuple(
        entry.requirement_id
        for entry in ledger.entries
        if entry.enforcement in {"blocking", "unresolved_blocking"}
        and entry.current_status not in _TERMINAL_STATUSES
    )
    advisory_failed = tuple(
        entry.requirement_id
        for entry in ledger.entries
        if entry.enforcement == "advisory" and entry.current_status == "failed"
    )
    closed = all(entry.current_status in _TERMINAL_STATUSES for entry in ledger.entries)
    publishable = closed and not blocking_failed and not blocking_unresolved
    return SemanticLedgerSummary(
        ledger_id=ledger.ledger_id,
        ledger_hash=ledger.content_hash,
        graph_id=ledger.graph_id,
        graph_hash=ledger.graph_hash,
        revision=ledger.revision,
        total_requirements=len(ledger.entries),
        status_counts=tuple(
            LedgerStatusCount(status=status, count=counts.get(status, 0))
            for status in (
                "extracted",
                "planned",
                "strategy_assigned",
                "constructed",
                "verified",
                "fulfilled",
                "failed",
            )
        ),
        closed=closed,
        publishable=publishable,
        blocking_failed=blocking_failed,
        blocking_unresolved=blocking_unresolved,
        advisory_failed=advisory_failed,
    )


def _persist_model(model: LedgerModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def persist_semantic_ledger(
    ledger: SemanticObligationLedger, output_path: Path
) -> None:
    _persist_model(ledger, output_path)


def load_semantic_ledger(path: Path) -> SemanticObligationLedger:
    return SemanticObligationLedger.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def load_or_initialize_semantic_ledger(
    path: Path,
    graph: SceneRequirementGraph,
    *,
    clock: Callable[[], datetime] = _utc_now,
) -> SemanticObligationLedger:
    if path.is_file():
        ledger = load_semantic_ledger(path)
        validate_ledger_against_graph(ledger, graph)
        return ledger
    ledger = initialize_semantic_ledger(graph, clock=clock)
    persist_semantic_ledger(ledger, path)
    return ledger


def start_certified_retry_ledger(
    ledger: SemanticObligationLedger,
    graph: SceneRequirementGraph,
    *,
    clock: Callable[[], datetime] = _utc_now,
) -> SemanticObligationLedger:
    """Start a new run-local attempt when an inherited ledger already failed.

    Terminal events remain immutable in the source run that produced them. A
    resumed run owns a separate ledger artifact, so a new successful certificate
    must be recorded as a fresh linear attempt instead of being blocked forever
    by the copied terminal failure state.
    """

    validate_ledger_against_graph(ledger, graph)
    if not any(entry.current_status == "failed" for entry in ledger.entries):
        return ledger
    return initialize_semantic_ledger(graph, clock=clock)


def persist_semantic_ledger_summary(
    ledger: SemanticObligationLedger, output_path: Path
) -> SemanticLedgerSummary:
    summary = semantic_ledger_summary(ledger)
    _persist_model(summary, output_path)
    return summary
