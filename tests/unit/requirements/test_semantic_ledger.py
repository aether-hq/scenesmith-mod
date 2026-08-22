"""Tests for the append-only semantic obligation ledger."""

from datetime import UTC, datetime

import pytest

from scenesmith.agent_utils.semantics.requirements.scene_requirements import (
    requirement_graph_from_prompt,
)
from scenesmith.agent_utils.semantics.requirements.semantic_ledger import (
    LedgerIntegrityError,
    LedgerTransitionError,
    StaleLedgerRevisionError,
    initialize_semantic_ledger,
    load_or_initialize_semantic_ledger,
    load_semantic_ledger,
    persist_semantic_ledger,
    persist_semantic_ledger_summary,
    semantic_ledger_summary,
    start_certified_retry_ledger,
    transition_requirement,
    validate_ledger_against_graph,
)

FIXED_TIME = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _clock():
    return FIXED_TIME


def _graph():
    return requirement_graph_from_prompt(
        "A calibration chamber with three phase pylons and a central lens."
    )


def _advance_to_fulfilled(ledger, requirement_id):
    for status in ("planned", "strategy_assigned"):
        ledger = transition_requirement(
            ledger,
            requirement_id,
            status,
            event_key=f"{status}:unit",
            actor="unit-test",
            stage=status,
            clock=_clock,
        )
    for status in ("constructed", "verified", "fulfilled"):
        ledger = transition_requirement(
            ledger,
            requirement_id,
            status,
            event_key=f"{status}:unit",
            actor="unit-test",
            stage=status,
            evidence_refs=(f"artifact:{status}",),
            clock=_clock,
        )
    return ledger


def test_initialization_hash_locks_every_hard_requirement():
    graph = _graph()
    ledger = initialize_semantic_ledger(graph, clock=_clock)

    hard_ids = {
        requirement.requirement_id
        for requirement in graph.requirements
        if requirement.strength == "hard"
    }
    assert {entry.requirement_id for entry in ledger.entries} == hard_ids
    assert ledger.graph_hash == graph.content_hash
    assert ledger.revision == 0
    assert all(entry.current_status == "extracted" for entry in ledger.entries)
    assert all(len(entry.events) == 1 for entry in ledger.entries)
    validate_ledger_against_graph(ledger, graph)


def test_linear_lifecycle_requires_evidence_and_terminal_is_immutable():
    ledger = initialize_semantic_ledger(_graph(), clock=_clock)
    requirement_id = ledger.entries[0].requirement_id

    with pytest.raises(LedgerTransitionError, match="illegal transition"):
        transition_requirement(
            ledger,
            requirement_id,
            "constructed",
            event_key="skip-ahead",
            actor="test",
            stage="placement",
        )
    ledger = transition_requirement(
        ledger,
        requirement_id,
        "planned",
        event_key="plan",
        actor="planner",
        stage="planning",
        clock=_clock,
    )
    ledger = transition_requirement(
        ledger,
        requirement_id,
        "strategy_assigned",
        event_key="strategy",
        actor="preflight",
        stage="capability",
        clock=_clock,
    )
    with pytest.raises(LedgerTransitionError, match="requires artifact evidence"):
        transition_requirement(
            ledger,
            requirement_id,
            "constructed",
            event_key="construct-without-evidence",
            actor="builder",
            stage="placement",
        )
    ledger = transition_requirement(
        ledger,
        requirement_id,
        "constructed",
        event_key="construct",
        actor="builder",
        stage="placement",
        evidence_refs=("object:phase_pylon_0",),
        clock=_clock,
    )
    ledger = transition_requirement(
        ledger,
        requirement_id,
        "verified",
        event_key="verify",
        actor="semantic-verifier",
        stage="semantic",
        evidence_refs=("measurement:phase_pylon_0",),
        clock=_clock,
    )
    ledger = transition_requirement(
        ledger,
        requirement_id,
        "fulfilled",
        event_key="fulfill",
        actor="publication-gate",
        stage="publication",
        evidence_refs=("certificate:phase_pylon_0",),
        clock=_clock,
    )

    with pytest.raises(LedgerTransitionError, match="terminal requirement"):
        transition_requirement(
            ledger,
            requirement_id,
            "failed",
            event_key="rewrite-terminal",
            actor="test",
            stage="test",
            failure_reason="must not rewrite history",
        )


def test_retries_are_idempotent_and_key_reuse_cannot_change_history():
    ledger = initialize_semantic_ledger(_graph(), clock=_clock)
    requirement_id = ledger.entries[0].requirement_id
    first = transition_requirement(
        ledger,
        requirement_id,
        "planned",
        event_key="planner-attempt-1",
        actor="planner",
        stage="planning",
        clock=_clock,
    )
    retry = transition_requirement(
        first,
        requirement_id,
        "planned",
        event_key="planner-attempt-1",
        actor="planner",
        stage="planning",
        expected_revision=0,
        clock=_clock,
    )

    assert retry == first
    assert retry.revision == 1
    with pytest.raises(LedgerTransitionError, match="reused"):
        transition_requirement(
            first,
            requirement_id,
            "failed",
            event_key="planner-attempt-1",
            actor="planner",
            stage="planning",
            failure_reason="changed history",
        )


def test_stale_revision_and_specific_failure_are_enforced():
    ledger = initialize_semantic_ledger(_graph(), clock=_clock)
    requirement_id = ledger.entries[0].requirement_id
    ledger = transition_requirement(
        ledger,
        requirement_id,
        "planned",
        event_key="plan",
        actor="planner",
        stage="planning",
        expected_revision=0,
        clock=_clock,
    )

    with pytest.raises(StaleLedgerRevisionError, match="expected revision 0"):
        transition_requirement(
            ledger,
            requirement_id,
            "strategy_assigned",
            event_key="stale",
            actor="preflight",
            stage="capability",
            expected_revision=0,
        )
    with pytest.raises(ValueError, match="specific failure reason"):
        transition_requirement(
            ledger,
            requirement_id,
            "failed",
            event_key="empty-failure",
            actor="preflight",
            stage="capability",
        )
    failed = transition_requirement(
        ledger,
        requirement_id,
        "failed",
        event_key="no-capability",
        actor="preflight",
        stage="capability",
        failure_reason="No catalog, composition, or procedural strategy is available.",
        clock=_clock,
    )
    assert failed.entries[0].current_status == "failed"


def test_graph_or_entry_tampering_fails_integrity_validation():
    graph = _graph()
    ledger = initialize_semantic_ledger(graph, clock=_clock)
    changed_graph = graph.model_copy(update={"analysis_model": "different-model"})

    with pytest.raises(LedgerIntegrityError, match="graph identity/hash"):
        validate_ledger_against_graph(ledger, changed_graph)
    missing_entry = ledger.model_copy(update={"entries": ledger.entries[1:]})
    with pytest.raises(LedgerIntegrityError, match="requirement set changed"):
        validate_ledger_against_graph(missing_entry, graph)
    changed_entry = ledger.entries[0].model_copy(update={"requirement_hash": "bad"})
    tampered = ledger.model_copy(
        update={"entries": (changed_entry,) + ledger.entries[1:]}
    )
    with pytest.raises(LedgerIntegrityError, match="requirement hash changed"):
        validate_ledger_against_graph(tampered, graph)


def test_serialization_resume_and_legacy_initialization_are_exact(tmp_path):
    graph = _graph()
    path = tmp_path / "semantic_obligation_ledger.json"
    ledger = initialize_semantic_ledger(graph, clock=_clock)
    requirement_id = ledger.entries[0].requirement_id
    ledger = transition_requirement(
        ledger,
        requirement_id,
        "planned",
        event_key="durable-plan",
        actor="planner",
        stage="planning",
        clock=_clock,
    )
    persist_semantic_ledger(ledger, path)

    assert load_semantic_ledger(path) == ledger
    assert load_or_initialize_semantic_ledger(path, graph, clock=_clock) == ledger
    legacy_path = tmp_path / "legacy" / "semantic_obligation_ledger.json"
    initialized = load_or_initialize_semantic_ledger(legacy_path, graph, clock=_clock)
    assert initialized.revision == 0
    assert legacy_path.is_file()


def test_summary_reports_open_closed_failed_and_publishable(tmp_path):
    ledger = initialize_semantic_ledger(_graph(), clock=_clock)
    initial = semantic_ledger_summary(ledger)
    assert not initial.closed
    assert not initial.publishable
    assert set(initial.blocking_unresolved) == {
        entry.requirement_id for entry in ledger.entries
    }

    for entry in tuple(ledger.entries):
        ledger = _advance_to_fulfilled(ledger, entry.requirement_id)
    complete = persist_semantic_ledger_summary(
        ledger, tmp_path / "semantic_obligation_summary.json"
    )
    assert complete.closed
    assert complete.publishable
    assert complete.blocking_failed == ()
    assert (tmp_path / "semantic_obligation_summary.json").is_file()


def test_certified_retry_starts_new_attempt_without_mutating_failed_source():
    graph = _graph()
    source = initialize_semantic_ledger(graph, clock=_clock)
    requirement_id = source.entries[0].requirement_id
    failed = transition_requirement(
        source,
        requirement_id,
        "failed",
        event_key="source-run-failed",
        actor="semantic-publication-gate",
        stage="semantic",
        failure_reason="The source run did not construct the required artifact.",
        clock=_clock,
    )

    retry = start_certified_retry_ledger(failed, graph, clock=_clock)

    assert failed.entries[0].current_status == "failed"
    assert retry is not failed
    assert retry.revision == 0
    assert all(entry.current_status == "extracted" for entry in retry.entries)


def test_certified_retry_keeps_nonfailed_ledger_attempt():
    graph = _graph()
    ledger = initialize_semantic_ledger(graph, clock=_clock)

    assert start_certified_retry_ledger(ledger, graph, clock=_clock) is ledger
