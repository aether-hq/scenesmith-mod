#!/usr/bin/env python3
"""Validate retained SceneSmith runs against the semantic regression corpus."""

from __future__ import annotations

import argparse

from pathlib import Path

from scenesmith.agent_utils.semantic_regression import (
    load_regression_corpus,
    validate_candidate_run,
    validate_reference_run,
)


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "test_data"
    / "semantic_regression"
    / "corpus.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--candidate-run", dest="candidate_run_id")
    parser.add_argument(
        "--operational-run",
        dest="operational_run_id",
        help=(
            "completed full-pipeline run supplying duration and LLM usage "
            "for a resumed candidate"
        ),
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--failed-attempts", type=int, default=0)
    parser.add_argument("--verify-visuals", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus = load_regression_corpus(args.corpus)
    if args.list:
        for case in corpus.cases:
            reference = case.reference.run_id if case.reference else "not captured"
            print(f"{case.case_id}\t{case.category}\t{reference}")
        return 0
    if not args.case_id or args.run_root is None:
        raise SystemExit("--case and --run-root are required unless --list is used")
    if args.candidate_run_id:
        if args.verify_visuals:
            raise SystemExit(
                "--verify-visuals validates immutable references, not candidates"
            )
        result = validate_candidate_run(
            args.corpus,
            case_id=args.case_id,
            run_root=args.run_root,
            candidate_run_id=args.candidate_run_id,
            operational_run_id=args.operational_run_id,
            attempts=args.attempts,
            failed_attempts=args.failed_attempts,
        )
        passed = result.passed
    else:
        result = validate_reference_run(
            args.corpus,
            case_id=args.case_id,
            run_root=args.run_root,
            verify_visuals=args.verify_visuals,
        )
        passed = result.matches_reference
    print(result.model_dump_json(indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
