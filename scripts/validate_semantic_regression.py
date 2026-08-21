#!/usr/bin/env python3
"""Validate retained SceneSmith runs against the semantic regression corpus."""

from __future__ import annotations

import argparse

from pathlib import Path

from scenesmith.agent_utils.semantic_regression import (
    load_regression_corpus,
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
    result = validate_reference_run(
        args.corpus,
        case_id=args.case_id,
        run_root=args.run_root,
        verify_visuals=args.verify_visuals,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.matches_reference else 1


if __name__ == "__main__":
    raise SystemExit(main())
