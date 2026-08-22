#!/usr/bin/env python3
"""Enforce bounded Python modules and directories across maintained code."""

from __future__ import annotations

import argparse
import json

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MAX_LINES = 800
DEFAULT_MAX_FILES = 10
DEFAULT_PATHS = ("scenesmith", "tests", "scripts")
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "generated",
    "vendor",
}


@dataclass(frozen=True, order=True)
class ArchitectureViolation:
    kind: str
    path: str
    observed: int
    maximum: int

    @property
    def message(self) -> str:
        noun = "physical lines" if self.kind == "file_lines" else "Python files"
        return f"{self.path}: {self.observed} {noun}; maximum is {self.maximum}"


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRECTORY_NAMES for part in path.parts)


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            if root.suffix == ".py" and not _is_ignored(root):
                yield root
            continue
        if not root.exists() or _is_ignored(root):
            continue
        yield from (
            path
            for path in root.rglob("*.py")
            if path.is_file() and not _is_ignored(path)
        )


def physical_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def scan_architecture(
    repository_root: Path,
    *,
    relative_paths: Sequence[str] = DEFAULT_PATHS,
    max_lines: int = DEFAULT_MAX_LINES,
    max_files: int = DEFAULT_MAX_FILES,
) -> tuple[ArchitectureViolation, ...]:
    """Return every deterministic architecture-policy violation."""

    if max_lines <= 0 or max_files <= 0:
        raise ValueError("architecture limits must be positive")
    roots = tuple(repository_root / path for path in relative_paths)
    files = tuple(sorted(set(iter_python_files(roots))))
    violations: list[ArchitectureViolation] = []
    for path in files:
        line_count = physical_line_count(path)
        if line_count > max_lines:
            violations.append(
                ArchitectureViolation(
                    kind="file_lines",
                    path=str(path.relative_to(repository_root)),
                    observed=line_count,
                    maximum=max_lines,
                )
            )

    files_by_directory: dict[Path, list[Path]] = {}
    for path in files:
        files_by_directory.setdefault(path.parent, []).append(path)
    for directory, children in files_by_directory.items():
        if len(children) > max_files:
            violations.append(
                ArchitectureViolation(
                    kind="directory_files",
                    path=str(directory.relative_to(repository_root)),
                    observed=len(children),
                    maximum=max_files,
                )
            )
    return tuple(sorted(violations))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the script's repository)",
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        help="relative maintained path; repeat to override defaults",
    )
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    violations = scan_architecture(
        root,
        relative_paths=tuple(args.paths or DEFAULT_PATHS),
        max_lines=args.max_lines,
        max_files=args.max_files,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "max_lines": args.max_lines,
                    "max_files": args.max_files,
                    "violation_count": len(violations),
                    "violations": [asdict(violation) for violation in violations],
                },
                indent=2,
            )
        )
    elif violations:
        print(f"Architecture policy failed with {len(violations)} violation(s):")
        for violation in violations:
            print(f"- {violation.message}")
    else:
        print(
            "Architecture policy passed: every Python file is at most "
            f"{args.max_lines} lines and every directory contains at most "
            f"{args.max_files} Python files."
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
