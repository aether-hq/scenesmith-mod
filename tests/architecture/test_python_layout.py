"""Tests for the repository architecture policy checker."""

from pathlib import Path

from scripts.architecture.check_python_layout import (
    ArchitectureViolation,
    physical_line_count,
    scan_architecture,
)


def _write_lines(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"line_{index}\n" for index in range(count)))


def test_physical_line_count_includes_comments_and_blank_lines(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("# comment\n\nvalue = 1\n")

    assert physical_line_count(path) == 3


def test_scan_reports_oversized_file(tmp_path):
    _write_lines(tmp_path / "scenesmith" / "oversized.py", 801)

    assert scan_architecture(tmp_path) == (
        ArchitectureViolation(
            kind="file_lines",
            path="scenesmith/oversized.py",
            observed=801,
            maximum=800,
        ),
    )


def test_scan_reports_directory_with_more_than_ten_python_files(tmp_path):
    for index in range(11):
        _write_lines(tmp_path / "tests" / f"test_{index}.py", 1)

    assert scan_architecture(tmp_path) == (
        ArchitectureViolation(
            kind="directory_files",
            path="tests",
            observed=11,
            maximum=10,
        ),
    )


def test_scan_ignores_caches_generated_and_non_python_files(tmp_path):
    for directory in ("__pycache__", "generated", "vendor"):
        _write_lines(tmp_path / "scenesmith" / directory / "ignored.py", 900)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "notes.txt").write_text("text\n" * 900)

    assert scan_architecture(tmp_path) == ()


def test_scan_supports_scoped_paths_and_custom_limits(tmp_path):
    _write_lines(tmp_path / "feature" / "one.py", 3)
    _write_lines(tmp_path / "ignored" / "large.py", 20)

    violations = scan_architecture(
        tmp_path,
        relative_paths=("feature",),
        max_lines=2,
        max_files=1,
    )

    assert violations == (
        ArchitectureViolation(
            kind="file_lines",
            path="feature/one.py",
            observed=3,
            maximum=2,
        ),
    )
