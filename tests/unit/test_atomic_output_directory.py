"""Tests for failure-safe, clean directory publication."""

import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from scenesmith.agent_utils.atomic_output import rebuild_directory_atomically


class TestAtomicOutputDirectory(unittest.TestCase):
    def test_success_replaces_old_tree_without_retaining_files(self) -> None:
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "generated"
            target.mkdir()
            (target / "stale.txt").write_text("old")

            result = rebuild_directory_atomically(
                target,
                lambda staging: (staging / "fresh.txt").write_text("new"),
            )

            self.assertEqual(result, 3)
            self.assertFalse((target / "stale.txt").exists())
            self.assertEqual((target / "fresh.txt").read_text(), "new")
            self.assertEqual(tuple(target.parent.glob(".generated.*")), ())

    def test_failure_preserves_previous_published_tree(self) -> None:
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "generated"
            target.mkdir()
            (target / "known-good.txt").write_text("preserved")

            def fail(staging: Path) -> None:
                (staging / "partial.txt").write_text("partial")
                raise RuntimeError("compile failed")

            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                rebuild_directory_atomically(target, fail)

            self.assertEqual(
                (target / "known-good.txt").read_text(),
                "preserved",
            )
            self.assertFalse((target / "partial.txt").exists())
            self.assertEqual(tuple(target.parent.glob(".generated.*")), ())

    def test_symlink_target_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            target = root / "generated"
            target.symlink_to(actual, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                rebuild_directory_atomically(target, lambda _: None)


if __name__ == "__main__":
    unittest.main()
