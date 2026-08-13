"""Regression tests for dependencies required by structural compilation."""

import importlib.metadata
import tomllib
import unittest

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_SHAPELY_VERSION = Version("2.1.2")


def _requirement_named(values: list[str], name: str) -> Requirement:
    matches = [
        Requirement(value) for value in values if Requirement(value).name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name} requirement, got {matches}")
    return matches[0]


class TestProjectDependencies(unittest.TestCase):
    def test_shapely_is_an_explicit_runtime_and_compatibility_requirement(self) -> None:
        project = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        runtime = _requirement_named(project["project"]["dependencies"], "shapely")
        self.assertTrue(runtime.specifier.contains(MINIMUM_SHAPELY_VERSION))
        self.assertFalse(runtime.specifier.contains(Version("2.1.1")))

        compatibility_requirements = [
            line.strip()
            for line in (REPOSITORY_ROOT / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "-"))
        ]
        compatibility = _requirement_named(compatibility_requirements, "shapely")
        self.assertTrue(compatibility.specifier.contains(MINIMUM_SHAPELY_VERSION))
        self.assertFalse(compatibility.specifier.contains(Version("2.1.1")))

    def test_lockfile_and_installed_environment_supply_required_shapely(self) -> None:
        lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))
        scenesmith = next(
            package for package in lock["package"] if package["name"] == "scenesmith"
        )
        locked = next(
            requirement
            for requirement in scenesmith["metadata"]["requires-dist"]
            if requirement["name"] == "shapely"
        )
        self.assertEqual(locked["specifier"], ">=2.1.2")

        installed = Version(importlib.metadata.version("shapely"))
        self.assertGreaterEqual(installed, MINIMUM_SHAPELY_VERSION)

        from shapely import constrained_delaunay_triangles

        self.assertTrue(callable(constrained_delaunay_triangles))
