"""Executable architecture boundaries for hardware-specific behavior."""

from __future__ import annotations

import ast
import tomllib
import unittest

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = PROJECT_ROOT / "scenesmith"


def _attribute_name(node: ast.AST) -> str | None:
    """Return a dotted name for an attribute expression when possible."""

    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


class TestHardwareProviderBoundaries(unittest.TestCase):
    def test_hardware_operations_are_confined_by_operation(self) -> None:
        allowed_by_operation = {
            "torch.cuda": {
                Path("agent_utils/execution_providers.py"),
            },
            "torch.backends.mps": {
                Path("agent_utils/execution_providers.py"),
            },
            "CUDA_VISIBLE_DEVICES": {
                Path("agent_utils/execution_providers.py"),
                Path("agent_utils/blender/process_provider.py"),
                Path("agent_utils/geometry_generation_server/execution_provider.py"),
            },
            "CUDA_HOME": {
                Path("agent_utils/geometry_generation_server/cuda_env_setup.py"),
                Path(
                    "agent_utils/geometry_generation_server/sam3d_pipeline_manager.py"
                ),
            },
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": {
                Path("agent_utils/geometry_generation_server/sam_provider.py"),
            },
            "nvidia-smi": {
                Path("agent_utils/execution_providers.py"),
                Path("agent_utils/geometry_generation_server/cuda_env_setup.py"),
            },
            "/dev/nvidia": {
                Path("agent_utils/blender/process_provider.py"),
            },
            "compute_device_type": {
                Path("agent_utils/blender/render_provider.py"),
            },
            "RastContext.cuda": {
                Path(
                    "agent_utils/geometry_generation_server/sam3d_pipeline_manager.py"
                ),
            },
        }
        violations: list[str] = []

        for source_path in SOURCE_ROOT.rglob("*.py"):
            relative = source_path.relative_to(SOURCE_ROOT)
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            operations: set[str] = set()
            for node in ast.walk(tree):
                dotted = _attribute_name(node)
                if dotted and (
                    dotted.startswith("torch.cuda")
                    or dotted.startswith("torch.backends.mps")
                ):
                    operations.add(
                        "torch.cuda"
                        if dotted.startswith("torch.cuda")
                        else "torch.backends.mps"
                    )
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "compute_device_type"
                ):
                    operations.add("compute_device_type")
                if (
                    isinstance(node, ast.Call)
                    and dotted == "utils3d.torch.RastContext"
                    and any(
                        keyword.arg == "backend"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value == "cuda"
                        for keyword in node.keywords
                    )
                ):
                    operations.add("RastContext.cuda")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for marker in (
                        "CUDA_VISIBLE_DEVICES",
                        "CUDA_HOME",
                        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                        "nvidia-smi",
                        "/dev/nvidia",
                    ):
                        if marker in node.value:
                            operations.add(marker)

            for operation in sorted(operations):
                if relative not in allowed_by_operation[operation]:
                    violations.append(f"{relative}: {operation}")

        self.assertEqual(
            violations,
            [],
            "Hardware-specific operations escaped their provider boundary:\n"
            + "\n".join(violations),
        )

    def test_production_identifiers_do_not_expose_numeric_gpu_slots(self) -> None:
        banned = {"gpu_id", "render_gpu_id"}
        violations: list[str] = []
        for source_path in SOURCE_ROOT.rglob("*.py"):
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in ast.walk(tree):
                identifier = None
                if isinstance(node, ast.Name):
                    identifier = node.id
                elif isinstance(node, ast.arg):
                    identifier = node.arg
                elif isinstance(node, ast.Attribute):
                    identifier = node.attr
                if identifier in banned:
                    violations.append(
                        f"{source_path.relative_to(SOURCE_ROOT)}:{node.lineno}: "
                        f"{identifier}"
                    )
        self.assertEqual(violations, [])

    def test_cuda_environment_setup_is_not_an_import_side_effect(self) -> None:
        source_path = (
            SOURCE_ROOT
            / "agent_utils/geometry_generation_server/sam3d_pipeline_manager.py"
        )
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        top_level_calls = {
            _attribute_name(node.value.func)
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        }
        self.assertNotIn("ensure_cuda_env", top_level_calls)

    def test_dependency_profiles_and_ci_cover_portable_and_cuda_hosts(self) -> None:
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        extras = project["project"]["optional-dependencies"]
        self.assertIn("portable", extras)
        self.assertIn("cuda", extras)
        self.assertTrue(any(item.startswith("torch") for item in extras["portable"]))
        self.assertTrue(any(item.startswith("torch") for item in extras["cuda"]))

        uv_config = project["tool"]["uv"]
        conflicts = uv_config["conflicts"]
        self.assertIn(
            [{"extra": "portable"}, {"extra": "cuda"}],
            conflicts,
        )
        torch_sources = uv_config["sources"]["torch"]
        self.assertTrue(any(item.get("extra") == "portable" for item in torch_sources))
        self.assertTrue(any(item.get("extra") == "cuda" for item in torch_sources))

        workflow = (PROJECT_ROOT / ".github/workflows/unit_test.yaml").read_text()
        self.assertIn("uv sync --extra portable", workflow)
        self.assertIn("uv sync --extra cuda", workflow)
        boundary_path = "tests/unit/test_hardware_provider_boundaries.py"
        self.assertIn(boundary_path, workflow)
        self.assertLess(workflow.index(boundary_path), workflow.index("--testmon"))


if __name__ == "__main__":
    unittest.main()
