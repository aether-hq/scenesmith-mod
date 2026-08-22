"""Linux Drake parser gate for generated semantic scene products."""

import json
import tempfile
import unittest

from pathlib import Path

from examples.prison_escape.generate_scene import generate_scene
from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.semantics.environment.semantic_environment_compiler import (
    SemanticCompileOptions,
    compile_semantic_environment,
)
from scenesmith.agent_utils.semantics.environment.semantic_environment_details import (
    compile_environment_details,
)
from scenesmith.agent_utils.structure.compiler.writing import write_compiled_structure

try:
    from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser
except ImportError:  # pragma: no cover - unsupported local environments
    AddMultibodyPlantSceneGraph = DiagramBuilder = Parser = None


def _parse_sdf(sdf_path: Path) -> int:
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    model_instances = Parser(plant).AddModels(str(sdf_path))
    plant.Finalize()
    builder.Build()
    if len(model_instances) != 1:
        raise AssertionError(f"expected one model in {sdf_path}")
    return plant.num_collision_geometries()


def _heldout_trial_paths() -> tuple[Path, ...]:
    results = (
        Path(__file__).parents[3]
        / "docs"
        / "geometry-extension"
        / "llm-trials"
        / "results"
    )
    return tuple(sorted(results.glob("heldout_*.json")))


class TestSemanticSceneTrialDiscovery(unittest.TestCase):
    def test_only_retained_trial_records_are_discovered(self) -> None:
        paths = _heldout_trial_paths()
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("semantic_environment", data)


@unittest.skipIf(Parser is None, "pydrake is unavailable on this host")
class TestSemanticSceneDrake(unittest.TestCase):
    def test_prison_escape_semantic_shell_loads_in_drake(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = generate_scene(root)
            self.assertGreater(_parse_sdf(root / manifest["tunnel"]["sdf_path"]), 0)

    def test_heldout_semantic_products_and_collision_policies_load_in_drake(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for result_path in _heldout_trial_paths():
                with self.subTest(trial=result_path.stem):
                    data = json.loads(result_path.read_text(encoding="utf-8"))
                    environment = SemanticEnvironmentSpec.from_dict(
                        data["semantic_environment"]
                    )
                    shell = compile_semantic_environment(
                        environment,
                        options=SemanticCompileOptions(
                            voxel_size=3.0,
                            max_cells=500_000,
                            max_triangles=500_000,
                        ),
                    )
                    shell_paths = write_compiled_structure(
                        shell,
                        root / result_path.stem / "shell",
                        source_content_hash=environment.content_hash(),
                    )
                    self.assertGreater(_parse_sdf(shell_paths.sdf_path), 0)
                    for detail in compile_environment_details(environment).structures:
                        paths = write_compiled_structure(
                            detail,
                            root / result_path.stem / detail.structure_id,
                            source_content_hash=environment.content_hash(),
                        )
                        collision_count = _parse_sdf(paths.sdf_path)
                        if detail.collision_enabled:
                            self.assertGreater(collision_count, 0)
                        else:
                            self.assertEqual(collision_count, 0)


if __name__ == "__main__":
    unittest.main()
