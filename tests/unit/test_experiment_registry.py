"""Every shipped Hydra experiment config must resolve to an executable experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from scenesmith.experiments import IndoorSceneGenerationExperiment, build_experiment
from scenesmith.experiments.indoor_scene_generation import _accepted_stage_input_prompt


class ExperimentRegistryTests(unittest.TestCase):
    def test_cpu_full_experiment_uses_complete_indoor_pipeline(self) -> None:
        cfg = OmegaConf.create(
            {
                "experiment": {
                    "_name": "aether_cpu_full",
                    "output_dir": "/tmp/scenesmith-test",
                }
            }
        )

        experiment = build_experiment(cfg)

        self.assertIsInstance(experiment, IndoorSceneGenerationExperiment)

    def test_accepted_stage_input_owns_the_scene_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage_input = Path(directory) / "stage-input.json"
            stage_input.write_text(
                json.dumps(
                    {
                        "realization_engine": "scenesmith",
                        "pipeline_profile": "full",
                        "people_allowed": False,
                        "room_prompt": "  accepted semantic environment prompt  ",
                    }
                )
            )

            prompt = _accepted_stage_input_prompt(stage_input)

        self.assertEqual(prompt, "accepted semantic environment prompt")

    def test_accepted_stage_input_rejects_people(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage_input = Path(directory) / "stage-input.json"
            stage_input.write_text(
                json.dumps(
                    {
                        "realization_engine": "scenesmith",
                        "pipeline_profile": "full",
                        "people_allowed": True,
                        "room_prompt": "invalid populated environment",
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "prohibit people"):
                _accepted_stage_input_prompt(stage_input)


if __name__ == "__main__":
    unittest.main()
