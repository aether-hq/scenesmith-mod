"""The public CPU profile must keep all SceneSmith stages and use ObjectThor."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir


class CpuObjaverseConfigTests(unittest.TestCase):
    def test_profile_routes_every_object_stage_to_objectthor(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configurations"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="cpu_full_objaverse")

        self.assertEqual(cfg.experiment.pipeline.start_stage, "floor_plan")
        self.assertEqual(cfg.experiment.pipeline.stop_stage, "manipuland")
        self.assertFalse(cfg.experiment.pipeline.parallel_rooms)
        for name in (
            "furniture_agent",
            "wall_agent",
            "ceiling_agent",
            "manipuland_agent",
        ):
            self.assertEqual(cfg[name].asset_manager.general_asset_source, "objaverse")
            self.assertFalse(cfg[name].asset_manager.router.enabled)
            natural = cfg[name].placement_noise.natural_profile
            perfect = cfg[name].placement_noise.perfect_profile
            self.assertTrue(all(float(value) == 0.0 for value in natural.values()))
            self.assertTrue(all(float(value) == 0.0 for value in perfect.values()))
        self.assertEqual(cfg.experiment.projection.furniture.solver_name, "ipopt")
        self.assertEqual(cfg.experiment.projection.final.solver_name, "ipopt")
        self.assertEqual(cfg.manipuland_agent.fill_simulation.nlp_solver_name, "ipopt")
        self.assertEqual(
            cfg.manipuland_agent.per_furniture_postprocessing.projection.solver_name,
            "ipopt",
        )


if __name__ == "__main__":
    unittest.main()
