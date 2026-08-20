"""Tests for passing experiment provider choices into agent subprocesses."""

import unittest

from unittest.mock import MagicMock

from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.experiments.indoor_scene_generation import (
    _asset_config_uses_generated_geometry,
    _resolve_geometry_runtime_configuration,
)


class _FloorAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class TestExperimentProviderWiring(unittest.TestCase):
    def test_render_provider_is_injected_into_agent_without_mutating_config(
        self,
    ) -> None:
        config = {
            "experiment": {
                "execution_providers": {"render": "metal"},
            },
            "floor_plan_agent": {
                "_name": "fake",
                "rendering": {"blender_server_port_range": [8000, 8010]},
            },
        }

        agent = BaseExperiment.build_floor_plan_agent(
            cfg_dict=config,
            compatible_agents={"fake": _FloorAgent},
            logger=MagicMock(),
        )

        self.assertEqual(agent.kwargs["cfg"].rendering.provider, "metal")
        self.assertNotIn("provider", config["floor_plan_agent"]["rendering"])

    def test_generated_agents_must_share_one_geometry_runtime(self) -> None:
        config = {
            "furniture_agent": {
                "asset_manager": {
                    "general_asset_source": "generated",
                    "backend": "sam3d",
                    "sam3d": {"provider": "mlx"},
                }
            },
            "manipuland_agent": {
                "asset_manager": {
                    "general_asset_source": "generated",
                    "backend": "hunyuan3d",
                }
            },
        }

        with self.assertRaisesRegex(ValueError, "same geometry backend"):
            _resolve_geometry_runtime_configuration(config)

    def test_matching_generated_agents_produce_one_server_configuration(self) -> None:
        sam_config = {"provider": "mlx", "mlx_steps": 8}
        config = {
            "furniture_agent": {
                "asset_manager": {
                    "general_asset_source": "generated",
                    "backend": "sam3d",
                    "sam3d": sam_config,
                }
            },
            "manipuland_agent": {
                "asset_manager": {
                    "general_asset_source": "generated",
                    "backend": "sam3d",
                    "sam3d": sam_config,
                }
            },
        }

        backend, resolved = _resolve_geometry_runtime_configuration(config)

        self.assertEqual(backend, "sam3d")
        self.assertEqual(resolved["provider"], "mlx")
        self.assertEqual(resolved["mlx_steps"], 8)

    def test_catalog_only_all_source_does_not_require_geometry_runtime(self) -> None:
        asset_config = {
            "general_asset_source": "all",
            "federated": {
                "source_order": ["polyhaven", "hssd", "objaverse"],
            },
            "router": {"strategies": {"generated": {"enabled": True}}},
        }

        self.assertFalse(_asset_config_uses_generated_geometry(asset_config))

    def test_all_source_requires_geometry_when_generated_is_a_fallback(self) -> None:
        asset_config = {
            "general_asset_source": "all",
            "federated": {
                "source_order": ["polyhaven", "hssd", "generated"],
            },
            "router": {"strategies": {"generated": {"enabled": True}}},
        }

        self.assertTrue(_asset_config_uses_generated_geometry(asset_config))


if __name__ == "__main__":
    unittest.main()
