"""Tests for the platform-neutral SAM provider facade."""

import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from PIL import Image

from scenesmith.agent_utils.geometry_generation_server.sam_provider import (
    _create_foreground_mask,
    _generate_with_mlx,
    resolve_sam_provider,
    sam_provider_config_from_mapping,
    validate_sam_provider_config,
)


class TestSamProvider(unittest.TestCase):
    def test_auto_selects_mlx_on_apple_silicon(self) -> None:
        self.assertEqual(
            resolve_sam_provider({}, system="Darwin", machine="arm64"), "mlx"
        )

    def test_auto_requires_detected_cuda_off_apple_silicon(self) -> None:
        self.assertEqual(
            resolve_sam_provider(
                {},
                system="Linux",
                machine="x86_64",
                cuda_detector=lambda: (0,),
            ),
            "cuda",
        )

    def test_auto_fails_without_a_capable_local_provider(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no supported local provider"):
            resolve_sam_provider(
                {},
                system="Linux",
                machine="x86_64",
                cuda_detector=lambda: (),
            )

    def test_environment_override_wins(self) -> None:
        with patch.dict("os.environ", {"SCENESMITH_SAM_PROVIDER": "cuda"}):
            self.assertEqual(
                resolve_sam_provider(
                    {"provider": "mlx"}, system="Darwin", machine="arm64"
                ),
                "cuda",
            )

    def test_mps_alias_resolves_to_mlx(self) -> None:
        self.assertEqual(resolve_sam_provider({"provider": "mps"}), "mlx")

    def test_config_is_process_safe_and_carries_mlx_options(self) -> None:
        config = sam_provider_config_from_mapping(
            {"provider": "mlx", "mlx_steps": 8}, object_description="wood chair"
        )
        self.assertEqual(config["provider"], "mlx")
        self.assertEqual(config["mlx_steps"], 8)
        self.assertEqual(config["object_description"], "wood chair")
        self.assertEqual(config["mlx_mps_high_watermark_ratio"], 0.8)

    def test_mlx_memory_policy_is_bounded_and_provider_owned(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "provider"
            output = root / "output.glb"
            output.write_bytes(b"fixture")
            config = {
                "mlx_repo_path": repo,
                "mlx_python_path": repo / "python",
                "mlx_checkpoint_dir": repo / "checkpoints" / "hf",
                "mlx_mps_high_watermark_ratio": 0.75,
            }
            completed = MagicMock(returncode=0, stdout="")
            with (
                patch(
                    "scenesmith.agent_utils.geometry_generation_server.sam_provider.validate_sam_provider_config"
                ),
                patch(
                    "scenesmith.agent_utils.geometry_generation_server.sam_provider._create_foreground_mask"
                ),
                patch(
                    "scenesmith.agent_utils.geometry_generation_server.sam_provider.subprocess.run",
                    return_value=completed,
                ) as run,
                patch.dict(
                    "os.environ",
                    {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"},
                    clear=True,
                ),
            ):
                _generate_with_mlx(
                    image_path=root / "input.png",
                    output_path=output,
                    config=config,
                    debug_folder=None,
                )

            self.assertEqual(
                run.call_args.kwargs["env"]["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
                "0.75",
            )
            self.assertLess(
                float(
                    run.call_args.kwargs["env"][
                        "PYTORCH_MPS_LOW_WATERMARK_RATIO"
                    ]
                ),
                0.75,
            )
            self.assertNotIn("__PYVENV_LAUNCHER__", run.call_args.kwargs["env"])
            self.assertEqual(run.call_args.kwargs["env"]["SPARSE_BACKEND"], "mps")
            self.assertEqual(
                run.call_args.kwargs["env"]["SPARSE_ATTN_BACKEND"], "sdpa"
            )

    def test_unbounded_mlx_memory_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "watermark"):
            sam_provider_config_from_mapping(
                {"provider": "mlx", "mlx_mps_high_watermark_ratio": 0.0}
            )

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected auto, mlx, or cuda"):
            resolve_sam_provider({"provider": "vulkan"})

    def test_uniform_background_mask_keeps_main_object(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.full((64, 64, 3), 255, dtype=np.uint8)
            pixels[18:48, 20:46] = (120, 45, 20)
            image_path = root / "asset.png"
            mask_path = root / "mask.png"
            Image.fromarray(pixels).save(image_path)

            _create_foreground_mask(image_path, mask_path, color_threshold=24.0)

            mask = np.asarray(Image.open(mask_path))
            self.assertEqual(mask[32, 32], 255)
            self.assertEqual(mask[0, 0], 0)
            self.assertGreater(mask.mean(), 20)
            self.assertLess(mask.mean(), 100)

    def test_mlx_validation_accepts_isolated_runtime_and_weights(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "provider"
            python = repo / ".venv" / "bin" / "python"
            checkpoints = repo / "checkpoints" / "hf"
            python.parent.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            (repo / "main.py").write_text("# fixture\n")
            python.write_text("#!/bin/sh\n")
            (checkpoints / "pipeline.yaml").write_text("{}\n")
            (checkpoints / "model.safetensors").write_text("fixture\n")

            provider = validate_sam_provider_config(
                {
                    "provider": "mlx",
                    "mlx_repo_path": repo,
                    "mlx_python_path": python,
                    "mlx_checkpoint_dir": checkpoints,
                }
            )
            self.assertEqual(provider, "mlx")


if __name__ == "__main__":
    unittest.main()
