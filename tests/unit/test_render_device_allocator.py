"""Render-device selection must not invent an NVIDIA GPU on CPU workers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scenesmith.experiments.indoor_scene_generation import RenderGPUAllocator


class RenderDeviceAllocatorTests(unittest.TestCase):
    def test_cpu_worker_allocates_no_gpu(self) -> None:
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True),
            patch("pathlib.Path.exists", return_value=False),
        ):
            allocator = RenderGPUAllocator()

        self.assertEqual(allocator.available_gpus, [None])
        self.assertIsNone(allocator.allocate())

    def test_explicit_cuda_devices_remain_round_robin(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}, clear=True):
            allocator = RenderGPUAllocator()

        self.assertEqual(allocator.available_gpus, [2, 5])
        self.assertEqual([allocator.allocate() for _ in range(3)], [2, 5, 2])


if __name__ == "__main__":
    unittest.main()
