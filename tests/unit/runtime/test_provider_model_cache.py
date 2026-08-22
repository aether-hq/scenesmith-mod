"""Tests for provider-aware model cache lifecycle."""

import threading
import unittest

from concurrent.futures import ThreadPoolExecutor

from scenesmith.agent_utils.runtime.provider_model_cache import ProviderModelCache


class TestProviderModelCache(unittest.TestCase):
    def test_concurrent_load_for_one_provider_happens_once(self) -> None:
        load_count = 0
        load_lock = threading.Lock()

        def load(key):
            nonlocal load_count
            with load_lock:
                load_count += 1
            return f"model:{key}"

        cache = ProviderModelCache(loader=load)

        def use_model(_):
            with cache.use("cpu") as model:
                return model

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(use_model, range(32)))

        self.assertEqual(load_count, 1)
        self.assertEqual(set(results), {"model:cpu"})

    def test_provider_transition_releases_old_model_after_last_lease(self) -> None:
        released = []
        cache = ProviderModelCache(
            loader=lambda key: f"model:{key}",
            releaser=lambda key, model: released.append((key, model)),
        )

        with cache.use("mps") as first:
            self.assertEqual(first, "model:mps")
            with cache.use("cpu") as second:
                self.assertEqual(second, "model:cpu")
                self.assertEqual(released, [])
            self.assertEqual(released, [])

        self.assertEqual(released, [("mps", "model:mps")])

    def test_reset_releases_idle_current_model(self) -> None:
        released = []
        cache = ProviderModelCache(
            loader=lambda key: object(),
            releaser=lambda key, model: released.append(key),
        )
        with cache.use("cpu"):
            pass

        cache.reset()

        self.assertEqual(released, ["cpu"])
        self.assertIsNone(cache.current_key)


if __name__ == "__main__":
    unittest.main()
