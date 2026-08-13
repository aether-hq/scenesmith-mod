"""OpenAI-compatible providers share one explicit authentication seam."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scenesmith.utils.inference import _client_options, agents_sdk_model_name


class InferenceClientTests(unittest.TestCase):
    def test_standard_provider_uses_bearer_api_key(self) -> None:
        environment = {
            "OPENAI_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://provider.test/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(
                _client_options(),
                {
                    "api_key": "secret",
                    "base_url": "https://provider.test/v1",
                },
            )

    def test_key_scheme_is_applied_without_leaking_into_model_configuration(self) -> None:
        environment = {
            "OPENAI_API_KEY": "secret",
            "SCENESMITH_INFERENCE_AUTH_SCHEME": "key",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(
                _client_options(),
                {
                    "api_key": "openai-compatible",
                    "default_headers": {"Authorization": "Key secret"},
                },
            )

    def test_unknown_auth_scheme_fails_loudly(self) -> None:
        environment = {
            "OPENAI_API_KEY": "secret",
            "SCENESMITH_INFERENCE_AUTH_SCHEME": "magic",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                _client_options()

    def test_gateway_model_id_is_forced_through_openai_provider(self) -> None:
        self.assertEqual(
            agents_sdk_model_name("google/gemini-2.5-flash-lite"),
            "openai/google/gemini-2.5-flash-lite",
        )
        self.assertEqual(agents_sdk_model_name("openai/gpt-5"), "openai/gpt-5")
        self.assertEqual(agents_sdk_model_name("gpt-5"), "gpt-5")
