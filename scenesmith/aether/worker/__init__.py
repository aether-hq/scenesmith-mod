"""Durable worker-side orchestration for Aether SceneSmith completion."""

from .author_client import AetherCompletionClient, CompletionAuthorError

__all__ = ["AetherCompletionClient", "CompletionAuthorError"]
