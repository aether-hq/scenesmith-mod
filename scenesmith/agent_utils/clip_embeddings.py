"""Core CLIP embedding functions using OpenCLIP.

This module provides text and image embedding functions using the OpenCLIP
ViT-H-14-378-quickgelu model with dfn5b pretrained weights (1024 dimensions).
"""

import logging
import time

from pathlib import Path

import numpy as np
import open_clip
import torch

from PIL import Image

from scenesmith.agent_utils.execution_providers import (
    release_torch_cache,
    resolve_torch_device,
)
from scenesmith.agent_utils.provider_model_cache import ProviderModelCache

console_logger = logging.getLogger(__name__)


def _load_clip_model(device: str):
    model_name = "ViT-H-14-378-quickgelu"
    pretrained = "dfn5b"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    console_logger.info(
        "Loaded OpenCLIP model: %s (%s) on %s", model_name, pretrained, device
    )
    return model, tokenizer, preprocess


def _release_clip_model(device: str, _value) -> None:
    release_torch_cache(device, torch_module=torch)


_clip_cache = ProviderModelCache(loader=_load_clip_model, releaser=_release_clip_model)


def _get_clip_model(device: str | None = None):
    """Get cached OpenCLIP model or load if not cached.

    Args:
        device: Target device or provider (for example ``cuda:0``, ``mps``,
            ``cpu``, or ``auto``). If None, uses the shared provider policy.

    Returns:
        Tuple of (model, tokenizer, preprocess, device_str).
    """
    target_device = _resolve_clip_device(device)
    with _clip_cache.use(target_device) as cached:
        return *cached, target_device


def _resolve_clip_device(device: str | None) -> str:
    if device is None and _clip_cache.current_key is not None:
        return _clip_cache.current_key
    return resolve_torch_device(requested=device or "auto", torch_module=torch)


def reset_clip_model_cache() -> None:
    """Release the current CLIP model after active embeddings finish."""

    _clip_cache.reset()


def get_text_embedding(text: str, device: str | None = None) -> np.ndarray:
    """Get CLIP text embedding using OpenCLIP.

    Uses ViT-H-14-378-quickgelu with dfn5b pretrained weights (1024 dimensions).

    Args:
        text: Text to embed.
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Text embedding as NumPy array (1024 dimensions), normalized.
    """
    target_device = _resolve_clip_device(device)
    with _clip_cache.use(target_device) as (model, tokenizer, _):
        text_tokens = tokenizer([text]).to(target_device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]


def warm_clip_text_encoder(device: str | None = None) -> None:
    """Load the shared CLIP model and execute one text-encoder pass.

    Loading only a catalog's precomputed embedding matrix does not initialize
    OpenCLIP.  Retrieval servers call this during startup so model download,
    deserialization, device transfer, and first-kernel compilation happen before
    a latency-bounded request is accepted.
    """
    started_at = time.monotonic()
    embedding = get_text_embedding("retrieval service warmup", device=device)
    if embedding.shape != (1024,):
        raise RuntimeError(
            f"Unexpected CLIP warmup embedding shape: {embedding.shape}"
        )
    console_logger.info(
        "OpenCLIP text encoder warmed in %.3fs", time.monotonic() - started_at
    )


def get_single_image_embedding(
    image_path: Path, device: str | None = None
) -> np.ndarray:
    """Get CLIP embedding for a single image.

    Args:
        image_path: Path to image file.
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Image embedding as NumPy array (1024 dimensions), normalized.
    """
    target_device = _resolve_clip_device(device)
    with _clip_cache.use(target_device) as (model, _, preprocess):
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(target_device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]


def get_multiview_image_embedding(
    image_paths: list[Path], device: str | None = None
) -> np.ndarray:
    """Get averaged CLIP image embedding from multiple views.

    Computes CLIP embeddings for each image and averages them,
    following the standard multi-view embedding approach.

    Args:
        image_paths: List of paths to image files (e.g., 8 rendered views).
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Averaged image embedding as NumPy array (1024 dimensions), normalized.

    Raises:
        ValueError: If image_paths is empty.
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    target_device = _resolve_clip_device(device)
    with _clip_cache.use(target_device) as (model, _, preprocess):
        image_tensors = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image_tensors.append(preprocess(image))
        batch_tensor = torch.stack(image_tensors).to(target_device)
        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        averaged_features = image_features.mean(dim=0)
        averaged_features = averaged_features / averaged_features.norm()
        return averaged_features.cpu().numpy()


def compute_clip_similarities(
    query_embedding: np.ndarray, embeddings: np.ndarray, indices: list[int]
) -> dict[int, float]:
    """Compute cosine similarities between query and candidate embeddings.

    Args:
        query_embedding: Query embedding (D,), should be normalized.
        embeddings: Candidate embeddings array (N, D).
        indices: List of indices in embeddings array to compare against.

    Returns:
        Dictionary mapping index to similarity score.
    """
    # Ensure query is normalized.
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Extract selected embeddings and normalize all at once.
    selected_embeddings = embeddings[indices]
    selected_norms = selected_embeddings / np.linalg.norm(
        selected_embeddings, axis=1, keepdims=True
    )

    # Vectorized dot product.
    similarities = selected_norms @ query_norm

    return dict(zip(indices, similarities))
