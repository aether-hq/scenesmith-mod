"""CLIP-based similarity search for Objaverse meshes.

Note: Objaverse uses ViT-L/14 embeddings (768-dim) which are different from
the default ViT-H-14 (1024-dim). We use a separate CLIP model loader that matches the
pre-computed embeddings.
"""

import logging
import re
import time

import numpy as np
import open_clip
import torch

from scenesmith.agent_utils.execution_providers import (
    release_torch_cache,
    resolve_torch_device,
)
from scenesmith.agent_utils.objaverse_retrieval.data_loader import (
    ObjaversePreprocessedData,
)
from scenesmith.agent_utils.provider_model_cache import ProviderModelCache

console_logger = logging.getLogger(__name__)

_LEXICAL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "of",
    "the",
    "with",
}


def _catalog_tokens(value: str) -> set[str]:
    normalized = value.lower().replace("armchair", "arm chair")
    return {
        token
        for token in re.findall(r"[a-z]+", normalized)
        if token not in _LEXICAL_STOP_WORDS
    }


def _load_objaverse_clip_model(device: str):
    model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    console_logger.info(
        "Loaded Objaverse CLIP model: %s (%s) on %s",
        model_name,
        pretrained,
        device,
    )
    return model, tokenizer


def _release_objaverse_clip_model(device: str, _value) -> None:
    release_torch_cache(device, torch_module=torch)


_objaverse_clip_cache = ProviderModelCache(
    loader=_load_objaverse_clip_model,
    releaser=_release_objaverse_clip_model,
)


def _get_objaverse_clip_model(device: str | None = None):
    """Get cached OpenCLIP model for Objaverse (ViT-L/14) or load if not cached.

    ObjectThor embeddings were computed with ViT-L/14 using laion2b_s32b_b82k
    pretrained weights (768 dimensions). This is different from:
    - HSSD: ViT-H-14-378-quickgelu (1024 dimensions)
    - OpenAI's original CLIP: ViT-L/14 with openai weights

    Using the wrong pretrained weights will result in embedding space mismatch
    and poor retrieval results, even if dimensions match.

    Args:
        device: Target device or provider (for example ``cuda:0``, ``mps``,
            ``cpu``, or ``auto``). If None, uses the shared provider policy.

    Returns:
        Tuple of (model, tokenizer, device_str).
    """
    target_device = _resolve_objaverse_clip_device(device)
    with _objaverse_clip_cache.use(target_device) as cached:
        return *cached, target_device


def _resolve_objaverse_clip_device(device: str | None) -> str:
    if device is None and _objaverse_clip_cache.current_key is not None:
        return _objaverse_clip_cache.current_key
    return resolve_torch_device(requested=device or "auto", torch_module=torch)


def reset_objaverse_clip_model_cache() -> None:
    _objaverse_clip_cache.reset()


def get_objaverse_text_embedding(text: str, device: str | None = None) -> np.ndarray:
    """Get CLIP text embedding for Objaverse matching.

    Uses ViT-L/14 with laion2b_s32b_b82k pretrained weights (768 dimensions)
    to match the pre-computed ObjectThor embeddings.

    Args:
        text: Text to embed.
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Text embedding as NumPy array (768 dimensions), normalized.
    """
    return get_objaverse_text_embeddings([text], device=device)[0]


def warm_objaverse_text_encoder(device: str | None = None) -> None:
    """Load and execute Objaverse's distinct ViT-L/14 text encoder once."""
    started_at = time.monotonic()
    embedding = get_objaverse_text_embedding(
        "retrieval service warmup", device=device
    )
    if embedding.shape != (768,):
        raise RuntimeError(
            f"Unexpected Objaverse CLIP warmup embedding shape: {embedding.shape}"
        )
    console_logger.info(
        "Objaverse CLIP text encoder warmed in %.3fs",
        time.monotonic() - started_at,
    )


def get_objaverse_text_embeddings(
    texts: list[str], device: str | None = None, batch_size: int = 64
) -> np.ndarray:
    """Embed catalog metadata in the same normalized space used for queries."""
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    target_device = _resolve_objaverse_clip_device(device)
    batches: list[np.ndarray] = []
    with _objaverse_clip_cache.use(target_device) as (model, tokenizer):
        for offset in range(0, len(texts), batch_size):
            text_tokens = tokenizer(texts[offset : offset + batch_size]).to(
                target_device
            )
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )
            batches.append(text_features.cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def filter_meshes_by_category(
    preprocessed_data: ObjaversePreprocessedData, category: str
) -> list[int]:
    """Filter mesh indices by object category.

    Args:
        preprocessed_data: Loaded preprocessed data.
        category: Object category (e.g., "large_objects", "small_objects").

    Returns:
        List of mesh embedding indices for meshes in this category.
    """
    if category not in preprocessed_data.object_categories:
        console_logger.warning(
            f"Category {category} not found in object_categories. "
            f"Available: {list(preprocessed_data.object_categories.keys())}"
        )
        return []

    uid_list = preprocessed_data.object_categories[category]

    mesh_indices = []
    for uid in uid_list:
        idx = preprocessed_data.get_embedding_index(uid)
        if idx is not None:
            mesh_indices.append(idx)

    console_logger.info(
        f"Filtered {len(mesh_indices)} meshes for category '{category}'"
    )

    return mesh_indices


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
    norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
    # Avoid division by zero.
    norms = np.maximum(norms, 1e-8)
    selected_norms = selected_embeddings / norms

    # Vectorized dot product.
    similarities = selected_norms @ query_norm

    return dict(zip(indices, similarities))


def get_top_k_similar_meshes(
    text_description: str,
    preprocessed_data: ObjaversePreprocessedData,
    category: str | None = None,
    top_k: int = 5,
    device: str | None = None,
) -> list[tuple[str, float]]:
    """Get top-K most similar meshes to text description.

    Args:
        text_description: Object description text.
        preprocessed_data: Loaded preprocessed data.
        category: Optional object category to filter by.
        top_k: Number of top candidates to return.
        device: Target CLIP device (e.g., "cuda:0"). If None, uses default.

    Returns:
        List of (uid, similarity_score) tuples, sorted by descending similarity.
    """
    console_logger.info(
        f"Computing CLIP similarities for '{text_description}' "
        f"(category={category}, top_k={top_k})"
    )

    text_embedding = get_objaverse_text_embedding(text_description, device=device)

    if category:
        mesh_indices = filter_meshes_by_category(preprocessed_data, category)
    else:
        mesh_indices = list(range(len(preprocessed_data.embedding_index)))

    if not mesh_indices:
        console_logger.warning("No meshes to search")
        return []

    similarities = compute_clip_similarities(
        query_embedding=text_embedding,
        embeddings=preprocessed_data.clip_embeddings,
        indices=mesh_indices,
    )

    # The normalized catalog supplies names, descriptions, aliases, tags, and
    # ontology paths. A bounded lexical boost corrects common CLIP near-neighbor
    # errors without replacing the semantic embedding signal.
    query_tokens = _catalog_tokens(text_description)
    for mesh_idx in mesh_indices:
        uid = preprocessed_data.embedding_index[mesh_idx]
        metadata = preprocessed_data.get_metadata(uid)
        if metadata is None:
            continue
        catalog_tokens = _catalog_tokens(
            " ".join(
                [
                    metadata.name,
                    metadata.description or "",
                    *metadata.aliases,
                    *metadata.tags,
                    metadata.ontology_path or "",
                ]
            )
        )
        denominator = min(len(query_tokens), len(catalog_tokens))
        if denominator:
            overlap = len(query_tokens & catalog_tokens) / denominator
            similarities[mesh_idx] += 0.12 * overlap

    source_groups: dict[str, list[tuple[int, float]]] = {}
    for mesh_idx, similarity in similarities.items():
        uid = preprocessed_data.embedding_index[mesh_idx]
        metadata = preprocessed_data.get_metadata(uid)
        source = metadata.asset_source if metadata is not None else "unknown"
        source_groups.setdefault(source, []).append((mesh_idx, similarity))

    if len(source_groups) == 1:
        top_k_items = sorted(
            similarities.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
    else:
        # Raw CLIP score distributions differ by source modality (some indexes
        # contain rendered-image embeddings, others curated text). Normalize by
        # rank within each source before merging so one modality cannot drown out
        # every other catalog. Keep a wide per-source pool, then globally rerank.
        normalized: list[tuple[int, float]] = []
        per_source_pool = max(top_k, 12)
        for items in source_groups.values():
            ranked = sorted(items, key=lambda item: item[1], reverse=True)[
                :per_source_pool
            ]
            denominator = max(len(ranked) - 1, 1)
            for rank, (mesh_idx, _raw_similarity) in enumerate(ranked):
                normalized.append((mesh_idx, 1.0 - rank / denominator))
        top_k_items = sorted(
            normalized,
            key=lambda item: (
                item[1],
                preprocessed_data.embedding_index[item[0]],
            ),
            reverse=True,
        )[:top_k]

    results = []
    for mesh_idx, similarity in top_k_items:
        uid = preprocessed_data.embedding_index[mesh_idx]
        results.append((uid, similarity))

    console_logger.info(
        f"Top-{len(results)} CLIP candidates: "
        f"{[(uid[:8], f'{sim:.3f}') for uid, sim in results]}"
    )

    return results
