"""Main Objaverse retrieval logic with two-stage process: CLIP -> size ranking."""

import logging
import re

from dataclasses import dataclass

import numpy as np
import trimesh

from scenesmith.agent_utils.assets.asset_semantics import (
    candidate_metadata_text,
    catalog_candidate_is_compatible,
    semantic_families,
)
from scenesmith.agent_utils.objaverse_retrieval.clip_similarity import (
    get_top_k_similar_meshes,
    warm_objaverse_text_encoder,
)
from scenesmith.agent_utils.objaverse_retrieval.config import ObjaverseConfig
from scenesmith.agent_utils.objaverse_retrieval.data_loader import (
    ObjaverseMeshMetadata,
    load_preprocessed_data,
    resolve_catalog_mesh_path,
)

console_logger = logging.getLogger(__name__)

_SEARCH_STOP_WORDS = {"a", "an", "and", "for", "of", "the", "with"}
_SEARCH_SYNONYM_GROUPS = (
    {"armchair", "chair", "seat", "seating", "stool"},
    {"bed", "cot", "examination", "gurney", "treatment"},
    {"cabinet", "cupboard", "locker", "storage", "wardrobe"},
    {
        "computer",
        "console",
        "diagnostic",
        "display",
        "monitor",
        "monitoring",
        "station",
        "terminal",
        "workstation",
    },
    {"couch", "loveseat", "sofa"},
    {"lamp", "light", "lighting", "luminaire"},
    {"screen", "television", "tv"},
)
_SEARCH_EXPANSIONS = {
    token: group for group in _SEARCH_SYNONYM_GROUPS for token in group
}


def _search_tokens(value: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(
            r"[a-z]+", value.lower().replace("armchair", "arm chair")
        )
        if token not in _SEARCH_STOP_WORDS
    }
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_SEARCH_EXPANSIONS.get(token, ()))
    return expanded


@dataclass
class RetrievalCandidate:
    """Candidate mesh for retrieval."""

    uid: str
    """Objaverse mesh UID."""

    mesh: trimesh.Trimesh | None
    """Loaded mesh, or None for a deferred global-catalog candidate."""

    metadata: ObjaverseMeshMetadata
    """Objaverse metadata."""

    clip_score: float
    """CLIP similarity score."""

    bbox_score: float
    """Bounding box size difference score (L1 distance)."""


class ObjaverseRetriever:
    """Objaverse (ObjectThor) asset retrieval system.

    Implements two-stage retrieval:
    1. CLIP semantic filtering (select top-K candidates)
    2. Size-based ranking (rank by dimension match)

    Unlike HSSD, Objaverse does not have pre-computed orientation metadata,
    so meshes are loaded directly without alignment transforms. Downstream
    VLM physics analysis will determine orientation during canonicalization.
    """

    def __init__(self, config: ObjaverseConfig, clip_device: str | None = None) -> None:
        """Initialize Objaverse retriever.

        Args:
            config: Objaverse configuration.
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default.
        """
        self.config = config
        self.clip_device = clip_device
        self.preprocessed_data = load_preprocessed_data(config.preprocessed_path)
        console_logger.info(
            f"Objaverse retriever initialized (clip_device={clip_device})"
        )

    def warmup(self) -> None:
        """Initialize the catalog-specific text encoder before serving requests."""
        warm_objaverse_text_encoder(device=self.clip_device)

    def _calculate_bbox_score(
        self, target_dimensions: np.ndarray, mesh_extents: np.ndarray
    ) -> float:
        """Calculate orientation-invariant bounding box score.

        Since Objaverse meshes are not pre-canonicalized (orientation is determined
        by VLM after retrieval), we sort dimensions before comparing. This ensures
        a mesh stored as (0.5, 0.9, 0.5) matches a target of (0.9, 0.5, 0.5).

        This matches Holodeck's approach.

        Args:
            target_dimensions: Desired dimensions (3,).
            mesh_extents: Actual mesh extents (3,).

        Returns:
            L1 distance score (lower is better).
        """
        sorted_target = np.sort(target_dimensions)
        sorted_extents = np.sort(mesh_extents)
        return float(np.sum(np.abs(sorted_target - sorted_extents)))

    def _load_mesh(self, metadata: ObjaverseMeshMetadata) -> trimesh.Trimesh:
        """Load mesh from Objaverse data directory.

        Unlike HSSD, Objaverse meshes are loaded directly without alignment
        transforms. VLM physics analysis handles orientation downstream.

        Args:
            metadata: Catalog metadata, including an optional explicit mesh path.

        Returns:
            Loaded mesh (Y-up GLB format).
        """
        mesh_path = resolve_catalog_mesh_path(
            data_path=self.config.data_path, metadata=metadata
        )

        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded mesh is not a Trimesh: {type(mesh)}")

        return mesh

    def retrieve(
        self,
        description: str,
        object_type: str,
        desired_dimensions: np.ndarray | None = None,
    ) -> tuple[trimesh.Trimesh, str, float, ObjaverseMeshMetadata]:
        """Retrieve best matching Objaverse mesh for description.

        Two-stage process:
        1. CLIP semantic filtering -> top-K candidates
        2. Size-based ranking -> best dimension match

        Args:
            description: Object description text.
            object_type: Object type (e.g., "FURNITURE", "MANIPULAND").
            desired_dimensions: Optional desired dimensions (width, height, depth).

        Returns:
            Tuple of (mesh, uid, clip_score, metadata) where:
            - mesh: Best matching mesh in Y-up coordinates
            - uid: Objaverse UID identifying the mesh
            - clip_score: CLIP similarity score (0.0 to 1.0)
            - metadata: Objaverse mesh metadata

        Raises:
            ValueError: If no suitable mesh is found.
        """
        candidates = self.retrieve_multiple(
            description=description,
            object_type=object_type,
            desired_dimensions=desired_dimensions,
            max_candidates=1,
        )

        if not candidates:
            raise ValueError(
                f"No suitable mesh found for '{description}' (type={object_type})"
            )

        best = candidates[0]
        return best.mesh, best.uid, best.clip_score, best.metadata

    def retrieve_multiple(
        self,
        description: str,
        object_type: str,
        desired_dimensions: np.ndarray | None = None,
        max_candidates: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve multiple matching Objaverse meshes for description.

        Same two-stage process as retrieve(), but returns all candidates
        sorted by bbox_score instead of just the best one.

        Args:
            description: Object description text.
            object_type: Object type (e.g., "FURNITURE", "MANIPULAND").
            desired_dimensions: Optional desired dimensions (width, depth, height).
            max_candidates: Maximum candidates to return. If None, returns all
                available (up to use_top_k CLIP candidates).

        Returns:
            List of RetrievalCandidate sorted by bbox_score (best first).
            Empty list if no suitable meshes found.
        """
        console_logger.info(
            f"Retrieving multiple Objaverse meshes: description='{description}', "
            f"type={object_type}, dimensions={desired_dimensions}"
        )

        category = self.config.object_type_mapping.get(object_type.upper())
        if category is None:
            console_logger.warning(
                f"Unknown object type: {object_type}. "
                f"Available: {list(self.config.object_type_mapping.keys())}"
            )
            return []

        top_k_meshes = get_top_k_similar_meshes(
            text_description=description,
            preprocessed_data=self.preprocessed_data,
            category=category,
            top_k=self.config.use_top_k,
            device=self.clip_device,
        )

        if not top_k_meshes:
            console_logger.warning(f"No meshes found for category: {category}")
            return []

        console_logger.info(f"Processing {len(top_k_meshes)} CLIP-filtered candidates")

        ranked_metadata: list[tuple[str, float, ObjaverseMeshMetadata, float]] = []

        for uid, clip_score in top_k_meshes:
            metadata = self.preprocessed_data.get_metadata(uid)
            if metadata is None:
                console_logger.warning(f"Metadata not found for mesh {uid}")
                continue

            if desired_dimensions is not None:
                bbox_score = self._calculate_bbox_score(
                    target_dimensions=desired_dimensions,
                    mesh_extents=np.asarray(metadata.bounding_box, dtype=float),
                )
            else:
                bbox_score = 0.0

            compatible, compatibility_reason = catalog_candidate_is_compatible(
                request_text=description,
                candidate_text=candidate_metadata_text(
                    name=metadata.name,
                    description=metadata.description or "",
                    aliases=metadata.aliases,
                    tags=metadata.tags,
                    ontology_path=metadata.ontology_path or "",
                ),
                quality_score=metadata.quality_score,
            )
            if not compatible:
                console_logger.info(
                    "Rejected catalog candidate %s for '%s': %s",
                    uid,
                    description,
                    compatibility_reason,
                )
                continue

            console_logger.debug(
                f"Candidate {uid[:8]}: CLIP={clip_score:.3f}, "
                f"bbox={bbox_score:.3f}, indexed_extents={metadata.bounding_box}"
            )

            ranked_metadata.append((uid, clip_score, metadata, bbox_score))

        # For recognized furniture/object families, prefer curated candidates
        # when the global pool contains them.  Low-score meshes remain a useful
        # fallback for long-tail requests whose category is unknown, but should
        # not outrank a curated chair, bed, table, or other known family merely
        # because its prose description repeats the query more literally.
        if semantic_families(description) and any(
            metadata.quality_score >= 0.70
            for _uid, _clip, metadata, _bbox in ranked_metadata
        ):
            ranked_metadata = [
                entry for entry in ranked_metadata if entry[2].quality_score >= 0.70
            ]

        # ObjectThor historically uses size as the second-stage ranker. Poly
        # Haven has much richer text metadata, so retain semantic relevance as
        # the primary signal and use dimensions as a bounded tie-breaker. This
        # avoids a correctly-sized clock outranking a fire alarm for "fire alarm".
        query_tokens = _search_tokens(description)

        def ranking_score(
            entry: tuple[str, float, ObjaverseMeshMetadata, float],
        ) -> float:
            _uid, clip_score, metadata, bbox_score = entry
            semantic_penalty = 1.0 - clip_score
            metadata_tokens = _search_tokens(
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
            lexical_coverage = (
                len(query_tokens & metadata_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            lexical_penalty = 0.35 * (1.0 - lexical_coverage)
            dimensions = np.asarray(metadata.bounding_box, dtype=float)
            has_dimensions = dimensions.shape == (3,) and np.all(dimensions > 0)
            size_penalty = 0.0
            if desired_dimensions is not None:
                if has_dimensions:
                    relative_size_error = bbox_score / max(
                        float(np.sum(np.abs(desired_dimensions))), 0.1
                    )
                    size_penalty = 0.12 * min(relative_size_error, 3.0)
                else:
                    size_penalty = 0.04
            quality_penalty = 0.08 * (1.0 - metadata.quality_score)
            completeness_penalty = (
                0.10
                if metadata.name.strip().lower() in {"", "none", "unknown"}
                and not (metadata.description or "").strip()
                else 0.0
            )
            return (
                semantic_penalty
                + lexical_penalty
                + size_penalty
                + quality_penalty
                + completeness_penalty
            )

        ranked_metadata.sort(key=ranking_score)

        # Bounding boxes are part of the semantic index. Rank with that metadata
        # first, then load only the assets we will actually return. The previous
        # implementation parsed every top-K GLB (and lazily converted every
        # ObjectThor bundle) before discarding all but one or two candidates.
        # That made a local lookup take several seconds despite sub-millisecond
        # vector ranking.
        candidates: list[RetrievalCandidate] = []
        for uid, clip_score, metadata, bbox_score in ranked_metadata:
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
            if metadata.deferred_loading:
                candidates.append(
                    RetrievalCandidate(
                        uid=uid,
                        mesh=None,
                        metadata=metadata,
                        clip_score=clip_score,
                        bbox_score=bbox_score,
                    )
                )
                continue
            try:
                mesh = self._load_mesh(metadata)
            except Exception as e:
                console_logger.warning(f"Failed to load mesh {uid}: {e}", exc_info=True)
                continue

            candidates.append(
                RetrievalCandidate(
                    uid=uid,
                    mesh=mesh,
                    metadata=metadata,
                    clip_score=clip_score,
                    bbox_score=bbox_score,
                )
            )

        if not candidates:
            console_logger.warning("No valid candidates found after mesh loading")
            return []

        console_logger.info(
            f"Returning {len(candidates)} candidates (sorted by bbox_score)"
        )

        return candidates
