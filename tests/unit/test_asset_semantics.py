"""Regression tests for deterministic catalog semantic guards."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from scenesmith.agent_utils.asset_semantics import (
    catalog_candidate_is_compatible,
    is_structural_architecture_request,
)
from scenesmith.agent_utils.objaverse_retrieval.data_loader import (
    ObjaverseMeshMetadata,
)
from scenesmith.agent_utils.objaverse_retrieval.retrieval import ObjaverseRetriever


def test_rejects_coffee_table_for_spiral_staircase() -> None:
    compatible, reason = catalog_candidate_is_compatible(
        request_text="Modern spiral staircase structure with metal railing",
        candidate_text=(
            "Coffee Table Round 01 round coffee table "
            "polyhaven/Furniture/Tables/Coffee Tables"
        ),
        quality_score=1.0,
    )

    assert not compatible
    assert "mismatch" in reason


def test_rejects_ceiling_lamp_for_wall_mirror() -> None:
    compatible, _ = catalog_candidate_is_compatible(
        request_text="Modern round wall mirror with brushed metal frame",
        candidate_text=(
            "Modern Ceiling Lamp pendant light polyhaven/Lighting/Ceiling"
        ),
        quality_score=1.0,
    )

    assert not compatible


def test_rejects_composite_table_chair_set_for_table_only() -> None:
    compatible, reason = catalog_candidate_is_compatible(
        request_text="Professional architect drafting table",
        candidate_text="Outdoor table and two chairs Furniture/Seating",
        quality_score=1.0,
    )

    assert not compatible
    assert "composite" in reason


def test_accepts_chair_and_bookshelf_ontology_matches() -> None:
    chair, _ = catalog_candidate_is_compatible(
        request_text="Ergonomic library study chair",
        candidate_text="An ergonomic office chair Furniture/Seating/Chairs",
        quality_score=0.76,
    )
    shelf, _ = catalog_candidate_is_compatible(
        request_text="Tall library bookshelf",
        candidate_text="Steel shelves Furniture/Storage/Shelving Bookcases",
        quality_score=1.0,
    )

    assert chair
    assert shelf


def test_quality_floor_rejects_weak_catalog_mesh() -> None:
    compatible, reason = catalog_candidate_is_compatible(
        request_text="Ergonomic study chair",
        candidate_text="Ergonomic office chair Furniture/Seating/Chair",
        quality_score=0.66,
        minimum_quality=0.70,
    )

    assert not compatible
    assert "quality" in reason


def test_structural_requests_are_not_furniture() -> None:
    assert is_structural_architecture_request("spiral staircase with railing")
    assert is_structural_architecture_request("raised mezzanine platform")
    assert not is_structural_architecture_request("round wall mirror")
    assert not is_structural_architecture_request("standing floor lamp")


def test_curated_candidate_beats_low_quality_literal_match_for_known_family() -> None:
    metadata = {
        "literal": ObjaverseMeshMetadata(
            uid="literal",
            name="Unknown",
            description="An ergonomic office chair",
            category="large_objects",
            bounding_box=(0.6, 0.6, 1.1),
            ontology_path="objaverse/large_objects",
            quality_score=0.66,
            deferred_loading=True,
        ),
        "curated": ObjaverseMeshMetadata(
            uid="curated",
            name="Job Office Chair",
            description="Swivel office chair",
            category="large_objects",
            bounding_box=(0.6, 0.6, 1.1),
            ontology_path="hssd/wordnet/swivel_chair.n.01",
            quality_score=0.76,
            deferred_loading=True,
        ),
    }
    retriever = object.__new__(ObjaverseRetriever)
    retriever.config = SimpleNamespace(
        object_type_mapping={"FURNITURE": "large_objects"}, use_top_k=50
    )
    retriever.clip_device = "cpu"
    retriever.preprocessed_data = MagicMock()
    retriever.preprocessed_data.get_metadata.side_effect = metadata.get

    with patch(
        "scenesmith.agent_utils.objaverse_retrieval.retrieval.get_top_k_similar_meshes",
        return_value=[("literal", 1.0), ("curated", 0.9)],
    ):
        candidates = retriever.retrieve_multiple(
            description="Ergonomic library study chair",
            object_type="furniture",
            desired_dimensions=np.array([0.6, 0.6, 1.1]),
            max_candidates=2,
        )

    assert [candidate.uid for candidate in candidates] == ["curated"]
