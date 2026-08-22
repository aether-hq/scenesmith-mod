"""Regression tests for deterministic catalog semantic guards."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from scenesmith.agent_utils.assets.asset_semantics import (
    catalog_candidate_is_compatible,
    catalog_candidate_satisfies_request_details,
    is_structural_architecture_request,
    tall_furniture_dimensions_are_compatible,
)
from scenesmith.agent_utils.objaverse_retrieval.data_loader import ObjaverseMeshMetadata
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
        candidate_text=("Modern Ceiling Lamp pendant light polyhaven/Lighting/Ceiling"),
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


def test_rejects_books_and_documents_prop_for_storage_furniture() -> None:
    compatible, reason = catalog_candidate_is_compatible(
        request_text=(
            "full-height Renaissance library bookcase densely filled with visible books"
        ),
        candidate_text=(
            "Book Encyclopedia Set 01 books bookshelf encyclopedia library "
            "polyhaven/Office & Stationery/Books & Documents/Books"
        ),
        quality_score=1.0,
    )

    assert not compatible
    assert "document" in reason.casefold()


def test_rejects_intrinsic_cabinet_renamed_as_bookcase() -> None:
    compatible, reason = catalog_candidate_is_compatible(
        request_text=(
            "full-height Renaissance library bookcase densely filled with visible books"
        ),
        candidate_text=(
            "Chinese Cabinet traditional shelving full-height Renaissance library "
            "bookcase densely filled with visible books "
            "polyhaven/Furniture/Storage Furniture/Cabinets & Cupboards"
        ),
        quality_score=1.0,
    )

    assert not compatible
    assert "cabinet" in reason.casefold()


def test_accepts_books_and_documents_prop_for_book_request() -> None:
    compatible, _ = catalog_candidate_is_compatible(
        request_text="vintage leather-bound encyclopedia books",
        candidate_text=(
            "Book Encyclopedia Set 01 polyhaven/Office & Stationery/Books & "
            "Documents/Books"
        ),
        quality_score=1.0,
    )

    assert compatible


def test_full_height_storage_requires_eighty_percent_of_target_height() -> None:
    compatible, reason = tall_furniture_dimensions_are_compatible(
        request_text="full-height Renaissance library bookcase",
        desired_dimensions=(1.0, 0.35, 2.0),
        bbox_min=(-0.414, -0.175, 0.0),
        bbox_max=(0.414, 0.175, 1.242),
    )

    assert not compatible
    assert "80%" in reason


def test_generic_tall_furniture_retains_sixty_percent_threshold() -> None:
    compatible, _ = tall_furniture_dimensions_are_compatible(
        request_text="classical marble statue on pedestal",
        desired_dimensions=(0.7, 0.7, 2.0),
        bbox_min=(-0.35, -0.35, 0.0),
        bbox_max=(0.35, 0.35, 1.242),
    )

    assert compatible


def test_dense_visible_book_request_rejects_non_fillable_storage_metadata() -> None:
    compatible, reason = catalog_candidate_satisfies_request_details(
        request_text=(
            "full-height Renaissance library bookcase densely filled with visible books"
        ),
        candidate_text=(
            "Wooden storage cabinet with closed drawers "
            "polyhaven/Furniture/Storage Furniture/Cabinets"
        ),
        supports_detail_fill=False,
    )

    assert not compatible
    assert "visible books" in reason


def test_dense_visible_book_request_accepts_fillable_catalog_bookcase() -> None:
    compatible, _ = catalog_candidate_satisfies_request_details(
        request_text=(
            "full-height Renaissance library bookcase densely filled with visible books"
        ),
        candidate_text="Dawson Tall Bookcase hssd/wordnet/bookcase.n.01",
        supports_detail_fill=False,
    )

    assert compatible


def test_stationary_chair_request_rejects_rocking_mechanism() -> None:
    rocking, reason = catalog_candidate_is_compatible(
        request_text="stationary upholstered library reading chair",
        candidate_text="Leather Chesterfield rocking chair rocking_chair.n.01",
        quality_score=0.76,
    )
    armchair, _ = catalog_candidate_is_compatible(
        request_text="stationary upholstered library reading chair",
        candidate_text="Wooden upholstered armchair Furniture/Seating/Chairs",
        quality_score=1.0,
    )

    assert not rocking
    assert "stationary" in reason
    assert armchair


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


def test_curated_sculpture_beats_low_quality_literal_statue_match() -> None:
    metadata = {
        "literal": ObjaverseMeshMetadata(
            uid="literal",
            name="Unknown",
            description="A statue of a man standing on a pedestal",
            category="large_objects",
            bounding_box=(0.7, 0.7, 2.0),
            ontology_path="objaverse/large_objects",
            quality_score=0.66,
            deferred_loading=True,
        ),
        "curated": ObjaverseMeshMetadata(
            uid="curated",
            name="Gothic Statue",
            description="Ornate classical stone human figure",
            category="large_objects",
            bounding_box=(1.48, 1.56, 1.74),
            ontology_path=(
                "polyhaven/Decor & Art/Sculptures & Figurines/Busts & Human Figures"
            ),
            quality_score=1.0,
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
            description="classical Renaissance marble statue on pedestal",
            object_type="furniture",
            desired_dimensions=np.array([0.7, 0.7, 2.0]),
            max_candidates=2,
        )

    assert [candidate.uid for candidate in candidates] == ["curated"]


def test_classical_human_statue_rejects_animal_sculpture() -> None:
    animal, reason = catalog_candidate_is_compatible(
        request_text="classical Renaissance marble human figure statue on pedestal",
        candidate_text=(
            "Bronze Whale Statue "
            "polyhaven/Decor & Art/Sculptures & Figurines/Animal Figures"
        ),
        quality_score=1.0,
    )
    human, _ = catalog_candidate_is_compatible(
        request_text="classical Renaissance marble human figure statue on pedestal",
        candidate_text=(
            "Gothic Statue ornate stone figure "
            "polyhaven/Decor & Art/Sculptures & Figurines/Busts & Human Figures"
        ),
        quality_score=1.0,
    )

    assert not animal
    assert "animal" in reason.casefold()
    assert human
