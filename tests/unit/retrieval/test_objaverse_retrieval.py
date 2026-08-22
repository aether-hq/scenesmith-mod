"""Unit tests for Objaverse (ObjectThor) retrieval module."""

import gzip
import json
import pickle
import shutil
import sqlite3
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh
import yaml

from PIL import Image

from scenesmith.agent_utils.objaverse_retrieval.clip_similarity import (
    compute_clip_similarities,
    filter_meshes_by_category,
)
from scenesmith.agent_utils.objaverse_retrieval.config import ObjaverseConfig
from scenesmith.agent_utils.objaverse_retrieval.data_loader import (
    ObjaverseMeshMetadata,
    ObjaversePreprocessedData,
    construct_objaverse_mesh_path,
    load_preprocessed_data,
    resolve_catalog_mesh_path,
)
from scenesmith.agent_utils.objaverse_retrieval.retrieval import ObjaverseRetriever


class TestObjaverseConfig(unittest.TestCase):
    """Test Objaverse configuration validation logic."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_missing_data_path(self):
        """Test config validation fails when data path doesn't exist."""
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()

        with self.assertRaises(FileNotFoundError) as cm:
            ObjaverseConfig(
                data_path=self.tmp_path / "nonexistent",
                preprocessed_path=preprocessed_path,
            )
        self.assertIn("Objaverse data path does not exist", str(cm.exception))

    def test_config_missing_preprocessed_path(self):
        """Test config validation fails when preprocessed path doesn't exist."""
        data_path = self.tmp_path / "objathor-assets"
        data_path.mkdir()

        with self.assertRaises(FileNotFoundError) as cm:
            ObjaverseConfig(
                data_path=data_path,
                preprocessed_path=self.tmp_path / "nonexistent",
            )
        self.assertIn("Preprocessed data path does not exist", str(cm.exception))

    def test_config_valid(self):
        """Test config creation with valid paths."""
        data_path = self.tmp_path / "objathor-assets"
        data_path.mkdir()
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()

        config = ObjaverseConfig(
            data_path=data_path,
            preprocessed_path=preprocessed_path,
            use_top_k=10,
        )

        self.assertEqual(config.data_path, data_path)
        self.assertEqual(config.preprocessed_path, preprocessed_path)
        self.assertEqual(config.use_top_k, 10)


class TestMeshPaths(unittest.TestCase):
    """Test Objaverse mesh path construction logic."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_construct_mesh_path(self):
        """Test path construction for Objaverse meshes."""
        objaverse_dir = self.tmp_path / "objathor-assets"
        uid = "abc123def456"

        # ObjectThor stores assets in assets/ subdirectory.
        expected_path = objaverse_dir / "assets" / uid / f"{uid}.glb"
        expected_path.parent.mkdir(parents=True)
        expected_path.touch()

        path = construct_objaverse_mesh_path(objaverse_dir, uid)
        self.assertEqual(path, expected_path)

    def test_construct_mesh_path_not_found(self):
        """Test validation fails when mesh file doesn't exist."""
        objaverse_dir = self.tmp_path / "objathor-assets"
        uid = "nonexistent123"

        with self.assertRaises(FileNotFoundError) as cm:
            construct_objaverse_mesh_path(objaverse_dir, uid)
        self.assertIn("Objaverse mesh not found", str(cm.exception))

    def test_resolve_generic_catalog_mesh_path(self):
        """Catalog metadata can point at an authored GLTF outside ObjectThor."""
        catalog_dir = self.tmp_path / "polyhaven"
        mesh_path = catalog_dir / "models" / "chair" / "chair_2k.gltf"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.touch()
        metadata = ObjaverseMeshMetadata(
            uid="polyhaven__chair",
            name="Chair",
            category="large_objects",
            bounding_box=(0.8, 0.8, 1.0),
            mesh_path="models/chair/chair_2k.gltf",
            asset_source="polyhaven",
            license="CC0-1.0",
        )

        self.assertEqual(resolve_catalog_mesh_path(catalog_dir, metadata), mesh_path)

    def test_construct_mesh_path_converts_objathor_bundle(self):
        """ObjectThor's native pickle and textures are converted and cached."""
        objaverse_dir = self.tmp_path / "objathor-assets"
        uid = "native123"
        asset_dir = objaverse_dir / "assets" / uid
        asset_dir.mkdir(parents=True)
        asset = {
            "vertices": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 1.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 1.0, "z": 0.0},
            ],
            "triangles": [0, 1, 2],
            "normals": [
                {"x": 0.0, "y": 0.0, "z": 1.0},
                {"x": 0.0, "y": 0.0, "z": 1.0},
                {"x": 0.0, "y": 0.0, "z": 1.0},
            ],
            "uvs": [
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 0.0},
                {"x": 0.0, "y": 1.0},
            ],
        }
        with gzip.open(asset_dir / f"{uid}.pkl.gz", "wb") as output:
            pickle.dump(asset, output)
        Image.new("RGB", (2, 2), (180, 90, 40)).save(asset_dir / "albedo.jpg")

        path = construct_objaverse_mesh_path(objaverse_dir, uid)
        self.assertEqual(path, asset_dir / f"{uid}.glb")
        self.assertTrue(path.exists())
        loaded = trimesh.load(path, force="scene")
        self.assertEqual(len(loaded.geometry), 1)

        # A second call reuses the cached conversion.
        self.assertEqual(construct_objaverse_mesh_path(objaverse_dir, uid), path)


class TestClipSimilarity(unittest.TestCase):
    """Test CLIP similarity computation and filtering algorithms."""

    def test_compute_clip_similarities(self):
        """Test cosine similarity computation ranks embeddings correctly."""
        query_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        mesh_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # similarity = 1.0
                [0.0, 1.0, 0.0, 0.0],  # similarity = 0.0
                [0.7, 0.7, 0.0, 0.0],  # similarity = 0.7
            ]
        )
        # Normalize for cosine similarity.
        mesh_embeddings[2] /= np.linalg.norm(mesh_embeddings[2])

        mesh_indices = [0, 1, 2]

        similarities = compute_clip_similarities(
            query_embedding=query_embedding,
            embeddings=mesh_embeddings,
            indices=mesh_indices,
        )

        # Should return dict with index → similarity.
        self.assertEqual(len(similarities), 3)
        # Index 0 has highest similarity (1.0).
        self.assertGreater(similarities[0], similarities[1])
        self.assertGreater(similarities[0], similarities[2])
        # Index 2 has higher similarity than index 1.
        self.assertGreater(similarities[2], similarities[1])

    def test_filter_meshes_by_category(self):
        """Test filtering returns correct mesh indices for object category."""
        preprocessed_data = ObjaversePreprocessedData(
            metadata_by_category={
                "large_objects": [
                    ObjaverseMeshMetadata(
                        uid="uid1",
                        name="Chair",
                        category="large_objects",
                        bounding_box=(0.5, 0.8, 0.5),
                    ),
                    ObjaverseMeshMetadata(
                        uid="uid2",
                        name="Table",
                        category="large_objects",
                        bounding_box=(1.0, 0.75, 0.6),
                    ),
                ],
                "small_objects": [
                    ObjaverseMeshMetadata(
                        uid="uid3",
                        name="Apple",
                        category="small_objects",
                        bounding_box=(0.08, 0.08, 0.08),
                    ),
                ],
            },
            clip_embeddings=np.zeros((3, 768)),
            embedding_index=["uid1", "uid2", "uid3"],
            object_categories={
                "large_objects": ["uid1", "uid2"],
                "small_objects": ["uid3"],
            },
        )

        # Filter for large_objects.
        indices = filter_meshes_by_category(preprocessed_data, "large_objects")
        self.assertEqual(len(indices), 2)
        self.assertIn(0, indices)  # uid1
        self.assertIn(1, indices)  # uid2

        # Filter for small_objects.
        indices = filter_meshes_by_category(preprocessed_data, "small_objects")
        self.assertEqual(len(indices), 1)
        self.assertIn(2, indices)  # uid3

    def test_filter_meshes_invalid_category(self):
        """Test filtering returns empty list for invalid category."""
        preprocessed_data = ObjaversePreprocessedData(
            metadata_by_category={},
            clip_embeddings=np.zeros((0, 768)),
            embedding_index=[],
            object_categories={"large_objects": []},
        )

        indices = filter_meshes_by_category(preprocessed_data, "invalid_category")
        self.assertEqual(len(indices), 0)


class TestDataLoader(unittest.TestCase):
    """Test file I/O and data loading logic."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_preprocessed_data(self):
        """Test loading preprocessed data from files."""
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()

        # Create test files.
        embeddings = np.random.rand(2, 768).astype(np.float32)
        np.save(preprocessed_path / "clip_embeddings.npy", embeddings)

        # Embedding index is just a list of UIDs.
        embedding_index = ["uid1", "uid2"]
        with open(preprocessed_path / "embedding_index.yaml", "w") as f:
            yaml.dump(embedding_index, f)

        metadata_index = {
            "uid1": {
                "name": "Chair",
                "category": "large_objects",
                "description": "A wooden chair",
                "bounding_box": [0.5, 0.8, 0.5],
            },
            "uid2": {
                "name": "Apple",
                "category": "small_objects",
                "description": "A red apple",
                "bounding_box": [0.08, 0.08, 0.08],
            },
        }
        with open(preprocessed_path / "metadata_index.json", "w") as f:
            json.dump(metadata_index, f)

        categories = {
            "large_objects": ["uid1"],
            "small_objects": ["uid2"],
        }
        with open(preprocessed_path / "object_categories.json", "w") as f:
            json.dump(categories, f)

        # Load and verify.
        data = load_preprocessed_data(preprocessed_path)

        # Data is organized by category, not by UID.
        self.assertIn("large_objects", data.metadata_by_category)
        self.assertIn("small_objects", data.metadata_by_category)
        self.assertEqual(len(data.metadata_by_category["large_objects"]), 1)
        self.assertEqual(len(data.metadata_by_category["small_objects"]), 1)
        self.assertEqual(data.metadata_by_category["large_objects"][0].name, "Chair")
        self.assertEqual(data.metadata_by_category["small_objects"][0].uid, "uid2")
        self.assertEqual(data.clip_embeddings.shape, (2, 768))
        self.assertEqual(data.embedding_index, ["uid1", "uid2"])
        self.assertIn("large_objects", data.object_categories)
        self.assertIn("small_objects", data.object_categories)

    def test_load_preprocessed_data_missing_files(self):
        """Test validation fails when required files are missing."""
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()

        with self.assertRaises(FileNotFoundError):
            load_preprocessed_data(preprocessed_path)

    def test_global_catalog_reads_sqlite_source_of_truth(self):
        """Global metadata wins over a stale compatibility JSON projection."""
        preprocessed_path = self.tmp_path / "preprocessed"
        preprocessed_path.mkdir()
        np.save(
            preprocessed_path / "clip_embeddings.npy",
            np.ones((1, 768), dtype=np.float32),
        )
        (preprocessed_path / "embedding_index.yaml").write_text("- global__chair\n")
        (preprocessed_path / "object_categories.json").write_text(
            json.dumps({"large_objects": ["global__chair"]})
        )
        (preprocessed_path / "metadata_index.json").write_text(
            json.dumps(
                {
                    "global__chair": {
                        "name": "STALE JSON",
                        "category": "small_objects",
                        "bounding_box": [0.1, 0.1, 0.1],
                    }
                }
            )
        )
        connection = sqlite3.connect(preprocessed_path / "catalog.sqlite3")
        connection.execute(
            """CREATE TABLE assets (
            uid TEXT PRIMARY KEY, source TEXT, source_id TEXT, name TEXT,
            description TEXT, aliases_json TEXT, tags_json TEXT,
            ontology_path TEXT, placement_class TEXT,
            placement_classes_json TEXT, dimensions_json TEXT,
            canonical_up TEXT, canonical_front TEXT, support_zones_json TEXT,
            clearance_zones_json TEXT, license TEXT, quality_score REAL,
            thumbnail TEXT, mesh_path TEXT, embedding_row INTEGER)"""
        )
        connection.execute(
            "INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "global__chair",
                "fixture",
                "chair",
                "SQLite Chair",
                "A normalized chair",
                '["seat"]',
                '["chair"]',
                "furniture/seating/chair",
                "large_objects",
                '["large_objects"]',
                "[0.8, 0.8, 1.0]",
                "0,1,0",
                "0,0,1",
                '[{"kind":"seat"}]',
                '[{"kind":"front"}]',
                "CC0-1.0",
                0.98,
                "chair.png",
                "/catalog/chair.glb",
                0,
            ),
        )
        connection.commit()
        connection.close()

        data = load_preprocessed_data(preprocessed_path)
        chair = data.get_metadata("global__chair")

        self.assertEqual(chair.name, "SQLite Chair")
        self.assertEqual(chair.ontology_path, "furniture/seating/chair")
        self.assertEqual(chair.canonical_front, "0,0,1")
        self.assertEqual(chair.support_zones[0]["kind"], "seat")
        self.assertEqual(chair.clearance_zones[0]["kind"], "front")
        self.assertEqual(chair.quality_score, 0.98)
        self.assertTrue(chair.deferred_loading)


class TestObjaverseMeshMetadata(unittest.TestCase):
    """Test ObjaverseMeshMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating metadata with all fields."""
        metadata = ObjaverseMeshMetadata(
            uid="test123",
            name="Test Chair",
            category="large_objects",
            bounding_box=[0.5, 0.8, 0.5],
            description="A comfortable wooden chair",
        )

        self.assertEqual(metadata.uid, "test123")
        self.assertEqual(metadata.name, "Test Chair")
        self.assertEqual(metadata.category, "large_objects")
        self.assertEqual(metadata.bounding_box, [0.5, 0.8, 0.5])
        self.assertEqual(metadata.description, "A comfortable wooden chair")

    def test_metadata_optional_description(self):
        """Test creating metadata without optional description."""
        metadata = ObjaverseMeshMetadata(
            uid="test456",
            name="Test Table",
            category="large_objects",
            bounding_box=[1.0, 0.75, 0.6],
        )

        self.assertEqual(metadata.uid, "test456")
        self.assertIsNone(metadata.description)


class TestObjaverseRetriever(unittest.TestCase):
    def test_ontology_aliases_rerank_terminal_ahead_of_electronic_safe(self):
        metadata = {
            "safe": ObjaverseMeshMetadata(
                uid="safe",
                name="Electronic Safe with Storage",
                description="Electronic fingerprint safe",
                category="large_objects",
                bounding_box=(0.0, 0.0, 0.0),
                ontology_path="hssd/wordnet/safe.n.01",
                quality_score=0.76,
                deferred_loading=True,
            ),
            "terminal": ObjaverseMeshMetadata(
                uid="terminal",
                name="Computer Terminal",
                description="Large console workstation with computer in center",
                category="large_objects",
                bounding_box=(0.0, 0.0, 0.0),
                ontology_path="electronics/computers/terminals",
                quality_score=0.66,
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
            return_value=[("safe", 1.0), ("terminal", 0.98)],
        ):
            candidates = retriever.retrieve_multiple(
                description="advanced diagnostic monitoring station",
                object_type="furniture",
                desired_dimensions=np.array([1.2, 0.6, 1.4]),
                max_candidates=2,
            )

        self.assertEqual(
            [candidate.uid for candidate in candidates], ["terminal", "safe"]
        )

    def test_ranks_indexed_bounds_before_loading_only_requested_meshes(self):
        metadata = {
            "too_small": ObjaverseMeshMetadata(
                uid="too_small",
                name="Small chair",
                category="large_objects",
                bounding_box=(0.1, 0.1, 0.1),
            ),
            "matching": ObjaverseMeshMetadata(
                uid="matching",
                name="Matching chair",
                category="large_objects",
                bounding_box=(0.8, 0.8, 1.0),
            ),
            "too_large": ObjaverseMeshMetadata(
                uid="too_large",
                name="Large chair",
                category="large_objects",
                bounding_box=(3.0, 3.0, 3.0),
            ),
        }
        retriever = object.__new__(ObjaverseRetriever)
        retriever.config = SimpleNamespace(
            object_type_mapping={"FURNITURE": "large_objects"}, use_top_k=3
        )
        retriever.clip_device = "cpu"
        retriever.preprocessed_data = MagicMock()
        retriever.preprocessed_data.get_metadata.side_effect = metadata.get
        loaded_mesh = trimesh.creation.box(extents=(0.8, 0.8, 1.0))
        retriever._load_mesh = MagicMock(return_value=loaded_mesh)

        with patch(
            "scenesmith.agent_utils.objaverse_retrieval.retrieval.get_top_k_similar_meshes",
            return_value=[
                ("too_small", 0.9),
                ("matching", 0.8),
                ("too_large", 0.7),
            ],
        ):
            candidates = retriever.retrieve_multiple(
                description="chair",
                object_type="furniture",
                desired_dimensions=np.array([0.8, 0.8, 1.0]),
                max_candidates=1,
            )

        self.assertEqual([candidate.uid for candidate in candidates], ["matching"])
        retriever._load_mesh.assert_called_once_with(metadata["matching"])


if __name__ == "__main__":
    unittest.main()
