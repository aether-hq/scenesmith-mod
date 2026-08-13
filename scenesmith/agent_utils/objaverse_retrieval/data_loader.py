"""Data loading utilities for Objaverse (ObjectThor) preprocessed indices and embeddings."""

import gzip
import json
import logging
import pickle

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml

from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

console_logger = logging.getLogger(__name__)


@dataclass
class ObjaverseMeshMetadata:
    """Metadata for a single Objaverse mesh."""

    uid: str
    """Objaverse/ObjectThor unique identifier."""

    name: str
    """Human-readable object name (from description)."""

    category: str
    """scenesmith category (large_objects, small_objects, etc.)."""

    bounding_box: tuple[float, float, float]
    """Bounding box dimensions (x, y, z) in meters from GLB."""

    description: str | None = None
    """Object description text (optional)."""


@dataclass
class ObjaversePreprocessedData:
    """Container for all preprocessed Objaverse data."""

    metadata_by_category: dict[str, list[ObjaverseMeshMetadata]]
    """Maps category names to mesh metadata lists."""

    clip_embeddings: np.ndarray
    """CLIP embeddings array (N, 768)."""

    embedding_index: list[str]
    """Maps array index to mesh UID."""

    object_categories: dict[str, list[str]]
    """Maps object types to UID lists."""

    _metadata_by_uid: dict[str, ObjaverseMeshMetadata] = field(
        init=False, default_factory=dict, repr=False
    )
    """Private O(1) lookup index from UID to metadata."""

    def __post_init__(self):
        """Build metadata lookup index after initialization."""
        self._metadata_by_uid = {
            m.uid: m for meshes in self.metadata_by_category.values() for m in meshes
        }

    def get_metadata(self, uid: str) -> ObjaverseMeshMetadata | None:
        """Get metadata for a specific mesh UID (O(1) lookup).

        Args:
            uid: Objaverse mesh UID to look up.

        Returns:
            Mesh metadata if found, None otherwise.
        """
        return self._metadata_by_uid.get(uid)

    def get_embedding_index(self, uid: str) -> int | None:
        """Get the embedding array index for a mesh UID.

        Args:
            uid: Objaverse mesh UID to look up.

        Returns:
            Array index if found, None otherwise.
        """
        try:
            return self.embedding_index.index(uid)
        except ValueError:
            return None


def load_preprocessed_data(preprocessed_path: Path) -> ObjaversePreprocessedData:
    """Load all preprocessed Objaverse data.

    Args:
        preprocessed_path: Path to directory containing preprocessed files.

    Returns:
        Loaded preprocessed data.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If data format is invalid.
    """
    console_logger.info(f"Loading Objaverse preprocessed data from {preprocessed_path}")

    metadata_path = preprocessed_path / "metadata_index.json"
    embeddings_path = preprocessed_path / "clip_embeddings.npy"
    embedding_index_path = preprocessed_path / "embedding_index.yaml"
    categories_path = preprocessed_path / "object_categories.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not embedding_index_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {embedding_index_path}")
    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_path}")

    with open(metadata_path, "r") as f:
        metadata_data = json.load(f)

    # Build metadata_by_category from flat metadata list.
    metadata_by_category: dict[str, list[ObjaverseMeshMetadata]] = {}
    total_entries = 0

    for uid, entry in metadata_data.items():
        total_entries += 1

        # Extract bounding box from metadata.
        bbox = entry.get("bounding_box", [1.0, 1.0, 1.0])
        if isinstance(bbox, dict):
            bbox = [bbox.get("x", 1.0), bbox.get("y", 1.0), bbox.get("z", 1.0)]

        metadata = ObjaverseMeshMetadata(
            uid=uid,
            name=entry.get("name", entry.get("description", uid)),
            category=entry.get("category", "small_objects"),
            bounding_box=tuple(bbox),
            description=entry.get("description", ""),
        )

        category = metadata.category
        if category not in metadata_by_category:
            metadata_by_category[category] = []
        metadata_by_category[category].append(metadata)

    console_logger.info(f"Loaded {total_entries} Objaverse entries")

    # Load CLIP embeddings.
    clip_embeddings = np.load(embeddings_path)
    console_logger.info(
        f"Loaded CLIP embeddings: shape={clip_embeddings.shape}, "
        f"dtype={clip_embeddings.dtype}"
    )

    # Load embedding index (UID list).
    with open(embedding_index_path, "r") as f:
        embedding_index_data = yaml.safe_load(f)
        # Handle both dict format {"uids": [...]} and plain list format.
        if isinstance(embedding_index_data, dict):
            embedding_index = embedding_index_data.get("uids", [])
        else:
            embedding_index = embedding_index_data

    # Load object categories.
    with open(categories_path, "r") as f:
        categories_data = json.load(f)

    object_categories = {
        category: uid_list for category, uid_list in categories_data.items()
    }

    console_logger.info(
        f"Loaded {len(metadata_by_category)} categories, "
        f"{len(embedding_index)} mesh embeddings, "
        f"{len(object_categories)} object categories"
    )

    return ObjaversePreprocessedData(
        metadata_by_category=metadata_by_category,
        clip_embeddings=clip_embeddings,
        embedding_index=embedding_index,
        object_categories=object_categories,
    )


def _vector_array(values: list[dict], *, keys: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[value[key] for key in keys] for value in values], dtype=np.float64
    )


def _load_texture(path: Path) -> Image.Image | None:
    return Image.open(path) if path.is_file() else None


def _convert_procedural_asset(asset_dir: Path, uid: str, output_path: Path) -> None:
    source_path = asset_dir / f"{uid}.pkl.gz"
    if not source_path.is_file():
        raise FileNotFoundError(f"ObjectThor procedural mesh not found: {source_path}")
    with gzip.open(source_path, "rb") as source:
        payload = pickle.load(source)
    vertices = _vector_array(payload["vertices"], keys=("x", "y", "z"))
    faces = np.asarray(payload["triangles"], dtype=np.int64).reshape((-1, 3))
    normals = _vector_array(payload["normals"], keys=("x", "y", "z"))
    uv = _vector_array(payload["uvs"], keys=("x", "y"))
    material = PBRMaterial(
        name=f"objathor-{uid}",
        baseColorTexture=_load_texture(asset_dir / "albedo.jpg"),
        normalTexture=_load_texture(asset_dir / "normal.jpg"),
        emissiveTexture=_load_texture(asset_dir / "emission.jpg"),
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        visual=TextureVisuals(uv=uv, material=material),
        process=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.glb")
    temporary.write_bytes(trimesh.exchange.gltf.export_glb(mesh))
    temporary.replace(output_path)


def construct_objaverse_mesh_path(
    data_path: Path, uid: str, derived_cache_path: Path | None = None
) -> Path:
    """Return a cached GLB, converting the official ObjectThor asset if needed.

    ObjectThor's official archive stores procedural mesh arrays in ``.pkl.gz``
    alongside texture maps. SceneSmith consumes GLB, so selected retrieval
    candidates are converted lazily and cached without rewriting the master.

    Args:
        data_path: Root directory of Objaverse data (containing assets/ subdir).
        uid: Objaverse mesh UID.
        derived_cache_path: Optional writable root for derived GLBs. When set,
            the immutable ObjectThor master is never modified.

    Returns:
        Path to the GLB mesh file.

    Raises:
        FileNotFoundError: If the constructed path does not exist.
    """
    asset_dir = data_path / "assets" / uid
    mesh_path = (
        derived_cache_path / uid / f"{uid}.glb"
        if derived_cache_path is not None
        else asset_dir / f"{uid}.glb"
    )
    if not mesh_path.is_file():
        _convert_procedural_asset(asset_dir, uid, mesh_path)
    return mesh_path
