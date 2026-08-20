"""Data loading utilities for Objaverse (ObjectThor) preprocessed indices and embeddings."""

import gzip
import json
import logging
import pickle
import sqlite3
import tempfile

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

    mesh_path: str | None = None
    """Catalog-relative mesh path. None uses the ObjectThor assets/{uid} layout."""

    asset_source: str = "objaverse"
    """Stable provenance label for the catalog that owns this mesh."""

    license: str | None = None
    """Asset license identifier, when supplied by the source catalog."""

    source_id: str | None = None
    """Identifier in the originating catalog (without a global source prefix)."""

    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    ontology_path: str | None = None
    placement_classes: tuple[str, ...] = ()
    canonical_up: str | None = None
    canonical_front: str | None = None
    support_zones: tuple[dict, ...] = ()
    clearance_zones: tuple[dict, ...] = ()
    quality_score: float = 0.5
    thumbnail: str | None = None
    deferred_loading: bool = False


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

    _embedding_row_by_uid: dict[str, int] = field(
        init=False, default_factory=dict, repr=False
    )
    """Private O(1) lookup from UID to embedding row."""

    def __post_init__(self):
        """Build metadata lookup index after initialization."""
        self._metadata_by_uid = {
            m.uid: m for meshes in self.metadata_by_category.values() for m in meshes
        }
        self._embedding_row_by_uid = {
            uid: index for index, uid in enumerate(self.embedding_index)
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
        return self._embedding_row_by_uid.get(uid)


def _load_metadata(preprocessed_path: Path) -> dict[str, dict]:
    """Load normalized catalog metadata, preferring its SQLite source of truth."""
    catalog_path = preprocessed_path / "catalog.sqlite3"
    if not catalog_path.exists():
        metadata_path = preprocessed_path / "metadata_index.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Catalog metadata not found: expected {catalog_path} or {metadata_path}"
            )
        with open(metadata_path, "r") as metadata_file:
            return json.load(metadata_file)

    metadata: dict[str, dict] = {}
    connection = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT uid, source, source_id, name, description, aliases_json,
                      tags_json, ontology_path, placement_class,
                      placement_classes_json, dimensions_json, canonical_up,
                      canonical_front, support_zones_json, clearance_zones_json,
                      license, quality_score, thumbnail, mesh_path
                 FROM assets
             ORDER BY embedding_row"""
        )
        for row in rows:
            dimensions = json.loads(row["dimensions_json"] or "null")
            metadata[row["uid"]] = {
                "name": row["name"],
                "description": row["description"],
                "category": row["placement_class"],
                "bounding_box": dimensions or [0.0, 0.0, 0.0],
                "mesh_path": row["mesh_path"],
                "asset_source": row["source"],
                "license": row["license"],
                "source_id": row["source_id"],
                "aliases": json.loads(row["aliases_json"]),
                "tags": json.loads(row["tags_json"]),
                "ontology_path": row["ontology_path"],
                "placement_class": row["placement_class"],
                "placement_classes": json.loads(row["placement_classes_json"]),
                "canonical_up": row["canonical_up"],
                "canonical_front": row["canonical_front"],
                "support_zones": json.loads(row["support_zones_json"]),
                "clearance_zones": json.loads(row["clearance_zones_json"]),
                "quality_score": row["quality_score"],
                "thumbnail": row["thumbnail"],
                "deferred_loading": True,
            }
    finally:
        connection.close()
    return metadata


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

    embeddings_path = preprocessed_path / "clip_embeddings.npy"
    embedding_index_path = preprocessed_path / "embedding_index.yaml"
    categories_path = preprocessed_path / "object_categories.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not embedding_index_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {embedding_index_path}")
    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_path}")

    metadata_data = _load_metadata(preprocessed_path)

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
            mesh_path=entry.get("mesh_path"),
            asset_source=entry.get("asset_source", "objaverse"),
            license=entry.get("license"),
            source_id=entry.get("source_id", uid),
            aliases=tuple(entry.get("aliases", [])),
            tags=tuple(entry.get("tags", [])),
            ontology_path=entry.get("ontology_path"),
            placement_classes=tuple(entry.get("placement_classes", [])),
            canonical_up=entry.get("canonical_up"),
            canonical_front=entry.get("canonical_front"),
            support_zones=tuple(entry.get("support_zones", [])),
            clearance_zones=tuple(entry.get("clearance_zones", [])),
            quality_score=float(entry.get("quality_score", 0.5)),
            thumbnail=entry.get("thumbnail"),
            deferred_loading=bool(entry.get("deferred_loading", False)),
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


def _vector_array(values: list[dict], dimensions: tuple[str, ...]) -> np.ndarray:
    """Convert ObjectThor's list-of-dicts vectors into a dense float array."""
    return np.asarray(
        [[value[dimension] for dimension in dimensions] for value in values],
        dtype=np.float64,
    )


def _load_texture(path: Path) -> Image.Image | None:
    """Load a texture eagerly so the source file can close before GLB export."""
    if not path.exists():
        return None
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


def convert_objathor_asset_to_glb(asset_dir: Path, uid: str) -> Path:
    """Convert an ObjectThor runtime bundle to a cached textured GLB.

    The public ObjectThor archive stores geometry in ``{uid}.pkl.gz`` plus
    adjacent albedo, normal, and emission textures. SceneSmith consumes GLB, so
    conversion happens lazily on the first retrieval and the result remains in
    the asset directory for all later scenes.
    """
    source_path = asset_dir / f"{uid}.pkl.gz"
    output_path = asset_dir / f"{uid}.glb"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Objaverse mesh not found: expected {output_path} or {source_path}"
        )

    with gzip.open(source_path, "rb") as source:
        asset = pickle.load(source)

    vertices = _vector_array(asset["vertices"], ("x", "y", "z"))
    faces = np.asarray(asset["triangles"], dtype=np.int64).reshape((-1, 3))
    normals = _vector_array(asset.get("normals", []), ("x", "y", "z"))
    uvs = _vector_array(asset.get("uvs", []), ("x", "y"))

    material = PBRMaterial(
        name=f"objathor-{uid}",
        baseColorTexture=_load_texture(asset_dir / "albedo.jpg"),
        normalTexture=_load_texture(asset_dir / "normal.jpg"),
        emissiveTexture=_load_texture(asset_dir / "emission.jpg"),
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    visual = TextureVisuals(
        uv=uvs if len(uvs) == len(vertices) else None,
        material=material,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals if len(normals) == len(vertices) else None,
        visual=visual,
        process=False,
    )

    glb = trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh))
    with tempfile.NamedTemporaryFile(
        dir=asset_dir,
        prefix=f".{uid}.",
        suffix=".glb.tmp",
        delete=False,
    ) as temporary:
        temporary.write(glb)
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)
    console_logger.info(f"Converted ObjectThor asset {uid} to {output_path}")
    return output_path


def construct_objaverse_mesh_path(data_path: Path, uid: str) -> Path:
    """Construct the file path for an Objaverse mesh.

    ObjectThor stores assets in ``{data_path}/assets/{uid}/{uid}.pkl.gz`` with
    adjacent textures. SceneSmith caches a GLB conversion in the same directory.

    Args:
        data_path: Root directory of Objaverse data (containing assets/ subdir).
        uid: Objaverse mesh UID.

    Returns:
        Path to the GLB mesh file.

    Raises:
        FileNotFoundError: If the constructed path does not exist.
    """
    # ObjectThor stores assets in assets/ subdirectory.
    asset_dir = data_path / "assets" / uid
    mesh_path = asset_dir / f"{uid}.glb"

    if not mesh_path.exists():
        mesh_path = convert_objathor_asset_to_glb(asset_dir=asset_dir, uid=uid)

    return mesh_path


def resolve_catalog_mesh_path(
    data_path: Path, metadata: ObjaverseMeshMetadata
) -> Path:
    """Resolve either a generic catalog mesh or the legacy ObjectThor bundle."""
    if metadata.mesh_path:
        mesh_path = Path(metadata.mesh_path)
        if not mesh_path.is_absolute():
            mesh_path = data_path / mesh_path
        if mesh_path.is_file():
            return mesh_path
        if metadata.asset_source == "objaverse" and mesh_path.suffix == ".glb":
            return convert_objathor_asset_to_glb(
                asset_dir=mesh_path.parent,
                uid=metadata.source_id or metadata.uid,
            )
        raise FileNotFoundError(
            f"{metadata.asset_source} mesh not found: {mesh_path}"
        )
    return construct_objaverse_mesh_path(
        data_path=data_path, uid=metadata.source_id or metadata.uid
    )
