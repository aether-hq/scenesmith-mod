#!/usr/bin/env python3
"""Build a SceneSmith-compatible semantic object index for Poly Haven models."""

from __future__ import annotations

import argparse
import json
import logging

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scenesmith.agent_utils.objaverse_retrieval.clip_similarity import (
    get_objaverse_text_embeddings,
)

LOGGER = logging.getLogger("polyhaven_index")
OBJECT_CATEGORIES = (
    "large_objects",
    "small_objects",
    "wall_objects",
    "ceiling_objects",
)

WALL_TERMS = {
    "wall",
    "wall-mounted",
    "painting",
    "picture",
    "frame",
    "poster",
    "mirror",
    "sign",
    "switch",
    "socket",
    "outlet",
    "fire alarm",
}
CEILING_TERMS = {
    "ceiling",
    "chandelier",
    "pendant light",
    "hanging light",
    "ceiling fan",
}
FURNITURE_TERMS = {
    "furniture",
    "seating",
    "chair",
    "stool",
    "sofa",
    "couch",
    "bench",
    "table",
    "desk",
    "bed",
    "shelf",
    "cabinet",
    "cupboard",
    "wardrobe",
    "appliance",
}


def _normalized_terms(asset: dict) -> str:
    values = [
        asset.get("name", ""),
        asset.get("description", ""),
        asset.get("category", ""),
        *asset.get("categories", []),
        *asset.get("tags", []),
    ]
    return " ".join(str(value).lower().replace("_", " ") for value in values)


def classify_polyhaven_asset(asset: dict) -> list[str]:
    """Map Poly Haven taxonomy into SceneSmith placement categories.

    Membership is intentionally multi-label: a wall lamp remains discoverable as
    a small object, and a large decorative prop remains available to furniture.
    """
    terms = _normalized_terms(asset)
    dimensions = [float(value) / 1000.0 for value in asset.get("dimensions", [])]
    max_dimension = max(dimensions, default=0.0)
    categories: set[str] = set()

    if any(term in terms for term in WALL_TERMS):
        categories.add("wall_objects")
    if any(term in terms for term in CEILING_TERMS):
        categories.add("ceiling_objects")
    if any(term in terms for term in FURNITURE_TERMS) or max_dimension >= 0.75:
        categories.add("large_objects")
    if max_dimension <= 1.25 or not categories:
        categories.add("small_objects")

    return [category for category in OBJECT_CATEGORIES if category in categories]


def _mesh_path(polyhaven_root: Path, asset_id: str, resolution: str) -> Path | None:
    model_dir = polyhaven_root / "models" / asset_id
    preferred = model_dir / f"{asset_id}_{resolution}.gltf"
    if preferred.is_file():
        return preferred
    candidates = sorted(model_dir.glob("*.gltf"))
    return candidates[0] if candidates else None


def _search_document(asset_id: str, asset: dict) -> str:
    attributes = asset.get("attributes", {})
    attribute_values = [
        item
        for values in attributes.values()
        for item in (values if isinstance(values, list) else [values])
    ]
    parts = [
        asset.get("name", asset_id),
        asset.get("description", ""),
        asset.get("category", ""),
        " ".join(asset.get("categories", [])),
        " ".join(asset.get("tags", [])),
        " ".join(str(value) for value in attribute_values),
    ]
    return ". ".join(str(part).strip() for part in parts if str(part).strip())


def collect_polyhaven_models(
    polyhaven_root: Path, resolution: str
) -> tuple[dict[str, dict], dict[str, list[str]], list[str]]:
    catalog_path = polyhaven_root / "catalog" / "assets.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    metadata: dict[str, dict] = {}
    object_categories = {category: [] for category in OBJECT_CATEGORIES}
    documents: list[str] = []

    for asset_id, asset in sorted(catalog.items()):
        mesh_path = _mesh_path(polyhaven_root, asset_id, resolution)
        if mesh_path is None:
            continue
        uid = f"polyhaven__{asset_id}"
        dimensions = [
            round(float(value) / 1000.0, 6)
            for value in asset.get("dimensions", [1.0, 1.0, 1.0])
        ]
        if len(dimensions) != 3:
            dimensions = [1.0, 1.0, 1.0]
        memberships = classify_polyhaven_asset(asset)
        for category in memberships:
            object_categories[category].append(uid)
        metadata[uid] = {
            "name": asset.get("name", asset_id),
            "description": asset.get("description", ""),
            "category": memberships[0],
            "bounding_box": dimensions,
            "mesh_path": str(mesh_path.relative_to(polyhaven_root)),
            "asset_source": "polyhaven",
            "license": "CC0-1.0",
            "source_id": asset_id,
            "source_category": asset.get("category"),
            "tags": asset.get("tags", []),
            "authors": sorted(asset.get("authors", {}).keys()),
            "thumbnail_url": asset.get("thumbnail_url"),
        }
        documents.append(_search_document(asset_id, asset))

    return metadata, object_categories, documents


def build_polyhaven_index(
    polyhaven_root: Path,
    output_dir: Path,
    resolution: str = "2k",
    device: str = "auto",
    batch_size: int = 64,
) -> dict:
    metadata, object_categories, documents = collect_polyhaven_models(
        polyhaven_root=polyhaven_root,
        resolution=resolution,
    )
    if not metadata:
        raise RuntimeError(f"No downloaded Poly Haven GLTF models found in {polyhaven_root}")

    uids = list(metadata)
    LOGGER.info("Embedding %d Poly Haven models on %s", len(uids), device)
    embeddings = get_objaverse_text_embeddings(
        documents, device=device, batch_size=batch_size
    )
    if embeddings.shape != (len(uids), 768):
        raise RuntimeError(
            f"Unexpected embedding shape {embeddings.shape}; expected ({len(uids)}, 768)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "clip_embeddings.npy", embeddings)
    (output_dir / "metadata_index.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "object_categories.json").write_text(
        json.dumps(object_categories, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "embedding_index.yaml").write_text(
        yaml.safe_dump({"uids": uids}, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "source": "polyhaven",
        "license": "CC0-1.0",
        "resolution": resolution,
        "embedding_model": "ViT-L-14/laion2b_s32b_b82k",
        "asset_count": len(uids),
        "category_counts": {
            category: len(values) for category, values in object_categories.items()
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/polyhaven"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", default="2k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_path = args.data_path.expanduser().resolve()
    output = (args.output or data_path / "preprocessed").expanduser().resolve()
    manifest = build_polyhaven_index(
        polyhaven_root=data_path,
        output_dir=output,
        resolution=args.resolution,
        device=args.device,
        batch_size=args.batch_size,
    )
    LOGGER.info(
        "Poly Haven index ready: %d assets at %s", manifest["asset_count"], output
    )


if __name__ == "__main__":
    main()
