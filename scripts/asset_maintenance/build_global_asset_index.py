#!/usr/bin/env python3
"""Build one normalized semantic catalog from every local object library.

The resulting directory is SceneSmith-compatible (metadata + embeddings) and
also contains ``catalog.sqlite3``, the source of truth for hierarchy, placement,
canonical-frame, licensing, quality, and provenance metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scenesmith.agent_utils.objaverse_retrieval.clip_similarity import (
    get_objaverse_text_embeddings,
)

LOGGER = logging.getLogger("global_asset_index")
PLACEMENT_CLASSES = (
    "large_objects",
    "small_objects",
    "wall_objects",
    "ceiling_objects",
)


def _load_embedding_index(path: Path) -> list[str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(value.get("uids", [])) if isinstance(value, dict) else list(value)


def _category_memberships(path: Path) -> dict[str, set[str]]:
    categories = json.loads(path.read_text(encoding="utf-8"))
    memberships: dict[str, set[str]] = defaultdict(set)
    for category, identifiers in categories.items():
        if category not in PLACEMENT_CLASSES:
            continue
        for identifier in identifiers:
            memberships[str(identifier)].add(category)
    return memberships


def _valid_dimensions(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    dimensions = [float(value) for value in values]
    if min(dimensions) <= 0:
        return None
    return dimensions


def _quality_score(
    source: str,
    *,
    dimensions: list[float] | None,
    description: str,
    canonical_up: str | None = None,
    canonical_front: str | None = None,
) -> float:
    base = {"polyhaven": 0.95, "hssd": 0.72, "objaverse": 0.62}[source]
    if dimensions:
        base += 0.08
    if description and description.lower() not in {"unknown", "none"}:
        base += 0.04
    if canonical_up and canonical_front:
        base += 0.06
    return round(min(base, 1.0), 4)


def _normalized_row(
    *,
    uid: str,
    source: str,
    source_id: str,
    name: str,
    description: str,
    aliases: list[str],
    tags: list[str],
    ontology_path: str,
    placements: set[str],
    dimensions: list[float] | None,
    canonical_up: str | None,
    canonical_front: str | None,
    license_id: str | None,
    thumbnail: str | None,
    mesh_path: Path,
    embedding_row: int,
) -> dict[str, Any]:
    placement_values = sorted(placements) or ["small_objects"]
    return {
        "uid": uid,
        "source": source,
        "source_id": source_id,
        "name": name.strip() or source_id,
        "description": description.strip(),
        "aliases": sorted({value.strip() for value in aliases if value.strip()}),
        "tags": sorted({value.strip().lower() for value in tags if value.strip()}),
        "ontology_path": ontology_path,
        "placement_class": placement_values[0],
        "placement_classes": placement_values,
        "bounding_box": dimensions,
        "canonical_up": canonical_up or None,
        "canonical_front": canonical_front or None,
        "support_zones": [],
        "clearance_zones": [],
        "license": license_id,
        "quality_score": _quality_score(
            source,
            dimensions=dimensions,
            description=description,
            canonical_up=canonical_up,
            canonical_front=canonical_front,
        ),
        "thumbnail": thumbnail,
        "mesh_path": str(mesh_path.resolve()),
        "asset_source": source,
        "deferred_loading": True,
        "embedding_row": embedding_row,
    }


def _catalog_rows(
    *,
    source: str,
    data_root: Path,
    preprocessed_root: Path,
    embedding_offset: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    metadata = json.loads(
        (preprocessed_root / "metadata_index.json").read_text(encoding="utf-8")
    )
    embedding_ids = _load_embedding_index(preprocessed_root / "embedding_index.yaml")
    embeddings = np.load(preprocessed_root / "clip_embeddings.npy")
    if len(embedding_ids) != len(embeddings):
        raise ValueError(f"{source} embedding IDs do not match the embedding matrix")
    source_rows = {identifier: row for identifier, row in metadata.items()}
    memberships = _category_memberships(preprocessed_root / "object_categories.json")
    rows: list[dict[str, Any]] = []
    selected_embeddings: list[np.ndarray] = []
    for identifier, embedding in zip(embedding_ids, embeddings):
        raw = source_rows.get(identifier)
        if raw is None:
            continue
        source_id = str(raw.get("source_id") or identifier)
        if source == "polyhaven":
            original_mesh_path = Path(str(raw["mesh_path"]))
            mesh_path = (
                original_mesh_path
                if original_mesh_path.is_absolute()
                else data_root / original_mesh_path
            )
            ontology = str(raw.get("source_category") or "uncategorized")
            tags = [str(value) for value in raw.get("tags", [])]
            aliases = [str(raw.get("name", "")), source_id, *tags]
            thumbnail = raw.get("thumbnail_url")
        else:
            mesh_path = data_root / "assets" / source_id / f"{source_id}.glb"
            ontology = str(raw.get("category") or "uncategorized")
            tags = []
            aliases = [str(raw.get("name", "")), source_id]
            thumbnail_path = data_root / "assets" / source_id / "albedo.jpg"
            thumbnail = str(thumbnail_path) if thumbnail_path.exists() else None
        uid = f"{source}__{source_id}"
        rows.append(
            _normalized_row(
                uid=uid,
                source=source,
                source_id=source_id,
                name=str(raw.get("name") or source_id),
                description=str(raw.get("description") or ""),
                aliases=aliases,
                tags=tags,
                ontology_path=f"{source}/{ontology}",
                placements=memberships.get(identifier, set()),
                dimensions=_valid_dimensions(raw.get("bounding_box")),
                canonical_up=None,
                canonical_front=None,
                license_id=raw.get("license"),
                thumbnail=thumbnail,
                mesh_path=mesh_path,
                embedding_row=embedding_offset + len(rows),
            )
        )
        selected_embeddings.append(np.asarray(embedding, dtype=np.float32))
    return rows, np.stack(selected_embeddings)


def _hssd_rows(
    *,
    data_root: Path,
    preprocessed_root: Path,
    embedding_offset: int,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    index = json.loads(
        (preprocessed_root / "hssd_wnsynsetkey_index.json").read_text(encoding="utf-8")
    )
    category_synsets = json.loads(
        (preprocessed_root / "object_categories.json").read_text(encoding="utf-8")
    )
    synset_placements: dict[str, set[str]] = defaultdict(set)
    for category, synsets in category_synsets.items():
        if category in PLACEMENT_CLASSES:
            for synset in synsets:
                synset_placements[str(synset)].add(category)

    rows: list[dict[str, Any]] = []
    documents: list[str] = []
    row_by_source_id: dict[str, dict[str, Any]] = {}
    for synset, entries in sorted(index.items()):
        friendly_synset = synset.replace("_", " ").replace(".n.", " ")
        for raw in entries:
            source_id = str(raw["id"])
            name = str(raw.get("name") or friendly_synset)
            up = str(raw.get("up") or "") or None
            front = str(raw.get("front") or "") or None
            uid = f"hssd__{source_id}"
            existing = row_by_source_id.get(source_id)
            if existing is not None:
                existing["aliases"] = sorted(
                    {*existing["aliases"], name, friendly_synset}
                )
                existing["tags"] = sorted({*existing["tags"], friendly_synset})
                existing["placement_classes"] = sorted(
                    {
                        *existing["placement_classes"],
                        *synset_placements.get(synset, set()),
                    }
                )
                continue
            documents.append(f"{name}. {friendly_synset}. HSSD object model")
            row = _normalized_row(
                uid=uid,
                source="hssd",
                source_id=source_id,
                name=name,
                description=f"{name}; category {friendly_synset}",
                aliases=[name, friendly_synset, source_id],
                tags=[friendly_synset],
                ontology_path=f"hssd/wordnet/{synset}",
                placements=synset_placements.get(synset, set()),
                dimensions=None,
                canonical_up=up,
                canonical_front=front,
                license_id="CC-BY-NC-4.0",
                thumbnail=None,
                mesh_path=(data_root / "objects" / source_id[0] / f"{source_id}.glb"),
                embedding_row=embedding_offset + len(rows),
            )
            rows.append(row)
            row_by_source_id[source_id] = row
    for row in rows:
        row["placement_class"] = row["placement_classes"][0]
    LOGGER.info("Embedding %d HSSD descriptions in the shared ViT-L space", len(rows))
    embeddings = get_objaverse_text_embeddings(
        documents, device=device, batch_size=batch_size
    )
    return rows, embeddings


def _write_sqlite(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE assets (
              uid TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              source_id TEXT NOT NULL,
              name TEXT NOT NULL,
              description TEXT NOT NULL,
              aliases_json TEXT NOT NULL,
              tags_json TEXT NOT NULL,
              ontology_path TEXT NOT NULL,
              placement_class TEXT NOT NULL,
              placement_classes_json TEXT NOT NULL,
              dimensions_json TEXT,
              canonical_up TEXT,
              canonical_front TEXT,
              support_zones_json TEXT NOT NULL,
              clearance_zones_json TEXT NOT NULL,
              license TEXT,
              quality_score REAL NOT NULL,
              thumbnail TEXT,
              mesh_path TEXT NOT NULL,
              embedding_row INTEGER NOT NULL UNIQUE
            );
            CREATE INDEX assets_source_idx ON assets(source);
            CREATE INDEX assets_ontology_idx ON assets(ontology_path);
            CREATE INDEX assets_placement_idx ON assets(placement_class);
            CREATE INDEX assets_quality_idx ON assets(quality_score DESC);
            CREATE VIRTUAL TABLE assets_fts USING fts5(
              uid UNINDEXED, name, description, aliases, tags, ontology_path
            );
            """
        )
        for row in rows:
            connection.execute(
                """INSERT INTO assets VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )""",
                (
                    row["uid"],
                    row["source"],
                    row["source_id"],
                    row["name"],
                    row["description"],
                    json.dumps(row["aliases"]),
                    json.dumps(row["tags"]),
                    row["ontology_path"],
                    row["placement_class"],
                    json.dumps(row["placement_classes"]),
                    json.dumps(row["bounding_box"]),
                    row["canonical_up"],
                    row["canonical_front"],
                    json.dumps(row["support_zones"]),
                    json.dumps(row["clearance_zones"]),
                    row["license"],
                    row["quality_score"],
                    row["thumbnail"],
                    row["mesh_path"],
                    row["embedding_row"],
                ),
            )
            connection.execute(
                "INSERT INTO assets_fts VALUES (?,?,?,?,?,?)",
                (
                    row["uid"],
                    row["name"],
                    row["description"],
                    " ".join(row["aliases"]),
                    " ".join(row["tags"]),
                    row["ontology_path"],
                ),
            )
        connection.commit()
    finally:
        connection.close()


def build_global_index(
    *,
    engine_root: Path,
    output_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []

    for source, data_path, preprocessed_path in (
        (
            "polyhaven",
            engine_root / "data/polyhaven",
            engine_root / "data/polyhaven/preprocessed",
        ),
        (
            "objaverse",
            engine_root / "data/objathor-assets",
            engine_root / "data/objathor-assets/preprocessed",
        ),
    ):
        source_rows, embeddings = _catalog_rows(
            source=source,
            data_root=data_path,
            preprocessed_root=preprocessed_path,
            embedding_offset=len(rows),
        )
        rows.extend(source_rows)
        matrices.append(embeddings)
        LOGGER.info("Added %d %s assets", len(source_rows), source)

    hssd_rows, hssd_embeddings = _hssd_rows(
        data_root=engine_root / "data/hssd-models",
        preprocessed_root=engine_root / "data/preprocessed",
        embedding_offset=len(rows),
        device=device,
        batch_size=batch_size,
    )
    rows.extend(hssd_rows)
    matrices.append(hssd_embeddings)
    embeddings = np.concatenate(matrices, axis=0).astype(np.float32)
    if len(rows) != len(embeddings):
        raise AssertionError("Global metadata and embedding row counts diverged")

    metadata = {
        row["uid"]: {
            "name": row["name"],
            "description": row["description"],
            "category": row["placement_class"],
            "bounding_box": row["bounding_box"] or [0.0, 0.0, 0.0],
            "mesh_path": row["mesh_path"],
            "asset_source": row["source"],
            "license": row["license"],
            "source_id": row["source_id"],
            "aliases": row["aliases"],
            "tags": row["tags"],
            "ontology_path": row["ontology_path"],
            "placement_class": row["placement_class"],
            "placement_classes": row["placement_classes"],
            "canonical_up": row["canonical_up"],
            "canonical_front": row["canonical_front"],
            "support_zones": row["support_zones"],
            "clearance_zones": row["clearance_zones"],
            "quality_score": row["quality_score"],
            "thumbnail": row["thumbnail"],
            "deferred_loading": True,
        }
        for row in rows
    }
    categories = {
        category: [row["uid"] for row in rows if category in row["placement_classes"]]
        for category in PLACEMENT_CLASSES
    }
    np.save(output_root / "clip_embeddings.npy", embeddings)
    (output_root / "metadata_index.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "object_categories.json").write_text(
        json.dumps(categories, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "embedding_index.yaml").write_text(
        yaml.safe_dump([row["uid"] for row in rows], sort_keys=False),
        encoding="utf-8",
    )
    _write_sqlite(output_root / "catalog.sqlite3", rows)
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": "ViT-L-14/laion2b_s32b_b82k",
        "embedding_dimensions": 768,
        "asset_count": len(rows),
        "source_counts": {
            source: sum(row["source"] == source for row in rows)
            for source in ("polyhaven", "hssd", "objaverse")
        },
        "category_counts": {key: len(value) for key, value in categories.items()},
        "candidate_pool_size": 50,
        "returned_candidates": 12,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/global-assets/preprocessed"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine_root = args.engine_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = engine_root / output
    manifest = build_global_index(
        engine_root=engine_root,
        output_root=output,
        device=args.device,
        batch_size=args.batch_size,
    )
    LOGGER.info("Global catalog ready: %s", json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
