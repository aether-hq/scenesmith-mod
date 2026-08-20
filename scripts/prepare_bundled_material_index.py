#!/usr/bin/env python3
"""Build a valid minimal retrieval index for SceneSmith's bundled PBR materials.

The repository contains Plaster001 and Wood094 textures. This script slices their
rows from the official precomputed AmbientCG index, preserving that complete
index in ``embeddings_ambientcg`` for a future full material-library download.
"""

from __future__ import annotations

import shutil

from pathlib import Path

import numpy as np
import yaml

MATERIAL_IDS = ("Plaster001", "Wood094")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" / "materials"
    destination = data_root / "embeddings"
    complete_index = data_root / "embeddings_ambientcg"

    if not complete_index.exists():
        with (destination / "embedding_index.yaml").open() as stream:
            current_ids = yaml.safe_load(stream)
        if len(current_ids) <= len(MATERIAL_IDS):
            print(f"Bundled material index is already ready at {destination}")
            return
        destination.rename(complete_index)

    with (complete_index / "embedding_index.yaml").open() as stream:
        all_ids = yaml.safe_load(stream)
    with (complete_index / "metadata_index.yaml").open() as stream:
        all_metadata = yaml.safe_load(stream)
    all_embeddings = np.load(complete_index / "clip_embeddings.npy")

    missing = [
        material_id for material_id in MATERIAL_IDS if material_id not in all_ids
    ]
    if missing:
        raise RuntimeError(f"Official AmbientCG index is missing: {missing}")

    indices = [all_ids.index(material_id) for material_id in MATERIAL_IDS]
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "clip_embeddings.npy", all_embeddings[indices])
    with (destination / "embedding_index.yaml").open("w") as stream:
        yaml.safe_dump(list(MATERIAL_IDS), stream, sort_keys=False)
    with (destination / "metadata_index.yaml").open("w") as stream:
        yaml.safe_dump(
            {material_id: all_metadata[material_id] for material_id in MATERIAL_IDS},
            stream,
            sort_keys=False,
        )

    for material_id in MATERIAL_IDS:
        source = project_root / "materials" / f"{material_id}_1K-JPG"
        target = data_root / material_id
        if not source.is_dir():
            raise FileNotFoundError(f"Bundled material textures not found: {source}")
        shutil.copytree(source, target, dirs_exist_ok=True)

    print(
        f"Prepared {len(MATERIAL_IDS)} bundled materials and a matching index at "
        f"{destination}"
    )


if __name__ == "__main__":
    main()
