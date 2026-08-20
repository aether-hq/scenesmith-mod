"""Tests for the repeatable Poly Haven semantic catalog setup."""

import json

from pathlib import Path

from scripts.index_polyhaven import classify_polyhaven_asset, collect_polyhaven_models


def test_polyhaven_taxonomy_supports_placement_hierarchy() -> None:
    wall_alarm = {
        "name": "Fire Alarm",
        "categories": ["wall decoration"],
        "tags": ["alarm", "safety"],
        "dimensions": [100, 30, 150],
    }
    chair = {
        "name": "Dining Chair",
        "categories": ["furniture", "seating"],
        "tags": ["chair"],
        "dimensions": [500, 500, 900],
    }

    assert "wall_objects" in classify_polyhaven_asset(wall_alarm)
    assert "large_objects" in classify_polyhaven_asset(chair)


def test_collect_polyhaven_models_preserves_path_and_license(tmp_path: Path) -> None:
    asset = {
        "name": "Dining Chair",
        "description": "An authored wooden dining chair.",
        "categories": ["furniture", "seating"],
        "category": "Furniture/Seating/Chairs",
        "tags": ["chair", "wood"],
        "dimensions": [500, 600, 900],
        "authors": {"Example Artist": "All"},
    }
    catalog_dir = tmp_path / "catalog"
    model_dir = tmp_path / "models" / "dining_chair"
    catalog_dir.mkdir()
    model_dir.mkdir(parents=True)
    (catalog_dir / "assets.json").write_text(
        json.dumps({"dining_chair": asset}), encoding="utf-8"
    )
    (model_dir / "dining_chair_2k.gltf").write_text("{}", encoding="utf-8")

    metadata, categories, documents = collect_polyhaven_models(tmp_path, "2k")

    item = metadata["polyhaven__dining_chair"]
    assert item["mesh_path"] == "models/dining_chair/dining_chair_2k.gltf"
    assert item["license"] == "CC0-1.0"
    assert item["asset_source"] == "polyhaven"
    assert item["authors"] == ["Example Artist"]
    assert "polyhaven__dining_chair" in categories["large_objects"]
    assert len(documents) == 1
