"""Regression coverage for floor-plan material family compatibility."""

import tempfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scenesmith.floor_plan_agents.tools.materials_resolver import (
    MaterialsConfig,
    MaterialsResolver,
)


def _material_directory(root: Path, material_id: str) -> Path:
    directory = root / material_id
    directory.mkdir(parents=True)
    for suffix in ("Color", "NormalGL", "Roughness"):
        (directory / f"{material_id}_{suffix}.jpg").touch()
    return directory


def test_plaster_wall_rejects_ground_family_retrieval() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        materials_dir = root / "materials"
        plaster = _material_directory(materials_dir, "Plaster001_1K-JPG")
        ground = _material_directory(root / "retrieved", "Ground096A")
        response = SimpleNamespace(
            results=[SimpleNamespace(material_id="Ground096A", material_path=ground)]
        )
        resolver = MaterialsResolver(
            MaterialsConfig(
                use_retrieval_server=True,
                default_wall_material=plaster.name,
                materials_dir=materials_dir,
                output_dir=root / "output",
            )
        )

        with patch(
            "scenesmith.floor_plan_agents.tools.materials_resolver."
            "MaterialsRetrievalClient"
        ) as client_type:
            client_type.return_value.retrieve_materials.return_value = [(0, response)]
            material = resolver.get_material(
                "warm ivory plaster wall with carved stone trim"
            )

        assert material is not None
        assert material.material_id == "Plaster001_1K-JPG"


def test_ground_surface_can_use_ground_family_retrieval() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        materials_dir = root / "materials"
        _material_directory(materials_dir, "Plaster001_1K-JPG")
        ground = _material_directory(root / "retrieved", "Ground096A")
        response = SimpleNamespace(
            results=[SimpleNamespace(material_id="Ground096A", material_path=ground)]
        )
        resolver = MaterialsResolver(
            MaterialsConfig(
                use_retrieval_server=True,
                materials_dir=materials_dir,
                output_dir=root / "output",
            )
        )

        with patch(
            "scenesmith.floor_plan_agents.tools.materials_resolver."
            "MaterialsRetrievalClient"
        ) as client_type:
            client_type.return_value.retrieve_materials.return_value = [(0, response)]
            material = resolver.get_material("warm sandstone ground surface")

        assert material is not None
        assert material.material_id == "Ground096A"
