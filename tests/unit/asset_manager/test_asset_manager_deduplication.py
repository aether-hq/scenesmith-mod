import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from omegaconf import OmegaConf

from scenesmith.agent_utils.assets.asset_manager import AssetManager
from scenesmith.agent_utils.assets.asset_models import (
    AssetGenerationRequest,
    AssetGenerationResult,
    FailedAsset,
)
from scenesmith.agent_utils.assets.image_generation import (
    AssetOperationType,
    OpenAIImageGenerator,
)
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerResponse,
)
from scenesmith.agent_utils.scene.room_parts.room_models import AgentType, ObjectType
from tests.unit.mock_utils import create_mock_logger


def create_mock_cfg():
    """Create mock configuration for AssetManager tests.

    Uses the config merge pattern to load actual config and override for testing.
    """
    # Load base configuration from actual config file.
    config_path = (
        Path(__file__).parents[3]
        / "configurations/furniture_agent/base_furniture_agent.yaml"
    )
    base_config = OmegaConf.load(config_path)

    # Define test overrides for fast testing.
    test_overrides = {
        "openai": {
            "model": "gpt-4o-mini",  # Cheaper model for testing
        },
        "asset_manager": {
            "general_asset_source": "generated",  # Avoid HSSD client initialization
            "reset_registry_based_on_style_change": True,  # Enable for testing
            "image_generation": {
                "parallel": False,  # Use sequential mode for tests
            },
            "router": {
                "enabled": False,  # Disable router for non-router tests
            },
        },
    }

    # Merge configurations (base config provides all other values).
    return OmegaConf.merge(base_config, test_overrides)


class TestAssetManager(unittest.TestCase):
    """Test AssetManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(self.temp_dir)
        self.mock_logger = create_mock_logger(self.output_dir)

        # Start persistent patches.
        self.patcher_image_gen = patch(
            "scenesmith.agent_utils.assets.asset_manager.create_image_generator"
        )
        self.patcher_geo_client = patch(
            "scenesmith.agent_utils.assets.asset_manager.GeometryGenerationClient"
        )
        # Patch the entire mesh-to-simulation pipeline to avoid complex file setup.
        self.patcher_mesh_conversion = patch.object(
            AssetManager, "_convert_mesh_to_simulation_asset"
        )

        self.patcher_image_gen.start()
        self.patcher_geo_client.start()
        mock_mesh_conversion = self.patcher_mesh_conversion.start()

        # Mock mesh conversion pipeline to return fake SDF path, bounding box, and scale.
        mock_mesh_conversion.return_value = (
            Path("/test/asset.sdf"),
            Path("/test/asset.gltf"),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            1.0,  # initial_scale
        )

        self.asset_manager = AssetManager(
            logger=self.mock_logger,
            vlm_service=MagicMock(),
            blender_server=MagicMock(),
            collision_client=MagicMock(),
            cfg=create_mock_cfg(),
            agent_type=AgentType.FURNITURE,
        )

        # Replace with proper mocks.
        self.mock_image_generator = MagicMock(spec=OpenAIImageGenerator)
        self.asset_manager.image_generator = self.mock_image_generator

        self.mock_geometry_client = MagicMock()
        self.asset_manager.geometry_client = self.mock_geometry_client

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop all patchers.
        self.patcher_mesh_conversion.stop()
        self.patcher_geo_client.stop()
        self.patcher_image_gen.stop()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("trimesh.load")
    def test_extract_bounds_from_visual_mesh_success(self, mock_trimesh_load):
        """Test that bounds extraction returns correct values from GLTF mesh."""
        # Mock trimesh mesh with known bounds and make it look like real Trimesh.
        mock_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_mesh.bounds = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]]
        mock_trimesh_load.return_value = mock_mesh

        # Create required file structure.
        sdf_path = self.temp_dir / "test_asset" / "test_asset.sdf"
        sdf_path.parent.mkdir(parents=True, exist_ok=True)
        sdf_path.write_text("<sdf></sdf>")

        # GLTF file should be alongside the SDF file.
        gltf_path = sdf_path.with_suffix(".gltf")
        gltf_path.write_text("{}")

        # Test the contract: returns tuple of min/max bounds.
        bbox_min, bbox_max = self.asset_manager._extract_bounds_from_visual_mesh(
            sdf_path
        )

        np.testing.assert_array_equal(bbox_min, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(bbox_max, [1.0, 2.0, 0.5])

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_with_duplicates_same_dimensions(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that duplicates with same dimensions are detected and removed."""
        # Mock SDF file discovery - return one SDF file per unique item (3 total).
        mock_sdf_paths = [
            Path("/test/desk.sdf"),
            Path("/test/chair.sdf"),
            Path("/test/printer.sdf"),
        ]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client to return only unique items (3).
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(mock_sdf_paths)
            ]
        )

        # Create request with duplicates.
        descriptions = [
            "Modern office desk",
            "Modern office desk",  # Duplicate of index 0
            "Ergonomic office chair",
            "Ergonomic office chair",  # Duplicate of index 2
            "Commercial laser printer",
        ]
        dimensions = [
            [1.5, 0.75, 0.75],
            [1.5, 0.75, 0.75],  # Same as index 0
            [0.6, 0.6, 1.0],
            [0.6, 0.6, 1.0],  # Same as index 2
            [0.5, 0.5, 0.4],
        ]
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=[
                "office_desk",
                "office_desk_2",
                "office_chair",
                "office_chair_2",
                "laser_printer",
            ],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=dimensions,
            style_context="Modern office",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Verify only unique items were generated (3 instead of 5).
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 3)

        # Verify duplicate info was stored.
        self.assertIsNotNone(self.asset_manager.last_duplicate_info)
        self.assertEqual(len(self.asset_manager.last_duplicate_info), 2)

        # Verify correct duplicates were detected.
        self.assertIn("Modern office desk", self.asset_manager.last_duplicate_info)
        self.assertIn("Ergonomic office chair", self.asset_manager.last_duplicate_info)
        self.assertEqual(
            self.asset_manager.last_duplicate_info["Modern office desk"], [1]
        )
        self.assertEqual(
            self.asset_manager.last_duplicate_info["Ergonomic office chair"], [3]
        )

        # Verify returned objects are correct.
        result_names = {obj.name for obj in result.successful_assets}
        expected_names = {"office_desk", "office_chair", "laser_printer"}
        self.assertEqual(result_names, expected_names)

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_with_duplicates_different_dimensions(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that duplicates with different dimensions are NOT deduplicated."""
        # Mock SDF file discovery.
        mock_sdf_paths = [Path("/test/table1.sdf"), Path("/test/table2.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(mock_sdf_paths)
            ]
        )

        # Same description but different dimensions.
        descriptions = ["Dining table", "Dining table"]
        dimensions = [[1.8, 0.9, 0.75], [2.0, 1.0, 0.75]]  # Different widths
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=["dining_table_1", "dining_table_2"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=dimensions,
            style_context="Modern dining room",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Both should be generated (no deduplication).
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 2)

        # No duplicates should be detected.
        self.assertIsNone(self.asset_manager.last_duplicate_info)

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_no_duplicates(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that no duplicates are detected when all items are unique."""
        # Mock SDF file discovery.
        mock_sdf_paths = [Path("/test/sofa.sdf"), Path("/test/table.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(mock_sdf_paths)
            ]
        )

        descriptions = ["Modern sofa", "Coffee table"]
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=["modern_sofa", "coffee_table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.8], [1.2, 0.6, 0.45]],
            style_context="Modern style",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # All items should be generated.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 2)

        # No duplicates should be detected.
        self.assertIsNone(self.asset_manager.last_duplicate_info)

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_multiple_duplicates_of_same_item(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test detection of multiple duplicates of the same item."""
        # Mock SDF file discovery.
        mock_sdf_paths = [Path("/test/chair.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (
                    0,
                    GeometryGenerationServerResponse(
                        geometry_path=str(mock_sdf_paths[0])
                    ),
                )
            ]
        )

        # Four identical chairs.
        descriptions = ["Dining chair"] * 4
        dimensions = [[0.5, 0.5, 0.9]] * 4
        request = AssetGenerationRequest(
            object_descriptions=descriptions,
            short_names=["chair_1", "chair_2", "chair_3", "chair_4"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=dimensions,
            style_context="Dining room",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Only one unique item should be generated.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 1)

        # Verify duplicate info.
        self.assertIsNotNone(self.asset_manager.last_duplicate_info)
        self.assertEqual(len(self.asset_manager.last_duplicate_info), 1)
        self.assertIn("Dining chair", self.asset_manager.last_duplicate_info)

        # Three duplicates should be detected (indices 1, 2, 3).
        self.assertEqual(
            self.asset_manager.last_duplicate_info["Dining chair"], [1, 2, 3]
        )

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_partial_success_continues_processing(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that partial success is handled gracefully.

        When one asset fails during conversion, remaining assets should still be
        processed.
        """
        # Mock SDF file discovery for successful assets.
        mock_sdf_paths = [Path("/test/bed.sdf"), Path("/test/chair.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client to return 3 geometries.
        all_geometry_paths = [
            Path("/test/bed.glb"),
            Path("/test/nightstand.glb"),
            Path("/test/chair.glb"),
        ]
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(all_geometry_paths)
            ]
        )

        # Mock _convert_mesh_to_simulation_asset to fail for index 1 (nightstand).
        original_convert = self.asset_manager._convert_mesh_to_simulation_asset

        def mock_convert_with_failure(
            geometry_path, config, object_type, desired_dimensions
        ):
            if "nightstand" in str(geometry_path):
                raise RuntimeError(
                    "Degenerate mesh: Z dimension is too small (0.000028m)"
                )
            # For successful assets, use the original mock behavior from setUp.
            return original_convert(
                geometry_path, config, object_type, desired_dimensions
            )

        with patch.object(
            self.asset_manager,
            "_convert_mesh_to_simulation_asset",
            side_effect=mock_convert_with_failure,
        ):
            request = AssetGenerationRequest(
                object_descriptions=["King bed", "Nightstand", "Accent chair"],
                short_names=["king_bed", "nightstand", "accent_chair"],
                object_type=ObjectType.FURNITURE,
                desired_dimensions=[[2.0, 2.0, 1.0], [0.5, 0.5, 0.6], [0.7, 0.7, 0.9]],
                style_context="Bedroom furniture",
                operation_type=AssetOperationType.INITIAL,
            )

            result = self.asset_manager.generate_assets(request)

        # Verify partial success structure.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertFalse(result.all_succeeded)
        self.assertTrue(result.has_failures)

        # Verify 2 assets succeeded (bed and chair).
        self.assertEqual(len(result.successful_assets), 2)
        success_names = {obj.name for obj in result.successful_assets}
        self.assertEqual(success_names, {"king_bed", "accent_chair"})

        # Verify 1 asset failed (nightstand).
        self.assertEqual(len(result.failed_assets), 1)
        failed_asset = result.failed_assets[0]
        self.assertIsInstance(failed_asset, FailedAsset)
        self.assertEqual(failed_asset.index, 1)
        self.assertEqual(failed_asset.description, "Nightstand")
        self.assertIn("Degenerate mesh", failed_asset.error_message)

        # Verify ALL geometries were attempted (critical benefit of issue #86).
        # The geometry client should have streamed all 3 geometries.
        self.mock_geometry_client.generate_geometries.assert_called_once()

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_multiple_failures_collected(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that multiple failures are collected and reported.

        Verifies that when multiple assets fail, all failures are collected and
        returned in the result.
        """
        # Mock SDF file discovery for the one successful asset.
        mock_sdf_path = Path("/test/table.sdf")
        mock_glob.side_effect = [[mock_sdf_path]]
        mock_scale_mesh.side_effect = [(mock_sdf_path, 1.0)]

        # Mock bounds extraction.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock geometry client to return 4 geometries.
        all_geometry_paths = [
            Path("/test/wardrobe.glb"),
            Path("/test/table.glb"),
            Path("/test/dresser.glb"),
            Path("/test/tv_stand.glb"),
        ]
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [
                (i, GeometryGenerationServerResponse(geometry_path=str(path)))
                for i, path in enumerate(all_geometry_paths)
            ]
        )

        # Mock _convert_mesh_to_simulation_asset to fail for indices 0, 2, 3.
        original_convert = self.asset_manager._convert_mesh_to_simulation_asset

        def mock_convert_with_failures(
            geometry_path, config, object_type, desired_dimensions
        ):
            path_str = str(geometry_path)
            if "wardrobe" in path_str:
                raise RuntimeError("Mesh too thin in X dimension")
            elif "dresser" in path_str:
                raise RuntimeError("VLM analysis failed")
            elif "tv_stand" in path_str:
                raise RuntimeError("CoACD decomposition timeout")
            # Table succeeds.
            return original_convert(
                geometry_path, config, object_type, desired_dimensions
            )

        with patch.object(
            self.asset_manager,
            "_convert_mesh_to_simulation_asset",
            side_effect=mock_convert_with_failures,
        ):
            request = AssetGenerationRequest(
                object_descriptions=["Wardrobe", "Coffee table", "Dresser", "TV stand"],
                short_names=["wardrobe", "coffee_table", "dresser", "tv_stand"],
                object_type=ObjectType.FURNITURE,
                desired_dimensions=[
                    [2.0, 0.6, 2.0],
                    [1.2, 0.6, 0.45],
                    [1.5, 0.5, 1.0],
                    [1.8, 0.4, 0.6],
                ],
                style_context="Modern furniture",
                operation_type=AssetOperationType.INITIAL,
            )

            result = self.asset_manager.generate_assets(request)

        # Verify partial success structure.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertFalse(result.all_succeeded)
        self.assertTrue(result.has_failures)

        # Verify 1 asset succeeded (coffee table).
        self.assertEqual(len(result.successful_assets), 1)
        self.assertEqual(result.successful_assets[0].name, "coffee_table")

        # Verify 3 assets failed.
        self.assertEqual(len(result.failed_assets), 3)

        # Verify failure details for each failed asset.
        failed_by_index = {fa.index: fa for fa in result.failed_assets}
        self.assertEqual(set(failed_by_index.keys()), {0, 2, 3})

        # Check wardrobe failure (index 0).
        self.assertEqual(failed_by_index[0].description, "Wardrobe")
        self.assertIn("too thin", failed_by_index[0].error_message)

        # Check dresser failure (index 2).
        self.assertEqual(failed_by_index[2].description, "Dresser")
        self.assertIn("VLM analysis", failed_by_index[2].error_message)

        # Check TV stand failure (index 3).
        self.assertEqual(failed_by_index[3].description, "TV stand")
        self.assertIn("CoACD", failed_by_index[3].error_message)
