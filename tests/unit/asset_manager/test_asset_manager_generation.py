import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from omegaconf import OmegaConf

from scenesmith.agent_utils.assets.asset_manager import AssetManager
from scenesmith.agent_utils.assets.asset_models import (
    AssetGenerationRequest,
    AssetGenerationResult,
    AssetPathConfig,
    _subscription_aware_worker_count,
)
from scenesmith.agent_utils.assets.image_generation import (
    AssetOperationType,
    OpenAIImageGenerator,
)
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerResponse,
)
from scenesmith.agent_utils.scene.room_parts.room_models import (
    AgentType,
    ObjectType,
    SceneObject,
)
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

    def test_subscription_cli_analysis_uses_one_honest_worker(self):
        with patch.dict(
            "os.environ", {"SCENESMITH_LLM_PROVIDER": "claude-cli"}, clear=False
        ):
            self.assertEqual(_subscription_aware_worker_count(8, 5), 1)

    def test_api_analysis_keeps_configured_parallelism(self):
        with patch.dict(
            "os.environ", {"SCENESMITH_LLM_PROVIDER": "anthropic"}, clear=False
        ):
            self.assertEqual(_subscription_aware_worker_count(3, 5), 3)

    def test_asset_generation_request_creation(self):
        """Test creating AssetGenerationRequest instances."""
        request = AssetGenerationRequest(
            object_descriptions=["Modern sofa", "Coffee table"],
            short_names=["modern_sofa", "coffee_table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.8], [1.2, 0.6, 0.45]],
            style_context="Modern minimalist living room",
            operation_type=AssetOperationType.INITIAL,
        )

        self.assertEqual(len(request.object_descriptions), 2)
        self.assertEqual(request.object_type, ObjectType.FURNITURE)
        self.assertEqual(request.style_context, "Modern minimalist living room")
        self.assertEqual(request.operation_type, AssetOperationType.INITIAL)

    def test_deterministic_router_preserves_structured_request(self):
        item = AssetManager._build_deterministic_asset_item(
            description="Sci-fi medical treatment bed",
            short_name="treatment_bed",
            dimensions=[2.1, 0.9, 0.8],
            object_type=ObjectType.FURNITURE,
        )

        self.assertEqual(item.short_name, "treatment_bed")
        self.assertEqual(item.dimensions, [2.1, 0.9, 0.8])
        self.assertEqual(item.object_type, ObjectType.FURNITURE)
        self.assertEqual(item.strategies, ["generated"])

    def test_deterministic_router_selects_specialized_strategies(self):
        cabinet = AssetManager._build_deterministic_asset_item(
            description="Sterile storage cabinet with sealed doors",
            short_name="storage_cabinet",
            dimensions=[1.0, 0.5, 1.8],
            object_type=ObjectType.FURNITURE,
        )
        poster = AssetManager._build_deterministic_asset_item(
            description="Framed medical poster",
            short_name="medical_poster",
            dimensions=[0.8, 0.03, 1.1],
            object_type=ObjectType.WALL_MOUNTED,
        )

        self.assertEqual(cabinet.strategies, ["articulated", "generated"])
        self.assertEqual(poster.strategies, ["thin_covering", "generated"])
        self.assertEqual(poster.thin_covering_type, "single_image")

    def test_deterministic_analysis_skips_router_model_call(self):
        self.asset_manager.cfg.asset_manager.router.deterministic_analysis = True
        self.asset_manager.router = MagicMock()
        request = AssetGenerationRequest(
            object_descriptions=["Medical bed"],
            short_names=["medical_bed"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.1, 0.9, 0.8]],
        )
        empty_result = AssetGenerationResult(successful_assets=[], failed_assets=[])

        with patch.object(
            self.asset_manager,
            "_generate_items_with_validation",
            return_value=empty_result,
        ) as generate:
            self.asset_manager._generate_assets_with_router(request)

        self.asset_manager.router.analyze_request.assert_not_called()
        routed_items = generate.call_args.kwargs["unique_items"]
        self.assertEqual(list(routed_items), ["Medical bed"])

    def test_catalog_axes_and_physics_are_deterministic(self):
        physics = AssetManager._deterministic_catalog_physics(
            description="Steel medical cabinet",
            desired_dimensions=[1.0, 0.5, 1.8],
            object_type=ObjectType.FURNITURE,
            canonical_up="0,0,-1",
            canonical_front="0,1,0",
        )

        self.assertEqual(physics.up_axis, "-Z")
        self.assertEqual(physics.front_axis, "+Y")
        self.assertEqual(physics.material, "metal")
        self.assertEqual(physics.mass_kg, 54.0)

    def test_missing_catalog_axes_use_scene_canonical_defaults(self):
        physics = AssetManager._deterministic_catalog_physics(
            description="Treatment bed",
            desired_dimensions=[2.0, 1.0, 0.75],
            object_type=ObjectType.FURNITURE,
            canonical_up=None,
            canonical_front=None,
        )

        self.assertEqual(physics.up_axis, "+Z")
        self.assertEqual(physics.front_axis, "+Y")

    def test_create_scene_object(self):
        """Test creating SceneObject from asset paths."""
        config = AssetPathConfig(
            description="A comfortable test chair",
            short_name="test_chair",
            image_path=Path("/test/image.png"),
            geometry_path=Path("/test/geometry.glb"),
            sdf_dir=Path("/test/sdf"),
        )
        sdf_path = Path("/test/asset.sdf")

        scene_obj = self.asset_manager._create_scene_object(
            config=config,
            object_type=ObjectType.FURNITURE,
            sdf_path=sdf_path,
            final_gltf_path=Path("/test/asset.gltf"),
        )

        self.assertIsInstance(scene_obj, SceneObject)
        self.assertEqual(scene_obj.name, "test_chair")
        self.assertEqual(scene_obj.description, "A comfortable test chair")
        self.assertEqual(scene_obj.object_type, ObjectType.FURNITURE)
        self.assertEqual(scene_obj.image_path, config.image_path)
        self.assertEqual(scene_obj.geometry_path, Path("/test/asset.gltf"))
        self.assertEqual(scene_obj.sdf_path, sdf_path)

    def test_initialization(self):
        """Test AssetManager initialization."""
        self.assertEqual(self.asset_manager.output_dir, self.output_dir)
        self.assertEqual(self.asset_manager.logger, self.mock_logger)

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_initial_operation(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test generate_assets with INITIAL operation type."""
        # Mock SDF file discovery - return one SDF file.
        mock_sdf_path = Path("/test/asset.sdf")
        mock_scale_mesh.return_value = (mock_sdf_path, 1.0)
        mock_glob.return_value = [mock_sdf_path]

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock asset client to return a valid response object.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [(0, GeometryGenerationServerResponse(geometry_path=str(mock_sdf_path)))]
        )

        request = AssetGenerationRequest(
            object_descriptions=["Modern sofa"],
            short_names=["modern_sofa"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.8]],
            style_context="Modern minimalist living room",
            operation_type=AssetOperationType.INITIAL,
        )

        result = self.asset_manager.generate_assets(request)

        # Verify image generation was called with correct contract.
        self.mock_image_generator.generate_images.assert_called_once()
        call_args = self.mock_image_generator.generate_images.call_args
        self.assertEqual(
            call_args.kwargs["style_prompt"], "Modern minimalist living room"
        )
        self.assertEqual(call_args.kwargs["object_descriptions"], ["Modern sofa"])
        self.assertEqual(len(call_args.kwargs["output_paths"]), 1)

        # Verify asset client was called.
        self.mock_geometry_client.generate_geometries.assert_called_once()

        # Verify result contract.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 1)
        self.assertIsInstance(result.successful_assets[0], SceneObject)
        self.assertEqual(result.successful_assets[0].name, "modern_sofa")
        self.assertEqual(result.successful_assets[0].object_type, ObjectType.FURNITURE)
        self.assertIn("generation_timestamp", result.successful_assets[0].metadata)

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_multiple_items(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test generate_assets with multiple items in a batch."""
        # Mock SDF file discovery - return one SDF file per call.
        mock_sdf_paths = [Path("/test/asset1.sdf"), Path("/test/asset2.sdf")]
        mock_glob.side_effect = [[path] for path in mock_sdf_paths]
        mock_scale_mesh.side_effect = [(path, 1.0) for path in mock_sdf_paths]

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock asset client to return valid response objects.
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
            operation_type=AssetOperationType.ADDITION,
        )

        result = self.asset_manager.generate_assets(request)

        # Verify batch processing contract.
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 2)
        self.mock_geometry_client.generate_geometries.assert_called_once()

        # Verify all items were processed with correct names.
        result_names = {obj.name for obj in result.successful_assets}
        expected_names = {"modern_sofa", "coffee_table"}
        self.assertEqual(result_names, expected_names)

        # Verify all results are proper SceneObjects.
        for obj in result.successful_assets:
            self.assertIsInstance(obj, SceneObject)
            self.assertEqual(obj.object_type, ObjectType.FURNITURE)

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_generate_assets_different_operation_types(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test generate_assets with different operation types."""
        # Mock SDF file discovery.
        mock_glob.return_value = [Path("/test/asset.sdf")]
        mock_scale_mesh.return_value = (Path("/test/asset.sdf"), 1.0)

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        for op_type in [
            AssetOperationType.INITIAL,
            AssetOperationType.ADDITION,
            AssetOperationType.REPLACEMENT,
        ]:
            with self.subTest(operation_type=op_type):
                # Set up mock for this iteration.
                self.mock_geometry_client.generate_geometries.return_value = iter(
                    [
                        (
                            0,
                            GeometryGenerationServerResponse(
                                geometry_path="/test/asset.sdf"
                            ),
                        )
                    ]
                )

                request = AssetGenerationRequest(
                    object_descriptions=["Test item"],
                    short_names=["test_item"],
                    object_type=ObjectType.FURNITURE,
                    desired_dimensions=[[1.0, 1.0, 1.0]],
                    operation_type=op_type,
                )

                result = self.asset_manager.generate_assets(request)

                # Verify contract: returns AssetGenerationResult with successful assets.
                self.assertIsInstance(result, AssetGenerationResult)
                self.assertTrue(result.all_succeeded)
                self.assertEqual(len(result.successful_assets), 1)
                self.assertIsInstance(result.successful_assets[0], SceneObject)

                # Verify image generation was called.
                self.mock_image_generator.generate_images.assert_called()

                # Reset mocks for next iteration.
                self.mock_image_generator.reset_mock()
                self.mock_geometry_client.reset_mock()

    def test_generate_assets_error_handling(self):
        """Test error handling in generate_assets."""
        # Mock image generation to fail.
        self.mock_image_generator.generate_images.side_effect = Exception(
            "Image generation failed"
        )

        request = AssetGenerationRequest(
            object_descriptions=["Test item"],
            short_names=["test_item"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[1.0, 1.0, 1.0]],
        )

        with self.assertRaises(Exception) as context:
            self.asset_manager.generate_assets(request)

        # Verify original error message is preserved (not wrapped).
        self.assertIn("Image generation failed", str(context.exception))

    def test_no_sdf_file_error(self):
        """Test error when no SDF file is generated."""
        # Create empty directory.
        sdf_dir = self.temp_dir / "empty_sdf"
        sdf_dir.mkdir()

        with self.assertRaises(RuntimeError) as context:
            self.asset_manager._find_sdf_file(sdf_dir)

        self.assertIn("No SDF file generated", str(context.exception))

    def test_multiple_sdf_files_error(self):
        """Test error when multiple SDF files are found."""
        # Create directory with multiple SDF files.
        sdf_dir = self.temp_dir / "multi_sdf"
        sdf_dir.mkdir()
        (sdf_dir / "asset1.sdf").touch()
        (sdf_dir / "asset2.sdf").touch()

        with self.assertRaises(RuntimeError) as context:
            self.asset_manager._find_sdf_file(sdf_dir)

        self.assertIn("Multiple SDF files generated", str(context.exception))

    def test_3d_generation_failure(self):
        """Test handling of 3D geometry generation failures."""
        self.mock_geometry_client.generate_geometries.side_effect = RuntimeError(
            "3D generation failed"
        )

        request = AssetGenerationRequest(
            object_descriptions=["Test chair"],
            short_names=["test_chair"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[0.5, 0.5, 0.9]],
        )

        with self.assertRaises(RuntimeError) as context:
            self.asset_manager.generate_assets(request)

        # Verify original error message is preserved (not wrapped at top level).
        self.assertIn("3D generation failed", str(context.exception))

    def test_asset_registry_integration(self):
        """Test that AssetManager integrates with AssetRegistry."""
        # Verify registry is initialized.
        self.assertIsNotNone(self.asset_manager.registry)
        self.assertEqual(self.asset_manager.registry.size(), 0)

    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    @patch("pathlib.Path.glob")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.AssetManager._extract_bounds_from_visual_mesh"
    )
    def test_assets_registered_after_generation(
        self, mock_extract_bounds, mock_glob, mock_scale_mesh
    ):
        """Test that generated assets are automatically registered."""
        # Mock SDF file discovery.
        mock_sdf_path = Path("/test/asset.sdf")
        mock_scale_mesh.return_value = (mock_sdf_path, 1.0)
        mock_glob.return_value = [mock_sdf_path]

        # Mock bounds extraction to return dummy bounds.
        mock_extract_bounds.return_value = (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        # Mock asset client to return a valid response object.
        self.mock_geometry_client.generate_geometries.return_value = iter(
            [(0, GeometryGenerationServerResponse(geometry_path=str(mock_sdf_path)))]
        )

        request = AssetGenerationRequest(
            object_descriptions=["Test chair"],
            short_names=["test_chair"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[0.5, 0.5, 0.9]],
        )

        result = self.asset_manager.generate_assets(request)

        # Verify asset was registered.
        self.assertEqual(self.asset_manager.registry.size(), 1)
        self.assertIsInstance(result, AssetGenerationResult)
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(result.successful_assets), 1)

        generated_asset = result.successful_assets[0]
        retrieved_asset = self.asset_manager.get_asset_by_id(generated_asset.object_id)
        self.assertEqual(retrieved_asset, generated_asset)
