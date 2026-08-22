import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from omegaconf import OmegaConf

from scenesmith.agent_utils.assets.asset_manager import AssetManager
from scenesmith.agent_utils.assets.asset_models import AssetGenerationRequest
from scenesmith.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerResponse,
)
from scenesmith.agent_utils.physics.mesh_physics_analyzer import MeshPhysicsAnalysis
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


class TestAssetManagerDimensionControl(unittest.TestCase):
    """Test AssetManager dimension control functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(self.temp_dir)
        self.mock_logger = create_mock_logger(self.output_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_asset_generation_request_with_dimensions(self):
        """Test AssetGenerationRequest with desired_dimensions."""
        request = AssetGenerationRequest(
            object_descriptions=["Modern sofa", "Coffee table"],
            short_names=["modern_sofa", "coffee_table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[[2.0, 0.9, 0.85], [1.2, 0.6, 0.45]],
        )

        self.assertEqual(len(request.desired_dimensions), 2)
        self.assertEqual(request.desired_dimensions[0], [2.0, 0.9, 0.85])
        self.assertEqual(request.desired_dimensions[1], [1.2, 0.6, 0.45])

    def test_validate_dimensions_mismatch(self):
        """Test validation error when dimensions don't match descriptions."""
        with (
            patch("scenesmith.agent_utils.assets.asset_manager.create_image_generator"),
            patch(
                "scenesmith.agent_utils.assets.asset_manager.GeometryGenerationClient"
            ),
        ):
            asset_manager = AssetManager(
                logger=self.mock_logger,
                vlm_service=MagicMock(),
                blender_server=MagicMock(),
                collision_client=MagicMock(),
                cfg=create_mock_cfg(),
                agent_type=AgentType.FURNITURE,
            )

        # Create request with mismatched dimensions.
        request = AssetGenerationRequest(
            object_descriptions=["Sofa", "Table"],
            short_names=["sofa", "table"],
            object_type=ObjectType.FURNITURE,
            desired_dimensions=[
                (2.0, 0.9, 0.85)
            ],  # Only one dimension for two objects.
        )

        with self.assertRaises(ValueError) as context:
            asset_manager.generate_assets(request)

        self.assertIn("Mismatch between desired_dimensions", str(context.exception))

    @patch("scenesmith.agent_utils.assets.asset_manager.generate_drake_sdf")
    @patch("scenesmith.agent_utils.assets.asset_manager.canonicalize_mesh")
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.analyze_mesh_orientation_and_material"
    )
    @patch(
        "scenesmith.agent_utils.assets.asset_manager.scale_mesh_uniformly_to_dimensions"
    )
    def test_mesh_scaling_when_dimensions_provided(
        self,
        mock_scale_mesh,
        mock_analyze,
        mock_canon,
        mock_sdf,
    ):
        """Test that mesh scaling is called when dimensions are provided."""
        # Mock VLM analysis.
        mock_analyze.return_value = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        with (
            patch("scenesmith.agent_utils.assets.asset_manager.create_image_generator"),
            patch(
                "scenesmith.agent_utils.assets.asset_manager.GeometryGenerationClient"
            ),
        ):
            asset_manager = AssetManager(
                logger=self.mock_logger,
                vlm_service=MagicMock(),
                blender_server=MagicMock(),
                collision_client=MagicMock(),
                cfg=create_mock_cfg(),
                agent_type=AgentType.FURNITURE,
            )

        # Mock geometry client to return a geometry path.
        mock_response = GeometryGenerationServerResponse(
            geometry_path=str(self.temp_dir / "test.glb")
        )
        asset_manager.geometry_client.generate_geometries = MagicMock(
            return_value=[(0, mock_response)]
        )

        # Mock image generator.
        asset_manager.image_generator.generate_images = MagicMock()

        # Create actual test geometry file.
        test_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        test_geometry_path = Path(mock_response.geometry_path)
        test_geometry_path.parent.mkdir(parents=True, exist_ok=True)
        test_mesh.export(test_geometry_path)

        # Mock the SDF creation and mesh files.
        sdf_dir = self.temp_dir / "generated_assets" / "sdf" / "test_1234567890"
        sdf_dir.mkdir(parents=True, exist_ok=True)
        sdf_path = sdf_dir / "test.sdf"
        gltf_path = sdf_dir / "test.gltf"

        # Create dummy SDF and GLTF files.
        sdf_path.write_text("<sdf></sdf>")
        test_mesh.export(gltf_path)

        # Mock canonicalize_mesh to create canonical file.
        def mock_canon_side_effect(gltf_path, output_path, **kwargs):
            test_mesh.export(output_path)

        mock_canon.side_effect = mock_canon_side_effect

        # Mock scale_mesh_uniformly_to_dimensions to create scaled file.
        def mock_scale_side_effect(
            mesh_path, desired_dimensions, output_path, **kwargs
        ):
            test_mesh.export(output_path)
            return (output_path, 1.5)

        mock_scale_mesh.side_effect = mock_scale_side_effect

        # Mock blender_server.convert_glb_to_gltf to create GLTF file.
        def mock_convert_side_effect(input_path, output_path, export_yup=False):
            test_mesh.export(output_path)
            return output_path

        asset_manager.blender_server.convert_glb_to_gltf.side_effect = (
            mock_convert_side_effect
        )

        # collision_client is already mocked in AssetManager init.

        # Mock _find_sdf_file and _extract_bounds_from_visual_mesh.
        with (
            patch.object(asset_manager, "_find_sdf_file", return_value=sdf_path),
            patch.object(
                asset_manager,
                "_extract_bounds_from_visual_mesh",
                return_value=(
                    np.array([0, 0, 0]),
                    np.array([1, 1, 1]),
                ),
            ),
        ):
            # Create request with dimensions.
            request = AssetGenerationRequest(
                object_descriptions=["Test object"],
                short_names=["test"],
                object_type=ObjectType.FURNITURE,
                desired_dimensions=[[1.8, 0.9, 0.75]],
            )

            # Generate assets.
            asset_manager.generate_assets(request)

        # Verify scale_mesh_uniformly_to_dimensions was called.
        mock_scale_mesh.assert_called_once()
        call_args = mock_scale_mesh.call_args
        self.assertEqual(call_args[1]["desired_dimensions"], [1.8, 0.75, 0.9])


if __name__ == "__main__":
    unittest.main()
