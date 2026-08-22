from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from omegaconf import OmegaConf

from scenesmith.agent_utils.assets.asset_manager import AssetManager
from scenesmith.agent_utils.assets.asset_models import AssetPathConfig
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType


def test_hssd_gltf_axes_are_converted_to_blender_import_frame():
    assert AssetManager._gltf_axis_to_blender("+X") == "+X"
    assert AssetManager._gltf_axis_to_blender("+Y") == "+Z"
    assert AssetManager._gltf_axis_to_blender("-Z") == "+Y"


def test_hssd_without_authored_axes_keeps_blender_defaults():
    assert AssetManager._canonical_axes_for_blender(
        is_hssd=True,
        analyzed_up="+Z",
        analyzed_front="+Y",
        authored_up=None,
        authored_front=None,
    ) == ("+Z", "+Y")


def test_hssd_explicit_source_axes_are_converted_to_blender_frame():
    assert AssetManager._canonical_axes_for_blender(
        is_hssd=True,
        analyzed_up="+Z",
        analyzed_front="+Y",
        authored_up="+Z",
        authored_front="+Y",
    ) == ("-Y", "+Z")


def test_hssd_canonical_dimensions_and_bounds_use_exported_y_up_frame():
    desired = [1.0, 0.35, 2.0]
    bounds = np.array([[0.0, 0.0, -0.175], [0.9, 1.7, 0.175]])

    assert AssetManager._canonical_mesh_target_dimensions(desired, is_hssd=True) == [
        1.0,
        2.0,
        0.35,
    ]
    bbox_min, bbox_max = AssetManager._canonical_bounds_to_drake(bounds, is_hssd=True)
    np.testing.assert_allclose(bbox_min, [0.0, -0.175, 0.0])
    np.testing.assert_allclose(bbox_max, [0.9, 0.175, 1.7])


def test_non_hssd_canonical_dimensions_keep_y_up_conversion():
    desired = [1.0, 0.35, 2.0]

    assert AssetManager._canonical_mesh_target_dimensions(desired, is_hssd=False) == [
        1.0,
        2.0,
        0.35,
    ]


def test_tall_furniture_rejects_converted_height_far_below_target():
    compatible, reason = AssetManager._converted_dimensions_are_compatible(
        object_type=ObjectType.FURNITURE,
        desired_dimensions=[1.0, 0.35, 2.0],
        bbox_min=np.array([-0.5, -0.148, 0.0]),
        bbox_max=np.array([0.5, 0.148, 0.431]),
    )

    assert not compatible
    assert "height" in reason


def test_tall_furniture_accepts_height_at_sixty_percent_of_target():
    compatible, _ = AssetManager._converted_dimensions_are_compatible(
        object_type=ObjectType.FURNITURE,
        desired_dimensions=[0.7, 0.7, 2.0],
        bbox_min=np.array([-0.35, -0.35, 0.0]),
        bbox_max=np.array([0.35, 0.35, 1.2]),
    )

    assert compatible


def test_unversioned_hssd_cache_entry_is_quarantined():
    stale_bookcase = SimpleNamespace(
        object_id="library_bookcase_0",
        name="library_bookcase",
        description="full-height Renaissance library bookcase with visible books",
        metadata={
            "asset_source": "hssd",
            "asset_quality_score": 0.76,
            "catalog_semantics": "wooden bookcase hssd/wordnet/bookcase.n.01",
        },
    )
    discarded = []
    manager = object.__new__(AssetManager)
    manager.registry = SimpleNamespace(
        list_all=lambda: [stale_bookcase],
        discard=discarded.append,
    )

    manager._quarantine_incompatible_cached_assets()

    assert discarded == ["library_bookcase_0"]


def test_short_bare_full_height_shelf_cache_entry_is_quarantined():
    bare_shelf = SimpleNamespace(
        object_id="library_bookshelf_0",
        name="library_bookshelf",
        description=(
            "Full-height library bookshelf in dark wood with adjustable shelves"
        ),
        bbox_min=np.array([-0.414, -0.175, 0.0]),
        bbox_max=np.array([0.414, 0.175, 1.242]),
        metadata={
            "asset_source": "polyhaven",
            "asset_quality_score": 1.0,
            "catalog_semantics": (
                "Wooden Bookshelf Worn antique bookcase rustic shelves storage "
                "polyhaven/Furniture/Storage Furniture/Shelving & Bookcases"
            ),
        },
    )
    discarded = []
    manager = object.__new__(AssetManager)
    manager.registry = SimpleNamespace(
        list_all=lambda: [bare_shelf],
        discard=discarded.append,
    )

    manager._quarantine_incompatible_cached_assets()

    assert discarded == ["library_bookshelf_0"]


def test_router_hssd_metadata_stamps_current_canonical_conversion():
    assert AssetManager._asset_conversion_metadata("hssd") == {
        "canonical_conversion_version": 4
    }
    assert AssetManager._asset_conversion_metadata("objaverse") == {}


def _conversion_boundary_manager(tmp_path: Path) -> AssetManager:
    manager = object.__new__(AssetManager)
    manager.collision_client = MagicMock()
    manager.blender_server = MagicMock()
    manager.cfg = SimpleNamespace(
        asset_manager=SimpleNamespace(floater_distance_threshold=0.05)
    )
    manager.debug_dir = tmp_path / "debug"
    manager.vlm_service = MagicMock()
    manager.side_view_elevation_degrees = 15.0
    manager.num_side_views_for_physics_analysis = 4
    return manager


def test_catalog_conversion_skips_generated_mesh_floater_cleanup(tmp_path):
    """Authored non-watertight catalog furniture must retain all components."""
    manager = _conversion_boundary_manager(tmp_path)
    config = AssetPathConfig(
        description="Full-height Renaissance library bookcase",
        short_name="bookcase",
        image_path=tmp_path / "bookcase.png",
        geometry_path=tmp_path / "bookcase.glb",
        sdf_dir=tmp_path / "sdf",
    )

    with (
        patch(
            "scenesmith.agent_utils.assets.asset_manager.remove_mesh_floaters"
        ) as remove,
        patch.object(
            manager,
            "_deterministic_catalog_physics",
            side_effect=RuntimeError("stop after cleanup boundary"),
        ),
    ):
        try:
            manager._convert_mesh_to_simulation_asset(
                config.geometry_path,
                config,
                ObjectType.FURNITURE,
                desired_dimensions=[1.0, 0.35, 2.0],
                asset_source="hssd",
            )
        except RuntimeError as error:
            assert str(error) == "stop after cleanup boundary"
        else:
            raise AssertionError("conversion did not reach catalog physics boundary")

    remove.assert_not_called()


def test_generated_conversion_keeps_mesh_floater_cleanup(tmp_path):
    manager = _conversion_boundary_manager(tmp_path)
    config = AssetPathConfig(
        description="Generated chair",
        short_name="chair",
        image_path=tmp_path / "chair.png",
        geometry_path=tmp_path / "chair.glb",
        sdf_dir=tmp_path / "sdf",
    )

    with (
        patch(
            "scenesmith.agent_utils.assets.asset_manager.remove_mesh_floaters"
        ) as remove,
        patch(
            "scenesmith.agent_utils.assets.asset_manager."
            "analyze_mesh_orientation_and_material",
            side_effect=RuntimeError("stop after cleanup boundary"),
        ),
    ):
        try:
            manager._convert_mesh_to_simulation_asset(
                config.geometry_path,
                config,
                ObjectType.FURNITURE,
                asset_source="generated",
            )
        except RuntimeError as error:
            assert str(error) == "stop after cleanup boundary"
        else:
            raise AssertionError("conversion did not reach generated analysis boundary")

    remove.assert_called_once()


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
