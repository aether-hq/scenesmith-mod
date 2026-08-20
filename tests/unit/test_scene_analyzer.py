import json
import shutil
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from omegaconf import OmegaConf

from scenesmith.agent_utils.room import ObjectType, RoomScene
from scenesmith.agent_utils.scene_analyzer import SceneAnalyzer


class TestSceneAnalyzer(unittest.TestCase):
    """Test SceneAnalyzer class contracts."""

    # Test configuration constants.
    TEST_MODEL = "gpt-4o-mini"
    TEST_REASONING_EFFORT = "low"

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_vlm_service = Mock()
        self.mock_rendering_manager = Mock()
        self.mock_scene = Mock(spec=RoomScene)

        # Create mock BlenderServer.
        self.mock_blender_server = Mock()
        self.mock_blender_server.is_running.return_value = True

        # Create test config (only OpenAI settings needed).
        test_config_dict = {
            "openai": {
                "model": self.TEST_MODEL,
                "vision_detail": "low",
                "reasoning_effort": {"scene_critique": self.TEST_REASONING_EFFORT},
                "verbosity": {"scene_critique": "low"},
            },
        }
        # Convert to OmegaConf to match expected structure.
        self.test_config = OmegaConf.create(test_config_dict)

        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.mock_vlm_service,
            rendering_manager=self.mock_rendering_manager,
            cfg=self.test_config,
            blender_server=self.mock_blender_server,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scene_analyzer_initialization(self):
        """Test that SceneAnalyzer initializes properly."""
        self.assertIsNotNone(self.scene_analyzer)
        self.assertEqual(self.scene_analyzer.vlm_service, self.mock_vlm_service)
        self.assertEqual(
            self.scene_analyzer.rendering_manager, self.mock_rendering_manager
        )
        self.assertEqual(self.scene_analyzer.cfg, self.test_config)
        self.assertEqual(self.scene_analyzer.blender_server, self.mock_blender_server)

    def test_configuration_access(self):
        """Test that SceneAnalyzer can access configuration values."""
        # Verify configuration was stored and accessible.
        self.assertEqual(self.scene_analyzer.cfg["openai"]["model"], self.TEST_MODEL)
        self.assertEqual(
            self.scene_analyzer.cfg["openai"]["reasoning_effort"]["scene_critique"],
            self.TEST_REASONING_EFFORT,
        )

def _support_surface(object_id: str, elevation: float):
    return SimpleNamespace(
        object_id=object_id,
        object_type=ObjectType.FURNITURE,
        immutable=False,
        bbox_min=np.array([-0.5, -0.2, 0.0]),
        bbox_max=np.array([0.5, 0.2, 2.4]),
        name="renaissance_bookcase",
        description="full-height library bookcase filled with visible books",
        transform=SimpleNamespace(
            translation=lambda: np.array([0.0, 0.0, elevation])
        ),
    )


def _deterministic_analyzer(max_furniture: int = 3):
    return SceneAnalyzer(
        vlm_service=Mock(),
        rendering_manager=Mock(),
        cfg=OmegaConf.create(
            {
                "furniture_selection": {
                    "mode": "deterministic",
                    "max_furniture": max_furniture,
                }
            }
        ),
        blender_server=Mock(),
    )


def test_collection_library_details_use_blueprint_style_across_levels(tmp_path):
    scene_dir = tmp_path / "scene_000" / "room_room"
    scene_dir.mkdir(parents=True)
    (scene_dir.parent / "scene_blueprint.json").write_text(
        json.dumps(
            {
                "source_prompt": (
                    "a multi-level library with thousands of books and gorgeous "
                    "Renaissance decor"
                ),
                "furniture_groups": [
                    {
                        "roles": {"bookshelf": 15},
                        "density": "layered",
                    }
                ],
                "design_tokens": {
                    "style_keywords": [
                        "Renaissance",
                        "ornate classical",
                        "grand library",
                    ],
                    "palette": ["dark walnut", "burgundy", "antique gold"],
                    "material_roles": {
                        "floor": "dark walnut parquet",
                        "walls": "warm ivory plaster with carved stone trim",
                    },
                    "lighting_mood": "warm dramatic gallery light",
                    "focal_hierarchy": [
                        "book-lined galleries",
                        "classical marble statues",
                    ],
                },
            }
        )
    )
    objects = {
        surface.object_id: surface
        for elevation in (0.0, 4.0, 8.0)
        for surface in (
            _support_surface(f"bookcase_{int(elevation)}_{index}", elevation)
            for index in range(4)
        )
    }
    scene = SimpleNamespace(
        scene_dir=scene_dir,
        text_description="large multi-level library with thousands of books",
        objects=objects,
        room_geometry=SimpleNamespace(floor=None),
    )

    selections = _deterministic_analyzer().analyze_furniture_for_manipulands(
        scene, prompt_enum=Mock()
    )

    assert len(selections) == 9
    selected_levels = [
        objects[str(selection.furniture_id)].transform.translation()[2]
        for selection in selections
    ]
    assert {level: selected_levels.count(level) for level in set(selected_levels)} == {
        0.0: 3,
        4.0: 3,
        8.0: 3,
    }
    assert all("Renaissance" in selection.style_notes for selection in selections)
    assert all("dark walnut" in selection.style_notes for selection in selections)
    assert all("Avoid sparse" in selection.style_notes for selection in selections)
    assert all(
        "thousands of books" in selection.prompt_constraints
        for selection in selections
    )
    assert all(
        "dense rows of visible" in selection.suggested_items
        for selection in selections
    )


def test_ordinary_room_keeps_configured_sparse_surface_limit(tmp_path):
    objects = {
        surface.object_id: surface
        for surface in (_support_surface(f"bookcase_{index}", 0.0) for index in range(6))
    }
    scene = SimpleNamespace(
        scene_dir=tmp_path,
        text_description="a practical neighborhood reading room",
        objects=objects,
        room_geometry=SimpleNamespace(floor=None),
    )

    selections = _deterministic_analyzer().analyze_furniture_for_manipulands(
        scene, prompt_enum=Mock()
    )

    assert len(selections) == 3
    assert all(
        selection.style_notes.startswith("Sparse, functional")
        for selection in selections
    )


if __name__ == "__main__":
    unittest.main()
