import logging

from pathlib import Path

import bpy

from scenesmith.agent_utils.blender.geometry.camera_utils import (
    configure_camera_from_params,
)
from scenesmith.agent_utils.blender.geometry.view_generation_mixin import (
    ViewGenerationMixin,
)
from scenesmith.agent_utils.blender.overlays.coordinate_grid_mixin import (
    CoordinateGridMixin,
)
from scenesmith.agent_utils.blender.params import RenderParams
from scenesmith.agent_utils.blender.render_settings import (
    apply_image_type_settings,
    apply_render_settings,
)
from scenesmith.agent_utils.blender.renderer_mixins import wall_views as _wall_views
from scenesmith.agent_utils.blender.renderer_mixins.articulated import (
    ArticulatedRenderingMixin,
)
from scenesmith.agent_utils.blender.renderer_mixins.camera_setup import (
    CameraSetupRenderingMixin,
)
from scenesmith.agent_utils.blender.renderer_mixins.floor_plan import (
    FloorPlanRenderingMixin,
)
from scenesmith.agent_utils.blender.renderer_mixins.multiview import (
    MultiviewRenderingMixin,
)
from scenesmith.agent_utils.blender.renderer_mixins.observation import (
    AgentObservationRenderingMixin,
)
from scenesmith.agent_utils.blender.renderer_mixins.render_view import (
    ViewPostprocessingMixin,
)
from scenesmith.agent_utils.blender.renderer_mixins.wall_views import (
    WallViewRenderingMixin,
)
from scenesmith.agent_utils.blender.surfaces.scene_setup_mixin import SceneSetupMixin
from scenesmith.agent_utils.blender.surfaces.surface_rendering_mixin import (
    SurfaceRenderingMixin,
)

console_logger = logging.getLogger(__name__)

# Rendering constants.
DEFAULT_LIGHT_ENERGY = 1000
DEFAULT_LIGHT_POSITION = (4.0, 1.0, 6.0)
DEFAULT_NUM_SIDE_VIEWS = 4
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
# EEVEE TAA samples for asset validation renders. Using 8 as a good balance
# between quality and speed. EEVEE is ~6x faster than CYCLES.
EEVEE_ASSET_VALIDATION_SAMPLES = 8
# CYCLES samples for offline CLIP embedding renders (higher quality, slower).
CYCLES_CLIP_SAMPLES = 20
VLM_ANALYSIS_LIGHT_ENERGY = 2000
# Lower light energy for articulated objects (more reflective materials).
ARTICULATED_LIGHT_ENERGY = 500
# Lower light energy for material/texture validation (avoid washing out colors).
MATERIAL_VALIDATION_LIGHT_ENERGY = 300

# Camera constants.
DEFAULT_CAMERA_LENS_MM = 50
DEFAULT_CAMERA_SENSOR_WIDTH_MM = 36
DEFAULT_CAMERA_CLIP_START = 0.01
DEFAULT_CAMERA_CLIP_END = 100000
CAMERA_DISTANCE_MARGIN_MULTIPLIER = (
    1 / 0.8
)  # Scene occupies ~80% of image (10% margin per side).
LIGHT_DISTANCE_RATIO = 0.1
# Offset above lower surfaces for camera near-plane clipping (meters).
# This clips furniture geometry above lower surfaces so they're visible from top-down views.
LOWER_SURFACE_CLIP_OFFSET_M = 0.05

# Multi-view rendering constants.
COORDINATE_FRAME_SCALE_FACTOR = 0.01


class BlenderRenderer(
    SurfaceRenderingMixin,
    CoordinateGridMixin,
    ViewGenerationMixin,
    SceneSetupMixin,
    MultiviewRenderingMixin,
    FloorPlanRenderingMixin,
    ArticulatedRenderingMixin,
    AgentObservationRenderingMixin,
    CameraSetupRenderingMixin,
    ViewPostprocessingMixin,
    WallViewRenderingMixin,
):
    """Encapsulates access to Blender rendering functionality.

    Note that even though this is a class, bpy is a singleton so likewise you
    should only ever create one instance of this class.
    """

    def __init__(
        self,
        blend_file: Path | None = None,
        bpy_settings_file: Path | None = None,
    ) -> None:
        """Initialize the Blender renderer.

        Args:
            blend_file: Optional path to a .blend file to use as base scene.
            bpy_settings_file: Optional path to a .py file with Blender settings.
        """
        self._blend_file = blend_file
        self._bpy_settings_file = bpy_settings_file
        self._client_objects = None

    def add_default_light_source(self) -> None:
        """Add a default point light source to the scene."""
        # Create a new light data block.
        light_data = bpy.data.lights.new(name="DefaultLight", type="POINT")
        light_data.energy = DEFAULT_LIGHT_ENERGY

        # Create new object with the light datablock.
        light_object = bpy.data.objects.new(name="DefaultLight", object_data=light_data)

        # Link light object to scene collection.
        bpy.context.collection.objects.link(light_object)

        # Set light position.
        light_object.location = DEFAULT_LIGHT_POSITION

    def render_image(self, params: RenderParams, output_path: Path) -> None:
        """Render the current scene with the given parameters.

        Args:
            params: The rendering parameters.
            output_path: Path where the rendered image will be saved.
        """
        # Set up scene and import glTF.
        self._setup_scene(params)
        self._import_and_organize_gltf(params.scene)

        # Configure camera.
        configure_camera_from_params(params=params)

        # Apply render settings.
        apply_render_settings(params=params)
        apply_image_type_settings(params=params, client_objects=self._client_objects)

        # Set output path and render.
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

    def save_blend_file(self, *args, **kwargs):
        """Preserve the historic patch point for Blender's singleton module."""

        _wall_views.bpy = bpy
        return super().save_blend_file(*args, **kwargs)
