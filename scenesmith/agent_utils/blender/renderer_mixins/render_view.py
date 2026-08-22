import logging

from pathlib import Path

import bpy

from mathutils import Vector
from omegaconf import DictConfig

from scenesmith.agent_utils.blender.geometry.camera_utils import (
    calculate_camera_distance,
    configure_metric_camera,
)
from scenesmith.agent_utils.blender.geometry.scene_utils import compute_scene_bounds
from scenesmith.agent_utils.blender.overlays.coordinate_frame import (
    remove_coordinate_frame,
    remove_wall_coordinate_frame,
)
from scenesmith.agent_utils.blender.overlays.image_annotations import (
    add_opening_labels_pil,
    add_set_of_mark_labels_pil,
    annotate_image_with_coordinates,
)
from scenesmith.agent_utils.blender.overlays.scene_annotations import (
    remove_annotation_objects,
)
from scenesmith.agent_utils.blender.overlays.wall_annotations import (
    add_wall_grid_annotations_pil,
    add_wall_labels_to_top_view,
    add_wall_surface_id_label,
)
from scenesmith.agent_utils.blender.params import RenderParams
from scenesmith.agent_utils.blender.render_dataclasses import OverlayRenderingSetup
from scenesmith.agent_utils.blender.render_settings import (
    apply_render_settings,
    setup_metric_world,
)
from scenesmith.agent_utils.blender.surfaces.surface_utils import (
    add_surface_id_label,
    add_surface_labels_to_side_view,
)
from scenesmith.agent_utils.scene.house_parts.openings import ClearanceOpeningData

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


class ViewPostprocessingMixin:
    """Single-view rendering, overlays, postprocessing, and state reset."""

    def _render_and_postprocess_view(
        self,
        view: dict[str, str | Vector | bool],
        output_path: Path,
        setup_data: OverlayRenderingSetup,
        scene_objects: list[dict] | None,
        annotations: DictConfig,
        openings: list[ClearanceOpeningData] | None = None,
    ) -> None:
        """Render view and apply PIL post-processing annotations.

        Args:
            view: Dictionary containing view information.
            output_path: Path where rendered image will be saved.
            setup_data: Metric rendering setup data.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Annotation config flags.
            openings: Optional opening metadata for door/window labels.

        Raises:
            RuntimeError: If rendering or post-processing fails.
        """
        # Render to output path.
        scene = bpy.context.scene
        scene.render.filepath = str(output_path)

        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            raise RuntimeError(f"Blender render failed for {view['name']}: {e}")

        if not output_path.exists():
            raise RuntimeError(f"Render failed to create file: {output_path}")

        # Add coordinate annotations (PIL post-processing for metric markers).
        is_top_view = not view.get("is_side", True)
        is_multi_surface_mode = (
            self._support_surfaces is not None and len(self._support_surfaces) > 1
        )

        # Check if this is a wall orthographic view (has its own grid annotations).
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        is_wall_context = view.get("is_wall_context", False)

        # For multi-surface side views, skip coordinate markers and add surface labels.
        if not is_top_view and is_multi_surface_mode:
            try:
                add_surface_labels_to_side_view(
                    image_path=output_path,
                    camera_obj=setup_data.camera_obj,
                    support_surfaces=self._support_surfaces,
                    surface_colors=self._surface_colors,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add surface labels {view['name']}: {e}"
                ) from e
        elif is_wall_orthographic:
            # Wall orthographic views use their own grid annotation system.
            # Skip regular room coordinate markers.
            pass
        elif is_wall_context:
            # Wall context views show furniture for context.
            # Skip red floor coordinate grid - it's not useful for wall placement.
            pass
        elif not getattr(annotations, "enable_coordinate_grid", True):
            # Skip coordinate grid when disabled (e.g., furniture_selection mode).
            pass
        else:
            # Regular coordinate markers for single-surface or top views.
            try:
                is_drawer_view = view.get("is_drawer_view", False)
                # Pass ceiling_height and room_bounds for stable grid markers.
                view_ceiling_height = view.get("ceiling_height", None)
                view_room_bounds = view.get("room_bounds", None)
                marks = self._get_visual_marks(
                    scene=scene,
                    camera_obj=setup_data.camera_obj,
                    is_top_view=is_top_view,
                    is_drawer_view=is_drawer_view,
                    ceiling_height=view_ceiling_height,
                    room_bounds=view_room_bounds,
                )
                if marks:
                    annotate_image_with_coordinates(image_path=output_path, marks=marks)
                # Debug: Visualize convex hull outline for multi-surface mode.
                if (
                    is_multi_surface_mode
                    and is_top_view
                    and getattr(annotations, "enable_convex_hull_debug", False)
                ):
                    self._debug_visualize_convex_hull(
                        image_path=output_path, camera_obj=setup_data.camera_obj
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to annotate {view['name']}: {e}") from e

        # Add set-of-mark labels (PIL post-processing for guaranteed top layer).
        # Include wall_context views for labels (is_wall_context flag is on view).
        is_wall_context = view.get("is_wall_context", False)
        should_add_labels = (is_top_view or is_wall_context) and scene_objects
        if should_add_labels and annotations.enable_set_of_mark_labels:
            try:
                # Extract rendering mode from annotations.
                rendering_mode = getattr(annotations, "rendering_mode", "furniture")
                # For per-surface views, filter labels by current surface.
                current_surface_id = (
                    self._current_surface_id
                    if hasattr(self, "_current_surface_id")
                    else None
                )
                # Get annotate_object_types filter if specified.
                annotate_object_types = getattr(
                    annotations, "annotate_object_types", None
                )
                # Filter scene_objects by current surface for per-surface top views.
                filtered_scene_objects = scene_objects
                if current_surface_id is not None:
                    filtered_scene_objects = self._filter_objects_by_surface(
                        scene_objects=scene_objects,
                        current_surface_id=current_surface_id,
                    )
                add_set_of_mark_labels_pil(
                    image_path=output_path,
                    scene_objects=filtered_scene_objects,
                    camera_obj=setup_data.camera_obj,
                    rendering_mode=rendering_mode,
                    current_surface_id=current_surface_id,
                    annotate_object_types=annotate_object_types,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add set-of-mark labels {view['name']}: {e}"
                ) from e

        # Add wall labels to wall_context top-down views.
        if is_wall_context and hasattr(self, "_wall_surfaces_for_labels"):
            wall_surfaces = self._wall_surfaces_for_labels
            if wall_surfaces:
                try:
                    add_wall_labels_to_top_view(
                        image_path=output_path,
                        camera_obj=setup_data.camera_obj,
                        wall_surfaces=wall_surfaces,
                    )
                except Exception as e:
                    console_logger.warning(
                        f"Failed to add wall labels for {view['name']}: {e}"
                    )

        # Add opening labels (door/window/open connection) for top views.
        # Skip for wall views - only show wall labels, not openings.
        is_wall_view = is_wall_context or is_wall_orthographic
        if is_top_view and openings and not is_wall_view:
            try:
                add_opening_labels_pil(
                    image_path=output_path,
                    openings=openings,
                    camera_obj=setup_data.camera_obj,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add opening labels {view['name']}: {e}"
                ) from e

        # Add surface ID label for multi-surface top views.
        if (
            is_top_view
            and hasattr(self, "_current_surface_id")
            and self._current_surface_id
        ):
            try:
                add_surface_id_label(
                    image_path=output_path,
                    surface_id=self._current_surface_id,
                    surface_colors=self._surface_colors,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add surface ID label {view['name']}: {e}"
                ) from e

        # Add wall orthographic annotations (grid and excluded regions).
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        if is_wall_orthographic:
            wall_surface = view.get("wall_surface", {})
            if wall_surface:
                try:
                    # Add coordinate grid overlay.
                    if getattr(annotations, "enable_wall_grid", True):
                        add_wall_grid_annotations_pil(
                            image_path=output_path,
                            wall_surface_data=wall_surface,
                            camera_obj=setup_data.camera_obj,
                            num_markers=getattr(annotations, "num_markers", 5),
                        )
                    surface_id = wall_surface.get(
                        "surface_id", wall_surface.get("wall_id", "")
                    )
                    if surface_id:
                        add_wall_surface_id_label(
                            image_path=output_path, wall_surface_id=surface_id
                        )
                except Exception as e:
                    console_logger.warning(
                        f"Failed to add wall annotations for {view['name']}: {e}"
                    )

        # Clean up overlays for next view.
        remove_coordinate_frame()
        remove_wall_coordinate_frame()
        remove_annotation_objects()

    def _render_single_view_with_metric_overlay_to_path(
        self,
        view: dict[str, str | Vector | bool],
        setup_data: OverlayRenderingSetup,
        output_path: Path,
        scene_objects: list[dict] | None = None,
        annotations: dict | None = None,
        openings: list[ClearanceOpeningData] | None = None,
    ) -> None:
        """Render a single view with metric overlays to specified path.

        Args:
            view: Dictionary containing view information.
            setup_data: Metric rendering setup data.
            output_path: Path where rendered image will be saved.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Optional annotation config flags.
            openings: Optional opening metadata for door/window labels.

        Raises:
            RuntimeError: If Blender rendering fails.
        """
        self._setup_camera_and_coordinate_frame(
            setup_data=setup_data, view=view, annotations=annotations
        )
        self._apply_view_annotations(
            view=view,
            scene_objects=scene_objects,
            annotations=annotations,
        )
        self._render_and_postprocess_view(
            view=view,
            output_path=output_path,
            setup_data=setup_data,
            scene_objects=scene_objects,
            annotations=annotations,
            openings=openings,
        )

    def _setup_overlay_rendering(
        self, params: RenderParams, view_size: int | None, margin_scale: float = 1.8
    ) -> OverlayRenderingSetup:
        """Setup scene for metric rendering and return rendering data.

        Args:
            params: Rendering parameters with scene path and camera settings.
            view_size: Optional view size for resolution settings.
            margin_scale: Camera distance margin scale factor. Default 1.8 (80% margin).
                Use 1.30 for 30% margin on side views.

        Returns:
            OverlayRenderingSetup containing camera, bbox, and distance data.
        """
        grid_size = params.width
        console_logger.debug(
            f"Grid size: {grid_size}, Individual view size: {view_size}"
        )

        self._setup_scene(params)
        self._import_and_organize_gltf(params.scene)

        bbox_center, max_dim = compute_scene_bounds(self._client_objects)
        camera_obj = configure_metric_camera(params=params)
        apply_render_settings(params=params, view_size=view_size)
        setup_metric_world()

        # Additional metric-specific settings.
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.film_transparent = True  # Enable alpha channel.
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.render.resolution_percentage = 100

        # EEVEE performance optimization settings.
        # Note: Some settings may not be available in EEVEE_NEXT (Blender 4.5+).
        # We set them conditionally to handle API changes gracefully.
        # TAA samples can be configured via _taa_samples attribute (default 16).
        taa_samples = getattr(self, "_taa_samples", 16)
        try:
            scene.eevee.taa_render_samples = taa_samples
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_gtao = False  # Disable ambient occlusion.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_bloom = False  # Disable bloom.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_ssr = False  # Disable screen space reflections.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_volumetric_shadows = False  # Disable volumetric shadows.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.shadow_cube_size = "128"  # Reduce from default 1024.
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.shadow_cascade_size = "128"
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        try:
            scene.eevee.use_shadows = False
        except AttributeError:
            pass  # Not available in EEVEE_NEXT.

        for light in bpy.data.lights:
            try:
                light.use_shadow = False
            except AttributeError:
                pass  # Not available in EEVEE_NEXT.

        camera_distance = calculate_camera_distance(
            camera_obj=camera_obj, max_dim=max_dim, margin_scale=margin_scale
        )

        return OverlayRenderingSetup(
            camera_obj=camera_obj,
            bbox_center=bbox_center,
            max_dim=max_dim,
            camera_distance=camera_distance,
        )

    def _reset_rendering_state(self) -> None:
        """Reset instance variables to prevent state leakage between renders."""
        self._support_surfaces = None
        self._surface_colors = {}
        self._wall_normals = {}
        self._show_support_surface = False
        self._scene_objects = None
        self._current_convex_hull = None
        self._current_surface_id = None
        self._overlay_mesh_objects = []
        self._surface_mesh_objects = []
        self._hidden_objects = []
        self._debug_camera_objects = []
