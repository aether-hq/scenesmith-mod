import logging

import bpy
import numpy as np

from mathutils import Vector
from omegaconf import DictConfig

from scenesmith.agent_utils.blender.geometry.camera_utils import look_at_target
from scenesmith.agent_utils.blender.geometry.scene_utils import get_floor_bounds
from scenesmith.agent_utils.blender.overlays.coordinate_frame import (
    add_coordinate_frame_top_view,
    add_coordinate_frame_wall_view,
)
from scenesmith.agent_utils.blender.overlays.image_overlays import (
    add_support_surface_debug_volume,
)
from scenesmith.agent_utils.blender.overlays.scene_annotations import (
    add_blender_scene_annotations,
)
from scenesmith.agent_utils.blender.render_dataclasses import OverlayRenderingSetup
from scenesmith.agent_utils.blender.surfaces.wall_utils import (
    looks_like_wall,
    should_hide_wall,
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

from scenesmith.agent_utils.blender.renderer_core import (
    _compute_wall_center_from_transform,
)


class CameraSetupRenderingMixin:
    """Camera, clipping, coordinate-frame, and annotation setup."""

    def _setup_camera_and_coordinate_frame(
        self,
        setup_data: OverlayRenderingSetup,
        view: dict[str, str | Vector | bool],
        annotations: DictConfig | None = None,
    ) -> None:
        """Position camera and add coordinate frame overlay for a view.

        Args:
            setup_data: Metric rendering setup data.
            view: Dictionary containing view information.
            annotations: Optional annotation config with rendering_mode.
        """
        # Reset camera clip_start to default before each view.
        # This ensures lower surface clipping from previous views doesn't persist.
        setup_data.camera_obj.data.clip_start = DEFAULT_CAMERA_CLIP_START

        # Position camera.
        direction = view["direction"].normalized()
        is_side = view.get("is_side", True)
        is_orthographic = view.get("is_orthographic", False)
        is_wall_orthographic = view.get("is_wall_orthographic", False)

        # Get rendering mode from annotations.
        rendering_mode = "furniture"
        if annotations and hasattr(annotations, "rendering_mode"):
            rendering_mode = annotations.rendering_mode

        # Handle orthographic camera for wall views.
        if is_orthographic and is_wall_orthographic:
            wall_surface = view.get("wall_surface", {})
            self._setup_wall_orthographic_camera(
                camera_obj=setup_data.camera_obj,
                wall_surface=wall_surface,
            )

            # Add coordinate frame for wall orthographic view.
            # Extract wall parameters for frame positioning.
            wall_length = wall_surface.get("length", 4.0)
            wall_height = wall_surface.get("height", 2.5)
            wall_direction = wall_surface.get("direction", "north")
            transform = wall_surface.get("transform", [0, 0, 0, 1, 0, 0, 0])

            # Calculate wall center from transform.
            wall_center = _compute_wall_center_from_transform(
                transform=transform, wall_length=wall_length, wall_height=wall_height
            )

            add_coordinate_frame_wall_view(
                wall_center=wall_center,
                wall_length=wall_length,
                wall_height=wall_height,
                wall_direction=wall_direction,
            )

            return  # Skip standard camera positioning for wall orthographic views.

        # Zoom in more for top view to reduce black borders.
        # Use less aggressive zoom for manipuland/ceiling mode to avoid cutting off edges.
        camera_distance = setup_data.camera_distance
        if not is_side:
            if rendering_mode == "manipuland":
                camera_distance *= 0.85  # Less aggressive zoom for manipuland mode.
            elif rendering_mode == "ceiling_perspective":
                camera_distance *= 0.8  # Moderate zoom for ceiling to show full grid.
            else:
                camera_distance *= 0.7  # More aggressive zoom for furniture mode.
        setup_data.camera_obj.location = (
            setup_data.bbox_center + direction * camera_distance
        )
        look_at_target(obj=setup_data.camera_obj, target=setup_data.bbox_center)

        # For manipuland mode top views, align camera with furniture orientation.
        if (
            not is_side
            and rendering_mode == "manipuland"
            and hasattr(self, "_furniture_rotation_z")
        ):
            # Apply roll rotation around viewing direction (Z-axis) to align camera
            # with furniture axes, making rotated furniture appear axis-aligned.
            setup_data.camera_obj.rotation_euler.rotate_axis(
                "Z", self._furniture_rotation_z
            )

        # Apply camera near-plane clipping for lower surfaces in per-surface top views.
        # This clips furniture geometry above the current surface so lower surfaces
        # (e.g., shelves under a table top) are visible from top-down views.
        surface_data = view.get("surface_data")
        is_per_surface_top_view = surface_data is not None and not is_side
        if (
            is_per_surface_top_view
            and rendering_mode == "manipuland"
            and self._support_surfaces is not None
            and len(self._support_surfaces) > 1
        ):
            self._apply_lower_surface_clipping(
                camera_obj=setup_data.camera_obj,
                surface_data=surface_data,
                camera_distance=camera_distance,
            )

        # Add support surface debug volumes for manipuland mode.
        if rendering_mode == "manipuland" and self._show_support_surface:
            if self._support_surfaces is not None and len(self._support_surfaces) > 1:
                # Multi-surface mode: draw colored volume for ALL surfaces.
                for surface in self._support_surfaces:
                    surface_id = surface.get("surface_id", "unknown")
                    corners = np.array(surface["corners"])
                    # Get surface color (RGB 0-255) and convert to Blender (RGBA 0-1).
                    if surface_id in self._surface_colors:
                        rgb = self._surface_colors[surface_id]
                        color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0)
                    else:
                        color = (0.0, 1.0, 0.0, 1.0)  # Fallback green.
                    add_support_surface_debug_volume(corners=corners, color=color)
            elif self._surface_corners is not None:
                # Single-surface mode: draw green volume for the one surface.
                add_support_surface_debug_volume(corners=self._surface_corners)

        # Add coordinate frame overlay.
        from mathutils import Vector

        floor_bounds = get_floor_bounds(self._client_objects)
        frame_origin = self._frame_origin if hasattr(self, "_frame_origin") else None
        # Convert numpy arrays to Vectors for coordinate frame functions.
        bbox_axis_x = (
            Vector(self._bbox_axis_x.tolist())
            if hasattr(self, "_bbox_axis_x") and self._bbox_axis_x is not None
            else None
        )
        bbox_axis_y = (
            Vector(self._bbox_axis_y.tolist())
            if hasattr(self, "_bbox_axis_y") and self._bbox_axis_y is not None
            else None
        )
        bbox_axis_z = (
            Vector(self._bbox_axis_z.tolist())
            if hasattr(self, "_bbox_axis_z") and self._bbox_axis_z is not None
            else None
        )

        # Only add coordinate frame for furniture/manipuland top views.
        # Skip for wall_context, wall_orthographic views, and furniture_selection mode.
        is_wall_context = view.get("is_wall_context", False)
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        show_coord_frame = getattr(annotations, "show_coordinate_frame", True)
        if (
            not is_side
            and not is_wall_context
            and not is_wall_orthographic
            and show_coord_frame
        ):
            add_coordinate_frame_top_view(
                bbox_center=setup_data.bbox_center,
                max_dim=setup_data.max_dim,
                floor_bounds=floor_bounds,
                rendering_mode=rendering_mode,
                frame_origin=frame_origin,
                bbox_axis_x=bbox_axis_x,
                bbox_axis_y=bbox_axis_y,
                bbox_axis_z=bbox_axis_z,
            )

    def _apply_lower_surface_clipping(
        self,
        camera_obj: bpy.types.Object,
        surface_data: dict,
        camera_distance: float,
    ) -> None:
        """Apply camera near-plane clipping for lower support surfaces.

        When rendering a top-down view of a surface that is NOT the highest surface,
        furniture geometry above the surface blocks the view. This method clips that
        geometry by adjusting the camera's near clipping plane.

        For a top-down camera at height camera_z looking down (direction -Z):
        - Near clip at distance d clips everything at z > camera_z - d
        - To clip at surface_z + offset: clip_start = camera_z - (surface_z + offset)

        Args:
            camera_obj: Blender camera object to modify.
            surface_data: Dictionary containing current surface info with 'corners' key.
            camera_distance: Distance from camera to bbox_center along view direction.
        """
        if self._support_surfaces is None or len(self._support_surfaces) <= 1:
            return

        # Get current surface's Z range from corners.
        current_corners = np.array(surface_data.get("corners", []))
        if current_corners.size == 0:
            return
        current_z_max = current_corners[:, 2].max()

        # Find the highest surface's Z max among all surfaces.
        highest_z_max = current_z_max
        for surface in self._support_surfaces:
            corners = np.array(surface.get("corners", []))
            if corners.size > 0:
                highest_z_max = max(highest_z_max, corners[:, 2].max())

        # Only apply clipping if current surface is NOT the highest surface.
        z_tolerance = 0.01  # 1cm tolerance for floating point comparison.
        if current_z_max >= highest_z_max - z_tolerance:
            # Current surface is the highest (or tied for highest), no clipping needed.
            console_logger.debug(
                f"Surface at z={current_z_max:.3f} is highest, no clipping needed"
            )
            return

        # Calculate clip height: just above the current surface.
        clip_z = current_z_max + LOWER_SURFACE_CLIP_OFFSET_M

        # Camera is at bbox_center + direction * camera_distance.
        # For top-down view, direction is (0, 0, 1), so camera_z = bbox_center_z + camera_distance.
        # bbox_center_z is approximately current_z_max (center of surface bounding box).
        # Using surface_z_max as approximation for camera target height.
        camera_z = current_z_max + camera_distance

        # Calculate clip_start to clip at clip_z.
        # Near plane clips objects at z > camera_z - clip_start.
        clip_start = camera_z - clip_z

        # Ensure clip_start is positive and reasonable.
        if clip_start < DEFAULT_CAMERA_CLIP_START:
            console_logger.warning(
                f"Calculated clip_start={clip_start:.3f} too small, using default"
            )
            clip_start = DEFAULT_CAMERA_CLIP_START

        # Apply clipping to camera.
        camera_obj.data.clip_start = clip_start
        console_logger.info(
            f"Applied lower surface clipping: clip_start={clip_start:.3f}m "
            f"(clips at z>{clip_z:.3f}m, surface z_max={current_z_max:.3f}m)"
        )

    def _apply_view_annotations(
        self,
        view: dict[str, str | Vector | bool],
        scene_objects: list[dict] | None,
        annotations: DictConfig,
    ) -> None:
        """Apply wall hiding and Blender 3D annotations for a view.

        Args:
            view: Dictionary containing view information.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Annotation config flags.
        """
        # Apply partial wall hiding if enabled.
        if annotations.enable_partial_walls:
            direction = view["direction"].normalized()
            is_top_view = not view.get("is_side", True)
            all_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
            wall_objects = [obj for obj in all_meshes if looks_like_wall(obj)]

            # Hide walls that should be hidden to not occlude the view.
            # Note: 'direction' is from center to camera, so we need to invert it
            # to get the camera viewing direction (from camera to center).
            camera_viewing_dir = -direction

            walls_hidden = 0
            for obj in wall_objects:
                should_hide = should_hide_wall(
                    obj=obj,
                    camera_direction=camera_viewing_dir,
                    is_top_view=is_top_view,
                    wall_normals=self._wall_normals,
                )

                if should_hide:
                    obj.hide_render = True
                    obj.hide_viewport = True
                    walls_hidden += 1

            # Force view layer update after visibility changes.
            if walls_hidden > 0:
                bpy.context.view_layer.update()

        # Add Blender 3D annotation objects before rendering (only for top views).
        # Skip for wall_orthographic views - they use PIL grid annotations instead.
        is_top_view = not view.get("is_side", True)
        is_wall_orthographic = view.get("is_wall_orthographic", False)
        annotations_enabled = any(
            [
                annotations.enable_set_of_mark_labels,
                annotations.enable_bounding_boxes,
                annotations.enable_direction_arrows,
            ]
        )
        console_logger.info(
            f"Annotation check: is_top_view={is_top_view}, "
            f"is_wall_orthographic={is_wall_orthographic}, "
            f"scene_objects={len(scene_objects) if scene_objects else 0}, "
            f"annotations_enabled={annotations_enabled}, annotations={annotations}"
        )
        # Skip Blender 3D annotations for wall orthographic views.
        if (
            is_top_view
            and scene_objects
            and annotations_enabled
            and not is_wall_orthographic
        ):
            # Filter scene_objects by current surface if in per-surface mode.
            filtered_scene_objects = scene_objects
            if hasattr(self, "_current_surface_id") and self._current_surface_id:
                console_logger.info(
                    f"Filtering scene_objects for surface {self._current_surface_id}"
                )
                filtered_scene_objects = self._filter_objects_by_surface(
                    scene_objects=scene_objects,
                    current_surface_id=self._current_surface_id,
                )
                console_logger.info(
                    f"Filtered from {len(scene_objects)} to "
                    f"{len(filtered_scene_objects)} objects"
                )

            try:
                add_blender_scene_annotations(
                    scene_objects=filtered_scene_objects, annotations=annotations
                )
                console_logger.info("Successfully added Blender annotations")
            except Exception as e:
                console_logger.error(
                    f"Failed to add Blender annotations: {e}", exc_info=True
                )
        else:
            console_logger.info(
                f"Skipping annotations: is_top_view={is_top_view}, "
                f"scene_objects_count={len(scene_objects) if scene_objects else 0}, "
                f"annotations_enabled={annotations_enabled}"
            )
