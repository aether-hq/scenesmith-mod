import logging
import math
import time

from pathlib import Path

import bpy
import numpy as np

from mathutils import Vector
from omegaconf import DictConfig, OmegaConf

from scenesmith.agent_utils.blender.geometry.camera_utils import (
    calculate_camera_distance,
)
from scenesmith.agent_utils.blender.params import RenderParams
from scenesmith.agent_utils.blender.render_dataclasses import OverlayRenderingSetup
from scenesmith.agent_utils.blender.surfaces.surface_utils import (
    generate_multi_surface_views,
    generate_surface_colors,
)
from scenesmith.agent_utils.blender.surfaces.wall_utils import restore_hidden_walls
from scenesmith.agent_utils.scene.house_parts.openings import ClearanceOpeningData
from scenesmith.utils.print_utils import suppress_stdout_stderr

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

from scenesmith.agent_utils.blender.renderer_core import _compute_bounds_from_corners


class AgentObservationRenderingMixin:
    """Annotated agent-observation view orchestration."""

    def render_agent_observation_views(
        self,
        params: RenderParams,
        output_dir: Path,
        layout: str,
        top_view_width: int,
        top_view_height: int,
        side_view_count: int,
        side_view_width: int,
        side_view_height: int,
        scene_objects: list[dict] | None = None,
        annotations: dict | None = None,
        wall_normals: dict[str, list[float]] | None = None,
        support_surfaces: list[dict] | None = None,
        show_support_surface: bool = False,
        current_furniture_id: str | None = None,
        context_furniture_ids: list[str] | None = None,
        render_single_view: dict | None = None,
        openings: list[ClearanceOpeningData] | None = None,
        wall_surfaces: list[dict] | None = None,
        wall_surfaces_for_labels: list[dict] | None = None,
        room_bounds: tuple[float, float, float, float] | None = None,
        ceiling_height: float | None = None,
        side_view_elevation_degrees: float | None = None,
        side_view_start_azimuth_degrees: float | None = None,
        include_vertical_views: bool = True,
    ) -> list[Path]:
        """Render scene views based on layout configuration.

        Each view is rendered individually at native resolution and saved
        directly to output_dir. Supports multiple layout types for ablations.

        For manipuland mode with multiple support surfaces, generates separate
        top views for each surface with filtered coordinate markers and labels.

        For wall rendering mode, generates context top-down view plus per-wall
        orthographic views for wall-mounted object placement.

        For ceiling perspective mode, generates an elevated corner view
        looking down at the ceiling plane with furniture context below.

        Args:
            params: Rendering parameters with scene path and camera settings.
            output_dir: Directory where rendered images will be saved.
            layout: Layout type - "grid_3x3", "single_top", "top_plus_sides",
                "wall_orthographic", "wall", or "ceiling_perspective".
            top_view_width: Width of top-down view in pixels.
            top_view_height: Height of top-down view in pixels.
            side_view_count: Number of side views to render.
            side_view_width: Width of each side view in pixels.
            side_view_height: Height of each side view in pixels.
            scene_objects: Optional scene object metadata for annotations.
            annotations: Optional annotation config flags.
            wall_normals: Pre-computed room-facing normals for walls.
            support_surfaces: Optional list of support surface data. Each surface dict
                contains: surface_id, corners (8 bbox corners), convex_hull_vertices
                (mesh vertices for marker filtering). For multi-surface furniture,
                generates one top view per surface.
            show_support_surface: If True, render green wireframe bbox showing support
                surface bounds for debugging.
            render_single_view: If provided, renders ONLY this single view instead of
                the full layout. Dict with keys: enabled, name, direction (list[float]).
                Used for per-drawer rendering.
            openings: Optional list of ClearanceOpeningData for door/window/open
                labels. Labels are rendered on top views using camera projection.
            wall_surfaces: List of wall surface dicts for wall rendering modes.
                Each dict contains wall_id, direction, length, height, transform,
                and excluded_regions.
            room_bounds: Room XY bounds (min_x, min_y, max_x, max_y) for ceiling mode.
            ceiling_height: Ceiling height in meters for ceiling mode.

        Returns:
            List of paths to rendered PNG files, ordered (top first, then sides).

        Raises:
            ValueError: If layout type is not recognized.
        """
        start_time = time.time()
        console_logger.info(
            f"Rendering {layout} layout with top ({top_view_width}x{top_view_height})"
        )

        # Reset state from previous renders to prevent leakage.
        self._reset_rendering_state()

        # Store wall normals, support surfaces, and debug flags for use during rendering.
        self._wall_normals = wall_normals or {}
        self._show_support_surface = show_support_surface
        self._wall_surfaces_for_labels = wall_surfaces_for_labels

        # Process support surfaces if provided.
        if support_surfaces is not None and len(support_surfaces) > 0:
            # Validate surface data structure.
            for i, surface in enumerate(support_surfaces):
                if "corners" not in surface:
                    raise ValueError(
                        f"Surface {i} missing required key 'corners'. "
                        f"Available keys: {list(surface.keys())}"
                    )
                if "surface_id" not in surface:
                    raise ValueError(
                        f"Surface {i} missing required key 'surface_id'. "
                        f"Available keys: {list(surface.keys())}"
                    )

            self._support_surfaces = support_surfaces

            # Use first surface for camera alignment (backward compatibility).
            # For multi-surface, all surfaces typically share same orientation.
            first_surface = support_surfaces[0]
            corners = first_surface["corners"]

            (
                self._surface_corners,
                self._surface_bounds_min,
                self._surface_bounds_max,
            ) = _compute_bounds_from_corners(corners)

            # Compute furniture rotation angle for camera alignment.
            # Corners are in Drake Z-up coordinates but need to be in Blender Y-up.
            # Apply the same 90° X rotation that's applied to GLTF imports:
            # Drake (x, y, z) → Blender (x, -z, y)

            def drake_to_blender(point):
                """Transform point from Drake Z-up to Blender Y-up coordinates."""
                return np.array([point[0], -point[2], point[1]])

            # Transform corners to Blender space.
            corners_blender = [
                drake_to_blender(corner) for corner in self._surface_corners
            ]

            # Extract furniture axes from corner edges in Blender space.
            edge_x = corners_blender[1] - corners_blender[0]
            edge_y = corners_blender[2] - corners_blender[0]

            # In Blender Y-up, for a top-down view (camera looking along -Y),
            # the camera's +X is "right" and +Z is "up" in the image.
            # Project furniture +X onto the XZ plane to find rotation angle.
            grid_axis_x_xz = np.array([edge_x[0], edge_x[2]])  # Project onto XZ plane
            grid_axis_x_xz_norm = grid_axis_x_xz / np.linalg.norm(grid_axis_x_xz)

            # Compute angle in XZ plane (rotation around Y axis in Blender).
            self._furniture_rotation_z = math.atan2(
                grid_axis_x_xz_norm[1], grid_axis_x_xz_norm[0]
            )

            # Position coordinate frame at corner for visual consistency with
            # furniture floor mode, even though (0,0) is at center.
            self._frame_origin = self._surface_corners[0]

            # Compute bbox axes for coordinate frame alignment.
            # These axes define the furniture's local coordinate system.
            # Compute from Drake-space corners (before Blender transformation).
            edge_x_drake = self._surface_corners[1] - self._surface_corners[0]
            edge_y_drake = self._surface_corners[2] - self._surface_corners[0]
            edge_z_drake = self._surface_corners[4] - self._surface_corners[0]

            extent_x_drake = np.linalg.norm(edge_x_drake)
            extent_y_drake = np.linalg.norm(edge_y_drake)
            extent_z_drake = np.linalg.norm(edge_z_drake)

            axis_x = (
                edge_x_drake / extent_x_drake
                if extent_x_drake > 0
                else np.array([1, 0, 0])
            )
            axis_y = (
                edge_y_drake / extent_y_drake
                if extent_y_drake > 0
                else np.array([0, 1, 0])
            )
            axis_z = (
                edge_z_drake / extent_z_drake
                if extent_z_drake > 0
                else np.array([0, 0, 1])
            )

            # Store as numpy arrays (coordinate_frame.py expects .tolist() to work).
            self._bbox_axis_x = axis_x
            self._bbox_axis_y = axis_y
            self._bbox_axis_z = axis_z
        else:
            self._support_surfaces = None
            self._surface_corners = None
            self._surface_bounds_min = None
            self._surface_bounds_max = None
            self._furniture_rotation_z = None
            self._frame_origin = None

        # Convert annotations dict to OmegaConf for consistent attribute access.
        # This handles the HTTP boundary where OmegaConf → JSON → plain dict.
        if annotations and not isinstance(annotations, DictConfig):
            annotations = OmegaConf.create(annotations)

        # Set default values.
        if scene_objects is None:
            scene_objects = []
        if annotations is None:
            annotations = OmegaConf.create({})

        # Store scene objects for filtering in per-surface rendering.
        self._scene_objects = scene_objects

        # Store current furniture ID for per-surface rendering.
        # In manipuland mode, this is the furniture whose surfaces are being rendered.
        self._current_furniture_id = current_furniture_id

        # Store context furniture IDs for per-surface rendering.
        # These nearby furniture objects should remain visible in top-down views
        # to provide spatial context for item placement orientation.
        self._context_furniture_ids = set(context_furniture_ids or [])

        # Generate surface colors for multi-surface mode.
        self._surface_colors: dict[str, tuple[int, int, int]] = {}
        if self._support_surfaces is not None and len(self._support_surfaces) > 1:
            self._surface_colors = generate_surface_colors(
                surface_ids=[str(s["surface_id"]) for s in self._support_surfaces]
            )
            console_logger.info(
                f"Generated {len(self._surface_colors)} unique colors for surfaces"
            )

        # Generate views based on layout.
        # Check for single view mode (per-drawer rendering).
        if render_single_view is not None and render_single_view.get("enabled", False):
            console_logger.info(
                f"Single view mode: rendering only '{render_single_view.get('name', 'drawer')}'"
            )
            direction = render_single_view.get("direction", [0.0, 0.7, 0.7])
            # If we have support surfaces, attach the first one's data to the view.
            surface_data = None
            if self._support_surfaces and len(self._support_surfaces) > 0:
                surface_data = self._support_surfaces[0]
            views = [
                {
                    "name": render_single_view.get("name", "drawer_view"),
                    "direction": Vector(direction),
                    "is_side": False,
                    "surface_data": surface_data,
                    "is_drawer_view": True,
                }
            ]
        # For multi-surface manipuland mode, generate per-surface top views + side views.
        elif self._support_surfaces is not None and len(self._support_surfaces) > 1:
            console_logger.info(
                f"Multi-surface mode: generating {len(self._support_surfaces)} top views "
                f"+ {side_view_count} side views"
            )
            # Generate per-surface top views.
            top_views = generate_multi_surface_views(
                support_surfaces=self._support_surfaces
            )
            # Generate standard side views for overall context.
            furniture_rotation = (
                self._furniture_rotation_z
                if hasattr(self, "_furniture_rotation_z")
                and self._furniture_rotation_z is not None
                else None
            )
            side_views = self._generate_top_plus_sides_views(
                count=side_view_count,
                furniture_rotation_z=furniture_rotation,
                is_multi_surface_mode=True,
                elevation_degrees=side_view_elevation_degrees,
                start_azimuth_degrees=side_view_start_azimuth_degrees,
                include_vertical_views=include_vertical_views,
            )
            # Combine: side views first, then top views.
            # This ensures original full-scene setup_data is computed and saved before
            # per-surface top views modify it with tight surface bounds.
            # Filter to only side views (skip the single top view from top_plus_sides).
            side_views_only = [v for v in side_views if v.get("is_side", False)]
            views = side_views_only + top_views
        elif layout == "grid_3x3":
            views = self._generate_grid_3x3_views()
        elif layout == "single_top":
            views = self._generate_single_top_view()
        elif layout == "top_plus_sides":
            # Pass furniture rotation for manipuland mode to align side views.
            furniture_rotation = (
                self._furniture_rotation_z
                if hasattr(self, "_furniture_rotation_z")
                and self._furniture_rotation_z is not None
                else None
            )
            views = self._generate_top_plus_sides_views(
                count=side_view_count,
                furniture_rotation_z=furniture_rotation,
                is_multi_surface_mode=False,
                elevation_degrees=side_view_elevation_degrees,
                start_azimuth_degrees=side_view_start_azimuth_degrees,
                include_vertical_views=include_vertical_views,
            )
        elif layout == "wall_orthographic":
            # Per-wall orthographic view with grid overlay.
            views = self._generate_wall_orthographic_view(wall_surfaces=wall_surfaces)
        elif layout == "wall":
            # Context-only view for wall mode.
            # Per-wall orthographic views are rendered separately via
            # render_per_wall_ortho_views with filtered furniture per wall.
            views = self._generate_wall_context_views()
        elif layout == "ceiling_perspective":
            # Elevated perspective view for ceiling observation.
            if room_bounds is None:
                raise ValueError("ceiling_perspective layout requires room_bounds")
            if ceiling_height is None:
                raise ValueError("ceiling_perspective layout requires ceiling_height")
            views = self._generate_ceiling_perspective_view(
                room_bounds=room_bounds,
                ceiling_height=ceiling_height,
            )
        else:
            raise ValueError(
                f"Unknown layout '{layout}'. "
                f"Options: grid_3x3, single_top, top_plus_sides, wall_orthographic, "
                f"wall, ceiling_perspective"
            )

        # Inject ceiling_height into all views for coordinate grid rendering.
        # This ensures the grid is drawn at ceiling level instead of floor.
        if ceiling_height is not None:
            for view in views:
                view["ceiling_height"] = ceiling_height

        # Create output directory.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Render each view.
        image_paths = []
        setup_data = None
        original_setup_data = None  # Store full-scene setup for side views.
        with suppress_stdout_stderr():
            for view in views:
                is_top_view = view["name"].endswith("_top") or "_top_" in view["name"]
                width = top_view_width if is_top_view else side_view_width
                height = top_view_height if is_top_view else side_view_height

                # For multi-surface views, update current surface data before rendering.
                is_per_surface_top_view = "surface_data" in view and is_top_view
                is_per_surface_side_view = "surface_data" in view and not is_top_view
                if "surface_data" in view:
                    surface_data = view["surface_data"]
                    # Update _surface_corners and bounds for this specific surface.
                    (
                        self._surface_corners,
                        self._surface_bounds_min,
                        self._surface_bounds_max,
                    ) = _compute_bounds_from_corners(surface_data["corners"])
                    # Store convex hull for coordinate marker filtering.
                    self._current_convex_hull = surface_data.get(
                        "convex_hull_vertices", None
                    )
                    self._current_surface_id = surface_data.get("surface_id", "unknown")

                    # For per-surface top views, set coordinate frame at surface corner.
                    if is_per_surface_top_view:
                        # Compute surface local coordinate frame.
                        # Use corner 0 as origin (min x, min y, min z).
                        corners = np.array(surface_data["corners"])
                        origin = corners[0]  # Corner at (min_x, min_y, min_z).

                        # Compute local axes from edges.
                        edge_x = corners[1] - corners[0]  # X axis direction.
                        edge_y = corners[2] - corners[0]  # Y axis direction.
                        edge_z = corners[4] - corners[0]  # Z axis direction.

                        # Normalize to get unit axes.
                        axis_x = edge_x / np.linalg.norm(edge_x)
                        axis_y = edge_y / np.linalg.norm(edge_y)
                        axis_z = edge_z / np.linalg.norm(edge_z)

                        # Store for coordinate frame rendering (keep as numpy arrays).
                        self._frame_origin = origin
                        self._bbox_axis_x = axis_x
                        self._bbox_axis_y = axis_y
                        self._bbox_axis_z = axis_z

                        # Clean up previous surface mesh first.
                        if (
                            hasattr(self, "_surface_mesh_objects")
                            and self._surface_mesh_objects
                        ):
                            self._cleanup_surface_meshes()
                        self._setup_per_surface_rendering(surface_data)
                else:
                    self._current_convex_hull = None
                    self._current_surface_id = None

                    # Restore object visibility for side views.
                    if (
                        hasattr(self, "_surface_mesh_objects")
                        and self._surface_mesh_objects
                    ):
                        self._cleanup_surface_meshes()
                        self._restore_object_visibility()

                # Setup scene on first iteration only.
                # Resolution is set per-view below, so initial value doesn't matter.
                if setup_data is None:
                    # For side views, use tighter margin (30%) to show full scene.
                    # For top views, use default margin (80%) with extra coordinate space.
                    is_side_view = view.get("is_side", False)
                    margin_scale = 1.40 if is_side_view else 1.8
                    setup_data = self._setup_overlay_rendering(
                        params=params, view_size=None, margin_scale=margin_scale
                    )
                    # Save original full-scene setup for side views ONLY if this is not
                    # a per-surface view. In multi-surface mode, side views come first and
                    # this will save the full-scene bounds with tight 5% margin.
                    if not is_per_surface_top_view:
                        original_setup_data = setup_data

                    # Create overlays AFTER scene and GLTF have been imported as
                    # reset_scene() deletes all objects. Must create overlays after GLTF
                    # import so they persist for rendering.
                    if (
                        self._support_surfaces is not None
                        and len(self._support_surfaces) > 1
                        and self._surface_colors
                        and not self._overlay_mesh_objects
                    ):
                        console_logger.info(
                            f"Creating {len(self._support_surfaces)} overlay meshes "
                            "for multi-surface mode (after scene setup)"
                        )
                        self._overlay_mesh_objects = []
                        for surface_data in self._support_surfaces:
                            surface_id = surface_data.get("surface_id", "unknown")
                            if surface_id in self._surface_colors:
                                overlay_color = self._surface_colors[surface_id]
                                overlay_obj = self._create_surface_overlay_mesh(
                                    surface_data=surface_data, color=overlay_color
                                )
                                if overlay_obj is not None:
                                    self._overlay_mesh_objects.append(overlay_obj)
                        console_logger.info(
                            f"Created {len(self._overlay_mesh_objects)} overlay meshes"
                        )
                else:
                    # Restore walls hidden in previous view.
                    restore_hidden_walls()

                # For per-surface top views, recompute camera setup after hiding objects.
                if is_per_surface_top_view:
                    console_logger.info(
                        "Recomputing camera setup for per-surface view "
                        f"(surface {self._current_surface_id})"
                    )
                    # Compute bounds from surface corners directly (not entire scene).
                    corners_array = np.array(surface_data["corners"])
                    bbox_min = corners_array.min(axis=0)
                    bbox_max = corners_array.max(axis=0)
                    bbox_center = Vector((bbox_min + bbox_max) / 2)
                    bbox_size = bbox_max - bbox_min
                    max_dim = max(bbox_size)

                    # Add margin for manipulands extending beyond surface bounds.
                    max_dim *= 1.1  # 10% margin.

                    camera_distance = calculate_camera_distance(
                        camera_obj=setup_data.camera_obj,
                        max_dim=max_dim,
                        margin_scale=1.1,  # 10% camera distance margin.
                    )
                    # Save original setup before modifying (for side views).
                    if original_setup_data is None:
                        original_setup_data = setup_data

                    # Update setup_data with new bounds.
                    setup_data = OverlayRenderingSetup(
                        camera_obj=setup_data.camera_obj,
                        bbox_center=bbox_center,
                        max_dim=max_dim,
                        camera_distance=camera_distance,
                    )
                elif is_per_surface_side_view:
                    # For side views, restore original full-scene setup.
                    # Side views should show full furniture, not just the surface.
                    if original_setup_data is not None:
                        setup_data = original_setup_data

                # Update render resolution for this view.
                scene = bpy.context.scene
                scene.render.resolution_x = width
                scene.render.resolution_y = height

                # Render view with metric overlays and annotations.
                output_path = output_dir / f"{view['name']}.png"
                self._render_single_view_with_metric_overlay_to_path(
                    view=view,
                    setup_data=setup_data,
                    output_path=output_path,
                    scene_objects=scene_objects,
                    annotations=annotations,
                    openings=openings,
                )
                image_paths.append(output_path)

        # Clean up temporary surface meshes and overlays.
        self._cleanup_surface_meshes()
        self._cleanup_overlay_meshes()

        console_logger.info(
            f"Rendered {len(image_paths)} views to {output_dir} in "
            f"{time.time() - start_time:.2f}s"
        )
        return image_paths
