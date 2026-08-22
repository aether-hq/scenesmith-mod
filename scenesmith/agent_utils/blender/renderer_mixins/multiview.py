import logging
import math
import time

from pathlib import Path

import bpy

from mathutils import Vector

from scenesmith.agent_utils.blender.geometry.camera_utils import look_at_target
from scenesmith.agent_utils.blender.geometry.scene_utils import disable_backface_culling
from scenesmith.agent_utils.blender.overlays.coordinate_frame import (
    create_coordinate_frame,
)
from scenesmith.agent_utils.blender.overlays.image_overlays import add_number_overlay
from scenesmith.agent_utils.blender.render_settings import setup_cycles_gpu_rendering
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

from scenesmith.agent_utils.blender.renderer_core import (
    _apply_eevee_speed_settings,
    _composite_onto_grey,
)


class MultiviewRenderingMixin:
    """General analysis and CLIP multiview rendering."""

    def render_multiview_for_analysis(
        self,
        mesh_path: Path,
        output_dir: Path,
        elevation_degrees: float,
        num_side_views: int = DEFAULT_NUM_SIDE_VIEWS,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        include_vertical_views: bool = True,
        light_energy: float | None = None,
        start_azimuth_degrees: float = 0.0,
        show_coordinate_frame: bool = True,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> list[Path]:
        """Render a mesh from multiple views for VLM physics analysis.

        This creates renders with optional coordinate frame visualization:
        - Image 0: Top view (+Z) [if include_vertical_views=True]
        - Image 1: Bottom view (-Z) [if include_vertical_views=True]
        - Images 2-(1+num_side_views): Equidistant side views with elevation angle
        - Each image shows RGB coordinate axes (+X=red, +Y=green, +Z=blue)
          [if show_coordinate_frame=True]
        - Each image has a numbered label overlay

        Args:
            mesh_path: Path to the mesh file (GLB/GLTF).
            output_dir: Directory where rendered images will be saved.
            elevation_degrees: Elevation angle in degrees for side view cameras.
                Cameras look down at objects from this angle above horizontal.
                Use 0 for ground-level horizontal views, ~20 for slightly elevated.
            num_side_views: Number of equidistant side views to render.
            width: Width of rendered images in pixels (default: 512).
            height: Height of rendered images in pixels (default: 512).
            include_vertical_views: If True, render top/bottom views. If False,
                only render side views (useful for constraining rotation to Z-axis).
            light_energy: Light energy in watts. If None, uses VLM_ANALYSIS_LIGHT_ENERGY.
            start_azimuth_degrees: Starting azimuth angle for side views (default: 0).
                Use 0 for first view at +X, 90 for first view at +Y. Useful for
                wall-mounted objects where front face is at +Y.
            show_coordinate_frame: If True, show RGB coordinate axes overlay.
                Set to False for cleaner validation renders.

        Returns:
            List of paths to rendered PNG images.
        """
        start_time = time.time()

        num_vertical_views = 2 if include_vertical_views else 0
        total_views = num_side_views + num_vertical_views
        console_logger.info(
            f"Rendering {total_views} views for VLM analysis ({width}x{height}px)"
        )

        # Suppress Blender's verbose rendering output.
        with suppress_stdout_stderr():
            # Clear existing scene.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Set up rendering with EEVEE for speed (6x faster than CYCLES).
            scene = bpy.context.scene
            scene.render.engine = "BLENDER_EEVEE_NEXT"
            scene.eevee.taa_render_samples = taa_samples
            _apply_eevee_speed_settings(scene)
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            # Use transparent background for VLM analysis.
            # This avoids bias toward light/dark objects.
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Add a camera-following point light to ensure visible surfaces
            # are illuminated from all viewing angles.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = (
                light_energy if light_energy is not None else VLM_ANALYSIS_LIGHT_ENERGY
            )
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import mesh.
            bpy.ops.import_scene.gltf(filepath=str(mesh_path))
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            disable_backface_culling(list(bpy.context.selected_objects))

            # Compute bounding box.
            mesh_objs = [
                obj for obj in bpy.context.selected_objects if obj.type == "MESH"
            ]
            bbox_min = Vector((float("inf"),) * 3)
            bbox_max = Vector((float("-inf"),) * 3)
            for obj in mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    bbox_min = Vector(map(min, bbox_min, world_corner))
                    bbox_max = Vector(map(max, bbox_max, world_corner))

            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            max_dim = max(bbox_size)

            # Create coordinate frame with labels (optional).
            if show_coordinate_frame:
                create_coordinate_frame(
                    position=bbox_center,
                    max_dim=max_dim,
                    scale_factor=COORDINATE_FRAME_SCALE_FACTOR,
                    add_labels=True,
                )

            # Setup camera.
            camera = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj
            camera.type = "PERSP"
            camera.lens = DEFAULT_CAMERA_LENS_MM
            camera.sensor_width = DEFAULT_CAMERA_SENSOR_WIDTH_MM
            camera.clip_start = DEFAULT_CAMERA_CLIP_START
            camera.clip_end = DEFAULT_CAMERA_CLIP_END

            # Compute camera distance.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Define views.
            views = []
            if include_vertical_views:
                views.append({"name": "0_top", "direction": Vector((0, 0, 1))})
                views.append({"name": "1_bottom", "direction": Vector((0, 0, -1))})
                side_index_offset = 2
            else:
                side_index_offset = 0

            # Convert elevation to radians for spherical coordinate calculation.
            elevation_rad = math.radians(elevation_degrees)
            cos_elev = math.cos(elevation_rad)
            sin_elev = math.sin(elevation_rad)

            # Convert start azimuth to radians.
            start_azimuth_rad = math.radians(start_azimuth_degrees)

            for i in range(num_side_views):
                azimuth = start_azimuth_rad + 2 * math.pi * i / num_side_views
                # Spherical to Cartesian: camera positioned at elevation, looking at center.
                # x = r * cos(elev) * cos(az)
                # y = r * cos(elev) * sin(az)
                # z = r * sin(elev)
                dir_vec = Vector(
                    (
                        cos_elev * math.cos(azimuth),
                        cos_elev * math.sin(azimuth),
                        sin_elev,
                    )
                )
                views.append(
                    {"name": f"{i + side_index_offset}_side", "direction": dir_vec}
                )

            # Render each view.
            image_paths = []
            output_dir.mkdir(parents=True, exist_ok=True)

            for idx, view in enumerate(views):
                direction = view["direction"].normalized()
                camera_obj.location = bbox_center + direction * camera_distance
                look_at_target(camera_obj, bbox_center)

                # Position light near camera to illuminate the viewed surface.
                light_obj.location = camera_obj.location + direction * (
                    camera_distance * LIGHT_DISTANCE_RATIO
                )

                # Render to file.
                output_path = output_dir / f"{view['name']}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)

                image_paths.append(output_path)

        # Add number overlays (outside suppression - uses PIL, not Blender).
        for idx, output_path in enumerate(image_paths):
            add_number_overlay(output_path, idx)

        console_logger.info(
            f"Rendered {len(image_paths)} views to {output_dir} in "
            f"{time.time()-start_time:.2f}s"
        )
        return image_paths

    def render_multiview_for_clip_embedding(
        self,
        mesh_path: Path,
        output_dir: Path,
        width: int = 224,
        height: int = 224,
        elevation_degrees: float = 30.0,
        light_energy: float | None = None,
    ) -> list[Path]:
        """Render clean multi-view images for CLIP embedding computation.

        Renders 8 views optimized for CLIP image encoding:
        - 4 views at +elevation (upper hemisphere) at 0°, 90°, 180°, 270° azimuth
        - 4 views at -elevation (lower hemisphere) at 0°, 90°, 180°, 270° azimuth

        Unlike render_multiview_for_analysis, this produces clean renders:
        - No coordinate frame overlay
        - No number labels
        - Neutral grey background (works for both light and dark objects)
        - 224x224 default resolution (CLIP's native input size)

        Args:
            mesh_path: Path to the mesh file (GLB/GLTF/OBJ).
            output_dir: Directory where rendered images will be saved.
            width: Image width in pixels (default: 224 for CLIP).
            height: Image height in pixels (default: 224 for CLIP).
            elevation_degrees: Elevation angle in degrees (default: 30).
            light_energy: Light energy in watts. If None, uses VLM_ANALYSIS_LIGHT_ENERGY.

        Returns:
            List of paths to rendered PNG images (8 images).
        """
        start_time = time.time()
        num_views = 8
        console_logger.info(
            f"Rendering {num_views} views for CLIP embedding ({width}x{height}px)"
        )

        with suppress_stdout_stderr():
            # Clear existing scene.
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Set up rendering with CYCLES (offline process, higher quality).
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            setup_cycles_gpu_rendering()
            scene.cycles.samples = CYCLES_CLIP_SAMPLES
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            # Render with transparent background, composite onto neutral grey.
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Black world background (will be transparent due to film_transparent).
            world = bpy.data.worlds.new("ClipWorld")
            scene.world = world
            world.use_nodes = True
            bg_node = world.node_tree.nodes["Background"]
            bg_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

            # Add camera-following point light.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = (
                light_energy if light_energy is not None else VLM_ANALYSIS_LIGHT_ENERGY
            )
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import mesh based on file extension.
            mesh_path_str = str(mesh_path)
            if mesh_path_str.lower().endswith(".obj"):
                bpy.ops.wm.obj_import(filepath=mesh_path_str)
            else:
                bpy.ops.import_scene.gltf(filepath=mesh_path_str)

            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            disable_backface_culling(list(bpy.context.selected_objects))

            # Compute bounding box.
            mesh_objs = [
                obj for obj in bpy.context.selected_objects if obj.type == "MESH"
            ]
            bbox_min = Vector((float("inf"),) * 3)
            bbox_max = Vector((float("-inf"),) * 3)
            for obj in mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    bbox_min = Vector(map(min, bbox_min, world_corner))
                    bbox_max = Vector(map(max, bbox_max, world_corner))

            bbox_center = (bbox_min + bbox_max) / 2
            bbox_size = bbox_max - bbox_min
            max_dim = max(bbox_size)

            # Setup camera.
            camera = bpy.data.cameras.new(name="Camera")
            camera_obj = bpy.data.objects.new("Camera", camera)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj
            camera.type = "PERSP"
            camera.lens = DEFAULT_CAMERA_LENS_MM
            camera.sensor_width = DEFAULT_CAMERA_SENSOR_WIDTH_MM
            camera.clip_start = DEFAULT_CAMERA_CLIP_START
            camera.clip_end = DEFAULT_CAMERA_CLIP_END

            # Compute camera distance.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Define 8 views: 4 upper + 4 lower at cardinal directions.
            elevation_rad = math.radians(elevation_degrees)
            azimuth_angles = [0, 90, 180, 270]  # Cardinal directions in degrees.
            elevations = [elevation_rad, -elevation_rad]  # Upper and lower.

            views = []
            for elev_idx, elev in enumerate(elevations):
                elev_name = "upper" if elev > 0 else "lower"
                cos_elev = math.cos(elev)
                sin_elev = math.sin(elev)

                for az_deg in azimuth_angles:
                    az_rad = math.radians(az_deg)
                    # Spherical to Cartesian: camera looks toward origin.
                    # x = r * cos(elev) * cos(az)
                    # y = r * cos(elev) * sin(az)
                    # z = r * sin(elev)
                    dir_vec = Vector(
                        (
                            cos_elev * math.cos(az_rad),
                            cos_elev * math.sin(az_rad),
                            sin_elev,
                        )
                    )
                    views.append(
                        {
                            "name": f"{elev_name}_az{az_deg}",
                            "direction": dir_vec,
                        }
                    )

            # Render each view.
            image_paths = []
            output_dir.mkdir(parents=True, exist_ok=True)

            for view in views:
                direction = view["direction"].normalized()
                camera_obj.location = bbox_center + direction * camera_distance
                look_at_target(camera_obj, bbox_center)

                # Position light near camera.
                light_obj.location = camera_obj.location + direction * (
                    camera_distance * LIGHT_DISTANCE_RATIO
                )

                # Render to file (RGBA with transparent background).
                output_path = output_dir / f"{view['name']}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)

                image_paths.append(output_path)

        # Composite all images onto neutral grey background.
        for img_path in image_paths:
            _composite_onto_grey(img_path)

        console_logger.info(
            f"Rendered {len(image_paths)} CLIP views to {output_dir} in "
            f"{time.time()-start_time:.2f}s"
        )
        return image_paths
