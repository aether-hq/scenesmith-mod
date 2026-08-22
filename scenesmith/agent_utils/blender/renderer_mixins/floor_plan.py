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

from scenesmith.agent_utils.blender.renderer_core import _apply_eevee_speed_settings


class FloorPlanRenderingMixin:
    """Floor-plan and object-directory rendering."""

    def render_floor_plan(
        self,
        mesh_path: Path,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        light_energy: float | None = None,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> Path:
        """Render a clean top-down view of a floor plan without coordinate frame.

        This produces a clean render suitable for floor plan visualization:
        - Single top-down view (looking down -Z axis)
        - No coordinate frame overlay
        - No number labels
        - Transparent background
        - Bright, even lighting to clearly show materials

        Args:
            mesh_path: Path to the floor plan GLB/GLTF file.
            output_path: Path where the rendered PNG will be saved.
            width: Image width in pixels (default: 1024).
            height: Image height in pixels (default: 1024).
            light_energy: Sun light energy. If None, uses default (5.0).

        Returns:
            Path to the rendered PNG image.
        """
        start_time = time.time()
        console_logger.info(f"Rendering floor plan top view ({width}x{height}px)")

        # Default sun energy for floor plans (brighter than typical scene).
        default_sun_energy = 5.0

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
            # Transparent background for compositing.
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Set up world with ambient lighting (adds fill light to shadows).
            scene.world = bpy.data.worlds.new("FloorPlanWorld")
            scene.world.use_nodes = True
            bg_node = scene.world.node_tree.nodes.get("Background")
            if bg_node:
                # Light gray ambient fill (0.3) - brightens shadows significantly.
                bg_node.inputs[0].default_value = (0.3, 0.3, 0.3, 1.0)
                bg_node.inputs[1].default_value = 1.0  # Strength.

            # Use SUN light for even illumination (no distance falloff).
            light = bpy.data.lights.new(name="Sun", type="SUN")
            light.energy = (
                light_energy if light_energy is not None else default_sun_energy
            )
            # Slightly warm color for natural look.
            light.color = (1.0, 0.98, 0.95)
            light_obj = bpy.data.objects.new("Sun", light)
            scene.collection.objects.link(light_obj)
            # Angle sun slightly (15 degrees from vertical) for some shadow definition.
            light_obj.rotation_euler = (math.radians(15), math.radians(15), 0)

            # Import mesh.
            bpy.ops.import_scene.gltf(filepath=str(mesh_path))
            bpy.ops.object.select_all(action="SELECT")

            # Apply rotation to counteract glTF import rotation.
            bpy.ops.transform.rotate(
                value=math.pi / 2,
                orient_axis="X",
                orient_type="GLOBAL",
                center_override=(0, 0, 0),
            )

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

            # Top-down view (looking down -Z axis).
            direction = Vector((0, 0, 1))  # Camera above, looking down.
            camera_obj.location = bbox_center + direction * camera_distance
            look_at_target(camera_obj, bbox_center)

            # SUN light doesn't need positioning - it's directional.

            # Render to file.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(output_path)
            bpy.ops.render.render(write_still=True)

        # Keep transparent background (RGBA) for floor plan renders.

        console_logger.info(
            f"Rendered floor plan to {output_path} in {time.time()-start_time:.2f}s"
        )
        return output_path

    def render_multiview_from_obj_directory(
        self,
        obj_directory: Path,
        output_dir: Path,
        num_side_views: int = DEFAULT_NUM_SIDE_VIEWS,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        include_vertical_views: bool = True,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> list[Path]:
        """Render multi-view images from a directory of OBJ files.

        This is designed for PartNet-Mobility assets which store each link's
        geometry as separate OBJ files. All OBJ files are loaded and rendered
        together as a single articulated object.

        The rendering setup matches render_multiview_for_analysis:
        - Transparent background
        - Coordinate frame visualization (+X=red, +Y=green, +Z=blue)
        - Numbered view labels
        - Camera-following point light

        Args:
            obj_directory: Path to directory containing OBJ files. All .obj
                files in this directory will be loaded.
            output_dir: Directory where rendered images will be saved.
            num_side_views: Number of equidistant side views to render.
            width: Width of rendered images in pixels (default: 512).
            height: Height of rendered images in pixels (default: 512).
            include_vertical_views: If True, render top/bottom views. If False,
                only render side views.

        Returns:
            List of paths to rendered PNG images.

        Raises:
            FileNotFoundError: If obj_directory does not exist.
            ValueError: If no OBJ files are found in the directory.
        """
        start_time = time.time()

        if not obj_directory.exists():
            raise FileNotFoundError(f"OBJ directory not found: {obj_directory}")

        # Find all OBJ files in the directory.
        obj_files = sorted(obj_directory.glob("*.obj"))
        if not obj_files:
            raise ValueError(f"No OBJ files found in {obj_directory}")

        num_vertical_views = 2 if include_vertical_views else 0
        total_views = num_side_views + num_vertical_views
        console_logger.info(
            f"Rendering {total_views} views from {len(obj_files)} OBJ files "
            f"({width}x{height}px)"
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
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Add a camera-following point light.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = VLM_ANALYSIS_LIGHT_ENERGY
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import all OBJ files.
            for obj_file in obj_files:
                bpy.ops.wm.obj_import(filepath=str(obj_file))

            # Select all mesh objects and apply transforms.
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            disable_backface_culling(list(bpy.context.scene.objects))

            # Compute combined bounding box.
            mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
            if not mesh_objs:
                raise ValueError(
                    f"No mesh objects after importing from {obj_directory}"
                )

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

            # Create coordinate frame with labels.
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

            for i in range(num_side_views):
                angle = 2 * math.pi * i / num_side_views
                dir_vec = Vector((math.cos(angle), math.sin(angle), 0))
                views.append(
                    {"name": f"{i + side_index_offset}_side", "direction": dir_vec}
                )

            # Render each view.
            image_paths = []
            output_dir.mkdir(parents=True, exist_ok=True)

            for view in views:
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
