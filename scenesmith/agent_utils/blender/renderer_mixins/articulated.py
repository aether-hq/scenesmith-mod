import logging
import math
import time

from pathlib import Path

import bpy

from mathutils import Matrix, Vector

from scenesmith.agent_utils.blender.geometry.camera_utils import look_at_target
from scenesmith.agent_utils.blender.geometry.scene_utils import disable_backface_culling
from scenesmith.agent_utils.blender.overlays.coordinate_frame import (
    create_coordinate_frame,
)
from scenesmith.agent_utils.blender.overlays.image_overlays import add_number_overlay
from scenesmith.agent_utils.blender.render_dataclasses import (
    ArticulatedRenderResult,
    LinkMeshInfo,
)
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


class ArticulatedRenderingMixin:
    """Articulated-object multiview rendering."""

    def render_multiview_articulated(
        self,
        link_meshes: list[LinkMeshInfo],
        output_dir: Path,
        num_combined_side_views: int = DEFAULT_NUM_SIDE_VIEWS,
        num_link_side_views: int = 4,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        light_energy: float | None = None,
        taa_samples: int = EEVEE_ASSET_VALIDATION_SAMPLES,
    ) -> ArticulatedRenderResult:
        """Render multi-view images for an articulated object with per-link views.

        This renders:
        1. Combined views showing all links together (combined_0.png, etc.)
        2. Per-link views showing each link in isolation (link_name_0.png, etc.)

        The combined views use the same setup as render_multiview_from_obj_directory.
        Per-link views use fewer angles since they're supplementary.

        Args:
            link_meshes: List of LinkMeshInfo with link names and OBJ paths.
            output_dir: Directory where rendered images will be saved.
            num_combined_side_views: Number of side views for combined render.
            num_link_side_views: Number of side views for each link (fewer than
                combined since they're supplementary).
            width: Width of rendered images in pixels.
            height: Height of rendered images in pixels.
            light_energy: Light energy in watts. If None, uses VLM_ANALYSIS_LIGHT_ENERGY.

        Returns:
            ArticulatedRenderResult with paths to all images and dimensions.

        Raises:
            ValueError: If no valid meshes are found.
        """
        start_time = time.time()

        # Collect all mesh files across all links.
        all_mesh_files = []
        for link_info in link_meshes:
            for mesh_path in link_info.mesh_paths:
                if mesh_path.exists():
                    all_mesh_files.append(mesh_path)

        if not all_mesh_files:
            raise ValueError("No valid mesh files found in link meshes")

        output_dir.mkdir(parents=True, exist_ok=True)
        combined_image_paths = []
        link_image_paths: dict[str, list[Path]] = {}
        link_dimensions: dict[str, tuple[float, float, float]] = {}

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
            scene.render.film_transparent = True
            scene.render.image_settings.color_mode = "RGBA"

            # Add a camera-following point light.
            light = bpy.data.lights.new(name="Light", type="POINT")
            light.energy = (
                light_energy if light_energy is not None else VLM_ANALYSIS_LIGHT_ENERGY
            )
            light_obj = bpy.data.objects.new("Light", light)
            scene.collection.objects.link(light_obj)

            # Import all mesh files and track which objects belong to which link.
            link_to_objects: dict[str, list[bpy.types.Object]] = {}

            for link_info in link_meshes:
                link_to_objects[link_info.link_name] = []

                # Build world transform matrix for this link.
                # Transform order: visual_origin -> link_world_transform.
                world_pos = link_info.world_position
                world_rot = link_info.world_rotation

                # Create world rotation matrix (identity if not provided).
                world_rot_matrix = (
                    Matrix(
                        (
                            (world_rot[0][0], world_rot[0][1], world_rot[0][2], 0),
                            (world_rot[1][0], world_rot[1][1], world_rot[1][2], 0),
                            (world_rot[2][0], world_rot[2][1], world_rot[2][2], 0),
                            (0, 0, 0, 1),
                        )
                    )
                    if world_rot is not None
                    else Matrix.Identity(4)
                )

                for mesh_path, origin in zip(
                    link_info.mesh_paths, link_info.origins, strict=True
                ):
                    if not mesh_path.exists():
                        continue

                    # Record objects before import.
                    objects_before = set(bpy.context.scene.objects)

                    # Import based on file extension.
                    ext = mesh_path.suffix.lower()
                    if ext == ".obj":
                        bpy.ops.wm.obj_import(filepath=str(mesh_path))
                    elif ext in {".gltf", ".glb"}:
                        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
                    else:
                        console_logger.warning(f"Unsupported mesh format: {ext}")
                        continue

                    # Find newly imported objects.
                    objects_after = set(bpy.context.scene.objects)
                    new_objects = objects_after - objects_before

                    for obj in new_objects:
                        if obj.type == "MESH":
                            # Apply visual origin offset first (in link's local frame).
                            obj.location.x += origin[0]
                            obj.location.y += origin[1]
                            obj.location.z += origin[2]

                            # Then apply link's world transform.
                            # First apply rotation around origin, then translate.
                            if world_rot is not None:
                                # Get current location as vector.
                                local_pos = Vector(obj.location)
                                # Rotate position by world rotation.
                                rotated_pos = world_rot_matrix @ Vector(
                                    (local_pos.x, local_pos.y, local_pos.z, 1)
                                )
                                obj.location = Vector(
                                    (rotated_pos.x, rotated_pos.y, rotated_pos.z)
                                )
                                # Apply rotation to object orientation.
                                obj.matrix_world = world_rot_matrix @ obj.matrix_world
                                # Reset location after matrix multiply.
                                obj.location = Vector(
                                    (rotated_pos.x, rotated_pos.y, rotated_pos.z)
                                )

                            # Apply world position offset.
                            obj.location.x += world_pos[0]
                            obj.location.y += world_pos[1]
                            obj.location.z += world_pos[2]

                            link_to_objects[link_info.link_name].append(obj)

            # Apply transforms to all mesh objects.
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Disable backface culling for all imported materials.
            # This ensures meshes render correctly from both sides, fixing issues
            # with single-sided meshes (common in PartNet-Mobility models).
            disable_backface_culling(list(bpy.context.scene.objects))

            # Compute combined bounding box of all mesh objects.
            all_mesh_objs = [
                obj for obj in bpy.context.scene.objects if obj.type == "MESH"
            ]
            if not all_mesh_objs:
                raise ValueError("No mesh objects after importing")

            combined_bbox_min = Vector((float("inf"),) * 3)
            combined_bbox_max = Vector((float("-inf"),) * 3)
            for obj in all_mesh_objs:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    combined_bbox_min = Vector(
                        map(min, combined_bbox_min, world_corner)
                    )
                    combined_bbox_max = Vector(
                        map(max, combined_bbox_max, world_corner)
                    )

            combined_bbox_center = (combined_bbox_min + combined_bbox_max) / 2
            combined_bbox_size = combined_bbox_max - combined_bbox_min
            combined_max_dim = max(combined_bbox_size)
            combined_dimensions = (
                combined_bbox_size.x,
                combined_bbox_size.y,
                combined_bbox_size.z,
            )

            # Compute per-link bounding boxes.
            for link_name, link_objs in link_to_objects.items():
                if not link_objs:
                    link_dimensions[link_name] = (0.0, 0.0, 0.0)
                    continue

                link_bbox_min = Vector((float("inf"),) * 3)
                link_bbox_max = Vector((float("-inf"),) * 3)
                for obj in link_objs:
                    for corner in obj.bound_box:
                        world_corner = obj.matrix_world @ Vector(corner)
                        link_bbox_min = Vector(map(min, link_bbox_min, world_corner))
                        link_bbox_max = Vector(map(max, link_bbox_max, world_corner))

                link_size = link_bbox_max - link_bbox_min
                link_dimensions[link_name] = (link_size.x, link_size.y, link_size.z)

            # Create coordinate frame.
            create_coordinate_frame(
                position=combined_bbox_center,
                max_dim=combined_max_dim,
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

            # Compute camera distance for combined view.
            fov = 2 * math.atan((camera.sensor_width / 2) / camera.lens)
            base_distance = (combined_max_dim / 2) / math.tan(fov / 2)
            camera_distance = base_distance * CAMERA_DISTANCE_MARGIN_MULTIPLIER

            # Define combined views (top, bottom, sides).
            combined_views = [
                {"name": "combined_0_top", "direction": Vector((0, 0, 1))},
                {"name": "combined_1_bottom", "direction": Vector((0, 0, -1))},
            ]
            for i in range(num_combined_side_views):
                angle = 2 * math.pi * i / num_combined_side_views
                dir_vec = Vector((math.cos(angle), math.sin(angle), 0))
                combined_views.append(
                    {"name": f"combined_{i + 2}_side", "direction": dir_vec}
                )

            # Render combined views.
            for view in combined_views:
                direction = view["direction"].normalized()
                camera_obj.location = combined_bbox_center + direction * camera_distance
                look_at_target(camera_obj, combined_bbox_center)
                light_obj.location = camera_obj.location + direction * (
                    camera_distance * LIGHT_DISTANCE_RATIO
                )

                output_path = output_dir / f"{view['name']}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)
                combined_image_paths.append(output_path)

            # Render per-link views.
            for link_name, link_objs in link_to_objects.items():
                if not link_objs:
                    link_image_paths[link_name] = []
                    continue

                # Hide all objects except this link's objects.
                for obj in all_mesh_objs:
                    obj.hide_render = obj not in link_objs

                # Compute link-specific camera distance.
                link_bbox_min = Vector((float("inf"),) * 3)
                link_bbox_max = Vector((float("-inf"),) * 3)
                for obj in link_objs:
                    for corner in obj.bound_box:
                        world_corner = obj.matrix_world @ Vector(corner)
                        link_bbox_min = Vector(map(min, link_bbox_min, world_corner))
                        link_bbox_max = Vector(map(max, link_bbox_max, world_corner))

                link_center = (link_bbox_min + link_bbox_max) / 2
                link_size = link_bbox_max - link_bbox_min
                link_max_dim = max(link_size)
                link_camera_distance = (
                    (link_max_dim / 2) / math.tan(fov / 2)
                ) * CAMERA_DISTANCE_MARGIN_MULTIPLIER

                # Define link views (fewer than combined).
                link_views = []
                for i in range(num_link_side_views):
                    angle = 2 * math.pi * i / num_link_side_views
                    dir_vec = Vector((math.cos(angle), math.sin(angle), 0))
                    link_views.append(
                        {"name": f"{link_name}_{i}_side", "direction": dir_vec}
                    )

                link_image_paths[link_name] = []
                for view in link_views:
                    direction = view["direction"].normalized()
                    camera_obj.location = link_center + direction * link_camera_distance
                    look_at_target(camera_obj, link_center)
                    light_obj.location = camera_obj.location + direction * (
                        link_camera_distance * LIGHT_DISTANCE_RATIO
                    )

                    output_path = output_dir / f"{view['name']}.png"
                    scene.render.filepath = str(output_path)
                    bpy.ops.render.render(write_still=True)
                    link_image_paths[link_name].append(output_path)

            # Restore all objects to visible.
            for obj in all_mesh_objs:
                obj.hide_render = False

        # Add number overlays to combined images.
        for idx, output_path in enumerate(combined_image_paths):
            add_number_overlay(output_path, idx)

        # Add number overlays to per-link images.
        # Note: image filenames already include link names (e.g., link_0_0_side.png).
        for link_name, paths in link_image_paths.items():
            for idx, output_path in enumerate(paths):
                add_number_overlay(output_path, idx)

        total_images = len(combined_image_paths) + sum(
            len(p) for p in link_image_paths.values()
        )
        console_logger.info(
            f"Rendered {total_images} articulated views ({len(combined_image_paths)} "
            f"combined, {len(link_image_paths)} links) to {output_dir} in "
            f"{time.time()-start_time:.2f}s"
        )

        return ArticulatedRenderResult(
            combined_image_paths=combined_image_paths,
            link_image_paths=link_image_paths,
            link_dimensions=link_dimensions,
            combined_dimensions=combined_dimensions,
        )
