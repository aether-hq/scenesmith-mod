"""Utilities for adding annotations to Blender scenes and rendered images."""

import logging

from pathlib import Path

import bpy
import numpy as np

from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

from scenesmith.agent_utils.blender.geometry.camera_utils import get_pixel_coordinates

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.blender.overlays.image_annotations import (
    _compute_wall_center,
    _wall_local_to_world,
    draw_coordinate_annotations,
    load_annotation_font,
)


def draw_wall_coordinate_grid(
    wall_surface_data: dict,
    grid_divisions: int = 5,
    line_color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.8),
    label_color: tuple[float, float, float, float] = (0.0, 0.0, 0.8, 1.0),
) -> list[bpy.types.Object]:
    """Draw coordinate grid on wall surface with position labels.

    Creates grid lines and corner/center labels for wall orthographic views.
    Grid lines are gray, labels are blue showing wall-local coordinates.

    Args:
        wall_surface_data: Wall surface info including length, height.
        grid_divisions: Number of grid divisions (default 5).
        line_color: RGBA color for grid lines.
        label_color: RGBA color for coordinate labels.

    Returns:
        List of created Blender objects (for cleanup).
    """
    wall_length = wall_surface_data.get("length", 4.0)
    wall_height = wall_surface_data.get("height", 2.5)

    created_objects = []

    # Create material for grid lines.
    line_mat = bpy.data.materials.new(name="WallGridLineMaterial")
    line_mat.use_nodes = True
    bsdf = line_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = line_color
        bsdf.inputs["Alpha"].default_value = line_color[3]
    line_mat.blend_method = "BLEND"

    # Create grid lines.
    for i in range(grid_divisions + 1):
        # Vertical lines (constant x).
        x = wall_length * i / grid_divisions
        v_line = _create_line_mesh(
            start=(x, 0.01, 0),
            end=(x, 0.01, wall_height),
            name=f"wall_grid_v_{i}",
        )
        if v_line:
            v_line.data.materials.append(line_mat)
            created_objects.append(v_line)

        # Horizontal lines (constant z).
        z = wall_height * i / grid_divisions
        h_line = _create_line_mesh(
            start=(0, 0.01, z),
            end=(wall_length, 0.01, z),
            name=f"wall_grid_h_{i}",
        )
        if h_line:
            h_line.data.materials.append(line_mat)
            created_objects.append(h_line)

    # Create label material.
    label_mat = bpy.data.materials.new(name="WallGridLabelMaterial")
    label_mat.use_nodes = True
    bsdf = label_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = label_color
        bsdf.inputs["Emission Color"].default_value = label_color
        bsdf.inputs["Emission Strength"].default_value = 2.0

    # Add coordinate labels at key positions (corners and center).
    label_positions = [
        (0, 0),  # Bottom-left.
        (wall_length, 0),  # Bottom-right.
        (0, wall_height),  # Top-left.
        (wall_length, wall_height),  # Top-right.
        (wall_length / 2, wall_height / 2),  # Center.
    ]

    for x, z in label_positions:
        label_text = f"({x:.1f}, {z:.1f})"
        label_obj = _create_text_label(
            position=(x, 0.02, z),
            text=label_text,
            size=0.1,
            name=f"wall_label_{x:.0f}_{z:.0f}",
        )
        if label_obj:
            label_obj.data.materials.append(label_mat)
            created_objects.append(label_obj)

    return created_objects


def draw_excluded_regions(
    wall_surface_data: dict,
    material_color: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.5),
) -> list[bpy.types.Object]:
    """Draw hatched rectangles for door/window regions.

    Creates semi-transparent gray planes with diagonal hatching pattern
    to indicate areas where wall objects cannot be placed.

    Args:
        wall_surface_data: Wall surface info including excluded_regions.
        material_color: RGBA color for excluded region overlay.

    Returns:
        List of created Blender objects (for cleanup).
    """
    excluded_regions = wall_surface_data.get("excluded_regions", [])
    if not excluded_regions:
        return []

    created_objects = []

    # Create material for excluded regions.
    excluded_mat = bpy.data.materials.new(name="ExcludedRegionMaterial")
    excluded_mat.use_nodes = True
    excluded_mat.blend_method = "BLEND"

    bsdf = excluded_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = material_color
        bsdf.inputs["Alpha"].default_value = material_color[3]

    for i, region in enumerate(excluded_regions):
        x_min, z_min, x_max, z_max = region

        # Create plane mesh for the region.
        width = x_max - x_min
        height = z_max - z_min
        center_x = x_min + width / 2
        center_z = z_min + height / 2

        bpy.ops.mesh.primitive_plane_add(
            size=1.0,
            location=(center_x, 0.005, center_z),  # Slight Y offset.
        )
        plane = bpy.context.active_object
        plane.name = f"excluded_region_{i}"

        # Scale to match region size.
        plane.scale = (width, height, 1)

        # Rotate to face into room (plane default faces up, need to face +Y).
        plane.rotation_euler = (1.5708, 0, 0)  # 90 degrees around X.

        # Apply material.
        plane.data.materials.append(excluded_mat)

        created_objects.append(plane)

    return created_objects


def _create_line_mesh(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    name: str,
    thickness: float = 0.005,
) -> bpy.types.Object | None:
    """Create a thin cylinder to represent a line.

    Args:
        start: Start point (x, y, z).
        end: End point (x, y, z).
        name: Object name.
        thickness: Line thickness (cylinder radius).

    Returns:
        Created cylinder object, or None if creation fails.
    """

    start_vec = Vector(start)
    end_vec = Vector(end)
    direction = end_vec - start_vec
    length = direction.length

    if length < 0.001:
        return None

    # Create cylinder.
    bpy.ops.mesh.primitive_cylinder_add(
        radius=thickness,
        depth=length,
        location=((start_vec + end_vec) / 2).to_tuple(),
    )
    line_obj = bpy.context.active_object
    line_obj.name = name

    # Rotate to align with direction.
    # Default cylinder is aligned with Z axis, need to rotate to align with direction.
    up = Vector((0, 0, 1))
    direction_normalized = direction.normalized()

    if abs(up.dot(direction_normalized)) < 0.9999:
        rotation_quat = up.rotation_difference(direction_normalized)
        line_obj.rotation_euler = rotation_quat.to_euler()

    return line_obj


def _create_text_label(
    position: tuple[float, float, float],
    text: str,
    size: float = 0.1,
    name: str = "label",
) -> bpy.types.Object | None:
    """Create a 3D text label.

    Args:
        position: Label position (x, y, z).
        text: Text to display.
        size: Text size.
        name: Object name.

    Returns:
        Created text object, or None if creation fails.
    """
    try:
        bpy.ops.object.text_add(location=position)
        text_obj = bpy.context.active_object
        text_obj.name = name
        text_obj.data.body = text
        text_obj.scale = (size, size, size)

        # Rotate to face camera (+Y direction) in wall orthographic view.
        text_obj.rotation_euler = (1.5708, 0, 0)  # 90 degrees around X.

        return text_obj
    except Exception as e:
        console_logger.warning(f"Failed to create text label: {e}")
        return None


def _add_wall_coordinate_frame(
    draw: ImageDraw.ImageDraw, image_size: tuple[int, int], font: ImageFont.FreeTypeFont
) -> None:
    """Add coordinate frame indicator for wall orthographic views.

    Draws X and Z axis arrows in bottom-left corner with labels indicating:
    - X: distance along wall (horizontal)
    - Z: height above floor (vertical)

    Args:
        draw: PIL ImageDraw object.
        image_size: (width, height) of image.
        font: Font for axis labels.
    """
    # Position in bottom-left corner with margin.
    margin = 40
    origin_x = margin + 10
    origin_y = image_size[1] - margin - 10
    arrow_length = 50

    # Colors matching the coordinate grid (red).
    axis_color = (200, 0, 0)
    label_color = (200, 0, 0)
    line_width = 2

    # Draw X axis (horizontal, pointing right).
    x_end = (origin_x + arrow_length, origin_y)
    draw.line([(origin_x, origin_y), x_end], fill=axis_color, width=line_width)
    # X arrowhead.
    draw.polygon(
        [
            (x_end[0], x_end[1]),
            (x_end[0] - 8, x_end[1] - 5),
            (x_end[0] - 8, x_end[1] + 5),
        ],
        fill=axis_color,
    )
    # X label.
    draw.text((x_end[0] + 5, x_end[1] - 8), "X", fill=label_color, font=font)

    # Draw Z axis (vertical, pointing up).
    z_end = (origin_x, origin_y - arrow_length)
    draw.line([(origin_x, origin_y), z_end], fill=axis_color, width=line_width)
    # Z arrowhead.
    draw.polygon(
        [
            (z_end[0], z_end[1]),
            (z_end[0] - 5, z_end[1] + 8),
            (z_end[0] + 5, z_end[1] + 8),
        ],
        fill=axis_color,
    )
    # Z label.
    draw.text((z_end[0] + 5, z_end[1] - 5), "Z", fill=label_color, font=font)

    # Add small labels for axis meanings.
    small_font = font  # Use same font for now.
    draw.text(
        (origin_x + arrow_length + 20, origin_y - 3),
        "(along wall)",
        fill=(100, 100, 100),
        font=small_font,
    )
    draw.text(
        (origin_x + 12, origin_y - arrow_length - 20),
        "(height)",
        fill=(100, 100, 100),
        font=small_font,
    )


def add_wall_grid_annotations_pil(
    image_path: Path,
    wall_surface_data: dict,
    camera_obj: bpy.types.Camera,
    num_markers: int = 5,
) -> None:
    """Add wall coordinate markers as PIL overlays (post-render).

    Uses wall direction to compute grid positions directly in world coordinates,
    then projects to pixels. Draws red coordinate points and labels matching
    the style of floor coordinate annotations.

    Wall coordinate system:
        x = distance along wall (0 at left when viewed from inside room)
        z = height above floor

    Args:
        image_path: Path to rendered image.
        wall_surface_data: Wall surface info (direction, length, height).
        camera_obj: Blender camera for 3D-to-2D projection.
        num_markers: Number of markers per axis (e.g., 5 gives 5x5=25 markers).
    """
    wall_length = wall_surface_data.get("length", 4.0)
    wall_height = wall_surface_data.get("height", 2.5)
    transform_data = wall_surface_data.get("transform")

    if not transform_data:
        console_logger.warning("Wall surface missing transform data")
        return

    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)
    scene = bpy.context.scene

    # Load font - smaller than furniture floor coordinates for wall ortho views.
    # Divisor 80 with min_font_size=10 gives ~10pt on 512px wall ortho images.
    font = load_annotation_font(
        pil_image.size[0], base_font_size_divisor=80, min_font_size=10
    )

    # Red color matching floor coordinate style.
    marker_color = (255, 0, 0)
    dot_radius = 2

    # Generate grid points using quaternion rotation for correct transformation.
    # Wall local coords: x = along wall (0 at start), z = height above floor.
    visual_marks = {}
    for i in range(num_markers):
        for j in range(num_markers):
            # Wall-local coordinates at exact positions.
            wall_x = wall_length * i / (num_markers - 1) if num_markers > 1 else 0
            wall_z = wall_height * j / (num_markers - 1) if num_markers > 1 else 0
            # Round for clean display labels.
            display_x = round(wall_x, 1)
            display_z = round(wall_z, 1)

            # Wall-local position (x along wall, y=0 on surface, z height).
            local_pos = np.array([wall_x, 0, wall_z])

            # Transform to world coordinates using quaternion rotation.
            world_pos = _wall_local_to_world(
                local_pos=local_pos, transform=transform_data
            )

            px = get_pixel_coordinates(
                scene=scene, camera=camera_obj, world_coord=Vector(world_pos.tolist())
            )

            if _is_valid_pixel(px=px, image_size=pil_image.size):
                # Store as (display_x, display_z) -> pixel position.
                # Use clean display values (0, 1, 2, ...) not margin-adjusted values.
                visual_marks[(display_x, display_z)] = px

    # Draw coordinate annotations using shared helper (same style as floor).
    draw_coordinate_annotations(
        draw=draw,
        visual_marks=visual_marks,
        font=font,
        dot_radius=dot_radius,
        text_color=marker_color,
        image_size=pil_image.size,
    )

    pil_image.save(str(image_path))


def _is_valid_pixel(px: tuple[float, float], image_size: tuple[int, int]) -> bool:
    """Check if pixel coordinates are within image bounds.

    Args:
        px: Pixel coordinates (x, y).
        image_size: Image size (width, height).

    Returns:
        True if pixel is within bounds, False otherwise.
    """
    if px[0] < 0 or px[1] < 0:
        return False
    if px[0] > image_size[0] or px[1] > image_size[1]:
        return False
    return True


def add_wall_surface_id_label(image_path: Path, wall_surface_id: str) -> None:
    """Add wall surface ID label to rendered wall orthographic image.

    Draws a label in the top-right corner showing the wall_surface_id that
    can be used with place/move tools.

    Args:
        image_path: Path to the rendered image file.
        wall_surface_id: Wall surface identifier to display.
    """
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Load font.
        # Divisor 25 gives ~20pt on 512px wall views for readable corner labels.
        font = load_annotation_font(img.width, base_font_size_divisor=25)

        # Build label text.
        label_text = f"Wall: {wall_surface_id}"

        # Get text bounding box.
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position in top-right corner with padding.
        padding = 8
        x = img.width - text_width - padding * 2
        y = padding

        # Draw background (blue for wall surfaces).
        bg_color = (70, 130, 180)  # Steel blue.
        draw.rectangle(
            [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ],
            fill=bg_color,
        )

        # Draw text in white.
        draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

        img.save(str(image_path))

    except Exception as e:
        console_logger.warning(f"Failed to add wall surface ID label: {e}")


def add_wall_labels_to_top_view(
    image_path: Path, camera_obj: bpy.types.Object, wall_surfaces: list[dict]
) -> None:
    """Add wall surface labels to a top-down view.

    Projects wall center positions to 2D and draws labels showing each
    wall's surface_id. Labels are positioned at wall midpoints.

    Args:
        image_path: Path to the rendered top-down image.
        camera_obj: Blender camera object for projection.
        wall_surfaces: List of wall surface dicts with surface_id, direction,
            length, height, and transform.
    """
    if not wall_surfaces:
        return

    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        scene = bpy.context.scene

        # Load font.
        font = load_annotation_font(image_with=img.width, base_font_size_divisor=70)

        for wall_data in wall_surfaces:
            surface_id = wall_data.get(
                "surface_id", wall_data.get("wall_id", "unknown")
            )
            wall_direction = wall_data.get("direction", "north")
            wall_length = wall_data.get("length", 4.0)
            wall_height = wall_data.get("height", 2.5)
            transform_data = wall_data.get("transform")

            if not transform_data:
                continue

            # Compute wall center from transform and dimensions.
            wall_center = _compute_wall_center(
                transform=transform_data,
                wall_length=wall_length,
                wall_height=wall_height,
            )

            # Project to pixel coordinates.
            px = get_pixel_coordinates(
                scene=scene,
                camera=camera_obj,
                world_coord=Vector(wall_center.tolist()),
            )

            # Debug: log wall center and pixel position.
            console_logger.info(
                f"Wall label {surface_id}: direction={wall_direction}, "
                f"transform={transform_data[:3]}, center={wall_center}, "
                f"px={px}, img_size={img.size}"
            )

            if not _is_valid_pixel(px=px, image_size=img.size):
                continue

            # Draw label.
            label_text = surface_id
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text on projected position.
            x = int(px[0] - text_width / 2)
            y = int(px[1] - text_height / 2)

            # Draw background.
            padding = 4
            bg_color = (70, 130, 180)  # Steel blue.
            draw.rectangle(
                [
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding,
                ],
                fill=bg_color,
            )

            # Draw text in white.
            draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

        img.save(str(image_path))

    except Exception as e:
        console_logger.warning(f"Failed to add wall labels to top view: {e}")
