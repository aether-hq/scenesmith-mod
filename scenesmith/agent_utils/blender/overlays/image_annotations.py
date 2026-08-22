"""Utilities for adding annotations to Blender scenes and rendered images."""

import logging

from pathlib import Path

import bpy
import numpy as np

from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

from scenesmith.agent_utils.blender.geometry.camera_utils import get_pixel_coordinates
from scenesmith.agent_utils.scene.house_parts.openings import ClearanceOpeningData

console_logger = logging.getLogger(__name__)


def _quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float):
    """Convert quaternion to 3x3 rotation matrix."""
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    return np.array(
        [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ]
    )


def _wall_local_to_world(local_pos: np.ndarray, transform: list[float]) -> np.ndarray:
    """Convert wall-local position to world coordinates.

    Wall local coordinate system:
    - X: along wall (from start to end)
    - Y: into room (wall normal)
    - Z: up (vertical)

    Args:
        local_pos: Position in wall-local frame [x, y, z].
        transform: [x, y, z, qw, qx, qy, qz] pose of wall origin in world frame.

    Returns:
        Position in world coordinates.
    """
    wall_origin = np.array(transform[:3])
    qw, qx, qy, qz = transform[3:7]
    rotation_matrix = _quaternion_to_rotation_matrix(qw, qx, qy, qz)
    return wall_origin + rotation_matrix @ local_pos


def _compute_wall_center(
    transform: list[float], wall_length: float, wall_height: float
) -> np.ndarray:
    """Compute wall center in world coordinates using quaternion rotation.

    Uses the quaternion from the transform to rotate the local center
    position to world coordinates. This correctly handles any wall origin
    convention (corner-based, edge-based, centered, etc.).

    Wall local frame:
        x = along wall (0 at origin end, wall_length at far end)
        y = 0 (on wall surface)
        z = height above floor

    Args:
        transform: [x, y, z, qw, qx, qy, qz] pose of wall origin in world frame.
        wall_length: Wall length in meters.
        wall_height: Wall height in meters.

    Returns:
        Wall center position in world coordinates.
    """
    wall_origin = np.array(transform[:3])
    qw, qx, qy, qz = transform[3], transform[4], transform[5], transform[6]

    # Build rotation matrix from quaternion.
    # This gives us the local-to-world rotation.
    rot_matrix = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

    # Local center: half length along X, zero along Y, half height along Z.
    local_center = np.array([wall_length / 2, 0, wall_height / 2])

    # Transform to world: rotate local center then add origin.
    world_offset = rot_matrix @ local_center
    return wall_origin + world_offset


def annotate_image_with_coordinates(
    image_path: Path,
    marks: dict[tuple[float, float], tuple[int, int]],
    dot_radius: int = 3,
    base_font_size_divisor: float = 60,
    text_color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Add coordinate labels and markers to rendered image.

    Args:
        image_path: Path to the rendered image file.
        marks: Markers with labels (world coords -> pixel coords).
        dot_radius: The radius of the dots in pixels.
        base_font_size_divisor: The divisor of the width of the image to determine
            the base font size.
        text_color: The color of the text and markers.
    """
    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)
    font = load_annotation_font(pil_image.size[0], base_font_size_divisor)

    # Draw markers with labels.
    draw_coordinate_annotations(
        draw=draw,
        visual_marks=marks,
        font=font,
        dot_radius=dot_radius,
        text_color=text_color,
        image_size=pil_image.size,
    )

    pil_image.save(str(image_path))


def load_annotation_font(
    image_width: int, base_font_size_divisor: float, min_font_size: int = 16
) -> ImageFont.ImageFont:
    """Load font for coordinate annotations with fallback logic.

    Args:
        image_width: Width of the image in pixels.
        base_font_size_divisor: Divisor to calculate font size (image_width / divisor).
        min_font_size: Minimum font size to use (default: 16).

    Returns:
        Loaded font with fallback to default if system fonts unavailable.
    """
    base_font_size = int(image_width / base_font_size_divisor)
    font_size = max(min_font_size, base_font_size)
    console_logger.debug(f"Base font size: {base_font_size}, Font size: {font_size}")

    font_paths = [
        "arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()


def draw_coordinate_annotations(
    draw: ImageDraw.ImageDraw,
    visual_marks: dict[tuple[float, float], tuple[int, int]],
    font: ImageFont.ImageFont,
    dot_radius: int,
    text_color: tuple[int, int, int],
    image_size: tuple[int, int],
) -> None:
    """Draw coordinate dots and labels on the image."""
    width, height = image_size

    for (world_x, world_y), (pixel_x, pixel_y) in visual_marks.items():
        dot_x, dot_y = int(pixel_x), int(pixel_y)

        # Draw dot.
        draw.ellipse(
            [
                dot_x - dot_radius,
                dot_y - dot_radius,
                dot_x + dot_radius,
                dot_y + dot_radius,
            ],
            fill=text_color,
        )

        # Draw text label (limited to 2 decimal places, drop trailing zeros).
        # Format with 2 decimals then strip trailing zeros and decimal point.
        x_str = f"{world_x:.2f}".rstrip("0").rstrip(".")
        y_str = f"{world_y:.2f}".rstrip("0").rstrip(".")
        text = f"({x_str},{y_str})"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = max(0, min(dot_x - text_width // 2, width - text_width))
        text_y = max(0, min(dot_y + dot_radius + 2, height - text_height))

        draw.text((text_x, text_y), text, fill=text_color, font=font)


def add_set_of_mark_labels_pil(
    image_path: Path,
    scene_objects: list[dict],
    camera_obj: bpy.types.Camera,
    rendering_mode: str = "furniture",
    current_surface_id: str | None = None,
    annotate_object_types: list[str] | None = None,
) -> None:
    """Add set-of-mark labels as PIL overlays (always on top).

    Uses existing camera projection utilities (get_pixel_coordinates)
    to convert 3D bbox centers to 2D pixel positions.

    Args:
        image_path: Path to rendered image.
        scene_objects: List of scene object metadata dicts.
        camera_obj: Blender camera for 3D-to-2D projection.
        rendering_mode: Rendering mode ("furniture", "manipuland", "wall").
        current_surface_id: Optional surface ID to filter manipulands by.
            If provided, only show labels for manipulands on this surface.
        annotate_object_types: Optional list of object types to annotate.
            If provided, only objects of these types get labels.
            Example: ["wall_mounted"] to only label wall objects.
    """
    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)

    # Smaller font to match 3D text scale (0.15).
    # 3D text was quite compact, so use larger divisor.
    font = load_annotation_font(pil_image.size[0], base_font_size_divisor=60)

    scene = bpy.context.scene

    for obj_meta in scene_objects:
        obj_type = obj_meta.get("object_type", "")

        # Skip room geometry and floor objects. Keep wall ones as they are useful
        # for snapping and facing checks.
        if obj_type in ["room_geometry", "floor"]:
            continue

        # Filter by annotate_object_types if provided.
        if annotate_object_types is not None and obj_type not in annotate_object_types:
            continue

        # In manipuland mode, only show manipuland labels (exclude all other types).
        if rendering_mode == "manipuland" and obj_type != "manipuland":
            continue

        # Filter by surface if current_surface_id is provided.
        if current_surface_id is not None:
            parent_surface_id = obj_meta.get("parent_surface_id")
            if parent_surface_id != current_surface_id:
                continue

        # Use object_id directly for labels (already short with 2-char UUID).
        obj_name = obj_meta.get("object_id", "object")
        bbox = obj_meta.get("bounding_box")
        if not bbox:
            continue

        # Determine object type for positioning.
        obj_type = obj_meta.get("object_type", "")
        is_furniture = obj_type == "furniture"

        if is_furniture:
            # For furniture (support surfaces), place label at surface center.
            # Use XY position of bbox center but Z at support surface height (bbox bottom).
            extents = bbox["extents"]
            surface_z = bbox["center"][2] - extents[2] / 2  # Bottom of bbox.
            label_3d_pos = Vector((bbox["center"][0], bbox["center"][1], surface_z))
        else:
            # For other objects (manipulands, etc.), use bbox center.
            label_3d_pos = Vector(bbox["center"])

        pixel_x, pixel_y = get_pixel_coordinates(
            scene=scene, camera=camera_obj, world_coord=label_3d_pos
        )

        # Apply vertical offset for non-furniture objects to avoid occlusion.
        # Furniture labels stay at surface level for better association.
        if not is_furniture:
            label_offset_pixels = 30  # Pixels above object center.
            pixel_y -= label_offset_pixels  # Move up in screen space.

        # Measure text size.
        text_bbox = draw.textbbox((0, 0), obj_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw blue background rectangle.
        padding = 5
        bg_left = int(pixel_x - text_width // 2 - padding)
        bg_top = int(pixel_y - text_height // 2 - padding)
        bg_right = int(pixel_x + text_width // 2 + padding)
        bg_bottom = int(pixel_y + text_height // 2 + padding)

        draw.rectangle(
            [bg_left, bg_top, bg_right, bg_bottom],
            fill=(77, 153, 255),  # Blue color for all labels (furniture and objects).
        )

        # Draw white text centered on background.
        text_x = int(pixel_x - text_width // 2)
        text_y = int(pixel_y - text_height // 2)
        draw.text((text_x, text_y), obj_name, fill=(255, 255, 255), font=font)

    pil_image.save(str(image_path))


def add_opening_labels_pil(
    image_path: Path, openings: list[ClearanceOpeningData], camera_obj: bpy.types.Camera
) -> None:
    """Add door/window/opening labels as PIL overlays.

    Uses camera projection to convert 3D opening centers to 2D pixel positions.
    Labels use distinct colors by opening type for visual clarity.

    Args:
        image_path: Path to rendered image.
        openings: List of ClearanceOpeningData from RoomGeometry.openings.
        camera_obj: Blender camera for 3D-to-2D projection.
    """
    if not openings:
        return

    pil_image = Image.open(str(image_path))

    draw = ImageDraw.Draw(pil_image)

    # Use slightly smaller font for opening labels.
    font = load_annotation_font(pil_image.size[0], base_font_size_divisor=70)

    scene = bpy.context.scene

    # Color scheme by opening type.
    type_colors = {
        "door": (34, 139, 34),  # Forest green for doors.
        "window": (255, 165, 0),  # Orange for windows.
        "open": (138, 43, 226),  # Blue-violet for open connections.
    }
    default_color = (100, 100, 100)  # Gray fallback.

    for opening in openings:
        label = opening.opening_id
        center_world = opening.center_world
        opening_type = opening.opening_type.lower()

        if not center_world:
            continue

        # Use actual opening center position for projection.
        label_3d_pos = Vector(center_world)
        pixel_x, pixel_y = get_pixel_coordinates(
            scene=scene, camera=camera_obj, world_coord=label_3d_pos
        )

        # Skip if projection failed (behind camera or out of frame).
        if pixel_x < 0 or pixel_y < 0:
            continue
        if pixel_x > pil_image.size[0] or pixel_y > pil_image.size[1]:
            continue

        # Measure text size.
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle with type-specific color.
        bg_color = type_colors.get(opening_type, default_color)
        padding = 4
        bg_left = int(pixel_x - text_width // 2 - padding)
        bg_top = int(pixel_y - text_height // 2 - padding)
        bg_right = int(pixel_x + text_width // 2 + padding)
        bg_bottom = int(pixel_y + text_height // 2 + padding)

        draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_color)

        # Draw white text centered on background.
        text_x = int(pixel_x - text_width // 2)
        text_y = int(pixel_y - text_height // 2)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    pil_image.save(str(image_path))
