import logging

from pathlib import Path

import bpy
import numpy as np

from PIL import Image

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


def _apply_eevee_speed_settings(scene, keep_shadows: bool = True) -> None:
    """Apply EEVEE speed optimizations for faster rendering.

    Disables expensive effects that aren't needed for asset validation or scene
    observation. Keeps shadows by default since they help VLMs understand 3D form.

    Args:
        scene: Blender scene object.
        keep_shadows: If True, keep shadows enabled (helps VLM see 3D form).
            If False, disable shadows entirely for maximum speed.
    """
    # Disable bloom (lens effect, not needed for validation).
    try:
        scene.eevee.use_bloom = False
    except AttributeError:
        pass  # Not available in EEVEE_NEXT.

    # Disable screen space reflections (expensive, not needed).
    try:
        scene.eevee.use_ssr = False
    except AttributeError:
        pass

    # Disable ambient occlusion (adds render time, not critical for validation).
    try:
        scene.eevee.use_gtao = False
    except AttributeError:
        pass

    # Disable volumetric shadows (expensive atmospheric effect).
    try:
        scene.eevee.use_volumetric_shadows = False
    except AttributeError:
        pass

    if keep_shadows:
        # Reduce shadow quality for speed while keeping depth cues.
        try:
            scene.eevee.shadow_cube_size = "256"  # Reduced from default 1024.
        except AttributeError:
            pass
        try:
            scene.eevee.shadow_cascade_size = "256"
        except AttributeError:
            pass
    else:
        # Disable shadows entirely for maximum speed.
        try:
            scene.eevee.use_shadows = False
        except AttributeError:
            pass
        for light in bpy.data.lights:
            try:
                light.use_shadow = False
            except AttributeError:
                pass


def _composite_onto_grey(image_path: Path, grey_value: int = 128) -> None:
    """Composite RGBA image onto neutral grey background.

    Args:
        image_path: Path to RGBA PNG image (modified in place).
        grey_value: Grey background value 0-255 (default 128 = 50% grey).
    """
    img = Image.open(image_path).convert("RGBA")
    background = Image.new("RGB", img.size, (grey_value, grey_value, grey_value))
    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask.
    background.save(image_path)


def _compute_bounds_from_corners(corners: list[list[float]]) -> tuple:
    """Compute axis-aligned and oriented bounding box from corner points.

    Args:
        corners: List of 8 corner points [x, y, z] in Drake Z-up world coordinates.

    Returns:
        Tuple of (corners_array, bbox_min, bbox_max) where:
        - corners_array: numpy array of shape (8, 3) with corner positions
        - bbox_min: numpy array of AABB minimum bounds
        - bbox_max: numpy array of AABB maximum bounds
    """
    # Convert to numpy array for easier manipulation.
    corners_array = np.array(corners)

    # Compute axis-aligned bounding box for positioning coordinate frame.
    bbox_min = corners_array.min(axis=0)
    bbox_max = corners_array.max(axis=0)

    return corners_array, bbox_min, bbox_max


def _compute_wall_center_from_transform(
    transform: list[float], wall_length: float, wall_height: float
) -> np.ndarray:
    """Compute wall center in world coordinates using quaternion rotation.

    Uses the quaternion from the transform to rotate the local center
    position to world coordinates. This correctly handles any wall origin
    convention (corner-based, edge-based, centered, etc.).

    Wall local frame:
    - Local X is along wall (from origin to far end)
    - Local Y is outward from room (wall normal)
    - Local Z is up

    Args:
        transform: [x, y, z, qw, qx, qy, qz] pose of wall origin in world frame.
        wall_length: Wall length in meters.
        wall_height: Wall height in meters.

    Returns:
        Wall center position in world coordinates as numpy array.
    """
    wall_origin = np.array(transform[:3])
    qw, qx, qy, qz = transform[3], transform[4], transform[5], transform[6]

    # Build rotation matrix from quaternion.
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
