import atexit
import copy
import logging
import os
import sys

from collections import defaultdict

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.scene.room_parts.room_models import (
    SceneObject,
    SupportSurface,
)
from scenesmith.utils.geometry.geometry_utils import compute_aabb_corners

console_logger = logging.getLogger(__name__)


# Track virtual display for cleanup on process exit.
_virtual_display = None


def _cleanup_virtual_display() -> None:
    """Clean up virtual display on process exit."""
    global _virtual_display
    if _virtual_display is not None:
        try:
            _virtual_display.stop()
        except Exception:
            pass  # Best effort cleanup.
        _virtual_display = None


def setup_virtual_display_if_needed() -> None:
    """Set up a virtual display for headless rendering if needed.

    On Linux without a DISPLAY, creates a virtual display. The DISPLAY env var
    check prevents duplicate creation within a process. Registers atexit handler
    to clean up Xvfb on process exit.
    """
    global _virtual_display
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        console_logger.info("Setting up virtual display for rendering.")
        from pyvirtualdisplay import Display

        _virtual_display = Display(visible=0, size=(1400, 900))
        _virtual_display.start()
        atexit.register(_cleanup_virtual_display)


def get_drake_model_name(obj: SceneObject) -> str:
    """Compute Drake model name matching to_drake_directive logic in scene.py.

    Drake model names are created by combining the object name (lowercased,
    spaces replaced with underscores) with a suffix derived from the object ID.
    """
    id_suffix = str(obj.object_id).split("_")[-1][:8]
    return f"{obj.name.lower().replace(' ', '_')}_{id_suffix}"


def apply_fk_to_surfaces(
    surfaces: list[SupportSurface],
    rest_transforms: dict[str, RigidTransform],
    open_transforms: dict[str, RigidTransform],
    link_to_joint: dict[str, str],
    open_joints: set[str],
) -> list[SupportSurface]:
    """Apply FK delta transforms to surfaces based on which joints are open.

    For articulated furniture, surfaces are extracted at rest (closed) position.
    When rendering with specific joints open, surfaces on those links need FK
    transforms to match the opened geometry.

    Args:
        surfaces: Support surfaces with link_name populated.
        rest_transforms: Link transforms at closed joint positions.
        open_transforms: Link transforms at current joint positions.
        link_to_joint: Mapping from child link name to controlling joint name.
        open_joints: Set of joint names that are open in this render.

    Returns:
        New list of surfaces with FK transforms applied where applicable.
        Surfaces on closed joints (or base link) keep their original transforms.
    """
    result = []

    for surface in surfaces:
        link_name = surface.link_name

        # No link association = base link = no FK needed.
        if not link_name or link_name not in link_to_joint:
            result.append(surface)
            continue

        joint_name = link_to_joint.get(link_name)

        # Only transform if this joint is in the open set.
        if joint_name and joint_name in open_joints:
            # Check we have transforms for this link.
            if link_name in rest_transforms and link_name in open_transforms:
                # Compute delta: open @ rest.inverse().
                rest_tf = rest_transforms[link_name]
                open_tf = open_transforms[link_name]
                delta = open_tf @ rest_tf.inverse()

                # Create new surface with transformed pose.
                new_surface = copy.copy(surface)
                new_surface.transform = delta @ surface.transform
                result.append(new_surface)
                console_logger.debug(
                    f"Applied FK to surface {surface.surface_id} on link {link_name}"
                )
            else:
                # Missing transform data - keep original.
                console_logger.warning(
                    f"Missing transform for link {link_name}, keeping original surface"
                )
                result.append(surface)
        else:
            # Joint not open - keep at rest position.
            result.append(surface)

    return result


def classify_surfaces_for_rendering(
    surfaces: list[SupportSurface], link_to_joint: dict[str, str]
) -> tuple[list[SupportSurface], dict[str, list[SupportSurface]]]:
    """Classify surfaces for per-joint rendering strategy.

    Surfaces are classified into:
    - Static surfaces: On base link (not controlled by any joint)
    - Per-joint surfaces: On child links (need per-joint renders)

    For articulated furniture like nightstands with drawers:
    - Body surfaces → static (main render)
    - Drawer surfaces → per-joint (each drawer gets its own render)

    Args:
        surfaces: Support surfaces with link_name populated.
        link_to_joint: Mapping from child link name to controlling joint name.

    Returns:
        Tuple of (static_surfaces, joint_surfaces) where joint_surfaces is a
        dict mapping joint_name to list of surfaces controlled by that joint.
    """
    static_surfaces: list[SupportSurface] = []
    joint_surfaces: dict[str, list[SupportSurface]] = defaultdict(list)

    for surface in surfaces:
        link_name = surface.link_name

        # No link association or base link = static surface.
        if not link_name or link_name not in link_to_joint:
            static_surfaces.append(surface)
            continue

        # Surface on a moving link = per-joint render.
        joint_name = link_to_joint[link_name]
        joint_surfaces[joint_name].append(surface)

    console_logger.debug(
        f"Classified {len(surfaces)} surfaces: {len(static_surfaces)} static, "
        f"{len(joint_surfaces)} joints with surfaces"
    )

    return static_surfaces, dict(joint_surfaces)


def compute_drawer_direction(
    rest_transform: RigidTransform, open_transform: RigidTransform
) -> list[float]:
    """Compute the direction a drawer moves when opening.

    Uses FK delta (open vs rest) to determine drawer sliding direction.
    This is used to position the camera to look into the drawer opening.

    Args:
        rest_transform: Link transform at rest (closed) position.
        open_transform: Link transform at open position.

    Returns:
        3D direction vector [x, y, z] of drawer movement (not normalized).
    """
    # FK delta = open @ rest.inverse()
    delta = open_transform @ rest_transform.inverse()
    # Translation component tells us how far and which way the drawer moved.
    translation = delta.translation()
    return translation.tolist()


def build_support_surfaces_data(surfaces: list[SupportSurface]) -> list[dict]:
    """Build serializable surface data for Blender overlay config.

    Args:
        surfaces: List of support surfaces (possibly FK-transformed).

    Returns:
        List of dicts with surface_id, corners, convex_hull_vertices, mesh_faces.
    """
    support_surfaces_data = []

    for surface in surfaces:
        # Compute 8 corners of the bounding box in local space.
        corners_local = compute_aabb_corners(
            bbox_min=surface.bounding_box_min,
            bbox_max=surface.bounding_box_max,
        )

        # Transform all corners to world space using Drake Z-up coordinates.
        corners_world = [
            (surface.transform @ corner).tolist() for corner in corners_local
        ]

        # Get convex hull vertices for coordinate marker filtering.
        convex_hull_vertices = None
        mesh_faces = None
        if surface.mesh is not None:
            # Get mesh vertices in world space (mesh-local -> surface -> world).
            mesh_vertices_local = surface.mesh.vertices
            transform_matrix = surface.transform.GetAsMatrix4()
            mesh_vertices_world = []
            for v in mesh_vertices_local:
                v_hom = np.append(v, 1.0)
                v_world_hom = transform_matrix @ v_hom
                mesh_vertices_world.append(v_world_hom[:3].tolist())
            convex_hull_vertices = mesh_vertices_world
            # Include face indices for rendering the surface mesh.
            mesh_faces = surface.mesh.faces.tolist()

        surface_data = {
            "surface_id": str(surface.surface_id),
            "corners": corners_world,
            "convex_hull_vertices": convex_hull_vertices,
            "mesh_faces": mesh_faces,
        }
        support_surfaces_data.append(surface_data)

    return support_surfaces_data
