#!/usr/bin/env python3
"""Export an existing scene to self-contained MuJoCo MJCF format.

Takes a scene directory (e.g., outputs/2025-12-05/13-39-27/scene_039) and exports
it to a self-contained MuJoCo directory with the scene.xml and all referenced
mesh assets.

Can also export a single Drake SDF file to MuJoCo MJCF format.

Usage:
    python scripts/export_scene_to_mujoco.py <scene_path> [--output <output_path>]

Example:
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039
    python scripts/export_scene_to_mujoco.py outputs/2025-12-05/13-39-27/scene_039 \
        --output /tmp/mujoco_scene
"""

import logging
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def parse_pose(pose_elem: ET.Element | None) -> tuple[list[float], list[float]]:
    """Parse SDF pose element into position and quaternion."""
    if pose_elem is None or pose_elem.text is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

    values = [float(v) for v in pose_elem.text.split()]
    pos = values[:3]

    if len(values) >= 6:
        roll, pitch, yaw = values[3:6]
        quat = rpy_to_quat(roll, pitch, yaw)
    else:
        quat = [1.0, 0.0, 0.0, 0.0]

    return pos, quat


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert roll-pitch-yaw to quaternion (w-first)."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [float(w), float(x), float(y), float(z)]


def quat_conjugate(q: list[float]) -> list[float]:
    """Return conjugate of quaternion (w, x, y, z)."""
    return [q[0], -q[1], -q[2], -q[3]]


def quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_rotate_vector(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by quaternion q."""
    # v' = q * v * q^-1 where v is treated as quaternion (0, vx, vy, vz).
    v_quat = [0.0, v[0], v[1], v[2]]
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return [result[1], result[2], result[3]]


def quat_to_rotation_matrix(q: list[float]) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def compute_relative_pose(
    parent_pos: list[float],
    parent_quat: list[float],
    child_pos: list[float],
    child_quat: list[float],
) -> tuple[list[float], list[float]]:
    """Compute child pose relative to parent.

    Given parent and child poses in world/model frame, compute the child's
    pose relative to the parent frame.

    Args:
        parent_pos: Parent position in model frame [x, y, z].
        parent_quat: Parent quaternion in model frame [w, x, y, z].
        child_pos: Child position in model frame [x, y, z].
        child_quat: Child quaternion in model frame [w, x, y, z].

    Returns:
        (relative_pos, relative_quat): Child pose in parent frame.
    """
    # Relative position: rotate (child_pos - parent_pos) by inverse of parent rotation.
    delta_pos = [
        child_pos[0] - parent_pos[0],
        child_pos[1] - parent_pos[1],
        child_pos[2] - parent_pos[2],
    ]
    parent_quat_inv = quat_conjugate(parent_quat)
    rel_pos = quat_rotate_vector(parent_quat_inv, delta_pos)

    # Relative rotation: q_rel = q_parent^-1 * q_child.
    rel_quat = quat_multiply(parent_quat_inv, child_quat)

    return rel_pos, rel_quat
