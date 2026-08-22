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

from pathlib import Path

import mujoco

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scripts.mujoco_export.mesh_conversion import create_mujoco_spec_with_environment
from scripts.mujoco_export.model_processing import process_sdf_model
from scripts.mujoco_export.scene_export import compile_and_write_mjcf
from scripts.mujoco_export.scene_io import parse_sdf_with_drake_namespace

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def export_sdf_to_mujoco(
    sdf_path: Path,
    output_dir: Path,
    is_static: bool = False,
) -> Path:
    """Export a single SDF file to MuJoCo MJCF format.

    This is a standalone export mode for testing articulated models directly,
    without needing a full scene directory structure.

    Args:
        sdf_path: Path to SDF file.
        output_dir: Output directory for MJCF and meshes.
        is_static: Whether the model is static (no freejoint).

    Returns:
        Path to the exported scene.xml file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    output_path = output_dir / "scene.xml"

    # Create MuJoCo spec with environment (skybox, ground with collision).
    spec = create_mujoco_spec_with_environment(
        model_name=sdf_path.stem, ground_collides=True
    )

    # Track mesh, texture, and color assets.
    mesh_assets: dict[str, str] = {}
    texture_assets: dict[str, str] = {}
    color_assets: dict[str, list[float]] = {}

    # Parse SDF to get model name.
    tree = parse_sdf_with_drake_namespace(sdf_path)
    model_elem = tree.getroot().find(".//model")
    if model_elem is None:
        raise ValueError(f"No model element in SDF: {sdf_path}")

    model_name = model_elem.get("name", sdf_path.stem)

    # Process SDF model with identity quaternion, raised 0.5m for testing.
    # room_id is empty since this is a single-model export, not a multi-room scene.
    process_sdf_model(
        sdf_path=sdf_path,
        sdf_dir=sdf_path.parent,
        model_name=model_name,
        transform_pos=[0.0, 0.0, 0.5],
        transform_quat=[1.0, 0.0, 0.0, 0.0],
        is_static=is_static,
        spec=spec,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
        color_assets=color_assets,
        room_id="",
    )

    # Add mesh assets to spec.
    for mesh_name, mesh_filename in mesh_assets.items():
        mesh = spec.add_mesh(name=mesh_name)
        mesh.file = mesh_filename

    # Add texture and material assets.
    for texture_name, texture_filename in texture_assets.items():
        texture = spec.add_texture(name=texture_name)
        texture.file = texture_filename
        texture.type = mujoco.mjtTexture.mjTEXTURE_2D
        material = spec.add_material(name=texture_name)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name

    console_logger.info(f"Processed model: {model_name}")

    return compile_and_write_mjcf(
        spec=spec,
        output_path=output_path,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
    )
