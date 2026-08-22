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
import re

from pathlib import Path

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


def fix_usd_texture_wrap_modes(materials_lib_path: Path) -> None:
    """Add repeat wrap mode to all UsdUVTexture shaders.

    The mujoco_usd_converter doesn't set wrapS/wrapT on texture shaders,
    causing UVs outside 0-1 to be clamped instead of tiled. Floor and wall
    meshes use UV coordinates > 1.0 for texture tiling, so they need repeat
    wrap mode to display correctly in usdview and Isaac Sim.

    Args:
        materials_lib_path: Path to MaterialsLibrary.usdc file.
    """
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.Open(str(materials_lib_path))

    textures_fixed = 0
    for prim in stage.TraverseAll():
        if prim.GetTypeName() == "Shader":
            shader = UsdShade.Shader(prim)
            shader_id = shader.GetIdAttr().Get()
            if shader_id == "UsdUVTexture":
                # Add wrap mode inputs for texture tiling.
                wrap_s = shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token)
                wrap_s.Set("repeat")
                wrap_t = shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token)
                wrap_t.Set("repeat")
                textures_fixed += 1

    stage.Save()
    console_logger.info(f"  Fixed wrap mode on {textures_fixed} texture shaders")


def export_to_usd(
    output_path: Path,
    output_dir: Path,
    apply_isaac_sim_fix: bool = True,
) -> None:
    """Export MuJoCo scene to USD format.

    Args:
        output_path: Path to scene.xml file.
        output_dir: Output directory containing meshes.

    Note:
        USD export is incompatible with bpy (Blender Python) in the same
        environment. Both packages install conflicting versions of the pxr
        (OpenUSD) library. If you need USD export, use a separate venv
        without bpy installed.
    """
    console_logger.info("\nExporting to USD format...")

    # Check for bpy/pxr conflict.
    try:
        import bpy  # noqa: F401

        console_logger.error(
            "USD export is incompatible with bpy (Blender) in the same environment.\n"
            "Both packages install conflicting versions of the pxr (OpenUSD) library.\n"
            "To use USD export, run the setup script to create a separate venv:\n"
            "  ./scripts/setup_mujoco_export.sh\n"
            "  source .mujoco_venv/bin/activate\n"
            "  python scripts/export_scene_to_mujoco.py --sdf ... --usd"
        )
        return
    except ImportError:
        pass  # bpy not installed, safe to proceed.

    try:
        import mujoco_usd_converter
        import usdex.core

        from pxr import Usd

        usd_dir = output_dir / "usd"
        usd_dir.mkdir(exist_ok=True)

        # Create a modified XML without checker texture (not supported in USD).
        # Remove grid texture, material, and ground plane geom.
        with open(output_path, "r") as f:
            xml_for_usd = f.read()

        # Remove grid texture (builtin checker).
        xml_for_usd = re.sub(r'<texture[^>]*name="grid"[^/]*/>\s*', "", xml_for_usd)
        # Remove grid material.
        xml_for_usd = re.sub(r'<material[^>]*name="grid"[^/]*/>\s*', "", xml_for_usd)
        # Remove ground plane geom that uses grid material.
        xml_for_usd = re.sub(
            r'<geom[^>]*name="ground_plane"[^/]*/>\s*', "", xml_for_usd
        )

        # Update meshdir/texturedir to point to parent directory's meshes folder.
        xml_for_usd = re.sub(r'meshdir="meshes"', 'meshdir="../meshes"', xml_for_usd)
        xml_for_usd = re.sub(
            r'texturedir="meshes"', 'texturedir="../meshes"', xml_for_usd
        )

        # Write temporary XML for USD conversion.
        usd_xml_path = usd_dir / "scene_for_usd.xml"
        with open(usd_xml_path, "w") as f:
            f.write(xml_for_usd)

        # Use mujoco-usd-converter for proper USD export with meshes.
        # Requires mujoco==3.3.5 and mujoco-usd-converter==0.1.0a3.
        converter = mujoco_usd_converter.Converter()
        asset = converter.convert(str(usd_xml_path), str(usd_dir))

        # Open stage and save with comment.
        stage: Usd.Stage = Usd.Stage.Open(asset.path)
        usdex.core.saveStage(
            stage, comment="Exported from scenesmith via mujoco-usd-converter"
        )

        # Fix texture wrap modes for proper tiling support.
        materials_lib_path = usd_dir / "Payload" / "MaterialsLibrary.usdc"
        if materials_lib_path.exists():
            fix_usd_texture_wrap_modes(materials_lib_path)

        # Fix physics for Isaac Sim compatibility.
        physics_path = usd_dir / "Payload" / "Physics.usda"
        if apply_isaac_sim_fix and physics_path.exists():
            from fix_usd_isaac_sim import fix_physics_layer

            fix_physics_layer(physics_path)

        # Clean up temporary XML.
        usd_xml_path.unlink()

        console_logger.info(f"  USD exported to: {asset.path}")
        console_logger.info(f"  USD payloads in: {usd_dir / 'Payload'}")
    except ImportError as e:
        console_logger.error(f"USD export requires mujoco-usd-converter: {e}")
    except Exception as e:
        console_logger.error(f"USD export failed: {e}")


def validate_mujoco_export(output_path: Path) -> bool:
    """Validate that the exported MJCF loads successfully in MuJoCo.

    Args:
        output_path: Path to scene.xml file.

    Returns:
        True if validation passed.
    """
    try:
        model = mujoco.MjModel.from_xml_path(str(output_path))
        data = mujoco.MjData(model)

        # Run a few simulation steps.
        for _ in range(10):
            mujoco.mj_step(model, data)

        # Check for NaN.
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            console_logger.error("Simulation produced NaN values")
            return False

        console_logger.info(
            f"Validation passed: {model.nbody} bodies, {model.njnt} joints, "
            f"{model.ngeom} geoms"
        )
        return True
    except Exception as e:
        console_logger.error(f"Validation failed: {e}")
        return False
