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

import json
import logging
import shutil

from pathlib import Path

import mujoco
import numpy as np
import trimesh

from PIL import Image

from scenesmith.agent_utils.scene.house import HouseScene
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scripts.mujoco_export.mesh_utils import apply_scale_to_trimesh

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def create_mujoco_spec_with_environment(
    model_name: str, ground_collides: bool = False
) -> mujoco.MjSpec:
    """Create MuJoCo spec with skybox and ground plane.

    Args:
        model_name: Name for the model.
        ground_collides: Whether ground plane should have collision.

    Returns:
        Configured MuJoCo spec.
    """
    spec = mujoco.MjSpec()
    spec.modelname = model_name
    spec.compiler.degree = False
    spec.compiler.balanceinertia = True
    spec.compiler.boundmass = 0.001
    spec.compiler.boundinertia = 0.001

    # Visual settings.
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]

    # Skybox.
    skybox = spec.add_texture(name="skybox")
    skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox.rgb1 = [0.3, 0.5, 0.7]
    skybox.rgb2 = [0.0, 0.0, 0.0]
    skybox.width = 512
    skybox.height = 512

    # Ground plane texture and material.
    grid_texture = spec.add_texture(name="grid")
    grid_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid_texture.rgb1 = [0.2, 0.3, 0.4]
    grid_texture.rgb2 = [0.1, 0.2, 0.3]
    grid_texture.width = 512
    grid_texture.height = 512
    grid_texture.mark = mujoco.mjtMark.mjMARK_EDGE
    grid_texture.markrgb = [0.8, 0.8, 0.8]

    grid_material = spec.add_material(name="grid")
    grid_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    grid_material.texrepeat = [10, 10]
    grid_material.reflectance = 0.0

    # Ground plane geom.
    ground = spec.worldbody.add_geom(name="ground_plane")
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [0, 0, 0.05]
    ground.pos = [0, 0, -0.001]
    ground.material = "grid"
    ground.contype = 1 if ground_collides else 0
    ground.conaffinity = 1 if ground_collides else 0

    return spec


def find_gltf_color_texture(gltf_path: Path) -> Path | None:
    """Find color texture referenced by a GLTF file.

    Parses the GLTF JSON to find external texture references.

    Args:
        gltf_path: Path to GLTF file.

    Returns:
        Path to color texture file, or None if not found.
    """
    try:
        with open(gltf_path, "r") as f:
            gltf_data = json.load(f)

        # Find images referenced in the GLTF.
        images = gltf_data.get("images", [])
        if not images:
            return None

        # Look for color texture (usually index 0 in PBR materials).
        # Check materials to find baseColorTexture index.
        materials = gltf_data.get("materials", [])
        color_texture_idx = None
        for mat in materials:
            pbr = mat.get("pbrMetallicRoughness", {})
            base_color = pbr.get("baseColorTexture", {})
            if "index" in base_color:
                # Get texture, then get its source (image index).
                tex_idx = base_color["index"]
                textures = gltf_data.get("textures", [])
                if tex_idx < len(textures):
                    color_texture_idx = textures[tex_idx].get("source", 0)
                    break

        # Default to first image if no specific color texture found.
        if color_texture_idx is None:
            color_texture_idx = 0

        if color_texture_idx < len(images):
            image_uri = images[color_texture_idx].get("uri")
            if image_uri:
                texture_path = gltf_path.parent / image_uri
                if texture_path.exists():
                    return texture_path
                # Try resolving relative path.
                texture_path = (gltf_path.parent / image_uri).resolve()
                if texture_path.exists():
                    return texture_path

        return None
    except Exception as e:
        console_logger.debug(f"Could not parse GLTF for textures: {e}")
        return None


def convert_gltf_to_obj(
    gltf_path: Path,
    obj_path: Path,
    texture_dir: Path | None = None,
    scale: list[float] | None = None,
) -> tuple[bool, Path | None, list[float] | None]:
    """Convert GLTF file to OBJ format using trimesh.

    GLTF uses Y-up coordinate system, OBJ uses Z-up. This function applies
    the necessary rotation during conversion. Optionally applies scale.

    Args:
        gltf_path: Path to input GLTF file.
        obj_path: Path for output OBJ file.
        texture_dir: Directory to save extracted textures.
        scale: Optional [sx, sy, sz] scale factors from SDF.

    Returns:
        Tuple of (success, texture_path or None, base_color_rgba or None).
    """
    try:
        # Use force='mesh' to get a single mesh with UV coordinates preserved.
        # Using Scene + concatenate loses UVs.
        mesh = trimesh.load(gltf_path, force="mesh")

        # Validate mesh has minimum vertices required by MuJoCo.
        if mesh.vertices.shape[0] < 4:
            console_logger.warning(
                f"Mesh {gltf_path.name} has only {mesh.vertices.shape[0]} vertices "
                f"(MuJoCo requires at least 4). Skipping conversion."
            )
            return False, None, None

        # GLTF is Y-up, MuJoCo uses Z-up. Apply the same Y-up to Z-up transform
        # that scenesmith uses for consistency with collision geometry.
        # This is a +90° rotation around X axis: x'=x, y'=-z, z'=y.
        yup_to_zup = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        mesh.apply_transform(yup_to_zup)

        # Apply scale if provided.
        if scale is not None and scale != [1.0, 1.0, 1.0]:
            apply_scale_to_trimesh(mesh, scale)

        mesh.export(obj_path)
        console_logger.info(
            f"Converted {gltf_path.name} -> {obj_path.name} (Y-up -> Z-up)"
        )

        # Extract texture if available.
        texture_path = None
        base_color = None

        if texture_dir:
            # Method 1: Try to find external texture referenced in GLTF.
            external_texture = find_gltf_color_texture(gltf_path)
            if external_texture:
                # MuJoCo only supports PNG textures, so convert if necessary.
                dest_texture = texture_dir / f"{obj_path.stem}_texture.png"
                if not dest_texture.exists():
                    if external_texture.suffix.lower() in (".jpg", ".jpeg"):
                        # Convert JPG to PNG.
                        img = Image.open(external_texture)
                        img.save(dest_texture, "PNG")
                        console_logger.info(
                            f"Converted texture: {external_texture.name} -> "
                            f"{dest_texture.name}"
                        )
                    else:
                        shutil.copy(external_texture, dest_texture)
                        console_logger.info(
                            f"Copied external texture: {dest_texture.name}"
                        )
                texture_path = dest_texture

            # Method 2: Try to extract embedded texture from mesh.
            if texture_path is None:
                try:
                    if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
                        material = mesh.visual.material
                        image = None

                        # Try different ways trimesh stores textures.
                        if hasattr(material, "image") and material.image is not None:
                            image = material.image
                        elif (
                            hasattr(material, "baseColorTexture")
                            and material.baseColorTexture is not None
                        ):
                            image = material.baseColorTexture

                        if image is not None:
                            texture_path = texture_dir / f"{obj_path.stem}_texture.png"
                            image.save(texture_path)
                            console_logger.info(
                                f"Extracted embedded texture: {texture_path.name}"
                            )
                except Exception as tex_err:
                    console_logger.debug(
                        f"Could not extract embedded texture: {tex_err}"
                    )

        # Method 3: Extract base color from PBR material if no texture found.
        if texture_path is None:
            base_color = get_gltf_base_color(gltf_path)

        return True, texture_path, base_color
    except Exception as e:
        console_logger.warning(f"Failed to convert {gltf_path}: {e}")
        return False, None, None


def get_gltf_base_color(gltf_path: Path) -> list[float] | None:
    """Extract base color from GLTF PBR material.

    Args:
        gltf_path: Path to GLTF file.

    Returns:
        RGBA color as [r, g, b, a] or None if not found.
    """
    try:
        with open(gltf_path, "r") as f:
            gltf_data = json.load(f)

        materials = gltf_data.get("materials", [])
        if materials:
            # Get first material's base color.
            mat = materials[0]
            pbr = mat.get("pbrMetallicRoughness", {})
            base_color = pbr.get("baseColorFactor")
            if base_color and len(base_color) >= 3:
                # Ensure we have RGBA.
                if len(base_color) == 3:
                    base_color = base_color + [1.0]
                return base_color
        return None
    except Exception:
        return None


def _remap_room_geometry_paths(state_dict: dict, scene_dir: Path) -> None:
    """Remap stale absolute sdf_path in room_geometry to valid paths.

    Scenes moved after generation have stale absolute paths pointing to the
    original outputs/ directory. The SDF files still exist at the same path
    relative to the scene root (e.g. room_geometry/room_geometry_bedroom.sdf).
    Since RoomGeometry.from_dict resolves paths relative to the room subdir
    (not the scene root), we rewrite the path as an absolute path pointing
    to the actual file location.
    """
    for room_data in state_dict.get("rooms", {}).values():
        rg = room_data.get("room_geometry", {})
        sdf_path_str = rg.get("sdf_path")
        if not sdf_path_str:
            continue
        sdf_path = Path(sdf_path_str)
        if sdf_path.is_absolute() and not sdf_path.exists():
            # Extract the portion after scene_NNN/.
            parts = sdf_path.parts
            for i, part in enumerate(parts):
                if part.startswith("scene_"):
                    relative = str(Path(*parts[i + 1 :]))
                    candidate = scene_dir / relative
                    if candidate.exists():
                        rg["sdf_path"] = str(candidate.resolve())
                        console_logger.debug(
                            f"Remapped sdf_path: {sdf_path_str} -> " f"{candidate}"
                        )
                    break


def load_house_from_directory(scene_dir: Path) -> HouseScene:
    """Load a house scene from a scene directory.

    Args:
        scene_dir: Path to scene directory (e.g., outputs/.../scene_039).

    Returns:
        Reconstructed HouseScene object with all rooms.

    Raises:
        FileNotFoundError: If house_state.json not found.
    """
    house_state_path = scene_dir / "combined_house" / "house_state.json"
    if not house_state_path.exists():
        raise FileNotFoundError(f"House state not found: {house_state_path}")

    with open(house_state_path, "r") as f:
        state_dict = json.load(f)

    # Remap absolute sdf_path values in room_geometry to be relative to the
    # scene directory. Scenes that were moved after generation have stale
    # absolute paths but the files still exist at the same relative location.
    _remap_room_geometry_paths(state_dict, scene_dir)

    # Use HouseScene.from_state_dict to restore the full house.
    house = HouseScene.from_state_dict(state_dict, house_dir=scene_dir.resolve())

    # Log summary.
    total_furniture = 0
    total_manipulands = 0
    for room_id, room in house.rooms.items():
        furniture_count = len(room.get_objects_by_type(ObjectType.FURNITURE))
        manipuland_count = len(room.get_manipulands())
        total_furniture += furniture_count
        total_manipulands += manipuland_count
        console_logger.info(
            f"Room '{room_id}': {furniture_count} furniture, {manipuland_count} manipulands"
        )

    console_logger.info(
        f"Loaded house with {len(house.rooms)} rooms, "
        f"{total_furniture} furniture, {total_manipulands} manipulands"
    )

    return house
