"""Architectural wall and floor GLTF construction."""

from pathlib import Path
from typing import Literal

from scenesmith.utils.geometry.gltf_generation import _create_box_mesh_gltf
from scenesmith.utils.geometry.material import Material


def create_wall_gltf(
    length: float,
    height: float,
    thickness: float,
    material: Material,
    output_path: Path,
    texture_scale: float = 0.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    plane: Literal["XZ", "YZ"] = "XZ",
) -> None:
    """
    Create a 3D box wall mesh with PBR material.

    Args:
        length: Horizontal length dimension in meters.
        height: Vertical height dimension in meters.
        thickness: Wall thickness in meters.
        material: PBR material with textures.
        output_path: Where to save the GLTF file.
        texture_scale: Meters per texture tile (default: 0.5m).
        center_x: X position of wall center in Drake Z-up coordinates (default: 0.0).
        center_y: Y position of wall center in Drake Z-up coordinates (default: 0.0).
        center_z: Z position of wall center in Drake Z-up coordinates (default: 0.0).
        plane: "XZ" for front/back walls, "YZ" for left/right walls (default: "XZ").
    """
    if plane == "XZ":
        # Front/back walls: extend along X (length), thin in Y (thickness), tall in Z
        # (height).
        _create_box_mesh_gltf(
            width=length,
            depth=thickness,
            height=height,
            material=material,
            output_path=output_path,
            texture_scale=texture_scale,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
        )
    elif plane == "YZ":
        # Left/right walls: thin in X (thickness), extend along Y (length), tall in Z
        # (height).
        _create_box_mesh_gltf(
            width=thickness,
            depth=length,
            height=height,
            material=material,
            output_path=output_path,
            texture_scale=texture_scale,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
        )
    else:
        raise ValueError(f"Invalid plane: {plane}. Must be 'XZ' or 'YZ'.")


def create_floor_gltf(
    width: float,
    depth: float,
    thickness: float,
    material: Material,
    output_path: Path,
    texture_scale: float = 0.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> None:
    """
    Create a 3D box floor mesh with PBR material.

    Args:
        width: X dimension in meters.
        depth: Y dimension in meters.
        thickness: Floor thickness in meters (Z dimension).
        material: PBR material with textures.
        output_path: Where to save the GLTF file.
        texture_scale: Meters per texture tile (default: 0.5m).
        center_x: X position of floor center in Drake Z-up coordinates (default: 0.0).
        center_y: Y position of floor center in Drake Z-up coordinates (default: 0.0).
        center_z: Z position of floor center in Drake Z-up coordinates (default: 0.0).
    """
    _create_box_mesh_gltf(
        width=width,
        depth=depth,
        height=thickness,
        material=material,
        output_path=output_path,
        texture_scale=texture_scale,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
    )
