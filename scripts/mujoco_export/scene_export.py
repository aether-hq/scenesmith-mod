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
import os
import re

from pathlib import Path

import mujoco
import numpy as np

from pydrake.all import Quaternion, RigidTransform, RotationMatrix

from scenesmith.agent_utils.physics.drake_utils import create_plant_from_dmd
from scenesmith.agent_utils.scene.house import HouseScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)

console_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from scripts.mujoco_export.mesh_conversion import create_mujoco_spec_with_environment
from scripts.mujoco_export.mesh_utils import (
    drop_bad_collision_mesh_from_spec,
    get_bad_mesh_name_from_compile_error,
)
from scripts.mujoco_export.model_processing import process_sdf_model
from scripts.mujoco_export.scene_io import (
    expand_composite_to_members,
    get_dmd_reference_link_name,
    get_model_directives,
    get_sdf_link_poses_and_roots,
    get_weld_directives_by_model,
    get_welded_models,
    infer_is_furniture_from_sdf_path,
    infer_room_id_from_scene_asset_path,
    parse_dmd_yaml,
    parse_sdf_with_drake_namespace,
    resolve_scene_file_uri,
)

# SDFormat to MuJoCo joint type mapping.
SDF_TO_MJCF_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Unlimited rotation.
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    # "fixed" joints are handled by not creating a joint (weld to parent).
}


def export_scene_to_mujoco(
    house: HouseScene,
    output_dir: Path,
    include_floor_plan: bool = True,
    weld_furniture: bool = False,
) -> Path:
    """Export house scene to self-contained MuJoCo directory.

    Creates a directory with:
    - scene.xml: Main MJCF file
    - meshes/: All referenced mesh files (converted to OBJ if necessary)

    Args:
        house: HouseScene object to export (contains all rooms).
        output_dir: Output directory path.
        include_floor_plan: Whether to include floor plan objects.
        weld_furniture: Whether to weld furniture (make static). Default False
            means furniture has freejoints and can move/fall.

    Returns:
        Path to the exported scene.xml file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    output_path = output_dir / "scene.xml"

    # Create MuJoCo spec with environment (skybox, ground).
    spec = create_mujoco_spec_with_environment(
        model_name=f"scene_{house.house_dir.name}",
        ground_collides=False,  # Scene has its own floor.
    )

    # Track mesh, texture, and color assets.
    mesh_assets: dict[str, str] = {}
    texture_assets: dict[str, str] = {}
    color_assets: dict[str, list[float]] = {}

    # Parse DMD to get welded models (should be static in MuJoCo).
    # Use house.dmd.yaml which only welds room geometry (floors, walls).
    # Furniture and manipulands keep their freejoints.
    dmd_path = house.house_dir / "combined_house" / "house.dmd.yaml"
    if dmd_path.exists():
        directives = parse_dmd_yaml(dmd_path)
        welded_models = get_welded_models(directives)
        console_logger.info(
            f"Found {len(welded_models)} welded models in " f"{dmd_path.name}"
        )
    else:
        welded_models = set()
        console_logger.debug("No DMD file found, using heuristic welding")

    # Track body names for articulated models (for self-collision filtering).
    articulated_model_bodies: list[list[str]] = []

    def process_scene_object(
        obj: SceneObject, is_static: bool, room_id: str = ""
    ) -> list[str]:
        """Process a SceneObject and return list of body names created."""
        if not obj.sdf_path or not obj.sdf_path.exists():
            console_logger.warning(f"SDF not found for {obj.name}: {obj.sdf_path}")
            return []

        # Parse SDF to get model name for unique naming.
        tree = parse_sdf_with_drake_namespace(obj.sdf_path)
        model_elem = tree.getroot().find(".//model")
        if model_elem is None:
            console_logger.warning(f"No model element in SDF: {obj.sdf_path}")
            return []

        # Use room_id and full object_id to ensure unique model names. Different
        # objects can share the same SDF model name, so the object_id is needed.
        room_prefix = f"{room_id}_" if room_id else ""
        model_name = f"{room_prefix}{obj.object_id}"

        # Extract transform.
        translation = obj.transform.translation()
        rotation = obj.transform.rotation().ToQuaternion()

        return process_sdf_model(
            sdf_path=obj.sdf_path,
            sdf_dir=obj.sdf_path.parent,
            model_name=model_name,
            transform_pos=[
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
            ],
            transform_quat=[
                float(rotation.w()),
                float(rotation.x()),
                float(rotation.y()),
                float(rotation.z()),
            ],
            is_static=is_static,
            spec=spec,
            meshes_dir=meshes_dir,
            mesh_assets=mesh_assets,
            texture_assets=texture_assets,
            color_assets=color_assets,
            room_id=room_id,
        )

    # Process all rooms in the house.
    for room_id, room in house.rooms.items():
        # Get room position offset for multi-room scenes.
        room_offset_x, room_offset_y = house._get_room_position(room_id)
        room_offset = np.array([room_offset_x, room_offset_y, 0.0])

        # Add floor plan (static).
        # The room_geometry.sdf contains both floor and walls as a single model.
        if include_floor_plan and room.room_geometry and room.room_geometry.sdf_path:
            sdf_path = room.room_geometry.sdf_path
            if sdf_path.exists():
                # Create a pseudo SceneObject for the room geometry.
                room_geometry_obj = SceneObject(
                    object_id=UniqueID(f"room_geometry_{room_id}"),
                    object_type=ObjectType.FLOOR,
                    name=f"room_geometry_{room_id}",
                    description=f"Room geometry for {room_id}",
                    transform=RigidTransform(p=room_offset),
                    sdf_path=sdf_path,
                )
                process_scene_object(room_geometry_obj, is_static=True, room_id=room_id)

        # Add furniture and manipulands with room offset applied.
        for obj in room.objects.values():
            if obj.object_type in (ObjectType.WALL, ObjectType.FLOOR):
                continue

            # Handle composite objects by expanding into member components.
            composite_type = obj.metadata.get("composite_type")
            if composite_type in ("stack", "pile", "filled_container"):
                members = expand_composite_to_members(obj, room_offset)
                # Use parent object_id directly for uniqueness.
                parent_id = str(obj.object_id)
                for idx, (sdf_path, transform, name) in enumerate(members):
                    if not sdf_path.exists():
                        console_logger.warning(
                            f"SDF not found for composite member: {sdf_path}"
                        )
                        continue
                    # Create pseudo SceneObject for member with unique ID.
                    # Use format that puts unique suffix at end for id_suffix extraction.
                    # Include both parent_id and member index for uniqueness.
                    unique_suffix = f"{parent_id}m{idx}"
                    member_obj = SceneObject(
                        object_id=UniqueID(unique_suffix),
                        object_type=ObjectType.MANIPULAND,
                        name=f"{name}_m{idx}",
                        description=f"Member of {obj.name}",
                        transform=transform,
                        sdf_path=sdf_path,
                    )
                    process_scene_object(member_obj, is_static=False, room_id=room_id)
                continue  # Don't process composite container itself.

            # Determine if object should be static (no freejoint).
            # DMD welding is the source of truth for which objects are welded.
            # The weld_furniture flag is an override for all furniture.
            # Model name format must match room.py:1413-1416.
            id_suffix = str(obj.object_id).split("_")[-1][:8]
            dmd_model_name = (
                f"{room_id}_{obj.name.lower().replace(' ', '_')}_{id_suffix}"
            )
            is_welded_in_dmd = dmd_model_name in welded_models

            should_be_static = is_welded_in_dmd or (
                weld_furniture and obj.object_type == ObjectType.FURNITURE
            )

            # Apply room offset to object transform.
            obj_translation = obj.transform.translation() + room_offset
            obj_with_offset = SceneObject(
                object_id=obj.object_id,
                object_type=obj.object_type,
                name=obj.name,
                description=obj.description,
                transform=RigidTransform(R=obj.transform.rotation(), p=obj_translation),
                sdf_path=obj.sdf_path,
                geometry_path=obj.geometry_path,
                image_path=obj.image_path,
                support_surfaces=obj.support_surfaces,
                placement_info=obj.placement_info,
                metadata=obj.metadata,
            )
            body_names = process_scene_object(
                obj_with_offset, is_static=should_be_static, room_id=room_id
            )
            # Track articulated models (multi-link with joints) for
            # self-collision filtering.
            if len(body_names) > 1:
                articulated_model_bodies.append(body_names)

    add_articulated_self_collision_exclusions(spec, articulated_model_bodies)
    add_mujoco_assets_to_spec(spec, mesh_assets, texture_assets)
    return compile_and_write_mjcf(
        spec=spec,
        output_path=output_path,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
    )


def export_dmd_scene_to_mujoco(
    scene_dir: Path,
    dmd_path: Path,
    output_dir: Path,
    include_floor_plan: bool = True,
    weld_furniture: bool = False,
) -> Path:
    """Export a scene directly from house.dmd.yaml plus referenced SDF assets.

    This is the clean fallback for archived scenes that still contain the
    authoritative Drake directives and packaged SDF assets but no longer keep
    house_state.json metadata around.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    output_path = output_dir / "scene.xml"

    spec = create_mujoco_spec_with_environment(
        model_name=f"scene_{scene_dir.name}",
        ground_collides=False,
    )

    mesh_assets: dict[str, str] = {}
    texture_assets: dict[str, str] = {}
    color_assets: dict[str, list[float]] = {}
    articulated_model_bodies: list[list[str]] = []

    directives = parse_dmd_yaml(dmd_path)
    model_directives = get_model_directives(directives)
    welded_models = get_welded_models(directives)
    weld_directives = get_weld_directives_by_model(directives)

    console_logger.info(
        f"Loaded {len(model_directives)} model directive(s) from {dmd_path.name}"
    )

    builder, plant, scene_graph = create_plant_from_dmd(dmd_path, scene_dir)
    del builder, scene_graph
    context = plant.CreateDefaultContext()

    for add_model_data in model_directives:
        model_name = add_model_data.get("name")
        file_uri = add_model_data.get("file")
        if not model_name or not file_uri:
            continue

        if not include_floor_plan and model_name.startswith("room_geometry_"):
            continue

        sdf_path = resolve_scene_file_uri(
            file_uri, scene_dir=scene_dir, dmd_dir=dmd_path.parent
        )
        if sdf_path is None or not sdf_path.exists():
            console_logger.warning(f"SDF not found for {model_name}: {file_uri}")
            continue

        room_id = infer_room_id_from_scene_asset_path(scene_dir, sdf_path)

        link_absolute_poses, root_links = get_sdf_link_poses_and_roots(sdf_path)
        weld_data = weld_directives.get(model_name)
        reference_link_name = get_dmd_reference_link_name(
            add_model_data=add_model_data,
            weld_data=weld_data,
            link_absolute_poses=link_absolute_poses,
            root_links=root_links,
        )
        if reference_link_name is None:
            console_logger.warning(
                f"Could not determine reference link for {model_name}: {sdf_path}"
            )
            continue

        try:
            model_instance = plant.GetModelInstanceByName(model_name)
            reference_body = plant.GetBodyByName(reference_link_name, model_instance)
        except RuntimeError as exc:
            console_logger.warning(
                f"Skipping {model_name}; Drake model lookup failed: {exc}"
            )
            continue

        x_wl = plant.EvalBodyPoseInWorld(context, reference_body)
        ref_pos, ref_quat = link_absolute_poses.get(
            reference_link_name, ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        )
        x_ml = RigidTransform(
            RotationMatrix(Quaternion(wxyz=ref_quat)),
            ref_pos,
        )
        x_wm = x_wl @ x_ml.inverse()
        q_wm = x_wm.rotation().ToQuaternion()

        is_static = model_name in welded_models or (
            weld_furniture and infer_is_furniture_from_sdf_path(sdf_path)
        )

        body_names = process_sdf_model(
            sdf_path=sdf_path,
            sdf_dir=sdf_path.parent,
            model_name=model_name,
            transform_pos=[
                float(x_wm.translation()[0]),
                float(x_wm.translation()[1]),
                float(x_wm.translation()[2]),
            ],
            transform_quat=[
                float(q_wm.w()),
                float(q_wm.x()),
                float(q_wm.y()),
                float(q_wm.z()),
            ],
            is_static=is_static,
            spec=spec,
            meshes_dir=meshes_dir,
            mesh_assets=mesh_assets,
            texture_assets=texture_assets,
            color_assets=color_assets,
            room_id=room_id,
        )
        if len(body_names) > 1:
            articulated_model_bodies.append(body_names)

    add_articulated_self_collision_exclusions(spec, articulated_model_bodies)
    add_mujoco_assets_to_spec(spec, mesh_assets, texture_assets)
    return compile_and_write_mjcf(
        spec=spec,
        output_path=output_path,
        meshes_dir=meshes_dir,
        mesh_assets=mesh_assets,
        texture_assets=texture_assets,
    )


def add_articulated_self_collision_exclusions(
    spec: mujoco.MjSpec,
    articulated_model_bodies: list[list[str]],
) -> None:
    """Disable self-collisions within each articulated model."""
    for body_names in articulated_model_bodies:
        for i in range(len(body_names)):
            for j in range(i + 1, len(body_names)):
                exclude = spec.add_exclude()
                exclude.name = f"selfcol_{body_names[i]}_{body_names[j]}"
                exclude.bodyname1 = body_names[i]
                exclude.bodyname2 = body_names[j]


def add_mujoco_assets_to_spec(
    spec: mujoco.MjSpec,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
) -> None:
    """Attach mesh and texture assets to the MuJoCo spec."""
    for mesh_name, mesh_filename in mesh_assets.items():
        mesh = spec.add_mesh(name=mesh_name)
        mesh.file = mesh_filename

    for texture_name, texture_filename in texture_assets.items():
        texture = spec.add_texture(name=texture_name)
        texture.file = texture_filename
        texture.type = mujoco.mjtTexture.mjTEXTURE_2D

        material = spec.add_material(name=texture_name)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name

    console_logger.info(f"Total textures: {len(texture_assets)}")


def compile_and_write_mjcf(
    spec: mujoco.MjSpec,
    output_path: Path,
    meshes_dir: Path,
    mesh_assets: dict[str, str],
    texture_assets: dict[str, str],
) -> Path:
    """Compile an MjSpec and write the final XML next to its mesh assets."""
    original_cwd = os.getcwd()
    try:
        os.chdir(meshes_dir)
        dropped_meshes: set[str] = set()
        while True:
            try:
                spec.compile()
                break
            except ValueError as e:
                err_str = str(e)
                bad_mesh = get_bad_mesh_name_from_compile_error(err_str)
                if bad_mesh is None or bad_mesh in dropped_meshes:
                    raise
                if not drop_bad_collision_mesh_from_spec(spec, bad_mesh, mesh_assets):
                    raise
                dropped_meshes.add(bad_mesh)
        xml_string = spec.to_xml()
    finally:
        os.chdir(original_cwd)

    xml_string = re.sub(
        r"<compiler([^/]*)/\s*>",
        r'<compiler\1 meshdir="meshes" texturedir="meshes"/>',
        xml_string,
    )
    xml_string = re.sub(
        r"<compiler([^/>]*)>",
        r'<compiler\1 meshdir="meshes" texturedir="meshes">',
        xml_string,
    )

    with open(output_path, "w") as f:
        f.write(xml_string)

    console_logger.info(f"Exported MJCF to: {output_path}")
    console_logger.info(f"Mesh assets in: {meshes_dir}")
    console_logger.info(f"Total meshes: {len(mesh_assets)}")
    console_logger.info(f"Total textures: {len(texture_assets)}")

    return output_path
