import copy
import logging

from pathlib import Path

import numpy as np
import requests

from omegaconf import DictConfig
from pydrake.all import (
    ApplyCameraConfig,
    CameraConfig,
    DiagramBuilder,
    RenderEngineGltfClientParams,
    Rgba,
    RigidTransform,
    Transform,
)

from scenesmith.agent_utils.blender import BlenderServer
from scenesmith.agent_utils.blender.surfaces.surface_utils import (
    generate_angled_drawer_view,
)
from scenesmith.agent_utils.physics.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
    get_all_link_transforms,
    get_closed_position,
    get_joint_limits,
    get_open_position,
    set_joints_to_config,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    SupportSurface,
)
from scenesmith.utils.geometry.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.rendering.pipeline.core import (
    apply_fk_to_surfaces,
    build_support_surfaces_data,
    compute_drawer_direction,
    get_drake_model_name,
)

# Track virtual display for cleanup on process exit.
_virtual_display = None


def render_per_drawer_views(
    plant,
    context,
    diagram,
    server: BlenderServer,
    drawer_surfaces: dict[str, list[SupportSurface]],
    all_surfaces: list[SupportSurface],
    scene_objects: list[SceneObject],
    link_to_joint: dict[str, str],
    rest_transforms: dict[str, RigidTransform],
    config_payload: dict,
    output_dir: Path,
    cfg: DictConfig,
) -> list[Path]:
    """Render per-drawer angled views with only one drawer open at a time.

    For each drawer joint, resets to rest position, opens only that drawer,
    computes FK transforms, and renders an angled view looking into the drawer.
    Manipulands placed on drawer surfaces are also FK-transformed to move with
    the opened drawer.

    Args:
        plant: Drake MultibodyPlant (already finalized).
        context: Plant context to modify joint positions.
        diagram: Built Drake diagram for evaluation.
        server: Running Blender server.
        drawer_surfaces: Mapping from joint_name to surfaces controlled by that joint.
        all_surfaces: All support surfaces (for surface_id to link_name lookup).
        scene_objects: All scene objects (to find manipulands for FK transform).
        link_to_joint: Mapping from link name to joint name.
        rest_transforms: Link transforms at rest (closed) position.
        config_payload: Base config payload for Blender (will be modified per drawer).
        output_dir: Directory for output images.
        cfg: Rendering config.

    Returns:
        List of paths to rendered drawer view images.
    """
    if not drawer_surfaces:
        return []

    drawer_images = []
    joint_limits = get_joint_limits(plant)

    # Build reverse mapping: joint_name -> link_name.
    joint_to_link = {v: k for k, v in link_to_joint.items()}

    # Build surface_id -> link_name lookup for manipuland FK transforms.
    surface_to_link = {str(s.surface_id): s.link_name for s in all_surfaces}

    config_url = f"{server.get_url()}/set_overlay_config"

    for joint_name, surfaces in drawer_surfaces.items():
        if not surfaces:
            continue

        console_logger.info(f"Rendering per-drawer view for joint: {joint_name}")

        # 1. Reset all joints to rest (closed) position.
        closed_config = {}
        for jname, (lower, upper) in joint_limits.items():
            closed_config[jname] = get_closed_position(lower, upper)
        set_joints_to_config(plant, context, closed_config)

        # 2. Open only this drawer.
        if joint_name in joint_limits:
            lower, upper = joint_limits[joint_name]
            open_pos = get_open_position(lower, upper)
            set_joints_to_config(plant, context, {joint_name: open_pos})

        # 3. Get transforms at this configuration.
        current_transforms = get_all_link_transforms(plant, context)

        # 4. Compute drawer direction from FK delta.
        link_name = joint_to_link.get(joint_name)
        drawer_direction = None
        if (
            link_name
            and link_name in rest_transforms
            and link_name in current_transforms
        ):
            drawer_direction = compute_drawer_direction(
                rest_transforms[link_name],
                current_transforms[link_name],
            )
            console_logger.debug(f"Drawer {joint_name} direction: {drawer_direction}")

        # 5. Apply FK transform to this drawer's surfaces only.
        transformed_surfaces = apply_fk_to_surfaces(
            surfaces=surfaces,
            rest_transforms=rest_transforms,
            open_transforms=current_transforms,
            link_to_joint=link_to_joint,
            open_joints={joint_name},
        )

        # 6. Build surface data for this drawer.
        surfaces_data = build_support_surfaces_data(transformed_surfaces)

        # 7. Generate angled view configuration.
        if surfaces_data:
            view = generate_angled_drawer_view(
                surface=surfaces_data[0],
                joint_name=joint_name,
                drawer_direction=drawer_direction,
            )

            # 8. Build drawer-specific config.
            drawer_config = copy.deepcopy(config_payload)
            drawer_config["support_surfaces"] = surfaces_data

            # Remove context furniture for drawer views - we only want to see the drawer
            # interior, not nearby furniture like beds that would affect camera framing.
            drawer_config.pop("context_furniture_ids", None)

            # 8a. Apply FK transform to scene_objects metadata for manipulands
            # on this drawer. This ensures bounding boxes and labels move with
            # the drawer in the rendered overlay.
            if (
                link_name
                and link_name in rest_transforms
                and link_name in current_transforms
            ):
                delta = (
                    current_transforms[link_name] @ rest_transforms[link_name].inverse()
                )
                for obj_meta in drawer_config.get("scene_objects", []):
                    parent_surface_id = obj_meta.get("parent_surface_id")
                    if not parent_surface_id:
                        continue
                    # Check if this object is on this drawer's surface.
                    obj_link = surface_to_link.get(parent_surface_id)
                    if not obj_link or link_to_joint.get(obj_link) != joint_name:
                        continue
                    # Apply FK transform to position.
                    old_pos = np.array(obj_meta["position"])
                    new_pos = delta @ old_pos
                    obj_meta["position"] = new_pos.tolist()
                    # Apply FK transform to bounding_box center.
                    if obj_meta.get("bounding_box"):
                        old_center = np.array(obj_meta["bounding_box"]["center"])
                        new_center = delta @ old_center
                        obj_meta["bounding_box"]["center"] = new_center.tolist()

            drawer_config["render_single_view"] = {
                "enabled": True,
                "name": view["name"],
                "direction": list(view["direction"]),
            }

            # 9. Send config to Blender.
            response = requests.post(config_url, json=drawer_config, timeout=10)
            if response.status_code != 200:
                console_logger.warning(
                    f"Failed to set drawer config for {joint_name}: "
                    f"{response.status_code} {response.text}"
                )
                continue

            # 10. Eval diagram (triggers Drake to send glTF with current joint positions).
            # Must use the root context that contains our modified plant context.
            root_context = diagram.CreateDefaultContext()
            plant_context = plant.GetMyContextFromRoot(root_context)
            # Re-apply joint configuration to the new context.
            set_joints_to_config(
                plant=plant, context=plant_context, joint_config=closed_config
            )
            set_joints_to_config(
                plant=plant, context=plant_context, joint_config={joint_name: open_pos}
            )

            # 10a. Apply FK transforms to manipulands on this drawer.
            # Objects placed on drawer surfaces at REST stay at REST world positions
            # unless we move them. Compute FK delta and set new poses.
            for obj in scene_objects:
                if obj.object_type != ObjectType.MANIPULAND:
                    continue  # Only transform free bodies (manipulands).
                if obj.placement_info is None:
                    continue

                # Get parent surface ID from placement info.
                parent_surface_id = obj.placement_info.parent_surface_id
                if not parent_surface_id:
                    continue

                # Find which link this surface belongs to.
                obj_link_name = surface_to_link.get(str(parent_surface_id))
                if not obj_link_name:
                    continue

                # Check if this object is on THIS drawer's link.
                obj_joint = link_to_joint.get(obj_link_name)
                if obj_joint != joint_name:
                    continue  # Object not on this drawer.

                # Compute FK delta: open_transform @ rest_transform.inverse().
                if (
                    obj_link_name not in rest_transforms
                    or obj_link_name not in current_transforms
                ):
                    continue
                delta = (
                    current_transforms[obj_link_name]
                    @ rest_transforms[obj_link_name].inverse()
                )
                new_pose = delta @ obj.transform

                # Set new pose in Drake using SetFreeBodyPose.
                try:
                    model_name = get_drake_model_name(obj)
                    base_link_name = extract_base_link_name_from_sdf(obj.sdf_path)
                    model_instance = plant.GetModelInstanceByName(model_name)
                    body = plant.GetBodyByName(base_link_name, model_instance)
                    plant.SetFreeBodyPose(plant_context, body, new_pose)
                    console_logger.debug(
                        f"Applied FK to manipuland {obj.name} on {joint_name}"
                    )
                except Exception as e:
                    console_logger.warning(f"Failed to set FK pose for {obj.name}: {e}")

            # 11. Track existing images before render, then find new image via
            # set difference (same pattern as wall rendering).
            existing_images = set(output_dir.glob("*.png"))

            _ = diagram.GetOutputPort("rgba_image").Eval(root_context)

            current_images = set(output_dir.glob("*.png"))
            new_images = current_images - existing_images

            if new_images:
                new_image = next(iter(new_images))
                drawer_image_name = f"drawer_{joint_name}.png"
                drawer_image_path = output_dir / drawer_image_name
                new_image.rename(drawer_image_path)
                drawer_images.append(drawer_image_path)
                console_logger.info(f"Rendered drawer view: {drawer_image_name}")

    return drawer_images


def render_per_wall_ortho_views(
    scene: "RoomScene",
    server: BlenderServer,
    wall_surfaces: list[dict],
    wall_furniture_map: dict[str, list],
    base_config_payload: dict,
    output_dir: Path,
    cfg: DictConfig,
) -> list[Path]:
    """Render per-wall orthographic views with filtered furniture per wall.

    For each wall, creates a new Drake plant with only furniture near that wall
    and renders an orthographic view facing the wall.

    Args:
        scene: RoomScene containing all objects.
        server: Running Blender server.
        wall_surfaces: List of wall surface dicts with surface_id, wall_id, direction, etc.
        wall_furniture_map: Mapping from surface_id to list of furniture UniqueIDs
            to include in that wall's render.
        base_config_payload: Base config payload for Blender (will be modified per wall).
        output_dir: Directory for output images.
        cfg: Rendering config.

    Returns:
        List of paths to rendered wall orthographic images.
    """
    if not wall_surfaces:
        return []

    wall_images = []
    config_url = f"{server.get_url()}/set_overlay_config"

    for wall_surface in wall_surfaces:
        surface_id = wall_surface.get("surface_id", "unknown")
        wall_id = wall_surface.get("wall_id", "unknown")

        # Get furniture IDs for this wall (keyed by surface_id).
        furniture_ids = wall_furniture_map.get(surface_id, [])

        # Get wall object IDs on this wall.
        wall_object_ids = []
        for obj in scene.objects.values():
            if obj.object_type != ObjectType.WALL_MOUNTED:
                continue
            if obj.placement_info is None:
                continue
            # Check if wall object is on this wall surface (match by surface_id).
            parent_surface_id = str(obj.placement_info.parent_surface_id)
            if parent_surface_id == surface_id:
                wall_object_ids.append(obj.object_id)

        include_objects = furniture_ids + wall_object_ids

        # Build scene objects metadata for this wall's objects only.
        scene_objects_metadata = []
        for obj in scene.objects.values():
            if obj.object_id not in include_objects:
                continue
            translation = obj.transform.translation()
            rotation = obj.transform.rotation()
            rotation_matrix = rotation.matrix().tolist()

            bbox = None
            if obj.bbox_min is not None and obj.bbox_max is not None:
                local_center = (obj.bbox_min + obj.bbox_max) / 2.0
                extents = obj.bbox_max - obj.bbox_min
                world_center = obj.transform @ local_center
                bbox = {"center": world_center.tolist(), "extents": extents.tolist()}

            parent_surface_id = None
            if obj.placement_info:
                parent_surface_id = str(obj.placement_info.parent_surface_id)

            scene_objects_metadata.append(
                {
                    "name": obj.name,
                    "object_id": str(obj.object_id),
                    "object_type": obj.object_type.value,
                    "position": translation.tolist(),
                    "rotation_matrix": rotation_matrix,
                    "bounding_box": bbox,
                    "parent_surface_id": parent_surface_id,
                }
            )

        # Create per-wall config payload with single wall in list.
        wall_config = base_config_payload.copy()
        wall_config["layout"] = "wall_orthographic"
        wall_config["wall_surfaces"] = [wall_surface]
        wall_config["wall_surfaces_for_labels"] = [wall_surface]
        wall_config["scene_objects"] = scene_objects_metadata

        # Set config on Blender server.
        response = requests.post(config_url, json=wall_config, timeout=10)
        if response.status_code != 200:
            console_logger.error(
                f"Failed to set wall config for {wall_id}: {response.text}"
            )
            continue

        # Create new Drake plant with only this wall's objects.
        # Wall rendering always includes room geometry (walls are needed).
        builder = DiagramBuilder()
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
            scene=scene,
            builder=builder,
            include_objects=include_objects,
            exclude_room_geometry=False,
        )

        # Placeholder camera (Blender handles actual camera).
        placeholder_pose = RigidTransform.Identity()
        camera_config = CameraConfig(
            X_PB=Transform(placeholder_pose),
            width=4,
            height=4,
            background=Rgba(
                cfg.background_color[0],
                cfg.background_color[1],
                cfg.background_color[2],
                1.0,
            ),
            renderer_class=RenderEngineGltfClientParams(
                base_url=server.get_url(),
                render_endpoint="render_overlay",
            ),
        )

        ApplyCameraConfig(
            config=camera_config,
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
        )

        builder.ExportOutput(
            builder.GetSubsystemByName(
                f"rgbd_sensor_{camera_config.name}"
            ).color_image_output_port(),
            "rgba_image",
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        # Track existing images before rendering.
        existing_images = set(output_dir.glob("*.png"))

        # Trigger render.
        _ = diagram.GetOutputPort("rgba_image").Eval(context)

        # Find the newly created image by diffing with existing.
        current_images = set(output_dir.glob("*.png"))
        new_images = current_images - existing_images

        if new_images:
            # Should be exactly one new image.
            new_image = next(iter(new_images))
            wall_image_name = f"wall_{wall_id}_ortho.png"
            wall_image_path = output_dir / wall_image_name
            new_image.rename(wall_image_path)
            wall_images.append(wall_image_path)
            console_logger.info(f"Rendered wall ortho view: {wall_image_name}")
        else:
            console_logger.error(
                f"No new image found after rendering wall {wall_id}. "
                f"Existing: {len(existing_images)}, Current: {len(current_images)}"
            )

    return wall_images
