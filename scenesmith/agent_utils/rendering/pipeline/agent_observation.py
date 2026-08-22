import logging
import tempfile

from pathlib import Path

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
from scenesmith.agent_utils.physics.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
    get_all_link_transforms,
    parse_joint_child_links,
    set_articulated_joints_to_max,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SupportSurface,
)
from scenesmith.utils.geometry.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.rendering.pipeline.core import (
    apply_fk_to_surfaces,
    build_support_surfaces_data,
    classify_surfaces_for_rendering,
    get_drake_model_name,
)
from scenesmith.agent_utils.rendering.pipeline.detailed_views import (
    render_per_drawer_views,
    render_per_wall_ortho_views,
)

# Track virtual display for cleanup on process exit.
_virtual_display = None


def render_scene_for_agent_observation(
    scene: RoomScene,
    cfg: DictConfig,
    blender_server: BlenderServer,
    include_objects: list | None = None,
    exclude_room_geometry: bool = False,
    rendering_mode: str = "furniture",
    support_surfaces: list["SupportSurface"] | None = None,
    show_support_surface: bool = False,
    articulated_open: bool = False,
    wall_surfaces: list[dict] | None = None,
    annotate_object_types: list[str] | None = None,
    wall_surfaces_for_labels: list[dict] | None = None,
    wall_furniture_map: dict[str, list] | None = None,
    room_bounds: tuple[float, float, float, float] | None = None,
    ceiling_height: float | None = None,
    taa_samples: int = 16,
    context_furniture_ids: list | None = None,
    side_view_elevation_degrees: float | None = None,
    side_view_start_azimuth_degrees: float | None = None,
    include_vertical_views: bool = True,
    override_side_view_count: int | None = None,
) -> list[Path]:
    """Render scene with config-driven layout for agent observation.

    This function uses Drake's rendering pipeline with a Blender server backend.
    Drake exports the scene to glTF internally and sends it to the /render_overlay
    endpoint, which saves individual view images to a temporary directory.

    For manipuland mode with multiple support surfaces, generates separate renders
    for each surface with appropriate labels and coordinate markers.

    For wall mode ("wall"), renders context top-down view first, then per-wall
    orthographic views with furniture filtered per wall.

    For ceiling_perspective mode, renders an elevated perspective view showing
    the ceiling plane with furniture context below.

    Args:
        scene: The scene to render.
        cfg: Configuration with layout and dimension settings.
        blender_server: BlenderServer instance for rendering. REQUIRED - forked
            workers cannot safely use embedded bpy due to GPU/OpenGL state
            corruption from fork. The caller owns the server lifecycle.
        include_objects: Optional list of UniqueID objects to include in rendering.
            If provided, only these objects will be rendered. Useful for focused
            rendering (e.g., manipuland agent viewing only current furniture).
        exclude_room_geometry: If True, completely exclude the floor plan from rendering.
            Useful for focused rendering of furniture + manipulands only.
        rendering_mode: Rendering mode - "furniture" for room-scale annotations,
            "manipuland" for surface-focused annotations, "wall" for combined
            context top-down view + per-wall orthographic views, "ceiling_perspective"
            for elevated ceiling view.
        support_surfaces: For manipuland mode, list of SupportSurface objects.
            Each surface generates separate rendering views with appropriate labels
            and coordinate markers filtered to surface convex hull.
        show_support_surface: If True, render green wireframe bbox showing support
            surface bounds for debugging.
        articulated_open: If True, render articulated furniture with doors/drawers
            open (joints at max values). Useful for manipuland placement to show
            internal surfaces.
        wall_surfaces: List of wall surface dicts for wall rendering modes.
            Each dict contains wall_id, direction, length, height, transform,
            and excluded_regions.
        annotate_object_types: Optional list of object types to annotate. If provided,
            only objects of these types get annotations (e.g., ["wall_mounted"] for
            wall_context mode). None means annotate all objects.
        wall_surfaces_for_labels: Wall surfaces for top-down wall labels.
        wall_furniture_map: For wall mode, mapping from surface_id to list of furniture
            UniqueIDs to include in that wall's orthographic render. Required when
            rendering_mode="wall".
        room_bounds: For ceiling_perspective mode, room XY bounds
            (min_x, min_y, max_x, max_y) in meters.
        ceiling_height: For ceiling_perspective mode, ceiling height in meters.
        context_furniture_ids: For manipuland mode, list of furniture IDs to keep
            visible in per-surface top-down renders. These provide spatial context
            for item placement orientation (e.g., chairs around a table).
        side_view_elevation_degrees: Optional elevation angle in degrees for side
            view cameras. Overrides default (30 degrees). Useful for context image
            rendering where different angles work better for different furniture.
        side_view_start_azimuth_degrees: Optional starting azimuth angle in degrees
            for side views. 90 degrees positions camera at +Y (front). Overrides
            default (0 degrees with 45° offset for corner views).
        include_vertical_views: Whether to include pure vertical views (top/bottom).
            Defaults to True. Set to False for angled-only context image rendering.
        override_side_view_count: Optional override for number of side views. If
            provided, overrides cfg.side_view_count. Set to 1 for single angled view.

    Returns:
        List of Paths to rendered PNG files.

    Raises:
        RuntimeError: If BlenderServer is not running or rendering fails.
    """
    # NOTE: Virtual display NOT needed for Blender rendering. Blender runs headless
    # natively and Xvfb causes a 6x slowdown. Only VTK/GLX rendering needs Xvfb.

    # Validate BlenderServer is running.
    # BlenderServer is REQUIRED - forked workers cannot safely use embedded bpy
    # due to GPU/OpenGL state corruption from fork.
    if not blender_server.is_running():
        raise RuntimeError(
            "BlenderServer is not running. Cannot render scene for agent observation. "
            "Forked workers cannot safely use embedded bpy."
        )

    # Create temporary directory for rendered outputs.
    temp_dir = Path(tempfile.mkdtemp(prefix="scene_render_"))
    output_dir = temp_dir / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Configure the server for overlay rendering.
        config_url = f"{blender_server.get_url()}/set_overlay_config"

        # Extract scene object metadata for annotations.
        # Filter by include_objects if provided to avoid rendering clutter.
        scene_objects_metadata = []
        objects_for_metadata = (
            [obj for obj in scene.objects.values() if obj.object_id in include_objects]
            if include_objects is not None
            else scene.objects.values()
        )
        for obj in objects_for_metadata:
            translation = obj.transform.translation()
            rotation = obj.transform.rotation()

            # Get rotation matrix as nested list for JSON serialization.
            rotation_matrix = rotation.matrix().tolist()

            # Get bounding box if available.
            bbox = None
            if obj.bbox_min is not None and obj.bbox_max is not None:
                # Compute center and extents from AABB bounds in local space.
                local_center = (obj.bbox_min + obj.bbox_max) / 2.0
                extents = obj.bbox_max - obj.bbox_min

                # Transform local center to world space.
                world_center = obj.transform @ local_center

                bbox = {
                    "center": world_center.tolist(),
                    "extents": extents.tolist(),
                }

            # Get parent surface ID for manipulands.
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

        # Get annotation config flags.
        # Direct attribute access to fail fast if any field is missing.
        annotations_cfg = cfg.annotations

        # Extract current furniture ID for manipuland mode.
        # In manipuland mode, include_objects[0] is always the current furniture.
        current_furniture_id = None
        if (
            rendering_mode == "manipuland"
            and include_objects is not None
            and len(include_objects) > 0
        ):
            current_furniture_id = str(include_objects[0])

        # Determine layout based on rendering_mode.
        # Wall modes use their own layout; other modes use config layout.
        if rendering_mode in ("wall_orthographic", "wall"):
            layout = rendering_mode
        else:
            layout = cfg.layout

        # Use override_side_view_count if provided, otherwise use config value.
        effective_side_view_count = (
            override_side_view_count
            if override_side_view_count is not None
            else cfg.side_view_count
        )

        config_payload = {
            "output_dir": str(output_dir.absolute()),
            "layout": layout,
            "top_view_width": cfg.top_view_width,
            "top_view_height": cfg.top_view_height,
            "side_view_count": effective_side_view_count,
            "side_view_width": cfg.side_view_width,
            "side_view_height": cfg.side_view_height,
            "scene_objects": scene_objects_metadata,
            "wall_normals": {
                name: normal.tolist()
                for name, normal in scene.room_geometry.wall_normals.items()
            },
            "annotations": {
                "enable_set_of_mark_labels": annotations_cfg.enable_set_of_mark_labels,
                "enable_bounding_boxes": annotations_cfg.enable_bounding_boxes,
                # Disable direction arrows for furniture_selection mode.
                "enable_direction_arrows": (
                    False
                    if rendering_mode == "furniture_selection"
                    else annotations_cfg.enable_direction_arrows
                ),
                "enable_partial_walls": annotations_cfg.enable_partial_walls,
                "rendering_mode": rendering_mode,
                "enable_support_surface_debug": annotations_cfg.enable_support_surface_debug,
                "enable_convex_hull_debug": annotations_cfg.enable_convex_hull_debug,
                "annotate_object_types": annotate_object_types,
                # Disable coordinate grid and frame for furniture_selection mode.
                "enable_coordinate_grid": rendering_mode != "furniture_selection",
                "show_coordinate_frame": rendering_mode != "furniture_selection",
            },
            "current_furniture_id": current_furniture_id,
            "openings": (
                [o.to_dict() for o in scene.room_geometry.openings]
                if scene.room_geometry
                else []
            ),
        }

        # Add wall surfaces for wall rendering modes.
        if wall_surfaces is not None:
            config_payload["wall_surfaces"] = wall_surfaces

        # Add wall surfaces for top-down wall labels.
        if wall_surfaces_for_labels is not None:
            config_payload["wall_surfaces_for_labels"] = wall_surfaces_for_labels

        # Add ceiling parameters for ceiling_perspective mode.
        if room_bounds is not None:
            config_payload["room_bounds"] = list(room_bounds)
        if ceiling_height is not None:
            config_payload["ceiling_height"] = ceiling_height

        # Add TAA samples for EEVEE render quality/speed control.
        config_payload["taa_samples"] = taa_samples

        # Add context furniture IDs for manipuland mode.
        # These furniture objects should remain visible in per-surface top-down views.
        if context_furniture_ids is not None and len(context_furniture_ids) > 0:
            config_payload["context_furniture_ids"] = [
                str(ctx_id) for ctx_id in context_furniture_ids
            ]

        # Add camera angle parameters for context image rendering.
        if side_view_elevation_degrees is not None:
            config_payload["side_view_elevation_degrees"] = side_view_elevation_degrees
        if side_view_start_azimuth_degrees is not None:
            config_payload["side_view_start_azimuth_degrees"] = (
                side_view_start_azimuth_degrees
            )
        config_payload["include_vertical_views"] = include_vertical_views

        # Create Drake plant and diagram FIRST to enable FK transforms.
        # This allows us to query link transforms before and after opening joints.
        builder = DiagramBuilder()
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
            scene=scene,
            builder=builder,
            include_objects=include_objects,
            exclude_room_geometry=exclude_room_geometry,
        )
        console_logger.info(
            f"Drake plant created with {plant.num_model_instances()} model instances"
        )

        # Placeholder camera pose (actual rendering uses configured views).
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
                base_url=blender_server.get_url(),
                render_endpoint="render_overlay",
            ),
        )

        # Apply camera config.
        ApplyCameraConfig(
            config=camera_config,
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
        )

        # Export the color image output.
        builder.ExportOutput(
            builder.GetSubsystemByName(
                f"rgbd_sensor_{camera_config.name}"
            ).color_image_output_port(),
            "rgba_image",
        )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        # Apply FK transforms to support surfaces for articulated objects.
        # When articulated_open=True, we transform surface bounding boxes to match
        # the opened joint positions.
        surfaces_for_rendering = support_surfaces
        if articulated_open:
            # Get REST transforms before opening joints.
            rest_transforms = get_all_link_transforms(plant, plant_context)

            # Open joints.
            set_articulated_joints_to_max(plant, plant_context)

            # Apply FK transforms to support surfaces.
            if support_surfaces is not None and len(support_surfaces) > 0:
                # Get OPEN transforms after opening joints.
                open_transforms = get_all_link_transforms(plant, plant_context)

                # Find the furniture object to get SDF path for link-to-joint mapping.
                # In manipuland mode, include_objects[0] is always the current furniture.
                # Otherwise, find the first articulated furniture in the scene.
                furniture_obj = None
                if include_objects is not None and len(include_objects) > 0:
                    furniture_obj = scene.get_object(include_objects[0])
                else:
                    # Find articulated furniture from scene objects.
                    for obj in scene.objects.values():
                        if obj.metadata.get("is_articulated", False):
                            furniture_obj = obj
                            break

                # Apply FK transforms if we have the furniture's SDF.
                if (
                    furniture_obj is not None
                    and furniture_obj.sdf_path is not None
                    and furniture_obj.metadata.get("is_articulated", False)
                ):
                    link_to_joint = parse_joint_child_links(furniture_obj.sdf_path)
                    if link_to_joint:
                        # All joints are open when using set_articulated_joints_to_max.
                        open_joints = set(link_to_joint.values())
                        surfaces_for_rendering = apply_fk_to_surfaces(
                            surfaces=support_surfaces,
                            rest_transforms=rest_transforms,
                            open_transforms=open_transforms,
                            link_to_joint=link_to_joint,
                            open_joints=open_joints,
                        )
                        console_logger.info(
                            f"Applied FK transforms to {len(surfaces_for_rendering)} "
                            f"support surfaces"
                        )

                        # Apply FK transforms to manipulands on articulated surfaces.
                        # Build surface_id -> link_name lookup.
                        surface_to_link = {
                            str(s.surface_id): s.link_name for s in support_surfaces
                        }

                        # Transform manipulands placed on surfaces of open joints.
                        manipuland_fk_count = 0
                        for obj in scene.objects.values():
                            if obj.object_type != ObjectType.MANIPULAND:
                                continue
                            if obj.placement_info is None:
                                continue

                            parent_surface_id = obj.placement_info.parent_surface_id
                            if not parent_surface_id:
                                continue

                            obj_link_name = surface_to_link.get(str(parent_surface_id))
                            if not obj_link_name:
                                continue

                            # Check if this link has an open joint.
                            obj_joint = link_to_joint.get(obj_link_name)
                            if obj_joint not in open_joints:
                                continue

                            # Compute FK delta and new pose.
                            if (
                                obj_link_name not in rest_transforms
                                or obj_link_name not in open_transforms
                            ):
                                continue
                            delta = (
                                open_transforms[obj_link_name]
                                @ rest_transforms[obj_link_name].inverse()
                            )
                            new_pose = delta @ obj.transform

                            # Set new pose in Drake.
                            try:
                                model_name = get_drake_model_name(obj)
                                base_link_name = extract_base_link_name_from_sdf(
                                    obj.sdf_path
                                )
                                model_instance = plant.GetModelInstanceByName(
                                    model_name
                                )
                                body = plant.GetBodyByName(
                                    base_link_name, model_instance
                                )
                                plant.SetFreeBodyPose(plant_context, body, new_pose)
                                manipuland_fk_count += 1
                            except Exception as e:
                                console_logger.warning(
                                    f"Failed to set FK pose for {obj.name}: {e}"
                                )

                        if manipuland_fk_count > 0:
                            console_logger.info(
                                f"Applied FK transforms to {manipuland_fk_count} "
                                f"manipulands"
                            )

        # Add support surfaces for manipuland mode using FK-transformed surfaces.
        if surfaces_for_rendering is not None and len(surfaces_for_rendering) > 0:
            config_payload["support_surfaces"] = build_support_surfaces_data(
                surfaces_for_rendering
            )

        # Add debug visualization flag.
        config_payload["show_support_surface"] = show_support_surface

        # Send overlay config to Blender server.
        response = requests.post(config_url, json=config_payload, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to set overlay config: {response.status_code} "
                f"{response.text}"
            )
        console_logger.info("Overlay config set on Blender server")

        # Evaluate diagram (triggers Drake to send glTF to /render_overlay).
        # Joints are already opened above if articulated_open=True.
        _ = diagram.GetOutputPort("rgba_image").Eval(context)

        # Collect rendered image paths from output directory.
        image_paths = sorted(output_dir.glob("*.png"))
        if not image_paths:
            raise RuntimeError(f"No images found in {output_dir}")

        console_logger.info(f"Rendered {len(image_paths)} main views successfully")

        # Per-drawer rendering for articulated furniture.
        # After main render, render each drawer separately with only that drawer open.
        if (
            articulated_open
            and support_surfaces is not None
            and len(support_surfaces) > 0
            and furniture_obj is not None
            and link_to_joint
        ):
            # Classify surfaces into static vs per-joint (drawer).
            _, drawer_surfaces = classify_surfaces_for_rendering(
                surfaces=support_surfaces, link_to_joint=link_to_joint
            )

            if drawer_surfaces:
                console_logger.info(
                    f"Rendering {len(drawer_surfaces)} per-drawer views"
                )
                drawer_images = render_per_drawer_views(
                    plant=plant,
                    context=plant_context,
                    diagram=diagram,
                    server=blender_server,
                    drawer_surfaces=drawer_surfaces,
                    all_surfaces=support_surfaces,
                    scene_objects=list(scene.objects.values()),
                    link_to_joint=link_to_joint,
                    rest_transforms=rest_transforms,
                    config_payload=config_payload,
                    output_dir=output_dir,
                    cfg=cfg,
                )
                image_paths.extend(drawer_images)
                console_logger.info(
                    f"Total rendered: {len(image_paths)} views "
                    f"({len(image_paths) - len(drawer_images)} main + "
                    f"{len(drawer_images)} drawer)"
                )

        # Per-wall orthographic rendering for combined wall mode.
        # After context render, render each wall with filtered furniture.
        if (
            rendering_mode == "wall"
            and wall_surfaces is not None
            and len(wall_surfaces) > 0
            and wall_furniture_map is not None
        ):
            console_logger.info(
                f"Rendering {len(wall_surfaces)} per-wall orthographic views"
            )
            wall_images = render_per_wall_ortho_views(
                scene=scene,
                server=blender_server,
                wall_surfaces=wall_surfaces,
                wall_furniture_map=wall_furniture_map,
                base_config_payload=config_payload,
                output_dir=output_dir,
                cfg=cfg,
            )
            image_paths.extend(wall_images)
            console_logger.info(
                f"Total rendered: {len(image_paths)} views "
                f"(1 context + {len(wall_images)} wall ortho)"
            )

        return image_paths

    except Exception as e:
        console_logger.error(f"Failed to render scene: {e}")
        raise
