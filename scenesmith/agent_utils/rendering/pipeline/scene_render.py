import copy
import logging
import time

import numpy as np

from pydrake.all import (
    ApplyCameraConfig,
    CameraConfig,
    DiagramBuilder,
    RenderEngineGltfClientParams,
    RenderEngineVtkParams,
    Rgba,
    RigidTransform,
    Transform,
)

from scenesmith.agent_utils.physics.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenesmith.agent_utils.scene.room import RoomScene

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.rendering.pipeline.core import (
    setup_virtual_display_if_needed,
)

# Track virtual display for cleanup on process exit.
_virtual_display = None


def render_plant(
    plant,
    scene_graph,
    builder: DiagramBuilder,
    camera_X_WC: RigidTransform,
    camera_width: int = 640,
    camera_height: int = 480,
    background_color: list[float] = [1.0, 1.0, 1.0],
    use_blender_server: bool = False,
    blender_server_url: str = "http://127.0.0.1:8000",
) -> np.ndarray:
    """
    Render a plant and scene graph by adding camera to existing DiagramBuilder.

    Args:
        plant: The MultibodyPlant to render.
        scene_graph: The SceneGraph to render.
        builder: The DiagramBuilder containing the plant and scene_graph.
        camera_X_WC (RigidTransform): The camera pose in the world frame.
        camera_width (int): The camera width.
        camera_height (int): The camera height.
        background_color (list[float]): The background color of the rendered image.
        use_blender_server (bool): Whether to send render requests to a Blender server.
        blender_server_url (str): The URL of the Blender server.

    Returns:
        np.ndarray: The rendered RGBA image. Shape (H, W, 4) where H is the image height
        and W is the image width.
    """
    # Add camera to the existing builder.
    camera_config = CameraConfig(
        X_PB=Transform(camera_X_WC),
        width=camera_width,
        height=camera_height,
        background=Rgba(
            background_color[0], background_color[1], background_color[2], 1.0
        ),
        renderer_class=(
            RenderEngineGltfClientParams(base_url=blender_server_url)
            if use_blender_server
            else RenderEngineVtkParams(backend="GLX")
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

    rgba_image = copy.deepcopy(diagram.GetOutputPort("rgba_image").Eval(context).data)
    return rgba_image


def render_scene(
    scene: RoomScene,
    camera_X_WC: RigidTransform,
    camera_width: int = 640,
    camera_height: int = 480,
    background_color: list[float] = [1.0, 1.0, 1.0],
    use_blender_server: bool = False,
    blender_server_url: str = "http://127.0.0.1:8000",
) -> np.ndarray:
    """
    Render a scene from a given camera pose.

    This function creates the plant and scene graph, then calls render_plant
    to do the actual rendering work.

    Args:
        scene (RoomScene): The scene to render.
        camera_X_WC (RigidTransform): The camera pose in the world frame.
        camera_width (int): The camera width.
        camera_height (int): The camera height.
        background_color (list[float]): The background color of the rendered image.
        use_blender_server (bool): Whether to send render requests to a Blender server.
        blender_server_url (str): The URL of the Blender server.

    Returns:
        np.ndarray: The rendered RGBA image. Shape (H, W, 4) where H is the image height
        and W is the image width.
    """
    # Virtual display only needed for VTK/GLX rendering, not Blender.
    # Xvfb causes 6x slowdown for Blender which runs headless natively.
    if not use_blender_server:
        setup_virtual_display_if_needed()

    start_time = time.time()

    # Create a diagram builder for rendering.
    builder = DiagramBuilder()

    # Create the plant and scene graph in the rendering diagram.
    plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
        scene=scene, builder=builder
    )

    # Render using the plant and scene_graph.
    image = render_plant(
        plant=plant,
        scene_graph=scene_graph,
        builder=builder,
        camera_X_WC=camera_X_WC,
        camera_width=camera_width,
        camera_height=camera_height,
        background_color=background_color,
        use_blender_server=use_blender_server,
        blender_server_url=blender_server_url,
    )

    end_time = time.time()
    console_logger.info(f"Rendered scene in {end_time - start_time:.2f} seconds.")

    return image
