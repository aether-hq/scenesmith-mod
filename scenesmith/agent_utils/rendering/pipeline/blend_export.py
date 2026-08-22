import logging

from pathlib import Path

import requests

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
    create_plant_from_dmd,
)
from scenesmith.agent_utils.scene.room import RoomScene

console_logger = logging.getLogger(__name__)


# Track virtual display for cleanup on process exit.
_virtual_display = None


def save_scene_as_blend(
    scene: RoomScene,
    output_path: Path,
    blender_server_host: str = "127.0.0.1",
    blender_server_port_range: tuple[int, int] = (8000, 8050),
    server_startup_delay: float = 0.1,
    port_cleanup_delay: float = 0.1,
) -> Path:
    """Export scene to a .blend file.

    Uses Drake to export scene to glTF, then Blender server imports and saves as .blend.

    Args:
        scene: The scene to export.
        output_path: Path where .blend file will be saved.
        blender_server_host: Host address for the Blender server.
        blender_server_port_range: Port range for the Blender server.
        server_startup_delay: Delay after starting server subprocess.
        port_cleanup_delay: Delay after stopping server.

    Returns:
        Path to the saved .blend file.

    Raises:
        RuntimeError: If Blender server fails or export fails.
    """
    # NOTE: Virtual display NOT needed for Blender. Blender runs headless natively.

    console_logger.info(f"Exporting scene to .blend file: {output_path}")

    server = BlenderServer(
        host=blender_server_host,
        port_range=blender_server_port_range,
        server_startup_delay=server_startup_delay,
        port_cleanup_delay=port_cleanup_delay,
    )
    server.start()
    server.wait_until_ready()

    try:
        # Configure server for blend export.
        config_url = f"{server.get_url()}/set_blend_config"
        config_payload = {"output_path": str(output_path.absolute())}

        response = requests.post(config_url, json=config_payload, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to set blend config: {response.text}")

        # Create Drake diagram to export glTF.
        builder = DiagramBuilder()
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(
            scene=scene,
            builder=builder,
            include_objects=None,
            exclude_room_geometry=False,
        )

        # Use minimal camera config (just to trigger glTF export).
        placeholder_pose = RigidTransform.Identity()
        camera_config = CameraConfig(
            X_PB=Transform(placeholder_pose),
            width=4,
            height=4,
            background=Rgba(1.0, 1.0, 1.0, 1.0),
            renderer_class=RenderEngineGltfClientParams(
                base_url=server.get_url(),
                render_endpoint="save_blend",
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

        # Trigger glTF export to /save_blend endpoint.
        _ = diagram.GetOutputPort("rgba_image").Eval(context)

        if not output_path.exists():
            raise RuntimeError(f"Blend file was not created at {output_path}")

        console_logger.info(f"Successfully saved .blend file to {output_path}")
        return output_path

    finally:
        if server.is_running():
            server.stop()


def save_directive_as_blend(
    directive_path: Path,
    output_path: Path,
    blender_server_host: str = "127.0.0.1",
    blender_server_port_range: tuple[int, int] = (8000, 8050),
    server_startup_delay: float = 0.1,
    port_cleanup_delay: float = 0.1,
    scene_dir: Path | None = None,
    max_retries: int = 3,
    additional_visuals: list[dict[str, object]] | None = None,
) -> Path:
    """Export a Drake model directive to a .blend file.

    Loads a Drake model directive YAML file and exports all models to a .blend file.
    Automatically retries with a fresh Blender server if the export fails.

    Args:
        directive_path: Path to the Drake model directive YAML file.
        output_path: Path where .blend file will be saved.
        blender_server_host: Host address for the Blender server.
        blender_server_port_range: Port range for the Blender server.
        server_startup_delay: Delay after starting server subprocess.
        port_cleanup_delay: Delay after stopping server.
        scene_dir: Optional scene root directory for package:// URI resolution.
            If not provided, searches parent directories for package.xml.
        max_retries: Maximum number of retry attempts if export fails.

    Returns:
        Path to the saved .blend file.

    Raises:
        RuntimeError: If Blender server fails or export fails after all retries.
        FileNotFoundError: If directive_path does not exist.
    """
    # NOTE: Virtual display NOT needed for Blender. Blender runs headless natively.

    console_logger.info(f"Exporting directive to .blend file: {output_path}")

    for attempt in range(max_retries):
        server = BlenderServer(
            host=blender_server_host,
            port_range=blender_server_port_range,
            server_startup_delay=server_startup_delay,
            port_cleanup_delay=port_cleanup_delay,
        )
        server.start()
        server.wait_until_ready()

        try:
            # Configure server for blend export.
            config_url = f"{server.get_url()}/set_blend_config"
            config_payload = {
                "output_path": str(output_path.absolute()),
                "additional_visuals": additional_visuals or [],
            }

            response = requests.post(config_url, json=config_payload, timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to set blend config: {response.text}")

            # Create Drake plant from directive.
            builder, plant, scene_graph = create_plant_from_dmd(
                directive_path, scene_dir=scene_dir
            )

            # Use minimal camera config (just to trigger glTF export).
            placeholder_pose = RigidTransform.Identity()
            camera_config = CameraConfig(
                X_PB=Transform(placeholder_pose),
                width=4,
                height=4,
                background=Rgba(1.0, 1.0, 1.0, 1.0),
                renderer_class=RenderEngineGltfClientParams(
                    base_url=server.get_url(),
                    render_endpoint="save_blend",
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

            # Trigger glTF export to /save_blend endpoint.
            _ = diagram.GetOutputPort("rgba_image").Eval(context)

            if output_path.exists():
                console_logger.info(f"Successfully saved .blend file to {output_path}")
                return output_path

            # File not created - server likely crashed during export.
            if attempt < max_retries - 1:
                console_logger.warning(
                    f"Blend export failed (file not created), "
                    f"retrying ({attempt + 1}/{max_retries})"
                )
                continue

            raise RuntimeError(f"Blend file was not created at {output_path}")

        finally:
            if server.is_running():
                server.stop()
