import logging
import math

from pathlib import Path

import bpy
import numpy as np

from mathutils import Matrix, Vector

from scenesmith.agent_utils.blender.params import RenderParams

console_logger = logging.getLogger(__name__)

# Rendering constants.
DEFAULT_LIGHT_ENERGY = 1000
DEFAULT_LIGHT_POSITION = (4.0, 1.0, 6.0)
DEFAULT_NUM_SIDE_VIEWS = 4
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
# EEVEE TAA samples for asset validation renders. Using 8 as a good balance
# between quality and speed. EEVEE is ~6x faster than CYCLES.
EEVEE_ASSET_VALIDATION_SAMPLES = 8
# CYCLES samples for offline CLIP embedding renders (higher quality, slower).
CYCLES_CLIP_SAMPLES = 20
VLM_ANALYSIS_LIGHT_ENERGY = 2000
# Lower light energy for articulated objects (more reflective materials).
ARTICULATED_LIGHT_ENERGY = 500
# Lower light energy for material/texture validation (avoid washing out colors).
MATERIAL_VALIDATION_LIGHT_ENERGY = 300

# Camera constants.
DEFAULT_CAMERA_LENS_MM = 50
DEFAULT_CAMERA_SENSOR_WIDTH_MM = 36
DEFAULT_CAMERA_CLIP_START = 0.01
DEFAULT_CAMERA_CLIP_END = 100000
CAMERA_DISTANCE_MARGIN_MULTIPLIER = (
    1 / 0.8
)  # Scene occupies ~80% of image (10% margin per side).
LIGHT_DISTANCE_RATIO = 0.1
# Offset above lower surfaces for camera near-plane clipping (meters).
# This clips furniture geometry above lower surfaces so they're visible from top-down views.
LOWER_SURFACE_CLIP_OFFSET_M = 0.05

# Multi-view rendering constants.
COORDINATE_FRAME_SCALE_FACTOR = 0.01

from scenesmith.agent_utils.blender.renderer_core import (
    _compute_wall_center_from_transform,
)


class WallViewRenderingMixin:
    """Wall camera setup, debug geometry, and blend-file persistence."""

    def _add_wall_camera_debug_cones(self, wall_surfaces: list[dict]) -> None:
        """Add debug cones showing camera positions and directions for wall views.

        Creates colored cones at each camera position, pointing toward wall centers.
        Colors: North=Red, South=Green, East=Blue, West=Yellow.

        Args:
            wall_surfaces: List of wall surface dicts with direction, length, height,
            transform.
        """
        colors = {
            "north": (1.0, 0.0, 0.0, 1.0),  # Red
            "south": (0.0, 1.0, 0.0, 1.0),  # Green
            "east": (0.0, 0.0, 1.0, 1.0),  # Blue
            "west": (1.0, 1.0, 0.0, 1.0),  # Yellow
        }

        for wall_surface in wall_surfaces:
            wall_direction = wall_surface.get("direction", "north").lower()
            wall_length = wall_surface.get("length", 5.0)
            wall_height = wall_surface.get("height", 2.5)
            transform = wall_surface.get("transform", [0, 0, 0, 1, 0, 0, 0])

            # Parse transform.
            wall_origin = np.array([transform[0], transform[1], transform[2]])
            qw, qx, qy, qz = transform[3], transform[4], transform[5], transform[6]

            # Build rotation matrix from quaternion.
            rotation_matrix = np.array(
                [
                    [
                        1 - 2 * (qy**2 + qz**2),
                        2 * (qx * qy - qw * qz),
                        2 * (qx * qz + qw * qy),
                    ],
                    [
                        2 * (qx * qy + qw * qz),
                        1 - 2 * (qx**2 + qz**2),
                        2 * (qy * qz - qw * qx),
                    ],
                    [
                        2 * (qx * qz - qw * qy),
                        2 * (qy * qz + qw * qx),
                        1 - 2 * (qx**2 + qy**2),
                    ],
                ]
            )

            # Compute wall center.
            wall_center_local = np.array([wall_length / 2, 0, wall_height / 2])
            wall_center_world = wall_origin + rotation_matrix @ wall_center_local

            # Compute camera position.
            camera_offset = 3.0
            if wall_direction == "north":
                camera_pos = np.array(
                    [
                        wall_center_world[0],
                        wall_center_world[1] - camera_offset,
                        wall_center_world[2],
                    ]
                )
            elif wall_direction == "south":
                camera_pos = np.array(
                    [
                        wall_center_world[0],
                        wall_center_world[1] + camera_offset,
                        wall_center_world[2],
                    ]
                )
            elif wall_direction == "east":
                camera_pos = np.array(
                    [
                        wall_center_world[0] - camera_offset,
                        wall_center_world[1],
                        wall_center_world[2],
                    ]
                )
            elif wall_direction == "west":
                camera_pos = np.array(
                    [
                        wall_center_world[0] + camera_offset,
                        wall_center_world[1],
                        wall_center_world[2],
                    ]
                )
            else:
                continue

            # Log positions for debugging.
            console_logger.info(
                f"DEBUG CONE {wall_direction}: wall_origin={wall_origin}, "
                f"wall_center={wall_center_world}, camera_pos={camera_pos}"
            )

            # Create cone mesh.
            bpy.ops.mesh.primitive_cone_add(
                radius1=0.15,
                radius2=0.0,
                depth=0.4,
                location=(camera_pos[0], camera_pos[1], camera_pos[2]),
            )
            cone = bpy.context.active_object
            cone.name = f"debug_camera_{wall_direction}"

            # Point cone toward wall center.
            direction = Vector(wall_center_world.tolist()) - Vector(camera_pos.tolist())
            if direction.length > 0:
                direction.normalize()
                quat = direction.to_track_quat("-Z", "Z")
                cone.rotation_euler = quat.to_euler()

            # Apply color.
            mat = bpy.data.materials.new(name=f"debug_mat_{wall_direction}")
            mat.use_nodes = False
            mat.diffuse_color = colors.get(wall_direction, (1.0, 1.0, 1.0, 1.0))
            cone.data.materials.append(mat)

            self._debug_camera_objects.append(cone)

    def _remove_wall_camera_debug_cones(self) -> None:
        """Remove debug camera cone objects."""
        for obj in self._debug_camera_objects:
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        self._debug_camera_objects = []

    def _setup_wall_orthographic_camera(
        self,
        camera_obj: bpy.types.Object,
        wall_surface: dict,
        margin_factor: float = 1.1,
    ) -> None:
        """Configure orthographic camera facing wall center.

        Sets up an orthographic camera perpendicular to the wall surface,
        positioned inside the room looking at the wall. Uses wall direction
        to compute camera position directly in world coordinates.

        Args:
            camera_obj: Blender camera object to configure.
            wall_surface: Wall surface data dict containing:
                - direction: Wall direction ("north", "south", "east", "west").
                - length: Wall length in meters.
                - height: Wall height in meters.
                - transform: [x, y, z, qw, qx, qy, qz] pose in world frame.
            direction: Camera viewing direction from view generator.
            margin_factor: Scale factor for orthographic view (1.0 = exact fit).
        """
        # Fail fast if required fields are missing (research codebase principle).
        if "direction" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'direction' field. Got: {wall_surface}"
            )
        if "transform" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'transform' field. Got: {wall_surface}"
            )
        if "length" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'length' field. Got: {wall_surface}"
            )
        if "height" not in wall_surface:
            raise ValueError(
                f"wall_surface missing required 'height' field. Got: {wall_surface}"
            )

        wall_length = wall_surface["length"]
        wall_height = wall_surface["height"]
        wall_direction = wall_surface["direction"]
        transform = wall_surface["transform"]
        wall_id = wall_surface.get("wall_id", "unknown")

        # Debug: log wall data for each wall.
        console_logger.info(
            f"Wall camera setup for {wall_id}:\n"
            f"  direction={wall_direction}\n"
            f"  length={wall_length}, height={wall_height}\n"
            f"  transform={transform}"
        )

        # Set camera to orthographic mode.
        camera_obj.data.type = "ORTHO"

        # Orthographic scale = max dimension * margin to fit entire wall.
        camera_obj.data.ortho_scale = max(wall_length, wall_height) * margin_factor

        # Compute wall center from transform.
        wall_center_world = _compute_wall_center_from_transform(
            transform=transform, wall_length=wall_length, wall_height=wall_height
        )
        wall_origin = np.array(transform[:3])
        wall_dir = wall_direction.lower()

        # Camera offset from wall center (inside room, looking at wall).
        # Use room_depth if available to avoid placing camera outside small rooms.
        room_depth = wall_surface.get("room_depth")
        if room_depth is not None and room_depth > 0:
            # Position camera at 80% of room depth, capped at 3m.
            # This ensures camera stays inside the room with some margin.
            camera_offset = min(room_depth * 0.8, 3.0)
        else:
            camera_offset = 3.0  # Default distance from wall center.

        # Compute camera position and look direction based on wall direction.
        if wall_dir == "north":
            camera_pos = np.array(
                [
                    wall_center_world[0],
                    wall_center_world[1] - camera_offset,
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((0, 1, 0))
        elif wall_dir == "south":
            camera_pos = np.array(
                [
                    wall_center_world[0],
                    wall_center_world[1] + camera_offset,
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((0, -1, 0))
        elif wall_dir == "east":
            camera_pos = np.array(
                [
                    wall_center_world[0] - camera_offset,
                    wall_center_world[1],
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((1, 0, 0))
        elif wall_dir == "west":
            camera_pos = np.array(
                [
                    wall_center_world[0] + camera_offset,
                    wall_center_world[1],
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((-1, 0, 0))
        else:
            camera_pos = np.array(
                [
                    wall_center_world[0],
                    wall_center_world[1] - camera_offset,
                    wall_center_world[2],
                ]
            )
            look_dir = Vector((0, 1, 0))

        # Debug logging.
        console_logger.info(
            f"Wall orthographic camera setup for {wall_dir}:\n"
            f"  wall_origin: {wall_origin}\n"
            f"  wall_center_world: {wall_center_world}\n"
            f"  camera_pos: {camera_pos}\n"
            f"  look_dir: {look_dir}"
        )

        # Set camera position.
        camera_obj.location = Vector(camera_pos.tolist())

        # Set camera rotation explicitly to ensure walls appear horizontal.
        # All walls should appear with width horizontal and height vertical.
        # rotation_euler.x = 90° points camera horizontally (from looking down).
        # rotation_euler.z controls which compass direction camera faces.
        if wall_dir == "north":
            # Looking toward +Y. (pi/2, 0, 0) gives forward = +Y.
            camera_obj.rotation_euler = (math.pi / 2, 0, 0)
        elif wall_dir == "south":
            # Looking toward -Y. (pi/2, 0, pi) gives forward = -Y.
            camera_obj.rotation_euler = (math.pi / 2, 0, math.pi)
        elif wall_dir == "east":
            # Looking toward +X.
            camera_obj.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
        elif wall_dir == "west":
            # Looking toward -X.
            camera_obj.rotation_euler = (math.pi / 2, 0, math.pi / 2)
        else:
            # Fallback to north.
            camera_obj.rotation_euler = (math.pi / 2, 0, math.pi)

    def save_blend_file(
        self,
        params: RenderParams,
        output_path: Path,
        additional_visuals: list[dict[str, object]] | None = None,
    ) -> Path:
        """Save the scene as a .blend file.

        Args:
            params: Rendering parameters containing scene path (glTF).
            output_path: Path where .blend file will be saved.
            additional_visuals: Compiled GLB visuals that Drake omitted from
                its exported scene, each with a room-frame translation/yaw.

        Returns:
            Path to the saved .blend file.
        """
        # Setup scene and import glTF (reuse existing methods).
        self._setup_scene(params)
        self._import_and_organize_gltf(params.scene)

        for visual in additional_visuals or []:
            visual_path = Path(str(visual["path"]))
            if visual_path.suffix.lower() not in {".glb", ".gltf"}:
                raise ValueError(f"Supplemental visual must be GLB/GLTF: {visual_path}")
            if not visual_path.is_file():
                raise FileNotFoundError(
                    f"Supplemental visual does not exist: {visual_path}"
                )
            translation = tuple(float(value) for value in visual["translation"])
            if len(translation) != 3 or not all(math.isfinite(v) for v in translation):
                raise ValueError(
                    "Supplemental visual translation must be 3D and finite"
                )
            yaw = float(visual.get("yaw_radians", 0.0))
            if not math.isfinite(yaw):
                raise ValueError("Supplemental visual yaw must be finite")

            bpy.ops.object.select_all(action="DESELECT")
            bpy.ops.import_scene.gltf(filepath=str(visual_path))
            imported = tuple(bpy.context.selected_objects)
            imported_ids = {id(obj) for obj in imported}
            transform = Matrix.Translation(Vector(translation)) @ Matrix.Rotation(
                yaw, 4, "Z"
            )
            roots = [
                obj
                for obj in imported
                if obj.parent is None or id(obj.parent) not in imported_ids
            ]
            for root in roots:
                root.matrix_world = transform @ root.matrix_world
            role = str(visual.get("role", "structural_detail"))
            source_id = str(visual.get("source_id", visual_path.stem))
            for obj in imported:
                if obj.type != "MESH":
                    continue
                obj["aether_role"] = role
                obj["aether_source_id"] = source_id

        # Ensure output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save blend file.
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))

        console_logger.info(f"Saved .blend file to {output_path}")
        return output_path
