"""Stateful floor plan agent using planner/designer/critic workflow.

This module implements the floor plan agent trio for designing house layouts
with rooms, doors, windows, and materials, then generates the geometry.
"""

import logging
import math

from pathlib import Path

import lxml.etree as ET
import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.scene.clearance_zones import compute_openings_data
from scenesmith.agent_utils.scene.house_parts.openings import OpeningType, PlacedRoom
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import (
    RoomSpec,
    legacy_openings_to_boundary_portals,
)
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.floor_plan_agents.tools.wall_geometry import WallOpening, WallSpec
from scenesmith.utils.geometry.material import Material

console_logger = logging.getLogger(__name__)


class FloorPlanGeometryExportMixin:
    """Stateful floor plan agent using planner/designer/critic workflow.

    This agent designs house layouts through an iterative process of:
    1. Designer proposes rooms, doors, windows using layout tools.
    2. Critic evaluates the design with VLM-based visual critique.
    3. Iteration continues until the design meets quality criteria.

    The layout is stored in a HouseLayout object that tracks:
    - Room specifications with adjacency constraints
    - Door and window placements on walls
    - Material assignments for floors and walls

    After design completion, geometry is generated for each room:
    - Floor meshes as GLTF
    - Wall meshes with door/window openings as GLTF
    - Full SDF/URDF assembly for Drake simulation
    """

    # Floor plan agent doesn't place objects, so no placement style tool.
    _is_placement_agent: bool = False

    def _generate_polygon_room_geometry(
        self,
        *,
        room_spec: RoomSpec,
        placed_room: PlacedRoom,
        output_dir: Path,
        wall_height: float,
        wall_thickness: float,
        floor_thickness: float,
    ) -> RoomGeometry:
        """Generate an arbitrary polygon room without cardinal-wall assumptions."""

        from scenesmith.agent_utils.structure.compiler.models import triangle_group_mesh
        from scenesmith.agent_utils.structure.compiler.polygon_spaces import (
            compile_polygon_space,
        )
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )
        from scenesmith.agent_utils.structure.geometry_models.surface_models import (
            Footprint2D,
            SurfaceRole,
        )

        source_footprint = room_spec.footprint or Footprint2D.rectangle(
            room_spec.length, room_spec.width
        )
        min_x, min_y, max_x, max_y = source_footprint.bounds
        footprint_width = max_x - min_x
        footprint_depth = max_y - min_y
        if not math.isclose(
            footprint_width, room_spec.length, rel_tol=0.0, abs_tol=1e-6
        ) or not math.isclose(
            footprint_depth, room_spec.width, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"Room '{room_spec.room_id}' footprint bounds are "
                f"{footprint_width:g} × {footprint_depth:g}, but length/width "
                f"are {room_spec.length:g} × {room_spec.width:g}"
            )

        footprint = source_footprint.centered_on_bounds()
        floor_footprint = (
            room_spec.floor_footprint or source_footprint
        ).centered_on_bounds()
        ceiling_footprint = (
            room_spec.ceiling_footprint or source_footprint
        ).centered_on_bounds()
        authored_portals = [
            portal
            for portal in self.layout.portals
            if portal.source_space_id == room_spec.room_id
            and portal.boundary_loop_index is not None
        ]
        authored_ids = {portal.portal_id for portal in authored_portals}
        legacy_portals = legacy_openings_to_boundary_portals(
            room_spec, placed_room, wall_height
        )
        compiled = compile_polygon_space(
            structure_id=f"room_geometry_{room_spec.room_id}",
            footprint=footprint,
            floor_footprint=floor_footprint,
            ceiling_footprint=ceiling_footprint,
            include_floor=not any(
                heightfield.space_id == room_spec.room_id and heightfield.replaces_floor
                for heightfield in self.layout.heightfields
            ),
            include_ceiling=room_spec.has_overhead_cover,
            floor_profile=room_spec.floor_profile,
            ceiling_profile=room_spec.ceiling_profile,
            wall_height=wall_height,
            floor_thickness=floor_thickness,
            ceiling_thickness=floor_thickness,
            portals=[
                *authored_portals,
                *(
                    portal
                    for portal in legacy_portals
                    if portal.portal_id not in authored_ids
                ),
            ],
        )
        room_materials = self.layout.room_materials.get(room_spec.room_id)
        floor_material = (
            room_materials.floor_material
            if room_materials is not None and room_materials.floor_material is not None
            else Material.from_path("materials/Wood094_1K-JPG")
        )
        floor_overlays = (
            {
                "floor_finish": (
                    triangle_group_mesh(
                        compiled.visual_mesh,
                        compiled.triangle_groups["floor_top"],
                        translation=(0.0, 0.0, 0.0015),
                    ),
                    floor_material,
                    floor_material.texture_scale or 0.5,
                )
            }
            if compiled.triangle_groups.get("floor_top")
            else None
        )
        paths = write_compiled_structure(
            compiled,
            output_dir / room_spec.room_id / "structural",
            model_name="room_geometry",
            link_name="room_geometry_body_link",
            visual_overlays=floor_overlays,
        )

        wall_objects: list[SceneObject] = []
        wall_normals: dict[str, np.ndarray] = {}
        for patch in compiled.surfaces:
            if SurfaceRole.BOUNDARY not in patch.surface.roles:
                continue
            start, end = patch.boundary[0], patch.boundary[1]
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            yaw = math.atan2(end[1] - start[1], end[0] - start[0])
            z_values = [point[2] for point in patch.boundary]
            z_min, z_max = min(z_values), max(z_values)
            name = patch.surface.surface_id
            wall_objects.append(
                SceneObject(
                    object_id=UniqueID(name),
                    object_type=ObjectType.WALL,
                    name=name,
                    description="Arbitrary structural boundary wall",
                    transform=RigidTransform(
                        RollPitchYaw(0.0, 0.0, yaw),
                        [
                            (start[0] + end[0]) / 2.0,
                            (start[1] + end[1]) / 2.0,
                            (z_min + z_max) / 2.0,
                        ],
                    ),
                    bbox_min=np.array(
                        [-length / 2.0, -wall_thickness / 2.0, -(z_max - z_min) / 2.0]
                    ),
                    bbox_max=np.array(
                        [length / 2.0, wall_thickness / 2.0, (z_max - z_min) / 2.0]
                    ),
                    metadata={"structural_surface_id": name},
                    immutable=True,
                )
            )
            wall_normals[name] = np.asarray(patch.normal[:2])

        floor_indices = compiled.triangle_groups["floor_top"]
        floor_vertices = [
            compiled.visual_mesh.vertices[vertex_index]
            for triangle_index in floor_indices
            for vertex_index in compiled.visual_mesh.triangles[triangle_index]
        ]
        floor_object = SceneObject(
            object_id=UniqueID(f"floor_{room_spec.room_id}"),
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Polygonal floor support surface",
            transform=RigidTransform(),
            geometry_path=paths.mesh_path,
            bbox_min=np.array(
                [
                    min(point[0] for point in floor_vertices),
                    min(point[1] for point in floor_vertices),
                    min(point[2] for point in floor_vertices) - floor_thickness,
                ]
            ),
            bbox_max=np.array(
                [
                    max(point[0] for point in floor_vertices),
                    max(point[1] for point in floor_vertices),
                    max(point[2] for point in floor_vertices),
                ]
            ),
            metadata={
                "structural_surface_id": f"room_geometry_{room_spec.room_id}_floor"
            },
            immutable=True,
        )

        return RoomGeometry(
            sdf_tree=ET.parse(paths.sdf_path),
            sdf_path=paths.sdf_path,
            walls=wall_objects,
            floor=floor_object,
            wall_normals=wall_normals,
            width=footprint_depth,
            length=footprint_width,
            wall_height=wall_height,
            has_overhead_cover=room_spec.has_overhead_cover,
            wall_thickness=wall_thickness,
            openings=compute_openings_data(
                placed_room=placed_room,
                wall_height=wall_height,
                door_clearance_distance=self.cfg.clearance_zones.door_clearance_distance,
                window_clearance_distance=(
                    self.cfg.clearance_zones.window_clearance_distance
                ),
            ),
            footprint=footprint,
            floor_footprint=floor_footprint,
            ceiling_footprint=ceiling_footprint,
            floor_profile=room_spec.floor_profile,
            ceiling_profile=room_spec.ceiling_profile,
            structural_surfaces=[patch.surface for patch in compiled.surfaces],
            structural_surface_path=paths.surfaces_path,
        )

    @staticmethod
    def _get_wall_specifications(
        length: float, width: float, wall_thickness: float = 0.05
    ) -> list[WallSpec]:
        """Get wall specifications for a rectangular room.

        Args:
            length: Room length in the x-direction (meters).
            width: Room width in the y-direction (meters).
            wall_thickness: Thickness of walls in meters.

        Returns:
            List of WallSpec objects defining all four walls.
        """
        half_length = length / 2.0
        half_width = width / 2.0

        return [
            WallSpec(
                name="left_wall",
                center_x=-half_length,
                center_y=0.0,
                bbox_width=wall_thickness,
                bbox_depth=width,
                thickness=wall_thickness,
            ),
            WallSpec(
                name="right_wall",
                center_x=half_length,
                center_y=0.0,
                bbox_width=wall_thickness,
                bbox_depth=width,
                thickness=wall_thickness,
            ),
            WallSpec(
                name="back_wall",
                center_x=0.0,
                center_y=-half_width,
                bbox_width=length,
                bbox_depth=wall_thickness,
                thickness=wall_thickness,
            ),
            WallSpec(
                name="front_wall",
                center_x=0.0,
                center_y=half_width,
                bbox_width=length,
                bbox_depth=wall_thickness,
                thickness=wall_thickness,
            ),
        ]

    @staticmethod
    def _add_gltf_wall_visual_with_pose(
        link_element: ET.Element,
        wall_name: str,
        gltf_relative_path: str,
        pose_x: float,
        pose_y: float,
        pose_z: float,
        is_horizontal: bool,
    ) -> None:
        """Add GLTF wall visual to SDF link element with pose.

        Args:
            link_element: SDF link element to add visual to.
            wall_name: Name of the wall.
            gltf_relative_path: Relative path to GLTF file.
            pose_x: X position of wall center.
            pose_y: Y position of wall center.
            pose_z: Z position of wall center.
            is_horizontal: True for north/south walls, False for east/west walls.
        """
        visual = ET.SubElement(link_element, "visual", name=f"{wall_name}_visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = gltf_relative_path

        # Add pose element.
        # Wall meshes are created centered at origin along X axis.
        # For north/south walls (horizontal), no rotation needed.
        # For east/west walls (vertical), rotate 90° around Z.
        pose = ET.SubElement(visual, "pose")
        if is_horizontal:
            pose.text = f"{pose_x} {pose_y} {pose_z} 0 0 0"
        else:
            # Rotate 90° around Z axis for vertical walls.
            pose.text = f"{pose_x} {pose_y} {pose_z} 0 0 1.5708"

    @staticmethod
    def _add_window_frame_visual(
        link_element: ET.Element,
        window_name: str,
        gltf_relative_path: str,
        pose_x: float,
        pose_y: float,
        pose_z: float,
    ) -> None:
        """Add GLTF window frame visual to SDF link element with pose.

        Args:
            link_element: SDF link element to add visual to.
            window_name: Name of the window.
            gltf_relative_path: Relative path to GLTF file.
            pose_x: X position of window center in local room coords.
            pose_y: Y position of window center in local room coords.
            pose_z: Z position of window center in local room coords.
        """
        visual = ET.SubElement(link_element, "visual", name=f"{window_name}_visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = gltf_relative_path

        # Window mesh rotation is baked in during GLTF export.
        pose = ET.SubElement(visual, "pose")
        pose.text = f"{pose_x} {pose_y} {pose_z} 0 0 0"

    def _create_wall_objects(
        self, wall_specs: list[WallSpec], wall_height: float
    ) -> list[SceneObject]:
        """Create wall objects from specifications.

        Args:
            wall_specs: Wall specifications defining wall geometry.
            wall_height: Wall height in the z-direction (meters).

        Returns:
            List of wall SceneObjects with proper transforms and bounding boxes.
        """
        walls = []

        for spec in wall_specs:
            # Create transform at wall center.
            transform = RigidTransform(
                p=[spec.center_x, spec.center_y, wall_height / 2.0]
            )

            # Bounding box in object frame (centered at origin).
            bbox_min = np.array(
                [-spec.bbox_width / 2.0, -spec.bbox_depth / 2.0, -wall_height / 2.0]
            )
            bbox_max = np.array(
                [spec.bbox_width / 2.0, spec.bbox_depth / 2.0, wall_height / 2.0]
            )

            wall_obj = SceneObject(
                object_id=UniqueID(spec.name),
                object_type=ObjectType.WALL,
                name=spec.name,
                description=f"Room {spec.name}",
                transform=transform,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                immutable=True,
            )
            walls.append(wall_obj)

        return walls

    @staticmethod
    def _add_gltf_floor_visual(
        link_element: ET.Element, gltf_relative_path: str
    ) -> None:
        """Add GLTF floor visual to SDF link element.

        Args:
            link_element: SDF link element to add visual to.
            gltf_relative_path: Relative path to GLTF file.
        """
        visual = ET.SubElement(link_element, "visual", name="floor_visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        uri = ET.SubElement(mesh, "uri")
        uri.text = gltf_relative_path

    @staticmethod
    def _add_floor_collision(
        link_element: ET.Element, length: float, width: float
    ) -> None:
        """Add floor collision geometry to SDF link element.

        Args:
            link_element: SDF link element to add collision to.
            length: Floor length in meters.
            width: Floor width in meters.
        """
        collision = ET.SubElement(link_element, "collision", name="floor_collision")
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = f"{length} {width} 0.1"
        pose = ET.SubElement(collision, "pose")
        pose.text = "0 0 -0.05 0 0 0"

    @staticmethod
    def _add_wall_collision(
        link_element: ET.Element, wall_spec: WallSpec, wall_height: float
    ) -> None:
        """Add wall collision geometry to SDF link element.

        Args:
            link_element: SDF link element to add collision to.
            wall_spec: Wall specification.
            wall_height: Wall height in meters.
        """
        collision = ET.SubElement(
            link_element, "collision", name=f"{wall_spec.name}_collision"
        )
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        size = ET.SubElement(box, "size")
        size.text = f"{wall_spec.bbox_width} {wall_spec.bbox_depth} {wall_height}"
        pose = ET.SubElement(collision, "pose")
        pose.text = (
            f"{wall_spec.center_x} {wall_spec.center_y} {wall_height / 2.0} 0 0 0"
        )

    @staticmethod
    def _add_wall_collision_with_openings(
        link_element: ET.Element,
        wall_spec: WallSpec,
        wall_height: float,
        openings: list[WallOpening],
        wall_length: float,
        is_horizontal: bool,
    ) -> None:
        """Add wall collision geometry with cutouts for doors and open connections.

        Creates multiple collision boxes that avoid door/open sections, allowing
        passage through doors while maintaining solid collision for windows.

        Args:
            link_element: SDF link element to add collision to.
            wall_spec: Wall specification.
            wall_height: Wall height in meters.
            openings: List of openings in this wall.
            wall_length: Full length of the wall (width or depth of room).
            is_horizontal: True for NORTH/SOUTH walls, False for EAST/WEST.
        """
        # Filter to only passable openings (doors and open connections, not windows).
        passable_openings = [
            o for o in openings if o.opening_type != OpeningType.WINDOW
        ]

        if not passable_openings:
            # No doors/open connections - use single solid box.
            FloorPlanGeometryExportMixin._add_wall_collision(
                link_element, wall_spec, wall_height
            )
            return

        # Sort openings by position along wall.
        sorted_openings = sorted(passable_openings, key=lambda o: o.position_along_wall)

        # Compute solid segments between openings.
        # Each segment is (start_pos, end_pos) along the wall.
        segments: list[tuple[float, float]] = []
        current_pos = 0.0

        for opening in sorted_openings:
            if opening.position_along_wall > current_pos:
                # Solid segment before this opening.
                segments.append((current_pos, opening.position_along_wall))
            current_pos = opening.position_along_wall + opening.width

        # Final segment after last opening.
        if current_pos < wall_length:
            segments.append((current_pos, wall_length))

        # Create collision box for each solid segment.
        for i, (start, end) in enumerate(segments):
            segment_length = end - start
            if segment_length <= 0.001:
                # Skip very small segments (floating point artifacts).
                continue

            # Segment center relative to wall center.
            segment_center_along_wall = start + segment_length / 2 - wall_length / 2

            collision = ET.SubElement(
                link_element, "collision", name=f"{wall_spec.name}_collision_{i}"
            )
            geometry = ET.SubElement(collision, "geometry")
            box = ET.SubElement(geometry, "box")
            size = ET.SubElement(box, "size")

            if is_horizontal:
                # NORTH/SOUTH wall: segments run along X axis.
                size.text = f"{segment_length} {wall_spec.bbox_depth} {wall_height}"
                pose_x = wall_spec.center_x + segment_center_along_wall
                pose_y = wall_spec.center_y
            else:
                # EAST/WEST wall: segments run along Y axis.
                size.text = f"{wall_spec.bbox_width} {segment_length} {wall_height}"
                pose_x = wall_spec.center_x
                pose_y = wall_spec.center_y + segment_center_along_wall

            pose = ET.SubElement(collision, "pose")
            pose.text = f"{pose_x} {pose_y} {wall_height / 2.0} 0 0 0"
