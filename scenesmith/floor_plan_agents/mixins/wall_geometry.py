"""Stateful floor plan agent using planner/designer/critic workflow.

This module implements the floor plan agent trio for designing house layouts
with rooms, doors, windows, and materials, then generates the geometry.
"""

import logging

from pathlib import Path

import lxml.etree as ET
import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.scene.clearance_zones import compute_openings_data
from scenesmith.agent_utils.scene.house_parts.openings import (
    OpeningType,
    PlacedRoom,
    Wall,
    WallDirection,
    compute_wall_normals,
)
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.floor_plan_agents.tools.geometry_cache import wall_cache_key
from scenesmith.floor_plan_agents.tools.wall_geometry import (
    WallDimensions,
    WallOpening,
    WallSpec,
    create_wall_gltf as create_wall_gltf_with_openings,
)
from scenesmith.utils.geometry.material import Material

console_logger = logging.getLogger(__name__)


class FloorPlanWallGeometryMixin:
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

    def _generate_single_wall(
        self,
        wall: Wall,
        placed_room: PlacedRoom,
        wall_height: float,
        wall_thickness: float,
        wall_material: Path,
        room_id: str,
        room_output_dir: Path,
        walls_dir: Path,
        link_element: ET.Element,
    ) -> WallSpec | None:
        """Generate geometry for a single wall.

        Args:
            wall: Wall specification from PlacedRoom.
            placed_room: The placed room with dimensions.
            wall_height: Wall height in meters.
            wall_thickness: Wall thickness in meters.
            wall_material: Path to wall material folder.
            room_id: Room identifier for path generation.
            room_output_dir: Room output directory.
            walls_dir: Directory to save wall GLTFs.
            link_element: SDF link element to add wall to.

        Returns:
            WallSpec for the wall, or None if wall was skipped.
        """
        # Determine wall length for this direction.
        if wall.direction in (WallDirection.NORTH, WallDirection.SOUTH):
            wall_length_dim = placed_room.width
        else:
            wall_length_dim = placed_room.depth

        # Skip walls that are entirely covered by an OPEN opening.
        if any(
            opening.opening_type == OpeningType.OPEN
            and opening.width >= wall_length_dim - 0.001  # 1mm tolerance.
            for opening in wall.openings
        ):
            console_logger.debug(
                f"Skipping wall {wall.wall_id} - fully covered by OPEN opening"
            )
            return None

        # Determine wall orientation and local position.
        offset = wall_thickness / 2
        is_horizontal = wall.direction in (WallDirection.NORTH, WallDirection.SOUTH)
        if is_horizontal:
            if wall.direction == WallDirection.NORTH:
                local_y = placed_room.depth / 2 - offset
            else:
                local_y = -placed_room.depth / 2 + offset
            local_x = 0.0
        else:
            if wall.direction == WallDirection.EAST:
                local_x = placed_room.width / 2 - offset
            else:
                local_x = -placed_room.width / 2 + offset
            local_y = 0.0

        wall_name = f"{wall.direction.value}_wall"

        # Convert openings to WallOpening format.
        wall_openings = []
        for opening in wall.openings:
            effective_height = (
                wall_height
                if opening.opening_type == OpeningType.OPEN
                else opening.height
            )
            wall_openings.append(
                WallOpening(
                    position_along_wall=opening.position_along_wall,
                    width=opening.width,
                    height=effective_height,
                    sill_height=opening.sill_height,
                    opening_type=opening.opening_type,
                    shape=opening.shape,
                )
            )

        # Determine if wall needs corner extension.
        length_override = None
        if wall.is_exterior and is_horizontal:
            length_override = wall_length_dim + wall_thickness
            console_logger.debug(
                f"Corner extension: {wall.wall_id} extended from "
                f"{wall_length_dim:.3f}m to {length_override:.3f}m"
            )

        # Create wall dimensions.
        effective_wall_length = length_override if length_override else wall_length_dim
        dimensions = WallDimensions(
            width=effective_wall_length,
            height=wall_height,
            thickness=wall_thickness,
        )

        # Generate wall GLTF with caching.
        wall_subdir = walls_dir / wall_name
        openings_dicts = [o.to_dict() for o in wall_openings] if wall_openings else None
        cache_key = wall_cache_key(
            width=dimensions.width,
            height=dimensions.height,
            thickness=dimensions.thickness,
            material=wall_material,
            openings=openings_dicts,
        )

        def create_wall_fn(output_path: Path) -> None:
            create_wall_gltf_with_openings(
                dimensions=dimensions,
                openings=wall_openings if wall_openings else None,
                output_path=output_path,
                uv_scale=0.5,
                material=wall_material,
            )

        assert self._geometry_cache is not None
        self._geometry_cache.get_or_create_wall(
            cache_key=cache_key,
            output_dir=wall_subdir,
            create_fn=create_wall_fn,
        )

        # Create WallSpec for SDF collision and SceneObject creation.
        if is_horizontal:
            bbox_width = placed_room.width
            bbox_depth = wall_thickness
        else:
            bbox_width = wall_thickness
            bbox_depth = placed_room.depth

        spec = WallSpec(
            name=wall_name,
            center_x=local_x,
            center_y=local_y,
            bbox_width=bbox_width,
            bbox_depth=bbox_depth,
            thickness=wall_thickness,
        )

        # Add wall visual to SDF.
        wall_gltf_rel = f"../floor_plans/{room_id}/walls/{wall_name}/wall.gltf"
        self._add_gltf_wall_visual_with_pose(
            link_element=link_element,
            wall_name=wall_name,
            gltf_relative_path=wall_gltf_rel,
            pose_x=local_x,
            pose_y=local_y,
            pose_z=0.0,
            is_horizontal=is_horizontal,
        )

        # Add wall collision with cutouts for doors/open connections.
        self._add_wall_collision_with_openings(
            link_element=link_element,
            wall_spec=spec,
            wall_height=wall_height,
            openings=wall_openings,
            wall_length=wall_length_dim,
            is_horizontal=is_horizontal,
        )

        # Generate window frames for WINDOW openings.
        for opening in wall.openings:
            if opening.opening_type == OpeningType.WINDOW:
                self._generate_window_frame(
                    opening=opening,
                    wall_thickness=wall_thickness,
                    is_horizontal=is_horizontal,
                    effective_wall_length=effective_wall_length,
                    local_x=local_x,
                    local_y=local_y,
                    room_id=room_id,
                    room_output_dir=room_output_dir,
                    link_element=link_element,
                )

        # Generate exterior wall layer if this is an exterior wall.
        if wall.is_exterior:
            self._generate_exterior_wall(
                wall=wall,
                dimensions=dimensions,
                wall_openings=wall_openings,
                placed_room=placed_room,
                offset=offset,
                local_x=local_x,
                local_y=local_y,
                is_horizontal=is_horizontal,
                room_id=room_id,
                walls_dir=walls_dir,
                link_element=link_element,
            )

        return spec

    def _generate_room_geometry(
        self, room_spec: RoomSpec, output_dir: Path
    ) -> RoomGeometry:
        """Generate geometry for a single room.

        Uses walls from placed_rooms which include door/window openings.
        Room geometry is generated in local coordinates (centered at origin),
        then positioned by Drake directive.

        Args:
            room_spec: Room specification.
            output_dir: Directory to save GLTF files.

        Returns:
            RoomGeometry with walls, floor, and SDF.
        """
        wall_height = self.layout.wall_height
        wall_thickness = self.cfg.wall_thickness
        floor_thickness = self.cfg.floor_thickness

        # Find the PlacedRoom for this spec.
        placed_room = None
        for pr in self.layout.placed_rooms:
            if pr.room_id == room_spec.room_id:
                placed_room = pr
                break

        if not placed_room:
            raise ValueError(
                f"No placed room found for room_id '{room_spec.room_id}'. "
                f"Ensure placement algorithm ran successfully."
            )

        from scenesmith.agent_utils.structure.geometry_models.surface_models import (
            ElevationProfile,
        )

        if (
            room_spec.footprint is not None
            or room_spec.floor_footprint is not None
            or room_spec.ceiling_footprint is not None
            or room_spec.floor_profile != ElevationProfile()
            or room_spec.ceiling_profile is not None
            or not room_spec.has_overhead_cover
            or abs(room_spec.yaw) > 1e-9
            or len(self.layout.levels) > 1
            or bool(self.layout.connectors)
            or any(
                portal.source_space_id == room_spec.room_id
                for portal in self.layout.portals
            )
            or any(
                heightfield.space_id == room_spec.room_id and heightfield.replaces_floor
                for heightfield in self.layout.heightfields
            )
        ):
            return self._generate_polygon_room_geometry(
                room_spec=room_spec,
                placed_room=placed_room,
                output_dir=output_dir,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                floor_thickness=floor_thickness,
            )

        # Get materials for this room.
        room_materials = self.layout.room_materials.get(room_spec.room_id)
        wall_material = Material.from_path("materials/Plaster001_1K-JPG")  # Default.
        floor_material = Material.from_path("materials/Wood094_1K-JPG")  # Default.

        if room_materials:
            if room_materials.wall_material:
                wall_material = room_materials.wall_material
            if room_materials.floor_material:
                floor_material = room_materials.floor_material

        # Create SDF structure.
        root_item = ET.Element("sdf", version="1.7", nsmap={"drake": "drake.mit.edu"})
        model_item = ET.SubElement(root_item, "model", name="room_geometry")
        link_item = ET.SubElement(model_item, "link", name="room_geometry_body_link")

        # Create subdirectories for GLTFs.
        room_output_dir = output_dir / room_spec.room_id
        walls_dir = room_output_dir / "walls"
        floors_dir = room_output_dir / "floors"
        walls_dir.mkdir(parents=True, exist_ok=True)
        floors_dir.mkdir(parents=True, exist_ok=True)

        # Generate floor geometry.
        floor_gltf_path = self._generate_floor_geometry(
            placed_room=placed_room,
            room_id=room_spec.room_id,
            floor_material=floor_material,
            floor_thickness=floor_thickness,
            floors_dir=floors_dir,
            link_element=link_item,
        )

        # Generate walls and collect wall specs.
        wall_specs_for_objects: list[WallSpec] = []
        for wall in placed_room.walls:
            spec = self._generate_single_wall(
                wall=wall,
                placed_room=placed_room,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                wall_material=wall_material,
                room_id=room_spec.room_id,
                room_output_dir=room_output_dir,
                walls_dir=walls_dir,
                link_element=link_item,
            )
            if spec is not None:
                wall_specs_for_objects.append(spec)

        # Save room geometry SDF (includes floor and walls).
        sdf_output_dir = self.logger.output_dir / "room_geometry"
        room_geometry_path = self.logger.log_sdf(
            name=f"room_geometry_{room_spec.room_id}",
            sdf_tree=ET.ElementTree(root_item),
            output_dir=sdf_output_dir,
        )

        # Create wall objects.
        walls = self._create_wall_objects(
            wall_specs=wall_specs_for_objects, wall_height=wall_height
        )
        wall_normals = compute_wall_normals(walls=walls)

        # Create floor object. Floor is part of room geometry SDF, not standalone.
        floor_object = SceneObject(
            object_id=UniqueID(f"floor_{room_spec.room_id}"),
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Floor surface",
            transform=RigidTransform(),
            geometry_path=floor_gltf_path,
            sdf_path=None,
            bbox_min=np.array(
                [-placed_room.width / 2, -placed_room.depth / 2, -floor_thickness]
            ),
            bbox_max=np.array([placed_room.width / 2, placed_room.depth / 2, 0.0]),
            immutable=True,
        )

        # Compute openings data for physics validation and label rendering.
        openings = compute_openings_data(
            placed_room=placed_room,
            wall_height=wall_height,
            door_clearance_distance=self.cfg.clearance_zones.door_clearance_distance,
            window_clearance_distance=self.cfg.clearance_zones.window_clearance_distance,
        )

        return RoomGeometry(
            sdf_tree=ET.ElementTree(root_item),
            sdf_path=room_geometry_path,
            walls=walls,
            floor=floor_object,
            wall_normals=wall_normals,
            # RoomGeometry: length=X-dim, width=Y-dim (matches RoomSpec convention).
            # PlacedRoom: width=X-dim, depth=Y-dim.
            width=placed_room.depth,
            length=placed_room.width,
            wall_height=wall_height,
            has_overhead_cover=room_spec.has_overhead_cover,
            wall_thickness=wall_thickness,
            openings=openings,
        )
