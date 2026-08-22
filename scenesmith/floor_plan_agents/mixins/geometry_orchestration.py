"""Stateful floor plan agent using planner/designer/critic workflow.

This module implements the floor plan agent trio for designing house layouts
with rooms, doors, windows, and materials, then generates the geometry.
"""

import logging

from pathlib import Path

import lxml.etree as ET
import numpy as np
import trimesh

from scenesmith.agent_utils.rendering.pipeline.blend_export import (
    save_directive_as_blend,
)
from scenesmith.agent_utils.scene.house_parts.openings import (
    Opening,
    PlacedRoom,
    Wall,
    WallDirection,
)
from scenesmith.floor_plan_agents.tools.geometry_cache import (
    floor_cache_key,
    wall_cache_key,
    window_cache_key,
)
from scenesmith.floor_plan_agents.tools.wall_geometry import (
    WallDimensions,
    WallOpening,
    create_wall_gltf as create_wall_gltf_with_openings,
)
from scenesmith.floor_plan_agents.tools.window_geometry import create_window_mesh
from scenesmith.utils.geometry.architectural_gltf import create_floor_gltf
from scenesmith.utils.geometry.gltf_generation import get_zup_to_yup_matrix
from scenesmith.utils.geometry.material import Material

console_logger = logging.getLogger(__name__)


class FloorPlanGeometryOrchestrationMixin:
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

    def _generate_all_room_geometries(self, output_dir: Path) -> None:
        """Generate geometry for rooms missing from the layout cache.

        Only regenerates geometry for rooms that are not in room_geometries.
        This allows per-room invalidation to work correctly - when a room's
        geometry is invalidated, only that room is regenerated on next render.

        Args:
            output_dir: Directory to save GLTF files.
        """
        replacement_space_ids = {
            mesh.space_id
            for mesh in self.layout.structural_meshes
            if mesh.replaces_room_shell
        }
        for room_spec in self.layout.room_specs:
            if room_spec.room_id in replacement_space_ids:
                # compile_structural_meshes below creates the room-compatible
                # SDF and RoomGeometry. Do not surround it with a legacy box.
                self.layout.invalidate_room_geometry(room_spec.room_id)
                continue
            # Skip rooms that already have geometry (not invalidated).
            if room_spec.room_id in self.layout.room_geometries:
                console_logger.debug(
                    f"Skipping geometry for {room_spec.room_id} (cached)"
                )
                continue

            console_logger.info(f"Generating geometry for {room_spec.room_id}")
            room_geometry = self._generate_room_geometry(
                room_spec=room_spec, output_dir=output_dir
            )
            self.layout.set_room_geometry(room_spec.room_id, room_geometry)

        # Structural assets are authored independently of the legacy room loop,
        # but must exist before ``to_drake_directive`` is called.  Compile them
        # after rooms so room-local surface sidecars can be attached to the
        # generated RoomGeometry and consumed by later placement agents.
        structural_dir = output_dir / "structural"
        if self.layout.connectors:
            self.layout.compile_connectors(structural_dir / "connectors")
        if self.layout.structural_meshes:
            self.layout.compile_structural_meshes(structural_dir / "meshes")
        if self.layout.platforms:
            self.layout.compile_platforms(structural_dir / "platforms")
        if self.layout.heightfields:
            self.layout.compile_heightfields(structural_dir / "heightfields")

        blocked_connectors = self.layout.geometrically_blocked_connectors()
        if blocked_connectors:
            raise ValueError(
                "Structural connectors fail support/headroom/width clearance: "
                + ", ".join(sorted(blocked_connectors))
                + ". Add or resize independent floor/ceiling openings, widen "
                "the connector, or revise its route."
            )

    def _export_floor_plan(self, output_dir: Path) -> None:
        """Export floor plan to .blend and .dmd.yaml files.

        Creates the final_floor_plan directory with:
        - floor_plan.blend: Blender file with PBR materials (matches renders)
        - floor_plan.dmd.yaml: Drake directive for simulation (references SDF files)

        The blend file is created from the DMD directive, which references the
        same SDF and GLTF files used in simulation and preview renders.

        Args:
            output_dir: Base directory for floor plan outputs.
        """
        # Create final floor plan directory.
        final_dir = output_dir / "final_floor_plan"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Export Drake directive first (used by both simulation and blend export).
        # Use house_dir as base for package://scene/ URIs (not final_dir subdirectory).
        directive_path = final_dir / "floor_plan.dmd.yaml"
        house_dir = self.layout.house_dir
        directive_content = self.layout.to_drake_directive(base_dir=house_dir)
        with open(directive_path, "w") as f:
            f.write(directive_content)
        console_logger.info(f"Floor plan directive saved to: {directive_path}")

        # Convert DMD to .blend for external use.
        # Pass house_dir as scene_dir for package://scene/ resolution.
        blend_path = final_dir / "floor_plan.blend"
        save_directive_as_blend(
            directive_path=directive_path,
            output_path=blend_path,
            scene_dir=house_dir,
        )
        console_logger.info(f"Floor plan .blend saved to: {blend_path}")

    def _generate_floor_geometry(
        self,
        placed_room: PlacedRoom,
        room_id: str,
        floor_material: Material,
        floor_thickness: float,
        floors_dir: Path,
        link_element: ET.Element,
    ) -> Path:
        """Generate floor GLTF and add to SDF.

        Uses geometry cache to reuse unchanged floors across iterations.

        Args:
            placed_room: The placed room with dimensions.
            room_id: Room identifier for path generation.
            floor_material: Floor material with PBR textures.
            floor_thickness: Floor thickness in meters.
            floors_dir: Directory to save floor GLTF.
            link_element: SDF link element to add floor to.

        Returns:
            Path to the generated floor GLTF file.
        """
        cache_key = floor_cache_key(
            width=placed_room.width,
            depth=placed_room.depth,
            thickness=floor_thickness,
            material=floor_material,
        )

        def create_fn(output_path: Path) -> None:
            create_floor_gltf(
                width=placed_room.width,
                depth=placed_room.depth,
                thickness=floor_thickness,
                material=floor_material,
                output_path=output_path,
                texture_scale=0.5,
                center_x=0.0,
                center_y=0.0,
                center_z=-floor_thickness / 2,
            )

        assert self._geometry_cache is not None
        floor_gltf_path = self._geometry_cache.get_or_create_floor(
            cache_key=cache_key, output_dir=floors_dir, create_fn=create_fn
        )

        # Add floor to SDF.
        floor_gltf_rel = f"../floor_plans/{room_id}/floors/floor.gltf"
        self._add_gltf_floor_visual(link_element, floor_gltf_rel)
        self._add_floor_collision(
            link_element, length=placed_room.width, width=placed_room.depth
        )

        return floor_gltf_path

    def _generate_window_frame(
        self,
        opening: Opening,
        wall_thickness: float,
        is_horizontal: bool,
        effective_wall_length: float,
        local_x: float,
        local_y: float,
        room_id: str,
        room_output_dir: Path,
        link_element: ET.Element,
    ) -> None:
        """Generate window frame mesh and add to SDF.

        Uses geometry cache to reuse unchanged windows across iterations.

        Args:
            opening: Window opening specification.
            wall_thickness: Wall thickness in meters.
            is_horizontal: Whether this is a horizontal (N/S) wall.
            effective_wall_length: Wall length (with corner extension if applicable).
            local_x: Wall X position in room coordinates.
            local_y: Wall Y position in room coordinates.
            room_id: Room identifier for path generation.
            room_output_dir: Room output directory.
            link_element: SDF link element to add window to.
        """
        cache_key = window_cache_key(
            width=opening.width,
            height=opening.height,
            depth=wall_thickness,
            is_horizontal=is_horizontal,
            shape=opening.shape.value,
        )

        def create_fn(output_path: Path) -> None:
            # Create window frame mesh (in Z-up coords, facing +Y).
            window_scene = create_window_mesh(
                width=opening.width,
                height=opening.height,
                depth=wall_thickness,
                shape=opening.shape.value,
            )

            # Z-up to Y-up transform for GLTF export.
            zup_to_yup = get_zup_to_yup_matrix()

            # Create new scene for export with transformed meshes.
            export_scene = trimesh.Scene()
            for part_name, part_mesh in window_scene.geometry.items():
                mesh_copy = part_mesh.copy()

                # For E/W walls (vertical), rotate 90° around Z in Z-up coords
                # BEFORE transforming to Y-up. This aligns window with wall.
                if not is_horizontal:
                    rotation = trimesh.transformations.rotation_matrix(
                        np.pi / 2, [0, 0, 1]  # Z is up in Z-up coordinates.
                    )
                    mesh_copy.apply_transform(rotation)

                # Transform from Z-up to Y-up for GLTF export.
                mesh_copy.apply_transform(zup_to_yup)
                export_scene.add_geometry(mesh_copy, geom_name=part_name)

            # Export window frame as GLTF.
            export_scene.export(str(output_path), file_type="gltf")

        # Create windows subdirectory.
        windows_dir = room_output_dir / "windows"
        window_subdir = windows_dir / opening.opening_id

        assert self._geometry_cache is not None
        self._geometry_cache.get_or_create_window(
            cache_key=cache_key, output_dir=window_subdir, create_fn=create_fn
        )

        # Calculate window position in local room coordinates.
        # opening.position_along_wall is LEFT EDGE, need CENTER.
        opening_center_along_wall = opening.position_along_wall + opening.width / 2
        # Window center height = sill_height + height/2.
        window_z = opening.sill_height + opening.height / 2

        # Convert position along wall to local room coords.
        if is_horizontal:
            # N/S walls: window moves along X axis.
            window_x = opening_center_along_wall - effective_wall_length / 2
            window_y = local_y
        else:
            # E/W walls: window moves along Y axis.
            window_x = local_x
            window_y = opening_center_along_wall - effective_wall_length / 2

        # Add window visual to SDF.
        window_gltf_rel = (
            f"../floor_plans/{room_id}/windows/{opening.opening_id}/window.gltf"
        )
        self._add_window_frame_visual(
            link_element=link_element,
            window_name=opening.opening_id,
            gltf_relative_path=window_gltf_rel,
            pose_x=window_x,
            pose_y=window_y,
            pose_z=window_z,
        )
        console_logger.debug(
            f"Added window frame {opening.opening_id} at "
            f"({window_x:.2f}, {window_y:.2f}, {window_z:.2f})"
        )

    def _generate_exterior_wall(
        self,
        wall: Wall,
        dimensions: WallDimensions,
        wall_openings: list[WallOpening],
        placed_room: PlacedRoom,
        offset: float,
        local_x: float,
        local_y: float,
        is_horizontal: bool,
        room_id: str,
        walls_dir: Path,
        link_element: ET.Element,
    ) -> None:
        """Generate exterior wall layer for a wall.

        Creates the outer layer of a dual-wall system for exterior walls.

        Args:
            wall: Wall specification with direction and exterior flag.
            dimensions: Wall dimensions (width, height, thickness).
            wall_openings: List of openings in the wall.
            placed_room: The placed room with dimensions.
            offset: Half wall thickness for positioning.
            local_x: Inner wall X position.
            local_y: Inner wall Y position.
            is_horizontal: Whether this is a horizontal (N/S) wall.
            room_id: Room identifier for path generation.
            walls_dir: Directory to save wall GLTFs.
            link_element: SDF link element to add wall to.
        """
        # Get exterior material.
        exterior_material = self.layout.exterior_material
        if exterior_material is None:
            exterior_material = Material.from_path(Path("materials/Plaster001_1K-JPG"))

        # Compute outer wall position (opposite offset from inner wall).
        if wall.direction == WallDirection.NORTH:
            outer_local_y = placed_room.depth / 2 + offset
            outer_local_x = local_x
        elif wall.direction == WallDirection.SOUTH:
            outer_local_y = -placed_room.depth / 2 - offset
            outer_local_x = local_x
        elif wall.direction == WallDirection.EAST:
            outer_local_x = placed_room.width / 2 + offset
            outer_local_y = local_y
        else:  # WEST
            outer_local_x = -placed_room.width / 2 - offset
            outer_local_y = local_y

        # Generate outer wall GLTF with exterior material and caching.
        outer_wall_name = f"{wall.direction.value}_wall_exterior"
        outer_wall_subdir = walls_dir / outer_wall_name
        openings_dicts = [o.to_dict() for o in wall_openings] if wall_openings else None
        cache_key = wall_cache_key(
            width=dimensions.width,
            height=dimensions.height,
            thickness=dimensions.thickness,
            material=exterior_material,
            openings=openings_dicts,
        )

        def create_exterior_wall_fn(output_path: Path) -> None:
            create_wall_gltf_with_openings(
                dimensions=dimensions,
                openings=wall_openings if wall_openings else None,
                output_path=output_path,
                uv_scale=0.5,
                material=exterior_material,
            )

        assert self._geometry_cache is not None
        self._geometry_cache.get_or_create_wall(
            cache_key=cache_key,
            output_dir=outer_wall_subdir,
            create_fn=create_exterior_wall_fn,
        )

        # Add outer wall visual with pose.
        outer_wall_gltf_rel = (
            f"../floor_plans/{room_id}/walls/{outer_wall_name}/wall.gltf"
        )
        self._add_gltf_wall_visual_with_pose(
            link_element=link_element,
            wall_name=outer_wall_name,
            gltf_relative_path=outer_wall_gltf_rel,
            pose_x=outer_local_x,
            pose_y=outer_local_y,
            pose_z=0.0,
            is_horizontal=is_horizontal,
        )

        console_logger.debug(
            f"Added exterior wall {outer_wall_name} at "
            f"({outer_local_x:.2f}, {outer_local_y:.2f})"
        )
