"""House layout and room geometry data structures."""

import logging
import math
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    ElevationProfile,
    Footprint2D,
    SurfaceRole,
)

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import (
    legacy_openings_to_boundary_portals,
)


class HouseLayoutCompilationMixin:
    """Compile polygon rooms, meshes, platforms, and heightfields."""

    def compile_polygon_rooms(self, output_dir: Path) -> dict[str, Path]:
        """Compile explicitly polygonal rooms into room-compatible SDF assets.

        Polygon coordinates are authored in a min-corner-local convention, like
        legacy ``RoomSpec.position``.  The compiler recenters them because room
        models are welded to a frame at the footprint bounding-box center.
        """

        from scenesmith.agent_utils.structure.compiler.models import triangle_group_mesh
        from scenesmith.agent_utils.structure.compiler.polygon_spaces import (
            compile_polygon_space,
        )
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )

        compiled_paths: dict[str, Path] = {}
        for room_spec in self.room_specs:
            if (
                room_spec.footprint is None
                and room_spec.floor_footprint is None
                and room_spec.ceiling_footprint is None
                and room_spec.floor_profile == ElevationProfile()
                and room_spec.ceiling_profile is None
                and abs(room_spec.yaw) <= 1e-9
                and not any(
                    heightfield.space_id == room_spec.room_id
                    and heightfield.replaces_floor
                    for heightfield in self.heightfields
                )
            ):
                continue
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
                    f"{footprint_width:g} × {footprint_depth:g}, but width/depth "
                    f"are {room_spec.length:g} × {room_spec.width:g}"
                )

            local_footprint = source_footprint.centered_on_bounds()
            local_floor_footprint = (
                room_spec.floor_footprint or source_footprint
            ).centered_on_bounds()
            local_ceiling_footprint = (
                room_spec.ceiling_footprint or source_footprint
            ).centered_on_bounds()
            authored_portals = [
                portal
                for portal in self.portals
                if portal.source_space_id == room_spec.room_id
                and portal.boundary_loop_index is not None
            ]
            placed_room = self.get_placed_room(room_spec.room_id)
            legacy_portals = (
                legacy_openings_to_boundary_portals(
                    room_spec, placed_room, self.wall_height
                )
                if placed_room is not None
                else []
            )
            authored_ids = {portal.portal_id for portal in authored_portals}
            compiled = compile_polygon_space(
                structure_id=f"room_geometry_{room_spec.room_id}",
                footprint=local_footprint,
                floor_footprint=local_floor_footprint,
                ceiling_footprint=local_ceiling_footprint,
                include_floor=not any(
                    heightfield.space_id == room_spec.room_id
                    and heightfield.replaces_floor
                    for heightfield in self.heightfields
                ),
                include_ceiling=room_spec.has_overhead_cover,
                floor_profile=room_spec.floor_profile,
                ceiling_profile=room_spec.ceiling_profile,
                wall_height=self.wall_height,
                portals=[
                    *authored_portals,
                    *(
                        portal
                        for portal in legacy_portals
                        if portal.portal_id not in authored_ids
                    ),
                ],
            )
            room_materials = self.room_materials.get(room_spec.room_id)
            floor_material = (
                room_materials.floor_material if room_materials is not None else None
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
                if floor_material is not None
                and compiled.triangle_groups.get("floor_top")
                else None
            )
            paths = write_compiled_structure(
                compiled,
                output_dir / room_spec.room_id,
                model_name="room_geometry",
                link_name="room_geometry_body_link",
                visual_overlays=floor_overlays,
            )
            geometry = RoomGeometry(
                sdf_tree=ET.parse(paths.sdf_path),
                sdf_path=paths.sdf_path,
                width=footprint_depth,
                length=footprint_width,
                wall_height=self.wall_height,
                has_overhead_cover=room_spec.has_overhead_cover,
                footprint=local_footprint,
                floor_footprint=local_floor_footprint,
                ceiling_footprint=local_ceiling_footprint,
                floor_profile=room_spec.floor_profile,
                ceiling_profile=room_spec.ceiling_profile,
                structural_surfaces=[patch.surface for patch in compiled.surfaces],
                structural_surface_path=paths.surfaces_path,
                wall_normals={
                    patch.surface.surface_id: np.asarray(patch.normal[:2])
                    for patch in compiled.surfaces
                    if SurfaceRole.BOUNDARY in patch.surface.roles
                },
            )
            self.set_room_geometry(room_spec.room_id, geometry)
            compiled_paths[room_spec.room_id] = paths.sdf_path
        return compiled_paths

    def compile_structural_meshes(
        self, output_dir: Path, *, repair: bool = False
    ) -> dict[str, Path]:
        """Compile cavern/freeform meshes into validated SDF and surface assets."""

        from scenesmith.agent_utils.structure.compiler.mesh_assembly import (
            compile_structural_mesh,
        )
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        # Replacement shells must establish RoomGeometry before additive mesh
        # sidecars are attached, independent of authoring array order.
        ordered_meshes = sorted(
            self.structural_meshes, key=lambda mesh: not mesh.replaces_room_shell
        )
        for mesh_spec in ordered_meshes:
            compiled = compile_structural_mesh(mesh_spec, repair=repair)
            paths = write_compiled_structure(
                compiled,
                output_dir / mesh_spec.mesh_id,
                model_name=("room_geometry" if mesh_spec.replaces_room_shell else None),
                link_name=(
                    "room_geometry_body_link"
                    if mesh_spec.replaces_room_shell
                    else "structure_link"
                ),
            )
            compiled_paths[mesh_spec.mesh_id] = paths.sdf_path
            if mesh_spec.replaces_room_shell:
                bounds_min, bounds_max = compiled.visual_mesh.bounds
                self.set_room_geometry(
                    mesh_spec.space_id,
                    RoomGeometry(
                        sdf_tree=ET.parse(paths.sdf_path),
                        sdf_path=paths.sdf_path,
                        width=bounds_max[1] - bounds_min[1],
                        length=bounds_max[0] - bounds_min[0],
                        wall_height=bounds_max[2] - bounds_min[2],
                        structural_surfaces=[
                            patch.surface for patch in compiled.surfaces
                        ],
                        structural_surface_path=paths.surfaces_path,
                        wall_normals={
                            patch.surface.surface_id: np.asarray(patch.normal[:2])
                            for patch in compiled.surfaces
                            if SurfaceRole.BOUNDARY in patch.surface.roles
                        },
                    ),
                )
            else:
                self._attach_structural_sidecar(mesh_spec.space_id, paths.surfaces_path)
        self.structural_mesh_geometry_paths = compiled_paths
        return dict(compiled_paths)

    def compile_platforms(self, output_dir: Path) -> dict[str, Path]:
        """Compile authored platforms and open edges into static SDF assets."""

        from scenesmith.agent_utils.structure.compiler.surfaces import compile_platform
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        for platform in self.platforms:
            room_materials = self.room_materials.get(platform.space_id)
            floor_material = (
                room_materials.floor_material if room_materials is not None else None
            )
            texture_scale = (
                floor_material.texture_scale or 0.5
                if floor_material is not None
                else 0.5
            )
            paths = write_compiled_structure(
                compile_platform(platform),
                output_dir / platform.platform_id,
                visual_material=floor_material,
                visual_texture_scale=texture_scale,
            )
            compiled_paths[platform.platform_id] = paths.sdf_path
            self._attach_structural_sidecar(platform.space_id, paths.surfaces_path)
        self.platform_geometry_paths = compiled_paths
        return dict(compiled_paths)

    def compile_heightfields(self, output_dir: Path) -> dict[str, Path]:
        """Compile sampled terrain/floors into SDF and semantic sidecars."""

        from scenesmith.agent_utils.structure.compiler.surfaces import (
            compile_heightfield,
        )
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        for heightfield in self.heightfields:
            paths = write_compiled_structure(
                compile_heightfield(heightfield),
                output_dir / heightfield.heightfield_id,
            )
            compiled_paths[heightfield.heightfield_id] = paths.sdf_path
            self._attach_structural_sidecar(heightfield.space_id, paths.surfaces_path)
        self.heightfield_geometry_paths = compiled_paths
        return dict(compiled_paths)

    def _attach_structural_sidecar(self, room_id: str, path: Path) -> None:
        """Attach one room-local compiled surface sidecar without duplication."""

        geometry = self.get_room_geometry(room_id)
        if geometry is None:
            return
        if path not in geometry.additional_structural_surface_paths:
            geometry.additional_structural_surface_paths.append(path)
