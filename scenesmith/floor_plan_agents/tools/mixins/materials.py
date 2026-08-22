"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import logging

from scenesmith.agent_utils.scene.house_parts.openings import RoomMaterials
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    PortalSpec,
    PortalType,
)
from scenesmith.agent_utils.structure.structural_topology import (
    EXTERIOR_NODE,
    StructuralTopology,
)
from scenesmith.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenesmith.floor_plan_agents.tools.floor_plan_models import (
    MaterialResult,
    MaterialsListResult,
    Result,
    ValidationResult,
)
from scenesmith.floor_plan_agents.tools.submission.placement.layout import (
    validate_connectivity,
)

console_logger = logging.getLogger(__name__)


class FloorPlanMaterialsMixin:
    """Material resolution, validation, and ASCII rendering operations."""

    def _get_material_impl(self, description: str) -> MaterialResult:
        """Find a material matching the description.

        Args:
            description: What the material should look like (e.g., "warm oak
                hardwood", "white hexagon tile", "red brick").

        Returns:
            MaterialResult with material_id.
        """
        console_logger.info(f"Tool called: get_material(description={description})")

        material = self.materials_resolver.get_material(description)

        if material:
            return MaterialResult(
                success=True,
                message=f"Found material: {material.material_id}",
                material_id=material.material_id,
            )
        else:
            msg = f"No material found matching '{description}'."
            console_logger.info(f"Tool failed: {msg}")
            return MaterialResult(success=False, message=msg)

    def _set_room_materials_impl(
        self, room_id: str, floor_material_id: str, wall_material_id: str
    ) -> Result:
        """Set wall and floor materials for a room using material IDs.

        Get material IDs from get_material() or list_room_materials().

        Args:
            room_id: Room to set materials for.
            floor_material_id: Material ID for floor (e.g., "Wood094_1K-JPG").
            wall_material_id: Material ID for walls (e.g., "Plaster001_1K-JPG").

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: set_room_materials(room_id={room_id}, "
            f"wall_material_id={wall_material_id}, floor_material_id={floor_material_id})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        spec = self.layout.get_room_spec(room_id)
        if not spec:
            return self._fail(f"Room '{room_id}' not found.")

        # Resolve materials.
        wall_mat = self.materials_resolver.get_material_by_id(wall_material_id)
        floor_mat = self.materials_resolver.get_material_by_id(floor_material_id)

        if not wall_mat:
            return self._fail(f"Wall material '{wall_material_id}' not found.")
        if not floor_mat:
            return self._fail(f"Floor material '{floor_material_id}' not found.")

        self.layout.room_materials[room_id] = RoomMaterials(
            wall_material=wall_mat,
            floor_material=floor_mat,
        )

        # Invalidate geometry for this room (materials baked into GLTF textures).
        if self.layout.invalidate_room_geometry(room_id):
            console_logger.debug(f"Invalidated geometry for room: {room_id}")

        return Result(success=True, message=f"Set materials for room '{room_id}'.")

    def _set_exterior_material_impl(self, material_id: str) -> Result:
        """Set material for exterior shell.

        Args:
            material_id: Material ID from get_material() (e.g., "Bricks001_1K-JPG").

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: set_exterior_material(material_id={material_id})"
        )
        material = self.materials_resolver.get_material_by_id(material_id)

        if not material:
            return self._fail(f"Exterior material '{material_id}' not found.")

        self.layout.exterior_material = material

        # Invalidate all geometry (exterior material affects rooms with exterior walls).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        return Result(
            success=True, message=f"Set exterior material to '{material_id}'."
        )

    def _list_room_materials_impl(self) -> MaterialsListResult:
        """List all materials currently assigned to rooms.

        Use to check existing materials before setting new ones for consistency.

        Returns:
            Dict mapping room_id to {wall_material_id, floor_material_id}.
            Also includes exterior_material_id if set.
        """
        console_logger.info("Tool called: list_room_materials")
        materials = {}

        for room_id, room_mat in self.layout.room_materials.items():
            materials[room_id] = {
                "wall_material_id": (
                    room_mat.wall_material.material_id if room_mat.wall_material else ""
                ),
                "floor_material_id": (
                    room_mat.floor_material.material_id
                    if room_mat.floor_material
                    else ""
                ),
            }

        exterior_id = ""
        if self.layout.exterior_material:
            exterior_id = self.layout.exterior_material.material_id

        return MaterialsListResult(
            success=True,
            message=f"Found materials for {len(materials)} room(s).",
            materials=materials,
            exterior_material_id=exterior_id,
        )

    def _validate_impl(self) -> ValidationResult:
        """Validate current design state.

        Call after room layout changes or door changes to catch issues early.
        Also use as final check before completing design.

        Returns:
            ValidationResult with status for each check:
            - layout: room placement (no overlaps, adjacencies satisfied)
            - connectivity: all rooms reachable from exterior via doors
        """
        console_logger.info("Tool called: validate")
        layout_status = "ok"
        connectivity_status = "ok"

        try:
            self.layout.validate_structure()
        except ValueError as exc:
            layout_status = f"error: structural geometry invalid: {exc}"

        # Check layout.
        if not self.layout.placement_valid:
            layout_status = "error: room placement not completed or invalid"
        elif not self.layout.placed_rooms:
            layout_status = "error: no rooms placed"

        # Check connectivity.
        if self.layout.placed_rooms and (
            self.layout.connectors or self.layout.portals or self.layout.platforms
        ):
            topology_portals = list(self.layout.portals)
            topology_portals.extend(
                PortalSpec(
                    portal_id=f"legacy_door_{door.id}",
                    portal_type=PortalType.DOOR,
                    source_space_id=door.room_a,
                    target_space_id=door.room_b,
                    width=door.width,
                    height=door.height,
                )
                for door in self.layout.doors
            )
            topology = StructuralTopology.build(
                space_ids=self.layout.room_ids,
                portals=topology_portals,
                connectors=self.layout.connectors,
            )
            if EXTERIOR_NODE not in topology.nodes:
                connectivity_status = "error: no exterior door or portal"
                self.layout.connectivity_valid = False
            else:
                reachable = topology.reachable(EXTERIOR_NODE, capabilities={"walk"})
                unreachable = sorted(set(self.layout.room_ids) - set(reachable))
                if unreachable:
                    connectivity_status = (
                        "error: structurally unreachable rooms: "
                        + ", ".join(unreachable)
                    )
                    self.layout.connectivity_valid = False
                else:
                    self.layout.connectivity_valid = True
        elif self.layout.placed_rooms:
            is_valid, msg = validate_connectivity(
                self.layout.placed_rooms,
                self.layout.doors,
                self.layout.room_specs,
            )
            if not is_valid:
                connectivity_status = f"error: {msg}"
            self.layout.connectivity_valid = is_valid
        else:
            connectivity_status = "error: no rooms to validate"

        # Log validation result.
        is_valid = layout_status == "ok" and connectivity_status == "ok"
        if is_valid:
            console_logger.info("Validation passed: layout=ok, connectivity=ok")
        else:
            console_logger.info(
                f"Validation failed: layout={layout_status}, "
                f"connectivity={connectivity_status}"
            )

        return ValidationResult(layout=layout_status, connectivity=connectivity_status)

    def _render_ascii_impl(self) -> str:
        """Generate text representation of floor plan.

        Shows room boundaries, room names, and wall segment labels (A, B, C...).
        Use for quick layout overview or when planning door/window placement.

        Returns:
            ASCII floor plan string.
        """
        console_logger.info("Tool called: render_ascii")
        if not self.layout.placed_rooms:
            return "(No rooms to render)"

        result = generate_ascii_floor_plan(self.layout.placed_rooms)
        return f"{result.ascii_art}\n\n{result.legend}"
