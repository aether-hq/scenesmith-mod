"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import json
import logging

from typing import Any

from agents import FunctionTool, function_tool

from scenesmith.agent_utils.llm.contracts.response_normalization import (
    extract_json_object,
)
from scenesmith.floor_plan_agents.tools.floor_plan_models import (
    MaterialResult,
    MaterialsListResult,
    Result,
    RoomSpecsResult,
    ValidationResult,
)
from scenesmith.floor_plan_agents.tools.submission.floor_plan_normalization import (
    normalize_floor_plan_submission,
)

console_logger = logging.getLogger(__name__)


class FloorPlanToolClosureMixin:
    """Agent-callable closure and submit-tool construction."""

    def _create_tool_closures(self) -> dict:
        """Create tool closures with access to instance data.

        Uses closure pattern to avoid including 'self' in OpenAI function schemas.
        Each tool is a local function that captures self via closure.

        Returns:
            Dictionary mapping tool names to tool functions.
        """

        @function_tool
        def generate_room_specs(room_specs_json: str) -> RoomSpecsResult:
            """Create rooms with the specified dimensions and adjacencies.

            MUST be called first. Room mode: fails if >1 room specified.

            Args:
                room_specs_json: JSON string with list of room specifications. Example:
                    '[{"type": "living_room", "width": 5.0, "depth": 4.0},
                      {"type": "kitchen", "width": 3.0, "depth": 4.0,
                       "connections": {"living_room": "DOOR"}},
                      {"type": "hallway", "width": 2.0, "depth": 6.0,
                       "connections": {"living_room": "DOOR"},
                       "exterior_walls": ["west"]}]'

                    exterior_walls: Optional list of wall directions ("north", "south",
                    "east", "west") that MUST remain exterior (no rooms placed adjacent).
                    Use for rooms needing guaranteed external door access, e.g., a hallway
                    with multiple room connections that still needs an entrance door.

                    has_overhead_cover: Optional boolean. Set false for an open-air
                    garden, courtyard, patio, terrace, yard, or rooftop. Covered or
                    indoor spaces default to true.

            Returns:
                RoomSpecsResult with placed rooms and wall segment labels.
            """
            return self._generate_room_specs_impl(room_specs_json)

        @function_tool
        def resize_room(room_id: str, width: float, depth: float) -> Result:
            """Change a room's dimensions.

            Args:
                room_id: Room to resize (e.g., "living_room").
                width: New width in meters.
                depth: New depth in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._resize_room_impl(room_id=room_id, width=width, depth=depth)

        @function_tool
        def add_adjacency(room_a: str, room_b: str) -> Result:
            """Require two rooms to share a wall.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._add_adjacency_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def remove_adjacency(room_a: str, room_b: str) -> Result:
            """Remove requirement for two rooms to share a wall.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_adjacency_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def add_open_connection(room_a: str, room_b: str) -> Result:
            """Remove wall between rooms for open floor plan (e.g., "living room open
            to kitchen").

            Creates floor-to-ceiling opening with NO wall - do NOT add doors after this.
            Use for: "open to", "open plan", "flows into", "combined" in prompt.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._add_open_connection_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def remove_open_connection(room_a: str, room_b: str) -> Result:
            """Remove an open floor plan connection and restore the wall.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_open_connection_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def set_wall_height(height_meters: float) -> Result:
            """Set the wall height for all rooms.

            Args:
                height_meters: Wall height in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._set_wall_height_impl(height_meters=height_meters)

        @function_tool(strict_mode=False)
        def set_structural_layout(structural_json: dict[str, Any]) -> Result:
            """Add levels, arbitrary footprints, slopes, connectors, and platforms.

            Call after generate_room_specs when the prompt is multilevel, sloped,
            non-rectangular, has stairs/ramps, or includes mezzanines/terrain.
            This operation is atomic: invalid geometry leaves the layout unchanged.

            Args:
                structural_json: Object with optional arrays: levels, rooms,
                    connectors, platforms, portals, heightfields, and
                    structural_meshes, plus an optional semantic_environment
                    object containing regions, chambers, passage networks,
                    physical sky/exterior openings, seeded detail fields, and
                    explicit hero features. Room overrides
                    identify an existing room by id and may include level_id,
                    elevation, house-frame min-corner `position` [x, y],
                    yaw_degrees, boundary `footprint`, independent
                    `floor_footprint`/`ceiling_footprint` slab holes, floor_profile,
                    or ceiling_profile. Connector/portal/platform/heightfield objects
                    use the version-2 serialized structural schema. A structural
                    mesh with `replaces_room_shell: true` becomes the room itself
                    rather than being added inside a rectangular shell. A
                    natural_passage or shaft connector may set
                    `parameters.geometry_embedded: true` when the imported room
                    mesh already embodies its full physical route.

                    Canonical example::

                        {
                          "levels": [
                            {"id": "ground", "elevation": 0,
                             "nominal_height": 4},
                            {"id": "upper", "elevation": 4,
                             "nominal_height": 3}
                          ],
                          "rooms": [
                            {"id": "library", "level_id": "ground",
                             "position": [0, 0]}
                          ],
                          "connectors": [{
                            "id": "stairs", "type": "stairs_straight",
                            "start": {"space_id": "library",
                                      "level_id": "ground",
                                      "position": [1, 1, 0]},
                            "end": {"space_id": "library",
                                    "level_id": "upper",
                                    "position": [7, 1, 4]},
                            "parameters": {"riser_count": 24}
                          }]
                        }

                    Room overrides use `id`; connector endpoints use `space_id`.
                    Pass the object directly, not a JSON-encoded string.

            Returns:
                Result indicating whether the complete structural spec validated.
            """
            return self._set_structural_layout_impl(structural_json)

        @function_tool
        def add_door(
            wall_id: str, position: str, width: float = 0.9, height: float = 2.1
        ) -> Result:
            """Add a door to a wall segment.

            Args:
                wall_id: Wall segment label (e.g., "A", "B") from the ASCII plan.
                position: "left", "center", or "right" third of the wall.
                width: Door width in meters.
                height: Door height in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._add_door_impl(
                wall_id=wall_id, position=position, width=width, height=height
            )

        @function_tool
        def remove_door(door_id: str) -> Result:
            """Remove a door.

            Args:
                door_id: Door identifier to remove.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_door_impl(door_id)

        @function_tool
        def add_window(
            wall_id: str,
            position: str,
            width: float = 1.2,
            height: float = 1.2,
            sill_height: float = 0.9,
            shape: str = "rectangular",
        ) -> Result:
            """Add a window to an exterior wall segment.

            Args:
                wall_id: Wall segment label (e.g., "A", "B") from the ASCII plan.
                position: "left", "center", or "right" third of the wall.
                width: Window width in meters.
                height: Window height in meters.
                sill_height: Height from floor to window bottom in meters.
                shape: Window silhouette: "rectangular" or "arched".

            Returns:
                Result indicating success or failure.
            """
            return self._add_window_impl(
                wall_id=wall_id,
                position=position,
                width=width,
                height=height,
                sill_height=sill_height,
                shape=shape,
            )

        @function_tool
        def remove_window(window_id: str) -> Result:
            """Remove a window.

            Args:
                window_id: Window identifier to remove.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_window_impl(window_id)

        @function_tool
        def get_material(description: str) -> MaterialResult:
            """Search for a material by description.

            Args:
                description: Material description (e.g., "light oak wood floor").

            Returns:
                MaterialResult with material_id if found.
            """
            return self._get_material_impl(description)

        @function_tool
        def set_room_materials(
            room_id: str,
            floor_material_id: str = "",
            wall_material_id: str = "",
        ) -> Result:
            """Set materials for a room's floor and/or walls.

            Args:
                room_id: Room to set materials for.
                floor_material_id: Floor material ID from get_material (empty to skip).
                wall_material_id: Wall material ID from get_material (empty to skip).

            Returns:
                Result indicating success or failure.
            """
            return self._set_room_materials_impl(
                room_id=room_id,
                floor_material_id=floor_material_id,
                wall_material_id=wall_material_id,
            )

        @function_tool
        def set_exterior_material(material_id: str) -> Result:
            """Set exterior wall material.

            Args:
                material_id: Material ID from get_material.

            Returns:
                Result indicating success or failure.
            """
            return self._set_exterior_material_impl(material_id)

        @function_tool
        def list_room_materials() -> MaterialsListResult:
            """List current material assignments for all rooms.

            Returns:
                MaterialsListResult with material assignments.
            """
            return self._list_room_materials_impl()

        @function_tool
        def validate() -> ValidationResult:
            """Validate the floor plan for completeness.

            Returns:
                ValidationResult with any issues found.
            """
            result = self._validate_impl()
            if result.layout == "ok" and result.connectivity == "ok":
                self._checkpoint_if_valid()
            return result

        @function_tool
        def render_ascii() -> str:
            """Generate ASCII representation of the floor plan.

            Returns:
                ASCII floor plan with wall labels and legend.
            """
            return self._render_ascii_impl()

        return {
            "generate_room_specs": generate_room_specs,
            "resize_room": resize_room,
            "add_adjacency": add_adjacency,
            "remove_adjacency": remove_adjacency,
            "add_open_connection": add_open_connection,
            "remove_open_connection": remove_open_connection,
            "set_wall_height": set_wall_height,
            "set_structural_layout": set_structural_layout,
            "add_door": add_door,
            "remove_door": remove_door,
            "add_window": add_window,
            "remove_window": remove_window,
            "get_material": get_material,
            "set_room_materials": set_room_materials,
            "set_exterior_material": set_exterior_material,
            "list_room_materials": list_room_materials,
            "validate": validate,
            "render_ascii": render_ascii,
        }

    def _create_submit_floor_plan_tool(self) -> FunctionTool:
        """Create the one-shot authoring tool used by the fast floor-plan path.

        The original designer exposes every primitive as an LLM tool. That turns a
        seven-step deterministic workflow into seven serial model requests. This
        facade asks the model for design intent once, then applies the primitives
        locally in their required order.
        """

        description = """Submit a complete floor-plan design in one call.

        Prefer room_specs plus optional structural, wall_height_meters,
        windows_per_room, window shape/dimensions, material descriptions, and
        exterior_door_room_id. Common
        aliases, camelCase, JSON-encoded fields, numeric IDs, design/plan envelopes,
        and room maps are accepted. Structural connector endpoints should identify a
        space, level, and 3D position. Doors, windows, materials, safe defaults,
        structural repair, geometry validation, and fallbacks run locally. Call this
        tool exactly once and do not narrate the design.
        """

        async def invoke(_context: Any, arguments_json: str) -> Result:
            try:
                raw = json.loads(extract_json_object(arguments_json))
            except Exception as exc:
                console_logger.warning(
                    "Could not parse floor-plan tool arguments; using prompt "
                    "fallback: %s",
                    exc,
                )
                raw = {}
            normalized = normalize_floor_plan_submission(
                raw,
                prompt=self.layout.house_prompt,
                mode=self.mode,
                room_dim_min=self.room_dim_min,
                room_dim_max=self.room_dim_max,
                wall_height_min=self.wall_height_min,
                wall_height_max=self.wall_height_max,
            )
            for repair in normalized.repairs:
                console_logger.info("Floor-plan input normalized: %s", repair)
            return self._submit_floor_plan_with_fallback(normalized)

        return FunctionTool(
            name="submit_floor_plan",
            description=description,
            params_json_schema={
                "type": "object",
                "properties": {
                    "room_specs": {},
                    "rooms": {},
                    "wall_height_meters": {},
                    "structural": {},
                    "windows_per_room": {},
                    "window_shape": {},
                    "window_width_m": {},
                    "window_height_m": {},
                    "window_sill_height_m": {},
                    "materials": {},
                    "design": {},
                    "plan": {},
                },
                "additionalProperties": True,
            },
            on_invoke_tool=invoke,
            strict_json_schema=False,
        )
