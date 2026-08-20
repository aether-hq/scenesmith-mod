"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import json
import logging
import math
from copy import deepcopy

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from agents import FunctionTool, function_tool

from scenesmith.agent_utils.house import (
    ConnectionType,
    HouseLayout,
    RoomMaterials,
    RoomSpec,
    Wall,
    WallDirection,
    default_ground_level,
)
from scenesmith.agent_utils.llm_harness import extract_json_object
from scenesmith.agent_utils.semantic_environments import SemanticEnvironmentSpec
from scenesmith.agent_utils.structural_geometry import (
    ConnectorSpec,
    Footprint2D,
    HeightfieldSpec,
    LevelSpec,
    PlatformSpec,
    PortalSpec,
    PortalType,
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structural_topology import EXTERIOR_NODE, StructuralTopology
from scenesmith.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenesmith.floor_plan_agents.tools.door_window_mixin import DoorWindowMixin
from scenesmith.floor_plan_agents.tools.materials_resolver import (
    MaterialsConfig,
    MaterialsResolver,
)
from scenesmith.floor_plan_agents.tools.open_plan_mixin import OpenPlanMixin
from scenesmith.floor_plan_agents.tools.floor_plan_submission import (
    NormalizedFloorPlanSubmission,
    normalize_floor_plan_submission,
    synthesize_structural_layout,
)
from scenesmith.floor_plan_agents.tools.room_placement import (
    PlacementConfig,
    PlacementError,
    ScoringWeights,
    create_placed_room,
    get_shared_edge,
    place_rooms,
    rooms_overlap,
    update_wall_connectivity,
    validate_connectivity,
)

console_logger = logging.getLogger(__name__)


@dataclass
class RoomSpecsResult:
    """Result from generate_room_specs tool."""

    success: bool
    message: str
    ascii_floor_plan: str = ""
    wall_segment_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Result:
    """Generic result from floor plan tools."""

    success: bool
    message: str


@dataclass
class MaterialResult:
    """Result from get_material tool."""

    success: bool
    message: str
    material_id: str = ""


@dataclass
class DoorWindowConfig:
    """Configuration for door and window constraints.

    All dimension values are in meters.
    """

    # Door constraints.
    door_width_min: float = 0.9
    door_width_max: float = 1.9
    door_height_min: float = 2.0
    door_height_max: float = 2.4
    door_default_width: float = 0.9
    door_default_height: float = 2.1

    # Window constraints.
    window_width_min: float = 0.6
    window_width_max: float = 4.0
    window_height_min: float = 0.6
    window_height_max: float = 4.0
    window_default_width: float = 1.2
    window_default_height: float = 1.2
    window_default_sill_height: float = 0.9
    window_segment_margin: float = 0.3  # Margin from segment boundary (meters).

    # Exterior door constraints.
    exterior_door_clearance_m: float = 1.0  # Min clearance outside exterior doors.


@dataclass
class MaterialsListResult:
    """Result from list_room_materials tool."""

    success: bool
    message: str
    materials: dict[str, dict[str, str]] = field(default_factory=dict)
    exterior_material_id: str = ""


@dataclass
class ValidationResult:
    """Result from validate tool."""

    layout: str  # "ok" or error message.
    connectivity: str  # "ok" or error message.


class FloorPlanTools(DoorWindowMixin, OpenPlanMixin):
    """Tools for floor plan designer agent.

    Follow the workflow phases:
    1. Room Layout - generate_room_specs, resize_room, add/remove_adjacency
    2. Wall Height - set_wall_height
    3. Doors - add_door, remove_door
    4. Windows - add_window, remove_window
    5. Materials - get_material, set_room_materials, set_exterior_material
    6. Validation - validate
    """

    def __init__(
        self,
        layout: HouseLayout,
        mode: Literal["room", "house"] = "room",
        materials_config: MaterialsConfig | None = None,
        min_opening_separation: float = 0.5,
        placement_timeout_seconds: float = 5.0,
        placement_scoring_weights: ScoringWeights | None = None,
        placement_exterior_wall_clearance_m: float = 20.0,
        door_window_config: DoorWindowConfig | None = None,
        wall_height_min: float = 2.0,
        wall_height_max: float = 12.0,
        room_dim_min: float = 1.5,
        room_dim_max: float = 20.0,
        checkpoint_callback: Callable[[], bool] | None = None,
    ):
        """Initialize floor plan tools.

        Args:
            layout: The HouseLayout to modify.
            mode: "room" (single room) or "house" (multi-room).
            materials_config: Materials resolver configuration.
            min_opening_separation: Minimum gap between door and window on same wall.
            placement_timeout_seconds: Backtracking search timeout for room placement.
            placement_scoring_weights: Weights for layout scoring (compactness, stability).
            placement_exterior_wall_clearance_m: Clearance zone for exterior_walls constraint.
            door_window_config: Door and window constraints configuration.
            wall_height_min: Minimum wall height in meters.
            wall_height_max: Maximum wall height in meters.
            room_dim_min: Minimum room dimension (width or depth) in meters.
            room_dim_max: Maximum room dimension (width or depth) in meters.
            checkpoint_callback: Optional durable checkpoint hook invoked only
                when the current layout is already valid.
        """
        self.layout = layout
        self.mode = mode
        self.materials_resolver = MaterialsResolver(materials_config)
        self.min_opening_separation = min_opening_separation
        self.placement_config = PlacementConfig(
            timeout_seconds=placement_timeout_seconds,
            scoring_weights=placement_scoring_weights or ScoringWeights(),
            exterior_wall_clearance_m=placement_exterior_wall_clearance_m,
        )
        self.door_window_config = door_window_config or DoorWindowConfig()
        self.wall_height_min = wall_height_min
        self.wall_height_max = wall_height_max
        self.room_dim_min = room_dim_min
        self.room_dim_max = room_dim_max
        self.checkpoint_callback = checkpoint_callback

        # Build tools dictionary using closure pattern.
        # This avoids including 'self' in OpenAI function schemas.
        self.tools = self._create_tool_closures()
        self.submit_floor_plan_tool = self._create_submit_floor_plan_tool()

    def _check_rooms_exist(self) -> Result | None:
        """Check if rooms have been defined.

        Returns:
            Error Result if no rooms, None if OK.
        """
        if not self.layout.room_specs:
            return Result(
                success=False,
                message="No rooms defined. Call generate_room_specs first.",
            )
        return None

    def _checkpoint_if_valid(self) -> bool:
        """Persist a valid layout without turning checkpoint I/O into a tool error."""

        if (
            self.checkpoint_callback is None
            or not self.layout.placement_valid
            or not self.layout.connectivity_valid
        ):
            return False
        try:
            return bool(self.checkpoint_callback())
        except Exception as exc:
            console_logger.warning("Floor-plan checkpoint hook failed: %s", exc)
            return False

    def _fail(self, message: str) -> Result:
        """Log failure and return Result with success=False.

        Args:
            message: Failure message to log and return.

        Returns:
            Result with success=False and the message.
        """
        console_logger.info(f"Tool failed: {message}")
        return Result(success=False, message=message)

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

    def _reset_structural_state_for_fallback(self) -> None:
        """Return the shared layout object to its safe flat compatibility state."""

        self.layout.levels = [default_ground_level()]
        self.layout.connectors.clear()
        self.layout.platforms.clear()
        self.layout.portals.clear()
        self.layout.heightfields.clear()
        self.layout.structural_meshes.clear()
        self.layout.connector_geometry_paths.clear()
        self.layout.platform_geometry_paths.clear()
        self.layout.heightfield_geometry_paths.clear()
        self.layout.structural_mesh_geometry_paths.clear()
        self.layout.semantic_environment = None
        self.layout.semantic_environment_geometry_path = None
        self.layout.semantic_detail_geometry_paths.clear()
        self.layout.connectivity_valid = False
        self.layout.invalidate_all_room_geometries()

    def _submit_floor_plan_with_fallback(
        self, submission: NormalizedFloorPlanSubmission
    ) -> Result:
        """Execute normalized intent, then degrade only as far as necessary."""

        def attempt(kwargs: dict[str, Any], *, label: str) -> Result:
            try:
                return self._submit_floor_plan_impl(**kwargs)
            except Exception as exc:
                console_logger.exception(
                    "%s floor-plan attempt raised; continuing to fallback", label
                )
                return Result(
                    success=False,
                    message=f"{label} attempt rejected malformed input: {exc}",
                )

        result = attempt(submission.tool_kwargs(), label="canonical")
        if result.success:
            return result
        failures = [result.message]
        console_logger.warning(
            "Canonical floor-plan submission failed: %s", result.message
        )

        synthesized = synthesize_structural_layout(
            self.layout.house_prompt,
            submission.room_specs,
            submission.wall_height_meters,
            max_total_height=self.wall_height_max,
        )
        if submission.structural is not None and synthesized is not None:
            self._reset_structural_state_for_fallback()
            retry_kwargs = submission.tool_kwargs()
            retry_kwargs["structural"] = synthesized
            result = attempt(retry_kwargs, label="synthesized structural fallback")
            if result.success:
                diagnostics = tuple(synthesized.get("_diagnostics", ()))
                for diagnostic in diagnostics:
                    console_logger.warning(
                        "Structural fallback degradation: %s", diagnostic
                    )
                return Result(
                    success=True,
                    message=(
                        result.message
                        + " Repaired the provider's structural payload with the "
                        "deterministic multi-level fallback."
                        + (" " + " ".join(diagnostics) if diagnostics else "")
                    ),
                )
            failures.append(result.message)
            console_logger.warning(
                "Synthesized structural fallback failed: %s", result.message
            )

        # Geometry is still useful when optional multi-level intent is irreparable.
        # Preserve the full room prompt so downstream furnishing still has all user
        # requirements, and make the degradation explicit in logs/result metadata.
        self._reset_structural_state_for_fallback()
        retry_kwargs = submission.tool_kwargs()
        retry_kwargs["structural"] = None
        retry_kwargs["wall_height_meters"] = max(
            self.wall_height_min,
            min(self.wall_height_max, submission.wall_height_meters),
        )
        result = attempt(retry_kwargs, label="safe flat fallback")
        if result.success:
            return Result(
                success=True,
                message=(
                    result.message
                    + " Used a safe flat structural fallback after: "
                    + " | ".join(dict.fromkeys(failures))
                ),
            )

        failures.append(result.message)
        console_logger.error(
            "All deterministic floor-plan fallbacks failed: %s",
            " | ".join(dict.fromkeys(failures)),
        )
        return Result(
            success=False,
            message="All deterministic floor-plan fallbacks failed: "
            + " | ".join(dict.fromkeys(failures)),
        )

    def _submit_floor_plan_impl(
        self,
        *,
        room_specs: list[dict[str, Any]],
        wall_height_meters: float = 3.0,
        structural: dict[str, Any] | None = None,
        windows_per_room: int = 1,
        window_shape: str = "rectangular",
        window_width_m: float = 1.2,
        window_height_m: float = 1.2,
        window_sill_height_m: float = 0.9,
        floor_material_description: str = "warm wood floor",
        wall_material_description: str = "neutral plaster wall",
        exterior_material_description: str = "neutral exterior plaster",
        exterior_door_room_id: str = "",
    ) -> Result:
        """Apply one model-produced design with deterministic finishing steps."""

        console_logger.info("Tool called: submit_floor_plan")
        if not isinstance(room_specs, list) or not room_specs:
            return self._fail("room_specs must be a non-empty array")
        if not all(isinstance(spec, dict) for spec in room_specs):
            return self._fail("every room_specs entry must be an object")

        rooms_result = self._generate_room_specs_impl(json.dumps(room_specs))
        if not rooms_result.success:
            return self._fail(rooms_result.message)

        if structural:
            if not isinstance(structural, dict):
                return self._fail("structural must be an object when provided")
            structural_result = self._set_structural_layout_impl(structural)
            if not structural_result.success:
                return self._fail(structural_result.message)

            # Models often return a per-storey height while the legacy room shell
            # needs the full vertical extent. Otherwise its ceiling cuts through
            # the first stair flight and the geometric clearance pass vetoes it.
            if self.layout.levels:
                lowest_elevation = min(level.elevation for level in self.layout.levels)
                structural_height = (
                    max(
                        level.elevation + level.nominal_height
                        for level in self.layout.levels
                    )
                    - lowest_elevation
                )
                wall_height_meters = max(float(wall_height_meters), structural_height)

        height_result = self._set_wall_height_impl(float(wall_height_meters))
        if not height_result.success:
            return height_result

        door_error = self._add_required_doors_deterministically(
            exterior_door_room_id=exterior_door_room_id
        )
        if door_error is not None:
            return door_error

        self._add_windows_deterministically(
            windows_per_room=windows_per_room,
            shape=window_shape,
            width=window_width_m,
            height=window_height_m,
            sill_height=window_sill_height_m,
        )
        self._assign_materials_deterministically(
            floor_description=floor_material_description,
            wall_description=wall_material_description,
            exterior_description=exterior_material_description,
        )

        validation = self._validate_impl()
        if validation.layout != "ok" or validation.connectivity != "ok":
            return self._fail(
                "Floor plan validation failed: "
                f"layout={validation.layout}; connectivity={validation.connectivity}"
            )

        checkpoint_saved = self._checkpoint_if_valid()
        if self.checkpoint_callback is not None and not checkpoint_saved:
            return self._fail(
                "The semantic layout passed, but geometry/checkpoint validation "
                "failed; applying a deterministic structural fallback."
            )
        return Result(
            success=True,
            message=(
                f"Completed and validated {len(self.layout.room_specs)} room(s), "
                f"{len(self.layout.doors)} door(s), and "
                f"{len(self.layout.windows)} window(s)."
            ),
        )

    def _add_required_doors_deterministically(
        self, *, exterior_door_room_id: str = ""
    ) -> Result | None:
        """Add required interior and exterior doors without another LLM turn."""

        for wall_id, (room_a, room_b, _direction) in sorted(
            self.layout.boundary_labels.items()
        ):
            if room_b is None:
                continue
            spec_a = self.layout.get_room_spec(room_a)
            spec_b = self.layout.get_room_spec(room_b)
            connection_a = spec_a.connections.get(room_b) if spec_a else None
            connection_b = spec_b.connections.get(room_a) if spec_b else None
            if ConnectionType.DOOR not in {connection_a, connection_b}:
                continue
            if any(
                {door.room_a, door.room_b} == {room_a, room_b}
                for door in self.layout.doors
            ):
                continue
            result = self._try_opening_positions(self._add_door_impl, wall_id=wall_id)
            if result is None or not result.success:
                return self._fail(
                    f"Could not add required door between {room_a} and {room_b}."
                )

        if any(door.room_b is None for door in self.layout.doors):
            return None

        preferred_room = (
            exterior_door_room_id
            if self.layout.get_room_spec(exterior_door_room_id)
            else ""
        )
        candidates = [
            (wall_id, room_a)
            for wall_id, (room_a, room_b, _direction) in sorted(
                self.layout.boundary_labels.items()
            )
            if room_b is None
        ]
        candidates.sort(key=lambda item: item[1] != preferred_room)
        for wall_id, _room_id in candidates:
            result = self._try_opening_positions(self._add_door_impl, wall_id=wall_id)
            if result is not None and result.success:
                return None
        return self._fail("Could not place an exterior door on any exterior wall.")

    @staticmethod
    def _try_opening_positions(operation: Callable[..., Result], *, wall_id: str):
        """Try stable wall segments in preferred order."""

        last_result = None
        for position in ("center", "left", "right"):
            last_result = operation(wall_id=wall_id, position=position)
            if last_result.success:
                return last_result
        return last_result

    def _add_windows_deterministically(
        self,
        *,
        windows_per_room: int,
        shape: str = "rectangular",
        width: float = 1.2,
        height: float = 1.2,
        sill_height: float = 0.9,
    ) -> None:
        """Add requested windows to viable exterior walls, best effort."""

        try:
            target_count = max(0, min(8, int(windows_per_room)))
        except (TypeError, ValueError):
            target_count = 1
        for room in self.layout.room_specs:
            existing_count = sum(
                window.room_id == room.room_id for window in self.layout.windows
            )
            if existing_count >= target_count:
                continue
            candidates = [
                wall_id
                for wall_id, (room_a, room_b, _direction) in sorted(
                    self.layout.boundary_labels.items()
                )
                if room_a == room.room_id
                and room_b is None
                and not any(
                    door.boundary_label == wall_id for door in self.layout.doors
                )
            ]
            for wall_id in candidates:
                for position in ("center", "left", "right"):
                    result = self._add_window_impl(
                        wall_id=wall_id,
                        position=position,
                        width=width,
                        height=height,
                        sill_height=sill_height,
                        shape=shape,
                    )
                    if result.success:
                        existing_count += 1
                        if existing_count >= target_count:
                            break
                if existing_count >= target_count:
                    break
            if existing_count < target_count:
                console_logger.warning(
                    "Placed %d of %d requested windows for room %s",
                    existing_count,
                    target_count,
                    room.room_id,
                )

    def _assign_materials_deterministically(
        self,
        *,
        floor_description: str,
        wall_description: str,
        exterior_description: str,
    ) -> None:
        """Resolve shared material intent once and apply it to every room."""

        floor_material = self._get_material_impl(floor_description)
        wall_material = self._get_material_impl(wall_description)
        if floor_material.success and wall_material.success:
            for room in self.layout.room_specs:
                result = self._set_room_materials_impl(
                    room_id=room.room_id,
                    floor_material_id=floor_material.material_id,
                    wall_material_id=wall_material.material_id,
                )
                if not result.success:
                    console_logger.warning(result.message)

        exterior_material = self._get_material_impl(exterior_description)
        if exterior_material.success:
            result = self._set_exterior_material_impl(exterior_material.material_id)
            if not result.success:
                console_logger.warning(result.message)

    def _set_structural_layout_impl(
        self, structural_json: str | dict[str, Any]
    ) -> Result:
        """Atomically apply version-2 structural authoring data to the layout."""

        console_logger.info("Tool called: set_structural_layout")
        if isinstance(structural_json, str):
            try:
                data = json.loads(structural_json)
            except json.JSONDecodeError as exc:
                return self._fail(f"Invalid structural JSON: {exc}")
        elif isinstance(structural_json, dict):
            # Never rewrite a caller-owned object while repairing common LLM aliases.
            data = deepcopy(structural_json)
        else:
            return self._fail("structural_json must be an object")
        if not isinstance(data, dict):
            return self._fail("structural_json must be a JSON object")
        diagnostics = data.pop("_diagnostics", ())
        if isinstance(diagnostics, (list, tuple)):
            for diagnostic in diagnostics:
                console_logger.warning(
                    "Structural authoring diagnostic: %s", diagnostic
                )

        def normalize_entity_collection(name: str) -> None:
            """Coerce nested provider collections before any iteration occurs."""

            if name not in data or data[name] is None:
                return
            value = data[name]
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    console_logger.warning(
                        "Ignoring malformed JSON string for structural %s", name
                    )
                    value = []
            if isinstance(value, dict):
                entity_markers = {
                    "id",
                    "level_id",
                    "space_id",
                    "room_id",
                    "type",
                    "elevation",
                    "position",
                    "start",
                    "source_space_id",
                    "mesh_path",
                    "heights",
                    "footprint",
                }
                if entity_markers.intersection(value):
                    value = [value]
                else:
                    expanded = []
                    for entity_id, entity in value.items():
                        if not isinstance(entity, dict):
                            continue
                        normalized_entity = dict(entity)
                        normalized_entity.setdefault("id", entity_id)
                        expanded.append(normalized_entity)
                    value = expanded
            elif isinstance(value, tuple):
                value = list(value)
            if not isinstance(value, list):
                console_logger.warning(
                    "Ignoring non-collection structural %s value: %r", name, value
                )
                value = []
            data[name] = value

        for collection_name in (
            "levels",
            "rooms",
            "connectors",
            "platforms",
            "portals",
            "heightfields",
            "structural_meshes",
        ):
            normalize_entity_collection(collection_name)

        def normalize_footprint(value: Any) -> Any:
            """Accept the common `polygon` alias at the authoring boundary."""

            if not isinstance(value, dict) or "outer" in value:
                return value
            if "polygon" not in value:
                return value
            normalized = dict(value)
            normalized["outer"] = normalized.pop("polygon")
            normalized.setdefault("holes", [])
            return normalized

        def stable_id(value: Any, *, prefix: str, index: int) -> str:
            """Return a deterministic schema-safe ID for LLM-authored entities."""

            raw_value = value
            if raw_value is None or not str(raw_value).strip():
                raw_value = f"{prefix}_{index + 1}"
            raw = str(raw_value).strip().lower()
            normalized = "".join(
                character if character.isalnum() or character in {"_", "-"} else "_"
                for character in raw
            ).strip("_-")
            return normalized or f"{prefix}_{index + 1}"

        # Providers routinely use `level_id` for a level definition and `name`
        # for connector identity. Repair bookkeeping aliases before strict schema
        # parsing; neither changes geometry intent.
        levels = data.get("levels", [])
        inferred_room_levels: dict[str, str] = {}
        level_id_aliases: dict[str, str] = {}
        if isinstance(levels, list):
            for index, level in enumerate(levels):
                if not isinstance(level, dict):
                    continue
                authored_level_id = level.get(
                    "id", level.get("level_id", level.get("name"))
                )
                level["id"] = stable_id(
                    authored_level_id,
                    prefix="level",
                    index=index,
                )
                if authored_level_id is not None:
                    level_id_aliases[str(authored_level_id)] = str(level["id"])
                if "nominal_height" not in level and "height" in level:
                    level["nominal_height"] = level["height"]
                level.pop("height", None)
                nested_rooms = level.pop("rooms", [])
                if isinstance(nested_rooms, list):
                    for nested_room in nested_rooms:
                        if not isinstance(nested_room, dict):
                            continue
                        space_id = nested_room.get(
                            "id",
                            nested_room.get("space_id", nested_room.get("room_id")),
                        )
                        if space_id:
                            inferred_room_levels.setdefault(
                                str(space_id), str(level["id"])
                            )
                level.pop("level_id", None)
                level.pop("name", None)

            valid_levels = [level for level in levels if isinstance(level, dict)]
            if valid_levels:
                base_level = min(
                    valid_levels,
                    key=lambda level: float(level.get("elevation", 0.0)),
                )
                for room_spec in self.layout.room_specs:
                    inferred_room_levels.setdefault(
                        room_spec.room_id, str(base_level["id"])
                    )

        level_elevations = {
            str(level.get("id")): float(level.get("elevation", 0.0))
            for level in data.get("levels", [])
            if isinstance(level, dict) and level.get("id") is not None
        }
        space_level_ids = {
            room.room_id: room.level_id for room in self.layout.room_specs
        }

        rooms = data.get("rooms", [])
        if rooms is None:
            rooms = []
        if "rooms" not in data or data.get("rooms") is None:
            data["rooms"] = rooms
        if isinstance(rooms, list):
            authored_room_ids = {
                str(room.get("id", room.get("space_id")))
                for room in rooms
                if isinstance(room, dict)
                and room.get("id", room.get("space_id")) is not None
            }
            for space_id, level_id in inferred_room_levels.items():
                if space_id not in authored_room_ids:
                    rooms.append({"id": space_id, "level_id": level_id})
        if isinstance(rooms, list):
            normalized_rooms: dict[str, dict[str, Any]] = {}
            anonymous_rooms: list[dict[str, Any]] = []
            for room in rooms:
                if not isinstance(room, dict):
                    continue
                if "id" not in room and "space_id" in room:
                    room["id"] = room["space_id"]
                room.pop("space_id", None)
                if room.get("level_id") is not None:
                    authored_level_id = str(room["level_id"])
                    room["level_id"] = level_id_aliases.get(
                        authored_level_id, authored_level_id
                    )
                for field_name in (
                    "footprint",
                    "floor_footprint",
                    "ceiling_footprint",
                ):
                    if field_name in room:
                        room[field_name] = normalize_footprint(room[field_name])

                room_id = str(room.get("id", "")).strip()
                if not room_id:
                    anonymous_rooms.append(room)
                    continue
                existing = normalized_rooms.get(room_id)
                if existing is None:
                    normalized_rooms[room_id] = room
                    continue
                # A single tall space is frequently repeated once per authored
                # level. Keep its lowest-level override; connector endpoints still
                # retain every vertical datum.
                existing_elevation = level_elevations.get(
                    str(existing.get("level_id")), float("inf")
                )
                candidate_elevation = level_elevations.get(
                    str(room.get("level_id")), float("inf")
                )
                if candidate_elevation < existing_elevation:
                    normalized_rooms[room_id] = room

            rooms[:] = [*normalized_rooms.values(), *anonymous_rooms]
            for room in rooms:
                if room.get("id") in space_level_ids and room.get("level_id"):
                    space_level_ids[str(room["id"])] = str(room["level_id"])

        platforms = data.get("platforms", [])
        if isinstance(platforms, list):
            for platform in platforms:
                if not isinstance(platform, dict):
                    continue
                if "space_id" not in platform and "room_id" in platform:
                    platform["space_id"] = platform["room_id"]
                platform.pop("room_id", None)
                # A platform's elevation is absolute; level_id is redundant.
                platform.pop("level_id", None)
                if "footprint" in platform:
                    platform["footprint"] = normalize_footprint(platform["footprint"])

        connectors = data.get("connectors", [])
        if isinstance(connectors, list):
            for connector_index, connector in enumerate(connectors):
                if not isinstance(connector, dict):
                    continue
                connector["id"] = stable_id(
                    connector.get("id", connector.get("name")),
                    prefix="connector",
                    index=connector_index,
                )
                connector.pop("name", None)
                if "clearance_height" not in connector and "clearance" in connector:
                    connector["clearance_height"] = connector["clearance"]
                connector.pop("clearance", None)
                default_space_id = connector.get("space_id", connector.get("room_id"))
                start_alias = connector.get("start", connector.get("source"))
                end_alias = connector.get("end", connector.get("target"))

                def normalize_endpoint(
                    endpoint: Any, *, is_start: bool
                ) -> dict[str, Any]:
                    if isinstance(endpoint, str):
                        normalized: dict[str, Any] = {"space_id": endpoint}
                    elif isinstance(endpoint, dict):
                        normalized = dict(endpoint)
                    else:
                        normalized = {}

                    if "space_id" not in normalized:
                        endpoint_space_aliases = (
                            ("room_id", "source_space_id")
                            if is_start
                            else ("room_id", "target_space_id")
                        )
                        for alias in endpoint_space_aliases:
                            if alias in normalized:
                                normalized["space_id"] = normalized[alias]
                                break
                        else:
                            if default_space_id is not None:
                                normalized["space_id"] = default_space_id

                    level_aliases = (
                        ("from_level", "source_level_id")
                        if is_start
                        else ("to_level", "target_level_id")
                    )
                    if "level_id" not in normalized:
                        for alias in level_aliases:
                            if alias in normalized:
                                normalized["level_id"] = normalized[alias]
                                break
                            if alias in connector:
                                normalized["level_id"] = connector[alias]
                                break
                    if normalized.get("level_id") is not None:
                        authored_level_id = str(normalized["level_id"])
                        normalized["level_id"] = level_id_aliases.get(
                            authored_level_id, authored_level_id
                        )

                    position_alias = "start_position" if is_start else "end_position"
                    if "position" not in normalized and position_alias in connector:
                        normalized["position"] = connector[position_alias]
                    position = normalized.get("position")
                    if isinstance(position, (list, tuple)) and len(position) == 2:
                        elevation = level_elevations.get(
                            str(normalized.get("level_id")), 0.0
                        )
                        normalized["position"] = [*position, elevation]

                    for alias in (
                        "room_id",
                        "source_space_id",
                        "target_space_id",
                        "from_level",
                        "to_level",
                        "source_level_id",
                        "target_level_id",
                    ):
                        normalized.pop(alias, None)
                    return normalized

                start = normalize_endpoint(start_alias, is_start=True)
                end = normalize_endpoint(end_alias, is_start=False)
                parameters = connector.get("parameters", {})
                parameters = dict(parameters) if isinstance(parameters, dict) else {}
                for parameter_name in (
                    "center",
                    "turns",
                    "direction",
                    "riser_count",
                    "riser_counts",
                    "waypoints",
                    "rung_count",
                    "yaw_degrees",
                ):
                    if parameter_name not in parameters and parameter_name in connector:
                        parameters[parameter_name] = connector[parameter_name]

                direction_aliases = {
                    "clockwise": "cw",
                    "counterclockwise": "ccw",
                    "counter-clockwise": "ccw",
                    "anticlockwise": "ccw",
                    "anti-clockwise": "ccw",
                }
                authored_direction = parameters.get("direction")
                if isinstance(authored_direction, str):
                    parameters["direction"] = direction_aliases.get(
                        authored_direction.strip().lower(),
                        authored_direction.strip().lower(),
                    )

                # A frequent spiral authoring form supplies center/radius and puts
                # both endpoints at the center. Convert that unambiguous shorthand
                # to the canonical centerline endpoints required by the compiler.
                if connector.get("type") == "stairs_spiral":
                    center = parameters.get("center")
                    radius = connector.get("radius", parameters.get("radius"))
                    if not isinstance(radius, (int, float)) or radius <= 0:
                        # Haiku commonly supplies the spiral center and stair width
                        # but omits a centerline radius. A width-sized radius keeps
                        # the inner edge positive and is a conservative default.
                        authored_width = connector.get("width", 1.0)
                        try:
                            radius = max(1.0, float(authored_width))
                        except (TypeError, ValueError):
                            radius = 1.0
                        parameters["radius"] = radius
                    turns = parameters.get("turns")
                    direction = parameters.get("direction")
                    if (
                        isinstance(center, (list, tuple))
                        and len(center) == 2
                        and isinstance(radius, (int, float))
                        and radius > 0
                        and isinstance(turns, (int, float))
                        and turns > 0
                        and direction in {"cw", "ccw"}
                    ):
                        start_position = start.get("position")
                        end_position = end.get("position")
                        start_z = (
                            start_position[2]
                            if isinstance(start_position, (list, tuple))
                            and len(start_position) == 3
                            else level_elevations.get(str(start.get("level_id")), 0.0)
                        )
                        end_z = (
                            end_position[2]
                            if isinstance(end_position, (list, tuple))
                            and len(end_position) == 3
                            else level_elevations.get(str(end.get("level_id")), 0.0)
                        )
                        if (
                            not isinstance(start_position, (list, tuple))
                            or len(start_position) < 2
                            or math.dist(start_position[:2], center) < 1e-6
                        ):
                            start_angle = 0.0
                        else:
                            start_angle = math.atan2(
                                start_position[1] - center[1],
                                start_position[0] - center[0],
                            )
                        end_angle = start_angle + (
                            (-1.0 if direction == "cw" else 1.0)
                            * 2.0
                            * math.pi
                            * float(turns)
                        )
                        start["position"] = [
                            center[0] + float(radius) * math.cos(start_angle),
                            center[1] + float(radius) * math.sin(start_angle),
                            start_z,
                        ]
                        end["position"] = [
                            center[0] + float(radius) * math.cos(end_angle),
                            center[1] + float(radius) * math.sin(end_angle),
                            end_z,
                        ]

                connector["start"] = start
                connector["end"] = end
                connector["parameters"] = parameters
                for alias in (
                    "source",
                    "target",
                    "room_id",
                    "space_id",
                    "from_level",
                    "to_level",
                    "source_level_id",
                    "target_level_id",
                    "start_position",
                    "end_position",
                    "center",
                    "radius",
                    "turns",
                    "direction",
                    "riser_count",
                    "riser_counts",
                    "waypoints",
                    "rung_count",
                    "yaw_degrees",
                ):
                    connector.pop(alias, None)

        portals = data.get("portals", [])
        if isinstance(portals, list):
            for portal_index, portal in enumerate(portals):
                if not isinstance(portal, dict):
                    continue
                portal["id"] = stable_id(
                    portal.get("id", portal.get("name")),
                    prefix="portal",
                    index=portal_index,
                )
                portal.pop("name", None)
                if "source_space_id" not in portal:
                    for alias in ("space_id", "room_id", "source_room_id"):
                        if alias in portal:
                            portal["source_space_id"] = portal[alias]
                            break
                if "target_space_id" not in portal and "target_room_id" in portal:
                    portal["target_space_id"] = portal["target_room_id"]
                for alias in (
                    "space_id",
                    "room_id",
                    "source_room_id",
                    "target_room_id",
                ):
                    portal.pop(alias, None)

        unknown = set(data) - {
            "levels",
            "rooms",
            "connectors",
            "platforms",
            "portals",
            "heightfields",
            "structural_meshes",
            "semantic_environment",
        }
        if unknown:
            return self._fail(
                "Unknown structural fields: " + ", ".join(sorted(unknown))
            )

        try:
            levels = (
                [LevelSpec.from_dict(level) for level in data["levels"]]
                if "levels" in data
                else list(self.layout.levels)
            )
            room_overrides = data.get("rooms", [])
            if not isinstance(room_overrides, list):
                raise ValueError("rooms must be an array")
            overrides_by_id = {}
            for override in room_overrides:
                room_id = str(override.get("id", "")).strip()
                if not room_id:
                    raise ValueError("each room override requires id")
                if room_id in overrides_by_id:
                    raise ValueError(f"duplicate room override '{room_id}'")
                overrides_by_id[room_id] = override

            def parse_footprint(footprint_data: dict) -> Footprint2D:
                if "circle" in footprint_data:
                    circle = footprint_data["circle"]
                    return Footprint2D.circle(
                        radius=circle["radius"],
                        chord_tolerance=circle.get("chord_tolerance", 0.02),
                        center=tuple(circle.get("center", (0.0, 0.0))),
                    )
                return Footprint2D.from_dict(footprint_data)

            known_room_ids = {room.room_id for room in self.layout.room_specs}
            unknown_rooms = set(overrides_by_id) - known_room_ids
            if unknown_rooms:
                raise ValueError(
                    "room overrides reference unknown rooms: "
                    + ", ".join(sorted(unknown_rooms))
                )

            updated_specs = []
            for spec in self.layout.room_specs:
                override = overrides_by_id.get(spec.room_id)
                if override is None:
                    updated_specs.append(RoomSpec.from_dict(spec.to_dict()))
                    continue
                allowed_room_fields = {
                    "id",
                    "level_id",
                    "elevation",
                    "yaw_degrees",
                    "position",
                    "footprint",
                    "floor_footprint",
                    "ceiling_footprint",
                    "floor_profile",
                    "ceiling_profile",
                }
                extra = set(override) - allowed_room_fields
                if extra:
                    raise ValueError(
                        f"room '{spec.room_id}' has unknown fields: "
                        + ", ".join(sorted(extra))
                    )
                state = spec.to_dict()
                state["level_id"] = override.get("level_id", spec.level_id)
                state["elevation"] = override.get("elevation", spec.elevation)
                state["yaw"] = math.radians(
                    float(override.get("yaw_degrees", math.degrees(spec.yaw)))
                )
                if "position" in override:
                    state["position"] = override["position"]
                if "footprint" in override:
                    footprint_data = override["footprint"]
                    footprint = parse_footprint(footprint_data)
                    state["footprint"] = footprint.to_dict()
                    min_x, min_y, max_x, max_y = footprint.bounds
                    state["length"] = max_x - min_x
                    state["width"] = max_y - min_y
                if "floor_footprint" in override:
                    state["floor_footprint"] = parse_footprint(
                        override["floor_footprint"]
                    ).to_dict()
                if "ceiling_footprint" in override:
                    state["ceiling_footprint"] = parse_footprint(
                        override["ceiling_footprint"]
                    ).to_dict()
                if "floor_profile" in override:
                    state["floor_profile"] = override["floor_profile"]
                if "ceiling_profile" in override:
                    state["ceiling_profile"] = override["ceiling_profile"]
                updated_specs.append(RoomSpec.from_dict(state))

            connectors = (
                [ConnectorSpec.from_dict(item) for item in data["connectors"]]
                if "connectors" in data
                else list(self.layout.connectors)
            )
            platforms = (
                [PlatformSpec.from_dict(item) for item in data["platforms"]]
                if "platforms" in data
                else list(self.layout.platforms)
            )
            portals = (
                [PortalSpec.from_dict(item) for item in data["portals"]]
                if "portals" in data
                else list(self.layout.portals)
            )
            heightfields = (
                [HeightfieldSpec.from_dict(item) for item in data["heightfields"]]
                if "heightfields" in data
                else list(self.layout.heightfields)
            )
            structural_meshes = (
                [
                    StructuralMeshSpec.from_dict(item)
                    for item in data["structural_meshes"]
                ]
                if "structural_meshes" in data
                else list(self.layout.structural_meshes)
            )
            semantic_environment = (
                SemanticEnvironmentSpec.from_dict(data["semantic_environment"])
                if data.get("semantic_environment") is not None
                else (
                    None
                    if "semantic_environment" in data
                    else self.layout.semantic_environment
                )
            )

            candidate = HouseLayout(
                room_specs=updated_specs,
                levels=levels,
                connectors=connectors,
                structural_meshes=structural_meshes,
                platforms=platforms,
                portals=portals,
                heightfields=heightfields,
                semantic_environment=semantic_environment,
            )
            candidate.validate_structure()
            specs_by_id = {spec.room_id: spec for spec in updated_specs}
            candidate_placed_rooms = []
            for placed in self.layout.placed_rooms:
                spec = specs_by_id[placed.room_id]
                override = overrides_by_id.get(placed.room_id, {})
                position = spec.position if "position" in override else placed.position
                candidate_placed_rooms.append(create_placed_room(spec, position))
            for index, room in enumerate(candidate_placed_rooms):
                for other in candidate_placed_rooms[index + 1 :]:
                    if rooms_overlap(room, other):
                        raise ValueError(
                            f"rooms '{room.room_id}' and '{other.room_id}' overlap "
                            f"on level '{room.level_id}'"
                        )
        except KeyError as exc:
            return self._fail(
                f"Invalid structural layout: missing required field {exc}. "
                "Canonical connector: {id, type, start: {space_id, level_id, "
                "position: [x,y,z]}, end: {space_id, level_id, position: "
                "[x,y,z]}, parameters: {...}}. Canonical portal: {id, type, "
                "source_space_id, target_space_id?}."
            )
        except (TypeError, ValueError) as exc:
            return self._fail(f"Invalid structural layout: {exc}")

        self.layout.levels = levels
        self.layout.room_specs = updated_specs
        self.layout.connectors = connectors
        self.layout.platforms = platforms
        self.layout.portals = portals
        self.layout.heightfields = heightfields
        self.layout.structural_meshes = structural_meshes
        self.layout.semantic_environment = semantic_environment
        self.layout.semantic_environment_geometry_path = None
        self.layout.semantic_detail_geometry_paths.clear()
        self.layout.connector_geometry_paths.clear()
        self.layout.platform_geometry_paths.clear()
        self.layout.heightfield_geometry_paths.clear()
        self.layout.structural_mesh_geometry_paths.clear()
        self.layout.invalidate_all_room_geometries()

        self.layout.placed_rooms = candidate_placed_rooms
        update_wall_connectivity(candidate_placed_rooms)
        # Rebuild compatibility walls after transforms/dimensions change, then
        # restore any pre-existing cardinal openings that still fit.
        self._reapply_openings_to_walls()
        ascii_result = generate_ascii_floor_plan(candidate_placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        result = Result(
            success=True,
            message=(
                f"Applied structural layout: {len(levels)} levels, "
                f"{len(connectors)} connectors, {len(platforms)} platforms, "
                f"{len(heightfields)} heightfields, {len(structural_meshes)} "
                f"structural meshes, {int(semantic_environment is not None)} "
                f"semantic environment, and {len(portals)} portals."
            ),
        )
        self._checkpoint_if_valid()
        return result

    def _generate_room_specs_impl(self, room_specs_json: str) -> RoomSpecsResult:
        """Create rooms with the specified dimensions and adjacencies.

        MUST be called first. Room mode: fails if >1 room specified.

        Args:
            room_specs_json: JSON string with list of room specifications.
                Each room must have: type and prompt (required).
                Optional fields: width, depth, connections.
                connections: dict mapping room_id to "DOOR" or "OPEN".
                Example:
                '[{"type": "living_room", "width": 5.0, "depth": 4.0,
                   "prompt": "A cozy modern living room with large windows."},
                  {"type": "kitchen", "width": 3.0, "depth": 4.0,
                   "prompt": "A bright kitchen with white cabinets.",
                   "connections": {"living_room": "DOOR"}}]'

        Returns:
            RoomSpecsResult with placed rooms and wall segment labels.
        """
        # Format JSON for readable logging.
        try:
            parsed_for_log = json.loads(room_specs_json)
            formatted_json = json.dumps(parsed_for_log, indent=2)
        except json.JSONDecodeError:
            formatted_json = room_specs_json  # Use raw string if invalid JSON.
        console_logger.info(
            f"Tool called: generate_room_specs(room_specs_json=\n{formatted_json})"
        )
        # Parse JSON input.
        try:
            room_specs = json.loads(room_specs_json)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON: {e}"
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        if not isinstance(room_specs, list):
            msg = "room_specs_json must be a JSON array of room specifications."
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Mode check.
        if self.mode == "room" and len(room_specs) > 1:
            msg = "Room mode: only 1 room allowed. Use house mode for multiple rooms."
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Convert to RoomSpec objects.
        specs = []
        room_type_counts: dict[str, int] = {}
        for spec_dict in room_specs:
            room_type = spec_dict.get("type", "room")

            # Generate room_id from type.
            room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            if room_type_counts[room_type] == 1:
                room_id = room_type
            else:
                room_id = f"{room_type}_{room_type_counts[room_type]}"

            # In room mode, always use house prompt directly (ignore agent's prompt).
            # In house mode, prompt is required for each room.
            if self.mode == "room":
                prompt = self.layout.house_prompt
            else:
                prompt = spec_dict.get("prompt", "")
                if not prompt:
                    msg = (
                        f"Room '{room_type}' is missing required 'prompt' field. "
                        f"Each room must have a descriptive prompt."
                    )
                    console_logger.info(f"Tool failed: {msg}")
                    return RoomSpecsResult(success=False, message=msg)

            # Validate room dimensions.
            room_width = spec_dict.get("width", 5.0)
            room_depth = spec_dict.get("depth", 4.0)
            if not (self.room_dim_min <= room_width <= self.room_dim_max):
                msg = (
                    f"Room '{room_type}' width must be {self.room_dim_min}-"
                    f"{self.room_dim_max}m. Got: {room_width}"
                )
                console_logger.info(f"Tool failed: {msg}")
                return RoomSpecsResult(success=False, message=msg)
            if not (self.room_dim_min <= room_depth <= self.room_dim_max):
                msg = (
                    f"Room '{room_type}' depth must be {self.room_dim_min}-"
                    f"{self.room_dim_max}m. Got: {room_depth}"
                )
                console_logger.info(f"Tool failed: {msg}")
                return RoomSpecsResult(success=False, message=msg)

            # Parse connections from JSON.
            connections_raw = spec_dict.get("connections", {})
            connections = {k: ConnectionType(v) for k, v in connections_raw.items()}

            # Parse exterior_walls constraint from JSON.
            exterior_walls_raw = spec_dict.get("exterior_walls", [])
            exterior_walls = {WallDirection(w) for w in exterior_walls_raw}

            explicit_cover = spec_dict.get(
                "has_overhead_cover",
                spec_dict.get(
                    "has_roof", spec_dict.get("has_ceiling", spec_dict.get("covered"))
                ),
            )
            if isinstance(explicit_cover, bool):
                has_overhead_cover = explicit_cover
            else:
                outdoor_terms = {
                    "garden",
                    "courtyard",
                    "patio",
                    "terrace",
                    "yard",
                    "rooftop",
                }
                covered_terms = {"covered", "roofed", "indoor", "enclosed"}
                semantic_text = f"{room_type} {prompt}".casefold()
                has_overhead_cover = not (
                    any(term in semantic_text for term in outdoor_terms)
                    and not any(term in semantic_text for term in covered_terms)
                )

            specs.append(
                RoomSpec(
                    room_id=room_id,
                    room_type=room_type,
                    prompt=prompt,
                    width=room_depth,  # Y dimension.
                    length=room_width,  # X dimension.
                    connections=connections,
                    exterior_walls=exterior_walls,
                    has_overhead_cover=has_overhead_cover,
                )
            )

        # Run placement algorithm.
        try:
            placed_rooms = place_rooms(room_specs=specs, config=self.placement_config)
        except PlacementError as e:
            msg = f"Room placement failed: {e}"
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Update layout.
        self.layout.room_specs = specs
        self.layout.placed_rooms = placed_rooms
        self.layout.placement_valid = True

        # Clear openings from previous layout (wall labels may have changed).
        self.layout.doors.clear()
        self.layout.windows.clear()

        # Invalidate all cached geometry (room specs completely changed).
        invalidated = self.layout.invalidate_all_room_geometries()
        console_logger.info(f"Invalidated {invalidated} room geometries")

        # Generate ASCII floor plan.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Log ASCII for visibility during runs.
        console_logger.info(
            "Floor plan layout:\n%s\n%s", ascii_result.ascii_art, ascii_result.legend
        )

        # Build wall segment labels description.
        labels_desc = {}
        for label, (room_a, room_b, direction) in ascii_result.boundary_labels.items():
            if room_b:
                labels_desc[label] = f"Interior: {room_a} <-> {room_b}"
            else:
                dir_str = f" ({direction})" if direction else ""
                labels_desc[label] = f"Exterior: {room_a}{dir_str}"

        return RoomSpecsResult(
            success=True,
            message=f"Created {len(specs)} room(s) successfully.",
            ascii_floor_plan=ascii_result.ascii_art,
            wall_segment_labels=labels_desc,
        )

    def _resize_room_impl(self, room_id: str, width: float, depth: float) -> Result:
        """Change a room's dimensions with layout stability.

        Doors and windows are preserved when possible:
        - Openings on walls whose length changed are proportionally repositioned
        - Openings that no longer fit after repositioning are removed
        - Open connections are preserved (positions recomputed from shared edges)

        Other rooms stay in approximately the same positions unless adjacency
        constraints require movement.

        Args:
            room_id: Room to resize (e.g., "living_room").
            width: New width in meters (within configured range).
            depth: New depth in meters (within configured range).

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: resize_room(room_id={room_id}, width={width}, depth={depth})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        # Validate dimensions.
        if not (self.room_dim_min <= width <= self.room_dim_max):
            return self._fail(
                f"Width must be {self.room_dim_min}-{self.room_dim_max}m. Got: {width}"
            )
        if not (self.room_dim_min <= depth <= self.room_dim_max):
            return self._fail(
                f"Depth must be {self.room_dim_min}-{self.room_dim_max}m. Got: {depth}"
            )

        # Find room spec.
        spec = self.layout.get_room_spec(room_id)
        if not spec:
            return self._fail(f"Room '{room_id}' not found.")

        # Store old state for rollback on failure.
        old_width = spec.length  # X dimension.
        old_depth = spec.width  # Y dimension.
        old_placed_rooms = self.layout.placed_rooms

        # Temporarily update dimensions for placement attempt.
        spec.length = width  # X dimension.
        spec.width = depth  # Y dimension.

        # Re-run placement with layout stability (other rooms stay in place).
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={r.room_id: r.position for r in old_placed_rooms},
                free_room_ids={room_id},
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: restore old dimensions, keep layout valid.
            spec.length = old_width
            spec.width = old_depth
            # Note: placed_rooms unchanged since assignment only happens on success.
            return self._fail(f"Resize failed (layout unchanged): {e}")

        # Invalidate geometry for resized room (dimensions changed).
        if self.layout.invalidate_room_geometry(room_id):
            console_logger.debug(f"Invalidated geometry for resized room: {room_id}")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Proportionally adjust opening positions for walls whose length changed.
        self._adjust_opening_positions_for_resize(
            room_id=room_id,
            old_width=old_width,
            old_depth=old_depth,
            new_width=width,
            new_depth=depth,
        )

        # Reapply all doors/windows and open connections. This validates positions
        # and removes openings that no longer fit after resize.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        # Build result message with removal info.
        msg = f"Room '{room_id}' resized to {width}m x {depth}m."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _add_adjacency_impl(self, room_a: str, room_b: str) -> Result:
        """Require two rooms to share a wall.

        Room mode: fails (single room has no adjacencies).

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: add_adjacency(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no adjacencies for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # Track whether we need to add connections (for rollback on failure).
        added_b_to_a = room_b not in spec_a.connections
        added_a_to_b = room_a not in spec_b.connections

        # Add connection with DOOR type.
        if added_b_to_a:
            spec_a.connections[room_b] = ConnectionType.DOOR
        if added_a_to_b:
            spec_b.connections[room_a] = ConnectionType.DOOR

        # If rooms are already placed and already adjacent, no re-placement needed.
        if self.layout.placed_rooms:
            placed_a = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_a), None
            )
            placed_b = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_b), None
            )
            if placed_a and placed_b:
                shared_edge = get_shared_edge(placed_a, placed_b)
                if shared_edge:
                    # Already adjacent - constraint already satisfied.
                    msg = f"Added adjacency: {room_a} <-> {room_b}."
                    return Result(success=True, message=msg)

        # Rooms not yet placed or not currently adjacent - run placement with stability.
        # Both rooms being made adjacent should have freedom to move.
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={
                    r.room_id: r.position for r in self.layout.placed_rooms
                },
                free_room_ids={room_a, room_b},
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: remove connections we added.
            if added_b_to_a:
                del spec_a.connections[room_b]
            if added_a_to_b:
                del spec_b.connections[room_a]
            return self._fail(f"Add adjacency failed (layout unchanged): {e}")

        # Invalidate all geometry (adjacency affects positions and wall labels).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore openings (doors, windows, open connections) to new walls.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Added adjacency: {room_a} <-> {room_b}."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _remove_adjacency_impl(self, room_a: str, room_b: str) -> Result:
        """Remove requirement for two rooms to share a wall.

        Room mode: fails.

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: remove_adjacency(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no adjacencies for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # Track connections we remove (for rollback on failure).
        removed_b_from_a = spec_a.connections.pop(room_b, None)
        removed_a_from_b = spec_b.connections.pop(room_a, None)

        # If rooms are already placed, no re-placement needed.
        # Removing a constraint doesn't invalidate existing placement.
        if self.layout.placed_rooms:
            msg = f"Removed adjacency: {room_a} <-> {room_b}."
            return Result(success=True, message=msg)

        # Rooms not yet placed - run placement with stability.
        # No special rooms need freedom since we're just removing a constraint.
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={
                    r.room_id: r.position for r in self.layout.placed_rooms
                },
                free_room_ids=set(),
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: restore connections we removed.
            if removed_b_from_a is not None:
                spec_a.connections[room_b] = removed_b_from_a
            if removed_a_from_b is not None:
                spec_b.connections[room_a] = removed_a_from_b
            return self._fail(f"Remove adjacency failed (layout unchanged): {e}")

        # Invalidate all geometry (adjacency affects positions and wall labels).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore openings (doors, windows, open connections) to new walls.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Removed adjacency: {room_a} <-> {room_b}."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _set_wall_height_impl(self, height_meters: float) -> Result:
        """Set wall/ceiling height for entire house.

        Args:
            height_meters: Height in valid range (from config).

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: set_wall_height(height_meters={height_meters})"
        )
        if not (self.wall_height_min <= height_meters <= self.wall_height_max):
            return self._fail(
                f"Height must be between {self.wall_height_min} and "
                f"{self.wall_height_max}m. Got: {height_meters}"
            )

        self.layout.wall_height = height_meters

        # Invalidate all geometry (wall height affects all rooms).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        return Result(success=True, message=f"Wall height set to {height_meters}m.")

    def _get_wall_by_boundary(self, wall_label: str, room_id: str) -> Wall:
        """Get a Wall object by boundary label and room ID.

        Args:
            wall_label: Boundary label (e.g., "A", "B") from boundary_labels.
            room_id: Room ID owning the wall.

        Returns:
            The Wall object.

        Raises:
            ValueError: If wall cannot be found (fail-fast per CLAUDE.md).
        """
        # Look up what this boundary label refers to.
        boundary_info = self.layout.boundary_labels.get(wall_label)
        if not boundary_info:
            raise ValueError(
                f"Unknown wall label '{wall_label}'. "
                f"Available labels: {list(self.layout.boundary_labels.keys())}"
            )

        room_a_label, room_b_label, direction = boundary_info

        # Determine which room we're looking for on the wall.
        # If room_id is room_a, look for wall facing room_b.
        # If room_id is room_b, look for wall facing room_a.
        if room_id == room_a_label:
            target_room = room_b_label
        elif room_id == room_b_label:
            target_room = room_a_label
        else:
            # room_id doesn't match either end of this boundary.
            raise ValueError(
                f"Room '{room_id}' is not part of boundary '{wall_label}' "
                f"which connects {room_a_label} <-> {room_b_label}."
            )

        # Find the placed room.
        placed_room = None
        for placed in self.layout.placed_rooms:
            if placed.room_id == room_id:
                placed_room = placed
                break

        if not placed_room:
            raise ValueError(f"Room '{room_id}' not found in placed_rooms.")

        # Find the wall matching this boundary.
        for wall in placed_room.walls:
            if target_room is None:
                # Exterior wall - match by direction.
                if wall.is_exterior and wall.direction.value == direction:
                    return wall
            else:
                # Interior wall - check if this wall faces the target room.
                if target_room in wall.faces_rooms:
                    return wall

        raise ValueError(
            f"Wall for boundary '{wall_label}' not found in room '{room_id}'."
        )

    def _get_wall_length(self, room_id: str, wall_label: str) -> float:
        """Get the length of a wall by room and boundary label.

        Args:
            room_id: Room ID owning the wall.
            wall_label: Boundary label (e.g., "A", "B") from boundary_labels.

        Returns:
            Wall length in meters.

        Raises:
            ValueError: If wall cannot be found (fail-fast per CLAUDE.md).
        """
        wall = self._get_wall_by_boundary(wall_label=wall_label, room_id=room_id)
        return wall.length

    def _wall_faces_nearby_room(
        self, room_id: str, wall_label: str, threshold: float = 0.5
    ) -> str | None:
        """Check if an exterior wall faces another room within threshold distance.

        Used to prevent windows on walls that technically exterior but face another
        room across a small gap (e.g., 10cm gap between non-adjacent rooms).

        Args:
            room_id: Room owning the wall.
            wall_label: Boundary label from boundary_labels.
            threshold: Maximum distance (meters) to consider as "facing".

        Returns:
            Room ID of nearby room if found, None otherwise.
        """
        wall = self._get_wall_by_boundary(wall_label=wall_label, room_id=room_id)
        placed_room = next(
            (r for r in self.layout.placed_rooms if r.room_id == room_id), None
        )
        if not placed_room:
            return None

        # Get wall position based on direction.
        for other in self.layout.placed_rooms:
            if other.room_id == room_id:
                continue

            other_min_x = other.position[0]
            other_max_x = other.position[0] + other.width
            other_min_y = other.position[1]
            other_max_y = other.position[1] + other.depth

            # Check if wall faces this room within threshold.
            if wall.direction == WallDirection.NORTH:
                # Wall at y = placed_room.position[1] + depth, facing +Y.
                wall_y = placed_room.position[1] + placed_room.depth
                # Check if other room's south edge is within threshold.
                if 0 < other_min_y - wall_y <= threshold:
                    # Check x overlap.
                    if max(wall.start_point[0], other_min_x) < min(
                        wall.end_point[0], other_max_x
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.SOUTH:
                # Wall at y = placed_room.position[1], facing -Y.
                wall_y = placed_room.position[1]
                # Check if other room's north edge is within threshold.
                if 0 < wall_y - other_max_y <= threshold:
                    if max(wall.start_point[0], other_min_x) < min(
                        wall.end_point[0], other_max_x
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.EAST:
                # Wall at x = placed_room.position[0] + width, facing +X.
                wall_x = placed_room.position[0] + placed_room.width
                # Check if other room's west edge is within threshold.
                if 0 < other_min_x - wall_x <= threshold:
                    if max(wall.start_point[1], other_min_y) < min(
                        wall.end_point[1], other_max_y
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.WEST:
                # Wall at x = placed_room.position[0], facing -X.
                wall_x = placed_room.position[0]
                # Check if other room's east edge is within threshold.
                if 0 < wall_x - other_max_x <= threshold:
                    if max(wall.start_point[1], other_min_y) < min(
                        wall.end_point[1], other_max_y
                    ):
                        return other.room_id

        return None

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
