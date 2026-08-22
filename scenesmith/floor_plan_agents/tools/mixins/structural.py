"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import json
import logging
import math

from copy import deepcopy
from typing import Any

from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    Footprint2D,
    HeightfieldSpec,
    LevelSpec,
    PlatformSpec,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorSpec,
    PortalSpec,
)
from scenesmith.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenesmith.floor_plan_agents.tools.floor_plan_models import Result
from scenesmith.floor_plan_agents.tools.submission.placement.geometry import (
    rooms_overlap,
)
from scenesmith.floor_plan_agents.tools.submission.placement.layout import (
    create_placed_room,
    update_wall_connectivity,
)

console_logger = logging.getLogger(__name__)


class FloorPlanStructuralMixin:
    """Versioned structural-layout parsing and atomic application."""

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
