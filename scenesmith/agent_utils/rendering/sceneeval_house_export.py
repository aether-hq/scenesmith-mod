"""House-level SceneEval export behavior."""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenesmith.agent_utils.rendering.sceneeval_exporter import (
        SceneEvalExportConfig,
    )
    from scenesmith.agent_utils.scene.house import HouseScene
    from scenesmith.agent_utils.scene.house_parts.openings import Wall

SCENEEVAL_VERSION = "scene@1.0.2"
ARCH_VERSION = "arch@1.0.2"
console_logger = logging.getLogger(__name__)


class SceneEvalHouseExportMixin:
    @classmethod
    def export_house(
        cls,
        house: "HouseScene",
        output_dir: Path,
        config: SceneEvalExportConfig,
    ) -> Path:
        """Export HouseScene to SceneEval format.

        Exports a complete SceneEval scene state including:
        - Architecture (floors, walls with doors/windows)
        - Objects (furniture, manipulands from all rooms)
        - Open room pairs (open floor plan connections)

        Args:
            house: HouseScene to export.
            output_dir: Directory to save sceneeval_state.json.
            config: Export configuration.

        Returns:
            Path to exported sceneeval_state.json.
        """
        # Build combined architecture and objects from all rooms.
        combined_arch = cls._build_house_architecture(house, config)
        combined_objects = cls._build_house_objects(house, config)

        scene_state = {
            "format": "sceneState",
            "scene": {
                "version": SCENEEVAL_VERSION,
                "id": f"scenesmith-{output_dir.name}",
                "unit": 1.0,
                "up": [0, 0, 1],
                "front": [0, 1, 0],
                "assetSource": [config.asset_id_prefix],
                "arch": combined_arch,
                "object": combined_objects,
            },
        }

        output_path = output_dir / "sceneeval_state.json"
        with open(output_path, "w") as f:
            json.dump(scene_state, f, indent=2)
        console_logger.info(f"Saved combined SceneEval state: {output_path}")

        return output_path

    @classmethod
    def _build_house_architecture(
        cls,
        house: "HouseScene",
        config: SceneEvalExportConfig,
    ) -> dict:
        """Build combined architecture for all rooms in a house.

        Returns:
            Architecture dictionary with floors, walls (with doors/windows),
            regions, and open_room_pairs.
        """
        # Local import to avoid circular import.
        from scenesmith.agent_utils.scene.house_parts.openings import ConnectionType

        elements = []
        regions = []
        element_index = 0

        # Get open room pairs from layout.
        open_pairs = []
        for spec in house.layout.room_specs:
            for other_room, conn_type in spec.connections.items():
                if conn_type != ConnectionType.OPEN:
                    continue
                pair = sorted([spec.room_id, other_room])
                if pair not in open_pairs:
                    open_pairs.append(pair)

        # Process each room.
        for room_id in house.rooms:
            placed_room = house.layout.get_placed_room(room_id)
            if not placed_room:
                continue

            room_pos_x, room_pos_y = cls._get_house_room_corner_position(house, room_id)
            wall_indices = []

            # Floor element.
            floor_points = cls._get_room_floor_polygon(
                house, room_id, room_pos_x, room_pos_y
            )
            elements.append(
                {
                    "id": f"floor|{room_id}",
                    "roomId": room_id,
                    "type": "Floor",
                    "depth": config.floor_thickness,
                    "points": floor_points,
                }
            )
            element_index += 1

            # Wall elements with holes.
            for wall in placed_room.walls:
                wall_elem = cls._wall_to_house_element(
                    wall=wall,
                    room_id=room_id,
                    index=element_index,
                    room_offset=(room_pos_x, room_pos_y),
                    wall_height=house.layout.wall_height,
                    wall_thickness=config.wall_thickness,
                )
                elements.append(wall_elem)
                wall_indices.append(element_index)
                element_index += 1

            # Region for this room.
            regions.append(
                {
                    "id": room_id,
                    "type": "Other",
                    "walls": wall_indices,
                }
            )

        arch = {
            "version": ARCH_VERSION,
            "id": "arch",
            "up": [0, 0, 1],
            "front": [0, 1, 0],
            "coords2d": [0, 1],
            "scaleToMeters": 1,
            "elements": elements,
            "regions": regions,
            "holes": [],
        }

        if open_pairs:
            arch["open_room_pairs"] = open_pairs

        return arch

    @classmethod
    def _get_house_room_corner_position(
        cls, house: "HouseScene", room_id: str
    ) -> tuple[float, float]:
        """Get room corner position for multi-room house.

        Used for floor/wall polygon construction where we need the corner
        (min x, min y) as the starting point.

        Single room (room_id="main") is at origin.
        Multi-room uses PlacedRoom positions.
        """
        if len(house.rooms) == 1:
            return (0.0, 0.0)

        placed_room = house.layout.get_placed_room(room_id)
        if placed_room:
            return placed_room.position
        return (0.0, 0.0)

    @classmethod
    def _get_house_room_center_position(
        cls, house: "HouseScene", room_id: str
    ) -> tuple[float, float]:
        """Get room center position for house export.

        Used for object positioning where room geometry is centered at origin,
        so we need the center position to transform room-local coordinates
        to world coordinates (corner-based, matching architecture).

        Computes center from corner position + dimensions/2.
        """
        placed_room = house.layout.get_placed_room(room_id)
        if placed_room:
            # PlacedRoom.width = X dimension, PlacedRoom.depth = Y dimension.
            center_x = placed_room.position[0] + placed_room.width / 2
            center_y = placed_room.position[1] + placed_room.depth / 2
            return (center_x, center_y)
        return (0.0, 0.0)

    @classmethod
    def _get_room_floor_polygon(
        cls,
        house: "HouseScene",
        room_id: str,
        offset_x: float,
        offset_y: float,
    ) -> list[list[float]]:
        """Get floor polygon for a room in world coordinates.

        Args:
            house: HouseScene containing the room.
            room_id: Room ID.
            offset_x: X offset for room position.
            offset_y: Y offset for room position.

        Returns:
            List of 4 corner points in counter-clockwise order.
        """
        placed_room = house.layout.get_placed_room(room_id)
        if not placed_room:
            return []

        # Use PlacedRoom dimensions (accounts for rotation during placement).
        # PlacedRoom.width = X dimension, PlacedRoom.depth = Y dimension.
        min_x = offset_x
        max_x = offset_x + placed_room.width
        min_y = offset_y
        max_y = offset_y + placed_room.depth

        # Counter-clockwise order at z=0.
        return [
            [float(min_x), float(min_y), 0.0],
            [float(max_x), float(min_y), 0.0],
            [float(max_x), float(max_y), 0.0],
            [float(min_x), float(max_y), 0.0],
        ]

    @classmethod
    def _wall_to_house_element(
        cls,
        wall: "Wall",
        room_id: str,
        index: int,
        room_offset: tuple[float, float],
        wall_height: float,
        wall_thickness: float,
    ) -> dict:
        """Convert Wall to SceneEval element with holes.

        Args:
            wall: Wall dataclass with openings.
            room_id: Room ID this wall belongs to.
            index: Element index.
            room_offset: (x, y) offset for room position.
            wall_height: Wall height in meters.
            wall_thickness: Wall thickness in meters.

        Returns:
            Wall element dictionary for SceneEval.
        """
        from scenesmith.agent_utils.scene.house_parts.openings import OpeningType

        offset_x, offset_y = room_offset

        # Wall points in world coordinates.
        p1 = [
            float(wall.start_point[0] + offset_x),
            float(wall.start_point[1] + offset_y),
            0.0,
        ]
        p2 = [
            float(wall.end_point[0] + offset_x),
            float(wall.end_point[1] + offset_y),
            0.0,
        ]

        # Convert openings to holes.
        holes = []
        for opening in wall.openings:
            # Skip OPEN connections - they're in open_room_pairs.
            if opening.opening_type == OpeningType.OPEN:
                continue

            # All opening types use LEFT EDGE convention for position_along_wall.
            x_min = opening.position_along_wall
            x_max = opening.position_along_wall + opening.width
            z_min = opening.sill_height
            z_max = opening.sill_height + opening.height

            hole_type = "Door" if opening.opening_type == OpeningType.DOOR else "Window"
            holes.append(
                {
                    "id": opening.opening_id,
                    "type": hole_type,
                    "box": {
                        "min": [float(x_min), float(z_min)],
                        "max": [float(x_max), float(z_max)],
                    },
                }
            )

        return {
            "id": f"wall|{room_id}|{wall.direction.value}|{index}",
            "roomId": room_id,
            "type": "Wall",
            "height": float(wall_height),
            "depth": float(wall_thickness),
            "points": [p1, p2],
            "holes": holes,
        }

    @classmethod
    def _build_house_objects(
        cls,
        house: "HouseScene",
        config: SceneEvalExportConfig,
    ) -> list[dict]:
        """Build combined objects list from all rooms in a house.

        Args:
            house: HouseScene containing rooms.
            config: Export configuration.

        Returns:
            List of object dictionaries.
        """
        combined_objects = []
        object_index = 0

        for room_id, room in house.rooms.items():
            # Create exporter for this room to reuse _build_objects.
            exporter = cls(
                scene=room,
                scene_dir=room.scene_dir,
                config=config,
                house_layout=house.layout,
            )

            # Get room CENTER position offset for objects.
            # Room geometry is centered at origin, so objects need center offset.
            pos_x, pos_y = cls._get_house_room_center_position(house, room_id)

            # Build objects for this room.
            room_objects = exporter._build_objects()

            # Transform objects to world coordinates.
            for obj in room_objects:
                # Update index.
                obj["index"] = object_index
                object_index += 1

                # Transform position in matrix (column 12, 13 are x, y translation).
                if "transform" in obj and "data" in obj["transform"]:
                    matrix = obj["transform"]["data"]
                    # Column-major: indices 12, 13 are x, y translation.
                    matrix[12] += pos_x
                    matrix[13] += pos_y

                combined_objects.append(obj)

        return combined_objects
