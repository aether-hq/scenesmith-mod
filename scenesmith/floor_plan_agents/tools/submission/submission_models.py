"""Canonical one-shot floor-plan submission models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class NormalizedFloorPlanSubmission:
    room_specs: list[dict[str, Any]]
    wall_height_meters: float
    structural: dict[str, Any] | None
    windows_per_room: int
    window_shape: Literal["rectangular", "arched"]
    window_width_m: float
    window_height_m: float
    window_sill_height_m: float
    floor_material_description: str
    wall_material_description: str
    exterior_material_description: str
    exterior_door_room_id: str
    repairs: tuple[str, ...] = ()

    def tool_kwargs(self) -> dict[str, Any]:
        """Return only arguments accepted by the deterministic executor."""

        return {
            "room_specs": self.room_specs,
            "wall_height_meters": self.wall_height_meters,
            "structural": self.structural,
            "windows_per_room": self.windows_per_room,
            "window_shape": self.window_shape,
            "window_width_m": self.window_width_m,
            "window_height_m": self.window_height_m,
            "window_sill_height_m": self.window_sill_height_m,
            "floor_material_description": self.floor_material_description,
            "wall_material_description": self.wall_material_description,
            "exterior_material_description": self.exterior_material_description,
            "exterior_door_room_id": self.exterior_door_room_id,
        }
