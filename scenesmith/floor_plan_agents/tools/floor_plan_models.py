"""Result and configuration models for floor-plan tools."""

from dataclasses import dataclass, field


@dataclass
class RoomSpecsResult:
    success: bool
    message: str
    ascii_floor_plan: str = ""
    wall_segment_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Result:
    success: bool
    message: str


@dataclass
class MaterialResult:
    success: bool
    message: str
    material_id: str = ""


@dataclass
class DoorWindowConfig:
    """Door, window, and exterior-clearance constraints in meters."""

    door_width_min: float = 0.9
    door_width_max: float = 1.9
    door_height_min: float = 2.0
    door_height_max: float = 2.4
    door_default_width: float = 0.9
    door_default_height: float = 2.1
    window_width_min: float = 0.6
    window_width_max: float = 4.0
    window_height_min: float = 0.6
    window_height_max: float = 4.0
    window_default_width: float = 1.2
    window_default_height: float = 1.2
    window_default_sill_height: float = 0.9
    window_segment_margin: float = 0.3
    exterior_door_clearance_m: float = 1.0


@dataclass
class MaterialsListResult:
    success: bool
    message: str
    materials: dict[str, dict[str, str]] = field(default_factory=dict)
    exterior_material_id: str = ""


@dataclass
class ValidationResult:
    layout: str
    connectivity: str
