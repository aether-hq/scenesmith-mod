"""Procedural SDF geometry for locked semantic blueprint groups."""

from __future__ import annotations

import hashlib
import math
import re

from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_") or "object"


def _material(
    rgba: tuple[float, float, float, float],
) -> str:
    values = " ".join(str(value) for value in rgba)
    return (
        f"<material><ambient>{values}</ambient><diffuse>{values}</diffuse></material>"
    )


def _box(
    name: str,
    size: tuple[float, float, float],
    pose: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    *,
    collision: bool = True,
) -> str:
    size_text = " ".join(f"{value:.6g}" for value in size)
    pose_text = " ".join(f"{value:.6g}" for value in (*pose, 0.0, 0.0, 0.0))
    collision_xml = (
        f"<collision name='{name}_collision'><pose>{pose_text}</pose>"
        f"<geometry><box><size>{size_text}</size></box></geometry></collision>"
        if collision
        else ""
    )
    return (
        f"<visual name='{name}_visual'><pose>{pose_text}</pose>"
        f"<geometry><box><size>{size_text}</size></box></geometry>"
        f"{_material(rgba)}</visual>{collision_xml}"
    )


def _cylinder(
    name: str,
    radius: float,
    length: float,
    pose: tuple[float, float, float, float, float, float],
    rgba: tuple[float, float, float, float],
    *,
    collision: bool = True,
) -> str:
    pose_text = " ".join(f"{value:.6g}" for value in pose)
    geometry = (
        f"<geometry><cylinder><radius>{radius:.6g}</radius>"
        f"<length>{length:.6g}</length></cylinder></geometry>"
    )
    collision_xml = (
        f"<collision name='{name}_collision'><pose>{pose_text}</pose>"
        f"{geometry}</collision>"
        if collision
        else ""
    )
    return (
        f"<visual name='{name}_visual'><pose>{pose_text}</pose>{geometry}"
        f"{_material(rgba)}</visual>{collision_xml}"
    )


def _variant_index(instance_index: int, instance_prompt: dict[str, Any] | None) -> int:
    prompt = str((instance_prompt or {}).get("construction_prompt") or "")
    digest = hashlib.sha1(prompt.encode("utf-8"), usedforsecurity=False).digest()
    return (instance_index + digest[0]) % 10


def _zone_geometry(
    dimensions: tuple[float, float, float],
    *,
    variant: int = 0,
) -> str:
    x, y, z = dimensions
    steel = (0.12, 0.16, 0.20, 1.0)
    hazard = (0.95, 0.55, 0.04, 1.0)
    pieces = [
        _box(
            "service_pad",
            (x, y, 0.16),
            (0.0, 0.0, 0.08),
            steel,
            collision=False,
        ),
        _box("back_wall", (x, 0.30, z), (0.0, -y / 2.0 + 0.15, z / 2.0), steel),
        _box(
            "left_wall", (0.30, y, z * 0.55), (-x / 2.0 + 0.15, 0.0, z * 0.275), steel
        ),
        _box(
            "right_wall", (0.30, y, z * 0.55), (x / 2.0 - 0.15, 0.0, z * 0.275), steel
        ),
        _box(
            "left_pylon",
            (0.65, 0.65, z),
            (-x / 2.0 + 0.325, y / 2.0 - 0.325, z / 2.0),
            hazard,
        ),
        _box(
            "right_pylon",
            (0.65, 0.65, z),
            (x / 2.0 - 0.325, y / 2.0 - 0.325, z / 2.0),
            hazard,
        ),
        _box("header", (x, 0.65, 0.65), (0.0, y / 2.0 - 0.325, z - 0.325), hazard),
    ]
    fixture_variant = variant % 5
    if fixture_variant == 0:
        pieces.extend(
            (
                _box(
                    "overhead_crane_rail",
                    (x * 0.72, 0.45, 0.45),
                    (0.0, 0.0, z * 0.84),
                    hazard,
                ),
                _cylinder(
                    "crane_drop",
                    0.16,
                    z * 0.28,
                    (x * 0.18, 0.0, z * 0.68, 0.0, 0.0, 0.0),
                    steel,
                ),
            )
        )
    elif fixture_variant == 1:
        pieces.extend(
            (
                _box(
                    "utility_spine",
                    (x * 0.58, 0.55, z * 0.18),
                    (0.0, -y * 0.43, z * 0.58),
                    hazard,
                ),
                _box(
                    "utility_cabinet",
                    (x * 0.18, y * 0.18, z * 0.35),
                    (-x * 0.34, -y * 0.36, z * 0.175),
                    steel,
                ),
            )
        )
    elif fixture_variant == 2:
        pieces.extend(
            (
                _box(
                    "service_trench",
                    (x * 0.60, y * 0.12, 0.08),
                    (0.0, 0.0, 0.18),
                    hazard,
                    collision=False,
                ),
                _box(
                    "trench_console",
                    (x * 0.16, y * 0.16, z * 0.28),
                    (x * 0.34, -y * 0.33, z * 0.14),
                    steel,
                ),
            )
        )
    elif fixture_variant == 3:
        pieces.extend(
            (
                _box(
                    "inspection_platform",
                    (x * 0.36, y * 0.16, 0.24),
                    (-x * 0.22, -y * 0.30, z * 0.36),
                    steel,
                ),
                _box(
                    "service_boom",
                    (x * 0.42, 0.30, 0.30),
                    (x * 0.10, -y * 0.30, z * 0.62),
                    hazard,
                ),
            )
        )
    else:
        pieces.extend(
            (
                _box(
                    "isolation_station",
                    (x * 0.15, y * 0.20, z * 0.42),
                    (-x * 0.35, -y * 0.34, z * 0.21),
                    hazard,
                ),
                _cylinder(
                    "hose_reel",
                    min(x, y) * 0.065,
                    0.28,
                    (x * 0.34, -y * 0.38, z * 0.48, math.pi / 2.0, 0.0, 0.0),
                    steel,
                    collision=False,
                ),
            )
        )
    return "".join(pieces)


def _fighter_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    hull = (0.24, 0.31, 0.38, 1.0)
    dark = (0.05, 0.07, 0.10, 1.0)
    accent = (0.92, 0.42, 0.04, 1.0)
    canopy = (0.08, 0.35, 0.50, 1.0)
    return "".join(
        (
            _box(
                "fuselage", (x * 0.74, y * 0.34, z * 0.24), (0.0, 0.0, z * 0.30), hull
            ),
            _box(
                "nose",
                (x * 0.23, y * 0.22, z * 0.17),
                (x * 0.43, 0.0, z * 0.30),
                accent,
            ),
            _box("wings", (x * 0.42, y, z * 0.07), (-x * 0.06, 0.0, z * 0.28), dark),
            _box(
                "canopy",
                (x * 0.22, y * 0.28, z * 0.17),
                (x * 0.16, 0.0, z * 0.48),
                canopy,
                collision=False,
            ),
            _box(
                "tail", (x * 0.20, y * 0.48, z * 0.30), (-x * 0.39, 0.0, z * 0.42), hull
            ),
            _cylinder(
                "engine_left",
                y * 0.11,
                x * 0.28,
                (-x * 0.30, -y * 0.22, z * 0.31, 0.0, math.pi / 2.0, 0.0),
                dark,
            ),
            _cylinder(
                "engine_right",
                y * 0.11,
                x * 0.28,
                (-x * 0.30, y * 0.22, z * 0.31, 0.0, math.pi / 2.0, 0.0),
                dark,
            ),
            _cylinder(
                "engine_glow_left",
                y * 0.075,
                0.10,
                (-x * 0.445, -y * 0.22, z * 0.31, 0.0, math.pi / 2.0, 0.0),
                accent,
                collision=False,
            ),
            _cylinder(
                "engine_glow_right",
                y * 0.075,
                0.10,
                (-x * 0.445, y * 0.22, z * 0.31, 0.0, math.pi / 2.0, 0.0),
                accent,
                collision=False,
            ),
            _box(
                "port_wingtip",
                (x * 0.22, y * 0.10, z * 0.20),
                (-x * 0.05, -y * 0.46, z * 0.33),
                hull,
            ),
            _box(
                "starboard_wingtip",
                (x * 0.22, y * 0.10, z * 0.20),
                (-x * 0.05, y * 0.46, z * 0.33),
                hull,
            ),
            _box(
                "vertical_fin",
                (x * 0.12, y * 0.08, z * 0.62),
                (-x * 0.37, 0.0, z * 0.69),
                accent,
            ),
            _box(
                "landing_skid_left",
                (x * 0.35, y * 0.07, z * 0.06),
                (0.0, -y * 0.26, z * 0.03),
                dark,
            ),
            _box(
                "landing_skid_right",
                (x * 0.35, y * 0.07, z * 0.06),
                (0.0, y * 0.26, z * 0.03),
                dark,
            ),
        )
    )


def _machine_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    steel = (0.18, 0.22, 0.25, 1.0)
    hazard = (0.95, 0.55, 0.04, 1.0)
    screen = (0.04, 0.50, 0.62, 1.0)
    return "".join(
        (
            _box("base", (x, y, z * 0.18), (0.0, 0.0, z * 0.09), steel),
            _box(
                "column",
                (x * 0.22, y * 0.34, z * 0.72),
                (-x * 0.28, 0.0, z * 0.54),
                steel,
            ),
            _box(
                "service_arm",
                (x * 0.72, y * 0.18, z * 0.14),
                (x * 0.08, 0.0, z * 0.82),
                hazard,
            ),
            _box(
                "tool_head",
                (x * 0.18, y * 0.45, z * 0.28),
                (x * 0.37, 0.0, z * 0.66),
                steel,
            ),
            _box(
                "control_screen",
                (x * 0.22, y * 0.06, z * 0.20),
                (-x * 0.05, -y * 0.20, z * 0.48),
                screen,
                collision=False,
            ),
        )
    )


def _rack_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    steel = (0.18, 0.21, 0.24, 1.0)
    parts = (0.78, 0.32, 0.08, 1.0)
    pieces = []
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            pieces.append(
                _box(
                    f"post_{x_sign}_{y_sign}",
                    (0.10, 0.10, z),
                    (x_sign * (x / 2.0 - 0.05), y_sign * (y / 2.0 - 0.05), z / 2.0),
                    steel,
                )
            )
    for index, height in enumerate((0.18, 0.45, 0.72, 0.96)):
        pieces.append(
            _box(f"shelf_{index}", (x, y, 0.10), (0.0, 0.0, z * height), steel)
        )
    pieces.append(
        _box("parts_bins", (x * 0.82, y * 0.78, z * 0.18), (0.0, 0.0, z * 0.58), parts)
    )
    return "".join(pieces)


def _equipment_geometry(dimensions: tuple[float, float, float]) -> str:
    x, y, z = dimensions
    steel = (0.20, 0.24, 0.28, 1.0)
    hazard = (0.92, 0.45, 0.04, 1.0)
    return "".join(
        (
            _box("equipment_case", (x, y, z * 0.72), (0.0, 0.0, z * 0.36), steel),
            _box(
                "hazard_band",
                (x * 1.01, y * 1.01, z * 0.10),
                (0.0, 0.0, z * 0.48),
                hazard,
                collision=False,
            ),
            _box(
                "top_module",
                (x * 0.62, y * 0.62, z * 0.28),
                (0.0, 0.0, z * 0.86),
                steel,
            ),
        )
    )


def _write_role_sdf(
    output_dir: Path,
    role: str,
    dimensions: tuple[float, float, float],
    kind: str,
    *,
    instance_index: int = 0,
    instance_prompt: dict[str, Any] | None = None,
) -> Path:
    role_dir = output_dir / _slug(role) / f"instance_{instance_index:03d}"
    role_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = role_dir / "model.sdf"
    role_words = role.casefold()
    variant = _variant_index(instance_index, instance_prompt)
    if kind == "semantic_repeated_zone" or any(
        word in role_words for word in ("bay", "zone", "booth")
    ):
        geometry = _zone_geometry(dimensions, variant=variant)
    elif kind == "semantic_hero_object" or any(
        word in role_words for word in ("fighter", "craft", "vehicle")
    ):
        geometry = _fighter_geometry(dimensions)
    elif "machine" in role_words:
        geometry = _machine_geometry(dimensions)
    elif any(word in role_words for word in ("rack", "shelf", "parts")):
        geometry = _rack_geometry(dimensions)
    else:
        geometry = _equipment_geometry(dimensions)
    model_name = f"{_slug(role)}_{instance_index:03d}"
    sdf_path.write_text(
        "<?xml version='1.0'?>"
        "<sdf version='1.9'>"
        f"<model name='{model_name}'>"
        f"<link name='base_link'>{geometry}</link></model></sdf>",
        encoding="utf-8",
    )
    return sdf_path
