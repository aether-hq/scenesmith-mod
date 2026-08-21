"""Deterministic presentation geometry for ornate Renaissance libraries."""

from __future__ import annotations

import math
import re

from pathlib import Path

import numpy as np
import trimesh

from scenesmith.agent_utils.house import (
    HouseLayout,
    OpeningType,
    Wall,
    WindowShape,
)
from scenesmith.utils.gltf_generation import get_zup_to_yup_matrix


ANTIQUE_GOLD = np.array([184, 134, 48, 255], dtype=np.uint8)
BURGUNDY = np.array([92, 18, 38, 255], dtype=np.uint8)


def _requests_renaissance_dressing(prompt: str) -> bool:
    folded = prompt.casefold()
    renaissance = bool(re.search(r"\brenai+ssance\b", folded))
    explicit_style = renaissance or "ornate" in folded or "antique gold" in folded
    return explicit_style and "librar" in folded


def _beam(start: np.ndarray, end: np.ndarray, thickness: float) -> trimesh.Trimesh:
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-6:
        raise ValueError("architectural beam endpoints must be distinct")
    mesh = trimesh.creation.box(extents=(length, thickness, thickness))
    transform = trimesh.geometry.align_vectors([1.0, 0.0, 0.0], delta / length)
    transform[:3, 3] = (start + end) / 2.0
    mesh.apply_transform(transform)
    return mesh


def _wall_box(
    *,
    center: np.ndarray,
    along: np.ndarray,
    inward: np.ndarray,
    along_extent: float,
    depth: float,
    height: float,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(along_extent, depth, height))
    transform = np.eye(4)
    transform[:3, 0] = (along[0], along[1], 0.0)
    transform[:3, 1] = (inward[0], inward[1], 0.0)
    transform[:3, 2] = (0.0, 0.0, 1.0)
    transform[:3, 3] = center
    mesh.apply_transform(transform)
    return mesh


def _wall_local_geometry(
    wall: Wall, room_center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(wall.start_point, dtype=float) - room_center
    end = np.asarray(wall.end_point, dtype=float) - room_center
    along = end - start
    along /= np.linalg.norm(along)
    inward = np.asarray(wall.direction.get_inward_normal(), dtype=float)
    return start, end, inward


def _append_window_dressing(
    *,
    wall: Wall,
    room_center: np.ndarray,
    gold: list[trimesh.Trimesh],
    burgundy: list[trimesh.Trimesh],
) -> int:
    start, _end, inward = _wall_local_geometry(wall, room_center)
    along = np.asarray(
        (
            wall.end_point[0] - wall.start_point[0],
            wall.end_point[1] - wall.start_point[1],
        ),
        dtype=float,
    )
    along /= np.linalg.norm(along)
    count = 0
    for opening in wall.openings:
        if (
            opening.opening_type != OpeningType.WINDOW
            or opening.shape != WindowShape.ARCHED
        ):
            continue
        count += 1
        center_xy = (
            start
            + along * (opening.position_along_wall + opening.width / 2.0)
            + inward * 0.08
        )
        radius = opening.width / 2.0
        crown_z = opening.sill_height + opening.height - radius
        arch_center = np.array([center_xy[0], center_xy[1], crown_z])

        for material, radius_offset, thickness in (
            (burgundy, 0.30, 0.22),
            (gold, 0.12, 0.13),
        ):
            arch_radius = radius + radius_offset
            points = []
            for index in range(17):
                theta = math.pi * index / 16.0
                point_xy = center_xy + along * (math.cos(theta) * arch_radius)
                points.append(
                    np.array(
                        [
                            point_xy[0],
                            point_xy[1],
                            crown_z + math.sin(theta) * arch_radius,
                        ]
                    )
                )
            material.extend(
                _beam(first, second, thickness)
                for first, second in zip(points, points[1:])
            )
            for side in (-1.0, 1.0):
                side_xy = center_xy + along * (side * arch_radius)
                material.append(
                    _beam(
                        np.array([side_xy[0], side_xy[1], opening.sill_height]),
                        np.array([side_xy[0], side_xy[1], crown_z]),
                        thickness,
                    )
                )

        for side in (-1.0, 1.0):
            pilaster_xy = center_xy + along * (side * (radius + 0.48))
            burgundy.append(
                _wall_box(
                    center=np.array([pilaster_xy[0], pilaster_xy[1], 2.05]),
                    along=along,
                    inward=inward,
                    along_extent=0.30,
                    depth=0.20,
                    height=4.10,
                )
            )
            for z, width in ((0.16, 0.48), (4.02, 0.52)):
                gold.append(
                    _wall_box(
                        center=np.array([pilaster_xy[0], pilaster_xy[1], z]),
                        along=along,
                        inward=inward,
                        along_extent=width,
                        depth=0.25,
                        height=0.18,
                    )
                )

        # Gold mullions and a transom make the arched silhouette legible from
        # the default interior camera even when the glass itself is subtle.
        gold.append(
            _beam(
                np.array([center_xy[0], center_xy[1], opening.sill_height + 0.08]),
                np.array(
                    [
                        center_xy[0],
                        center_xy[1],
                        opening.sill_height + opening.height - 0.12,
                    ]
                ),
                0.07,
            )
        )
        transom_start = center_xy - along * (radius - 0.12)
        transom_end = center_xy + along * (radius - 0.12)
        gold.append(
            _beam(
                np.array([transom_start[0], transom_start[1], crown_z]),
                np.array([transom_end[0], transom_end[1], crown_z]),
                0.07,
            )
        )
    return count


def _append_cornices(
    *,
    walls: list[Wall],
    room_center: np.ndarray,
    elevations: set[float],
    gold: list[trimesh.Trimesh],
    burgundy: list[trimesh.Trimesh],
) -> None:
    for wall in walls:
        start, end, inward = _wall_local_geometry(wall, room_center)
        for elevation in elevations:
            for material, z, thickness, offset in (
                (burgundy, elevation, 0.18, 0.08),
                (gold, elevation - 0.15, 0.07, 0.13),
            ):
                first_xy = start + inward * offset
                second_xy = end + inward * offset
                material.append(
                    _beam(
                        np.array([first_xy[0], first_xy[1], z]),
                        np.array([second_xy[0], second_xy[1], z]),
                        thickness,
                    )
                )


def _append_gallery_finials(
    *,
    layout: HouseLayout,
    room_id: str,
    gold: list[trimesh.Trimesh],
    burgundy: list[trimesh.Trimesh],
) -> int:
    finial_count = 0
    for platform in layout.platforms:
        if platform.space_id != room_id:
            continue
        for hole_index in platform.guarded_hole_indices:
            loop = platform.footprint.holes[hole_index]
            for start, end in zip(loop, loop[1:] + loop[:1]):
                start_xy = np.asarray(start, dtype=float)
                end_xy = np.asarray(end, dtype=float)
                edge_length = float(np.linalg.norm(end_xy - start_xy))
                intervals = max(1, math.ceil(edge_length / 0.75))
                for index in range(intervals):
                    fraction = index / intervals
                    point = start_xy + (end_xy - start_xy) * fraction
                    finial = trimesh.creation.icosphere(subdivisions=1, radius=0.11)
                    finial.apply_translation(
                        (point[0], point[1], platform.elevation + 1.08)
                    )
                    gold.append(finial)
                    medallion = trimesh.creation.icosphere(subdivisions=1, radius=0.075)
                    midpoint = point + (end_xy - start_xy) / (2.0 * intervals)
                    medallion.apply_translation(
                        (midpoint[0], midpoint[1], platform.elevation + 0.62)
                    )
                    burgundy.append(medallion)
                    finial_count += 1
    return finial_count


def _colored_mesh(
    meshes: list[trimesh.Trimesh], color: np.ndarray, name: str
) -> trimesh.Trimesh:
    combined = trimesh.util.concatenate(meshes)
    combined.visual.face_colors = np.tile(color, (len(combined.faces), 1))
    combined.metadata["name"] = name
    combined.apply_transform(get_zup_to_yup_matrix())
    return combined


def write_renaissance_dressing_visuals(
    layout: HouseLayout, output_dir: Path
) -> list[dict[str, object]]:
    """Write bounded, collision-free presentation details for ornate libraries."""

    if not _requests_renaissance_dressing(layout.house_prompt):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals: list[dict[str, object]] = []
    for room in layout.placed_rooms:
        gold: list[trimesh.Trimesh] = []
        burgundy: list[trimesh.Trimesh] = []
        room_center = np.array(
            [room.position[0] + room.width / 2.0, room.position[1] + room.depth / 2.0]
        )
        arch_count = sum(
            _append_window_dressing(
                wall=wall,
                room_center=room_center,
                gold=gold,
                burgundy=burgundy,
            )
            for wall in room.walls
        )
        level_cornices = {
            platform.elevation - 0.10
            for platform in layout.platforms
            if platform.space_id == room.room_id and platform.elevation > 0.25
        }
        level_cornices.add(layout.wall_height - 0.28)
        _append_cornices(
            walls=room.walls,
            room_center=room_center,
            elevations=level_cornices,
            gold=gold,
            burgundy=burgundy,
        )
        finial_count = _append_gallery_finials(
            layout=layout,
            room_id=room.room_id,
            gold=gold,
            burgundy=burgundy,
        )
        if not gold or not burgundy:
            continue

        scene = trimesh.Scene()
        scene.add_geometry(
            _colored_mesh(gold, ANTIQUE_GOLD, "renaissance_antique_gold"),
            geom_name="renaissance_antique_gold",
            node_name="renaissance_antique_gold",
        )
        scene.add_geometry(
            _colored_mesh(burgundy, BURGUNDY, "renaissance_burgundy"),
            geom_name="renaissance_burgundy",
            node_name="renaissance_burgundy",
        )
        artifact = output_dir / f"renaissance_dressing_{room.room_id}.glb"
        artifact.write_bytes(trimesh.exchange.gltf.export_glb(scene))
        visuals.append(
            {
                "path": str(artifact),
                "translation": [
                    float(room_center[0]),
                    float(room_center[1]),
                    float(layout.get_room_elevation(room.room_id)),
                ],
                "yaw_radians": float(room.yaw),
                "role": "room_structure",
                "source_id": f"renaissance_dressing_{room.room_id}",
                "arched_window_surrounds": arch_count,
                "gallery_finials": finial_count,
            }
        )
    return visuals
