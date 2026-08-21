"""Deterministic presentation geometry for ornate Renaissance libraries."""

from __future__ import annotations

import math
import re

from collections import Counter
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
WALNUT = np.array([73, 39, 24, 255], dtype=np.uint8)
BOOK_SPINE_COLORS = (
    np.array([112, 24, 39, 255], dtype=np.uint8),
    np.array([34, 68, 55, 255], dtype=np.uint8),
    np.array([31, 53, 91, 255], dtype=np.uint8),
    np.array([151, 102, 30, 255], dtype=np.uint8),
    np.array([122, 71, 39, 255], dtype=np.uint8),
    np.array([202, 177, 119, 255], dtype=np.uint8),
)


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


def _append_gallery_panels(
    *,
    layout: HouseLayout,
    room_id: str,
    gold: list[trimesh.Trimesh],
    burgundy: list[trimesh.Trimesh],
) -> int:
    panel_count = 0
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
                    interval_start = start_xy + (end_xy - start_xy) * (
                        index / intervals
                    )
                    interval_end = start_xy + (end_xy - start_xy) * (
                        (index + 1) / intervals
                    )
                    edge_direction = (interval_end - interval_start) / np.linalg.norm(
                        interval_end - interval_start
                    )
                    panel_start = interval_start + edge_direction * 0.07
                    panel_end = interval_end - edge_direction * 0.07
                    lower_z = platform.elevation + 0.34
                    upper_z = platform.elevation + 0.84

                    for z in (lower_z, upper_z):
                        gold.append(
                            _beam(
                                np.array([panel_start[0], panel_start[1], z]),
                                np.array([panel_end[0], panel_end[1], z]),
                                0.035,
                            )
                        )
                    for point in (panel_start, panel_end):
                        gold.append(
                            _beam(
                                np.array([point[0], point[1], lower_z]),
                                np.array([point[0], point[1], upper_z]),
                                0.035,
                            )
                        )

                    midpoint = (interval_start + interval_end) / 2.0
                    burgundy.append(
                        _beam(
                            np.array([midpoint[0], midpoint[1], lower_z + 0.03]),
                            np.array([midpoint[0], midpoint[1], upper_z - 0.03]),
                            0.05,
                        )
                    )
                    for z in (platform.elevation + 0.28, platform.elevation + 0.91):
                        collar = trimesh.creation.box(extents=(0.11, 0.11, 0.05))
                        collar.apply_translation(
                            (interval_start[0], interval_start[1], z)
                        )
                        gold.append(collar)
                    panel_count += 1
    return panel_count


def _colored_mesh(
    meshes: list[trimesh.Trimesh], color: np.ndarray, name: str
) -> trimesh.Trimesh:
    combined = trimesh.util.concatenate(meshes)
    combined.visual.face_colors = np.tile(color, (len(combined.faces), 1))
    combined.metadata["name"] = name
    combined.apply_transform(get_zup_to_yup_matrix())
    return combined


def _pbr_colored_mesh(
    meshes: list[trimesh.Trimesh],
    color: np.ndarray,
    name: str,
    *,
    metallic: float = 0.0,
) -> trimesh.Trimesh:
    """Combine deterministic primitives under one explicit glTF PBR material."""

    combined = trimesh.util.concatenate(meshes)
    combined.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=color,
            metallicFactor=metallic,
            roughnessFactor=0.58 if metallic else 0.72,
        )
    )
    combined.metadata["name"] = name
    combined.apply_transform(get_zup_to_yup_matrix())
    return combined


def _append_populated_bookcase(
    *,
    owner: object,
    along: np.ndarray,
    inward: np.ndarray,
    walnut: list[trimesh.Trimesh],
    gold: list[trimesh.Trimesh],
    spines: list[list[trimesh.Trimesh]],
) -> int:
    """Add a full-height walnut case and deterministic visible book spines."""

    minimum = np.asarray(owner.bbox_min, dtype=float)
    maximum = np.asarray(owner.bbox_max, dtype=float)
    dimensions = maximum - minimum
    width = float(max(dimensions[0], dimensions[1]))
    depth = float(max(min(dimensions[0], dimensions[1]), 0.24))
    height = float(dimensions[2])
    if width < 0.65 or height < 1.6:
        return 0

    translation = np.asarray(owner.transform.translation(), dtype=float)
    center_xy = translation[:2]
    bottom = float(translation[2] + minimum[2])
    center_z = bottom + height / 2.0
    front_xy = center_xy + inward * (depth * 0.36)
    back_xy = center_xy - inward * (depth * 0.38)

    walnut.append(
        _wall_box(
            center=np.array([back_xy[0], back_xy[1], center_z]),
            along=along,
            inward=inward,
            along_extent=width * 0.94,
            depth=0.035,
            height=height * 0.94,
        )
    )
    for side in (-1.0, 1.0):
        side_xy = center_xy + along * side * (width / 2.0 - 0.035)
        walnut.append(
            _wall_box(
                center=np.array([side_xy[0], side_xy[1], center_z]),
                along=along,
                inward=inward,
                along_extent=0.07,
                depth=depth * 0.90,
                height=height,
            )
        )
        trim_xy = front_xy + along * side * (width / 2.0 - 0.065)
        gold.append(
            _wall_box(
                center=np.array([trim_xy[0], trim_xy[1], center_z]),
                along=along,
                inward=inward,
                along_extent=0.025,
                depth=0.045,
                height=height * 0.88,
            )
        )

    for z in (bottom + 0.045, bottom + height - 0.045):
        walnut.append(
            _wall_box(
                center=np.array([center_xy[0], center_xy[1], z]),
                along=along,
                inward=inward,
                along_extent=width,
                depth=depth,
                height=0.09,
            )
        )
    gold.append(
        _wall_box(
            center=np.array([front_xy[0], front_xy[1], bottom + height - 0.105]),
            along=along,
            inward=inward,
            along_extent=width * 0.96,
            depth=0.055,
            height=0.055,
        )
    )

    tier_fractions = (0.06, 0.205, 0.35, 0.495, 0.64, 0.785)
    usable_width = width * 0.76
    books_per_tier = 10
    book_step = usable_width / books_per_tier
    book_depth = max(0.06, depth * 0.24)
    spine_count = 0
    owner_seed = sum(ord(character) for character in str(owner.object_id))
    for tier_index, tier_fraction in enumerate(tier_fractions):
        shelf_z = bottom + height * tier_fraction
        walnut.append(
            _wall_box(
                center=np.array([center_xy[0], center_xy[1], shelf_z]),
                along=along,
                inward=inward,
                along_extent=width * 0.88,
                depth=depth * 0.84,
                height=0.035,
            )
        )
        for book_index in range(books_per_tier):
            pattern = owner_seed + tier_index * 11 + book_index * 7
            book_width = book_step * (0.72 + 0.06 * (pattern % 4))
            book_height = height * (0.095 + 0.01 * (pattern % 4))
            along_offset = -usable_width / 2.0 + (book_index + 0.5) * book_step
            book_xy = front_xy + along * along_offset
            spine = _wall_box(
                center=np.array(
                    [
                        book_xy[0],
                        book_xy[1],
                        shelf_z + 0.025 + book_height / 2.0,
                    ]
                ),
                along=along,
                inward=inward,
                along_extent=book_width,
                depth=book_depth,
                height=book_height,
            )
            spines[pattern % len(spines)].append(spine)
            spine_count += 1
    return spine_count


def write_renaissance_bookcase_visuals(
    layout: HouseLayout,
    rooms: dict[str, object],
    output_dir: Path,
) -> list[dict[str, object]]:
    """Write populated presentation shells for physically validated wall runs."""

    if not _requests_renaissance_dressing(layout.house_prompt):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals: list[dict[str, object]] = []
    for placed_room in layout.placed_rooms:
        room = rooms.get(placed_room.room_id)
        if room is None:
            continue
        row_counts = Counter(
            str((obj.metadata or {}).get("dense_library_owner_bound"))
            for obj in room.objects.values()
            if bool((obj.metadata or {}).get("dense_library_book_row"))
        )
        grouped: dict[float, list[object]] = {}
        for obj in room.objects.values():
            metadata = obj.metadata or {}
            marker = metadata.get("dense_library_grouped_run")
            if (
                marker is None
                or row_counts[str(obj.object_id)] < 3
                or obj.bbox_min is None
                or obj.bbox_max is None
            ):
                continue
            grouped.setdefault(float(marker), []).append(obj)

        walnut: list[trimesh.Trimesh] = []
        gold: list[trimesh.Trimesh] = []
        spines: list[list[trimesh.Trimesh]] = [[] for _color in BOOK_SPINE_COLORS]
        populated = 0
        spine_count = 0
        for level in sorted(grouped):
            owners = grouped[level]
            if len(owners) < 3:
                continue
            points = np.asarray(
                [owner.transform.translation()[:2] for owner in owners], dtype=float
            )
            farthest = max(
                (
                    (left, right)
                    for left in range(len(points))
                    for right in range(left + 1, len(points))
                ),
                key=lambda pair: float(
                    np.linalg.norm(points[pair[1]] - points[pair[0]])
                ),
            )
            along = points[farthest[1]] - points[farthest[0]]
            along /= np.linalg.norm(along)
            inward = np.array([-along[1], along[0]])
            centroid = np.mean(points, axis=0)
            if float(np.dot(inward, -centroid)) < 0.0:
                inward *= -1.0
            for owner in owners:
                added = _append_populated_bookcase(
                    owner=owner,
                    along=along,
                    inward=inward,
                    walnut=walnut,
                    gold=gold,
                    spines=spines,
                )
                if added:
                    populated += 1
                    spine_count += added
        if populated < 3 or not walnut or not gold:
            continue

        scene = trimesh.Scene()
        scene.add_geometry(
            _pbr_colored_mesh(walnut, WALNUT, "renaissance_bookcase_walnut"),
            geom_name="renaissance_bookcase_walnut",
            node_name="renaissance_bookcase_walnut",
        )
        scene.add_geometry(
            _pbr_colored_mesh(
                gold,
                ANTIQUE_GOLD,
                "renaissance_bookcase_antique_gold",
                metallic=0.55,
            ),
            geom_name="renaissance_bookcase_antique_gold",
            node_name="renaissance_bookcase_antique_gold",
        )
        for index, (meshes, color) in enumerate(
            zip(spines, BOOK_SPINE_COLORS, strict=True)
        ):
            if not meshes:
                continue
            name = f"renaissance_book_spines_{index}"
            scene.add_geometry(
                _pbr_colored_mesh(meshes, color, name),
                geom_name=name,
                node_name=name,
            )

        artifact = output_dir / f"renaissance_bookcases_{placed_room.room_id}.glb"
        artifact.write_bytes(trimesh.exchange.gltf.export_glb(scene))
        room_center = [
            placed_room.position[0] + placed_room.width / 2.0,
            placed_room.position[1] + placed_room.depth / 2.0,
        ]
        visuals.append(
            {
                "path": str(artifact),
                "translation": [
                    float(room_center[0]),
                    float(room_center[1]),
                    float(layout.get_room_elevation(placed_room.room_id)),
                ],
                "yaw_radians": float(placed_room.yaw),
                "role": "structural_detail",
                "source_id": f"renaissance_bookcases_{placed_room.room_id}",
                "populated_bookcases": populated,
                "shelf_tiers_per_bookcase": 6,
                "visible_book_spines": spine_count,
            }
        )
    return visuals


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
        panel_count = _append_gallery_panels(
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
                "gallery_panels": panel_count,
            }
        )
    return visuals
