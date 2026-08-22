"""Provider-neutral normalization for one-shot floor-plan submissions.

Model tool arguments are design intent, not trusted internal state.  This module
accepts common OpenAI/Anthropic/open-model shapes and converts them to the small
canonical boundary consumed by :class:`FloorPlanTools`.
"""

from __future__ import annotations

import math
import re

from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from scenesmith.agent_utils.semantics.requirements.scene_blueprint import (
        SceneBlueprint,
    )

_MULTILEVEL_PATTERN = re.compile(
    r"\b(multi[- ]?level|multi[- ]?story|multiple levels?|two[- ]?story|"
    r"three[- ]?story|two[- ]?levels?|three[- ]?levels?|four[- ]?levels?|"
    r"mezzanine|platforms?|upper floor|spiral stairs?|stairs?|staircases?|ramp)\b",
    re.IGNORECASE,
)


def synthesize_structural_layout(
    prompt: str,
    room_specs: list[dict[str, Any]],
    storey_height: float,
    *,
    max_total_height: float = 12.0,
    level_count_hint: int | None = None,
    exact_level_count: int | None = None,
    stair_width_hint: float | None = None,
) -> dict[str, Any] | None:
    """Build conservative walkable levels, slabs, and stairs from prompt intent.

    ``storey_height`` is often authored as a per-storey height while the legacy
    room shell validates one total height. Fit the requested number of levels
    into that envelope before emitting geometry; otherwise the deterministic
    fallback would repeat the provider's invalid height and silently flatten.
    """

    if not _MULTILEVEL_PATTERN.search(prompt) and exact_level_count is None:
        return None
    lowered = prompt.lower()
    level_count = (
        4
        if "four" in lowered
        else (
            3
            if any(token in lowered for token in ("three", "multi", "multiple"))
            or re.search(
                r"\btwo\s+(?:raised\s+)?(?:mezzanine\s+)?platforms?\b",
                lowered,
            )
            else 2
        )
    )
    if exact_level_count is not None:
        level_count = int(exact_level_count)
    elif level_count_hint is not None:
        level_count = max(level_count, int(level_count_hint))
    level_count = max(2, min(4, level_count))
    requested_level_count = level_count
    requested_storey_height = float(storey_height)
    diagnostics: list[str] = []
    minimum_storey_height = 2.8
    maximum_storey_height = 4.5
    max_total_height = max(2 * minimum_storey_height, float(max_total_height))
    max_feasible_levels = max(2, int(max_total_height // minimum_storey_height))
    level_count = min(level_count, max_feasible_levels)
    height = max(
        minimum_storey_height,
        min(
            maximum_storey_height, float(storey_height), max_total_height / level_count
        ),
    )
    if level_count < requested_level_count:
        diagnostics.append(
            f"Reduced the requested {requested_level_count} levels to {level_count} "
            f"because the {max_total_height:g}m shell cannot provide safe headroom."
        )
    elif requested_storey_height * level_count > max_total_height + 1e-9:
        diagnostics.append(
            f"Reduced each storey from {requested_storey_height:g}m to {height:g}m "
            f"to fit {level_count} usable levels inside the {max_total_height:g}m shell."
        )
    room = room_specs[0]
    room_id = str(room["type"])
    width = float(room["width"])
    depth = float(room["depth"])
    replacement_target = re.search(
        r"\b(?:replace|swap|change)\b.*?\b(?:with|to)\b(?P<target>.+)$",
        lowered,
    )
    family_text = replacement_target.group("target") if replacement_target else lowered
    requested_family = (
        "stairs_spiral"
        if "spiral" in family_text
        else (
            "stairs_u"
            if any(
                token in family_text
                for token in (
                    "switchback",
                    "switch-back",
                    "u-shaped",
                    "u shaped",
                    "u-stair",
                    "u stair",
                    "double-back",
                    "double back",
                )
            )
            else (
                "stairs_l"
                if any(
                    token in family_text
                    for token in (
                        "l-shaped",
                        "l shaped",
                        "l-stair",
                        "l stair",
                        "quarter-turn",
                        "quarter turn",
                    )
                )
                else "stairs_straight"
            )
        )
    )
    stair_width = max(0.8, float(stair_width_hint or 1.1))
    riser_count = max(12, round(height / 0.18))
    tread_depth = 0.28
    clearance_padding = 0.2
    # A landing must overlap the surrounding slab by enough to read and behave
    # as a real destination. The stairwell includes tread-width clearance past
    # each endpoint, so a short 0.9 m landing left only a ~0.15 m connection.
    landing_depth = 1.5

    # Author stair paths in a local long/short-axis frame, then map them back to
    # room XY.  This keeps ordinary stairs viable in portrait and landscape rooms.
    swap_axes = depth > width
    long_extent, short_extent = (depth, width) if swap_axes else (width, depth)

    def room_xy(along: float, across: float) -> list[float]:
        return [across, along] if swap_axes else [along, across]

    def point(along: float, across: float, elevation: float) -> list[float]:
        return [*room_xy(along, across), elevation]

    # Connector centerlines are authored in the house frame, while platforms
    # are welded to the room's center frame. Platform footprints therefore use
    # the same centered-local convention as compiled room geometry.
    def platform_xy(point_xy: Sequence[float]) -> list[float]:
        return [point_xy[0] - width / 2.0, point_xy[1] - depth / 2.0]

    margin = landing_depth + stair_width / 2.0 + clearance_padding
    family = requested_family
    half_risers = riser_count // 2
    remaining_risers = riser_count - half_risers
    first_run = half_risers * tread_depth
    second_run = remaining_risers * tread_depth
    straight_run = riser_count * tread_depth
    switchback_gap = stair_width + 0.35

    if family == "stairs_straight" and long_extent - 2 * margin < riser_count * 0.22:
        family = "stairs_spiral"
    elif family == "stairs_l" and (
        long_extent - 2 * margin < first_run or short_extent - 2 * margin < second_run
    ):
        family = "stairs_u"
    if family == "stairs_u" and (
        long_extent - 2 * margin < max(first_run, second_run)
        or short_extent - 2 * margin < switchback_gap
    ):
        family = "stairs_spiral"
    if family != requested_family:
        diagnostics.append(
            f"Changed {requested_family} to {family} because the room footprint "
            "cannot fit safe tread and landing geometry for the requested form."
        )

    spiral_radius = max(
        stair_width / 2.0 + 0.25,
        riser_count * 0.22 / (2.0 * math.pi),
    )
    spiral_outer_radius = spiral_radius + stair_width / 2.0
    spiral_center = [
        max(
            spiral_outer_radius + clearance_padding,
            min(
                width - spiral_outer_radius - clearance_padding,
                width * 0.25,
            ),
        ),
        max(
            spiral_outer_radius + clearance_padding,
            min(
                depth - spiral_outer_radius - clearance_padding,
                depth * 0.25,
            ),
        ),
    ]
    levels = [
        {"id": f"level_{index}", "elevation": index * height, "nominal_height": height}
        for index in range(level_count)
    ]
    connectors = []
    clearance_points: list[list[float]] = []
    landing_platforms: list[dict[str, Any]] = []
    for index in range(level_count - 1):
        bottom = index * height
        top = (index + 1) * height
        mid = bottom + height * half_risers / riser_count
        start = point(margin, margin, bottom)
        end = point(margin + straight_run, margin, top)
        parameters: dict[str, Any] = {"riser_count": riser_count}

        if family == "stairs_spiral":
            start = [
                spiral_center[0] + spiral_radius,
                spiral_center[1],
                bottom,
            ]
            end = [
                spiral_center[0] + spiral_radius,
                spiral_center[1],
                top,
            ]
            parameters.update(
                {
                    "center": spiral_center,
                    "radius": spiral_radius,
                    "turns": 1.0,
                    "direction": "cw",
                }
            )
            clearance_points.extend(
                [
                    [
                        spiral_center[0] - spiral_outer_radius,
                        spiral_center[1] - spiral_outer_radius,
                    ],
                    [
                        spiral_center[0] + spiral_outer_radius,
                        spiral_center[1] + spiral_outer_radius,
                    ],
                ]
            )
        elif family == "stairs_l":
            turn = point(margin + first_run, margin, mid)
            end = point(margin + first_run, margin + second_run, top)
            parameters = {
                "waypoints": [turn],
                "riser_counts": [half_risers, remaining_risers],
                "landing_length": stair_width,
            }
            clearance_points.extend([start[:2], turn[:2], end[:2]])
        elif family == "stairs_u":
            first_turn = point(margin + first_run, margin, mid)
            second_turn = point(
                margin + first_run,
                margin + switchback_gap,
                mid,
            )
            end = point(margin, margin + switchback_gap, top)
            parameters = {
                "waypoints": [first_turn, second_turn],
                "riser_counts": [half_risers, remaining_risers],
            }
            clearance_points.extend(
                [start[:2], first_turn[:2], second_turn[:2], end[:2]]
            )
        else:
            clearance_points.extend([start[:2], end[:2]])

        if index % 2 == 1 and family != "stairs_spiral":
            previous_start = start
            start = [*end[:2], bottom]
            end = [*previous_start[:2], top]
            if family == "stairs_l":
                parameters["waypoints"] = [[*parameters["waypoints"][0][:2], mid]]
                parameters["riser_counts"] = list(reversed(parameters["riser_counts"]))
            elif family == "stairs_u":
                parameters["waypoints"] = [
                    [*waypoint[:2], mid]
                    for waypoint in reversed(parameters["waypoints"])
                ]
                parameters["riser_counts"] = list(reversed(parameters["riser_counts"]))

        last_waypoint = parameters.get("waypoints", ())[-1:] or [start]
        approach_from = last_waypoint[0]
        if family == "stairs_spiral":
            direction_x = end[0] - spiral_center[0]
            direction_y = end[1] - spiral_center[1]
        else:
            direction_x = end[0] - approach_from[0]
            direction_y = end[1] - approach_from[1]
        direction_length = max(math.hypot(direction_x, direction_y), 1e-9)
        direction_x /= direction_length
        direction_y /= direction_length
        perpendicular_x, perpendicular_y = -direction_y, direction_x
        landing_half_width = stair_width / 2.0 + 0.1
        landing_start = [
            end[0] - direction_x * 0.08,
            end[1] - direction_y * 0.08,
        ]
        landing_end = [
            end[0] + direction_x * landing_depth,
            end[1] + direction_y * landing_depth,
        ]
        landing_outer = [
            [
                landing_start[0] - perpendicular_x * landing_half_width,
                landing_start[1] - perpendicular_y * landing_half_width,
            ],
            [
                landing_end[0] - perpendicular_x * landing_half_width,
                landing_end[1] - perpendicular_y * landing_half_width,
            ],
            [
                landing_end[0] + perpendicular_x * landing_half_width,
                landing_end[1] + perpendicular_y * landing_half_width,
            ],
            [
                landing_start[0] + perpendicular_x * landing_half_width,
                landing_start[1] + perpendicular_y * landing_half_width,
            ],
        ]
        landing_platforms.append(
            {
                "id": f"level_{index + 1}_stair_landing",
                "space_id": room_id,
                "footprint": {
                    "outer": [platform_xy(point_xy) for point_xy in landing_outer],
                    "holes": [],
                },
                "elevation": top,
                "thickness": 0.2,
                "traversable": True,
            }
        )

        connector: dict[str, Any] = {
            "id": f"fallback_stairs_{index + 1}",
            "type": family,
            "start": {
                "space_id": room_id,
                "level_id": f"level_{index}",
                "position": start,
            },
            "end": {
                "space_id": room_id,
                "level_id": f"level_{index + 1}",
                "position": end,
            },
            "width": stair_width,
            "parameters": parameters,
        }
        connectors.append(connector)

    # Each upper datum needs actual walkable geometry. Use a full floor slab with
    # a stairwell void rather than treating the level list as metadata or adding a
    # decorative furniture staircase. Clockwise hole winding is intentional.
    inset = min(0.15, width * 0.01, depth * 0.01)
    outer_house = [
        [inset, inset],
        [width - inset, inset],
        [width - inset, depth - inset],
        [inset, depth - inset],
    ]
    outer = [platform_xy(point_xy) for point_xy in outer_house]
    # Spiral clearance points already describe the outer tread edge. Straight,
    # L, and U paths describe their centerlines and still need half-width added.
    footprint_half_width = 0.0 if family == "stairs_spiral" else stair_width / 2.0
    min_x = max(
        inset + 0.02,
        min(point[0] for point in clearance_points)
        - footprint_half_width
        - clearance_padding,
    )
    max_x = min(
        width - inset - 0.02,
        max(point[0] for point in clearance_points)
        + footprint_half_width
        + clearance_padding,
    )
    min_y = max(
        inset + 0.02,
        min(point[1] for point in clearance_points)
        - footprint_half_width
        - clearance_padding,
    )
    max_y = min(
        depth - inset - 0.02,
        max(point[1] for point in clearance_points)
        + footprint_half_width
        + clearance_padding,
    )
    hole_house = [
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
    ]
    hole = [platform_xy(point_xy) for point_xy in hole_house]
    platform_holes = [hole]
    guarded_hole_indices: list[int] = []
    folded_prompt = prompt.casefold()
    gallery_library = (
        "library" in folded_prompt
        and any(
            token in folded_prompt
            for token in (
                "multi-level",
                "multilevel",
                "mezzanine",
                "two-story",
                "two storey",
                "three-story",
                "three storey",
            )
        )
        and any(
            token in folded_prompt
            for token in ("large", "grand", "vast", "thousands of books")
        )
    )
    if gallery_library:
        minimum_extent = min(width, depth)
        gallery_gap = max(0.4, minimum_extent * 0.04)
        perimeter_width = max(1.5, minimum_extent * 0.14)
        gallery_min_x = max(width * 0.36, max_x + gallery_gap)
        gallery_min_y = max(depth * 0.36, max_y + gallery_gap)
        gallery_max_x = width - inset - perimeter_width
        gallery_max_y = depth - inset - perimeter_width
        if (
            gallery_max_x - gallery_min_x >= 5.0
            and gallery_max_y - gallery_min_y >= 5.0
        ):
            gallery_hole_house = [
                [gallery_min_x, gallery_min_y],
                [gallery_min_x, gallery_max_y],
                [gallery_max_x, gallery_max_y],
                [gallery_max_x, gallery_min_y],
            ]
            platform_holes.append(
                [platform_xy(point_xy) for point_xy in gallery_hole_house]
            )
            guarded_hole_indices.append(len(platform_holes) - 1)
            diagnostics.append(
                "Added a large gallery atrium void to each upper library level."
            )
    platforms = [
        {
            "id": f"level_{index}_walkable_slab",
            "space_id": room_id,
            "footprint": {"outer": outer, "holes": platform_holes},
            "elevation": index * height,
            "thickness": 0.2,
            "guarded_hole_indices": guarded_hole_indices,
            "traversable": True,
        }
        for index in range(1, level_count)
    ] + landing_platforms
    return {
        "levels": levels,
        "rooms": [{"id": room_id, "level_id": "level_0"}],
        "connectors": connectors,
        "platforms": platforms,
        "_diagnostics": diagnostics,
    }


def structural_submission_from_blueprint(
    blueprint: "SceneBlueprint",
    room_specs: list[dict[str, Any]],
    *,
    max_total_height: float,
) -> dict[str, Any] | None:
    """Project an accepted semantic blueprint into deterministic construction."""

    if len(blueprint.levels) <= 1 and not blueprint.connectors:
        return None
    if not room_specs:
        raise ValueError("blueprint structural projection requires a primary room")
    ordered_levels = sorted(blueprint.levels, key=lambda level: level.elevation_m)
    primary_room_id = str(room_specs[0]["type"])
    widest_connector = max(
        (connector.width_m for connector in blueprint.connectors),
        default=1.1,
    )
    structural = synthesize_structural_layout(
        blueprint.source_prompt,
        room_specs,
        max(level.clear_height_m for level in ordered_levels),
        max_total_height=max_total_height,
        exact_level_count=len(ordered_levels),
        stair_width_hint=widest_connector,
    )
    if structural is None:
        return None

    authored_levels = list(structural.get("levels", []))
    if len(authored_levels) != len(ordered_levels):
        raise ValueError(
            "blueprint levels cannot fit the configured structural height envelope: "
            f"requested {len(ordered_levels)}, constructible {len(authored_levels)}"
        )
    elevation_by_old_id = {
        str(level.get("id")): float(level.get("elevation", 0.0))
        for level in authored_levels
        if isinstance(level, Mapping)
    }
    level_by_index = {index: level for index, level in enumerate(ordered_levels)}
    old_level_ids = [str(level.get("id")) for level in authored_levels]
    old_to_new = {
        old_id: level_by_index[index].level_id
        for index, old_id in enumerate(old_level_ids)
        if index in level_by_index
    }
    structural["levels"] = [
        {
            "id": level.level_id,
            "elevation": level.elevation_m,
            "nominal_height": level.clear_height_m,
        }
        for level in ordered_levels
    ]
    structural["rooms"] = [
        {"id": primary_room_id, "level_id": ordered_levels[0].level_id}
    ]

    def new_elevation(old_elevation: float) -> float:
        if not elevation_by_old_id:
            return old_elevation
        closest_old_id = min(
            elevation_by_old_id,
            key=lambda level_id: abs(elevation_by_old_id[level_id] - old_elevation),
        )
        new_level_id = old_to_new.get(closest_old_id)
        return next(
            (
                level.elevation_m
                for level in ordered_levels
                if level.level_id == new_level_id
            ),
            old_elevation,
        )

    for platform in structural.get("platforms", []):
        if isinstance(platform, dict):
            platform["space_id"] = primary_room_id
            platform["elevation"] = new_elevation(float(platform.get("elevation", 0.0)))

    templates = [
        item for item in structural.get("connectors", []) if isinstance(item, dict)
    ]
    level_by_id = {level.level_id: level for level in ordered_levels}
    level_index = {level.level_id: index for index, level in enumerate(ordered_levels)}
    connectors: list[dict[str, Any]] = []
    for connector in blueprint.connectors:
        landing_level_ids = [
            str(landing.get("level_id", ""))
            for landing in connector.parameters.get("intermediate_landings", [])
            if isinstance(landing, Mapping)
        ]
        served_level_ids = list(
            dict.fromkeys(
                [
                    connector.start.level_id,
                    *landing_level_ids,
                    connector.end.level_id,
                ]
            )
        )
        served_level_ids.sort(key=level_index.__getitem__)
        level_pairs = list(zip(served_level_ids, served_level_ids[1:]))
        for segment_index, (start_level_id, end_level_id) in enumerate(
            level_pairs, start=1
        ):
            template_index = min(level_index[start_level_id], len(templates) - 1)
            template = dict(templates[template_index]) if templates else {}
            parameters = dict(template.get("parameters", {}))
            start_level = level_by_id[start_level_id]
            end_level = level_by_id[end_level_id]
            if connector.kind == "stairs_spiral":
                center = parameters.get(
                    "center",
                    [
                        float(room_specs[0]["width"]) * 0.25,
                        float(room_specs[0]["depth"]) * 0.25,
                    ],
                )
                riser_count = max(
                    12,
                    round(abs(end_level.elevation_m - start_level.elevation_m) / 0.18),
                )
                radius = max(
                    float(parameters.get("radius", 0.0)),
                    connector.width_m / 2.0 + 0.25,
                    riser_count * 0.221 / (2.0 * math.pi),
                )
                parameters.update(
                    {
                        "center": center,
                        "radius": radius,
                        # One full turn keeps each compiled span aligned with its
                        # declared endpoint and intermediate landing.
                        "turns": 1.0,
                        "direction": parameters.get("direction", "cw"),
                        "riser_count": riser_count,
                    }
                )
                start_xy = [float(center[0]) + radius, float(center[1])]
                end_xy = list(start_xy)
            else:
                template_start = template.get("start", {}).get("position", [2.0, 2.0])
                template_end = template.get("end", {}).get("position", [4.0, 2.0])
                start_xy = list(template_start[:2])
                end_xy = list(template_end[:2])
            connectors.append(
                {
                    "id": (
                        connector.connector_id
                        if len(level_pairs) == 1 or segment_index == 1
                        else f"{connector.connector_id}-segment-{segment_index}"
                    ),
                    "type": connector.kind,
                    "start": {
                        "space_id": primary_room_id,
                        "level_id": start_level.level_id,
                        "position": [*start_xy, start_level.elevation_m],
                    },
                    "end": {
                        "space_id": primary_room_id,
                        "level_id": end_level.level_id,
                        "position": [*end_xy, end_level.elevation_m],
                    },
                    "width": connector.width_m,
                    "clearance_height": min(start_level.clear_height_m, 2.4),
                    "parameters": parameters,
                }
            )
    structural["connectors"] = connectors
    return structural
