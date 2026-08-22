"""
Stateful furniture agent with natural conversation between persistent agents.

This module implements a furniture placement workflow using persistent
SQLiteSession agents that maintain conversation memory across interactions.
"""

import json
import logging
import math
import re

from collections import Counter
from typing import Any

from scenesmith.agent_utils.assets.asset_models import AssetGenerationRequest
from scenesmith.agent_utils.assets.asset_semantics import (
    catalog_candidate_is_compatible,
    catalog_candidate_satisfies_request_details,
    tall_furniture_dimensions_are_compatible,
)
from scenesmith.agent_utils.design.placement_noise import PlacementNoiseMode
from scenesmith.agent_utils.design.room_kits import RoomKitSelection
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType

console_logger = logging.getLogger(__name__)

from scenesmith.furniture_agents.room_kit.planning import (
    _bookcase_wall_run_level_counts,
    _chair_cluster_poses,
    _nearest_level,
    _normalize_dense_library_bookcases,
    _object_matches_room_kit_slot,
    _patron_ensemble_level_counts,
    _required_room_kit_exact_level_counts,
    _required_room_kit_level_coverage,
    _required_room_kit_role_count,
    _room_kit_role_level_counts,
    _stable_chair_faces_table,
)


class FurnitureRoomKitPlacementMixin:
    """Deterministic room-kit placement and deficit recovery."""

    @staticmethod
    def _semantic_tokens(value: str) -> set[str]:
        """Normalize a short role or asset label for deterministic matching."""

        stop_words = {"a", "an", "and", "for", "of", "the", "with"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
            if token not in stop_words
        }

    @classmethod
    def _slot_relevance(cls, asset: Any, slot: Any) -> tuple[int, int, float, str]:
        """Rank one cached asset against a semantic room-kit slot."""

        role_names = (slot.role, *getattr(slot, "aliases", ()))
        normalized_roles = {
            " ".join(sorted(cls._semantic_tokens(role))) for role in role_names
        }
        asset_name = " ".join(sorted(cls._semantic_tokens(str(asset.name))))
        asset_tokens = cls._semantic_tokens(
            f"{asset.name} {getattr(asset, 'description', '')}"
        )
        role_tokens = set().union(*(cls._semantic_tokens(role) for role in role_names))
        exact = int(asset_name in normalized_roles)
        overlap = len(asset_tokens & role_tokens)
        metadata = getattr(asset, "metadata", None) or {}
        quality = float(metadata.get("asset_quality_score", 0.0))
        catalog_text = str(
            metadata.get("catalog_semantics")
            or metadata.get("ontology_path")
            or f"{asset.name} {getattr(asset, 'description', '')}"
        )
        detail_text = f"{asset.name} {getattr(asset, 'description', '')} {catalog_text}"
        compatible, _ = catalog_candidate_is_compatible(
            request_text=str(getattr(slot, "query", slot.role)),
            candidate_text=catalog_text,
            quality_score=quality,
        )
        if not compatible:
            return (-1, 0, quality, str(asset.object_id))
        compatible, _ = catalog_candidate_satisfies_request_details(
            request_text=str(getattr(slot, "query", slot.role)),
            candidate_text=catalog_text,
            supports_detail_fill=bool(metadata.get("support_zones")),
        )
        if not compatible:
            return (-1, 0, quality, str(asset.object_id))
        compatible_dimensions, _ = tall_furniture_dimensions_are_compatible(
            request_text=str(getattr(slot, "query", slot.role)),
            desired_dimensions=getattr(slot, "nominal_dimensions_m", None),
            bbox_min=getattr(asset, "bbox_min", None),
            bbox_max=getattr(asset, "bbox_max", None),
        )
        if not compatible_dimensions:
            return (-1, 0, quality, str(asset.object_id))
        query_tokens = cls._semantic_tokens(str(getattr(slot, "query", slot.role)))
        candidate_tokens = asset_tokens | cls._semantic_tokens(detail_text)
        query_overlap = len(query_tokens & candidate_tokens)
        return (
            query_overlap * 1000 + exact * 100 + overlap,
            query_overlap,
            quality,
            str(asset.object_id),
        )

    def _deterministic_room_positions(
        self, *, wall: bool
    ) -> list[tuple[float, float, float]]:
        """Return conservative unique SE(2) poses inside the room envelope."""

        half_x = max(0.5, float(self.scene.room_geometry.length) / 2.0 - 0.65)
        half_y = max(0.5, float(self.scene.room_geometry.width) / 2.0 - 0.65)
        if wall:
            return [
                (-0.55 * half_x, 0.88 * half_y, 180.0),
                (0.55 * half_x, 0.88 * half_y, 180.0),
                (-0.55 * half_x, -0.88 * half_y, 0.0),
                (0.55 * half_x, -0.88 * half_y, 0.0),
                (-0.88 * half_x, 0.45 * half_y, -90.0),
                (-0.88 * half_x, -0.45 * half_y, -90.0),
                (0.88 * half_x, 0.45 * half_y, 90.0),
                (0.88 * half_x, -0.45 * half_y, 90.0),
            ]
        return [
            (0.0, -0.18 * half_y, 0.0),
            (-0.42 * half_x, -0.18 * half_y, -90.0),
            (0.42 * half_x, -0.18 * half_y, 90.0),
            (-0.42 * half_x, 0.38 * half_y, -135.0),
            (0.42 * half_x, 0.38 * half_y, 135.0),
            (0.0, 0.58 * half_y, 180.0),
            (-0.68 * half_x, -0.58 * half_y, -45.0),
            (0.68 * half_x, -0.58 * half_y, 45.0),
            (-0.68 * half_x, 0.68 * half_y, -135.0),
            (0.68 * half_x, 0.68 * half_y, 135.0),
            (0.0, -0.72 * half_y, 0.0),
            (-0.72 * half_x, 0.0, -90.0),
            (0.72 * half_x, 0.0, 90.0),
        ]

    def _bookcase_wall_run_candidates(
        self, asset: Any
    ) -> list[list[tuple[float, float, float]]]:
        """Return bounded three-case runs along each wall, away from corners."""

        try:
            width = max(
                abs(float(asset.bbox_max[0]) - float(asset.bbox_min[0])),
                abs(float(asset.bbox_max[1]) - float(asset.bbox_min[1])),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            width = 1.0
        spacing = min(1.45, max(0.8, width + 0.12))
        half_x = max(0.5, float(self.scene.room_geometry.length) / 2.0 - 0.65)
        half_y = max(0.5, float(self.scene.room_geometry.width) / 2.0 - 0.65)
        wall_x = 0.88 * half_x
        wall_y = 0.88 * half_y
        centers = (-2.0 * spacing, 0.0, 2.0 * spacing)
        runs: list[list[tuple[float, float, float]]] = []
        for center in centers:
            offsets = (center - spacing, center, center + spacing)
            if max(abs(offset) for offset in offsets) <= half_x - 0.4:
                runs.append([(offset, wall_y, 180.0) for offset in offsets])
                runs.append([(offset, -wall_y, 0.0) for offset in offsets])
            if max(abs(offset) for offset in offsets) <= half_y - 0.4:
                runs.append([(-wall_x, offset, -90.0) for offset in offsets])
                runs.append([(wall_x, offset, 90.0) for offset in offsets])
        return runs

    def _place_bookcase_wall_run_deterministically(
        self,
        asset: Any,
        slot: Any,
        elevation: float,
        support_elevations: tuple[float, ...],
    ) -> int:
        """Place one complete collision-validated wall run or leave no partial run."""

        if (
            _bookcase_wall_run_level_counts(
                self.scene.objects.values(), slot, support_elevations
            )[elevation]
            >= 3
        ):
            return 0
        for candidate_run in self._bookcase_wall_run_candidates(asset):
            added_ids: list[str] = []
            complete = True
            for x, y, yaw in candidate_run:
                before_ids = set(self.scene.objects)
                raw_result = self.furniture_tools._add_furniture_to_scene_impl(
                    asset_id=str(asset.object_id),
                    x=x,
                    y=y,
                    z=elevation,
                    roll=0.0,
                    pitch=0.0,
                    yaw=yaw,
                )
                try:
                    result_payload = json.loads(raw_result)
                    success = bool(result_payload.get("success"))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    result_payload = {}
                    success = False
                object_id = str(result_payload.get("object_id") or "")
                if success and not object_id:
                    new_ids = sorted(set(self.scene.objects) - before_ids)
                    object_id = new_ids[0] if len(new_ids) == 1 else ""
                placed_object = self.scene.objects.get(object_id)
                actual_level = (
                    _nearest_level(placed_object, support_elevations)
                    if placed_object is not None
                    else None
                )
                if not success or not object_id or actual_level != elevation:
                    if object_id and object_id in self.scene.objects:
                        self.furniture_tools._remove_furniture_impl(object_id)
                    complete = False
                    break
                added_ids.append(object_id)
            if complete and len(added_ids) == 3:
                run_size = _bookcase_wall_run_level_counts(
                    self.scene.objects.values(), slot, support_elevations
                )[elevation]
                if run_size >= 3:
                    console_logger.info(
                        "Deterministic recovery placed a contiguous 3-case "
                        "bookshelf wall run at %.3fm",
                        elevation,
                    )
                    return 3
            for object_id in reversed(added_ids):
                self.furniture_tools._remove_furniture_impl(object_id)
        console_logger.warning(
            "Deterministic recovery could not place a complete bookshelf wall "
            "run at %.3fm without violating placement constraints",
            elevation,
        )
        return 0

    def _place_room_kit_minimums_deterministically(
        self, room_kit: RoomKitSelection
    ) -> int:
        """Recover required kit roles from acquired assets without another model call.

        Every attempt goes through ``FurnitureTools`` so structural support,
        enclosure, contextual, and collision validation remain authoritative.
        """

        assets = [
            asset
            for asset in self.asset_manager.list_available_assets()
            if asset.object_type == ObjectType.FURNITURE
        ]

        self.furniture_tools.set_noise_profile(PlacementNoiseMode.PERFECT)
        support_elevations = self.furniture_tools._major_support_elevations()
        level_requirements = _required_room_kit_level_coverage(
            str(getattr(self.scene, "text_description", "")),
            room_kit,
            support_elevations,
        )
        table_slot = next(
            (slot for slot in room_kit.slots if slot.role == "reading_table"), None
        )
        chair_slot = next(
            (slot for slot in room_kit.slots if slot.role == "reading_chair"), None
        )
        attempted_positions: set[tuple[float, float, float]] = set()
        level_counts = {elevation: 0 for elevation in support_elevations}
        for scene_object in self.scene.objects.values():
            if scene_object.object_type != ObjectType.FURNITURE:
                continue
            try:
                object_elevation = float(scene_object.transform.translation()[2])
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            nearest = min(
                support_elevations,
                key=lambda elevation: abs(elevation - object_elevation),
            )
            level_counts[nearest] += 1
        placed = 0

        for slot in room_kit.slots:
            if not slot.required:
                continue
            existing = sum(
                obj.object_type == ObjectType.FURNITURE
                and _object_matches_room_kit_slot(obj, slot)
                for obj in self.scene.objects.values()
            )
            aggregate_missing = max(
                0,
                _required_room_kit_role_count(room_kit, slot) - existing,
            )
            level_targets: list[float] = []
            wall_run_targets: list[float] = []
            required_per_level = level_requirements.get(slot.role)
            if required_per_level is not None:
                targets_by_level = _required_room_kit_exact_level_counts(
                    self.scene,
                    room_kit,
                    slot,
                    support_elevations,
                ) or {elevation: required_per_level for elevation in support_elevations}
                if (
                    slot.role == "reading_chair"
                    and table_slot is not None
                    and chair_slot is not None
                ):
                    role_level_counts = _patron_ensemble_level_counts(
                        self.scene.objects.values(),
                        table_slot,
                        chair_slot,
                        support_elevations,
                    )
                else:
                    role_level_counts = _room_kit_role_level_counts(
                        self.scene.objects.values(),
                        slot,
                        support_elevations,
                    )
                for elevation in support_elevations:
                    level_targets.extend(
                        [elevation]
                        * max(
                            0,
                            targets_by_level[elevation] - role_level_counts[elevation],
                        )
                    )
                if slot.role == "bookshelf":
                    wall_run_counts = _bookcase_wall_run_level_counts(
                        self.scene.objects.values(), slot, support_elevations
                    )
                    wall_run_targets = [
                        elevation
                        for elevation in support_elevations
                        if wall_run_counts[elevation] < 3
                    ]
            missing = max(
                aggregate_missing,
                len(level_targets),
                3 * len(wall_run_targets),
            )
            if missing == 0:
                continue

            ranked = sorted(
                assets,
                key=lambda asset: self._slot_relevance(asset, slot),
                reverse=True,
            )
            if not ranked or self._slot_relevance(ranked[0], slot)[0] <= 0:
                try:
                    result = self.asset_manager.generate_assets(
                        AssetGenerationRequest(
                            object_descriptions=[
                                str(getattr(slot, "query", slot.role))
                            ],
                            short_names=[slot.role],
                            object_type=ObjectType.FURNITURE,
                            desired_dimensions=[
                                list(
                                    getattr(
                                        slot,
                                        "nominal_dimensions_m",
                                        (1.0, 1.0, 1.0),
                                    )
                                )
                            ],
                            style_context=str(
                                getattr(self.scene, "text_description", "")
                            ),
                            scene_id=getattr(
                                getattr(self.scene, "scene_dir", None),
                                "name",
                                None,
                            ),
                        )
                    )
                    console_logger.info(
                        "Deterministic recovery acquired %d asset(s) for missing "
                        "room-kit role %s",
                        len(result.successful_assets),
                        slot.role,
                    )
                except Exception as exc:
                    console_logger.warning(
                        "Deterministic recovery could not acquire missing room-kit "
                        "role %s: %s",
                        slot.role,
                        exc,
                    )
                assets = [
                    asset
                    for asset in self.asset_manager.list_available_assets()
                    if asset.object_type == ObjectType.FURNITURE
                ]
                ranked = sorted(
                    assets,
                    key=lambda asset: self._slot_relevance(asset, slot),
                    reverse=True,
                )
            if not ranked or self._slot_relevance(ranked[0], slot)[0] <= 0:
                console_logger.warning(
                    "No cached furniture asset matched required room-kit role %s",
                    slot.role,
                )
                continue
            asset = ranked[0]
            if slot.role == "bookshelf" and wall_run_targets:
                for elevation in wall_run_targets:
                    placed += self._place_bookcase_wall_run_deterministically(
                        asset,
                        slot,
                        elevation,
                        support_elevations,
                    )
                existing = sum(
                    obj.object_type == ObjectType.FURNITURE
                    and _object_matches_room_kit_slot(obj, slot)
                    for obj in self.scene.objects.values()
                )
                aggregate_missing = max(
                    0,
                    _required_room_kit_role_count(room_kit, slot) - existing,
                )
                role_level_counts = _room_kit_role_level_counts(
                    self.scene.objects.values(),
                    slot,
                    support_elevations,
                )
                level_targets = []
                if required_per_level is not None:
                    for elevation in support_elevations:
                        level_targets.extend(
                            [elevation]
                            * max(
                                0,
                                targets_by_level[elevation]
                                - role_level_counts[elevation],
                            )
                        )
                missing = max(aggregate_missing, len(level_targets))
                if missing == 0:
                    continue
            positions = self._deterministic_room_positions(
                wall=getattr(slot, "placement_class", "floor") == "wall"
            )
            if slot.role == "reading_table" and level_targets:
                role_anchors: list[tuple[float, float, float]] = []
                for scene_object in self.scene.objects.values():
                    if not _object_matches_room_kit_slot(scene_object, slot):
                        continue
                    try:
                        translation = scene_object.transform.translation()
                        yaw = math.degrees(
                            scene_object.transform.rotation()
                            .ToRollPitchYaw()
                            .yaw_angle()
                        )
                    except (AttributeError, IndexError, TypeError, ValueError):
                        try:
                            translation = scene_object.transform.translation()
                        except (AttributeError, IndexError, TypeError, ValueError):
                            continue
                        yaw = 0.0
                    anchor = (float(translation[0]), float(translation[1]), yaw)
                    if anchor not in role_anchors:
                        role_anchors.append(anchor)
                positions = [*role_anchors, *positions]

            cluster_ids: dict[float, list[str]] = {
                elevation: [] for elevation in set(level_targets)
            }
            for recovery_index in range(missing):
                success = False
                target_elevation = (
                    level_targets[recovery_index]
                    if recovery_index < len(level_targets)
                    else None
                )
                candidate_elevations = (
                    (target_elevation,)
                    if target_elevation is not None
                    else tuple(
                        sorted(
                            support_elevations,
                            key=lambda value: (level_counts[value], value),
                        )
                    )
                )
                for elevation in candidate_elevations:
                    candidate_positions = positions
                    if slot.role == "reading_chair" and table_slot is not None:
                        tables = sorted(
                            (
                                obj
                                for obj in self.scene.objects.values()
                                if _object_matches_room_kit_slot(obj, table_slot)
                                and _nearest_level(obj, support_elevations) == elevation
                            ),
                            key=lambda table: (
                                -sum(
                                    _stable_chair_faces_table(chair, table)
                                    for chair in (
                                        obj
                                        for obj in self.scene.objects.values()
                                        if _object_matches_room_kit_slot(obj, slot)
                                        and _nearest_level(obj, support_elevations)
                                        == elevation
                                    )
                                ),
                                str(getattr(table, "object_id", "")),
                            ),
                        )
                        if tables:
                            candidate_positions = []
                            seen_poses: set[tuple[float, float, float]] = set()
                            for table in tables:
                                for pose in _chair_cluster_poses(table, asset):
                                    pose_key = tuple(round(value, 4) for value in pose)
                                    if pose_key in seen_poses:
                                        continue
                                    seen_poses.add(pose_key)
                                    candidate_positions.append(pose)
                    for x, y, yaw in candidate_positions:
                        position_key = (
                            round(x, 4),
                            round(y, 4),
                            round(elevation, 4),
                        )
                        if position_key in attempted_positions:
                            continue
                        attempted_positions.add(position_key)
                        raw_result = self.furniture_tools._add_furniture_to_scene_impl(
                            asset_id=str(asset.object_id),
                            x=x,
                            y=y,
                            z=elevation,
                            roll=0.0,
                            pitch=0.0,
                            yaw=yaw,
                        )
                        try:
                            result_payload = json.loads(raw_result)
                            success = bool(result_payload.get("success"))
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            result_payload = {}
                            success = False
                        object_id = str(result_payload.get("object_id") or "")
                        placed_object = self.scene.objects.get(object_id)
                        actual_level = (
                            _nearest_level(placed_object, support_elevations)
                            if placed_object is not None
                            else None
                        )
                        if (
                            success
                            and target_elevation is not None
                            and actual_level is not None
                            and actual_level != elevation
                        ):
                            self.furniture_tools._remove_furniture_impl(object_id)
                            console_logger.warning(
                                "Deterministic recovery rejected %s at %.3fm: "
                                "support resolution placed it at %.3fm",
                                slot.role,
                                elevation,
                                actual_level,
                            )
                            success = False
                        if success:
                            level_counts[elevation] += 1
                            if (
                                slot.role == "reading_chair"
                                and elevation in cluster_ids
                            ):
                                if object_id:
                                    cluster_ids[elevation].append(object_id)
                            break
                    if success:
                        break
                if success:
                    placed += 1
                if not success:
                    console_logger.warning(
                        "Deterministic recovery exhausted valid poses for room-kit "
                        "role %s after placing %d of %d missing instances",
                        slot.role,
                        placed,
                        missing,
                    )
                    break

            if slot.role == "reading_chair" and cluster_ids:
                required_by_level = Counter(level_targets)
                for elevation, object_ids in cluster_ids.items():
                    if len(object_ids) >= required_by_level[elevation]:
                        continue
                    for object_id in reversed(object_ids):
                        self.furniture_tools._remove_furniture_impl(object_id)
                    placed -= len(object_ids)
                    level_counts[elevation] -= len(object_ids)
                    console_logger.warning(
                        "Rolled back incomplete patron chair cluster at %.3fm: "
                        "placed %d of %d required chairs",
                        elevation,
                        len(object_ids),
                        required_by_level[elevation],
                    )

        return placed

    def _preprune_and_recover_room_kit(
        self, room_kit: RoomKitSelection
    ) -> tuple[int, int]:
        """Clear dense-library surplus before bounded wall-run recovery."""

        support_elevations = self.furniture_tools._major_support_elevations()
        pruned = _normalize_dense_library_bookcases(
            self.scene,
            room_kit,
            support_elevations,
            remove_object=self.furniture_tools._remove_furniture_impl,
        )
        recovered = self._place_room_kit_minimums_deterministically(room_kit)
        return pruned, recovered
