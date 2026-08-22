"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import json
import logging
import re

from types import SimpleNamespace

from agents.exceptions import ModelBehaviorError

from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.manipuland_agents.mixins.dense_assets import DenseBookRowAssetMixin

console_logger = logging.getLogger(__name__)


class DenseBookRowPlacementMixin:
    """Collision-safe dense-row placement and completion validation."""

    def _dense_book_row_pose_is_collision_free(
        self,
        furniture_id: UniqueID,
        row_id: UniqueID,
    ) -> bool:
        """Validate a tentative row against its owner and room structure."""

        furniture = self.scene.get_object(furniture_id)
        row = self.scene.get_object(row_id)
        if furniture is None or row is None:
            return False
        surface = next(
            (
                candidate
                for candidate in furniture.support_surfaces
                if row.placement_info is not None
                and candidate.surface_id == row.placement_info.parent_surface_id
            ),
            None,
        )
        if surface is None or not self._dense_book_row_is_contained(row, surface):
            return False
        invalid = self._physically_invalid_dense_book_row_ids(
            self.scene,
            self.cfg,
            row_ids={row_id},
        )
        if invalid:
            console_logger.warning(
                "Rejected dense book-row pose for %s on %s due to containment or "
                "non-owner collision",
                row_id,
                furniture_id,
            )
        return not invalid

    @staticmethod
    def _dense_book_rows_on_furniture(
        scene: RoomScene,
        furniture: SceneObject,
    ) -> list[SceneObject]:
        surface_ids = {surface.surface_id for surface in furniture.support_surfaces}
        return [
            obj
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.MANIPULAND
            and bool((obj.metadata or {}).get("dense_library_book_row"))
            and obj.placement_info is not None
            and obj.placement_info.parent_surface_id in surface_ids
        ]

    def _place_dense_book_rows_deterministically(
        self,
        furniture: SceneObject,
    ) -> int:
        """Place one proven multi-book artifact on every compatible internal tier."""

        if not self._requests_dense_book_rows(
            furniture, self.current_furniture_selection
        ):
            return 0
        row_asset = self._ensure_dense_book_row_asset()
        if (
            row_asset is None
            or row_asset.bbox_min is None
            or row_asset.bbox_max is None
        ):
            return 0

        row_height = float(row_asset.bbox_max[2] - row_asset.bbox_min[2])
        existing_surface_ids = {
            obj.placement_info.parent_surface_id
            for obj in getattr(self.scene, "objects", {}).values()
            if obj.object_type == ObjectType.MANIPULAND
            and obj.placement_info is not None
            and bool((obj.metadata or {}).get("dense_library_book_row"))
        }
        previous_noise_profile = getattr(
            self.manipuland_tools, "active_noise_profile", None
        )
        if previous_noise_profile is not None:
            self.manipuland_tools.active_noise_profile = SimpleNamespace(
                position_xy_std_meters=0.0,
                rotation_yaw_std_degrees=0.0,
            )
        placed = 0
        try:
            for surface in self._internal_bookcase_surfaces(furniture):
                if surface.surface_id in existing_surface_ids:
                    continue
                clearance = float(
                    surface.bounding_box_max[2] - surface.bounding_box_min[2]
                )
                if row_height > clearance + 1e-6:
                    continue
                minimum = surface.bounding_box_min
                maximum = surface.bounding_box_max
                center_x = float((minimum[0] + maximum[0]) / 2.0)
                center_y = float((minimum[1] + maximum[1]) / 2.0)
                span_x = float(maximum[0] - minimum[0])
                positions = tuple(
                    (center_x + fraction * span_x, center_y)
                    for fraction in (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3)
                )
                candidates = (
                    (position_x, position_y, rotation_degrees)
                    for rotation_degrees in (0.0, 180.0)
                    for position_x, position_y in positions
                )
                for position_x, position_y, rotation_degrees in candidates:
                    raw_result = (
                        self.manipuland_tools._place_manipuland_on_surface_impl(
                            asset_id=str(row_asset.object_id),
                            surface_id=str(surface.surface_id),
                            position_x=position_x,
                            position_z=position_y,
                            rotation_degrees=rotation_degrees,
                            _action_metadata={
                                "furniture_id": str(self.current_furniture_id),
                                "surface_id": str(surface.surface_id),
                                "placement_method": "deterministic_dense_book_row",
                            },
                        )
                    )
                    try:
                        result = json.loads(raw_result)
                    except (json.JSONDecodeError, TypeError):
                        result = {}
                    if not result.get("success"):
                        continue
                    object_id = result.get("object_id")
                    placed_object = (
                        self.scene.get_object(UniqueID(object_id))
                        if object_id
                        else None
                    )
                    if placed_object is not None:
                        if not self._dense_book_row_pose_is_collision_free(
                            furniture.object_id,
                            placed_object.object_id,
                        ):
                            self.scene.remove_object(placed_object.object_id)
                            continue
                        self._bind_dense_book_row_to_owner_surface(
                            placed_object,
                            furniture,
                            surface,
                        )
                    placed += 1
                    break
        finally:
            if previous_noise_profile is not None:
                self.manipuland_tools.active_noise_profile = previous_noise_profile
        return placed

    @staticmethod
    def _validate_dense_library_book_rows(
        scene: RoomScene,
        *,
        invalid_row_ids: set[UniqueID] | None = None,
    ) -> int:
        """Require surviving intrinsic book rows on every authored library story."""

        normalized = str(getattr(scene, "text_description", "")).casefold()
        explicit_dense_library = (
            "library" in normalized
            and "large" in normalized
            and "thousand" in normalized
            and bool(re.search(r"\bmulti[ -]?level\b", normalized))
        )
        if not explicit_dense_library:
            return 0

        bookcases = [
            obj
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            and any(
                term in f"{obj.name} {obj.description}".casefold()
                for term in ("shelf", "bookcase")
            )
        ]
        bookcase_levels = DenseBookRowAssetMixin._dense_bookcase_story_levels(bookcases)
        support_levels: dict[UniqueID, float] = {}
        levels = set(bookcase_levels.values())
        for bookcase in bookcases:
            level = bookcase_levels.get(bookcase.object_id)
            if level is None:
                continue
            support_levels.update(
                {surface.surface_id: level for surface in bookcase.support_surfaces}
            )
        if len(levels) < 2:
            return 0

        invalid_row_ids = invalid_row_ids or set()
        tagged_invalid = sorted(
            str(obj.object_id)
            for obj in scene.objects.values()
            if obj.object_id in invalid_row_ids
            and obj.object_type == ObjectType.MANIPULAND
            and bool((obj.metadata or {}).get("dense_library_book_row"))
        )
        if tagged_invalid:
            raise ModelBehaviorError(
                "Dense library contains physically invalid tagged book rows: "
                f"{', '.join(tagged_invalid)}. The detail stage cannot publish "
                "this checkpoint."
            )

        counts = {level: 0 for level in levels}
        owner_by_surface = DenseBookRowAssetMixin._dense_book_row_owner_by_surface(
            scene
        )
        row_counts_by_owner = {bookcase.object_id: 0 for bookcase in bookcases}
        for obj in scene.objects.values():
            if (
                obj.object_type != ObjectType.MANIPULAND
                or not bool((obj.metadata or {}).get("dense_library_book_row"))
                or obj.placement_info is None
            ):
                continue
            level = support_levels.get(obj.placement_info.parent_surface_id)
            if level is not None:
                counts[level] += 1
                owner_id = owner_by_surface.get(obj.placement_info.parent_surface_id)
                if owner_id in row_counts_by_owner:
                    row_counts_by_owner[owner_id] += 1

        required_per_level = 12
        deficits = [
            (level, counts[level])
            for level in sorted(levels)
            if counts[level] < required_per_level
        ]
        if deficits:
            details = "; ".join(
                f"{level:.3f}m placed {placed}, required {required_per_level}"
                for level, placed in deficits
            )
            raise ModelBehaviorError(
                "Dense library book-row coverage deficits: "
                f"{details}. The detail stage cannot publish this checkpoint."
            )

        grouped_run_deficits: list[tuple[float, int]] = []
        for level in sorted(levels):
            grouped_bookcases = [
                bookcase
                for bookcase in bookcases
                if bookcase_levels.get(bookcase.object_id) == level
                and (bookcase.metadata or {}).get("dense_library_grouped_run")
                is not None
            ]
            if not grouped_bookcases:
                continue
            populated = sum(
                row_counts_by_owner.get(bookcase.object_id, 0) >= 3
                for bookcase in grouped_bookcases
            )
            if populated < 3:
                grouped_run_deficits.append((level, populated))
        if grouped_run_deficits:
            details = "; ".join(
                f"populated bookcase wall run at {level:.3f}m has {populated}, "
                "required 3"
                for level, populated in grouped_run_deficits
            )
            raise ModelBehaviorError(
                "Dense library grouped-run population deficits: "
                f"{details}. The detail stage cannot publish this checkpoint."
            )
        return sum(counts.values())
