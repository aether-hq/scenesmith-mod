"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import logging
import re

from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    PlacementInfo,
    SceneObject,
    UniqueID,
)

console_logger = logging.getLogger(__name__)


class ManipulandTemplateRecoveryMixin:
    """Template cloning, deficit recovery, and surplus normalization."""

    @staticmethod
    def _furniture_template_key(furniture: SceneObject) -> tuple[str, float] | None:
        """Identify interchangeable instances without an LLM call."""
        if furniture.geometry_path is None:
            return None
        return (
            str(furniture.geometry_path.resolve()),
            round(furniture.scale_factor, 6),
        )

    def _clone_manipulands_between_identical_furniture(
        self,
        source_id: UniqueID,
        target_id: UniqueID,
        *,
        dense_book_rows_only: bool = False,
        excluded_object_ids: set[UniqueID] | None = None,
    ) -> int:
        """Copy a composed surface arrangement into an identical asset frame.

        Surface-relative rigid transforms are transferred exactly, so repeated
        beds/tables/shelves do not require another planner/designer tool loop.
        """
        source = self.scene.get_object(source_id)
        target = self.scene.get_object(target_id)
        if source is None or target is None:
            return 0
        if len(source.support_surfaces) != len(target.support_surfaces):
            return 0

        surface_pairs = {
            source_surface.surface_id: target_surface
            for source_surface, target_surface in zip(
                source.support_surfaces, target.support_surfaces, strict=True
            )
        }
        excluded_object_ids = excluded_object_ids or set()
        originals = [
            obj
            for obj in list(self.scene.objects.values())
            if obj.object_type == ObjectType.MANIPULAND
            and obj.placement_info is not None
            and obj.placement_info.parent_surface_id in surface_pairs
            and obj.object_id not in excluded_object_ids
            and (
                not dense_book_rows_only
                or bool((obj.metadata or {}).get("dense_library_book_row"))
            )
        ]
        for original in originals:
            source_surface = next(
                surface
                for surface in source.support_surfaces
                if surface.surface_id == original.placement_info.parent_surface_id
            )
            target_surface = surface_pairs[source_surface.surface_id]
            relative_transform = source_surface.transform.inverse() @ original.transform
            target_transform = target_surface.transform @ relative_transform
            position_2d, rotation_2d = target_surface.from_world_pose(target_transform)
            clone = SceneObject(
                object_id=self.scene.generate_unique_id(original.name),
                object_type=original.object_type,
                name=original.name,
                description=original.description,
                transform=target_transform,
                geometry_path=original.geometry_path,
                sdf_path=original.sdf_path,
                image_path=original.image_path,
                support_surfaces=[],
                placement_info=PlacementInfo(
                    parent_surface_id=target_surface.surface_id,
                    position_2d=position_2d,
                    rotation_2d=rotation_2d,
                    placement_method="template_transfer",
                ),
                metadata=original.metadata.copy(),
                bbox_min=(
                    original.bbox_min.copy() if original.bbox_min is not None else None
                ),
                bbox_max=(
                    original.bbox_max.copy() if original.bbox_max is not None else None
                ),
                immutable=original.immutable,
                scale_factor=original.scale_factor,
            )
            if bool(clone.metadata.get("dense_library_book_row")):
                self._bind_dense_book_row_to_owner_surface(
                    clone,
                    target,
                    target_surface,
                )
            self.scene.add_object(clone)
        return len(originals)

    def _recover_dense_library_book_row_deficits(self) -> int:
        """Fill story deficits from proven rows on identical local bookcases."""

        normalized = str(getattr(self.scene, "text_description", "")).casefold()
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
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            and any(
                term in f"{obj.name} {obj.description}".casefold()
                for term in ("shelf", "bookcase")
            )
        ]
        bookcase_levels = self._dense_bookcase_story_levels(bookcases)
        bookcases_by_level: dict[float, list[SceneObject]] = {}
        for bookcase in bookcases:
            level = bookcase_levels.get(bookcase.object_id)
            if level is None:
                continue
            bookcases_by_level.setdefault(level, []).append(bookcase)
        if len(bookcases_by_level) < 2:
            return 0

        invalid_row_ids = self._physically_invalid_dense_book_row_ids(
            self.scene,
            self.cfg,
        )
        required_per_level = 12
        recovered = 0
        for level in sorted(bookcases_by_level):
            level_bookcases = sorted(
                bookcases_by_level[level], key=lambda obj: str(obj.object_id)
            )
            rows_by_bookcase = {
                bookcase.object_id: [
                    row
                    for row in self._dense_book_rows_on_furniture(self.scene, bookcase)
                    if row.object_id not in invalid_row_ids
                ]
                for bookcase in level_bookcases
            }
            level_count = sum(len(rows) for rows in rows_by_bookcase.values())
            grouped_bookcase_ids = {
                bookcase.object_id
                for bookcase in level_bookcases
                if (bookcase.metadata or {}).get("dense_library_grouped_run")
                is not None
            }

            def populated_grouped_cases() -> int:
                return sum(
                    len(rows_by_bookcase[bookcase_id]) >= 3
                    for bookcase_id in grouped_bookcase_ids
                )

            if level_count >= required_per_level and (
                not grouped_bookcase_ids or populated_grouped_cases() >= 3
            ):
                continue

            targets = [
                bookcase
                for bookcase in level_bookcases
                if not rows_by_bookcase[bookcase.object_id]
            ]
            targets.sort(
                key=lambda bookcase: (
                    (bookcase.metadata or {}).get("dense_library_grouped_run") is None,
                    str(bookcase.object_id),
                )
            )
            for target in targets:
                target_key = self._furniture_template_key(target)
                if target_key is None:
                    continue
                compatible_sources = [
                    source
                    for source in level_bookcases
                    if source.object_id != target.object_id
                    and self._furniture_template_key(source) == target_key
                    and rows_by_bookcase[source.object_id]
                ]
                if not compatible_sources:
                    continue
                source = max(
                    compatible_sources,
                    key=lambda obj: (
                        len(rows_by_bookcase[obj.object_id]),
                        str(obj.object_id),
                    ),
                )
                before_ids = set(self.scene.objects)
                self._clone_manipulands_between_identical_furniture(
                    source.object_id,
                    target.object_id,
                    dense_book_rows_only=True,
                    excluded_object_ids=invalid_row_ids,
                )
                new_row_ids = {
                    object_id
                    for object_id in set(self.scene.objects) - before_ids
                    if bool(
                        (self.scene.objects[object_id].metadata or {}).get(
                            "dense_library_book_row"
                        )
                    )
                }
                if not new_row_ids:
                    continue
                newly_invalid = self._physically_invalid_dense_book_row_ids(
                    self.scene,
                    self.cfg,
                ).intersection(new_row_ids)
                for object_id in newly_invalid:
                    self.scene.remove_object(object_id)
                surviving = new_row_ids - newly_invalid
                rows_by_bookcase[target.object_id] = [
                    self.scene.objects[object_id] for object_id in surviving
                ]
                recovered += len(surviving)
                level_count += len(surviving)
                if level_count >= required_per_level and (
                    not grouped_bookcase_ids or populated_grouped_cases() >= 3
                ):
                    break

        return recovered

    def _normalize_dense_library_book_row_surplus(self) -> int:
        """Prune dense rows to the canonical count while preserving wall runs."""

        normalized = str(getattr(self.scene, "text_description", "")).casefold()
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
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            and any(
                term in f"{obj.name} {obj.description}".casefold()
                for term in ("shelf", "bookcase")
            )
        ]
        bookcase_levels = self._dense_bookcase_story_levels(bookcases)
        levels = sorted(set(bookcase_levels.values()))
        if len(levels) < 2:
            return 0

        invalid_row_ids = self._physically_invalid_dense_book_row_ids(
            self.scene,
            self.cfg,
        )
        required_per_level = 12
        removed = 0
        for level in levels:
            level_bookcases = sorted(
                (
                    bookcase
                    for bookcase in bookcases
                    if bookcase_levels.get(bookcase.object_id) == level
                ),
                key=lambda obj: str(obj.object_id),
            )
            rows_by_bookcase = {
                bookcase.object_id: sorted(
                    (
                        row
                        for row in self._dense_book_rows_on_furniture(
                            self.scene,
                            bookcase,
                        )
                        if row.object_id not in invalid_row_ids
                    ),
                    key=lambda row: str(row.object_id),
                )
                for bookcase in level_bookcases
            }
            level_rows = [row for rows in rows_by_bookcase.values() for row in rows]
            if len(level_rows) <= required_per_level:
                continue

            retained_ids: set[UniqueID] = set()
            grouped_bookcases = [
                bookcase
                for bookcase in level_bookcases
                if (bookcase.metadata or {}).get("dense_library_grouped_run")
                is not None
                and len(rows_by_bookcase[bookcase.object_id]) >= 3
            ]
            for bookcase in grouped_bookcases[:3]:
                retained_ids.update(
                    row.object_id for row in rows_by_bookcase[bookcase.object_id][:3]
                )

            remaining = sorted(
                (row for row in level_rows if row.object_id not in retained_ids),
                key=lambda row: str(row.object_id),
            )
            retained_ids.update(
                row.object_id
                for row in remaining[: max(0, required_per_level - len(retained_ids))]
            )
            for row in level_rows:
                if row.object_id in retained_ids:
                    continue
                self.scene.remove_object(row.object_id)
                removed += 1

        return removed
