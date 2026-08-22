"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import json
import logging
import re

import numpy as np

from omegaconf import DictConfig
from pydrake.math import RigidTransform

from scenesmith.agent_utils.assets.asset_models import AssetGenerationRequest
from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    SupportSurface,
    UniqueID,
)
from scenesmith.agent_utils.scene.scene_analyzer import FurnitureSelection

console_logger = logging.getLogger(__name__)


class DenseBookRowAssetMixin:
    """Dense-row asset selection, binding, containment, and normalization."""

    @staticmethod
    def _semantic_tokens(value: str) -> set[str]:
        """Return stable content tokens for cheap cached-asset matching."""
        stop_words = {
            "a",
            "an",
            "and",
            "for",
            "of",
            "on",
            "small",
            "the",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
            if token not in stop_words
        }

    def _place_cached_assets_deterministically(self) -> int:
        """Place locally cached semantic matches on the current support surface.

        This is deliberately a fallback, not another generative path. It is
        network-free, uses the same validated placement primitive as the agent,
        and keeps a useful checkpoint when any LLM provider is slow or malformed.
        """
        selection = self.current_furniture_selection
        if selection is None or not self.manipuland_tools.support_surfaces:
            return 0

        suggestion_text = str(selection.suggested_items or "")
        suggestion_tokens = self._semantic_tokens(suggestion_text)
        assets = [
            asset
            for asset in self.asset_manager.list_available_assets()
            if asset.object_type == ObjectType.MANIPULAND
        ]

        def relevance(asset: SceneObject) -> tuple[int, float, str]:
            label = f"{asset.name} {asset.description}"
            label_tokens = self._semantic_tokens(label)
            normalized_name = asset.name.lower().replace("_", " ")
            exact = int(normalized_name in suggestion_text.lower())
            overlap = len(suggestion_tokens & label_tokens)
            quality = float(asset.metadata.get("asset_quality_score", 0.0))
            return (exact * 100 + overlap, quality, str(asset.object_id))

        ranked = sorted(assets, key=relevance, reverse=True)
        matched = [asset for asset in ranked if relevance(asset)[0] > 0][:3]
        if not matched:
            return 0

        surfaces = sorted(
            self.manipuland_tools.support_surfaces.values(),
            key=lambda surface: (surface.area, str(surface.surface_id)),
            reverse=True,
        )
        fractions_by_count = {
            1: [0.0],
            2: [-0.22, 0.22],
            3: [-0.30, 0.0, 0.30],
        }
        fractions = fractions_by_count[len(matched)]
        placed = 0

        for index, asset in enumerate(matched):
            for surface in surfaces:
                minimum = surface.bounding_box_min
                maximum = surface.bounding_box_max
                center_x = float((minimum[0] + maximum[0]) / 2.0)
                center_y = float((minimum[1] + maximum[1]) / 2.0)
                span_x = float(maximum[0] - minimum[0])
                span_y = float(maximum[1] - minimum[1])
                offset = fractions[index]
                if span_x >= span_y:
                    primary = (center_x + offset * span_x, center_y)
                else:
                    primary = (center_x, center_y + offset * span_y)
                # The first pose provides semantic spacing. Center and modest
                # cross-axis offsets make the fallback resilient to non-rectangular
                # support meshes while all bounds remain validated by the tool.
                candidates = [
                    primary,
                    (center_x, center_y),
                    (center_x + 0.12 * span_x, center_y - 0.12 * span_y),
                    (center_x - 0.12 * span_x, center_y + 0.12 * span_y),
                ]
                for position_x, position_y in candidates:
                    raw_result = (
                        self.manipuland_tools._place_manipuland_on_surface_impl(
                            asset_id=str(asset.object_id),
                            surface_id=str(surface.surface_id),
                            position_x=position_x,
                            position_z=position_y,
                            rotation_degrees=0.0,
                            _action_metadata={
                                "furniture_id": str(self.current_furniture_id),
                                "surface_id": str(surface.surface_id),
                                "placement_method": "deterministic_llm_fallback",
                            },
                        )
                    )
                    try:
                        success = bool(json.loads(raw_result).get("success"))
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        success = False
                    if success:
                        placed += 1
                        break
                if success:
                    break

        return placed

    @staticmethod
    def _is_intrinsic_catalog_book_row_asset(asset: SceneObject) -> bool:
        """Require catalog evidence for the visible multi-book row artifact."""

        if asset.object_type != ObjectType.MANIPULAND:
            return False
        catalog_id = str((asset.metadata or {}).get("catalog_id") or "").casefold()
        return catalog_id.endswith("book_encyclopedia_set_01")

    @staticmethod
    def _requests_dense_book_rows(
        furniture: SceneObject,
        selection: FurnitureSelection | None,
    ) -> bool:
        if selection is None:
            return False
        furniture_text = f"{furniture.name} {furniture.description}".casefold()
        suggestion_text = str(selection.suggested_items or "").casefold()
        return (
            any(term in furniture_text for term in ("shelf", "bookcase"))
            and "dense rows" in suggestion_text
            and "book" in suggestion_text
        )

    def _ensure_dense_book_row_asset(self) -> SceneObject | None:
        """Reuse or acquire the exact authored encyclopedia-set catalog mesh."""

        available = self.asset_manager.list_available_assets()
        row_asset = next(
            (
                asset
                for asset in available
                if self._is_intrinsic_catalog_book_row_asset(asset)
            ),
            None,
        )
        if row_asset is not None:
            return row_asset

        selection = self.current_furniture_selection
        result = self.asset_manager.generate_assets(
            AssetGenerationRequest(
                object_descriptions=[
                    "upright encyclopedia book set row with tightly packed visible "
                    "leather-bound volumes and spines facing forward"
                ],
                short_names=["encyclopedia_book_row"],
                object_type=ObjectType.MANIPULAND,
                desired_dimensions=[[0.45, 0.12, 0.22]],
                style_context=str(getattr(selection, "style_notes", "") or ""),
            )
        )
        return next(
            (
                asset
                for asset in result.successful_assets
                if self._is_intrinsic_catalog_book_row_asset(asset)
            ),
            None,
        )

    @staticmethod
    def _internal_bookcase_surfaces(
        furniture: SceneObject,
    ) -> list[SupportSurface]:
        """Return authored support planes inside an upright bookcase shell."""

        if furniture.bbox_min is None or furniture.bbox_max is None:
            return []
        object_height = float(furniture.bbox_max[2] - furniture.bbox_min[2])
        if object_height <= 0.0:
            return []
        object_elevation = float(furniture.transform.translation()[2])
        bottom = object_elevation + float(furniture.bbox_min[2])
        top = object_elevation + float(furniture.bbox_max[2])
        edge_margin = max(0.10, 0.075 * object_height)
        return sorted(
            [
                surface
                for surface in furniture.support_surfaces
                if bottom + edge_margin
                < float(surface.transform.translation()[2])
                < top - edge_margin
            ],
            key=lambda surface: (
                float(surface.transform.translation()[2]),
                str(surface.surface_id),
            ),
        )

    @staticmethod
    def _dense_book_row_owner_by_surface(
        scene: RoomScene,
    ) -> dict[UniqueID, UniqueID]:
        return {
            surface.surface_id: obj.object_id
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            for surface in obj.support_surfaces
        }

    @staticmethod
    def _dense_bookcase_story_levels(
        bookcases: list[SceneObject],
        *,
        elevation_tolerance_m: float = 0.05,
    ) -> dict[UniqueID, float]:
        """Cluster bounded post-simulation drift into authored story levels."""

        positioned: list[tuple[float, SceneObject]] = []
        for bookcase in bookcases:
            try:
                positioned.append(
                    (float(bookcase.transform.translation()[2]), bookcase)
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
        clusters: list[list[tuple[float, SceneObject]]] = []
        for elevation, bookcase in sorted(
            positioned,
            key=lambda item: (item[0], str(item[1].object_id)),
        ):
            if (
                clusters
                and abs(elevation - clusters[-1][0][0]) <= elevation_tolerance_m
            ):
                clusters[-1].append((elevation, bookcase))
            else:
                clusters.append([(elevation, bookcase)])

        levels: dict[UniqueID, float] = {}
        for cluster in clusters:
            representative = round(
                sum(elevation for elevation, _ in cluster) / len(cluster),
                3,
            )
            levels.update(
                {bookcase.object_id: representative for _, bookcase in cluster}
            )
        return levels

    @staticmethod
    def _bind_dense_book_row_to_owner_surface(
        row: SceneObject,
        owner: SceneObject,
        surface: SupportSurface,
    ) -> None:
        """Persist the authored tier pose in its owning furniture frame."""

        row.metadata["dense_library_book_row"] = True
        row.metadata["dense_library_owner_bound"] = str(owner.object_id)
        row.metadata["dense_library_owner_surface_local_transform"] = (
            (owner.transform.inverse() @ surface.transform).GetAsMatrix4().tolist()
        )

    def _normalize_intrinsic_dense_book_rows(self) -> tuple[int, int]:
        """Bind exact catalog rows placed by any workflow to their bookcase tier."""

        normalized = str(getattr(self.scene, "text_description", "")).casefold()
        explicit_dense_library = (
            "library" in normalized
            and "large" in normalized
            and "thousand" in normalized
            and bool(re.search(r"\bmulti[ -]?level\b", normalized))
        )
        if not explicit_dense_library:
            return 0, 0

        owner_by_surface = self._dense_book_row_owner_by_surface(self.scene)
        surface_by_id = {
            surface.surface_id: surface
            for obj in self.scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            for surface in obj.support_surfaces
        }
        bound = 0
        discarded = 0
        for row in list(self.scene.objects.values()):
            if (
                not self._is_intrinsic_catalog_book_row_asset(row)
                or row.placement_info is None
            ):
                continue
            surface_id = row.placement_info.parent_surface_id
            owner_id = owner_by_surface.get(surface_id)
            owner = self.scene.get_object(owner_id) if owner_id is not None else None
            surface = surface_by_id.get(surface_id)
            if (
                owner is None
                or surface is None
                or owner.object_type != ObjectType.FURNITURE
                or not any(
                    term in f"{owner.name} {owner.description}".casefold()
                    for term in ("shelf", "bookcase")
                )
            ):
                continue
            effective_surface_transform = (
                self._dense_book_row_effective_surface_transform(
                    row,
                    owner,
                    surface,
                )
            )
            if not self._dense_book_row_is_contained(
                row,
                surface,
                surface_transform=effective_surface_transform,
            ):
                self.scene.remove_object(row.object_id)
                discarded += 1
                continue
            was_bound = bool((row.metadata or {}).get("dense_library_book_row"))
            self._bind_dense_book_row_to_owner_surface(row, owner, surface)
            if not was_bound:
                bound += 1
        return bound, discarded

    @staticmethod
    def _dense_book_row_effective_surface_transform(
        row: SceneObject,
        owner: SceneObject,
        surface: SupportSurface,
    ) -> RigidTransform:
        """Resolve a cached authored tier through its owner's current pose."""

        metadata = row.metadata or {}
        if metadata.get("dense_library_owner_bound") != str(owner.object_id):
            return surface.transform
        local_transform = metadata.get("dense_library_owner_surface_local_transform")
        if local_transform is None:
            return surface.transform
        try:
            matrix = np.asarray(local_transform, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                return surface.transform
            return owner.transform @ RigidTransform(matrix)
        except (RuntimeError, TypeError, ValueError):
            return surface.transform

    @staticmethod
    def _dense_book_row_is_contained(
        row: SceneObject,
        surface: SupportSurface,
        *,
        edge_clearance_m: float = 0.002,
        surface_transform: RigidTransform | None = None,
    ) -> bool:
        """Require the row's actual footprint to remain inside its authored tier."""

        if row.bbox_min is None or row.bbox_max is None:
            return False
        row_height = float(row.bbox_max[2] - row.bbox_min[2])
        surface_clearance = float(
            surface.bounding_box_max[2] - surface.bounding_box_min[2]
        )
        if row_height > surface_clearance + 1e-6:
            return False

        effective_surface_transform = surface_transform or surface.transform
        relative = effective_surface_transform.inverse() @ row.transform
        minimum = surface.bounding_box_min
        maximum = surface.bounding_box_max
        for x in (float(row.bbox_min[0]), float(row.bbox_max[0])):
            for y in (float(row.bbox_min[1]), float(row.bbox_max[1])):
                point = relative @ np.array([x, y, 0.0])
                point_2d = point[:2]
                if not (
                    minimum[0] + edge_clearance_m
                    <= point_2d[0]
                    <= maximum[0] - edge_clearance_m
                    and minimum[1] + edge_clearance_m
                    <= point_2d[1]
                    <= maximum[1] - edge_clearance_m
                ):
                    return False
                if not surface.contains_point_2d(point_2d):
                    return False
        return True

    @classmethod
    def _physically_invalid_dense_book_row_ids(
        cls,
        scene: RoomScene,
        cfg: DictConfig,
        *,
        row_ids: set[UniqueID] | None = None,
    ) -> set[UniqueID]:
        """Return tagged rows with collision beyond allowed owner support contact."""

        if row_ids is None:
            row_ids = {
                obj.object_id
                for obj in scene.objects.values()
                if obj.object_type == ObjectType.MANIPULAND
                and bool((obj.metadata or {}).get("dense_library_book_row"))
            }
        if not row_ids:
            return set()

        owner_by_surface = cls._dense_book_row_owner_by_surface(scene)
        surface_by_id = {
            surface.surface_id: surface
            for obj in scene.objects.values()
            if obj.object_type == ObjectType.FURNITURE
            for surface in obj.support_surfaces
        }
        invalid: set[UniqueID] = set()
        for row_id in row_ids:
            row = scene.get_object(row_id)
            parent_surface_id = (
                row.placement_info.parent_surface_id
                if row is not None and row.placement_info is not None
                else None
            )
            surface = surface_by_id.get(parent_surface_id)
            owner_id = owner_by_surface.get(parent_surface_id)
            owner = scene.get_object(owner_id) if owner_id is not None else None
            effective_surface_transform = (
                cls._dense_book_row_effective_surface_transform(row, owner, surface)
                if row is not None and owner is not None and surface is not None
                else None
            )
            if (
                row is None
                or owner is None
                or surface is None
                or not cls._dense_book_row_is_contained(
                    row,
                    surface,
                    surface_transform=effective_surface_transform,
                )
            ):
                invalid.add(row_id)

        physics_cfg = cfg.physics_validation
        collisions = compute_scene_collisions(
            scene=scene,
            penetration_threshold=physics_cfg.object_penetration_threshold_m,
            floor_penetration_tolerance=physics_cfg.floor_penetration_tolerance_m,
            current_furniture_id=None,
            manipuland_furniture_tolerance_m=0.0,
        )
        for collision in collisions:
            pair = {
                UniqueID(collision.object_a_id),
                UniqueID(collision.object_b_id),
            }
            colliding_rows = pair & row_ids
            for row_id in colliding_rows:
                row = scene.get_object(row_id)
                owner_id = (
                    owner_by_surface.get(row.placement_info.parent_surface_id)
                    if row is not None and row.placement_info is not None
                    else None
                )
                other_ids = pair - {row_id}
                is_owner_bound_contact = owner_id is not None and owner_id in other_ids
                if not is_owner_bound_contact:
                    invalid.add(row_id)
        return invalid
