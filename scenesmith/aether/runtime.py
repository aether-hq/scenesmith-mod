"""Concrete SceneSmith runtime for accepted contextual-completion operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .assets import acquire_completion_assets
from .room_runtime import annotate_room_scene_instances
from .scene_census import CensusError


class DeterministicPlacementAdapter(Protocol):
    """Operation-specific, non-LLM candidate solver and SceneSmith placement adapter."""

    def place(
        self,
        scene: Any,
        asset: Any,
        operation: dict[str, Any],
        asset_brief: dict[str, Any],
        *,
        instance_index: int,
        round_index: int,
    ) -> str | None:
        """Return the raw id of one accepted placement, or None after bounded rejection."""


class SceneSmithCompletionRuntime:
    """Route typed assets and execute only adapter-approved SceneSmith placements.

    Asset acquisition is concrete and uses SceneSmith's validated HSSD, Objaverse,
    ArtVIP, or SAM3D routing. Geometry transforms remain in operation-specific
    deterministic adapters so neither this runtime nor an LLM guesses coordinates.
    """

    def __init__(
        self,
        *,
        scene: Any,
        asset_managers: dict[str, Any],
        placement_adapters: dict[str, DeterministicPlacementAdapter],
        evidence_provider: Callable[[], dict[str, Any]],
        style_context: str,
        asset_acquirer: Callable[..., Any] = acquire_completion_assets,
    ) -> None:
        self.scene = scene
        self.asset_managers = asset_managers
        self.placement_adapters = placement_adapters
        self.evidence_provider = evidence_provider
        self.style_context = style_context
        self.asset_acquirer = asset_acquirer

    def place_asset_brief(
        self,
        operation: dict[str, Any],
        asset_brief: dict[str, Any],
        *,
        round_index: int,
    ) -> tuple[str, ...]:
        kind = str(operation["operation"])
        asset_manager = self.asset_managers.get(kind)
        adapter = self.placement_adapters.get(kind)
        if asset_manager is None or adapter is None:
            raise CensusError(
                f"SceneSmith completion operation {kind} is not configured"
            )
        result = self.asset_acquirer(
            asset_manager,
            operation,
            asset_brief,
            style_context=self.style_context,
            scene_id=self.scene.scene_dir.name,
        )
        if result.has_failures:
            messages = "; ".join(item.error_message for item in result.failed_assets)
            raise CensusError(
                f"asset routing failed for {asset_brief['variant_id']}: {messages}"
            )
        if len(result.successful_assets) != 1:
            raise CensusError(
                f"asset routing for {asset_brief['variant_id']} returned "
                f"{len(result.successful_assets)} templates; expected exactly one"
            )
        asset = result.successful_assets[0]
        placed: list[str] = []
        for index in range(int(asset_brief["instance_count"])):
            raw_id = adapter.place(
                self.scene,
                asset,
                operation,
                asset_brief,
                instance_index=index,
                round_index=round_index,
            )
            if raw_id is not None:
                placed.append(str(raw_id))
        return tuple(placed)

    def scene_state(self) -> dict[str, Any]:
        return self.scene.to_state_dict()

    def restore_scene_state(self, state: dict[str, Any]) -> None:
        self.scene.restore_from_state_dict(state)

    def validation_evidence(self) -> dict[str, Any]:
        return self.evidence_provider()

    def annotate_instances(
        self,
        raw_ids: tuple[str, ...],
        operation: dict[str, Any],
        asset_brief: dict[str, Any],
        *,
        round_index: int,
    ) -> None:
        annotate_room_scene_instances(
            self.scene,
            raw_ids,
            operation,
            asset_brief,
            round_index=round_index,
        )
