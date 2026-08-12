"""Deterministic execution boundary for Aether-authored completion patches."""

from __future__ import annotations

import copy

from typing import Any, Protocol

from .scene_census import (
    CensusError,
    build_scene_census,
    canonical_digest,
    normalize_instance_id,
)


class CompletionPlacementRuntime(Protocol):
    """Linux-CUDA SceneSmith runtime owned by the deterministic placement worker."""

    def place_asset_brief(
        self,
        operation: dict[str, Any],
        asset_brief: dict[str, Any],
        *,
        round_index: int,
    ) -> tuple[str, ...]:
        """Acquire routed assets, place valid instances, and return their raw scene ids."""

    def scene_state(self) -> dict[str, Any]:
        """Return RoomScene.to_state_dict() after all accepted placements."""

    def restore_scene_state(self, state: dict[str, Any]) -> None:
        """Roll back every placement when a patch fails its post-placement gates."""

    def validation_evidence(self) -> dict[str, Any]:
        """Return measured collision, support, route, story, PBR, and view evidence."""

    def annotate_instances(
        self,
        raw_ids: tuple[str, ...],
        operation: dict[str, Any],
        asset_brief: dict[str, Any],
        *,
        round_index: int,
    ) -> None:
        """Persist semantic provenance on the actual RoomScene instances."""


def execute_completion_patch(
    stage_input: dict[str, Any],
    patch: dict[str, Any],
    input_census: dict[str, Any],
    runtime: CompletionPlacementRuntime,
    *,
    scene_root=None,
) -> dict[str, Any]:
    """Execute one validated add-only patch, then remeasure the resulting scene."""
    if patch.get("job_id") != stage_input.get("job_id") or patch.get("job_id") != input_census.get(
        "job_id"
    ):
        raise CensusError("completion patch identifies a different immutable job")
    if patch.get("round_index") != input_census.get("round_index"):
        raise CensusError("completion patch round does not match the input census")
    if patch.get("base_census_sha256") != canonical_digest(input_census):
        raise CensusError("completion patch was authored against a stale census")
    checkpoint = copy.deepcopy(runtime.scene_state())
    try:
        return _execute_validated_patch(
            stage_input,
            patch,
            input_census,
            runtime,
            scene_root=scene_root,
        )
    except Exception:
        runtime.restore_scene_state(checkpoint)
        raise


def _execute_validated_patch(
    stage_input: dict[str, Any],
    patch: dict[str, Any],
    input_census: dict[str, Any],
    runtime: CompletionPlacementRuntime,
    *,
    scene_root=None,
) -> dict[str, Any]:
    requested = sum(int(operation["count"]) for operation in patch["operations"])
    placed_ids: list[str] = []
    diagnostics: list[str] = []
    for operation in patch["operations"]:
        for asset_brief in operation["asset_briefs"]:
            raw_ids = runtime.place_asset_brief(
                operation,
                asset_brief,
                round_index=int(patch["round_index"]),
            )
            if len(raw_ids) > int(asset_brief["instance_count"]):
                raise CensusError(
                    f"placement runtime overfilled asset brief {asset_brief['variant_id']}"
                )
            placed_ids.extend(raw_ids)
            runtime.annotate_instances(
                raw_ids,
                operation,
                asset_brief,
                round_index=int(patch["round_index"]),
            )
            diagnostics.append(
                f"{asset_brief['variant_id']}: placed {len(raw_ids)}/"
                f"{asset_brief['instance_count']}"
            )
    if len(placed_ids) != len(set(placed_ids)):
        raise CensusError("placement runtime returned a scene instance more than once")
    next_census = build_scene_census(
        stage_input,
        runtime.scene_state(),
        runtime.validation_evidence(),
        round_index=int(input_census["round_index"]) + 1,
        scene_root=scene_root,
    )
    placed = len(next_census["objects"]) - len(input_census["objects"])
    if placed < 0:
        raise CensusError("completion patch removed existing scene objects")
    if placed != len(placed_ids):
        raise CensusError("placement runtime changed objects outside the accepted patch")
    previous_objects = {item["instance_id"]: item for item in input_census["objects"]}
    next_objects = {item["instance_id"]: item for item in next_census["objects"]}
    changed = sorted(
        instance_id
        for instance_id, item in previous_objects.items()
        if next_objects.get(instance_id) != item
    )
    if changed:
        raise CensusError(f"placement runtime mutated or removed existing instances: {changed}")
    expected_new_ids = {normalize_instance_id(raw_id) for raw_id in placed_ids}
    measured_new_ids = set(next_objects) - set(previous_objects)
    if measured_new_ids != expected_new_ids:
        raise CensusError(
            "placement runtime did not link returned ids to the measured new scene instances"
        )
    unsupported = sorted(
        instance_id for instance_id in measured_new_ids if not next_objects[instance_id]["supported"]
    )
    if unsupported:
        raise CensusError(f"completion placed unsupported scene instances: {unsupported}")
    return {
        "patch_sha256": canonical_digest(patch),
        "requested_count": requested,
        "placed_count": placed,
        "rejected_count": requested - placed,
        "next_census": next_census,
        "diagnostics": diagnostics,
    }
