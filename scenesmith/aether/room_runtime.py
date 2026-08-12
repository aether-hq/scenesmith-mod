"""Shared semantic provenance behavior for concrete SceneSmith placement runtimes."""

from __future__ import annotations

from typing import Any

from .scene_census import CensusError, normalize_instance_id


def annotate_room_scene_instances(
    scene,
    raw_ids: tuple[str, ...],
    operation: dict[str, Any],
    asset_brief: dict[str, Any],
    *,
    round_index: int,
) -> None:
    """Stamp the actual RoomScene objects before their next serialized checkpoint."""
    by_string_id = {str(object_id): item for object_id, item in scene.objects.items()}
    for raw_id in raw_ids:
        item = by_string_id.get(raw_id)
        if item is None:
            raise CensusError(f"placement runtime reported absent RoomScene object {raw_id}")
        item.metadata.update(
            {
                "aether_instance_id": normalize_instance_id(raw_id),
                "aether_role_id": operation["role_id"],
                "aether_object_class": "scenic-object",
                "aether_functional_zone_ids": list(operation["functional_zone_ids"]),
                "aether_completion_operation_id": operation["operation_id"],
                "aether_completion_variant_id": asset_brief["variant_id"],
                "aether_completion_round": round_index,
            }
        )
