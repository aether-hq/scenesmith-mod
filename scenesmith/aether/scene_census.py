"""Translate a real serialized SceneSmith scene into Aether census evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_ARCHITECTURE_TYPES = {"wall", "floor"}
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CensusError(RuntimeError):
    """Scene state cannot truthfully produce the required census."""


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _slug(value: str, *, prefix: str = "item") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or prefix
    if len(normalized) <= 80 and _SLUG_PATTERN.fullmatch(normalized):
        return normalized
    suffix = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{normalized[:69].rstrip('-')}-{suffix}"


def _instance_id(raw_id: str) -> str:
    if len(raw_id) <= 80 and _SLUG_PATTERN.fullmatch(raw_id):
        return raw_id
    return _slug(raw_id, prefix="scene-object")


def normalize_instance_id(raw_id: str) -> str:
    """Expose the worker's stable SceneSmith-to-Aether instance-id mapping."""
    return _instance_id(raw_id)


def _scene_instance_id(item: dict[str, Any], raw_id: str) -> str:
    """Prefer the immutable Aether id over SceneSmith's internal object key."""
    explicit = (item.get("metadata") or {}).get("aether_instance_id")
    if isinstance(explicit, str) and explicit:
        return _instance_id(explicit)
    return _instance_id(raw_id)


def _evidence_instance_id(
    value: object,
    objects: dict[str, dict[str, Any]],
) -> str:
    """Resolve measured raw ids through the same semantic-id mapping as census rows."""
    raw_id = str(value)
    item = objects.get(raw_id)
    if item is not None:
        return _scene_instance_id(item, raw_id)
    normalized = _instance_id(raw_id)
    for candidate_raw_id, candidate in objects.items():
        semantic_id = _scene_instance_id(candidate, candidate_raw_id)
        if normalized in {_instance_id(candidate_raw_id), semantic_id}:
            return semantic_id
    return normalized


def _asset_id(item: dict[str, Any], scene_root: Path | None) -> str:
    metadata = item.get("metadata") or {}
    explicit = metadata.get("aether_asset_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    source = str(metadata.get("asset_source") or "unknown")
    source_keys = (
        "hssd_mesh_id",
        "objaverse_mesh_id",
        "articulated_id",
    )
    for key in source_keys:
        identifier = metadata.get(key)
        if identifier is not None and str(identifier):
            return f"{source}:{identifier}"
    geometry = item.get("geometry_path")
    if geometry:
        path = Path(geometry)
        if scene_root is not None and not path.is_absolute():
            path = scene_root / path
        if path.is_file():
            return f"geometry-sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    raise CensusError(
        f"scene object {item.get('object_id')} has no content-addressed asset identity"
    )


def _semantic_role(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    role = metadata.get("aether_role_id")
    if isinstance(role, str) and role:
        return _slug(role, prefix="scene-detail")
    # SceneSmith names are semantic (``bottle``, ``dining chair``, ...).  Keep
    # that vocabulary intact so a pre-completion census can satisfy an Aether
    # role target even before an Aether repair operation has stamped explicit
    # provenance.  Prefixing every fallback as ``scene-detail-*`` made real
    # bottles invisible to a ``bottle`` target and manufactured deficits.
    return _slug(str(item.get("name") or item.get("object_type")), prefix="scene-detail")


def _object_class(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    explicit = metadata.get("aether_object_class")
    if explicit in {"scenic-object", "person"}:
        return explicit
    return "person" if str(item.get("object_type")).lower() in {"person", "human"} else "scenic-object"


def _zone_ids(item: dict[str, Any]) -> tuple[str, ...]:
    raw = (item.get("metadata") or {}).get("aether_functional_zone_ids", ())
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (list, tuple)):
        raise CensusError(
            f"scene object {item.get('object_id')} has malformed functional-zone metadata"
        )
    return tuple(dict.fromkeys(_slug(str(value), prefix="zone") for value in raw))


def _support_parent_by_surface(objects: dict[str, dict[str, Any]]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for raw_id, item in objects.items():
        for surface in item.get("support_surfaces") or ():
            surface_id = surface.get("surface_id")
            if surface_id:
                parents[str(surface_id)] = _scene_instance_id(item, raw_id)
    return parents


def _require_validation_evidence(
    stage_input: dict[str, Any], evidence: dict[str, Any]
) -> None:
    expected = stage_input.get("locked_architecture_sha256")
    if evidence.get("architecture_sha256") != expected:
        raise CensusError("physical validation did not preserve the locked architecture digest")
    baseline_geometry = evidence.get("baseline_room_geometry_sha256")
    current_geometry = evidence.get("current_room_geometry_sha256")
    if not baseline_geometry or not current_geometry:
        raise CensusError("physical validation omitted room-geometry digests")
    if baseline_geometry != current_geometry:
        raise CensusError("completion mutated the compiled SceneSmith room geometry")
    required_keys = {
        "clear_circulation_route_ids",
        "clear_story_position_ids",
        "collision_instance_ids",
        "supported_instance_ids",
        "pbr_complete_instance_ids",
        "visible_view_ids_by_instance",
    }
    missing = sorted(required_keys - evidence.keys())
    if missing:
        raise CensusError(f"physical validation evidence is incomplete: {missing}")


def build_scene_census(
    stage_input: dict[str, Any],
    scene_state: dict[str, Any],
    validation_evidence: dict[str, Any],
    *,
    round_index: int,
    scene_root: Path | None = None,
) -> dict[str, Any]:
    """Build the exact Genesis SceneCensus JSON shape from measured scene evidence."""
    _require_validation_evidence(stage_input, validation_evidence)
    raw_objects = scene_state.get("objects")
    if not isinstance(raw_objects, dict):
        raise CensusError("SceneSmith state has no serialized object dictionary")
    support_parents = _support_parent_by_surface(raw_objects)
    supported = {
        _evidence_instance_id(value, raw_objects)
        for value in validation_evidence["supported_instance_ids"]
    }
    pbr_complete = {
        _evidence_instance_id(value, raw_objects)
        for value in validation_evidence["pbr_complete_instance_ids"]
    }
    visible = validation_evidence["visible_view_ids_by_instance"]
    if not isinstance(visible, dict):
        raise CensusError("visible_view_ids_by_instance must be an object map")
    objects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_id, item in sorted(raw_objects.items()):
        if item.get("object_type") in _ARCHITECTURE_TYPES:
            continue
        instance_id = _scene_instance_id(item, str(raw_id))
        if instance_id in seen_ids:
            raise CensusError(f"normalizing SceneSmith ids produced duplicate {instance_id}")
        seen_ids.add(instance_id)
        placement = item.get("placement_info") or {}
        surface_id = placement.get("parent_surface_id")
        parent_id = support_parents.get(str(surface_id)) if surface_id else None
        view_ids = visible.get(raw_id, visible.get(str(raw_id), visible.get(instance_id, ())))
        if not isinstance(view_ids, (list, tuple)):
            raise CensusError(f"visibility evidence for {raw_id} must be a list")
        objects.append(
            {
                "instance_id": instance_id,
                "role_id": _semantic_role(item),
                "object_class": _object_class(item),
                "asset_id": _asset_id(item, scene_root),
                "functional_zone_ids": list(_zone_ids(item)),
                "supported_by_instance_id": parent_id,
                "supported": instance_id in supported,
                "pbr_complete": instance_id in pbr_complete,
                "visible_view_ids": [
                    _slug(str(view_id), prefix="inspection-view") for view_id in view_ids
                ],
                "locked": bool(item.get("immutable") or (item.get("metadata") or {}).get("aether_locked")),
            }
        )
    return {
        "contract_version": 1,
        "job_id": stage_input["job_id"],
        "round_index": round_index,
        "architecture_sha256": validation_evidence["architecture_sha256"],
        "objects": objects,
        "clear_circulation_route_ids": validation_evidence["clear_circulation_route_ids"],
        "clear_story_position_ids": validation_evidence["clear_story_position_ids"],
        "collision_instance_ids": [
            _evidence_instance_id(value, raw_objects)
            for value in validation_evidence["collision_instance_ids"]
        ],
    }
