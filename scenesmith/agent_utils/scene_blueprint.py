"""Provider-neutral, versioned scene intent used before geometry construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CURRENT_SCHEMA_VERSION = 1


def stable_blueprint_id(kind: str, label: str, ordinal: int = 0) -> str:
    """Create a readable ID stable across providers and repeated builds."""

    slug = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-") or kind
    digest = hashlib.sha1(
        f"{kind}:{slug}:{ordinal}".encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:8]
    return f"{kind}-{slug[:32]}-{digest}"


class BlueprintModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BlueprintConstraint(BlueprintModel):
    constraint_id: str
    kind: str
    target_ids: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    strength: Literal["hard", "soft"] = "hard"
    source: Literal["user", "inferred", "system"] = "inferred"


class LevelBlueprint(BlueprintModel):
    level_id: str
    name: str
    elevation_m: float = 0.0
    clear_height_m: float = 3.0


class SpaceBlueprint(BlueprintModel):
    space_id: str
    name: str
    room_type: str
    level_id: str
    dimensions_m: tuple[float, float]
    covered: bool = True
    prompt: str = ""


class OpeningBlueprint(BlueprintModel):
    opening_id: str
    kind: Literal["door", "window", "open_connection"]
    host_space_id: str
    connects_to_space_id: str | None = None
    width_m: float = 0.9
    height_m: float = 2.1


class ConnectorEndpoint(BlueprintModel):
    space_id: str
    level_id: str
    position_m: tuple[float, float, float]


class ConnectorBlueprint(BlueprintModel):
    connector_id: str
    kind: Literal[
        "stairs_straight",
        "stairs_l",
        "stairs_u",
        "stairs_spiral",
        "ramp",
        "ladder",
        "elevator",
    ]
    start: ConnectorEndpoint
    end: ConnectorEndpoint
    width_m: float = 1.0
    parameters: dict[str, Any] = Field(default_factory=dict)


class FurnitureGroupBlueprint(BlueprintModel):
    group_id: str
    name: str
    space_id: str
    roles: dict[str, int]
    focal_target: str | None = None
    density: Literal["sparse", "balanced", "layered"] = "balanced"


class BlueprintDesignTokens(BlueprintModel):
    style_keywords: tuple[str, ...] = ()
    palette: tuple[str, ...] = ()
    material_roles: dict[str, str] = Field(default_factory=dict)
    lighting_mood: str = ""
    focal_hierarchy: tuple[str, ...] = ()


class SceneBlueprint(BlueprintModel):
    """Canonical scene intent consumed by all construction stages."""

    schema_version: Literal[1] = CURRENT_SCHEMA_VERSION
    blueprint_id: str
    source_prompt: str
    mode: Literal["room", "house"] = "room"
    levels: tuple[LevelBlueprint, ...]
    spaces: tuple[SpaceBlueprint, ...]
    openings: tuple[OpeningBlueprint, ...] = ()
    connectors: tuple[ConnectorBlueprint, ...] = ()
    furniture_groups: tuple[FurnitureGroupBlueprint, ...] = ()
    design_tokens: BlueprintDesignTokens = Field(default_factory=BlueprintDesignTokens)
    constraints: tuple[BlueprintConstraint, ...] = ()
    locked_ids: tuple[str, ...] = ()
    repair_log: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references_and_topology(self) -> "SceneBlueprint":
        ids = [self.blueprint_id]
        ids.extend(level.level_id for level in self.levels)
        ids.extend(space.space_id for space in self.spaces)
        ids.extend(opening.opening_id for opening in self.openings)
        ids.extend(connector.connector_id for connector in self.connectors)
        ids.extend(group.group_id for group in self.furniture_groups)
        ids.extend(constraint.constraint_id for constraint in self.constraints)
        if len(ids) != len(set(ids)):
            raise ValueError("blueprint IDs must be globally unique")

        level_ids = {level.level_id for level in self.levels}
        space_ids = {space.space_id for space in self.spaces}
        if not self.levels or not self.spaces:
            raise ValueError("a blueprint requires at least one level and one space")
        for space in self.spaces:
            if space.level_id not in level_ids:
                raise ValueError(f"space {space.space_id} references an unknown level")
            if min(space.dimensions_m) <= 0:
                raise ValueError(f"space {space.space_id} has non-positive dimensions")
        for opening in self.openings:
            if opening.host_space_id not in space_ids:
                raise ValueError(f"opening {opening.opening_id} has an unknown host")
            if (
                opening.connects_to_space_id is not None
                and opening.connects_to_space_id not in space_ids
            ):
                raise ValueError(
                    f"opening {opening.opening_id} has an unknown destination"
                )
        connected_level_ids: set[str] = set()
        for connector in self.connectors:
            for endpoint in (connector.start, connector.end):
                if endpoint.level_id not in level_ids:
                    raise ValueError(
                        f"connector {connector.connector_id} references unknown level"
                    )
                if endpoint.space_id not in space_ids:
                    raise ValueError(
                        f"connector {connector.connector_id} references unknown space"
                    )
                connected_level_ids.add(endpoint.level_id)
            if connector.start.level_id == connector.end.level_id:
                raise ValueError(
                    f"connector {connector.connector_id} does not connect two levels"
                )
        ground_level = min(self.levels, key=lambda level: level.elevation_m).level_id
        for level in self.levels:
            if (
                level.level_id != ground_level
                and level.level_id not in connected_level_ids
            ):
                raise ValueError(f"elevated level {level.level_id} is unreachable")
        for group in self.furniture_groups:
            if group.space_id not in space_ids:
                raise ValueError(f"furniture group {group.group_id} has unknown space")
            if any(count < 0 for count in group.roles.values()):
                raise ValueError(f"furniture group {group.group_id} has negative count")
        valid_ids = set(ids)
        for constraint in self.constraints:
            unknown = set(constraint.target_ids) - valid_ids
            if unknown:
                raise ValueError(
                    f"constraint {constraint.constraint_id} targets unknown IDs: "
                    + ", ".join(sorted(unknown))
                )
        if set(self.locked_ids) - valid_ids:
            raise ValueError("locked_ids contains an unknown blueprint ID")
        return self

    def to_prompt_brief(self) -> str:
        """Serialize the contract for a model without vendor-specific syntax."""

        payload = self.model_dump(mode="json", exclude={"repair_log"})
        return (
            f"Original request: {self.source_prompt}\n\n"
            "Canonical SceneBlueprint v1 (preserve IDs and hard constraints):\n"
            + json.dumps(payload, separators=(",", ":"), sort_keys=True)
        )


class BlueprintDiff(BlueprintModel):
    changed_paths: tuple[str, ...]
    invalidated_stages: tuple[
        Literal["floor_plan", "furniture", "walls", "ceiling", "details", "render"],
        ...,
    ]


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [dict(item, name=item.get("name", key)) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def migrate_blueprint_payload(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Migrate older provider payloads to the current external field names."""

    payload = dict(raw)
    repairs: list[str] = []
    version = int(payload.get("schema_version", 0) or 0)
    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported future SceneBlueprint version {version}")
    if version == 0:
        renames = {
            "rooms": "spaces",
            "stories": "levels",
            "storeys": "levels",
            "stairs": "connectors",
            "style": "design_tokens",
            "furniture": "furniture_groups",
        }
        for old, new in renames.items():
            if old in payload and new not in payload:
                payload[new] = payload.pop(old)
                repairs.append(f"renamed {old} to {new}")
        payload["schema_version"] = CURRENT_SCHEMA_VERSION
        repairs.append("migrated schema_version 0 to 1")
    return payload, repairs


def _prompt_level_count(prompt: str) -> int:
    folded = prompt.casefold()
    number_words = {"two": 2, "three": 3, "four": 4}
    explicit = re.search(
        r"\b(\d+|two|three|four)\s*[- ]?(?:level|storey|story|floor)s?\b",
        folded,
    )
    if explicit:
        token = explicit.group(1)
        count = int(token) if token.isdigit() else number_words[token]
        return max(1, min(4, count))
    if any(term in folded for term in ("multi-level", "multilevel", "mezzanine")):
        return 2
    return 1


def _connector_kind(prompt: str) -> str:
    folded = prompt.casefold()
    if "spiral" in folded:
        return "stairs_spiral"
    if "u-shaped" in folded or "u shaped" in folded:
        return "stairs_u"
    if "l-shaped" in folded or "l shaped" in folded:
        return "stairs_l"
    if "ramp" in folded:
        return "ramp"
    if "ladder" in folded:
        return "ladder"
    return "stairs_straight"


def blueprint_from_prompt(
    prompt: str,
    *,
    mode: Literal["room", "house"] = "room",
    default_dimensions_m: tuple[float, float] = (7.0, 7.0),
) -> SceneBlueprint:
    """Create a structurally valid deterministic blueprint from plain language."""

    blueprint_id = stable_blueprint_id("scene", prompt)
    level_count = _prompt_level_count(prompt)
    clear_height = max(2.4, min(4.0, 9.0 / level_count if level_count > 1 else 3.2))
    levels = tuple(
        LevelBlueprint(
            level_id=stable_blueprint_id("level", f"level {index}", index),
            name=f"Level {index + 1}",
            elevation_m=index * clear_height,
            clear_height_m=clear_height,
        )
        for index in range(level_count)
    )
    room_name = "Main room" if mode == "room" else "Primary space"
    spaces = tuple(
        SpaceBlueprint(
            space_id=stable_blueprint_id("space", f"{room_name} {index}", index),
            name=room_name if index == 0 else f"{room_name} upper {index}",
            room_type="room",
            level_id=level.level_id,
            dimensions_m=default_dimensions_m,
            prompt=prompt,
        )
        for index, level in enumerate(levels)
    )
    connectors: list[ConnectorBlueprint] = []
    for index in range(1, level_count):
        lower = levels[index - 1]
        upper = levels[index]
        lower_space = spaces[index - 1]
        upper_space = spaces[index]
        connectors.append(
            ConnectorBlueprint(
                connector_id=stable_blueprint_id(
                    "connector", f"levels {index-1}-{index}"
                ),
                kind=_connector_kind(prompt),
                start=ConnectorEndpoint(
                    space_id=lower_space.space_id,
                    level_id=lower.level_id,
                    position_m=(-2.0, 0.0, lower.elevation_m),
                ),
                end=ConnectorEndpoint(
                    space_id=upper_space.space_id,
                    level_id=upper.level_id,
                    position_m=(-2.0, 0.0, upper.elevation_m),
                ),
                parameters=(
                    {"direction": "ccw"} if "spiral" in prompt.casefold() else {}
                ),
            )
        )
    constraints = (
        BlueprintConstraint(
            constraint_id=stable_blueprint_id("constraint", "walkable circulation"),
            kind="minimum_circulation_width",
            target_ids=tuple(space.space_id for space in spaces),
            parameters={"width_m": 0.9},
            strength="hard",
            source="system",
        ),
    )
    return SceneBlueprint(
        blueprint_id=blueprint_id,
        source_prompt=prompt,
        mode=mode,
        levels=levels,
        spaces=spaces,
        connectors=tuple(connectors),
        constraints=constraints,
    )


def normalize_scene_blueprint(
    raw: Mapping[str, Any] | str | None,
    *,
    prompt: str,
    mode: Literal["room", "house"] = "room",
) -> SceneBlueprint:
    """Repair common provider envelopes and aliases into SceneBlueprint v1."""

    source = _as_mapping(raw)
    repairs: list[str] = []
    for envelope in ("arguments", "arguments_json", "blueprint", "scene", "plan"):
        nested = _as_mapping(source.get(envelope))
        if nested:
            source = {**nested, **{k: v for k, v in source.items() if k != envelope}}
            repairs.append(f"unwrapped {envelope} envelope")
            break
    source, migration_repairs = migrate_blueprint_payload(source)
    repairs.extend(migration_repairs)
    fallback = blueprint_from_prompt(prompt, mode=mode)

    raw_levels = _as_sequence(source.get("levels"))
    levels: list[LevelBlueprint] = []
    for index, item in enumerate(raw_levels):
        data = _as_mapping(item)
        name = str(data.get("name") or data.get("label") or f"Level {index + 1}")
        levels.append(
            LevelBlueprint(
                level_id=str(
                    data.get("level_id")
                    or data.get("id")
                    or stable_blueprint_id("level", name, index)
                ),
                name=name,
                elevation_m=float(
                    data.get("elevation_m", data.get("elevation", index * 3.0))
                ),
                clear_height_m=float(
                    data.get("clear_height_m", data.get("height", 3.0))
                ),
            )
        )
    if not levels:
        levels = list(fallback.levels)
        repairs.append("synthesized missing levels")

    raw_spaces = _as_sequence(source.get("spaces"))
    spaces: list[SpaceBlueprint] = []
    for index, item in enumerate(raw_spaces):
        data = _as_mapping(item)
        name = str(data.get("name") or data.get("type") or f"Room {index + 1}")
        dimensions = data.get("dimensions_m", data.get("dimensions", (7.0, 7.0)))
        if isinstance(dimensions, Mapping):
            dimensions = (
                dimensions.get("length", dimensions.get("x", 7.0)),
                dimensions.get("width", dimensions.get("y", 7.0)),
            )
        dimensions_list = (
            list(dimensions) if isinstance(dimensions, (list, tuple)) else [7, 7]
        )
        level_ref = str(
            data.get("level_id")
            or data.get("level")
            or levels[min(index, len(levels) - 1)].level_id
        )
        if level_ref not in {level.level_id for level in levels}:
            # Providers commonly emit numeric level indices.
            try:
                level_ref = levels[int(level_ref)].level_id
                repairs.append(f"resolved numeric level reference for {name}")
            except (ValueError, IndexError):
                level_ref = levels[0].level_id
                repairs.append(f"repaired unknown level reference for {name}")
        spaces.append(
            SpaceBlueprint(
                space_id=str(
                    data.get("space_id")
                    or data.get("id")
                    or stable_blueprint_id("space", name, index)
                ),
                name=name,
                room_type=str(data.get("room_type") or data.get("type") or "room"),
                level_id=level_ref,
                dimensions_m=(float(dimensions_list[0]), float(dimensions_list[1])),
                covered=bool(data.get("covered", data.get("has_roof", True))),
                prompt=str(data.get("prompt") or data.get("description") or prompt),
            )
        )
    if not spaces:
        spaces = list(fallback.spaces)
        # Align fallback spaces to any provider-authored levels.
        spaces = [
            space.model_copy(
                update={"level_id": levels[min(i, len(levels) - 1)].level_id}
            )
            for i, space in enumerate(spaces[: len(levels)])
        ]
        repairs.append("synthesized missing spaces")

    # Use deterministic connectors whenever provider connector topology is missing
    # or malformed. This is preferable to accepting decorative stairs to nowhere.
    connectors: list[ConnectorBlueprint] = []
    raw_connectors = _as_sequence(source.get("connectors"))
    space_by_level = {space.level_id: space for space in spaces}
    for index in range(1, len(levels)):
        lower, upper = levels[index - 1], levels[index]
        raw_connector = _as_mapping(
            raw_connectors[index - 1] if index - 1 < len(raw_connectors) else {}
        )
        lower_space = space_by_level.get(lower.level_id, spaces[0])
        upper_space = space_by_level.get(upper.level_id, spaces[-1])
        kind = str(
            raw_connector.get("kind")
            or raw_connector.get("type")
            or _connector_kind(prompt)
        )
        allowed = ConnectorBlueprint.model_fields["kind"].annotation.__args__
        if kind not in allowed:
            kind = _connector_kind(prompt)
            repairs.append(f"repaired unsupported connector kind at index {index - 1}")
        connectors.append(
            ConnectorBlueprint(
                connector_id=str(
                    raw_connector.get("connector_id")
                    or raw_connector.get("id")
                    or stable_blueprint_id("connector", f"levels {index-1}-{index}")
                ),
                kind=kind,
                start=ConnectorEndpoint(
                    space_id=lower_space.space_id,
                    level_id=lower.level_id,
                    position_m=(-2.0, 0.0, lower.elevation_m),
                ),
                end=ConnectorEndpoint(
                    space_id=upper_space.space_id,
                    level_id=upper.level_id,
                    position_m=(-2.0, 0.0, upper.elevation_m),
                ),
                width_m=float(
                    raw_connector.get("width_m", raw_connector.get("width", 1.0))
                ),
                parameters=_as_mapping(raw_connector.get("parameters")),
            )
        )

    blueprint = SceneBlueprint(
        blueprint_id=str(
            source.get("blueprint_id") or source.get("id") or fallback.blueprint_id
        ),
        source_prompt=str(
            source.get("source_prompt") or source.get("prompt") or prompt
        ),
        mode=mode,
        levels=tuple(levels),
        spaces=tuple(spaces),
        connectors=tuple(connectors),
        design_tokens=BlueprintDesignTokens.model_validate(
            _as_mapping(source.get("design_tokens"))
        ),
        repair_log=tuple(repairs),
    )
    return blueprint


def diff_blueprints(before: SceneBlueprint, after: SceneBlueprint) -> BlueprintDiff:
    """Return changed paths and the minimal construction stages to invalidate."""

    before_data = before.model_dump(mode="json", exclude={"repair_log"})
    after_data = after.model_dump(mode="json", exclude={"repair_log"})
    paths: list[str] = []
    for key in before_data.keys() | after_data.keys():
        if before_data.get(key) != after_data.get(key):
            paths.append(key)
    stages: set[str] = {"render"} if paths else set()
    if set(paths) & {"levels", "spaces", "openings", "connectors"}:
        stages.update({"floor_plan", "furniture", "walls", "ceiling", "details"})
    if "furniture_groups" in paths:
        stages.update({"furniture", "details"})
    if "design_tokens" in paths:
        stages.update({"furniture", "walls", "ceiling", "details"})
    if "constraints" in paths:
        stages.update({"floor_plan", "furniture", "walls", "ceiling", "details"})
    order = ("floor_plan", "furniture", "walls", "ceiling", "details", "render")
    return BlueprintDiff(
        changed_paths=tuple(sorted(paths)),
        invalidated_stages=tuple(stage for stage in order if stage in stages),
    )


def floor_plan_submission_from_blueprint(blueprint: SceneBlueprint) -> dict[str, Any]:
    """Project canonical intent into the floor-plan tool's provider schema."""

    spaces = blueprint.spaces[:1] if blueprint.mode == "room" else blueprint.spaces
    room_specs = [
        {
            "id": space.space_id,
            "type": space.room_type,
            "width": space.dimensions_m[0],
            "depth": space.dimensions_m[1],
            "prompt": space.prompt or blueprint.source_prompt,
        }
        for space in spaces
    ]
    clear_height = max(level.clear_height_m for level in blueprint.levels)
    payload: dict[str, Any] = {
        "room_specs": room_specs,
        "wall_height_meters": min(12.0, clear_height * len(blueprint.levels)),
    }
    if blueprint.design_tokens.material_roles:
        roles = blueprint.design_tokens.material_roles
        payload["materials"] = {
            "floor": roles.get("floor", "warm wood floor"),
            "walls": roles.get("walls", "neutral plaster wall"),
            "exterior": roles.get("exterior", roles.get("walls", "neutral plaster")),
        }
    return payload


def persist_scene_blueprint(blueprint: SceneBlueprint, output_path: Path) -> None:
    """Atomically persist canonical scene intent for resume and revisions."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = blueprint.model_dump_json(indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
