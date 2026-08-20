"""Deterministic semantic room kits for reliable first-pass furnishing.

Room kits are deliberately asset-agnostic.  They describe roles, counts, scale,
and relationships, while the global asset catalog resolves each role to a local
mesh.  This keeps layout intent stable when catalog contents or LLM providers
change.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "sixteen": 16,
    "twenty": 20,
}


@dataclass(frozen=True)
class RoomKitSlot:
    """One semantic role in a room kit."""

    role: str
    query: str
    aliases: tuple[str, ...] = ()
    minimum_count: int = 1
    target_count: int = 1
    maximum_count: int = 1
    nominal_dimensions_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    placement_class: str = "floor"
    support_role: str | None = None
    facing_target: str | None = None
    required: bool = True
    notes: str = ""


@dataclass(frozen=True)
class RoomKitRelationship:
    """A deterministic relationship which placement must preserve."""

    subject_role: str
    relation: str
    target_role: str
    distance_m: tuple[float, float] | None = None
    required: bool = True


@dataclass(frozen=True)
class RoomKitDefinition:
    """Reusable semantic arrangement for a common room archetype."""

    kit_id: str
    label: str
    triggers: tuple[str, ...]
    slots: tuple[RoomKitSlot, ...]
    relationships: tuple[RoomKitRelationship, ...]
    source_policy: str = "catalog_only"
    circulation_width_m: float = 0.9


@dataclass(frozen=True)
class RoomKitSelection:
    """Prompt- and room-sized expansion of a kit."""

    kit_id: str
    label: str
    source_policy: str
    circulation_width_m: float
    slot_counts: dict[str, int]
    slots: tuple[RoomKitSlot, ...]
    relationships: tuple[RoomKitRelationship, ...]
    match_score: int
    matched_triggers: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_brief(self) -> str:
        """Render an exact, provider-neutral furnishing contract."""

        slot_lines = []
        for slot in self.slots:
            count = self.slot_counts[slot.role]
            facing = f"; face toward {slot.facing_target}" if slot.facing_target else ""
            support = f"; supported by {slot.support_role}" if slot.support_role else ""
            requirement = "required" if slot.required else "optional"
            slot_lines.append(
                f"- {slot.role}: exactly {count} {requirement}; catalog query "
                f"'{slot.query}'; nominal size {slot.nominal_dimensions_m}m; "
                f"placement={slot.placement_class}{support}{facing}. {slot.notes}"
            )
        relationship_lines = [
            "- "
            + relationship.subject_role
            + " "
            + relationship.relation.replace("_", " ")
            + " "
            + relationship.target_role
            + (
                f" at {relationship.distance_m[0]:.2f}-"
                f"{relationship.distance_m[1]:.2f}m"
                if relationship.distance_m
                else ""
            )
            for relationship in self.relationships
        ]
        return "\n".join(
            [
                f"Semantic room kit: {self.label} ({self.kit_id})",
                "Use local catalog assets only for ordinary furniture; do not "
                "generate replacement geometry when a compatible catalog item exists.",
                f"Preserve at least {self.circulation_width_m:.2f}m circulation.",
                "Required roles and counts:",
                *slot_lines,
                "Required relationships:",
                *relationship_lines,
                "Treat these counts, facing targets, support relationships, and "
                "functional pairings as hard constraints.",
            ]
        )


def _slot(
    role: str,
    query: str,
    *,
    aliases: tuple[str, ...] = (),
    counts: tuple[int, int, int] = (1, 1, 1),
    dimensions: tuple[float, float, float] = (1.0, 1.0, 1.0),
    placement: str = "floor",
    support: str | None = None,
    facing: str | None = None,
    required: bool = True,
    notes: str = "",
) -> RoomKitSlot:
    return RoomKitSlot(
        role=role,
        query=query,
        aliases=aliases,
        minimum_count=counts[0],
        target_count=counts[1],
        maximum_count=counts[2],
        nominal_dimensions_m=dimensions,
        placement_class=placement,
        support_role=support,
        facing_target=facing,
        required=required,
        notes=notes,
    )


BUILTIN_ROOM_KITS: tuple[RoomKitDefinition, ...] = (
    RoomKitDefinition(
        kit_id="library-reading-hall-v1",
        label="Library / reading hall",
        triggers=("library", "reading room", "reading hall", "bookshelves"),
        slots=(
            _slot(
                "bookshelf",
                "full-height library bookshelf",
                aliases=("bookcase", "shelving"),
                counts=(2, 4, 10),
                dimensions=(1.0, 0.35, 2.0),
                placement="wall",
                facing="room_center",
            ),
            _slot(
                "classical_statue",
                "classical marble statue on pedestal",
                aliases=("statue", "sculpture"),
                counts=(0, 0, 4),
                dimensions=(0.7, 0.7, 2.0),
                facing="room_center",
                required=False,
            ),
            _slot(
                "reading_table",
                "library reading table",
                aliases=("research table", "work table"),
                counts=(1, 2, 4),
                dimensions=(2.0, 0.9, 0.75),
                facing="primary_aisle",
            ),
            _slot(
                "reading_chair",
                "comfortable library reading chair",
                aliases=("chair", "seat"),
                counts=(4, 8, 20),
                dimensions=(0.55, 0.55, 0.9),
                facing="reading_table",
            ),
            _slot(
                "task_lamp",
                "library task lamp",
                counts=(1, 2, 4),
                dimensions=(0.3, 0.3, 0.55),
                placement="surface",
                support="reading_table",
                required=False,
            ),
        ),
        relationships=(
            RoomKitRelationship("reading_chair", "faces", "reading_table"),
            RoomKitRelationship(
                "reading_chair", "paired_around", "reading_table", (0.15, 0.75)
            ),
            RoomKitRelationship("bookshelf", "faces", "room_center"),
        ),
    ),
    RoomKitDefinition(
        kit_id="bar-lounge-v1",
        label="Bar / lounge",
        triggers=("bar", "cocktail lounge", "pub", "tavern", "nightclub"),
        slots=(
            _slot(
                "bar_counter",
                "commercial bar counter",
                counts=(1, 1, 2),
                dimensions=(2.8, 0.75, 1.1),
                facing="bar_stool",
            ),
            _slot(
                "bar_stool",
                "bar stool",
                aliases=("stool",),
                counts=(3, 5, 10),
                dimensions=(0.45, 0.45, 1.0),
                facing="bar_counter",
            ),
            _slot(
                "back_bar_storage",
                "back bar bottle shelving cabinet",
                counts=(1, 2, 4),
                dimensions=(1.2, 0.4, 2.0),
                placement="wall",
                facing="bar_counter",
            ),
            _slot(
                "lounge_seat",
                "upholstered lounge chair",
                counts=(0, 2, 6),
                dimensions=(0.8, 0.8, 0.85),
                facing="lounge_table",
                required=False,
            ),
            _slot(
                "lounge_table",
                "small cocktail table",
                counts=(0, 1, 3),
                dimensions=(0.65, 0.65, 0.55),
                required=False,
            ),
        ),
        relationships=(
            RoomKitRelationship("bar_stool", "faces", "bar_counter"),
            RoomKitRelationship(
                "bar_stool", "evenly_spaced_along", "bar_counter", (0.20, 0.65)
            ),
            RoomKitRelationship(
                "back_bar_storage", "behind", "bar_counter", (0.7, 1.4)
            ),
            RoomKitRelationship("lounge_seat", "faces", "lounge_table", required=False),
        ),
    ),
    RoomKitDefinition(
        kit_id="dining-room-v1",
        label="Dining room / restaurant",
        triggers=("dining room", "restaurant", "cafe", "cafeteria", "dining"),
        slots=(
            _slot(
                "dining_table",
                "dining table",
                counts=(1, 2, 6),
                dimensions=(1.5, 0.85, 0.75),
            ),
            _slot(
                "dining_chair",
                "dining chair",
                aliases=("chair", "seat"),
                counts=(4, 8, 24),
                dimensions=(0.5, 0.5, 0.9),
                facing="dining_table",
            ),
            _slot(
                "sideboard",
                "dining sideboard credenza",
                counts=(0, 1, 2),
                dimensions=(1.5, 0.45, 0.85),
                placement="wall",
                facing="room_center",
                required=False,
            ),
        ),
        relationships=(
            RoomKitRelationship("dining_chair", "faces", "dining_table"),
            RoomKitRelationship(
                "dining_chair", "paired_around", "dining_table", (0.10, 0.65)
            ),
        ),
    ),
    RoomKitDefinition(
        kit_id="medical-exam-room-v1",
        label="Medical examination / treatment room",
        triggers=(
            "medical",
            "clinic",
            "exam room",
            "treatment room",
            "hospital room",
            "med bay",
            "medbay",
        ),
        slots=(
            _slot(
                "patient_bed",
                "medical examination bed",
                aliases=("hospital bed", "gurney", "exam table"),
                dimensions=(2.0, 0.85, 0.8),
                facing="clinical_work_zone",
            ),
            _slot(
                "medical_device",
                "freestanding medical monitoring device",
                aliases=("patient monitor", "medical equipment"),
                counts=(1, 1, 2),
                dimensions=(0.65, 0.55, 1.4),
                facing="patient_bed",
                notes="Must stand beside the bed, never on top of it.",
            ),
            _slot(
                "clinician_stool",
                "clinical rolling stool",
                counts=(1, 1, 2),
                dimensions=(0.5, 0.5, 0.65),
                facing="patient_bed",
            ),
            _slot(
                "medical_storage",
                "medical supply cabinet",
                counts=(1, 1, 2),
                dimensions=(0.9, 0.45, 1.8),
                placement="wall",
                facing="room_center",
            ),
        ),
        relationships=(
            RoomKitRelationship(
                "medical_device", "beside_not_on", "patient_bed", (0.25, 1.0)
            ),
            RoomKitRelationship("clinician_stool", "faces", "patient_bed"),
            RoomKitRelationship("patient_bed", "clear_on_both_sides", "primary_aisle"),
        ),
        circulation_width_m=1.1,
    ),
    RoomKitDefinition(
        kit_id="creative-studio-v1",
        label="Creative / broadcast studio",
        triggers=(
            "studio",
            "radio studio",
            "broadcast studio",
            "recording studio",
            "podcast studio",
            "architect's studio",
            "architect studio",
        ),
        slots=(
            _slot(
                "studio_desk",
                "professional studio work desk",
                aliases=("broadcast desk", "workstation"),
                dimensions=(1.8, 0.85, 0.75),
                facing="focal_wall",
            ),
            _slot(
                "studio_chair",
                "ergonomic studio office chair",
                aliases=("office chair",),
                counts=(1, 2, 4),
                dimensions=(0.65, 0.65, 1.1),
                facing="studio_desk",
            ),
            _slot(
                "equipment_rack",
                "professional studio equipment rack",
                counts=(1, 1, 2),
                dimensions=(0.6, 0.7, 1.5),
                placement="wall",
                facing="studio_desk",
            ),
            _slot(
                "guest_seat",
                "studio guest lounge chair",
                counts=(0, 2, 4),
                dimensions=(0.75, 0.75, 0.85),
                facing="studio_desk",
                required=False,
            ),
        ),
        relationships=(
            RoomKitRelationship("studio_chair", "faces", "studio_desk"),
            RoomKitRelationship(
                "equipment_rack", "reachable_from", "studio_desk", (0.5, 1.8)
            ),
            RoomKitRelationship("guest_seat", "faces", "studio_desk", required=False),
        ),
    ),
)


def _explicit_count(prompt: str, slot: RoomKitSlot) -> int | None:
    alternatives = (slot.role.replace("_", " "), slot.query, *slot.aliases)
    prompt_folded = prompt.casefold()
    for alternative in alternatives:
        noun = re.escape(alternative.casefold())
        match = re.search(
            rf"\b(\d+|{'|'.join(_NUMBER_WORDS)})\s+{noun}s?\b", prompt_folded
        )
        if not match:
            continue
        token = match.group(1)
        return int(token) if token.isdigit() else _NUMBER_WORDS[token]
    return None


def select_room_kit(
    prompt: str,
    *,
    room_area_m2: float,
    registry: tuple[RoomKitDefinition, ...] = BUILTIN_ROOM_KITS,
) -> RoomKitSelection | None:
    """Select and deterministically size the strongest matching room kit."""

    folded = prompt.casefold()
    ranked: list[tuple[int, int, RoomKitDefinition, tuple[str, ...]]] = []
    for index, kit in enumerate(registry):
        matches = tuple(trigger for trigger in kit.triggers if trigger in folded)
        score = sum(len(trigger.split()) * 10 + len(trigger) for trigger in matches)
        ranked.append((score, -index, kit, matches))
    score, _, kit, matches = max(ranked, key=lambda item: (item[0], item[1]))
    if score <= 0:
        return None

    selected_slots = kit.slots
    if kit.kit_id == "library-reading-hall-v1":
        collection_scale = bool(
            re.search(r"\b(?:hundreds|thousands) of books\b", folded)
        )
        renaissance = bool(re.search(r"\brena+i+s+ance\b", folded))
        statues_requested = bool(re.search(r"\b(?:statues?|sculptures?)\b", folded))
        if collection_scale or statues_requested:
            rewritten_slots: list[RoomKitSlot] = []
            for slot in selected_slots:
                if collection_scale and slot.role == "bookshelf":
                    query = "full-height library bookcase densely filled with visible books"
                    if renaissance:
                        query = "full-height Renaissance library bookcase densely filled with visible books"
                    slot = replace(
                        slot,
                        query=query,
                        minimum_count=12,
                        target_count=14,
                        maximum_count=20,
                        notes="Fill gallery walls across every level with visible books.",
                    )
                elif collection_scale and slot.role == "reading_table":
                    slot = replace(
                        slot,
                        minimum_count=4,
                        target_count=4,
                        maximum_count=6,
                    )
                elif collection_scale and slot.role == "reading_chair":
                    slot = replace(
                        slot,
                        minimum_count=12,
                        target_count=12,
                        maximum_count=20,
                    )
                elif statues_requested and slot.role == "classical_statue":
                    slot = replace(
                        slot,
                        query=(
                            "classical Renaissance marble statue on pedestal"
                            if renaissance
                            else slot.query
                        ),
                        minimum_count=2,
                        target_count=2,
                        required=True,
                        notes="Use as repeated gallery focal points on separate levels.",
                    )
                rewritten_slots.append(slot)
            selected_slots = tuple(rewritten_slots)

    # Scale target density at two coarse thresholds; explicit prompt counts win.
    density_delta = -1 if room_area_m2 < 12.0 else 1 if room_area_m2 >= 32.0 else 0
    counts: dict[str, int] = {}
    for slot in selected_slots:
        explicit = _explicit_count(prompt, slot)
        requested = (
            explicit if explicit is not None else slot.target_count + density_delta
        )
        counts[slot.role] = max(slot.minimum_count, min(slot.maximum_count, requested))
    return RoomKitSelection(
        kit_id=kit.kit_id,
        label=kit.label,
        source_policy=kit.source_policy,
        circulation_width_m=kit.circulation_width_m,
        slot_counts=counts,
        slots=selected_slots,
        relationships=kit.relationships,
        match_score=score,
        matched_triggers=matches,
    )


def persist_room_kit(selection: RoomKitSelection, output_path: Path) -> None:
    """Atomically persist the exact kit contract used for a scene."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n"
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
