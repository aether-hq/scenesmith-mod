"""Tests for deterministic semantic room-kit selection and expansion."""

import json

from scenesmith.agent_utils.design.room_kits import (
    BUILTIN_ROOM_KITS,
    persist_room_kit,
    select_room_kit,
)


def test_builtin_registry_covers_five_high_value_room_types():
    assert {kit.kit_id for kit in BUILTIN_ROOM_KITS} == {
        "library-reading-hall-v1",
        "bar-lounge-v1",
        "dining-room-v1",
        "medical-exam-room-v1",
        "creative-studio-v1",
    }
    assert all(kit.source_policy == "catalog_only" for kit in BUILTIN_ROOM_KITS)


def test_library_kit_expands_counts_and_preserves_chair_facing():
    selection = select_room_kit(
        "A large library with four reading tables and sixteen chairs",
        room_area_m2=64.0,
    )

    assert selection is not None
    assert selection.kit_id == "library-reading-hall-v1"
    assert selection.slot_counts["reading_table"] == 4
    assert selection.slot_counts["reading_chair"] == 16
    chair = next(slot for slot in selection.slots if slot.role == "reading_chair")
    assert chair.facing_target == "reading_table"
    assert "exactly 16 required" in selection.to_prompt_brief()


def test_collection_scale_library_requires_dense_shelves_and_statues():
    prompt = (
        "a large, multi-level library with thousands of books and a bunch of tables "
        "and chairs for patrons. A spiral staircase connects the floors, and there "
        "are huge archted windows, statues, and so on, as it has a renaiissance , "
        "gorgeous decor."
    )

    selection = select_room_kit(prompt, room_area_m2=190.44)

    assert selection is not None
    assert selection.slot_counts["bookshelf"] >= 12
    bookshelf = next(slot for slot in selection.slots if slot.role == "bookshelf")
    assert "renaissance" in bookshelf.query.casefold()
    assert "filled" in bookshelf.query.casefold()
    statue = next(slot for slot in selection.slots if slot.role == "classical_statue")
    assert statue.required
    assert selection.slot_counts["classical_statue"] >= 2
    assert "human figure" in statue.query.casefold()
    assert statue.nominal_dimensions_m == (1.4, 1.4, 1.8)
    gothic_extents = (1.477464, 1.739898, 1.563758)
    target_extents = (
        statue.nominal_dimensions_m[0],
        statue.nominal_dimensions_m[2],
        statue.nominal_dimensions_m[1],
    )
    uniform_scale = min(
        target / extent for target, extent in zip(target_extents, gothic_extents)
    )
    assert gothic_extents[1] * uniform_scale >= 0.6 * statue.nominal_dimensions_m[2]
    chair = next(slot for slot in selection.slots if slot.role == "reading_chair")
    assert "stationary" in chair.query.casefold()


def test_medical_kit_keeps_equipment_beside_bed():
    selection = select_room_kit(
        "A compact medical exam room with a hospital bed and patient monitor",
        room_area_m2=10.0,
    )

    assert selection is not None
    assert selection.kit_id == "medical-exam-room-v1"
    assert any(
        relationship.subject_role == "medical_device"
        and relationship.relation == "beside_not_on"
        and relationship.target_role == "patient_bed"
        for relationship in selection.relationships
    )
    assert "never on top" in selection.to_prompt_brief()


def test_room_area_scaling_is_bounded_and_deterministic():
    small = select_room_kit("A dining room", room_area_m2=8.0)
    large = select_room_kit("A dining room", room_area_m2=40.0)

    assert small is not None and large is not None
    assert small.slot_counts["dining_table"] == 1
    assert large.slot_counts["dining_table"] == 3
    assert select_room_kit("A dining room", room_area_m2=40.0) == large


def test_no_match_does_not_force_an_unrelated_kit():
    assert select_room_kit("An empty moonlit atrium", room_area_m2=20.0) is None


def test_room_kit_persistence_round_trips(tmp_path):
    selection = select_room_kit("A warm cocktail lounge", room_area_m2=24.0)
    assert selection is not None
    output = tmp_path / "room_kit.json"

    persist_room_kit(selection, output)

    payload = json.loads(output.read_text())
    assert payload["kit_id"] == "bar-lounge-v1"
    assert payload["slot_counts"]["bar_stool"] == 5
