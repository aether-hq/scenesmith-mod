from pathlib import Path
from types import SimpleNamespace

import pytest

from scenesmith.agent_utils.scene_blueprint import (
    BlueprintConstraint,
    FurnitureGroupBlueprint,
    LevelBlueprint,
    SceneBlueprint,
    SpaceBlueprint,
)
from scenesmith.agent_utils.semantic_group_materializer import (
    materialize_locked_semantic_groups,
)


class _Scene:
    def __init__(self, *, length: float = 200.0, width: float = 100.0) -> None:
        self.room_geometry = SimpleNamespace(length=length, width=width)
        self.objects = {}

    def add_object(self, obj) -> None:
        self.objects[obj.object_id] = obj

    def generate_unique_id(self, name: str):
        raise AssertionError(f"fixture generated an unexpected duplicate for {name}")


def _blueprint(*, dock: bool = True) -> SceneBlueprint:
    level = LevelBlueprint(
        level_id="level-main",
        name="Main",
        elevation_m=0.0,
        clear_height_m=40.0,
    )
    space = SpaceBlueprint(
        space_id="space-main",
        name="Main",
        room_type="hangar" if dock else "library",
        level_id=level.level_id,
        dimensions_m=(200.0, 100.0),
    )
    if not dock:
        group = FurnitureGroupBlueprint(
            group_id="group-reading",
            name="Reading furniture",
            space_id=space.space_id,
            roles={"reading_table": 3, "reading_chair": 1},
        )
        return SceneBlueprint(
            blueprint_id="small-room-kit",
            source_prompt="A library with reading tables",
            levels=(level,),
            spaces=(space,),
            furniture_groups=(group,),
            locked_ids=(group.group_id,),
        )

    fighter = FurnitureGroupBlueprint(
        group_id="group-fighter",
        name="Hero Space Fighter",
        space_id=space.space_id,
        roles={"hero_space_fighter": 1},
        density="sparse",
    )
    bays = FurnitureGroupBlueprint(
        group_id="group-bays",
        name="Repair Bay Ring",
        space_id=space.space_id,
        roles={"repair_bay": 10},
        focal_target=fighter.group_id,
        density="layered",
    )
    equipment = FurnitureGroupBlueprint(
        group_id="group-equipment",
        name="Bay Workshop Equipment",
        space_id=space.space_id,
        roles={
            "repair_parts_rack": 20,
            "repair_machine": 10,
            "misc_equipment_prop": 15,
        },
        focal_target=bays.group_id,
        density="layered",
    )
    constraints = (
        BlueprintConstraint(
            constraint_id="constraint-fighter",
            kind="semantic_hero_object",
            target_ids=(fighter.group_id,),
            parameters={
                "role_key": "hero_space_fighter",
                "preferred_dimensions_m": [20.0, 6.0, 14.0],
            },
            source="user",
        ),
        BlueprintConstraint(
            constraint_id="constraint-bays",
            kind="semantic_repeated_zone",
            target_ids=(bays.group_id,),
            parameters={
                "role_key": "repair_bay",
                "preferred_dimensions_m": [25.0, 15.0, 20.0],
            },
            source="user",
        ),
        BlueprintConstraint(
            constraint_id="constraint-parts",
            kind="semantic_object_group",
            target_ids=(equipment.group_id,),
            parameters={
                "role_key": "repair_parts_rack",
                "relationships": [{"predicate": "distributed_across", "target": "bay"}],
            },
            source="user",
        ),
        BlueprintConstraint(
            constraint_id="constraint-machines",
            kind="semantic_object_group",
            target_ids=(equipment.group_id,),
            parameters={
                "role_key": "repair_machine",
                "relationships": [{"predicate": "distributed_across", "target": "bay"}],
            },
            source="user",
        ),
        BlueprintConstraint(
            constraint_id="constraint-misc",
            kind="semantic_object_group",
            target_ids=(equipment.group_id,),
            parameters={
                "role_key": "misc_equipment_prop",
                "relationships": [{"predicate": "distributed_across", "target": "bay"}],
            },
            source="user",
        ),
    )
    return SceneBlueprint(
        blueprint_id="large-dock",
        source_prompt="A huge space-fighter and ten repair bays",
        levels=(level,),
        spaces=(space,),
        furniture_groups=(fighter, bays, equipment),
        constraints=constraints,
        locked_ids=tuple(
            [group.group_id for group in (fighter, bays, equipment)]
            + [constraint.constraint_id for constraint in constraints]
        ),
    )


def test_large_locked_groups_become_exact_surviving_geometry(tmp_path: Path):
    scene = _Scene()

    created = materialize_locked_semantic_groups(
        scene,
        _blueprint(),
        tmp_path,
    )

    assert len(created) == 56
    assert len(scene.objects) == 56
    by_role = {}
    for obj in scene.objects.values():
        by_role.setdefault(obj.metadata["role"], []).append(obj)
        assert obj.immutable
        assert obj.sdf_path.is_file()
        assert obj.metadata["generated_from"] == "locked_scene_blueprint"
    assert {role: len(objects) for role, objects in by_role.items()} == {
        "hero_space_fighter": 1,
        "repair_bay": 10,
        "repair_parts_rack": 20,
        "repair_machine": 10,
        "misc_equipment_prop": 15,
    }
    fighter = by_role["hero_space_fighter"][0]
    assert tuple(fighter.bbox_max - fighter.bbox_min) == (20.0, 6.0, 14.0)
    assert tuple(fighter.transform.translation()) == (0.0, 0.0, 0.0)
    bays = by_role["repair_bay"]
    assert all(
        tuple(item.bbox_max - item.bbox_min) == (25.0, 15.0, 20.0) for item in bays
    )
    assert len({round(float(item.transform.translation()[0]), 3) for item in bays}) == 5
    assert {round(float(item.transform.translation()[1]), 3) for item in bays} == {
        -42.0,
        42.0,
    }

    assert materialize_locked_semantic_groups(scene, _blueprint(), tmp_path) == ()
    assert len(scene.objects) == 56


def test_small_room_kit_groups_remain_owned_by_the_furniture_pipeline(tmp_path: Path):
    scene = _Scene(length=20.0, width=14.0)

    assert (
        materialize_locked_semantic_groups(scene, _blueprint(dock=False), tmp_path)
        == ()
    )
    assert scene.objects == {}


def test_object_budget_fails_without_reducing_counts_or_dimensions(tmp_path: Path):
    with pytest.raises(RuntimeError, match="explicit construction budget of 55"):
        materialize_locked_semantic_groups(
            _Scene(),
            _blueprint(),
            tmp_path,
            max_objects=55,
        )


def test_repeated_zones_fail_instead_of_shrinking_into_a_small_room(tmp_path: Path):
    with pytest.raises(RuntimeError, match="will not silently clamp dimensions"):
        materialize_locked_semantic_groups(
            _Scene(length=80.0, width=40.0),
            _blueprint(),
            tmp_path,
        )
