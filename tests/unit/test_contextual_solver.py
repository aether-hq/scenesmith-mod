"""Tests for deterministic contextual zones and topology guards."""

import math

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.contextual_solver import (
    OrientedZone,
    ZoneKind,
    evaluate_spatial_entities,
    scene_object_to_spatial_entity,
    solve_candidate_poses,
    validate_blueprint_topology,
    validate_hosted_object,
    zones_intersect,
)
from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenesmith.agent_utils.scene_blueprint import blueprint_from_prompt


def _object(
    object_id: str,
    name: str,
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    yaw_degrees: float = 0,
    context: str = "active",
) -> SceneObject:
    return SceneObject(
        object_id=UniqueID(object_id),
        object_type=ObjectType.FURNITURE,
        name=name,
        description=name,
        transform=RigidTransform(RollPitchYaw(0, 0, math.radians(yaw_degrees)), center),
        bbox_min=-np.asarray(dimensions) / 2,
        bbox_max=np.asarray(dimensions) / 2,
        metadata={"placement_context": context},
    )


def test_oriented_zone_sat_distinguishes_separated_and_overlapping():
    first = OrientedZone(
        "a", "a", ZoneKind.OCCUPANCY, (0, 0), (1, 0.25), math.radians(45)
    )
    overlap = OrientedZone("b", "b", ZoneKind.OCCUPANCY, (0.5, 0), (0.5, 0.5))
    separate = OrientedZone("c", "c", ZoneKind.OCCUPANCY, (4, 0), (0.5, 0.5))

    assert zones_intersect(first, overlap)
    assert not zones_intersect(first, separate)


def test_active_chair_must_face_nearby_table_but_stacked_chair_does_not():
    table = scene_object_to_spatial_entity(
        _object("table_0", "dining table", (0, 0, 0.375), (1.5, 0.8, 0.75))
    )
    wrong_chair = scene_object_to_spatial_entity(
        _object("chair_0", "dining chair", (0, -1.1, 0.45), (0.5, 0.5, 0.9), 180)
    )
    stacked_chair = scene_object_to_spatial_entity(
        _object(
            "chair_1",
            "dining chair",
            (2, -1.1, 0.45),
            (0.5, 0.5, 0.9),
            180,
            "stacked",
        )
    )
    assert table is not None and wrong_chair is not None and stacked_chair is not None

    active_result = evaluate_spatial_entities([table, wrong_chair])
    stacked_result = evaluate_spatial_entities([table, stacked_chair])

    assert any(
        violation.code == "seat_faces_away_from_table"
        for violation in active_result.violations
    )
    assert not any(
        violation.code == "seat_faces_away_from_table"
        for violation in stacked_result.violations
    )


def test_storage_access_zone_is_hard_while_seat_access_is_advisory():
    cabinet = scene_object_to_spatial_entity(
        _object("cabinet_0", "storage cabinet", (0, 0, 0.9), (1, 0.5, 1.8))
    )
    blocker = scene_object_to_spatial_entity(
        _object("box_0", "ottoman", (0, 0.7, 0.25), (0.8, 0.8, 0.5))
    )
    assert cabinet is not None and blocker is not None

    result = evaluate_spatial_entities([cabinet, blocker])

    assert not result.valid
    assert any(
        violation.code == "access_zone_blocked" and violation.severity == "hard"
        for violation in result.violations
    )


def test_wall_art_must_fit_and_align_with_host():
    result = validate_hosted_object(
        object_id="art_0",
        kind="wall_art",
        center_xyz=(0, 0, 1.5),
        dimensions_xyz=(3, 0.1, 2),
        host_bounds_min=(-1, -0.1, 0),
        host_bounds_max=(1, 0.1, 3),
        normal_alignment=0.2,
    )

    assert not result.valid
    assert {violation.code for violation in result.violations} == {
        "host_bounds_exceeded",
        "host_orientation_mismatch",
    }


def test_blueprint_connectors_have_valid_landings_and_headroom():
    blueprint = blueprint_from_prompt("A two-level library with spiral stairs")

    result = validate_blueprint_topology(blueprint)

    assert result.valid


def test_bounded_solver_is_deterministic_and_stops_early():
    existing = [_object("table_0", "dining table", (0, 0, 0.375), (1.5, 0.8, 0.75))]

    def factory(pose):
        return _object(
            "chair_0",
            "dining chair",
            (pose[0], pose[1], 0.45),
            (0.5, 0.5, 0.9),
            pose[2],
        )

    poses = ((0, 0, 0), (0, -1, 0), (0, -1, 180))
    first = solve_candidate_poses(factory, existing, poses, timeout_ms=100)
    second = solve_candidate_poses(factory, existing, poses, timeout_ms=100)

    assert first.valid
    assert first.selected_pose == second.selected_pose
    assert first.elapsed_ms < 100
