"""Physical feasibility post-processing for scene collision resolution.

This module provides two-stage post-processing adapted from scene_gen repository:
1. Projection - IK-based collision resolution with configurable DOF constraints
2. Simulation - Physics settling to static equilibrium (always full 6DOF)

See: https://github.com/nepfaff/steerable-scene-generation/blob/main/steerable_scene_generation/algorithms/scene_diffusion/postprocessing.py
"""

import logging

from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.physics.feasibility import (
    postprocessing as _postprocessing,
    projection as _projection,
)
from scenesmith.agent_utils.physics.feasibility.ik import _create_drake_plant_for_ik
from scenesmith.agent_utils.physics.feasibility.postprocessing import (
    apply_per_furniture_postprocessing as _apply_per_furniture_postprocessing,
    apply_physical_feasibility_postprocessing as _apply_physical_feasibility_postprocessing,
)
from scenesmith.agent_utils.physics.feasibility.projection import (
    _apply_floor_penetration_fallback,
    _get_colliding_object_ids,
    apply_non_penetration_projection as _apply_non_penetration_projection,
)
from scenesmith.agent_utils.physics.feasibility.simulation import (
    apply_forward_simulation,
)


def apply_non_penetration_projection(*args, **kwargs):
    """Compatibility facade preserving patchable IK projection dependencies."""

    _projection._create_drake_plant_for_ik = _create_drake_plant_for_ik
    _projection._get_colliding_object_ids = _get_colliding_object_ids
    return _apply_non_penetration_projection(*args, **kwargs)


def apply_physical_feasibility_postprocessing(*args, **kwargs):
    """Run postprocessing while preserving the historic patch surface."""

    _postprocessing.apply_non_penetration_projection = apply_non_penetration_projection
    _postprocessing._apply_floor_penetration_fallback = (
        _apply_floor_penetration_fallback
    )
    _postprocessing.apply_forward_simulation = apply_forward_simulation
    _postprocessing.compute_scene_collisions = compute_scene_collisions
    return _apply_physical_feasibility_postprocessing(*args, **kwargs)


def apply_per_furniture_postprocessing(*args, **kwargs):
    """Run per-furniture processing through the patch-compatible facade."""

    _postprocessing.apply_physical_feasibility_postprocessing = (
        apply_physical_feasibility_postprocessing
    )
    return _apply_per_furniture_postprocessing(*args, **kwargs)
