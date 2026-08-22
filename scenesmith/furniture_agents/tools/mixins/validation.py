import json
import logging
import math
import os

import numpy as np

from pydrake.all import RollPitchYaw, RotationMatrix

from scenesmith.agent_utils.physics.physics_validation import compute_scene_collisions
from scenesmith.agent_utils.scene.contextual_solver import (
    validate_scene_object_placement,
)
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.furniture_agents.tools.response_dataclasses import (
    FurnitureErrorType,
    FurnitureOperationResult,
)

console_logger = logging.getLogger(__name__)


class FurnitureValidationMixin:
    """Spatial-envelope, collision, contextual-zone, and loop validation."""

    def _check_floor_bounds(
        self,
        x: float,
        y: float,
        reference_z: float | None = None,
    ) -> tuple[bool, str]:
        """Check if position (center point) is within floor plan bounds.

        Args:
            x: X coordinate in meters.
            y: Y coordinate in meters.
            reference_z: Requested support elevation for stacked floor plans.

        Returns:
            (is_valid, error_message) - error_message is empty string if valid.
        """
        room_geometry = self.scene.room_geometry

        surface_index = self._get_structural_surface_index()
        if surface_index is not None:
            pose = surface_index.support_pose(x, y, reference_z=reference_z)
            if pose is None:
                return (
                    False,
                    f"Position ({x:.3f}, {y:.3f}) is not on a traversable "
                    "structural support surface (it may be outside the room, "
                    "inside a hole/shaft, or on an overly steep patch).",
                )
            return True, ""

        # Floor bounds: [-length/2, length/2] × [-width/2, width/2].
        min_x = -room_geometry.length / 2
        max_x = room_geometry.length / 2
        min_y = -room_geometry.width / 2
        max_y = room_geometry.width / 2

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            error_msg = (
                f"Position ({x:.3f}, {y:.3f}) is out of floor plan bounds. "
                f"Valid bounds: X=[{min_x:.3f}, {max_x:.3f}], "
                f"Y=[{min_y:.3f}, {max_y:.3f}]"
            )
            return False, error_msg

        return True, ""

    def _get_structural_surface_index(self):
        """Lazily load compiled surface patches for irregular geometry."""

        if self._structural_surface_index is False:
            return None
        if self._structural_surface_index is not None:
            return self._structural_surface_index
        room_geometry = self.scene.room_geometry
        additional = getattr(room_geometry, "additional_structural_surface_paths", ())
        if not isinstance(additional, (list, tuple, set, frozenset)):
            additional = ()
        candidates = (
            getattr(room_geometry, "structural_surface_path", None),
            *additional,
        )
        sidecar_paths = [
            path
            for candidate in candidates
            if isinstance(candidate, (str, os.PathLike))
            and (path := os.fspath(candidate))
            and os.path.isfile(path)
        ]
        if not sidecar_paths:
            self._structural_surface_index = False
            return None
        from scenesmith.agent_utils.structure.structural_surfaces import (
            StructuralSurfaceIndex,
            load_surface_patches,
        )

        self._structural_surface_index = StructuralSurfaceIndex(
            patch
            for sidecar_path in sidecar_paths
            for patch in load_surface_patches(sidecar_path)
        )
        return self._structural_surface_index

    def _major_support_elevations(self) -> tuple[float, ...]:
        """Return story-floor elevations, excluding small landings and stair treads."""

        surface_index = self._get_structural_surface_index()
        if surface_index is None:
            return (0.0,)

        from scenesmith.agent_utils.structure.geometry_models.surface_models import (
            SurfaceRole,
        )

        candidates: list[tuple[float, float]] = []
        for query in surface_index.by_role(SurfaceRole.SUPPORT):
            if float(query.normal[2]) < math.cos(math.radians(5.0)):
                continue
            boundary = query.patch.boundary
            if len(boundary) < 3:
                continue
            area = (
                abs(
                    sum(
                        float(start[0]) * float(end[1])
                        - float(end[0]) * float(start[1])
                        for start, end in zip(boundary, boundary[1:] + boundary[:1])
                    )
                )
                / 2.0
            )
            if area <= 0.0:
                continue
            elevation = sum(float(point[2]) for point in boundary) / len(boundary)
            candidates.append((area, elevation))

        if not candidates:
            return (0.0,)
        minimum_story_area = max(area for area, _ in candidates) * 0.25
        elevations: list[float] = []
        for area, elevation in sorted(candidates, key=lambda item: item[1]):
            if area < minimum_story_area:
                continue
            if not any(abs(elevation - existing) <= 0.05 for existing in elevations):
                elevations.append(elevation)
        return tuple(elevations) or (0.0,)

    def _surface_aligned_pose(
        self,
        x: float,
        y: float,
        yaw_degrees: float,
        reference_z: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Return z/roll/pitch/yaw aligned to the support surface, if present."""

        surface_index = self._get_structural_surface_index()
        if surface_index is None:
            return None
        pose = surface_index.support_pose(
            x,
            y,
            reference_z=reference_z,
            yaw=math.radians(yaw_degrees),
        )
        if pose is None:
            return None
        rotation = RotationMatrix(
            np.column_stack((pose.tangent_x, pose.tangent_y, pose.normal))
        )
        rpy = RollPitchYaw(rotation)
        return (
            pose.position[2],
            math.degrees(rpy.roll_angle()),
            math.degrees(rpy.pitch_angle()),
            math.degrees(rpy.yaw_angle()),
        )

    def _validate_spatial_envelope(self, scene_object: SceneObject) -> tuple[bool, str]:
        """Require an object's complete AABB to fit its support and enclosure.

        Structural overhead patches are authoritative when present: a missing
        overhead patch means that XY location is open air. Legacy rectangular
        rooms use ``has_overhead_cover`` and ``wall_height``.
        """
        try:
            bounds = scene_object.compute_world_bounds()
        except (TypeError, ValueError, IndexError):
            # Compatibility for legacy/test registry entries whose bbox fields
            # predate concrete numeric bounds.
            return True, ""
        if (
            bounds is None
            or not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or not all(isinstance(item, np.ndarray) for item in bounds)
        ):
            # Older hand-authored test assets may not carry bounds. Production
            # catalog/generated assets do, and are validated here.
            return True, ""
        world_min, world_max = bounds
        center_x = float((world_min[0] + world_max[0]) / 2.0)
        center_y = float((world_min[1] + world_max[1]) / 2.0)
        footprint_samples = (
            (center_x, center_y),
            (float(world_min[0]), float(world_min[1])),
            (float(world_min[0]), float(world_max[1])),
            (float(world_max[0]), float(world_min[1])),
            (float(world_max[0]), float(world_max[1])),
        )
        epsilon = 1e-4
        surface_index = self._get_structural_surface_index()
        if surface_index is not None:
            reference_z = float(world_min[2]) + 0.10
            for sample_x, sample_y in footprint_samples:
                support = surface_index.support_pose(
                    sample_x,
                    sample_y,
                    reference_z=reference_z,
                    max_drop=0.20,
                )
                if support is None:
                    return (
                        False,
                        "Furniture footprint extends beyond its traversable support "
                        f"near ({sample_x:.3f}, {sample_y:.3f}).",
                    )

            center_overhead = surface_index.overhead_pose(
                center_x, center_y, reference_z=float(world_min[2])
            )
            if center_overhead is None:
                return True, ""

            overhead_heights: list[float] = []
            for sample_x, sample_y in footprint_samples:
                overhead = surface_index.overhead_pose(
                    sample_x, sample_y, reference_z=float(world_min[2])
                )
                if overhead is None:
                    return (
                        False,
                        "Furniture footprint straddles the edge of a covered area; "
                        "move it fully under the roof or fully into open air.",
                    )
                overhead_heights.append(float(overhead.position[2]))
            lowest_overhead = min(overhead_heights)
            if float(world_max[2]) > lowest_overhead + epsilon:
                return (
                    False,
                    "Furniture exceeds the available overhead clearance: object top "
                    f"is {float(world_max[2]):.3f}m, overhead is "
                    f"{lowest_overhead:.3f}m.",
                )
            return True, ""

        room_geometry = self.scene.room_geometry
        half_length = float(room_geometry.length) / 2.0
        half_width = float(room_geometry.width) / 2.0
        if (
            float(world_min[0]) < -half_length - epsilon
            or float(world_max[0]) > half_length + epsilon
            or float(world_min[1]) < -half_width - epsilon
            or float(world_max[1]) > half_width + epsilon
        ):
            return (
                False,
                "Furniture footprint exceeds the room floor bounds: "
                f"object X=[{world_min[0]:.3f}, {world_max[0]:.3f}], "
                f"Y=[{world_min[1]:.3f}, {world_max[1]:.3f}].",
            )

        has_overhead = bool(getattr(room_geometry, "has_overhead_cover", True))
        if has_overhead:
            overhead_height = float(room_geometry.wall_height)
            if float(world_max[2]) > overhead_height + epsilon:
                return (
                    False,
                    "Furniture exceeds the available overhead clearance: object top "
                    f"is {float(world_max[2]):.3f}m, overhead is "
                    f"{overhead_height:.3f}m.",
                )
        return True, ""

    def _placement_collisions_for(
        self, object_id: UniqueID
    ) -> tuple[list[str], list[str]]:
        """Return furniture and structural collisions involving one proposed pose."""

        furniture = self.scene.get_objects_by_type(ObjectType.FURNITURE)
        if not isinstance(furniture, (list, tuple)):
            return [], []
        physics_cfg = self.cfg.physics_validation
        collisions = compute_scene_collisions(
            scene=self.scene,
            penetration_threshold=physics_cfg.object_penetration_threshold_m,
            floor_penetration_tolerance=physics_cfg.floor_penetration_tolerance_m,
            manipuland_furniture_tolerance_m=(
                physics_cfg.manipuland_furniture_tolerance_m
            ),
        )
        current = str(object_id)
        furniture_descriptions: list[str] = []
        structural_descriptions: list[str] = []
        for collision in collisions:
            ids = {collision.object_a_id, collision.object_b_id}
            if current not in ids:
                continue
            other_id = next((item for item in ids if item != current), None)
            if other_id is None:
                continue
            try:
                other = self.scene.get_object(UniqueID(other_id))
            except Exception:
                other = None
            description = collision.to_description()
            if other is not None and other.object_type == ObjectType.FURNITURE:
                furniture_descriptions.append(description)
            else:
                structural_descriptions.append(description)
        return furniture_descriptions, structural_descriptions

    def _furniture_collisions_for(self, object_id: UniqueID) -> list[str]:
        """Return concrete furniture collisions involving one proposed pose."""

        furniture_collisions, _ = self._placement_collisions_for(object_id)
        return furniture_collisions

    def _validate_contextual_zones(self, scene_object: SceneObject) -> tuple[bool, str]:
        """Run the fast semantic solver before invoking heavyweight physics."""

        furniture = self.scene.get_objects_by_type(ObjectType.FURNITURE)
        if not isinstance(furniture, (list, tuple)):
            # Compatibility for legacy tests and partially restored scenes. The
            # production RoomScene API always returns a concrete list.
            furniture = []
        existing = [
            item for item in furniture if item.object_id != scene_object.object_id
        ]
        result = validate_scene_object_placement(scene_object, existing)
        hard = [
            violation for violation in result.violations if violation.severity == "hard"
        ]
        soft = [
            violation for violation in result.violations if violation.severity == "soft"
        ]
        for violation in soft:
            console_logger.info(
                "Contextual placement advisory %s: %s",
                violation.code,
                violation.message,
            )
        if not hard:
            return True, ""
        payload = {
            "message": "Contextual placement constraints rejected this pose",
            "violations": [violation.to_dict() for violation in hard],
        }
        return False, json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def _create_loop_error_response(
        self, method_name: str, attempt_count: int, args: tuple, kwargs: dict
    ) -> str:
        """Create furniture-specific error response for loop detection."""
        # Extract object_id from kwargs or args if available.
        object_id = kwargs.get("object_id", "")
        if not object_id and args and len(args) > 1:
            object_id = str(args[1])  # First arg after self

        # Create context-specific diagnostic message.
        if method_name == "_remove_furniture_impl":
            base_name = object_id.rsplit("_", 1)[0] if "_" in object_id else object_id
            diagnostic_message = (
                f"Loop detected: You've tried to remove '{object_id}' {attempt_count} "
                f"times.\n\n"
                f"This means one of:\n"
                f"1. Wrong object name - missing ID postfix (e.g., using '{base_name}' "
                f"instead of '{base_name}_0', '{base_name}_1', etc.)\n"
                f"2. Object was already removed\n"
                f"3. Object doesn't exist with that ID\n\n"
                f"CRITICAL: ALL objects have sequential postfixes (_0, _1, _2, ...).\n"
                f"Base names without postfixes NEVER exist.\n\n"
                f"Recovery procedure (execute in order):\n"
                f"1. Call get_current_scene_state() to see current object IDs with "
                f"postfixes\n"
                f"2. Find objects whose names start with '{base_name}' "
                f"(e.g., '{base_name}_0', '{base_name}_1')\n"
                f"3. Use exact object_id from get_current_scene_state() including postfix\n"
                f"4. If object not in scene, report it was already removed\n\n"
                f"First call get_current_scene_state(), then retry with correct ID."
            )
            suggested_action = (
                "Call get_current_scene_state() to discover object IDs with postfixes"
            )
        elif method_name == "_move_furniture_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to move '{object_id}' {attempt_count} "
                f"times.\n\n"
                f"Causes:\n"
                f"1. Wrong object ID - IDs have sequential postfixes (_0, _1, _2, ...)\n"
                f"2. Object doesn't exist\n"
                f"3. Position/rotation causing collision or validation failure\n\n"
                f"CRITICAL: ALL objects have sequential postfixes. Base names without "
                f"postfixes NEVER exist.\n\n"
                f"Recovery procedure:\n"
                f"1. Call get_current_scene_state() to verify object exists with correct "
                f"ID\n"
                f"2. If collision issue, try different coordinates\n"
                f"3. Check for obstacles blocking the target position"
            )
            suggested_action = "Call get_current_scene_state() to verify object ID"
        else:
            diagnostic_message = (
                f"Loop detected: {attempt_count} identical calls to {method_name}"
            )
            suggested_action = (
                "Call get_current_scene_state() to refresh state, then try different "
                "approach"
            )

        return FurnitureOperationResult(
            success=False,
            message=diagnostic_message,
            object_id=object_id,
            error_type=FurnitureErrorType.LOOP_DETECTED,
            suggested_action=suggested_action,
        ).to_json()
