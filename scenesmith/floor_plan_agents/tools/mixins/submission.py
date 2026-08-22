"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import json
import logging

from typing import Any, Callable

from scenesmith.agent_utils.scene.house_parts.openings import ConnectionType
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    default_ground_level,
)
from scenesmith.floor_plan_agents.tools.floor_plan_models import Result
from scenesmith.floor_plan_agents.tools.submission.structural_submission import (
    synthesize_structural_layout,
)
from scenesmith.floor_plan_agents.tools.submission.submission_models import (
    NormalizedFloorPlanSubmission,
)

console_logger = logging.getLogger(__name__)


class FloorPlanSubmissionMixin:
    """One-shot submission, fallback, openings, and deterministic materials."""

    def _reset_structural_state_for_fallback(self) -> None:
        """Return the shared layout object to its safe flat compatibility state."""

        self.layout.levels = [default_ground_level()]
        self.layout.connectors.clear()
        self.layout.platforms.clear()
        self.layout.portals.clear()
        self.layout.heightfields.clear()
        self.layout.structural_meshes.clear()
        self.layout.connector_geometry_paths.clear()
        self.layout.platform_geometry_paths.clear()
        self.layout.heightfield_geometry_paths.clear()
        self.layout.structural_mesh_geometry_paths.clear()
        self.layout.semantic_environment = None
        self.layout.semantic_environment_geometry_path = None
        self.layout.semantic_detail_geometry_paths.clear()
        self.layout.connectivity_valid = False
        self.layout.invalidate_all_room_geometries()

    def _submit_floor_plan_with_fallback(
        self, submission: NormalizedFloorPlanSubmission
    ) -> Result:
        """Execute normalized intent, then degrade only as far as necessary."""

        def attempt(kwargs: dict[str, Any], *, label: str) -> Result:
            try:
                return self._submit_floor_plan_impl(**kwargs)
            except Exception as exc:
                console_logger.exception(
                    "%s floor-plan attempt raised; continuing to fallback", label
                )
                return Result(
                    success=False,
                    message=f"{label} attempt rejected malformed input: {exc}",
                )

        result = attempt(submission.tool_kwargs(), label="canonical")
        if result.success:
            return result
        failures = [result.message]
        console_logger.warning(
            "Canonical floor-plan submission failed: %s", result.message
        )

        synthesized = synthesize_structural_layout(
            self.layout.house_prompt,
            submission.room_specs,
            submission.wall_height_meters,
            max_total_height=self.wall_height_max,
        )
        if submission.structural is not None and synthesized is not None:
            self._reset_structural_state_for_fallback()
            retry_kwargs = submission.tool_kwargs()
            retry_kwargs["structural"] = synthesized
            result = attempt(retry_kwargs, label="synthesized structural fallback")
            if result.success:
                diagnostics = tuple(synthesized.get("_diagnostics", ()))
                for diagnostic in diagnostics:
                    console_logger.warning(
                        "Structural fallback degradation: %s", diagnostic
                    )
                return Result(
                    success=True,
                    message=(
                        result.message
                        + " Repaired the provider's structural payload with the "
                        "deterministic multi-level fallback."
                        + (" " + " ".join(diagnostics) if diagnostics else "")
                    ),
                )
            failures.append(result.message)
            console_logger.warning(
                "Synthesized structural fallback failed: %s", result.message
            )

        # Geometry is still useful when optional multi-level intent is irreparable.
        # Preserve the full room prompt so downstream furnishing still has all user
        # requirements, and make the degradation explicit in logs/result metadata.
        self._reset_structural_state_for_fallback()
        retry_kwargs = submission.tool_kwargs()
        retry_kwargs["structural"] = None
        retry_kwargs["wall_height_meters"] = max(
            self.wall_height_min,
            min(self.wall_height_max, submission.wall_height_meters),
        )
        result = attempt(retry_kwargs, label="safe flat fallback")
        if result.success:
            return Result(
                success=True,
                message=(
                    result.message
                    + " Used a safe flat structural fallback after: "
                    + " | ".join(dict.fromkeys(failures))
                ),
            )

        failures.append(result.message)
        console_logger.error(
            "All deterministic floor-plan fallbacks failed: %s",
            " | ".join(dict.fromkeys(failures)),
        )
        return Result(
            success=False,
            message="All deterministic floor-plan fallbacks failed: "
            + " | ".join(dict.fromkeys(failures)),
        )

    def _submit_floor_plan_impl(
        self,
        *,
        room_specs: list[dict[str, Any]],
        wall_height_meters: float = 3.0,
        structural: dict[str, Any] | None = None,
        windows_per_room: int = 1,
        window_shape: str = "rectangular",
        window_width_m: float = 1.2,
        window_height_m: float = 1.2,
        window_sill_height_m: float = 0.9,
        floor_material_description: str = "warm wood floor",
        wall_material_description: str = "neutral plaster wall",
        exterior_material_description: str = "neutral exterior plaster",
        exterior_door_room_id: str = "",
    ) -> Result:
        """Apply one model-produced design with deterministic finishing steps."""

        console_logger.info("Tool called: submit_floor_plan")
        if not isinstance(room_specs, list) or not room_specs:
            return self._fail("room_specs must be a non-empty array")
        if not all(isinstance(spec, dict) for spec in room_specs):
            return self._fail("every room_specs entry must be an object")

        rooms_result = self._generate_room_specs_impl(json.dumps(room_specs))
        if not rooms_result.success:
            return self._fail(rooms_result.message)

        if structural:
            if not isinstance(structural, dict):
                return self._fail("structural must be an object when provided")
            structural_result = self._set_structural_layout_impl(structural)
            if not structural_result.success:
                return self._fail(structural_result.message)

            # Models often return a per-storey height while the legacy room shell
            # needs the full vertical extent. Otherwise its ceiling cuts through
            # the first stair flight and the geometric clearance pass vetoes it.
            if self.layout.levels:
                lowest_elevation = min(level.elevation for level in self.layout.levels)
                structural_height = (
                    max(
                        level.elevation + level.nominal_height
                        for level in self.layout.levels
                    )
                    - lowest_elevation
                )
                wall_height_meters = max(float(wall_height_meters), structural_height)

        height_result = self._set_wall_height_impl(float(wall_height_meters))
        if not height_result.success:
            return height_result

        door_error = self._add_required_doors_deterministically(
            exterior_door_room_id=exterior_door_room_id
        )
        if door_error is not None:
            return door_error

        self._add_windows_deterministically(
            windows_per_room=windows_per_room,
            shape=window_shape,
            width=window_width_m,
            height=window_height_m,
            sill_height=window_sill_height_m,
        )
        self._assign_materials_deterministically(
            floor_description=floor_material_description,
            wall_description=wall_material_description,
            exterior_description=exterior_material_description,
        )

        validation = self._validate_impl()
        if validation.layout != "ok" or validation.connectivity != "ok":
            return self._fail(
                "Floor plan validation failed: "
                f"layout={validation.layout}; connectivity={validation.connectivity}"
            )

        checkpoint_saved = self._checkpoint_if_valid()
        if self.checkpoint_callback is not None and not checkpoint_saved:
            return self._fail(
                "The semantic layout passed, but geometry/checkpoint validation "
                "failed; applying a deterministic structural fallback."
            )
        return Result(
            success=True,
            message=(
                f"Completed and validated {len(self.layout.room_specs)} room(s), "
                f"{len(self.layout.doors)} door(s), and "
                f"{len(self.layout.windows)} window(s)."
            ),
        )

    def _add_required_doors_deterministically(
        self, *, exterior_door_room_id: str = ""
    ) -> Result | None:
        """Add required interior and exterior doors without another LLM turn."""

        for wall_id, (room_a, room_b, _direction) in sorted(
            self.layout.boundary_labels.items()
        ):
            if room_b is None:
                continue
            spec_a = self.layout.get_room_spec(room_a)
            spec_b = self.layout.get_room_spec(room_b)
            connection_a = spec_a.connections.get(room_b) if spec_a else None
            connection_b = spec_b.connections.get(room_a) if spec_b else None
            if ConnectionType.DOOR not in {connection_a, connection_b}:
                continue
            if any(
                {door.room_a, door.room_b} == {room_a, room_b}
                for door in self.layout.doors
            ):
                continue
            result = self._try_opening_positions(self._add_door_impl, wall_id=wall_id)
            if result is None or not result.success:
                return self._fail(
                    f"Could not add required door between {room_a} and {room_b}."
                )

        if any(door.room_b is None for door in self.layout.doors):
            return None

        preferred_room = (
            exterior_door_room_id
            if self.layout.get_room_spec(exterior_door_room_id)
            else ""
        )
        candidates = [
            (wall_id, room_a)
            for wall_id, (room_a, room_b, _direction) in sorted(
                self.layout.boundary_labels.items()
            )
            if room_b is None
        ]
        candidates.sort(key=lambda item: item[1] != preferred_room)
        for wall_id, _room_id in candidates:
            result = self._try_opening_positions(self._add_door_impl, wall_id=wall_id)
            if result is not None and result.success:
                return None
        return self._fail("Could not place an exterior door on any exterior wall.")

    @staticmethod
    def _try_opening_positions(operation: Callable[..., Result], *, wall_id: str):
        """Try stable wall segments in preferred order."""

        last_result = None
        for position in ("center", "left", "right"):
            last_result = operation(wall_id=wall_id, position=position)
            if last_result.success:
                return last_result
        return last_result

    def _add_windows_deterministically(
        self,
        *,
        windows_per_room: int,
        shape: str = "rectangular",
        width: float = 1.2,
        height: float = 1.2,
        sill_height: float = 0.9,
    ) -> None:
        """Add requested windows to viable exterior walls, best effort."""

        try:
            target_count = max(0, min(8, int(windows_per_room)))
        except (TypeError, ValueError):
            target_count = 1
        for room in self.layout.room_specs:
            existing_count = sum(
                window.room_id == room.room_id for window in self.layout.windows
            )
            if existing_count >= target_count:
                continue
            candidates = [
                wall_id
                for wall_id, (room_a, room_b, _direction) in sorted(
                    self.layout.boundary_labels.items()
                )
                if room_a == room.room_id
                and room_b is None
                and not any(
                    door.boundary_label == wall_id for door in self.layout.doors
                )
            ]
            for wall_id in candidates:
                for position in ("center", "left", "right"):
                    result = self._add_window_impl(
                        wall_id=wall_id,
                        position=position,
                        width=width,
                        height=height,
                        sill_height=sill_height,
                        shape=shape,
                    )
                    if result.success:
                        existing_count += 1
                        if existing_count >= target_count:
                            break
                if existing_count >= target_count:
                    break
            if existing_count < target_count:
                console_logger.warning(
                    "Placed %d of %d requested windows for room %s",
                    existing_count,
                    target_count,
                    room.room_id,
                )

    def _assign_materials_deterministically(
        self,
        *,
        floor_description: str,
        wall_description: str,
        exterior_description: str,
    ) -> None:
        """Resolve shared material intent once and apply it to every room."""

        floor_material = self._get_material_impl(floor_description)
        wall_material = self._get_material_impl(wall_description)
        if floor_material.success and wall_material.success:
            for room in self.layout.room_specs:
                result = self._set_room_materials_impl(
                    room_id=room.room_id,
                    floor_material_id=floor_material.material_id,
                    wall_material_id=wall_material.material_id,
                )
                if not result.success:
                    console_logger.warning(result.message)

        exterior_material = self._get_material_impl(exterior_description)
        if exterior_material.success:
            result = self._set_exterior_material_impl(exterior_material.material_id)
            if not result.success:
                console_logger.warning(result.message)
