"""Stateful floor plan agent using planner/designer/critic workflow.

This module implements the floor plan agent trio for designing house layouts
with rooms, doors, windows, and materials, then generates the geometry.
"""

import copy
import json
import logging
import shutil

from typing import Any

import yaml

from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.runtime.agent_runtime import (
    BoundedRunner as Runner,
    agent_run_timeout_seconds,
)
from scenesmith.agent_utils.runtime.base_stateful_agent import log_agent_usage
from scenesmith.agent_utils.runtime.scoring import (
    FloorPlanCritiqueWithScores,
    format_score_deltas_for_planner,
    log_agent_response,
    log_critique_scores,
    scores_to_dict,
)
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import (
    Door,
    Opening,
    OpeningType,
    WallDirection,
    Window,
    WindowShape,
)
from scenesmith.agent_utils.semantics.requirements.compilation.models import (
    SpatialCompilationError,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    PortalSpec,
    PortalType,
)
from scenesmith.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenesmith.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenesmith.floor_plan_agents.tools.submission.floor_plan_submission import (
    opening_placements_from_blueprint,
)
from scenesmith.floor_plan_agents.tools.submission.placement.layout import (
    create_placed_room,
    update_wall_connectivity,
)
from scenesmith.prompts.registry import FloorPlanAgentPrompts

console_logger = logging.getLogger(__name__)


class FloorPlanCheckpointWorkflowMixin:
    """Stateful floor plan agent using planner/designer/critic workflow.

    This agent designs house layouts through an iterative process of:
    1. Designer proposes rooms, doors, windows using layout tools.
    2. Critic evaluates the design with VLM-based visual critique.
    3. Iteration continues until the design meets quality criteria.

    The layout is stored in a HouseLayout object that tracks:
    - Room specifications with adjacency constraints
    - Door and window placements on walls
    - Material assignments for floors and walls

    After design completion, geometry is generated for each room:
    - Floor meshes as GLTF
    - Wall meshes with door/window openings as GLTF
    - Full SDF/URDF assembly for Drake simulation
    """

    # Floor plan agent doesn't place objects, so no placement style tool.
    _is_placement_agent: bool = False

    async def _request_critique_impl(self, update_checkpoint: bool = True) -> str:
        """Implementation for critique request.

        Runs critic which calls observe_scene, render_ascii, and validate tools.
        Images persist in session via ToolOutputImage.

        Args:
            update_checkpoint: Whether to shift checkpoints. Set to False for
                final critique calls to preserve N-1 checkpoint for reset check.

        Returns:
            Critique text with scores.
        """
        console_logger.info("Tool called: request_critique")

        # Critiques can involve multiple long-running vision/model turns. Save a
        # portable, geometry-complete layout before entering that boundary so a
        # malformed model response or interruption can resume at furniture.
        self._write_resumable_layout_checkpoint()

        # Get critique instruction.
        critique_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FloorPlanAgentPrompts.CRITIC_RUNNER_INSTRUCTION,
        )

        # Run critic.
        # Critic will call observe_scene, render_ascii, and validate tools.
        result = await Runner.run(
            starting_agent=self.critic,
            input=critique_instruction,
            session=self.critic_session,
            max_turns=self.cfg.agents.critic_agent.max_turns,
            run_config=self._create_run_config(),
            timeout_seconds=agent_run_timeout_seconds(
                "critic",
                max_turns=self.cfg.agents.critic_agent.max_turns,
            ),
        )
        log_agent_usage(result=result, agent_name="CRITIC (FLOOR PLAN)")
        vision_tools = self._get_vision_tools()

        # Parse structured output.
        response = result.final_output_as(FloorPlanCritiqueWithScores)

        # Log critique.
        log_agent_response(response=response.critique, agent_name="CRITIC")
        log_critique_scores(response, title="FLOOR PLAN CRITIQUE SCORES")

        # Save scores to render directory.
        scores_dict = scores_to_dict(response)
        render_dir = vision_tools.last_render_dir

        # Always track the final render directory (separate from checkpoint logic).
        # This is needed because final critique uses update_checkpoint=False, but we
        # still need to know the actual last render dir for copying to final output.
        self.final_render_dir = render_dir

        scores_path = render_dir / "scores.yaml"
        with open(scores_path, "w") as f:
            yaml.dump(scores_dict, f, default_flow_style=False, sort_keys=False)
        console_logger.info(f"Scores saved to: {scores_path}")

        # Shift checkpoints only during iteration critiques, not final critique.
        # This preserves N-1 checkpoint for reset check in _finalize_scene_and_scores.
        if update_checkpoint:
            # Update checkpoint state (shift current to previous before saving new).
            self.previous_scene_checkpoint = self.scene_checkpoint
            self.previous_checkpoint_scores = self.checkpoint_scores
            self.previous_checkpoint_render_dir = self.checkpoint_render_dir

            # Save new checkpoint (current scene state).
            self.scene_checkpoint = copy.deepcopy(self.layout.to_dict())
            self.checkpoint_scores = response
            self.checkpoint_render_dir = (
                render_dir if render_dir and render_dir.exists() else None
            )

            # Reuse render cache hash for checkpoint change detection.
            self.checkpoint_scene_hash = self.layout.content_hash()

        # Compute score deltas BEFORE updating previous_scores.
        score_change_msg = ""
        if self.previous_scores is not None:
            score_change_msg = format_score_deltas_for_planner(
                current_scores=response,
                previous_scores=self.previous_scores,
                format_style="detailed",
            )

        # Always update previous_scores for delta formatting in planner.
        self.previous_scores = response

        return response.critique + score_change_msg

    def _write_resumable_layout_checkpoint(self) -> bool:
        """Atomically persist the latest structurally valid floor-plan state."""
        if (
            not self.layout.room_specs
            or not self.layout.placement_valid
            or not self.layout.connectivity_valid
        ):
            console_logger.info(
                "Skipping floor-plan checkpoint because the layout is not yet valid"
            )
            return False
        try:
            self.layout.validate_structure()
            self._generate_all_room_geometries(
                output_dir=self.logger.output_dir / "floor_plans"
            )
            checkpoint_path = self.logger.output_dir / "house_layout.json"
            pending_path = checkpoint_path.with_suffix(".json.pending")
            with pending_path.open("w") as file:
                json.dump(
                    self.layout.to_dict(scene_dir=self.logger.output_dir),
                    file,
                    indent=2,
                )
            pending_path.replace(checkpoint_path)
            console_logger.info(
                "Saved resumable floor-plan checkpoint to %s", checkpoint_path
            )
            return True
        except Exception as exc:
            console_logger.warning(
                "Could not save resumable floor-plan checkpoint: %s", exc
            )
            return False

    def _apply_locked_blueprint_topology(self) -> None:
        """Make accepted semantic dimensions and apertures authoritative."""

        if self.blueprint is None or not self.layout.room_specs:
            return

        blueprint = self.blueprint
        actual_by_id = {spec.room_id: spec for spec in self.layout.room_specs}
        space_to_room: dict[str, str] = {}
        unclaimed_room_ids = set(actual_by_id)
        for space in blueprint.spaces:
            if space.space_id in actual_by_id:
                room_id = space.space_id
            elif self.mode == "room" and len(actual_by_id) == 1:
                room_id = next(iter(actual_by_id))
            else:
                room_id = next(
                    (
                        candidate_id
                        for candidate_id in sorted(unclaimed_room_ids)
                        if actual_by_id[candidate_id].room_type == space.room_type
                    ),
                    "",
                )
                if not room_id:
                    continue
            space_to_room[space.space_id] = room_id
            unclaimed_room_ids.discard(room_id)

        # A repeated room-mode space represents the same coherent volume on
        # another level. Its ground footprint remains the first authored space.
        primary_space_by_room: dict[str, Any] = {}
        for space in blueprint.spaces:
            room_id = space_to_room.get(space.space_id)
            if room_id is not None and room_id not in primary_space_by_room:
                primary_space_by_room[room_id] = space
        previous_positions = {
            room.room_id: room.position for room in self.layout.placed_rooms
        }
        for room_id, space in primary_space_by_room.items():
            spec = actual_by_id[room_id]
            spec.length = float(space.dimensions_m[0])
            spec.width = float(space.dimensions_m[1])
            spec.has_overhead_cover = bool(space.covered)

        self.layout.placed_rooms = [
            create_placed_room(
                spec,
                previous_positions.get(spec.room_id, tuple(spec.position)),
            )
            for spec in self.layout.room_specs
        ]
        update_wall_connectivity(self.layout.placed_rooms)
        ascii_plan = generate_ascii_floor_plan(self.layout.placed_rooms)
        self.layout.boundary_labels = ascii_plan.boundary_labels
        self.layout.placement_valid = True

        lowest_elevation = min(level.elevation_m for level in blueprint.levels)
        semantic_height = (
            max(level.elevation_m + level.clear_height_m for level in blueprint.levels)
            - lowest_elevation
        )
        if any(
            opening.sill_height_m + opening.height_m > semantic_height + 1e-9
            for opening in blueprint.openings
        ):
            raise SpatialCompilationError(
                "accepted blueprint contains an opening taller than its host shell"
            )
        self.layout.wall_height = semantic_height

        placements = opening_placements_from_blueprint(blueprint)
        authored_kinds = {placement.kind for placement in placements}
        has_exterior_walkable_opening = any(
            placement.kind in {"door", "open_connection"}
            and placement.connects_to_space_id is None
            for placement in placements
        )
        if "window" in authored_kinds:
            self.layout.windows.clear()
        if has_exterior_walkable_opening:
            self.layout.doors.clear()
        removable_types = set()
        if "window" in authored_kinds:
            removable_types.add(OpeningType.WINDOW)
        if has_exterior_walkable_opening:
            removable_types.update({OpeningType.DOOR, OpeningType.OPEN})
        if removable_types:
            for room in self.layout.placed_rooms:
                for wall in room.walls:
                    wall.openings[:] = [
                        opening
                        for opening in wall.openings
                        if opening.opening_type not in removable_types
                    ]

        opening_ids = {placement.opening_id for placement in placements}
        self.layout.portals[:] = [
            portal
            for portal in self.layout.portals
            if portal.portal_id not in opening_ids
            and portal.portal_id not in {f"topology-{item}" for item in opening_ids}
        ]
        edge_directions = (
            WallDirection.SOUTH,
            WallDirection.EAST,
            WallDirection.NORTH,
            WallDirection.WEST,
        )
        for placement in placements:
            room_id = space_to_room.get(placement.host_space_id)
            if room_id is None:
                raise SpatialCompilationError(
                    f"blueprint opening {placement.opening_id!r} has no constructed host"
                )
            placed_room = next(
                room for room in self.layout.placed_rooms if room.room_id == room_id
            )
            direction = edge_directions[placement.boundary_edge_index]
            wall = next(
                candidate
                for candidate in placed_room.walls
                if candidate.direction == direction
            )
            reverse = placement.boundary_edge_index in {2, 3}
            position_exact = (
                wall.length - placement.position_along_m - placement.width_m / 2.0
                if reverse
                else placement.position_along_m - placement.width_m / 2.0
            )
            opening_type = {
                "door": OpeningType.DOOR,
                "window": OpeningType.WINDOW,
                "open_connection": OpeningType.OPEN,
            }[placement.kind]
            wall.openings.append(
                Opening(
                    opening_id=placement.opening_id,
                    opening_type=opening_type,
                    position_along_wall=position_exact,
                    width=placement.width_m,
                    height=placement.height_m,
                    sill_height=placement.sill_height_m,
                    shape=WindowShape(placement.shape),
                )
            )
            boundary_label = next(
                (
                    label
                    for label, (label_room, other_room, label_direction) in (
                        self.layout.boundary_labels.items()
                    )
                    if label_room == room_id
                    and (
                        label_direction == direction.value
                        if placement.connects_to_space_id is None
                        else other_room
                        == space_to_room.get(placement.connects_to_space_id)
                    )
                ),
                wall.wall_id,
            )
            target_room_id = (
                space_to_room.get(placement.connects_to_space_id)
                if placement.connects_to_space_id is not None
                else None
            )
            if placement.kind == "door":
                self.layout.doors.append(
                    Door(
                        id=placement.opening_id,
                        boundary_label=boundary_label,
                        position_segment="center",
                        position_exact=position_exact,
                        door_type="interior" if target_room_id else "exterior",
                        room_a=room_id,
                        room_b=target_room_id,
                        width=placement.width_m,
                        height=placement.height_m,
                    )
                )
            elif placement.kind == "window":
                self.layout.windows.append(
                    Window(
                        id=placement.opening_id,
                        boundary_label=boundary_label,
                        position_along_wall=position_exact,
                        room_id=room_id,
                        wall_direction=direction,
                        width=placement.width_m,
                        height=placement.height_m,
                        sill_height=placement.sill_height_m,
                        shape=WindowShape(placement.shape),
                    )
                )
            else:
                self.layout.portals.append(
                    PortalSpec(
                        portal_id=f"topology-{placement.opening_id}",
                        portal_type=PortalType.OPEN,
                        source_space_id=room_id,
                        target_space_id=target_room_id,
                        width=placement.width_m,
                        height=placement.height_m,
                    )
                )

        self.layout.invalidate_all_room_geometries()
        validation = FloorPlanTools(
            layout=self.layout,
            mode=self.mode,
            wall_height_min=self.cfg.wall_height.min,
            wall_height_max=self._construction_wall_height_max(),
            room_dim_min=self.cfg.min_floor_plan_dim_m,
            room_dim_max=self._construction_room_dim_max(),
        )._validate_impl()
        if validation.layout != "ok" or validation.connectivity != "ok":
            raise SpatialCompilationError(
                "accepted blueprint could not be constructed: "
                f"layout={validation.layout}; connectivity={validation.connectivity}"
            )

    @log_scene_action
    def _perform_checkpoint_reset(self, checkpoint_state_dict: dict) -> None:
        """Restore layout and scores to previous checkpoint (N-1).

        Override of base class method to restore HouseLayout instead of RoomScene.

        Args:
            checkpoint_state_dict: Checkpoint state dictionary to restore from.
                During normal operation, this is self.previous_scene_checkpoint.
                During replay, this is the logged checkpoint state.
        """
        # Restore layout from checkpoint (N-1 iteration).
        self.layout = HouseLayout.from_dict(
            data=checkpoint_state_dict, house_dir=self.layout.house_dir
        )

        # Force SDF regeneration since files on disk are not versioned.
        # Without this, room_geometries from checkpoint would be used,
        # but SDF files on disk have door positions from later iterations.
        self.layout.room_geometries.clear()

        # Update vision tools with restored layout (preserve render counter).
        if self._vision_tools is not None:
            self._vision_tools.update_layout(self.layout)
            self._vision_tools.clear_cache()

        # Recreate designer/critic tools (they reference self.layout directly).
        self._recreate_tools_with_layout()

        # Reset score tracking to previous checkpoint state.
        if self.previous_checkpoint_scores is not None:
            self.checkpoint_scores = copy.deepcopy(self.previous_checkpoint_scores)
            self.previous_scores = copy.deepcopy(self.previous_checkpoint_scores)

        # Invalidate current checkpoint since we went back.
        if self.previous_scene_checkpoint is not None:
            self.scene_checkpoint = self.previous_scene_checkpoint
            self.checkpoint_render_dir = self.previous_checkpoint_render_dir

    def _recreate_tools_with_layout(self) -> None:
        """Recreate tools after layout restoration to ensure they reference current layout."""
        # Designer tools reference self.layout, need to recreate them.
        designer_tools = self._create_designer_tools()
        self.designer = self._create_designer_agent(tools=designer_tools)

        # Critic tools also reference self.layout.
        critic_tools = self._create_critic_tools()
        self.critic = self._create_critic_agent(tools=critic_tools)

    async def _finalize_scene_and_scores(self) -> None:
        """Validate final scene against thresholds and save scores.

        Override of base class to use FloorPlanVisionTools instead of rendering_manager.
        The base class assumes `self.rendering_manager` and `self.scene` exist, but
        floor plan agent uses `FloorPlanVisionTools` and `self.layout` instead.
        """
        # Check if final scores warrant resetting to previous checkpoint.
        # Use previous_scores (actual final critique) vs checkpoint_scores (last checkpoint).
        # Note: Final critique uses update_checkpoint=False, so previous_scores holds the
        # actual final scores while checkpoint_scores holds the last iteration's scores.
        if self.previous_scores is not None and self.checkpoint_scores is not None:
            should_reset, reason = self._should_reset_to_checkpoint(
                current_scores=self.previous_scores,
                previous_scores=self.checkpoint_scores,
            )
            console_logger.info(
                f"Reset check result: should_reset={should_reset}, reason={reason}"
            )

            if should_reset:
                console_logger.info(
                    f"Final scene scores are degraded ({reason}). "
                    f"Resetting to checkpoint (N-1)."
                )

                # Restore layout to checkpoint (N-1) directly. Don't use
                # _perform_checkpoint_reset() here since that's designed for mid-loop
                # resets and modifies checkpoint tracking variables.
                self.layout = HouseLayout.from_dict(
                    data=self.scene_checkpoint, house_dir=self.layout.house_dir
                )

                # Force SDF regeneration since files on disk are not versioned.
                # Without this, room_geometries from checkpoint would be used,
                # but SDF files on disk have dimensions from later iterations.
                self.layout.room_geometries.clear()

                self._vision_tools = None
                self._recreate_tools_with_layout()

                # Render the reset state using vision tools.
                console_logger.info("Rendering final scene after reset")
                vision_tools = self._get_vision_tools()
                vision_tools.clear_cache()  # Force new render.
                vision_tools._observe_scene_impl()
                render_dir = vision_tools.last_render_dir
                self.checkpoint_render_dir = render_dir
                self.final_render_dir = render_dir  # Update so correct dir is copied.

                # Save scores to the new render directory.
                # Use checkpoint_scores (N-1) since we reset to that state.
                if self.checkpoint_scores is not None:
                    scores_dict = scores_to_dict(self.checkpoint_scores)
                    scores_path = render_dir / "scores.yaml"
                    with open(scores_path, "w") as f:
                        yaml.dump(
                            scores_dict,
                            f,
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    console_logger.info(f"Scores saved to: {scores_path}")

                console_logger.info(f"Final scene restored to checkpoint state.")

        # Copy final scores and renders to final_floor_plan/ directory.
        # Use final_render_dir (tracks actual last render) instead of checkpoint_render_dir
        # (which may be stale when final critique uses update_checkpoint=False).
        render_dir_to_copy = self.final_render_dir or self.checkpoint_render_dir
        if render_dir_to_copy is not None:
            final_scene_dir = self._get_final_scores_directory()
            final_scene_dir.mkdir(parents=True, exist_ok=True)

            # Copy scores.
            scores_source = render_dir_to_copy / "scores.yaml"
            if scores_source.exists():
                scores_dest = final_scene_dir / "scores.yaml"
                shutil.copy(scores_source, scores_dest)
                console_logger.info(f"Saved final scores to {scores_dest}")
            else:
                console_logger.warning(
                    f"Scores file not found at {scores_source}, cannot copy"
                )

            # Copy render images.
            render_images = list(render_dir_to_copy.glob("*.png"))
            if render_images:
                for img_path in render_images:
                    img_dest = final_scene_dir / img_path.name
                    shutil.copy(img_path, img_dest)
                console_logger.info(
                    f"Copied {len(render_images)} render images to {final_scene_dir}"
                )
            else:
                console_logger.warning(
                    f"No render images found in {render_dir_to_copy}"
                )

    async def _request_initial_design_impl(self) -> str:
        """Implementation for initial design request.

        Returns:
            Designer's report of initial design.
        """
        console_logger.info("Tool called: request_initial_design")

        # Get instruction.
        one_shot = (
            len(self.designer.tools) == 1
            and getattr(self.designer.tools[0], "name", "") == "submit_floor_plan"
        )
        instruction = self.prompt_registry.get_prompt(
            prompt_enum=(
                FloorPlanAgentPrompts.DESIGNER_ONE_SHOT_INSTRUCTION
                if one_shot
                else FloorPlanAgentPrompts.DESIGNER_INITIAL_INSTRUCTION
            ),
        )

        # Run designer.
        result = await Runner.run(
            starting_agent=self.designer,
            input=instruction,
            session=self.designer_session,
            max_turns=self.cfg.agents.designer_agent.max_turns,
            run_config=self._create_run_config(),
            timeout_seconds=agent_run_timeout_seconds(
                "designer",
                max_turns=self.cfg.agents.designer_agent.max_turns,
            ),
        )
        log_agent_usage(result=result, agent_name="DESIGNER (INITIAL FLOOR PLAN)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (INITIAL)"
            )

        return result.final_output

    async def _request_design_change_impl(self, instruction: str) -> str:
        """Implementation for design change request.

        Args:
            instruction: Changes to make based on critique.

        Returns:
            Designer's report of changes.
        """
        console_logger.info("Tool called: request_design_change")

        # Get instruction.
        full_instruction = self.prompt_registry.get_prompt(
            prompt_enum=FloorPlanAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION,
            instruction=instruction,
        )

        # Run designer.
        result = await Runner.run(
            starting_agent=self.designer,
            input=full_instruction,
            session=self.designer_session,
            max_turns=self.cfg.agents.designer_agent.max_turns,
            run_config=self._create_run_config(),
            timeout_seconds=agent_run_timeout_seconds(
                "designer",
                max_turns=self.cfg.agents.designer_agent.max_turns,
            ),
        )
        log_agent_usage(result=result, agent_name="DESIGNER (CHANGE FLOOR PLAN)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="DESIGNER (CHANGE)"
            )

        return result.final_output
