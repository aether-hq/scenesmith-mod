"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import logging

from pathlib import Path

from agents import custom_span

from scenesmith.agent_utils.geometry.support_surfaces.models import (
    SupportSurfaceExtractionConfig,
)
from scenesmith.agent_utils.physics.physical_feasibility import (
    apply_per_furniture_postprocessing,
)
from scenesmith.agent_utils.scene.room import RoomScene
from scenesmith.agent_utils.scene.room_parts.room_models import UniqueID
from scenesmith.agent_utils.scene.room_parts.room_support import (
    extract_and_propagate_support_surfaces,
)
from scenesmith.agent_utils.scene.scene_analyzer import FurnitureSelection
from scenesmith.prompts.registry import ManipulandAgentPrompts

console_logger = logging.getLogger(__name__)


class ManipulandAgentWorkflowMixin:
    """Final scoring, per-furniture workflow, and placement analysis."""

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving per-furniture manipuland placement state.

        Returns:
            Path to scene_states/manipuland_furniture_{id} directory.
        """
        return (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{self.current_furniture_id}"
        )

    async def add_manipulands(self, scene: RoomScene) -> None:
        """Add manipulands to furniture surfaces in the scene.

        This method implements a two-phase workflow:
        1. VLM-based furniture analysis to identify which pieces need manipulands
        2. Per-furniture multi-agent workflow (planner/designer/critic) to
           populate selected furniture with appropriate small objects

        The scene is mutated in place to add manipuland objects. Fresh agent
        contexts are created for each furniture piece to bound token usage.

        Side effects:
        - Scene objects are added (manipulands placed on furniture)
        - Support surfaces are extracted and assigned to furniture
        - Render cache is cleared before processing
        - Per-furniture subdirectories created under logger output directory
        - Checkpoint state saved after each critique iteration
        - Final scores copied to furniture_<id>/final_scene/ directories

        Requirements:
        - Furniture must have geometry_path (non-None)
        - Furniture must have valid bounding boxes (bbox_min, bbox_max)
        - Scene must have text_description for agent context

        Args:
            scene: RoomScene with furniture already placed. Furniture objects must
                have geometry and bounding boxes to be considered for manipuland
                placement.

        Raises:
            Exception: If support surface extraction fails (indicates invalid
                furniture geometry). Agent execution errors are logged but do
                not halt processing of remaining furniture.
        """
        console_logger.info("Starting manipuland placement")
        self.scene = scene

        # Clear render cache to ensure fresh renders for manipulands.
        # This prevents cache key collisions when object IDs are reused.
        self.rendering_manager.clear_cache()

        # Phase 1: Initial analysis - identify which furniture to populate.
        furniture_data = await self._analyze_furniture_for_placement(scene)

        if not furniture_data:
            console_logger.info("No furniture identified for manipuland placement")
            return

        console_logger.info(
            f"Identified {len(furniture_data)} furniture pieces to populate"
        )

        # Phase 1b: Select context furniture for each selection.
        if self.cfg.context_furniture.enabled:
            # Get path to furniture_selection images (already rendered).
            furniture_selection_dir = (
                self.rendering_manager._base_output_dir
                / "scene_renders"
                / "furniture_selection"
            )
            images_dir = (
                furniture_selection_dir if furniture_selection_dir.exists() else None
            )

            context_map = self.scene_analyzer.select_context_furniture(
                scene=scene,
                furniture_selections=furniture_data,
                furniture_selection_images_dir=images_dir,
            )

            # Attach context to each selection.
            for selection in furniture_data:
                selection.context_furniture_ids = context_map.get(
                    selection.furniture_id, []
                )

        # Phase 2: Per-furniture loop. Identical asset instances share one
        # composed template in their canonical surface frame.
        populated_templates: dict[tuple[str, float], UniqueID] = {}
        for furniture_selection in furniture_data:
            furniture_id = furniture_selection.furniture_id
            # Create custom span for this furniture's manipuland placement.
            with custom_span(
                name=f"manipulands_{furniture_id}",
                data={"furniture_id": str(furniture_id)},
            ):
                console_logger.info(f"Populating furniture: {furniture_id}")
                if furniture_selection.suggested_items:
                    console_logger.info(
                        f"Suggested items: {furniture_selection.suggested_items}"
                    )
                    console_logger.info(
                        f"Prompt constraints: {furniture_selection.prompt_constraints}"
                    )
                    console_logger.info(
                        f"Style notes: {furniture_selection.style_notes}"
                    )

                # Extract support surface for this furniture.
                furniture = scene.get_object(furniture_id)
                if not furniture:
                    console_logger.warning(
                        f"Furniture {furniture_id} not found, skipping"
                    )
                    continue

                # Extract all support surfaces using HSM algorithm.
                hsm_config = SupportSurfaceExtractionConfig.from_config(
                    cfg=self.cfg.support_surface_extraction
                )
                surfaces = extract_and_propagate_support_surfaces(
                    scene=self.scene, furniture_object=furniture, config=hsm_config
                )

                console_logger.info(
                    f"Extracted {len(surfaces)} support surface(s) for {furniture_id}"
                )

                # Skip furniture with no support surfaces (e.g., plants, unsuitable geometry).
                if not surfaces:
                    console_logger.warning(
                        f"No support surfaces found for {furniture_id}, skipping manipuland placement"
                    )
                    continue

                template_key = self._furniture_template_key(furniture)
                template_source = (
                    populated_templates.get(template_key)
                    if template_key is not None
                    else None
                )
                if template_source is not None:
                    clone_count = self._clone_manipulands_between_identical_furniture(
                        template_source, furniture_id
                    )
                    if clone_count:
                        console_logger.info(
                            "Transferred %d manipuland(s) from identical furniture "
                            "%s to %s without another LLM workflow",
                            clone_count,
                            template_source,
                            furniture_id,
                        )
                        continue

                try:
                    # Set up per-furniture context.
                    self._setup_furniture_context(furniture_selection)

                    # Generate context image for manipuland placement (if enabled).
                    self.manipuland_context_image_path = (
                        self._generate_manipuland_context_image()
                    )

                    # Initialize checkpoint state.
                    self._initialize_checkpoint_state()

                    # Get furniture description for agent prompts.
                    furniture_obj = scene.get_object(furniture_id)
                    furniture_description = (
                        furniture_obj.description if furniture_obj else "furniture"
                    )

                    # Create agents and sessions.
                    self._setup_furniture_agents(
                        furniture_id=furniture_id,
                        furniture_description=furniture_description,
                    )

                    book_rows_placed = self._place_dense_book_rows_deterministically(
                        furniture
                    )
                    if book_rows_placed:
                        console_logger.info(
                            "Deterministically populated %d internal bookcase tiers "
                            "for %s",
                            book_rows_placed,
                            furniture_id,
                        )

                    dense_rows = self._dense_book_rows_on_furniture(
                        self.scene,
                        furniture,
                    )
                    if len(dense_rows) >= 4:
                        console_logger.info(
                            "Skipping manipuland LLM workflow for %s: %d clean "
                            "deterministic book rows already satisfy this bookcase",
                            furniture_id,
                            len(dense_rows),
                        )
                    else:
                        # Run multi-agent workflow.
                        await self._run_furniture_workflow(furniture_id)

                    # Per-furniture post-processing (after manipulands placed).
                    if self.cfg.per_furniture_postprocessing.enabled:
                        sim_cfg = self.cfg.per_furniture_postprocessing.simulation
                        sim_html_path = None
                        if sim_cfg.save_html:
                            sim_html_path = (
                                self.scene.scene_dir
                                / "simulation"
                                / "per_furniture"
                                / f"{furniture_id}_simulation.html"
                            )
                        self.scene = apply_per_furniture_postprocessing(
                            full_scene=self.scene,
                            furniture_id=furniture_id,
                            config=self.cfg.per_furniture_postprocessing,
                            simulation_html_path=sim_html_path,
                        )

                    if template_key is not None:
                        populated_templates[template_key] = furniture_id

                except Exception as e:
                    console_logger.error(
                        f"Error populating furniture {furniture_id}: {e}", exc_info=True
                    )
                    # Continue to next furniture piece.
                    continue

        normalized_rows, discarded_rows = self._normalize_intrinsic_dense_book_rows()
        if normalized_rows or discarded_rows:
            console_logger.info(
                "Normalized exact dense book rows before final dynamics: "
                "%d owner-bound, %d uncontained discarded",
                normalized_rows,
                discarded_rows,
            )

        recovered_dense_book_rows = self._recover_dense_library_book_row_deficits()
        if recovered_dense_book_rows:
            console_logger.info(
                "Recovered %d dense book rows across additional compatible "
                "same-story bookcases",
                recovered_dense_book_rows,
            )
        removed_dense_book_rows = self._normalize_dense_library_book_row_surplus()
        if removed_dense_book_rows:
            console_logger.info(
                "Pruned %d surplus dense book rows to the canonical per-story " "count",
                removed_dense_book_rows,
            )
        invalid_dense_book_rows = self._physically_invalid_dense_book_row_ids(
            self.scene,
            self.cfg,
        )
        dense_book_rows = self._validate_dense_library_book_rows(
            self.scene,
            invalid_row_ids=invalid_dense_book_rows,
        )
        if dense_book_rows:
            console_logger.info(
                "Dense library book-row completion passed with %d surviving rows",
                dense_book_rows,
            )
        console_logger.info("Manipuland placement complete")

    async def _analyze_furniture_for_placement(
        self, scene: RoomScene
    ) -> list[FurnitureSelection]:
        """Analyze which furniture should have manipulands.

        Delegates to SceneAnalyzer for VLM-based furniture selection.

        Args:
            scene: RoomScene with furniture.

        Returns:
            List of FurnitureSelection objects with assignment context.
        """
        return self.scene_analyzer.analyze_furniture_for_manipulands(
            scene=scene,
            prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
        )
