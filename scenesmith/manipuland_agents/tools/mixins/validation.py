import logging

from pathlib import Path

import numpy as np
import trimesh

from scipy.spatial import ConvexHull, QhullError

from scenesmith.agent_utils.scene.room_parts.room_models import (
    SceneObject,
    SupportSurface,
)
from scenesmith.manipuland_agents.tools.response_dataclasses import (
    ManipulandErrorType,
    ManipulandOperationResult,
    PenetrationResolutionResult,
    PileCreationResult,
    StackCreationResult,
)

console_logger = logging.getLogger(__name__)


class ManipulandValidationMixin:
    """Support semantics, convex-hull, surface, and loop validation."""

    @staticmethod
    def _support_semantics_allow(
        furniture: SceneObject, asset: SceneObject
    ) -> tuple[bool, str | None]:
        """Enforce deterministic contextual support-zone compatibility."""
        furniture_text = f"{furniture.name} {furniture.description}".lower()
        asset_text = (
            " ".join(
                [
                    asset.name,
                    asset.description,
                    str(asset.metadata.get("catalog_id", "")),
                    str(asset.metadata.get("source_id", "")),
                ]
            )
            .lower()
            .replace("_", " ")
        )
        is_patient_surface = any(
            term in furniture_text
            for term in ("medical bed", "med bed", "hospital bed", "treatment bed")
        )
        if not is_patient_surface:
            return True, None

        forbidden = (
            "accessory",
            "computer",
            "device",
            "equipment",
            "machine",
            "monitor",
            "stand",
            "tape",
            "terminal",
            "tool",
        )
        matched = next((term for term in forbidden if term in asset_text), None)
        if matched is None:
            return True, None
        return (
            False,
            f"'{asset.name}' is classified as hard equipment ({matched}) and cannot "
            "be placed in a medical bed's patient-support zone. Place it on a "
            "bedside cabinet/equipment stand, or choose soft bedding instead.",
        )

    def _create_loop_error_response(
        self, method_name: str, attempt_count: int, _args: tuple, kwargs: dict
    ) -> str:
        """Create manipuland-specific error response for loop detection."""
        # Extract object_id or asset_id from kwargs/args if available.
        identifier = kwargs.get("object_id", kwargs.get("asset_id", ""))

        if method_name == "_remove_manipuland_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to remove '{identifier}' "
                f"{attempt_count} times.\n\n"
                f"Possible causes:\n"
                f"1. Object doesn't exist\n"
                f"2. Object was already removed\n"
                f"3. Wrong object ID\n\n"
                f"Recovery: Call get_current_scene_state() to see actual object IDs."
            )
        elif method_name == "_place_manipuland_on_surface_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to place '{identifier}' "
                f"{attempt_count} times with the same parameters.\n\n"
                f"This suggests placement is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Position is out of surface bounds\n"
                f"2. Asset doesn't exist\n"
                f"3. Invalid placement parameters\n\n"
                f"Recovery: Check surface bounds and available assets."
            )
        elif method_name == "_move_manipuland_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to move '{identifier}' "
                f"{attempt_count} times with the same parameters.\n\n"
                f"This suggests movement is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Position is out of surface bounds\n"
                f"2. Object doesn't exist\n"
                f"3. Already at target position\n\n"
                f"Recovery: Check surface bounds and object status."
            )
        elif method_name == "_create_stack_impl":
            asset_ids = kwargs.get("asset_ids", [])
            diagnostic_message = (
                f"Loop detected: You've tried to create the same stack "
                f"{attempt_count} times with {len(asset_ids)} assets.\n\n"
                f"This suggests stack creation is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Stack is unstable (physics simulation failing)\n"
                f"2. Stack height exceeds surface clearance\n"
                f"3. Invalid asset IDs in stack\n\n"
                f"Recovery: Check simulation feedback and try with fewer items "
                f"or different base objects."
            )
            # Return StackCreationResult for stack operations.
            stack_result = StackCreationResult(
                success=False,
                message=diagnostic_message,
                stack_object_id=None,
                stack_height=None,
                parent_surface_id=(
                    self._current_surface.surface_id if self._current_surface else ""
                ),
                num_items=len(asset_ids),
                error_type=ManipulandErrorType.LOOP_DETECTED,
            )
            return stack_result.to_json()
        elif method_name == "_create_pile_impl":
            asset_ids = kwargs.get("asset_ids", [])
            diagnostic_message = (
                f"Loop detected: You've tried to create the same pile "
                f"{attempt_count} times with {len(asset_ids)} assets.\n\n"
                f"This suggests pile creation is failing repeatedly.\n\n"
                f"Possible causes:\n"
                f"1. Objects falling off surface edge\n"
                f"2. Position too close to surface boundary\n"
                f"3. Invalid asset IDs in pile\n\n"
                f"Recovery: Move position toward center of surface or use fewer items."
            )
            # Return PileCreationResult for pile operations.
            pile_result = PileCreationResult(
                success=False,
                message=diagnostic_message,
                pile_object_id=None,
                parent_surface_id=(
                    self._current_surface.surface_id if self._current_surface else ""
                ),
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.LOOP_DETECTED,
            )
            return pile_result.to_json()
        elif method_name == "_resolve_penetrations_impl":
            diagnostic_message = (
                f"Loop detected: You've tried to resolve penetrations on the same "
                f"surface {attempt_count} times.\n\n"
                f"This suggests the solver cannot find a valid configuration.\n\n"
                f"Possible causes:\n"
                f"1. Too many objects for surface area\n"
                f"2. Objects too large to fit\n\n"
                f"Recovery: Remove some objects or use a larger surface."
            )
            return PenetrationResolutionResult(
                success=False,
                message=diagnostic_message,
                num_objects_considered=0,
                num_objects_moved=0,
                moved_object_ids=[],
                max_displacement_m=0.0,
                error_type=ManipulandErrorType.LOOP_DETECTED,
            ).to_json()
        else:
            diagnostic_message = (
                f"Loop detected in {method_name}: {attempt_count} attempts with same "
                f"parameters."
            )

        result = ManipulandOperationResult(
            success=False,
            message=diagnostic_message,
            error_type=ManipulandErrorType.LOOP_DETECTED,
            object_id=identifier if identifier else None,
        )

        return result.to_json()

    def _get_object_convex_hull_2d(
        self, geometry_path: Path, scale_factor: float = 1.0
    ) -> np.ndarray:
        """Extract 2D convex hull vertices from object mesh.

        Loads the object mesh and computes its 2D convex hull by projecting vertices
        to the XY plane. This provides an accurate footprint for placement validation.

        Args:
            geometry_path: Path to object GLB/OBJ file.
            scale_factor: Scale factor to apply to mesh vertices (default 1.0).

        Returns:
            Array of 2D vertices [(x, y), ...] in object-local frame (XY plane).
            Vertices are ordered counter-clockwise around the hull.

        Raises:
            ValueError: If mesh cannot be loaded (fail fast - manipulands must have mesh).
        """
        try:
            mesh = trimesh.load(geometry_path, force="mesh")
        except Exception as e:
            raise ValueError(
                f"Failed to load mesh from {geometry_path} for convex hull "
                f"computation: {e}. Manipulands must have valid geometry."
            )

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(
                f"Loaded geometry from {geometry_path} is not a mesh "
                f"(got {type(mesh)}). Manipulands must have valid mesh geometry."
            )

        # Apply scale factor to mesh vertices.
        if scale_factor != 1.0:
            mesh.vertices *= scale_factor

        # Project mesh vertices to XY plane (drop Z coordinate).
        mesh_xy_vertices = mesh.vertices[:, :2]

        # Compute 2D convex hull.
        try:
            hull = ConvexHull(mesh_xy_vertices)
        except QhullError as e:
            raise ValueError(
                f"Failed to compute convex hull for mesh from {geometry_path}: {e}. "
                f"Mesh may have degenerate geometry."
            )

        # Return hull vertices in counter-clockwise order.
        hull_vertices = mesh_xy_vertices[hull.vertices]
        return hull_vertices

    def _is_top_surface(self, surface_id: str) -> bool:
        """Check if the given surface is the highest (top) surface of its parent object.

        Top surfaces are the highest support surfaces of furniture pieces. They allow
        natural overlap (e.g., books extending over table edges) and skip strict
        boundary validation. Lower surfaces (shelves) require objects to fit entirely
        within their boundaries.

        Args:
            surface_id: ID of the surface to check.

        Returns:
            True if the surface is the highest surface of its parent furniture object.
        """
        # Get the parent object for this surface.
        parent_object = None
        for obj in self.scene.objects.values():
            for surface in obj.support_surfaces:
                if str(surface.surface_id) == surface_id:
                    parent_object = obj
                    break
            if parent_object:
                break

        if parent_object is None:
            console_logger.warning(
                f"Could not find parent object for surface {surface_id}, "
                f"treating as non-top surface (strict validation)"
            )
            return False

        # Find the highest surface by Z-coordinate in world frame.
        max_height = float("-inf")
        highest_surface_id = None
        for surface in parent_object.support_surfaces:
            surface_height = surface.transform.translation()[2]
            if surface_height > max_height:
                max_height = surface_height
                highest_surface_id = str(surface.surface_id)

        return surface_id == highest_surface_id

    def _validate_convex_hull_footprint(
        self,
        target_surface: SupportSurface,
        geometry_path: Path,
        position_2d: np.ndarray,
        rotation_degrees: float,
        allow_overlap_ratio: float = 0.0,
        scale_factor: float = 1.0,
    ) -> tuple[bool, str | None]:
        """Validate that object's convex hull fits within surface with optional overlap.

        Uses the object's actual mesh convex hull for accurate validation (not the
        conservative bounding box). For overlap tolerance, the convex hull is first
        centered at the origin, then shrunk toward the origin before checking
        containment. This ensures correct validation regardless of mesh centering.

        For top surfaces, an overlap tolerance can be specified to allow natural
        overhang (e.g., books extending slightly over table edges). The tolerance
        is relative to the object's size.

        Args:
            target_surface: Surface to validate against.
            geometry_path: Path to object mesh.
            position_2d: Placement position in surface frame [x, y].
            rotation_degrees: Placement rotation in degrees.
            allow_overlap_ratio: Ratio by which to shrink the convex hull
                (0.0 = no shrinking/strict containment, 0.15 = shrink by 15%).
            scale_factor: Scale factor to apply to mesh vertices (default 1.0).

        Returns:
            Tuple of (is_valid, error_message):
            - is_valid: True if shrunk hull vertices are within surface boundary.
            - error_message: Descriptive error if validation fails, None otherwise.
        """
        # Get object convex hull vertices.
        hull_vertices = self._get_object_convex_hull_2d(
            geometry_path=geometry_path, scale_factor=scale_factor
        )

        # Compute hull centroid and center the hull at origin.
        # This ensures shrinking works correctly even if mesh is not perfectly
        # centered. When we place an object at position (x, y), we expect the
        # object's geometric center to be at (x, y), not its mesh origin.
        hull_centroid = hull_vertices.mean(axis=0)
        hull_vertices_centered = hull_vertices - hull_centroid  # Center at (0, 0).

        # Shrink centered hull toward origin by the overlap ratio.
        shrink_factor = 1.0 - allow_overlap_ratio
        shrunk_hull_vertices = shrink_factor * hull_vertices_centered

        # Convert rotation to radians.
        rotation_radians = np.deg2rad(rotation_degrees)

        # Build 2D rotation matrix.
        cos_theta = np.cos(rotation_radians)
        sin_theta = np.sin(rotation_radians)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Check each shrunk hull vertex.
        for i, vertex in enumerate(shrunk_hull_vertices):
            # Apply rotation.
            rotated_vertex = rotation_matrix @ vertex

            # Translate to placement position.
            transformed_vertex = rotated_vertex + position_2d

            # Check if vertex is within surface boundary.
            if not target_surface.contains_point_2d(transformed_vertex):
                allowed_percent = allow_overlap_ratio * 100
                return (
                    False,
                    f"Object convex hull extends beyond surface boundary by more than "
                    f"{allowed_percent:.1f}% of object size. "
                    f"Shrunk hull vertex {i+1} at "
                    f"({transformed_vertex[0]:.3f}, {transformed_vertex[1]:.3f}) "
                    f"is outside surface {target_surface.surface_id}.\n\n"
                    f"Try:\n"
                    f"- Moving object toward the center of the surface\n"
                    f"- Using a smaller object that fits within the surface\n"
                    f"- Placing on a different surface",
                )

        return (True, None)
