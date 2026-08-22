"""House layout and room geometry data structures."""

import json
import logging
import os

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
)

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)


class HouseLayoutDirectiveMixin:
    """Drake directives for rooms and compiled structural artifacts."""

    def to_drake_directive(self, base_dir: Path | None = None) -> str:
        """Generate a Drake directive string for all room geometries.

        Creates a directive that includes all room geometry SDFs, with a
        house_frame at the root and room frames as children. Each room
        geometry is welded to its room frame.

        Args:
            base_dir: If provided, SDF paths are relative to this directory
                (for portable directives). The directive YAML file should be
                saved in this directory for Drake to resolve paths correctly.
                If None, absolute paths with file:// scheme are used.

        Returns:
            Drake directive in YAML format.

        Raises:
            ValueError: If no room geometries have been generated.
        """
        if (
            not self.room_geometries
            and not self.structural_meshes
            and self.semantic_environment is None
            and not self.platforms
            and not self.heightfields
        ):
            raise ValueError(
                "No room or freeform structural geometries have been defined. "
                "Generate or compile structural geometry first."
            )

        def format_sdf_path(sdf_path: Path | str | None) -> str:
            """Format SDF path as package:// URI or absolute file:// URI."""
            if sdf_path is None:
                return ""
            sdf_path = Path(sdf_path)
            if base_dir is not None:
                # Use package://scene/ for portable scenes.
                # Drake resolves this via PackageMap (set ROS_PACKAGE_PATH or
                # call parser.package_map().Add("scene", scene_dir)).
                rel_path = os.path.relpath(sdf_path, base_dir)
                return f"package://scene/{rel_path}"
            else:
                return f"file://{sdf_path.absolute()}"

        # Build lookup from room_id to PlacedRoom for positions.
        placed_room_lookup = {room.room_id: room for room in self.placed_rooms}

        directive = """directives:
- add_frame:
    name: house_frame
    X_PF:
      base_frame: world
      translation: [0, 0, 0]"""

        for room_id, room_geometry in self.room_geometries.items():
            # Get room position from placed_rooms (not room_specs).
            placed_room = placed_room_lookup.get(room_id)
            if placed_room is None:
                console_logger.warning(
                    f"Room '{room_id}' not found in placed_rooms, skipping"
                )
                continue

            # PlacedRoom.position is (x, y) of min corner.
            # Room geometry is centered at origin, so translate to room center.
            room_center_x = placed_room.position[0] + placed_room.width / 2
            room_center_y = placed_room.position[1] + placed_room.depth / 2
            room_center_z = self.get_room_elevation(room_id)
            room_yaw_deg = placed_room.yaw * 180.0 / np.pi

            room_frame_name = f"room_{room_id}_frame"
            model_name = f"room_geometry_{room_id}"
            room_geom_path = format_sdf_path(room_geometry.sdf_path)

            # Add room frame as child of house_frame.
            directive += f"""
- add_frame:
    name: {room_frame_name}
    X_PF:
      base_frame: house_frame
      translation: [{room_center_x}, {room_center_y}, {room_center_z}]
      rotation: !AngleAxis
        angle_deg: {room_yaw_deg}
        axis: [0, 0, 1]
- add_model:
    name: {model_name}
    file: {room_geom_path}
- add_weld:
    parent: {room_frame_name}
    child: {model_name}::room_geometry_body_link"""

        directive += self._connector_drake_directives(base_dir=base_dir)
        directive += self._structural_mesh_drake_directives(base_dir=base_dir)
        directive += self._semantic_environment_drake_directive(base_dir=base_dir)
        directive += self._semantic_detail_drake_directives(base_dir=base_dir)
        directive += self._platform_drake_directives(base_dir=base_dir)
        directive += self._heightfield_drake_directives(base_dir=base_dir)

        return directive

    def _connector_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for compiled structural connectors."""
        if not self.connectors:
            return ""
        missing = [
            connector.connector_id
            for connector in self.connectors
            if not self._connector_geometry_is_embedded(connector)
            if connector.connector_id not in self.connector_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Connector geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_connectors() before exporting the house."
            )

        directives = ""
        for connector in self.connectors:
            if self._connector_geometry_is_embedded(connector):
                continue
            sdf_path = self.connector_geometry_paths[connector.connector_id]
            if base_dir is None:
                formatted_path = f"file://{sdf_path.absolute()}"
            else:
                formatted_path = (
                    f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
                )
            model_name = f"structure_{connector.connector_id}"
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: house_frame
    child: {model_name}::structure_link"""
        return directives

    def _platform_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for platforms in room-local frames."""

        if not self.platforms:
            return ""
        missing = [
            platform.platform_id
            for platform in self.platforms
            if platform.platform_id not in self.platform_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Platform geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_platforms() before exporting the house."
            )
        directives = ""
        placed_ids = {room.room_id for room in self.placed_rooms}
        for platform in self.platforms:
            sdf_path = self.platform_geometry_paths[platform.platform_id]
            formatted_path = (
                f"file://{sdf_path.absolute()}"
                if base_dir is None
                else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
            )
            model_name = f"structure_{platform.platform_id}"
            parent_frame = (
                f"room_{platform.space_id}_frame"
                if platform.space_id in placed_ids
                else "house_frame"
            )
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::structure_link"""
        return directives

    def _structural_mesh_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for compiled freeform structures."""

        if not self.structural_meshes:
            return ""
        missing = [
            mesh.mesh_id
            for mesh in self.structural_meshes
            if mesh.mesh_id not in self.structural_mesh_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Structural mesh geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_structural_meshes() before exporting the house."
            )
        directives = ""
        placed_ids = {room.room_id for room in self.placed_rooms}
        for mesh in self.structural_meshes:
            if mesh.replaces_room_shell:
                # Its room-compatible SDF is already emitted by the standard
                # room directive and welded to the room frame there.
                continue
            sdf_path = self.structural_mesh_geometry_paths[mesh.mesh_id]
            formatted_path = (
                f"file://{sdf_path.absolute()}"
                if base_dir is None
                else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
            )
            model_name = f"structure_{mesh.mesh_id}"
            parent_frame = (
                f"room_{mesh.space_id}_frame"
                if mesh.space_id in placed_ids
                else "house_frame"
            )
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::structure_link"""
        return directives

    def _semantic_environment_drake_directive(
        self, base_dir: Path | None = None
    ) -> str:
        """Generate a house-frame directive for compiled semantic void geometry."""

        if self.semantic_environment is None:
            return ""
        if self.semantic_environment_geometry_path is None:
            raise ValueError(
                "Semantic environment geometry has not been compiled. "
                "Call compile_semantic_environment() before exporting the house."
            )
        if (
            self.semantic_environment_source_hash
            != self.semantic_environment.content_hash()
        ):
            raise ValueError(
                "Semantic environment geometry is stale. "
                "Call compile_semantic_environment() before exporting the house."
            )
        self._validate_semantic_artifact(
            self.semantic_environment_geometry_path,
            self.semantic_environment_source_hash,
            expected_compiler_version="semantic-environment-v2",
        )
        sdf_path = self.semantic_environment_geometry_path
        formatted_path = (
            f"file://{sdf_path.absolute()}"
            if base_dir is None
            else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
        )
        return f"""
- add_model:
    name: structure_semantic_environment
    file: {formatted_path}
- add_weld:
    parent: house_frame
    child: structure_semantic_environment::structure_link"""

    def _semantic_detail_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate house-frame directives for compiled semantic details."""

        if self.semantic_environment is None or not (
            self.semantic_environment.detail_fields
            or self.semantic_environment.hero_features
        ):
            return ""
        expected = {
            item.field_id for item in self.semantic_environment.detail_fields
        } | {item.feature_id for item in self.semantic_environment.hero_features}
        missing = expected - set(self.semantic_detail_geometry_paths)
        if missing:
            raise ValueError(
                "Semantic detail geometry has not been compiled for: "
                + ", ".join(sorted(missing))
                + ". Call compile_semantic_environment_details() before exporting."
            )
        if self.semantic_detail_source_hash != self.semantic_environment.content_hash():
            raise ValueError(
                "Semantic detail geometry is stale. Call "
                "compile_semantic_environment_details() before exporting."
            )
        directives = ""
        for detail_id in sorted(expected):
            sdf_path = self.semantic_detail_geometry_paths[detail_id]
            self._validate_semantic_artifact(
                sdf_path,
                self.semantic_detail_source_hash,
                expected_compiler_version="semantic-environment-details-v1",
            )
            formatted_path = (
                f"file://{sdf_path.absolute()}"
                if base_dir is None
                else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
            )
            directives += f"""
- add_model:
    name: environment_detail_{detail_id}
    file: {formatted_path}
- add_weld:
    parent: house_frame
    child: environment_detail_{detail_id}::structure_link"""
        return directives

    @staticmethod
    def _validate_semantic_artifact(
        sdf_path: Path,
        expected_source_hash: str,
        *,
        expected_compiler_version: str,
    ) -> None:
        """Reject missing, stale, or corrupted content-addressed products."""
        try:
            from scenesmith.agent_utils.structure.compiler.models import ArtifactRef

            sidecar_path = sdf_path.with_suffix(".surfaces.json")
            manifest = json.loads(sidecar_path.read_text(encoding="utf-8"))
            ArtifactRef(
                mesh_path=sidecar_path.parent / manifest["mesh"],
                sdf_path=sdf_path,
                surfaces_path=sidecar_path,
                collision_mesh_path=(
                    sidecar_path.parent / manifest["collision_mesh"]
                    if manifest.get("collision_mesh")
                    else None
                ),
            ).verify(
                expected_source_hash=expected_source_hash,
                expected_compiler_version=expected_compiler_version,
            )
        except (
            OSError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            GeometryValidationError,
        ) as exc:
            message = str(exc)
            if "source" in message:
                label = "source hash mismatch"
            elif "identity" in message or "compiler" in message:
                label = "artifact identity mismatch"
            elif "surface semantics" in message:
                label = "surface hash mismatch"
            elif "product hash" in message:
                label = "product hash mismatch"
            else:
                label = "artifact manifest is invalid"
            raise ValueError(f"Semantic {label}: {sdf_path}") from exc

    def _heightfield_drake_directives(self, base_dir: Path | None = None) -> str:
        """Generate model/weld directives for room-local heightfields."""

        if not self.heightfields:
            return ""
        missing = [
            heightfield.heightfield_id
            for heightfield in self.heightfields
            if heightfield.heightfield_id not in self.heightfield_geometry_paths
        ]
        if missing:
            raise ValueError(
                "Heightfield geometry has not been compiled for: "
                + ", ".join(missing)
                + ". Call compile_heightfields() before exporting the house."
            )
        directives = ""
        placed_ids = {room.room_id for room in self.placed_rooms}
        for heightfield in self.heightfields:
            sdf_path = self.heightfield_geometry_paths[heightfield.heightfield_id]
            formatted_path = (
                f"file://{sdf_path.absolute()}"
                if base_dir is None
                else f"package://scene/{os.path.relpath(sdf_path, base_dir)}"
            )
            model_name = f"structure_{heightfield.heightfield_id}"
            parent_frame = (
                f"room_{heightfield.space_id}_frame"
                if heightfield.space_id in placed_ids
                else "house_frame"
            )
            directives += f"""
- add_model:
    name: {model_name}
    file: {formatted_path}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::structure_link"""
        return directives
