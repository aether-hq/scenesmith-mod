"""House layout and room geometry data structures."""

import logging
import math

from pathlib import Path
from typing import TYPE_CHECKING

from scenesmith.agent_utils.structure.geometry_models.common import SCHEMA_VERSION
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    LevelSpec,
    Transform3D,
    default_ground_level,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorSpec,
    ConnectorType,
    validate_structural_references,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    validate_global_identifiers,
)
from scenesmith.utils.package_utils import create_package_xml

if TYPE_CHECKING:
    from scenesmith.agent_utils.structure.compiler.models import CompiledStructurePaths

console_logger = logging.getLogger(__name__)


class HouseLayoutTopologyMixin:
    """Structural validation, topology, semantic compilation, and connectors."""

    def __post_init__(self) -> None:
        """Normalize v2 defaults and create package metadata when needed."""
        if not self.levels:
            self.levels = [default_ground_level()]
        if self.schema_version not in (1, SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported house layout schema_version={self.schema_version}; "
                f"supported versions are 1 and {SCHEMA_VERSION}"
            )
        if self.house_dir is not None:
            package_xml_path = self.house_dir / "package.xml"
            if not package_xml_path.exists():
                self.house_dir.mkdir(parents=True, exist_ok=True)
                create_package_xml(self.house_dir)
                console_logger.debug(
                    f"Created package.xml at {package_xml_path} for scene portability"
                )

    def get_level(self, level_id: str) -> LevelSpec | None:
        """Get a structural level by stable ID."""
        for level in self.levels:
            if level.level_id == level_id:
                return level
        return None

    def get_room_elevation(self, room_id: str) -> float:
        """Resolve a room's absolute floor Z using placement/spec/level precedence."""
        placed_room = self.get_placed_room(room_id)
        if placed_room is not None and placed_room.elevation is not None:
            return placed_room.elevation
        room_spec = self.get_room_spec(room_id)
        if room_spec is not None:
            if room_spec.elevation is not None:
                return room_spec.elevation
            level = self.get_level(room_spec.level_id)
            if level is not None:
                return level.elevation
        return 0.0

    def validate_structure(self) -> None:
        """Validate v2 level, room, portal, and connector references."""
        space_level_ids = {spec.room_id: spec.level_id for spec in self.room_specs}
        validate_structural_references(
            levels=self.levels,
            space_level_ids=space_level_ids,
            connectors=self.connectors,
            portals=self.portals,
            structural_meshes=self.structural_meshes,
            platforms=self.platforms,
            heightfields=self.heightfields,
        )
        if self.semantic_environment is not None:
            self.semantic_environment.validate_layout_bindings(
                space_level_ids=space_level_ids,
                level_ids=[level.level_id for level in self.levels],
            )
        # Wall openings are a compatibility projection of canonical Door and
        # Window records and may appear on both sides of a shared wall.  Count
        # each logical legacy opening once so aliases do not look like separate
        # scene entities, while still checking them against every other kind.
        canonical_opening_ids = {door.id for door in self.doors} | {
            window.id for window in self.windows
        }
        legacy_opening_ids = sorted(
            {
                opening.opening_id
                for room in self.placed_rooms
                for wall in room.walls
                for opening in wall.openings
                if opening.opening_id not in canonical_opening_ids
            }
        )
        primary_identifiers: list[tuple[str, str]] = [
            *((spec.room_id, "room") for spec in self.room_specs),
            *((level.level_id, "level") for level in self.levels),
            *((connector.connector_id, "connector") for connector in self.connectors),
            *((portal.portal_id, "portal") for portal in self.portals),
            *((mesh.mesh_id, "structural_mesh") for mesh in self.structural_meshes),
            *((platform.platform_id, "platform") for platform in self.platforms),
            *(
                (heightfield.heightfield_id, "heightfield")
                for heightfield in self.heightfields
            ),
            *((door.id, "door") for door in self.doors),
            *((window.id, "window") for window in self.windows),
            *((opening_id, "legacy_opening") for opening_id in legacy_opening_ids),
        ]
        if self.semantic_environment is not None:
            semantic = self.semantic_environment
            primary_identifiers.extend(
                [
                    *(
                        (item.region_id, "environment_region")
                        for item in semantic.regions
                    ),
                    *(
                        (item.chamber_id, "cavern_chamber")
                        for item in semantic.chambers
                    ),
                    *(
                        (item.network_id, "passage_network")
                        for item in semantic.passage_networks
                    ),
                    *(
                        (item.opening_id, "environment_opening")
                        for item in semantic.openings
                    ),
                    *(
                        (item.field_id, "detail_field")
                        for item in semantic.detail_fields
                    ),
                    *(
                        (item.feature_id, "hero_feature")
                        for item in semantic.hero_features
                    ),
                    *(
                        (item.junction_id, "passage_junction")
                        for network in semantic.passage_networks
                        for item in network.junctions
                    ),
                    *(
                        (item.segment_id, "passage_segment")
                        for network in semantic.passage_networks
                        for item in network.segments
                    ),
                    *(
                        (f"{item.field_id}_{index:04d}", "detail_instance")
                        for item in semantic.detail_fields
                        for index in range(item.count)
                    ),
                ]
            )
        validate_global_identifiers(primary_identifiers)
        replacement_spaces = [
            mesh.space_id for mesh in self.structural_meshes if mesh.replaces_room_shell
        ]
        duplicate_spaces = sorted(
            space_id
            for space_id in set(replacement_spaces)
            if replacement_spaces.count(space_id) > 1
        )
        if duplicate_spaces:
            raise GeometryValidationError(
                "duplicate_room_shell",
                "only one structural mesh may replace each room shell; duplicates: "
                + ", ".join(duplicate_spaces),
            )
        replacement_floor_spaces = [
            heightfield.space_id
            for heightfield in self.heightfields
            if heightfield.replaces_floor
        ]
        duplicate_floor_spaces = sorted(
            space_id
            for space_id in set(replacement_floor_spaces)
            if replacement_floor_spaces.count(space_id) > 1
        )
        if duplicate_floor_spaces:
            raise GeometryValidationError(
                "duplicate_floor_replacement",
                "only one heightfield may replace each room floor; duplicates: "
                + ", ".join(duplicate_floor_spaces),
            )

    def compile_semantic_environment(
        self,
        output_dir: Path,
        *,
        voxel_size: float = 0.5,
        max_cells: int = 2_000_000,
        max_triangles: int = 500_000,
        structure_id: str = "semantic_environment",
    ) -> "CompiledStructurePaths":
        """Compile authored chambers and passage networks into one SDF shell."""

        if self.semantic_environment is None:
            raise ValueError("No semantic environment has been defined.")
        from scenesmith.agent_utils.semantics.environment.semantic_environment_compiler import (
            SemanticCompileOptions,
            compile_semantic_environment,
        )
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )

        compiled = compile_semantic_environment(
            self.semantic_environment,
            options=SemanticCompileOptions(
                voxel_size=voxel_size,
                max_cells=max_cells,
                max_triangles=max_triangles,
                structure_id=structure_id,
            ),
        )
        source_hash = self.semantic_environment.content_hash()
        paths = write_compiled_structure(
            compiled,
            output_dir,
            source_content_hash=source_hash,
            compiler_version="semantic-environment-v2",
            compile_options={
                "max_cells": max_cells,
                "max_triangles": max_triangles,
                "structure_id": structure_id,
                "voxel_size": voxel_size,
            },
        )
        self.semantic_environment_geometry_path = paths.sdf_path
        self.semantic_environment_source_hash = source_hash
        return paths

    def derive_scene_contract(
        self,
        output_dir: Path,
        *,
        voxel_size: float = 0.5,
        max_cells: int = 2_000_000,
        max_triangles: int = 500_000,
    ):
        """Derive semantic geometry, collision, topology, and artifacts together.

        New callers should use this method instead of invoking the shell and
        detail compilers independently.  The older compile methods remain as
        compatibility shims for existing scene exporters.
        """

        from scenesmith.agent_utils.semantics.publication.scene_contract import (
            derive_scene_contract,
        )

        return derive_scene_contract(
            self,
            output_dir,
            voxel_size=voxel_size,
            max_cells=max_cells,
            max_triangles=max_triangles,
        )

    def compile_semantic_environment_details(self, output_dir: Path) -> dict[str, Path]:
        """Compile seeded geological details and hero features to SDF assets."""

        if self.semantic_environment is None:
            raise ValueError("No semantic environment has been defined.")
        from scenesmith.agent_utils.semantics.environment.semantic_environment_details import (
            compile_environment_details,
        )
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )

        source_hash = self.semantic_environment.content_hash()
        compiled = compile_environment_details(self.semantic_environment)
        paths: dict[str, Path] = {}
        for structure in compiled.structures:
            written = write_compiled_structure(
                structure,
                output_dir / structure.structure_id,
                source_content_hash=source_hash,
                compiler_version="semantic-environment-details-v1",
                compile_options={"sampler_version": 1},
            )
            paths[structure.structure_id] = written.sdf_path
        self.semantic_detail_geometry_paths = paths
        self.semantic_detail_source_hash = source_hash
        return dict(paths)

    def build_topology(self):
        """Build the capability-aware semantic topology for this layout."""

        from scenesmith.agent_utils.structure.structural_topology import (
            StructuralTopology,
        )

        self.validate_structure()
        return StructuralTopology.build(
            space_ids=self.room_ids,
            portals=self.portals,
            connectors=self.connectors,
        )

    def build_structural_surface_index(self, *, include_connectors: bool = True):
        """Build one house-frame query index from all compiled room surfaces."""

        from scenesmith.agent_utils.structure.structural_surfaces import (
            StructuralSurfaceIndex,
            load_surface_patches,
            transform_surface_patches,
        )

        patches = []
        for room_id, geometry in self.room_geometries.items():
            placed = self.get_placed_room(room_id)
            if placed is None:
                continue
            paths = {
                Path(path)
                for path in (
                    geometry.structural_surface_path,
                    *geometry.additional_structural_surface_paths,
                )
                if path is not None and Path(path).exists()
            }
            room_transform = Transform3D(
                translation=(
                    placed.position[0] + placed.width / 2.0,
                    placed.position[1] + placed.depth / 2.0,
                    self.get_room_elevation(room_id),
                ),
                rotation_rpy=(0.0, 0.0, placed.yaw),
            )
            for path in sorted(paths, key=str):
                patches.extend(
                    transform_surface_patches(
                        load_surface_patches(path), room_transform
                    )
                )
        if include_connectors:
            for connector_id, sdf_path in sorted(self.connector_geometry_paths.items()):
                surface_path = Path(sdf_path).with_suffix(".surfaces.json")
                if not surface_path.exists():
                    raise ValueError(
                        f"compiled connector '{connector_id}' is missing surface "
                        f"sidecar {surface_path}"
                    )
                patches.extend(load_surface_patches(surface_path))
            if self.semantic_environment_geometry_path is not None:
                surface_path = self.semantic_environment_geometry_path.with_suffix(
                    ".surfaces.json"
                )
                if not surface_path.exists():
                    raise ValueError(
                        "compiled semantic environment is missing surface sidecar "
                        f"{surface_path}"
                    )
                patches.extend(load_surface_patches(surface_path))
        return StructuralSurfaceIndex(patches)

    @staticmethod
    def _connector_centerline_samples(
        connector: ConnectorSpec, *, sample_spacing: float
    ) -> tuple[tuple[float, float, float], ...]:
        """Sample a connector centerline in the house structural frame."""

        if not math.isfinite(sample_spacing) or sample_spacing <= 0:
            raise ValueError("sample_spacing must be finite and positive")
        if connector.connector_type.value == "stairs_spiral":
            center = connector.parameters.get("center")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError(
                    f"spiral connector '{connector.connector_id}' has no center"
                )
            low, high = (
                (connector.start.position, connector.end.position)
                if connector.start.position[2] <= connector.end.position[2]
                else (connector.end.position, connector.start.position)
            )
            center_x, center_y = float(center[0]), float(center[1])
            radius = math.hypot(low[0] - center_x, low[1] - center_y)
            turns = float(connector.parameters.get("turns", 1.0))
            direction = (
                1.0
                if str(connector.parameters.get("direction", "ccw")).lower() == "ccw"
                else -1.0
            )
            total_angle = direction * math.tau * turns
            start_angle = math.atan2(low[1] - center_y, low[0] - center_x)
            length = math.hypot(radius * total_angle, high[2] - low[2])
            count = max(1, math.ceil(length / sample_spacing))
            return tuple(
                (
                    center_x + radius * math.cos(start_angle + total_angle * i / count),
                    center_y + radius * math.sin(start_angle + total_angle * i / count),
                    low[2] + (high[2] - low[2]) * i / count,
                )
                for i in range(count + 1)
            )

        raw_waypoints = connector.parameters.get("waypoints", ())
        waypoints = tuple(
            tuple(float(value) for value in point) for point in raw_waypoints
        )
        points = (connector.start.position, *waypoints, connector.end.position)
        samples: list[tuple[float, float, float]] = []
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            length = math.dist(start, end)
            count = max(1, math.ceil(length / sample_spacing))
            samples.extend(
                tuple(
                    start[axis] + (end[axis] - start[axis]) * i / count
                    for axis in range(3)
                )
                for i in range(0 if segment_index == 0 else 1, count + 1)
            )
        return tuple(samples)

    def geometrically_blocked_connectors(
        self,
        *,
        capabilities: tuple[str, ...] = ("walk",),
        agent_height: float = 1.8,
        agent_radius: float = 0.25,
        max_step_height: float = 0.3,
        sample_spacing: float = 0.15,
    ) -> frozenset[str]:
        """Veto semantically walkable connectors that fail local clearance."""

        missing_geometry = [
            connector.connector_id
            for connector in self.connectors
            if not self._connector_geometry_is_embedded(connector)
            and connector.connector_id not in self.connector_geometry_paths
        ]
        if missing_geometry:
            raise ValueError("compile_connectors() before checking route clearance")
        index = self.build_structural_surface_index(include_connectors=True)
        available = frozenset(capabilities)
        blocked: set[str] = set()
        for connector in self.connectors:
            if not connector.required_capabilities.issubset(available):
                continue
            # Climb-only ladders use a different body model; this local support
            # sampler is intentionally restricted to walkable connectors.
            if "walk" not in connector.required_capabilities:
                continue
            if connector.width / 2.0 + 1e-9 < agent_radius:
                blocked.add(connector.connector_id)
                continue
            for x, y, z in self._connector_centerline_samples(
                connector, sample_spacing=sample_spacing
            ):
                clearance = index.clearance_at(
                    x,
                    y,
                    agent_height=agent_height,
                    agent_radius=0.0,
                    reference_z=z + max_step_height,
                    max_drop=max_step_height * 1.5,
                )
                if not clearance.fits:
                    blocked.add(connector.connector_id)
                    break
        return frozenset(blocked)

    @staticmethod
    def _connector_geometry_is_embedded(connector: ConnectorSpec) -> bool:
        """Whether a connector centerline is embodied by imported room geometry.

        Natural passages and shafts frequently are not useful as additive mesh
        primitives: the tunnel/chimney is already part of a scanned or authored
        cavern shell.  Such connectors still participate in semantic topology and
        route-clearance checks, but must not produce a duplicate simulation model.
        """

        return (
            connector.connector_type
            in {
                ConnectorType.NATURAL_PASSAGE,
                ConnectorType.SHAFT,
            }
            and connector.parameters.get("geometry_embedded") is True
        )

    def compile_connectors(self, output_dir: Path) -> dict[str, Path]:
        """Compile all supported semantic connectors into simulation assets."""
        from scenesmith.agent_utils.structure.compiler.connector_dispatch import (
            compile_connector,
        )
        from scenesmith.agent_utils.structure.compiler.writing import (
            write_compiled_structure,
        )

        self.validate_structure()
        compiled_paths: dict[str, Path] = {}
        for connector in self.connectors:
            if self._connector_geometry_is_embedded(connector):
                continue
            connector_output = output_dir / connector.connector_id
            paths = write_compiled_structure(
                compile_connector(connector), connector_output
            )
            compiled_paths[connector.connector_id] = paths.sdf_path
        self.connector_geometry_paths = compiled_paths
        return dict(compiled_paths)
