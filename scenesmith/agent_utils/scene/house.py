"""House layout and room geometry data structures."""

import json
import logging
import time
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omegaconf import DictConfig

from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.structure.geometry_models.common import SCHEMA_VERSION
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    HeightfieldSpec,
    LevelSpec,
    PlatformSpec,
    default_ground_level,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorSpec,
    PortalSpec,
)
from scenesmith.utils.geometry.material import Material
from scenesmith.utils.package_utils import create_package_xml

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.room import RoomScene
    from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType

console_logger = logging.getLogger(__name__)

from scenesmith.agent_utils.scene.house_parts.layout_compilation import (
    HouseLayoutCompilationMixin,
)
from scenesmith.agent_utils.scene.house_parts.layout_directives import (
    HouseLayoutDirectiveMixin,
)
from scenesmith.agent_utils.scene.house_parts.layout_persistence import (
    HouseLayoutPersistenceMixin,
)
from scenesmith.agent_utils.scene.house_parts.layout_state import HouseLayoutStateMixin
from scenesmith.agent_utils.scene.house_parts.layout_topology import (
    HouseLayoutTopologyMixin,
)
from scenesmith.agent_utils.scene.house_parts.openings import (
    Door,
    PlacedRoom,
    RoomMaterials,
    Window,
)
from scenesmith.agent_utils.scene.house_parts.room_geometry import RoomGeometry
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec


@dataclass
class HouseLayout(
    HouseLayoutTopologyMixin,
    HouseLayoutCompilationMixin,
    HouseLayoutStateMixin,
    HouseLayoutDirectiveMixin,
    HouseLayoutPersistenceMixin,
):
    """Layout specification for a house with one or more rooms.

    HouseLayout is the unified data structure for both room mode (single room)
    and house mode (multiple rooms). Room mode is simply a HouseLayout with
    one room. This eliminates separate code paths for the two modes.

    The floor plan generator receives a HouseLayout and populates the
    room_geometries dict with generated geometry for each room. Following stage agents
    don't interact with HouseLayout directly - they receive RoomScene instances
    with RoomGeometry.
    """

    schema_version: int = SCHEMA_VERSION
    """Serialized semantic schema version (v1 inputs migrate to v2)."""

    wall_height: float = 2.5
    """Wall height in meters (default 2.5m, agent can override via set_wall_height)."""

    house_prompt: str = ""
    """Original user prompt for the house/room."""

    room_specs: list[RoomSpec] = field(default_factory=list)
    """Specifications for each room in the house."""

    levels: list[LevelSpec] = field(default_factory=lambda: [default_ground_level()])
    """Vertical datums. Legacy layouts contain one zero-elevation ground level."""

    connectors: list[ConnectorSpec] = field(default_factory=list)
    """Stairs, ramps, ladders, lifts, shafts, and natural passages."""

    connector_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled connector SDF paths keyed by connector ID."""

    structural_meshes: list[StructuralMeshSpec] = field(default_factory=list)
    """Imported/freeform structural meshes, including cavern chambers/tunnels."""

    structural_mesh_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled structural-mesh SDF paths keyed by mesh ID."""

    semantic_environment: SemanticEnvironmentSpec | None = None
    """LLM-authored chambers and passage graphs compiled as navigable voids."""

    semantic_environment_geometry_path: Path | None = None
    """Compiled SDF path for the semantic environment shell."""

    semantic_environment_source_hash: str | None = None
    """Semantic content hash used to compile the current environment shell."""

    semantic_detail_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled SDF paths for semantic detail fields and hero features."""

    semantic_detail_source_hash: str | None = None
    """Semantic content hash used to compile the current detail products."""

    platforms: list[PlatformSpec] = field(default_factory=list)
    """Raised/sunken platforms, mezzanines, balconies, bridges, and catwalks."""

    platform_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled platform SDF paths keyed by platform ID."""

    heightfields: list[HeightfieldSpec] = field(default_factory=list)
    """Sampled terrain and organic floor surfaces."""

    heightfield_geometry_paths: dict[str, Path] = field(default_factory=dict)
    """Compiled heightfield SDF paths keyed by heightfield ID."""

    portals: list[PortalSpec] = field(default_factory=list)
    """General apertures; legacy doors/windows remain available during migration."""

    room_geometries: dict[str, RoomGeometry] = field(default_factory=dict)
    """Generated room geometry for each room (room_id -> RoomGeometry)."""

    house_dir: Path | None = None
    """Directory for house-level outputs."""

    # Placed rooms (derived from specs via placement algorithm).
    placed_rooms: list[PlacedRoom] = field(default_factory=list)
    """Rooms with computed positions after placement algorithm."""

    # Doors and windows.
    doors: list[Door] = field(default_factory=list)
    """All doors in the house."""

    windows: list[Window] = field(default_factory=list)
    """All windows in the house."""

    # Materials per room (interior walls + floors).
    room_materials: dict[str, RoomMaterials] = field(default_factory=dict)
    """Materials for each room (room_id -> RoomMaterials)."""

    # Exterior shell material (consistent for entire house).
    exterior_material: Material | None = None
    """Exterior material (brick, siding, etc.) with PBR textures."""

    # Validation state.
    placement_valid: bool = False
    """True if room placement satisfies all constraints."""

    connectivity_valid: bool = False
    """True if all rooms are reachable from exterior via doors."""

    # ASCII boundary labels (generated dynamically).
    boundary_labels: dict[str, tuple[str, str | None, str | None]] = field(
        default_factory=dict
    )
    """Maps label (A, B, C...) to (room_a, room_b, direction).

    For interior walls: (room_a, room_b, None) - direction not needed.
    For exterior walls: (room_a, None, direction) - direction is wall facing (north, south, etc).
    """


@dataclass
class HouseScene:
    """Complete house scene: layout + populated rooms.

    Always use HouseScene as the top-level container. Room mode is just a
    HouseScene with a single room (room_id="main"). This unified model avoids
    code duplication between modes.

    HouseScene contains the HouseLayout (floor plan data) and populated
    RoomScene instances.
    """

    layout: HouseLayout
    """House layout containing room specs, geometry, and doors/windows."""

    rooms: dict[str, "RoomScene"] = field(default_factory=dict)
    """Dictionary mapping room_id to RoomScene instances."""

    @property
    def house_dir(self) -> Path:
        """Base directory for the house (from layout)."""
        if self.layout.house_dir is None:
            raise ValueError("HouseLayout.house_dir is not set")
        return self.layout.house_dir

    def _get_room_position(self, room_id: str) -> tuple[float, float]:
        """Get legacy XY room-center position from the full v2 transform."""

        x, y, _, _ = self._get_room_transform(room_id)
        return (x, y)

    def _get_room_transform(self, room_id: str) -> tuple[float, float, float, float]:
        """Get room center XYZ and yaw from layout.

        Room geometry is centered at origin, so we need the center position
        (not corner) when placing rooms in the combined directive.

        Args:
            room_id: Room ID to look up.

        Returns:
            (x, y, z, yaw) tuple. Returns identity if room is not found.
        """
        for placed in self.layout.placed_rooms:
            if placed.room_id == room_id:
                # Convert from corner to center position.
                center_x = placed.position[0] + placed.width / 2
                center_y = placed.position[1] + placed.depth / 2
                return (
                    center_x,
                    center_y,
                    self.layout.get_room_elevation(room_id),
                    placed.yaw,
                )
        # Default to origin for single room mode or if placement not done.
        return (0.0, 0.0, 0.0, 0.0)

    def add_room(self, room: "RoomScene") -> None:
        """Add a room to the house.

        Args:
            room: RoomScene to add. room.room_id must be unique within this house.

        Raises:
            ValueError: If a room with the same room_id already exists.
        """
        if room.room_id in self.rooms:
            raise ValueError(f"Room with id '{room.room_id}' already exists")
        self.rooms[room.room_id] = room

    def get_room(self, room_id: str) -> "RoomScene | None":
        """Get a room by ID.

        Args:
            room_id: The room ID to look up.

        Returns:
            RoomScene if found, None otherwise.
        """
        return self.rooms.get(room_id)

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize HouseScene to dictionary for checkpointing.

        Returns:
            Dictionary containing complete house state including layout.
        """
        rooms_dict = {}
        for room_id, room in self.rooms.items():
            rooms_dict[room_id] = room.to_state_dict()

        return {
            "layout": self.layout.to_dict(scene_dir=self.house_dir),
            "rooms": rooms_dict,
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], house_dir: Path
    ) -> "HouseScene":
        """Create HouseScene from serialized dictionary.

        Args:
            state_dict: State dictionary from to_state_dict().
            house_dir: Base directory for the house (needed for path resolution).

        Returns:
            Restored HouseScene instance.
        """
        # Import here to avoid circular import.
        from scenesmith.agent_utils.scene.room import RoomScene

        # Restore layout.
        layout = HouseLayout.from_dict(state_dict["layout"], house_dir=house_dir)

        # Create HouseScene with restored layout.
        house_scene = cls(layout=layout)

        # Restore rooms.
        for room_id, room_data in state_dict["rooms"].items():
            room_dir = house_dir / f"room_{room_id}"
            room = RoomScene(
                room_geometry=None,  # Will be restored.
                scene_dir=room_dir,
                room_id=room_id,
            )
            room.restore_from_state_dict(room_data)
            house_scene.rooms[room_id] = room

        return house_scene

    def assemble(
        self,
        cfg: dict | DictConfig | None = None,
        output_name: str = "combined_house",
        include_object_types: "list[ObjectType] | None" = None,
    ) -> Path:
        """Assemble all rooms into combined house outputs.

        Creates the output directory with:
        - house.dmd.yaml: Drake directive with furniture as free bodies
          (only wall/ceiling-mounted objects welded)
        - house_furniture_welded.dmd.yaml: Drake directive with furniture welded
        - house_state.json: Combined state for all rooms
        - sceneeval_state.json: Combined SceneEval format
        - house.blend: Blender file for visualization (uses house.dmd.yaml)

        Single room is treated as a house with one room at identity transform.

        Note: Composite manipulands (stacks, piles) are always free bodies in
        both output files. This is only for final output - internal simulation
        still uses welded furniture and composites for physics.

        Args:
            cfg: Configuration (dict or OmegaConf). Required for blend export.
                If None, blend file will not be generated.
            output_name: Name of output directory (default: "combined_house").
                Use "combined_house_after_furniture" for intermediate saves.
            include_object_types: If provided, only include objects of these
                types in the output. Useful for intermediate snapshots.

        Returns:
            Path to the output directory.
        """
        combined_dir = self.house_dir / output_name
        combined_dir.mkdir(parents=True, exist_ok=True)

        # Generate house.dmd.yaml: furniture as free bodies, composites as free bodies.
        directive_free = self._generate_combined_directive(
            include_object_types=include_object_types,
            weld_furniture=False,
            weld_composite_members=False,
        )
        directive_path_free = combined_dir / "house.dmd.yaml"
        with open(directive_path_free, "w") as f:
            f.write(directive_free)
        console_logger.info(
            f"Saved Drake directive (furniture free): {directive_path_free}"
        )

        # Generate house_furniture_welded.dmd.yaml: furniture welded, composites free.
        directive_welded = self._generate_combined_directive(
            include_object_types=include_object_types,
            weld_furniture=True,
            weld_composite_members=False,
        )
        directive_path_welded = combined_dir / "house_furniture_welded.dmd.yaml"
        with open(directive_path_welded, "w") as f:
            f.write(directive_welded)
        console_logger.info(
            f"Saved Drake directive (furniture welded): {directive_path_welded}"
        )

        # Create package.xml for portability (only once per scene).
        package_xml_path = self.house_dir / "package.xml"
        if not package_xml_path.exists():
            create_package_xml(self.house_dir)
            console_logger.info(f"Created package.xml for scene portability")

        # Save combined house state.
        state_dict = self.to_state_dict()
        state_dict["timestamp"] = time.time()
        state_path = combined_dir / "house_state.json"
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)
        console_logger.info(f"Saved combined house state: {state_path}")

        # Export combined SceneEval format.
        # Imported lazily so layout serialization and validation do not require
        # the full Drake/scene runtime to be importable.
        from scenesmith.agent_utils.rendering.sceneeval_exporter import (
            SceneEvalExportConfig,
            SceneEvalExporter,
        )

        floor_thickness = cfg["floor_plan_agent"]["floor_thickness"] if cfg else 0.1
        config = SceneEvalExportConfig(floor_thickness=floor_thickness)
        SceneEvalExporter.export_house(
            house=self, output_dir=combined_dir, config=config
        )

        # Generate combined blend file.
        if cfg is not None:
            self._export_blend(output_dir=combined_dir, cfg=cfg)

        return combined_dir

    def _generate_combined_directive(
        self,
        include_object_types: "list[ObjectType] | None" = None,
        weld_furniture: bool = True,
        weld_composite_members: bool = True,
    ) -> str:
        """Generate Drake directive combining all rooms.

        Single room is just a house with one room at identity transform.
        Multi-room uses frames to position each room at its layout position.

        Args:
            include_object_types: If provided, only include objects of these
                types. Useful for intermediate snapshots.
            weld_furniture: If True (default), weld furniture to world frame.
                If False, furniture is added as free bodies.
            weld_composite_members: If True (default), weld composite manipuland
                members (stacks, piles) to their base. If False, all members
                are free bodies.

        Returns:
            Drake directive YAML string with package://scene/ URIs for portability.
        """
        directive = """directives:
- add_frame:
    name: house_frame
    X_PF:
      base_frame: world
      translation: [0, 0, 0]"""

        for room_id, room in self.rooms.items():
            geometry_name = f"room_geometry_{room_id}"
            room_frame_name = f"room_{room_id}_frame"

            # Get full room transform from the v2 layout.
            pos_x, pos_y, pos_z, yaw = self._get_room_transform(room_id)
            yaw_deg = yaw * 180.0 / np.pi

            # Add room frame as child of house_frame.
            directive += f"""
- add_frame:
    name: {room_frame_name}
    X_PF:
      base_frame: house_frame
      translation: [{pos_x}, {pos_y}, {pos_z}]
      rotation: !AngleAxis
        angle_deg: {yaw_deg}
        axis: [0, 0, 1]"""

            # Get room directive with parent_frame so all objects use
            # room-local coordinates relative to the room frame.
            room_directive = room.to_drake_directive(
                weld_room_geometry=False,
                room_geometry_name=geometry_name,
                model_name_prefix=f"{room_id}_",
                include_object_types=include_object_types,
                base_dir=self.house_dir,
                weld_furniture=weld_furniture,
                weld_stack_members=weld_composite_members,
                parent_frame=room_frame_name,
                include_additional_structural_geometry=False,
            )

            # Strip the "directives:" header.
            if room_directive.startswith("directives:"):
                room_directive = room_directive[len("directives:") :]
            directive += room_directive

            # Weld room geometry to room frame (no translation needed,
            # room geometry is centered at origin).
            directive += f"""
- add_weld:
    parent: {room_frame_name}
    child: {geometry_name}::room_geometry_body_link"""

        directive += self.layout._connector_drake_directives(base_dir=self.house_dir)
        directive += self.layout._structural_mesh_drake_directives(
            base_dir=self.house_dir
        )
        directive += self.layout._semantic_environment_drake_directive(
            base_dir=self.house_dir
        )
        directive += self.layout._semantic_detail_drake_directives(
            base_dir=self.house_dir
        )
        directive += self.layout._platform_drake_directives(base_dir=self.house_dir)
        directive += self.layout._heightfield_drake_directives(base_dir=self.house_dir)

        return directive

    def _room_floor_blender_visuals(self) -> list[dict[str, object]]:
        """Return PBR polygon-floor finishes omitted by Drake's glTF renderer.

        The compiled room OBJ remains the authoritative visual/collision shell.
        Polygon rooms add a named, visual-only GLB just above the floor so the
        selected PBR finish survives the combined Blender export.  As with
        platform GLBs, Drake does not forward this nested SDF visual, so import
        only the explicitly named finish in the owning room frame.
        """

        visuals: list[dict[str, object]] = []
        for room_id, geometry in self.layout.room_geometries.items():
            sdf_path = geometry.sdf_path
            for visual in ET.parse(sdf_path).findall(".//visual"):
                if "floor_finish" not in visual.get("name", ""):
                    continue
                visual_uri = visual.findtext("geometry/mesh/uri")
                if not visual_uri or Path(visual_uri).suffix.lower() not in {
                    ".glb",
                    ".gltf",
                }:
                    continue
                visual_path = Path(visual_uri)
                if not visual_path.is_absolute():
                    visual_path = sdf_path.parent / visual_path
                if not visual_path.is_file():
                    raise FileNotFoundError(
                        f"Compiled room floor visual does not exist: {visual_path}"
                    )
                x, y, z, yaw = self._get_room_transform(room_id)
                visuals.append(
                    {
                        "path": str(visual_path),
                        "translation": [x, y, z],
                        "yaw_radians": yaw,
                        "role": "structural_detail",
                        "source_id": f"{room_id}_floor_finish",
                    }
                )
        return visuals

    def _platform_blender_visuals(self) -> list[dict[str, object]]:
        """Return PBR platform visuals omitted by Drake's glTF renderer.

        Platform SDFs keep their collision OBJ as the authoritative physics
        geometry.  Drake's RenderEngineGltfClient does not currently forward a
        nested GLB visual referenced by an SDF, so the Blender export imports
        that authenticated compiled visual separately in the same room frame.
        """

        visuals: list[dict[str, object]] = []
        for platform in self.layout.platforms:
            sdf_path = self.layout.platform_geometry_paths.get(platform.platform_id)
            if sdf_path is None:
                raise ValueError(
                    f"Platform geometry has not been compiled for: {platform.platform_id}"
                )
            visual_uri = ET.parse(sdf_path).findtext(".//visual/geometry/mesh/uri")
            if not visual_uri or Path(visual_uri).suffix.lower() not in {
                ".glb",
                ".gltf",
            }:
                # OBJ-backed visuals are already forwarded by Drake and must
                # not be duplicated in the Blender presentation artifact.
                continue
            visual_path = Path(visual_uri)
            if not visual_path.is_absolute():
                visual_path = sdf_path.parent / visual_path
            if not visual_path.is_file():
                raise FileNotFoundError(
                    f"Compiled platform visual does not exist: {visual_path}"
                )
            x, y, z, yaw = self._get_room_transform(platform.space_id)
            visuals.append(
                {
                    "path": str(visual_path),
                    "translation": [x, y, z],
                    "yaw_radians": yaw,
                    "role": "structural_detail",
                    "source_id": platform.platform_id,
                }
            )
        return visuals

    def _architectural_blender_visuals(
        self, output_dir: Path
    ) -> list[dict[str, object]]:
        """Return deterministic style-specific, presentation-only structure."""

        from scenesmith.agent_utils.design.renaissance_dressing import (
            write_renaissance_dressing_visuals,
        )

        return write_renaissance_dressing_visuals(
            self.layout, output_dir / "architectural_dressing"
        )

    def _renaissance_bookcase_blender_visuals(
        self, output_dir: Path
    ) -> list[dict[str, object]]:
        """Return populated presentation shells for validated library wall runs."""

        from scenesmith.agent_utils.design.renaissance_dressing import (
            write_renaissance_bookcase_visuals,
        )

        return write_renaissance_bookcase_visuals(
            self.layout,
            self.rooms,
            output_dir,
        )

    def _export_blend(self, output_dir: Path, cfg: dict | DictConfig) -> None:
        """Export Blender file for all rooms to combined directory.

        Uses the welded visualization directive for both single and multi-room
        cases.  The free-body directive remains the simulation artifact, but
        Drake's Blender renderer does not apply ``default_free_body_pose`` to
        imported visual geometry.  Rendering that directive collapses every
        movable asset to its model origin.

        Args:
            output_dir: Directory to save house.blend.
            cfg: Configuration with rendering settings (dict or OmegaConf).
        """
        from scenesmith.agent_utils.rendering.pipeline.blend_export import (
            save_directive_as_blend,
        )

        directive_path = output_dir / "house_furniture_welded.dmd.yaml"
        if not directive_path.exists():
            console_logger.error(
                "Welded visualization directive not found, skipping house.blend"
            )
            return

        blend_output_path = output_dir / "house.blend"
        rendering_cfg = cfg["furniture_agent"]["rendering"]

        try:
            save_directive_as_blend(
                directive_path=directive_path,
                output_path=blend_output_path,
                blender_server_host=rendering_cfg["blender_server_host"],
                blender_server_port_range=tuple(
                    rendering_cfg["blender_server_port_range"]
                ),
                server_startup_delay=rendering_cfg["server_startup_delay"],
                port_cleanup_delay=rendering_cfg["port_cleanup_delay"],
                scene_dir=self.house_dir,
                additional_visuals=[
                    *self._room_floor_blender_visuals(),
                    *self._platform_blender_visuals(),
                    *self._architectural_blender_visuals(output_dir),
                    *self._renaissance_bookcase_blender_visuals(
                        output_dir / "bookcase_dressing"
                    ),
                ],
            )
            console_logger.info(f"Saved combined blend file: {blend_output_path}")
        except Exception as e:
            console_logger.error(f"Failed to export combined .blend file: {e}")
