import logging

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scenesmith.agent_utils.assets.asset_semantics import (
    tall_furniture_dimensions_are_compatible,
)
from scenesmith.agent_utils.geometry.mesh_canonicalization import canonicalize_mesh
from scenesmith.agent_utils.geometry.mesh_utils import (
    load_mesh_as_trimesh,
    remove_mesh_floaters,
    scale_mesh_uniformly_to_dimensions,
)
from scenesmith.agent_utils.geometry.sdf_generator import generate_drake_sdf
from scenesmith.agent_utils.physics.mesh_physics_analyzer import (
    MeshPhysicsAnalysis,
    analyze_mesh_orientation_and_material,
)
from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType
from scenesmith.agent_utils.structure.thin_covering_generator import (
    generate_thin_covering_sdf,
)

if TYPE_CHECKING:
    pass

console_logger = logging.getLogger(__name__)

HSSD_CANONICAL_CONVERSION_VERSION = 4

from scenesmith.agent_utils.assets.asset_models import AssetPathConfig


class SimulationAssetConversionMixin:
    """Canonical mesh and thin-covering conversion into simulation assets."""

    def _convert_mesh_to_simulation_asset(
        self,
        geometry_path: Path,
        config: AssetPathConfig,
        object_type: ObjectType,
        desired_dimensions: list[float] | None = None,
        asset_source: str = "generated",
        canonical_up: str | None = None,
        canonical_front: str | None = None,
    ) -> tuple[Path, Path, np.ndarray, np.ndarray, float]:
        """Convert mesh to a simulatable Drake SDF.

        Pipeline:
        - Convert GLB → Y-up GLTF (enables VLM analysis in Blender's Z-up space)
        - Remove mesh floaters from generated geometry (trusted catalog geometry
          may contain valid authored non-watertight components)
        - VLM analysis → orientation + material + mass (in Blender coords)
        - Canonicalize in Blender → rotate to canonical orientation + placement
          (Y-up GLTF input → Z-up GLTF output for Drake)
        - Scale to desired dimensions (if provided)
        - Collision → CoACD decomposition
        - SDF → Drake format with physics properties

        Multi-view images used for VLM physics analysis are saved to
        generated_assets/debug/<base_name>/ where <base_name> follows the pattern
        {sanitized_short_name}_{timestamp} (e.g., "office_desk_A_1759997032").

        Args:
            geometry_path: Path to raw GLB mesh from Hunyuan3D or HSSD.
            config: Asset path configuration.
            object_type: Type of object (determines placement strategy).
            desired_dimensions: Optional dimensions (width, depth, height) from agent.
            asset_source: Source of the asset ("generated" or "hssd"). HSSD assets
                use specialized VLM prompts and skip vertical views since they're
                already upright.

        Returns:
            Tuple of (sdf_path, final_gltf_path, bbox_min, bbox_max, scale_factor).
            The scale_factor is the uniform scaling applied during mesh scaling
            (1.0 if no scaling was applied). This is needed to correctly scale
            HSSD pre-computed support surfaces.
        """
        if self.collision_client is None:
            raise RuntimeError(
                "Collision client not available. Cannot generate collision geometry."
            )

        console_logger.info(
            f"Processing mesh ({geometry_path}) to simulation asset "
            f"(object_type={object_type.value})"
        )

        # Convert GLB to Y-up GLTF (enables VLM analysis in Blender's Z-up space).
        # Uses BlenderServer for crash isolation.
        gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
        self.blender_server.convert_glb_to_gltf(
            input_path=geometry_path,
            output_path=gltf_path,
            export_yup=True,
        )

        # Generated geometry can contain disconnected junk. Authored catalog meshes
        # commonly contain valid non-watertight components (shelves, doors, trim);
        # trimesh's watertight-only split would silently drop those components.
        if asset_source == "generated":
            console_logger.info("Removing disconnected mesh floaters")
            remove_mesh_floaters(
                mesh_path=gltf_path,
                output_path=gltf_path,
                distance_threshold=self.cfg.asset_manager.floater_distance_threshold,
            )
        else:
            console_logger.info(
                "Preserving authored catalog mesh components (source=%s)",
                asset_source,
            )

        # VLM analysis for orientation, material, mass.
        # Create debug directory for saving multi-view physics analysis images.
        # Use geometry_path stem to match asset naming pattern (e.g., "desk_A_1234567890").
        debug_dir = self.debug_dir / config.geometry_path.stem

        # Authored catalog assets already have a canonical source frame and are
        # deterministically reranked. Six-image VLM physics inference took 50-145s
        # per local asset, even though mass/friction only need stable estimates.
        # Keep visual inference for genuinely generated, untrusted meshes.
        is_hssd = asset_source == "hssd"
        if asset_source != "generated":
            physics_analysis = self._deterministic_catalog_physics(
                description=config.description,
                desired_dimensions=desired_dimensions,
                object_type=object_type,
                canonical_up=canonical_up,
                canonical_front=canonical_front,
            )
            console_logger.info(
                "Using deterministic catalog physics: source=%s, up=%s, "
                "front=%s, material=%s, mass=%.2fkg",
                asset_source,
                physics_analysis.up_axis,
                physics_analysis.front_axis,
                physics_analysis.material,
                physics_analysis.mass_kg,
            )
        else:
            console_logger.info(
                "Running VLM analysis for generated mesh physics " "(asset_source=%s)",
                asset_source,
            )
            physics_analysis = analyze_mesh_orientation_and_material(
                mesh_path=gltf_path,
                vlm_service=self.vlm_service,
                cfg=self.cfg,
                elevation_degrees=self.side_view_elevation_degrees,
                blender_server=self.blender_server,
                num_side_views=self.num_side_views_for_physics_analysis,
                debug_output_dir=debug_dir,
                prompt_type="generated",
                include_vertical_views=True,
            )

        console_logger.info(
            f"VLM analysis complete: up={physics_analysis.up_axis}, "
            f"front={physics_analysis.front_axis}, material={physics_analysis.material}, "
            f"mass={physics_analysis.mass_kg}kg"
        )

        # HSSD catalog axes describe the source glTF/Habitat frame, while the
        # canonicalizer receives axes after Blender has imported that Y-up file.
        # Convert those authored axes exactly once. Other catalog sources use
        # Blender-frame deterministic defaults and must not be remapped here.
        canonical_up_axis, canonical_front_axis = self._canonical_axes_for_blender(
            is_hssd=is_hssd,
            analyzed_up=physics_analysis.up_axis,
            analyzed_front=physics_analysis.front_axis,
            authored_up=canonical_up,
            authored_front=canonical_front,
        )
        if is_hssd and (canonical_up is not None or canonical_front is not None):
            console_logger.info(
                "Converted HSSD source axes to Blender frame: up=%s, front=%s",
                canonical_up_axis,
                canonical_front_axis,
            )

        # Canonicalize mesh in Blender (rotate to canonical orientation + placement).
        # Input: Y-up GLTF, Output: Z-up GLTF for Drake.
        canonical_path = config.sdf_dir / f"{config.short_name}_canonical.gltf"
        canonicalize_mesh(
            gltf_path=gltf_path,
            output_path=canonical_path,
            up_axis=canonical_up_axis,
            front_axis=canonical_front_axis,
            blender_server=self.blender_server,
            object_type=object_type,
        )

        # Scale mesh to desired dimensions (if provided).
        # For generated assets: scale_factor=1.0 because support surface extraction runs
        # on the already-scaled mesh, so surfaces are at correct dimensions.
        # For HSSD assets: scale_factor=applied_scale because pre-computed surfaces
        # are at original HSSD dimensions and need scaling.
        final_gltf_path = canonical_path
        initial_scale = 1.0
        if desired_dimensions is not None:
            # Canonicalization exports every source through the same Y-up glTF
            # contract; express scene [width, depth, height] in that frame.
            mesh_target_dimensions = self._canonical_mesh_target_dimensions(
                desired_dimensions,
                is_hssd=is_hssd,
            )
            console_logger.info(
                "Scaling mesh to scene dimensions %s (glTF containing box %s)",
                desired_dimensions,
                mesh_target_dimensions,
            )
            final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
            final_gltf_path, applied_scale = scale_mesh_uniformly_to_dimensions(
                mesh_path=canonical_path,
                desired_dimensions=mesh_target_dimensions,
                output_path=final_gltf_path,
                min_dimension_meters=self.min_mesh_dimension_meters,
                relative_threshold=self.mesh_relative_dimension_threshold,
                allow_wall_plane_quarter_turn=(object_type == ObjectType.WALL_MOUNTED),
            )
            # HSSD pre-computed surfaces are at original mesh dimensions.
            # They need scale_factor to match the physical scaling applied above.
            if is_hssd:
                initial_scale = applied_scale
        else:
            # Rename canonical to final name if no scaling needed.
            final_gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
            canonical_path.rename(final_gltf_path)

        # Generate collision geometry via convex decomposition server.
        collision_pieces = self._generate_collision_geometry(final_gltf_path)

        # Load mesh for bounding box calculation.
        mesh = load_mesh_as_trimesh(final_gltf_path, force_merge=True)

        # Generate Drake SDF.
        sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
        generate_drake_sdf(
            visual_mesh_path=final_gltf_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics_analysis,
            output_path=sdf_path,
            asset_name=config.short_name,
        )

        # Extract bounding box from scaled mesh.
        bounds = mesh.bounds
        bbox_min, bbox_max = self._canonical_bounds_to_drake(
            bounds,
            is_hssd=is_hssd,
        )

        compatible_dimensions, dimension_reason = (
            self._converted_dimensions_are_compatible(
                object_type=object_type,
                request_text=config.description,
                desired_dimensions=desired_dimensions,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            )
        )
        if not compatible_dimensions:
            raise ValueError(
                f"Converted asset '{config.description}' is incompatible with its "
                f"requested dimensions: {dimension_reason}"
            )

        console_logger.info(
            f"Drake SDF complete: SDF at {sdf_path}, bounds: {bbox_min} to {bbox_max}"
        )

        return sdf_path, final_gltf_path, bbox_min, bbox_max, initial_scale

    @staticmethod
    def _converted_dimensions_are_compatible(
        *,
        object_type: ObjectType,
        request_text: str = "",
        desired_dimensions: list[float] | None,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> tuple[bool, str]:
        """Validate the produced height contract for requested tall furniture."""

        if object_type != ObjectType.FURNITURE:
            return True, "object is not furniture"
        return tall_furniture_dimensions_are_compatible(
            request_text=request_text,
            desired_dimensions=desired_dimensions,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

    @staticmethod
    def _gltf_axis_to_blender(axis: str) -> str:
        """Map a signed source glTF axis into Blender's imported frame."""
        mapping = {
            "+X": "+X",
            "-X": "-X",
            "+Y": "+Z",
            "-Y": "-Z",
            "+Z": "-Y",
            "-Z": "+Y",
        }
        return mapping.get(axis.upper(), axis.upper())

    @classmethod
    def _canonical_axes_for_blender(
        cls,
        *,
        is_hssd: bool,
        analyzed_up: str,
        analyzed_front: str,
        authored_up: str | None,
        authored_front: str | None,
    ) -> tuple[str, str]:
        """Resolve catalog axes without remapping inferred Blender defaults."""

        if not is_hssd:
            return analyzed_up, analyzed_front
        return (
            (
                cls._gltf_axis_to_blender(analyzed_up)
                if authored_up is not None
                else analyzed_up
            ),
            (
                cls._gltf_axis_to_blender(analyzed_front)
                if authored_front is not None
                else analyzed_front
            ),
        )

    @staticmethod
    def _canonical_mesh_target_dimensions(
        desired_dimensions: list[float],
        *,
        is_hssd: bool,
    ) -> list[float]:
        """Express scene dimensions in the canonical mesh's coordinate frame."""
        return [
            desired_dimensions[0],
            desired_dimensions[2],
            desired_dimensions[1],
        ]

    @staticmethod
    def _canonical_bounds_to_drake(
        bounds: np.ndarray,
        *,
        is_hssd: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Express canonical mesh bounds in the scene/Drake Z-up frame."""
        # Y-up → Z-up transformation: (x, y, z) → (x, -z, y).
        bbox_min_yup = bounds[0]
        bbox_max_yup = bounds[1]
        bbox_min = np.array([bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]])
        bbox_max = np.array([bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]])
        return np.minimum(bbox_min, bbox_max), np.maximum(bbox_min, bbox_max)

    @staticmethod
    def _canonical_axis(value: str | None, default: str) -> str:
        """Normalize catalog vectors such as ``0,0,-1`` to signed axes."""
        if not value:
            return default
        normalized = value.strip().upper()
        if normalized in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            return normalized
        try:
            components = [float(part.strip()) for part in value.split(",")]
        except ValueError:
            return default
        if (
            len(components) != 3
            or max(abs(component) for component in components) < 1e-6
        ):
            return default
        index = max(range(3), key=lambda item: abs(components[item]))
        sign = "+" if components[index] >= 0 else "-"
        return f"{sign}{'XYZ'[index]}"

    @classmethod
    def _deterministic_catalog_physics(
        cls,
        *,
        description: str,
        desired_dimensions: list[float] | None,
        object_type: ObjectType,
        canonical_up: str | None,
        canonical_front: str | None,
    ) -> MeshPhysicsAnalysis:
        """Derive stable simulation properties without a model or render pass."""
        normalized = description.casefold()
        material_terms = (
            ("metal", ("metal", "steel", "aluminum", "chrome", "iron")),
            ("glass", ("glass", "crystal")),
            ("plastic", ("plastic", "polymer", "acrylic")),
            ("fabric", ("fabric", "upholstered", "cloth", "velvet", "linen")),
            ("wood", ("wood", "oak", "walnut", "timber", "plywood")),
        )
        material = next(
            (
                name
                for name, terms in material_terms
                if any(term in normalized for term in terms)
            ),
            "wood" if object_type == ObjectType.FURNITURE else "plastic",
        )

        volume = float(np.prod(desired_dimensions or [0.3, 0.3, 0.3]))
        density_by_type = {
            ObjectType.FURNITURE: 60.0,
            ObjectType.MANIPULAND: 180.0,
            ObjectType.WALL_MOUNTED: 40.0,
            ObjectType.CEILING_MOUNTED: 35.0,
        }
        bounds_by_type = {
            ObjectType.FURNITURE: (2.0, 200.0),
            ObjectType.MANIPULAND: (0.05, 15.0),
            ObjectType.WALL_MOUNTED: (0.1, 30.0),
            ObjectType.CEILING_MOUNTED: (0.2, 40.0),
        }
        density = density_by_type.get(object_type, 60.0)
        lower, upper = bounds_by_type.get(object_type, (0.1, 200.0))
        mass = min(upper, max(lower, volume * density))

        return MeshPhysicsAnalysis(
            up_axis=cls._canonical_axis(canonical_up, "+Z"),
            front_axis=cls._canonical_axis(canonical_front, "+Y"),
            material=material,
            mass_kg=mass,
            mass_range_kg=(mass * 0.6, mass * 1.4),
        )

    def _convert_thin_covering_to_simulation_asset(
        self,
        geometry_path: Path,
        config: AssetPathConfig,
        collision_dims: tuple[float, float, float] | None = None,
        collision_shape: str = "rectangular",
    ) -> tuple[Path, Path, np.ndarray, np.ndarray]:
        """Convert thin covering mesh to Drake SDF (simplified pipeline).

        Thin coverings are static decorative objects that don't require:
        - VLM orientation analysis (already correctly oriented)
        - Canonicalization (already in correct pose)
        - Collision geometry for floor/manipuland coverings (decorative only)

        Wall thin coverings (paintings, posters) DO get collision geometry so
        Drake can detect furniture collisions.

        Pipeline:
        - Convert GLB → GLTF with separate textures (for Drake)
        - Generate static SDF (with optional collision for wall coverings)
        - Compute bounding box from mesh

        Args:
            geometry_path: Path to thin covering GLB file.
            config: Asset path configuration.
            collision_dims: Optional (width, depth, height) for collision geometry.
                Used for wall thin coverings.
            collision_shape: Shape of collision ("rectangular" or "circular").

        Returns:
            Tuple of (sdf_path, final_gltf_path, bbox_min, bbox_max).
        """
        console_logger.info(f"Processing thin covering ({geometry_path}) to static SDF")

        # Convert GLB to GLTF with separate textures for Drake.
        # Uses BlenderServer for crash isolation.
        gltf_path = config.sdf_dir / f"{config.short_name}.gltf"
        self.blender_server.convert_glb_to_gltf(
            input_path=geometry_path,
            output_path=gltf_path,
            export_yup=True,
        )

        # Generate static SDF (with optional collision geometry for wall coverings).
        sdf_path = config.sdf_dir / f"{config.short_name}.sdf"
        generate_thin_covering_sdf(
            visual_mesh_path=gltf_path,
            output_path=sdf_path,
            model_name=config.short_name,
            collision_dims=collision_dims,
            collision_shape=collision_shape,
        )

        # Load mesh for bounding box calculation.
        mesh = load_mesh_as_trimesh(gltf_path, force_merge=True)
        bounds = mesh.bounds  # In Y-up coordinates (GLTF native format).

        # Transform from Y-up (GLTF) to Z-up (Drake) coordinate system.
        bbox_min_yup = bounds[0]
        bbox_max_yup = bounds[1]

        # Apply coordinate transformation: (x, y, z)_Yup → (x, -z, y)_Zup
        bbox_min = np.array([bbox_min_yup[0], -bbox_min_yup[2], bbox_min_yup[1]])
        bbox_max = np.array([bbox_max_yup[0], -bbox_max_yup[2], bbox_max_yup[1]])

        # Ensure min < max after transformation.
        bbox_min, bbox_max = (
            np.minimum(bbox_min, bbox_max),
            np.maximum(bbox_min, bbox_max),
        )

        console_logger.info(
            f"Thin covering SDF complete: {sdf_path}, bounds: {bbox_min} to {bbox_max}"
        )

        return sdf_path, gltf_path, bbox_min, bbox_max
