import logging

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenesmith.agent_utils.scene.room import RoomScene

import numpy as np
import trimesh

from pydrake.all import RigidTransform, RollPitchYaw

from scenesmith.agent_utils.geometry.support_surfaces.articulated_extraction import (
    extract_support_surfaces_articulated,
)
from scenesmith.agent_utils.geometry.support_surfaces.mesh_extraction import (
    extract_support_surfaces_from_mesh,
)
from scenesmith.agent_utils.geometry.support_surfaces.models import (
    SupportSurfaceExtractionConfig,
)

console_logger = logging.getLogger(__name__)


from scenesmith.agent_utils.scene.room_parts.room_models import (
    SceneObject,
    SupportSurface,
    UniqueID,
)


def _catalog_aabb_support_surfaces(
    furniture_object: SceneObject,
    config: SupportSurfaceExtractionConfig,
) -> list[SupportSurface] | None:
    """Create one bounded support plane from canonical catalog metadata.

    Dense catalog display meshes can contain hundreds of thousands of faces;
    applying HSM face clustering to them is both unnecessary and unbounded. The
    catalog importer already canonicalizes the frame and records object bounds,
    which are sufficient for a conservative top support plane.
    """
    catalog_sources = {
        "articulated",
        "catalog",
        "generated",
        "hssd",
        "objathor",
        "objaverse",
        "polyhaven",
    }
    ontology_path = str(furniture_object.metadata.get("ontology_path") or "")
    intrinsic_shelving = any(
        term in ontology_path.casefold()
        for term in ("bookcase", "bookshelf", "shelving")
    )
    if (
        not config.use_catalog_aabb_fast_path
        or furniture_object.metadata.get("asset_source") not in catalog_sources
        or intrinsic_shelving
        or furniture_object.bbox_min is None
        or furniture_object.bbox_max is None
    ):
        return None

    # SceneObject bounds have already absorbed scale_factor. Convert back to the
    # unscaled object-local frame because propagation below applies scale once.
    scale = max(float(furniture_object.scale_factor), 1e-9)
    bounds_min = furniture_object.bbox_min.astype(float) / scale
    bounds_max = furniture_object.bbox_max.astype(float) / scale
    dimensions = bounds_max - bounds_min
    if dimensions[0] <= 0.05 or dimensions[1] <= 0.05 or dimensions[2] <= 0.05:
        return None

    searchable = f"{furniture_object.name} {furniture_object.description}".lower()
    if "bed" in searchable:
        surface_z = bounds_min[2] + dimensions[2] * config.bed_surface_height_ratio
    else:
        surface_z = bounds_max[2]

    inset = min(max(float(config.aabb_inset_ratio), 0.0), 0.4)
    half_width = dimensions[0] * (0.5 - inset)
    half_depth = dimensions[1] * (0.5 - inset)
    center_x = (bounds_min[0] + bounds_max[0]) / 2.0
    center_y = (bounds_min[1] + bounds_max[1]) / 2.0
    clearance = max(config.min_clearance_m, config.top_surface_clearance_m)

    return [
        SupportSurface(
            surface_id=UniqueID("surface_catalog_aabb"),
            bounding_box_min=np.array([-half_width, -half_depth, 0.0]),
            bounding_box_max=np.array([half_width, half_depth, clearance]),
            transform=RigidTransform(
                p=[center_x, center_y, surface_z + config.surface_offset_m]
            ),
            mesh=None,
        )
    ]


def extract_and_propagate_support_surfaces(
    scene: "RoomScene",
    furniture_object: SceneObject,
    config: SupportSurfaceExtractionConfig | None = None,
) -> list[SupportSurface]:
    """Extract all support surfaces using HSM algorithm and propagate to identical furniture.

    When furniture is duplicated, all instances share the same geometry but have
    different world poses. This function extracts support surfaces once using the
    HSM face clustering algorithm and propagates them to all furniture with the
    same geometry_path, saving computation.

    Args:
        scene: The scene containing all furniture objects.
        furniture_object: The furniture object to extract support surfaces from.
        config: HSM algorithm configuration (uses defaults if None).

    Returns:
        List of SupportSurface objects for the selected furniture, sorted by area
        (largest first).

    Raises:
        ValueError: If furniture object has no geometry path.
    """
    config = config or SupportSurfaceExtractionConfig()

    # Return existing support surfaces if already computed.
    if furniture_object.support_surfaces:
        console_logger.info(
            f"Support surfaces already extracted for {furniture_object.object_id} "
            f"({len(furniture_object.support_surfaces)} surfaces)"
        )
        return furniture_object.support_surfaces

    # Validate furniture has geometry.
    if furniture_object.geometry_path is None:
        raise ValueError(
            f"Furniture object {furniture_object.object_id} has no geometry path"
        )

    surfaces = _catalog_aabb_support_surfaces(furniture_object, config)
    hssd_mesh_id = ""
    if surfaces is not None:
        source = "canonical catalog AABB"
    else:
        hssd_mesh_id = str(furniture_object.metadata.get("hssd_mesh_id") or "")
        if not hssd_mesh_id:
            catalog_id = str(furniture_object.metadata.get("catalog_id") or "")
            if catalog_id.startswith("hssd__"):
                hssd_mesh_id = catalog_id.removeprefix("hssd__")

    # Check if HSSD asset with pre-validated surfaces.
    # Determine surface loading strategy and source.
    if surfaces is None and (
        furniture_object.metadata.get("asset_source") == "hssd"
        and hssd_mesh_id
        and not config.recompute_hssd_surfaces
    ):
        from scenesmith.agent_utils.hssd_retrieval.support_surface_loader import (
            load_hssd_support_surfaces,
        )

        surfaces = load_hssd_support_surfaces(
            mesh_id=hssd_mesh_id, config=config, scene=scene
        )
        source = "HSSD"

        if surfaces is None:
            # Fallback to HSM algorithm.
            console_logger.info(
                f"Falling back to HSM algorithm for {furniture_object.object_id}"
            )
            surfaces = extract_support_surfaces_from_mesh(
                mesh_path=furniture_object.geometry_path, config=config
            )
            source = "HSM (HSSD fallback)"
    elif surfaces is None:
        # Extract all surfaces using HSM algorithm.
        # For articulated objects with per-link meshes, use per-link extraction
        # for accurate link association.
        is_articulated = furniture_object.metadata.get("is_articulated", False)

        # Use sdf_path.parent for articulated objects (per-link meshes are there).
        if furniture_object.sdf_path:
            sdf_dir = furniture_object.sdf_path.parent
        else:
            sdf_dir = furniture_object.geometry_path.parent

        if is_articulated and furniture_object.sdf_path:
            # Use per-link extraction for articulated objects.
            surfaces = extract_support_surfaces_articulated(
                sdf_dir=sdf_dir, config=config, sdf_path=furniture_object.sdf_path
            )
            source = "HSM (per-link)"
        else:
            surfaces = extract_support_surfaces_from_mesh(
                mesh_path=furniture_object.geometry_path, config=config
            )
            source = "HSM"

        if (
            furniture_object.metadata.get("asset_source") == "hssd"
            and config.recompute_hssd_surfaces
        ):
            source = "HSM (HSSD recomputed)"

    # Log loaded surfaces with areas and clearances.
    if surfaces:
        surfaces_summary = ", ".join(
            [
                f"{surf.surface_id} (area={surf.area:.2f}m², "
                f"clearance={surf.bounding_box_max[2] - surf.bounding_box_min[2]:.2f}m)"
                for surf in surfaces
            ]
        )
        console_logger.info(
            f"Loaded {len(surfaces)} surfaces for {furniture_object.object_id} "
            f"(source: {source}): {surfaces_summary}"
        )

    # Transform surfaces from object-local to world frame.
    # The HSM algorithm extracts surfaces in mesh-local frame (identity transform).
    # We need to transform them to world frame using furniture's transform and scale.
    world_surfaces = []
    scale = furniture_object.scale_factor
    for surface in surfaces:
        # Scale surface translation to match the scaled collision geometry.
        scaled_translation = surface.transform.translation() * scale
        scaled_surface_transform = RigidTransform(
            surface.transform.rotation(), scaled_translation
        )
        world_transform = furniture_object.transform @ scaled_surface_transform

        # Scale bounding box to match scaled geometry.
        scaled_bbox_min = surface.bounding_box_min * scale
        scaled_bbox_max = surface.bounding_box_max * scale

        # Scale mesh vertices to match scaled collision geometry.
        scaled_mesh = None
        if surface.mesh is not None:
            scaled_vertices = surface.mesh.vertices * scale
            scaled_mesh = trimesh.Trimesh(
                vertices=scaled_vertices, faces=surface.mesh.faces
            )

        # Create new surface with world transform and short unique ID.
        # Use scene's generate_surface_id for base-36 sequential IDs (S_0, S_1, ...).
        world_surface = SupportSurface(
            surface_id=scene.generate_surface_id(),
            bounding_box_min=scaled_bbox_min,
            bounding_box_max=scaled_bbox_max,
            transform=world_transform,
            mesh=scaled_mesh,  # Scaled mesh for convex hull computation.
            link_name=surface.link_name,  # Preserve link for FK transforms.
        )
        world_surfaces.append(world_surface)

    furniture_object.support_surfaces = world_surfaces

    console_logger.info(
        f"Extracted {len(world_surfaces)} support surfaces for "
        f"{furniture_object.object_id}"
    )

    # Propagate to all identical furniture (same geometry_path).
    target_geometry_path = furniture_object.geometry_path

    for obj in scene.objects.values():
        # Skip the furniture we just processed.
        if obj.object_id == furniture_object.object_id:
            continue

        # Only propagate to identical furniture (same geometry file).
        if obj.geometry_path != target_geometry_path:
            continue

        # Transform each surface to this object's world frame with scaling.
        obj_surfaces = []
        obj_scale = obj.scale_factor
        for surface in surfaces:
            # Apply this object's scale_factor to surface translation.
            obj_scaled_translation = surface.transform.translation() * obj_scale
            obj_scaled_surface_transform = RigidTransform(
                surface.transform.rotation(), obj_scaled_translation
            )
            obj_world_transform = obj.transform @ obj_scaled_surface_transform

            # Scale bounding box to match this object's scaled geometry.
            obj_scaled_bbox_min = surface.bounding_box_min * obj_scale
            obj_scaled_bbox_max = surface.bounding_box_max * obj_scale

            # Scale mesh vertices to match this object's scale factor.
            obj_scaled_mesh = None
            if surface.mesh is not None:
                obj_scaled_vertices = surface.mesh.vertices * obj_scale
                obj_scaled_mesh = trimesh.Trimesh(
                    vertices=obj_scaled_vertices, faces=surface.mesh.faces
                )

            obj_surface = SupportSurface(
                surface_id=scene.generate_surface_id(),
                bounding_box_min=obj_scaled_bbox_min,
                bounding_box_max=obj_scaled_bbox_max,
                transform=obj_world_transform,
                mesh=obj_scaled_mesh,  # Scaled mesh for convex hull computation.
                link_name=surface.link_name,  # Preserve link for FK transforms.
            )
            obj_surfaces.append(obj_surface)

        obj.support_surfaces = obj_surfaces

        console_logger.info(
            f"Propagated {len(obj_surfaces)} support surfaces from "
            f"{furniture_object.object_id} to {obj.object_id}"
        )

    return world_surfaces


def copy_scene_object_with_new_pose(
    scene: "RoomScene",
    original: SceneObject,
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> SceneObject:
    """
    Create a copy of a SceneObject with a new pose (position + rotation) and unique ID.

    Generates a guaranteed-unique ID using the scene's sequential numbering system
    (e.g., first "chair", second "chair_2", 11th "chair_a").

    Args:
        scene: Scene instance for generating unique IDs.
        original: Original SceneObject to copy.
        x: New X position.
        y: New Y position.
        z: New Z position.
        roll: Roll rotation in radians (default 0.0).
        pitch: Pitch rotation in radians (default 0.0).
        yaw: Yaw rotation in radians (default 0.0).

    Returns:
        New SceneObject with same asset data but new pose and unique ID.
    """
    # Create new transform with both position and rotation.
    new_transform = RigidTransform(
        rpy=RollPitchYaw(roll=roll, pitch=pitch, yaw=yaw), p=[x, y, z]
    )

    # Create new metadata.
    new_metadata = deepcopy(original.metadata)

    # Create new SceneObject with unique ID.
    return SceneObject(
        object_id=scene.generate_unique_id(original.name),
        object_type=original.object_type,
        name=original.name,
        description=original.description,
        transform=new_transform,
        geometry_path=original.geometry_path,
        sdf_path=original.sdf_path,
        image_path=original.image_path,
        support_surfaces=original.support_surfaces,
        metadata=new_metadata,
        bbox_min=original.bbox_min,
        bbox_max=original.bbox_max,
        scale_factor=original.scale_factor,
    )
