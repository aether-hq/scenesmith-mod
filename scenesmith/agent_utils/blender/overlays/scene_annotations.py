"""Utilities for adding annotations to Blender scenes and rendered images."""

import logging

import bpy

from mathutils import Matrix, Vector
from omegaconf import DictConfig

console_logger = logging.getLogger(__name__)


def create_annotation_collection() -> bpy.types.Collection:
    """Create or get the annotations collection for organizing annotation objects.

    Returns:
        Blender collection for annotations.
    """
    collection_name = "SceneAnnotations"

    # Check if collection already exists.
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]

    # Create new collection.
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    return collection


def add_bounding_box_annotation(
    bbox: dict,
    rotation_matrix: list[list[float]],
    collection: bpy.types.Collection,
    object_name: str,
) -> None:
    """Add a 3D wireframe bounding box annotation.

    Args:
        bbox: Bounding box dict with 'center' and 'extents'.
        rotation_matrix: 3x3 rotation matrix as list of lists.
        collection: Collection to add annotation to.
        object_name: Name for the annotation object.
    """
    # Create cube primitive.
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    bbox_obj = bpy.context.active_object
    bbox_obj.name = f"bbox_{object_name}"

    # Set position and scale.
    center = Vector(bbox["center"])
    extents = bbox["extents"]
    bbox_obj.location = center
    bbox_obj.scale = (extents[0], extents[1], extents[2])

    # Apply rotation from object.
    rot_matrix_3x3 = Matrix(rotation_matrix)
    bbox_obj.rotation_euler = rot_matrix_3x3.to_euler()

    # Add wireframe modifier (SceneWeaver pattern) with thick lines for visibility.
    mod = bbox_obj.modifiers.new("Wireframe", type="WIREFRAME")
    mod.thickness = 0.05
    mod.use_even_offset = True

    # Set display type to wireframe.
    bbox_obj.display_type = "WIRE"

    # Create blue Principled BSDF material (SceneWeaver color).
    mat = bpy.data.materials.new(name=f"mat_bbox_{object_name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (
            0.0,
            0.3,
            1.0,
            1.0,
        )  # Match PIL label color exactly.
        # No emission to match flat PIL color appearance.
        bsdf.inputs["Emission Strength"].default_value = 0.0
    bbox_obj.data.materials.append(mat)

    # Link to annotations collection.
    if bbox_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(bbox_obj)
    collection.objects.link(bbox_obj)

    console_logger.debug(
        f"Created bbox: {bbox_obj.name}, location={bbox_obj.location}, "
        f"scale={bbox_obj.scale}, hide_render={bbox_obj.hide_render}, "
        f"in_collection={bbox_obj.name in collection.objects}"
    )


def add_set_of_mark_label_annotation(
    position: tuple[float, float, float],
    text: str,
    collection: bpy.types.Collection,
    object_name: str,
) -> None:
    """Add a 3D text label annotation with background plane.

    Args:
        position: (x, y, z) position for the label.
        text: Text to display.
        collection: Collection to add annotation to.
        object_name: Name for the annotation object.
    """
    # Create 3D text object.
    bpy.ops.object.text_add(location=position)
    text_obj = bpy.context.active_object
    text_obj.name = f"label_{object_name}"

    # Configure text properties (default alignment, not centered).
    text_obj.data.body = text

    # Scale text.
    scale = 0.15
    text_obj.scale = (scale, scale, scale)

    # Update view layer to get accurate text dimensions before positioning.
    bpy.context.view_layer.update()

    # Position text above bbox center with offset to center left-aligned text.
    text_offset = Vector((-text_obj.dimensions[0] * text_obj.scale[0] / 2, 0.1, 0.5))
    text_obj.location = Vector(position) + text_offset

    # Create bright white material for text with emission.
    text_mat = bpy.data.materials.new(name=f"mat_text_{object_name}")
    text_mat.use_nodes = True
    bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 5.0
    text_obj.data.materials.append(text_mat)

    # Create background plane.
    bpy.ops.mesh.primitive_plane_add(size=1)
    bg_plane = bpy.context.active_object
    bg_plane.name = f"label_bg_{object_name}"

    # Scale background based on text dimensions with padding.
    padding = 0.05
    text_size = text_obj.dimensions
    bg_plane.scale.x = (text_size.x + padding) / text_obj.scale[0]
    bg_plane.scale.y = (text_size.y + padding) / text_obj.scale[1]

    # Position background to center on left-aligned text.
    bg_plane.location.x = (text_size.x / 2) / text_obj.scale[0]
    bg_plane.location.y = (text_size.y / 2) / text_obj.scale[1]
    bg_plane.location.z = -0.01  # Slightly behind text.

    # Create blue material for background.
    bg_mat = bpy.data.materials.new(name=f"mat_label_bg_{object_name}")
    bg_mat.use_nodes = True
    bsdf = bg_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (
            0.0,
            0.3,
            1.0,
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = 1.0
    bg_plane.data.materials.append(bg_mat)

    # Parent background to text so they move together.
    bg_plane.parent = text_obj

    # Link both to annotations collection.
    if text_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(text_obj)
    collection.objects.link(text_obj)

    if bg_plane.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(bg_plane)
    collection.objects.link(bg_plane)

    console_logger.debug(
        f"Created label: {text_obj.name}, location={text_obj.location}, "
        f"text='{text_obj.data.body}', hide_render={text_obj.hide_render}, "
        f"in_collection={text_obj.name in collection.objects}"
    )
    console_logger.debug(
        f"Created label bg: {bg_plane.name}, location={bg_plane.location}, "
        f"hide_render={bg_plane.hide_render}, "
        f"in_collection={bg_plane.name in collection.objects}"
    )


def _calculate_arrow_dimensions(
    rendering_mode: str,
    y_direction: Vector,
    bbox_extents: tuple[float, float, float] | None,
) -> tuple[float, float, float, float]:
    """Calculate arrow dimensions based on rendering mode and bbox.

    Args:
        rendering_mode: Rendering mode - "furniture" or "manipuland".
        y_direction: Direction vector for the arrow (normalized Y-axis).
        bbox_extents: Optional bbox extents for furniture mode scaling.

    Returns:
        Tuple of (shaft_length, shaft_radius, head_length, head_radius).
    """
    if rendering_mode == "manipuland":
        # Smaller, thinner arrows for manipuland mode.
        return (0.05, 0.005, 0.02, 0.015)

    if bbox_extents:
        # Furniture mode with bbox: size based on object.
        # Compute bbox dimension along arrow direction (Y-axis).
        # For AABB, projection = sum of absolute dot products with each axis.
        projected_length = (
            abs(y_direction.x) * bbox_extents[0]
            + abs(y_direction.y) * bbox_extents[1]
            + abs(y_direction.z) * bbox_extents[2]
        )
        shaft_length = projected_length * 0.75
        return (shaft_length, 0.02, 0.1, 0.1)

    # Fallback to fixed size.
    return (0.6, 0.02, 0.1, 0.1)


def _create_arrow_geometry(
    shaft_length: float,
    shaft_radius: float,
    head_length: float,
    head_radius: float,
    object_name: str,
) -> bpy.types.Object:
    """Create arrow geometry from cylinder shaft and cone head.

    Args:
        shaft_length: Length of the arrow shaft.
        shaft_radius: Radius of the arrow shaft.
        head_length: Length of the arrow head (cone).
        head_radius: Radius of the arrow head base.
        object_name: Name for the arrow object.

    Returns:
        Combined arrow object (shaft + head joined).
    """
    # Create arrow shaft (cylinder).
    bpy.ops.mesh.primitive_cylinder_add(
        radius=shaft_radius,
        depth=shaft_length,
        location=(0, 0, 0),
    )
    shaft = bpy.context.active_object
    shaft.name = f"arrow_shaft_{object_name}"

    # Create arrowhead (cone).
    bpy.ops.mesh.primitive_cone_add(
        radius1=head_radius,
        radius2=0,
        depth=head_length,
        location=(0, 0, shaft_length / 2 + head_length / 2),
    )
    head = bpy.context.active_object
    head.name = f"arrow_head_{object_name}"

    # Select both and join.
    shaft.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join()

    # Combined arrow object.
    arrow_obj = bpy.context.active_object
    arrow_obj.name = f"arrow_{object_name}"
    return arrow_obj


def _create_cyan_emission_material(object_name: str) -> bpy.types.Material:
    """Create bright cyan material with strong emission for arrows.

    Args:
        object_name: Name for the material (used in naming).

    Returns:
        Configured cyan emission material.
    """
    arrow_mat = bpy.data.materials.new(name=f"mat_arrow_{object_name}")
    arrow_mat.use_nodes = True
    bsdf = arrow_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Color"].default_value = (0.0, 1.0, 1.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 10.0
    return arrow_mat


def add_direction_arrow_annotation(
    position: tuple[float, float, float],
    rotation_matrix: list[list[float]],
    collection: bpy.types.Collection,
    object_name: str,
    bbox_extents: tuple[float, float, float] | None = None,
    rendering_mode: str = "furniture",
) -> None:
    """Add a 3D direction arrow annotation showing Y-axis (forward) orientation.

    Args:
        position: (x, y, z) position for arrow base (bbox center).
        rotation_matrix: 3x3 rotation matrix as list of lists.
        collection: Collection to add annotation to.
        object_name: Name for the annotation object.
        bbox_extents: Optional bbox extents for positioning at top of bbox.
        rendering_mode: Rendering mode - "furniture" or "manipuland".
    """
    # Extract Y-axis direction from rotation matrix (Drake uses Y for "forward").
    rot_matrix = Matrix(rotation_matrix)
    y_direction = rot_matrix @ Vector((0, 1, 0))

    # Calculate arrow dimensions based on rendering mode.
    shaft_length, shaft_radius, head_length, head_radius = _calculate_arrow_dimensions(
        rendering_mode=rendering_mode,
        y_direction=y_direction,
        bbox_extents=bbox_extents,
    )

    # Create arrow geometry.
    arrow_obj = _create_arrow_geometry(
        shaft_length=shaft_length,
        shaft_radius=shaft_radius,
        head_length=head_length,
        head_radius=head_radius,
        object_name=object_name,
    )

    # Position at top of bbox.
    # Add half bbox height to Z coordinate to place arrow at top.
    arrow_location = Vector(position)
    if bbox_extents:
        arrow_location.z += bbox_extents[2] / 2

    # Offset arrow so it starts at bbox center (not centered there).
    # Move arrow forward by half its shaft length.
    arrow_obj.location = arrow_location + y_direction * (shaft_length / 2)

    # Rotate to align with Y-axis direction.
    # Default arrow points up (+Z), rotate to point in y_direction.
    default_up = Vector((0, 0, 1))
    rotation_quat = default_up.rotation_difference(y_direction)
    arrow_obj.rotation_euler = rotation_quat.to_euler()

    # Apply cyan emission material.
    arrow_mat = _create_cyan_emission_material(object_name=object_name)
    arrow_obj.data.materials.append(arrow_mat)

    # Link to annotations collection.
    if arrow_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(arrow_obj)
    collection.objects.link(arrow_obj)

    console_logger.debug(
        f"Created arrow: {arrow_obj.name}, location={arrow_obj.location}, "
        f"rotation={arrow_obj.rotation_euler}, hide_render={arrow_obj.hide_render}, "
        f"in_collection={arrow_obj.name in collection.objects}"
    )


def add_blender_scene_annotations(
    scene_objects: list[dict], annotations: DictConfig
) -> None:
    """Add Blender 3D annotation objects to the scene before rendering.

    Args:
        scene_objects: List of scene object metadata dictionaries.
        annotations: Annotation config flags.
    """
    # Create or get annotations collection.
    collection = create_annotation_collection()

    console_logger.info(f"Adding annotations for {len(scene_objects)} objects")

    # Get rendering mode from annotations.
    rendering_mode = "furniture"
    if hasattr(annotations, "rendering_mode"):
        rendering_mode = annotations.rendering_mode

    # Get annotate_object_types filter if specified.
    annotate_object_types = getattr(annotations, "annotate_object_types", None)

    # Process each scene object.
    for obj_meta in scene_objects:
        obj_type = obj_meta.get("object_type", "")

        # Skip room geometry, wall, and floor objects.
        if obj_type in ["room_geometry", "wall", "floor"]:
            continue

        # Filter by annotate_object_types if specified.
        if annotate_object_types is not None and obj_type not in annotate_object_types:
            continue

        # In manipuland mode, only annotate manipulands (skip furniture).
        if rendering_mode == "manipuland" and obj_type != "manipuland":
            continue

        obj_name = obj_meta.get("name", "object")
        position = obj_meta.get("position", [0, 0, 0])
        rotation_matrix = obj_meta.get(
            "rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        # Add bounding box if enabled.
        if annotations.enable_bounding_boxes:
            bbox = obj_meta.get("bounding_box")
            if bbox:
                console_logger.info(f"Adding bbox for {obj_name}: {bbox}")
                add_bounding_box_annotation(
                    bbox=bbox,
                    rotation_matrix=rotation_matrix,
                    collection=collection,
                    object_name=obj_name,
                )
            else:
                console_logger.warning(f"No bbox data for {obj_name}")

        # Add direction arrow if enabled.
        if annotations.enable_direction_arrows:
            console_logger.info(f"Adding arrow for {obj_name}")
            # Use bbox center and extents if available for proper positioning.
            arrow_position = position
            bbox_extents = None
            bbox = obj_meta.get("bounding_box")
            if bbox:
                arrow_position = bbox["center"]
                bbox_extents = tuple(bbox["extents"])
            add_direction_arrow_annotation(
                position=tuple(arrow_position),
                rotation_matrix=rotation_matrix,
                collection=collection,
                object_name=obj_name,
                bbox_extents=bbox_extents,
                rendering_mode=rendering_mode,
            )

    console_logger.debug(
        f"Annotation collection has {len(collection.objects)} objects total"
    )
    for obj in collection.objects:
        console_logger.debug(
            f"  - {obj.name}: location={obj.location}, "
            f"hide_render={obj.hide_render}, visible={not obj.hide_viewport}"
        )


def remove_annotation_objects() -> None:
    """Remove all annotation objects from the scene after rendering."""
    collection_name = "SceneAnnotations"

    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]

        # Remove all objects in the collection.
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

        # Remove the collection itself.
        bpy.data.collections.remove(collection)
