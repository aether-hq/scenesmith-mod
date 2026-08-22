import logging
import os

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np

from pydrake.all import Quaternion, RigidTransform, RotationMatrix

from scenesmith.utils.geometry.sdf_utils import (
    deserialize_rigid_transform,
    extract_base_link_name_from_sdf,
    is_static_sdf_model,
)

console_logger = logging.getLogger(__name__)


from scenesmith.agent_utils.scene.room_parts.room_models import ObjectType, UniqueID


class RoomDirectiveMixin:
    """Serialize a room scene into Drake model directives."""

    def to_drake_directive(
        self,
        include_objects: list[UniqueID] | None = None,
        include_object_types: list[ObjectType] | None = None,
        weld_furniture: bool = True,
        free_objects: list[UniqueID] | None = None,
        exclude_room_geometry: bool = False,
        include_additional_structural_geometry: bool = True,
        weld_stack_members: bool = True,
        weld_room_geometry: bool = True,
        room_geometry_name: str = "room_geometry",
        model_name_prefix: str = "",
        base_dir: Path | None = None,
        free_mounted_objects_for_collision: bool = False,
        parent_frame: str = "world",
    ) -> str:
        """Generate a Drake directive string from the current scene.

        Args:
            include_objects: If provided, only include these objects in the
                directive. Useful for fast collision checking between 2-3
                objects.
            include_object_types: If provided, only include objects of these
                types (e.g., [ObjectType.FURNITURE, ObjectType.WALL_MOUNTED]).
                Useful for intermediate house snapshots.
            weld_furniture: If True (default), weld furniture to world frame.
                If False, add furniture as free bodies for IK optimization.
            free_objects: If provided, these specific objects will be free bodies
                regardless of weld_furniture setting. Useful for IK when you want
                most furniture welded but one object free to optimize.
            exclude_room_geometry: If True, completely exclude the room geometry
                from the directive. Useful for focused rendering (e.g., manipuland
                agent viewing only furniture + manipulands).
            include_additional_structural_geometry: Include collision SDFs adjacent
                to persisted structural-surface sidecars. HouseScene disables this
                because HouseLayout emits the same platform models globally.
            weld_stack_members: If True (default), weld upper stack members to
                the bottom member, treating stacks as rigid units. If False, all
                stack members are free bodies (legacy behavior).
            weld_room_geometry: If True (default), weld room geometry to world at
                origin. If False, only add the model without weld. HouseScene
                sets this to False and handles welding with transforms.
            room_geometry_name: Model name for room geometry. Use unique names for
                multi-room houses (e.g., "room_geometry_living_room").
            model_name_prefix: Prefix to prepend to all model names. Used by
                HouseScene to ensure globally unique model names across rooms
                (e.g., "living_room_" makes "rug_0" become "living_room_rug_0").
            base_dir: If provided, SDF paths are relative to this directory
                (for portable directives). The directive YAML file should be
                saved in this directory for Drake to resolve paths correctly.
                If None, absolute paths with file:// scheme are used (for temp
                file usage in physics simulation).
            free_mounted_objects_for_collision: If True, wall-mounted and
                ceiling-mounted objects are treated as free bodies instead of
                welded. Used for collision checking where Drake's broadphase
                needs free bodies to detect collisions properly.
            parent_frame: Frame to parent objects to. All object poses and
                welds reference this frame. Free body poses include
                base_frame for Drake frame resolution. Defaults to "world".

        Returns:
            Drake directive in YAML format that can be loaded by Drake's
            ProcessModelDirectives.
        """

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

        def format_free_body_pose(
            link_name: str,
            tx: float,
            ty: float,
            tz: float,
            angle_deg: float,
            axis: list[float],
        ) -> str:
            """Format default_free_body_pose with base_frame."""
            return f"""
    default_free_body_pose:
      {link_name}:
        base_frame: {parent_frame}
        translation: [{tx}, {ty}, {tz}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""

        if exclude_room_geometry:
            directive = "directives:"
        else:
            room_geom_path = format_sdf_path(self.room_geometry.sdf_path)
            directive = f"""directives:
- add_model:
    name: {room_geometry_name}
    file: {room_geom_path}"""
            if weld_room_geometry:
                directive += f"""
- add_weld:
    parent: {parent_frame}
    child: {room_geometry_name}::room_geometry_body_link"""

            if include_additional_structural_geometry:
                for index, surface_path in enumerate(
                    self.room_geometry.additional_structural_surface_paths
                ):
                    suffix = ".surfaces.json"
                    if not surface_path.name.endswith(suffix):
                        console_logger.warning(
                            "Cannot resolve structural collision geometry from %s",
                            surface_path,
                        )
                        continue
                    structure_name = surface_path.name[: -len(suffix)]
                    sdf_path = surface_path.with_name(f"{structure_name}.sdf")
                    if not sdf_path.is_file():
                        console_logger.warning(
                            "Structural collision geometry is missing for %s: %s",
                            surface_path,
                            sdf_path,
                        )
                        continue
                    model_name = f"{room_geometry_name}_additional_support_{index}"
                    directive += f"""
- add_model:
    name: {model_name}
    file: {format_sdf_path(sdf_path)}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::structure_link"""

        # Filter objects by ID and/or type.
        objects_to_add = list(self.objects.values())
        if include_objects is not None:
            objects_to_add = [
                obj for obj in objects_to_add if obj.object_id in include_objects
            ]
        if include_object_types is not None:
            objects_to_add = [
                obj for obj in objects_to_add if obj.object_type in include_object_types
            ]

        console_logger.info(
            f"to_drake_directive filtering: "
            f"include_objects={len(include_objects) if include_objects else 'None'}, "
            f"total scene objects={len(self.objects)}, "
            f"filtered objects_to_add={len(objects_to_add)}"
        )

        # Add scene objects.
        for obj in objects_to_add:
            # Handle composite objects (e.g., stacks) by expanding member assets.
            if obj.metadata.get("composite_type") == "stack":
                member_assets = obj.metadata.get("member_assets", [])

                # Clear cached model names (regenerated below for collision lookup).
                obj.metadata["member_model_names"] = []

                # Track bottom member info for welding upper members.
                bottom_model_name: str | None = None
                bottom_base_link: str | None = None
                bottom_transform: RigidTransform | None = None

                for i, member in enumerate(member_assets):
                    member_sdf = member.get("sdf_path")
                    if not member_sdf:
                        continue

                    # Deserialize stored transform (format from serialize_rigid_transform).
                    transform_data = member.get("transform", {})
                    translation = transform_data.get("translation", [0, 0, 0])

                    # Convert quaternion to angle-axis for Drake directive.
                    rotation_wxyz = transform_data.get("rotation_wxyz", [1, 0, 0, 0])
                    quaternion = Quaternion(wxyz=rotation_wxyz)
                    angle_axis = RotationMatrix(quaternion).ToAngleAxis()
                    angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                    axis = [float(x) for x in angle_axis.axis()]

                    # Generate unique model name for each stack member.
                    # The prefix is used by HouseScene to ensure globally unique names.
                    member_name = member.get("name", "stack_member")
                    member_id = member.get("asset_id", "unknown")
                    id_suffix = member_id.split("_")[-1][:8]
                    stack_suffix = str(obj.object_id).split("_")[-1][:4]
                    model_name = (
                        f"{model_name_prefix}{member_name.lower().replace(' ', '_')}_"
                        f"{id_suffix}_s{stack_suffix}_{i}"
                    )

                    # Store model name for direct lookup in collision detection.
                    obj.metadata["member_model_names"].append(model_name)

                    # Extract base link name from member SDF.
                    try:
                        base_link_name = extract_base_link_name_from_sdf(
                            Path(member_sdf)
                        )
                    except ValueError as e:
                        console_logger.warning(
                            f"Warning: {e}. Using 'base_link' as fallback."
                        )
                        base_link_name = "base_link"

                    if weld_stack_members and i == 0:
                        # Bottom member: free body, store info for welding upper members.
                        bottom_model_name = model_name
                        bottom_base_link = base_link_name
                        bottom_transform = deserialize_rigid_transform(transform_data)
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )
                    elif weld_stack_members and i > 0:
                        # Upper member: welded to bottom member.
                        member_transform = deserialize_rigid_transform(transform_data)
                        t_rel = bottom_transform.inverse() @ member_transform
                        rel_translation = t_rel.translation()
                        rel_angle_axis = t_rel.rotation().ToAngleAxis()
                        rel_angle_deg = float(rel_angle_axis.angle()) * 180.0 / np.pi
                        rel_axis = [float(x) for x in rel_angle_axis.axis()]

                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}
- add_weld:
    parent: {bottom_model_name}::{bottom_base_link}
    child: {model_name}::{base_link_name}
    X_PC:
      translation: [{rel_translation[0]}, {rel_translation[1]}, {rel_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {rel_angle_deg}
        axis: [{rel_axis[0]}, {rel_axis[1]}, {rel_axis[2]}]"""
                    else:
                        # weld_stack_members=False: all members as free bodies.
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                continue

            # Handle filled containers (container + fill objects inside).
            if obj.metadata.get("composite_type") == "filled_container":
                container_asset = obj.metadata.get("container_asset")
                fill_assets = obj.metadata.get("fill_assets", [])

                # Clear cached model names (regenerated below for collision lookup).
                obj.metadata["member_model_names"] = []

                # Track container info for welding fill objects.
                container_model_name: str | None = None
                container_base_link: str | None = None
                container_transform: RigidTransform | None = None

                # Add container as free body (reference member).
                if container_asset:
                    container_sdf = container_asset.get("sdf_path")
                    if container_sdf:
                        transform_data = container_asset.get("transform", {})
                        translation = transform_data.get("translation", [0, 0, 0])
                        rotation_wxyz = transform_data.get(
                            "rotation_wxyz", [1, 0, 0, 0]
                        )
                        quaternion = Quaternion(wxyz=rotation_wxyz)
                        angle_axis = RotationMatrix(quaternion).ToAngleAxis()
                        angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                        axis = [float(x) for x in angle_axis.axis()]

                        # Generate unique model name for container.
                        container_name = container_asset.get("name", "container")
                        container_id = container_asset.get("asset_id", "unknown")
                        id_suffix = container_id.split("_")[-1][:8]
                        fill_suffix = str(obj.object_id).split("_")[-1][:4]
                        container_model_name = (
                            f"{model_name_prefix}{container_name.lower().replace(' ', '_')}_"
                            f"{id_suffix}_f{fill_suffix}_c"
                        )

                        obj.metadata["member_model_names"].append(container_model_name)

                        try:
                            container_base_link = extract_base_link_name_from_sdf(
                                Path(container_sdf)
                            )
                        except ValueError as e:
                            console_logger.warning(
                                f"Warning: {e}. Using 'base_link' as fallback."
                            )
                            container_base_link = "base_link"

                        container_transform = deserialize_rigid_transform(
                            transform_data
                        )

                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]

                        container_sdf_formatted = format_sdf_path(container_sdf)
                        directive += f"""
- add_model:
    name: {container_model_name}
    file: {container_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=container_base_link,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                # Add fill objects welded to container.
                for i, fill_asset in enumerate(fill_assets):
                    fill_sdf = fill_asset.get("sdf_path")
                    if not fill_sdf:
                        continue

                    transform_data = fill_asset.get("transform", {})
                    translation = transform_data.get("translation", [0, 0, 0])
                    rotation_wxyz = transform_data.get("rotation_wxyz", [1, 0, 0, 0])
                    quaternion = Quaternion(wxyz=rotation_wxyz)
                    angle_axis = RotationMatrix(quaternion).ToAngleAxis()
                    angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                    axis = [float(x) for x in angle_axis.axis()]

                    fill_name = fill_asset.get("name", "fill_item")
                    fill_id = fill_asset.get("asset_id", "unknown")
                    id_suffix = fill_id.split("_")[-1][:8]
                    fill_suffix = str(obj.object_id).split("_")[-1][:4]
                    fill_model_name = (
                        f"{model_name_prefix}{fill_name.lower().replace(' ', '_')}_"
                        f"{id_suffix}_f{fill_suffix}_{i}"
                    )

                    obj.metadata["member_model_names"].append(fill_model_name)

                    try:
                        fill_base_link = extract_base_link_name_from_sdf(Path(fill_sdf))
                    except ValueError as e:
                        console_logger.warning(
                            f"Warning: {e}. Using 'base_link' as fallback."
                        )
                        fill_base_link = "base_link"

                    if (
                        weld_stack_members
                        and container_model_name
                        and container_transform
                    ):
                        # Fill object welded to container.
                        fill_transform = deserialize_rigid_transform(transform_data)
                        t_rel = container_transform.inverse() @ fill_transform
                        rel_translation = t_rel.translation()
                        rel_angle_axis = t_rel.rotation().ToAngleAxis()
                        rel_angle_deg = float(rel_angle_axis.angle()) * 180.0 / np.pi
                        rel_axis = [float(x) for x in rel_angle_axis.axis()]

                        fill_sdf_formatted = format_sdf_path(fill_sdf)
                        directive += f"""
- add_model:
    name: {fill_model_name}
    file: {fill_sdf_formatted}
- add_weld:
    parent: {container_model_name}::{container_base_link}
    child: {fill_model_name}::{fill_base_link}
    X_PC:
      translation: [{rel_translation[0]}, {rel_translation[1]}, {rel_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {rel_angle_deg}
        axis: [{rel_axis[0]}, {rel_axis[1]}, {rel_axis[2]}]"""
                    else:
                        # weld_stack_members=False: fill objects as free bodies.
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        fill_sdf_formatted = format_sdf_path(fill_sdf)
                        directive += f"""
- add_model:
    name: {fill_model_name}
    file: {fill_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=fill_base_link,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                continue

            # Handle piles (member assets similar to stack structure).
            if obj.metadata.get("composite_type") == "pile":
                member_assets = obj.metadata.get("member_assets", [])

                # Clear cached model names (regenerated below for collision lookup).
                obj.metadata["member_model_names"] = []

                # Track first member info for welding other members.
                first_model_name: str | None = None
                first_base_link: str | None = None
                first_transform: RigidTransform | None = None

                for i, member in enumerate(member_assets):
                    member_sdf = member.get("sdf_path")
                    if not member_sdf:
                        continue

                    # Deserialize stored transform (format from serialize_rigid_transform).
                    transform_data = member.get("transform", {})
                    translation = transform_data.get("translation", [0, 0, 0])

                    # Convert quaternion to angle-axis for Drake directive.
                    rotation_wxyz = transform_data.get("rotation_wxyz", [1, 0, 0, 0])
                    quaternion = Quaternion(wxyz=rotation_wxyz)
                    angle_axis = RotationMatrix(quaternion).ToAngleAxis()
                    angle_deg = float(angle_axis.angle()) * 180.0 / np.pi
                    axis = [float(x) for x in angle_axis.axis()]

                    # Generate unique model name for each pile member.
                    member_name = member.get("name", "pile_member")
                    member_id = member.get("asset_id", "unknown")
                    id_suffix = member_id.split("_")[-1][:8]
                    pile_suffix = str(obj.object_id).split("_")[-1][:4]
                    model_name = (
                        f"{model_name_prefix}{member_name.lower().replace(' ', '_')}_"
                        f"{id_suffix}_p{pile_suffix}_{i}"
                    )

                    # Store model name for direct lookup in collision detection.
                    obj.metadata["member_model_names"].append(model_name)

                    # Extract base link name from member SDF.
                    try:
                        base_link_name = extract_base_link_name_from_sdf(
                            Path(member_sdf)
                        )
                    except ValueError as e:
                        console_logger.warning(
                            f"Warning: {e}. Using 'base_link' as fallback."
                        )
                        base_link_name = "base_link"

                    if weld_stack_members and i == 0:
                        # First member: free body, store info for welding other members.
                        first_model_name = model_name
                        first_base_link = base_link_name
                        first_transform = deserialize_rigid_transform(transform_data)
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )
                    elif weld_stack_members and i > 0:
                        # Other members: welded to first member.
                        member_transform = deserialize_rigid_transform(transform_data)
                        t_rel = first_transform.inverse() @ member_transform
                        rel_translation = t_rel.translation()
                        rel_angle_axis = t_rel.rotation().ToAngleAxis()
                        rel_angle_deg = float(rel_angle_axis.angle()) * 180.0 / np.pi
                        rel_axis = [float(x) for x in rel_angle_axis.axis()]

                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}
- add_weld:
    parent: {first_model_name}::{first_base_link}
    child: {model_name}::{base_link_name}
    X_PC:
      translation: [{rel_translation[0]}, {rel_translation[1]}, {rel_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {rel_angle_deg}
        axis: [{rel_axis[0]}, {rel_axis[1]}, {rel_axis[2]}]"""
                    else:
                        # weld_stack_members=False: all members as free bodies.
                        tx = translation[0]
                        ty = translation[1]
                        tz = translation[2]
                        member_sdf_formatted = format_sdf_path(member_sdf)
                        directive += f"""
- add_model:
    name: {model_name}
    file: {member_sdf_formatted}"""
                        directive += format_free_body_pose(
                            link_name=base_link_name,
                            tx=tx,
                            ty=ty,
                            tz=tz,
                            angle_deg=angle_deg,
                            axis=axis,
                        )

                continue

            if obj.sdf_path is None:
                continue

            # Extract position and orientation from RigidTransform.
            translation = obj.transform.translation()
            angle_axis = obj.transform.rotation().ToAngleAxis()
            # Create unique model name by combining name with ID suffix.
            # This ensures Drake model instances are unique even for reused assets.
            # The prefix is used by HouseScene to ensure globally unique names.
            id_suffix = str(obj.object_id).split("_")[-1][:8]
            model_name = (
                f"{model_name_prefix}{obj.name.lower().replace(' ', '_')}_{id_suffix}"
            )

            # Extract the base link name from the SDF file.
            try:
                base_link_name = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError as e:
                # Fallback to "base_link" if extraction fails.
                console_logger.warning(f"Warning: {e}. Using 'base_link' as fallback.")
                base_link_name = "base_link"

            # Convert angle to degrees.
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            # Determine if this object should be welded or free.
            # Thin coverings are always welded - they have no collision geometry
            # so would fall through the floor during simulation.
            # Wall-mounted objects are normally welded - they're mounted on walls
            # and shouldn't move during physics simulation.
            # Ceiling-mounted objects are normally welded - they hang from ceiling
            # and shouldn't move during physics simulation.
            # For collision checking, wall/ceiling objects must be free bodies
            # for Drake's broadphase to detect collisions between them.
            is_thin_covering = obj.metadata.get("asset_source") == "thin_covering"
            is_wall_mounted = obj.object_type == ObjectType.WALL_MOUNTED
            is_ceiling_mounted = obj.object_type == ObjectType.CEILING_MOUNTED
            is_owner_bound_decor = bool(obj.metadata.get("dense_library_owner_bound"))
            is_immutable = obj.immutable
            if free_mounted_objects_for_collision:
                # Immutable authored structures remain anchored. Other mounted
                # and owner-bound decor become free so broadphase queries can see
                # their contacts with room geometry and non-owner objects.
                always_welded = is_thin_covering or is_immutable
            else:
                always_welded = (
                    is_thin_covering
                    or is_wall_mounted
                    or is_ceiling_mounted
                    or is_owner_bound_decor
                    or is_immutable
                )
            if free_objects is not None:
                # Exclusive mode: ONLY objects in free_objects are free.
                # Used by large scene optimization to reduce DOFs.
                is_free = obj.object_id in free_objects and not always_welded
            else:
                # Original logic when free_objects not specified.
                is_free = (
                    (obj.object_type != ObjectType.FURNITURE) or not weld_furniture
                ) and not always_welded

            if is_free:
                # Free body (in free_objects list or manipuland).
                tx = translation[0]
                ty = translation[1]
                tz = translation[2]
                obj_sdf_formatted = format_sdf_path(obj.sdf_path)
                directive += f"""
- add_model:
    name: {model_name}
    file: {obj_sdf_formatted}"""
                directive += format_free_body_pose(
                    link_name=base_link_name,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                    angle_deg=angle_deg,
                    axis=axis,
                )
            else:
                # Welded (furniture, wall-mounted, or thin covering).
                # Check if model is static (auto-welded by Drake).
                sdf_path = Path(obj.sdf_path).absolute() if obj.sdf_path else None
                is_static = sdf_path and is_static_sdf_model(sdf_path)

                if is_static:
                    # Static models are auto-welded by Drake at their pose.
                    # Use default_free_body_pose to set initial position.
                    tx = translation[0]
                    ty = translation[1]
                    tz = translation[2]
                    obj_sdf_formatted = format_sdf_path(obj.sdf_path)
                    directive += f"""
- add_model:
    name: {model_name}
    file: {obj_sdf_formatted}"""
                    directive += format_free_body_pose(
                        link_name=base_link_name,
                        tx=tx,
                        ty=ty,
                        tz=tz,
                        angle_deg=angle_deg,
                        axis=axis,
                    )
                else:
                    # Non-static models need explicit weld.
                    tx = translation[0]
                    ty = translation[1]
                    tz = translation[2]
                    obj_sdf_formatted = format_sdf_path(obj.sdf_path)
                    directive += f"""
- add_model:
    name: {model_name}
    file: {obj_sdf_formatted}
- add_weld:
    parent: {parent_frame}
    child: {model_name}::{base_link_name}
    X_PC:
      translation: [{tx}, {ty}, {tz}]
      rotation: !AngleAxis
        angle_deg: {angle_deg}
        axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""

        return directive
