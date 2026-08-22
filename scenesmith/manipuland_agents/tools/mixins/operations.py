import json
import logging
import math
import time

from dataclasses import asdict

from pydrake.all import RollPitchYaw

from scenesmith.agent_utils.assets.asset_models import AssetGenerationRequest
from scenesmith.agent_utils.core.action_logger import log_scene_action
from scenesmith.agent_utils.core.response_datatypes import (
    AssetGenerationResult,
    AssetInfo,
    BoundingBox3D,
    GeneratedAsset,
)
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    SupportSurface,
)
from scenesmith.manipuland_agents.tools.arrangement_tools import create_arrangement_impl
from scenesmith.manipuland_agents.tools.fill_tools import fill_container_tool_impl
from scenesmith.manipuland_agents.tools.response_dataclasses import (
    AvailableAssetsResult,
    ManipulandErrorType,
    ManipulandInfo,
    ManipulandOperationResult,
    Position2D,
    Position3D,
    Rotation3D,
    SimplifiedFurnitureInfo,
    SimplifiedManipulandInfo,
    SupportSurfaceWithManipulands,
)
from scenesmith.manipuland_agents.tools.stacking_tools.pile_creation import (
    create_pile_tool_impl,
)
from scenesmith.manipuland_agents.tools.stacking_tools.stack_tools import (
    create_stack_tool_impl,
)

console_logger = logging.getLogger(__name__)

from scenesmith.manipuland_agents.tools.manipuland_tool_models import FillAssetItem


class ManipulandOperationsMixin:
    """Asset, scene-state, arrangement, and DTO operations."""

    def _generate_assets_impl(self, request: AssetGenerationRequest) -> str:
        """Implementation for generating manipuland assets."""
        console_logger.info(
            f"Generating batch of {len(request.object_descriptions)} manipuland assets"
        )
        start_time = time.time()

        # Generate assets using asset manager.
        result = self.asset_manager.generate_assets(request)

        # Convert successful assets to DTOs.
        generated_assets = [
            GeneratedAsset(
                name=obj.name,
                object_id=str(obj.object_id),
                description=obj.description,
                width=(
                    float(obj.bbox_max[0] - obj.bbox_min[0])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                depth=(
                    float(obj.bbox_max[1] - obj.bbox_min[1])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
                height=(
                    float(obj.bbox_max[2] - obj.bbox_min[2])
                    if obj.bbox_min is not None and obj.bbox_max is not None
                    else None
                ),
            )
            for obj in result.successful_assets
        ]

        elapsed_time = time.time() - start_time

        # Handle partial success (failures exist).
        if result.has_failures:
            failures_detail = "\n".join(
                [f"- {f.description}: {f.error_message}" for f in result.failed_assets]
            )

            message = (
                f"Partially successful: {len(result.successful_assets)} succeeded, "
                f"{len(result.failed_assets)} failed in {elapsed_time:.1f}s"
            )

            console_logger.warning(message)
            console_logger.warning(f"Failures:\n{failures_detail}")

            return AssetGenerationResult(
                success=False,
                assets=generated_assets,
                message=message,
                successful_count=len(result.successful_assets),
                failed_count=len(result.failed_assets),
                failures=failures_detail,
            ).to_json()

        # All succeeded.
        message = (
            f"Successfully generated {len(generated_assets)} manipuland(s) "
            f"in {elapsed_time:.1f}s"
        )
        console_logger.info(message)

        return AssetGenerationResult(
            success=True,
            assets=generated_assets,
            message=message,
        ).to_json()

    def _get_current_scene_state_impl(self) -> str:
        """Implementation for getting current scene state (filtered)."""
        console_logger.info("Tool called: get_current_scene_state")

        # Get current furniture object.
        furniture = self.scene.get_object(self.current_furniture_id)
        if not furniture:
            return ManipulandOperationResult(
                success=False,
                message=f"Furniture {self.current_furniture_id} not found",
                error_type=ManipulandErrorType.SURFACE_NOT_FOUND,
            ).to_json()

        # Convert furniture to simplified DTO.
        furniture_info = self._scene_object_to_simplified_furniture_info(furniture)

        # Build surfaces with their manipulands grouped together.
        total_manipuland_count = 0
        surface_infos = []
        for surface in self.support_surfaces.values():
            manipulands_on_surface = self.scene.get_objects_on_surface(
                surface.surface_id
            )
            total_manipuland_count += len(manipulands_on_surface)
            surface_info = self._support_surface_to_dto_with_manipulands(
                surface=surface, manipulands=manipulands_on_surface
            )
            surface_infos.append(surface_info)

        # Build structured response with manipulands grouped by surface.
        result = {
            "current_furniture": asdict(furniture_info),
            "surfaces": [asdict(s) for s in surface_infos],
            "num_surfaces": len(surface_infos),
            "total_manipuland_count": total_manipuland_count,
        }

        return json.dumps(result, indent=2)

    def _list_available_assets_impl(self) -> str:
        """Implementation for listing all available manipuland assets."""
        console_logger.info("Tool called: list_available_assets")

        # Get all assets from registry and filter for manipulands.
        all_assets = self.asset_manager.list_available_assets()
        available_assets = [
            asset for asset in all_assets if asset.object_type == ObjectType.MANIPULAND
        ]

        # Convert to simplified DTOs.
        asset_dtos = [AssetInfo.from_scene_object(asset) for asset in available_assets]

        result = AvailableAssetsResult(
            assets=asset_dtos,
            total_count=len(asset_dtos),
            message=f"Found {len(asset_dtos)} available manipuland assets",
        )

        return result.to_json()

    @log_scene_action
    def _create_stack_impl(
        self,
        asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for creating a stack of objects on a support surface.

        Delegates to create_stack_tool_impl in stack_tools.py.
        """
        return create_stack_tool_impl(
            asset_ids=asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
        )

    @log_scene_action
    def _fill_container_impl(
        self,
        container_asset_id: str,
        fill_asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for filling a container with objects.

        Delegates to fill_container_tool_impl in fill_tools.py.
        """
        return fill_container_tool_impl(
            container_asset_id=container_asset_id,
            fill_asset_ids=fill_asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
            top_surface_overlap_tolerance=self.top_surface_overlap_tolerance,
            is_top_surface_fn=self._is_top_surface,
            validate_footprint_fn=self._validate_convex_hull_footprint,
        )

    @log_scene_action
    def _create_arrangement_impl(
        self,
        container_asset_id: str,
        fill_assets: list[FillAssetItem],
        surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for creating a controlled arrangement on a flat container.

        Delegates to create_arrangement_impl in arrangement_tools.py.
        """
        return create_arrangement_impl(
            container_asset_id=container_asset_id,
            fill_assets=fill_assets,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_degrees=rotation_degrees,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
            validate_footprint_fn=self._validate_convex_hull_footprint,
            top_surface_overlap_tolerance=self.top_surface_overlap_tolerance,
            is_top_surface_fn=self._is_top_surface,
        )

    @log_scene_action
    def _create_pile_impl(
        self,
        asset_ids: list[str],
        surface_id: str,
        position_x: float,
        position_z: float,
        **kwargs,
    ) -> str:
        """Implementation for creating a pile of objects.

        Delegates to create_pile_tool_impl in pile_tools.py.
        """
        return create_pile_tool_impl(
            asset_ids=asset_ids,
            surface_id=surface_id,
            position_x=position_x,
            position_z=position_z,
            scene=self.scene,
            cfg=self.cfg,
            asset_manager=self.asset_manager,
            support_surfaces=self.support_surfaces,
            generate_unique_id=self.scene.generate_unique_id,
        )

    def _scene_object_to_manipuland_info(self, obj: SceneObject) -> ManipulandInfo:
        """Convert SceneObject to ManipulandInfo DTO."""
        position = obj.transform.translation()
        rpy = RollPitchYaw(obj.transform.rotation())

        # Extract placement info if available.
        surface_position = None
        surface_rotation_deg = None
        parent_surface_id = None

        if obj.placement_info:
            surface_position = Position2D(
                x=float(obj.placement_info.position_2d[0]),
                y=float(obj.placement_info.position_2d[1]),
            )
            surface_rotation_deg = math.degrees(obj.placement_info.rotation_2d)
            parent_surface_id = str(obj.placement_info.parent_surface_id)

        return ManipulandInfo(
            object_id=str(obj.object_id),
            description=obj.description,
            object_type=obj.object_type.value,
            position=Position3D(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            rotation=Rotation3D(
                roll=math.degrees(rpy.roll_angle()),
                pitch=math.degrees(rpy.pitch_angle()),
                yaw=math.degrees(rpy.yaw_angle()),
            ),
            surface_position=surface_position,
            surface_rotation_deg=surface_rotation_deg,
            parent_surface_id=parent_surface_id,
            has_geometry=obj.geometry_path is not None,
        )

    def _scene_object_to_simplified_manipuland_info(
        self, obj: SceneObject
    ) -> SimplifiedManipulandInfo:
        """Convert SceneObject to SimplifiedManipulandInfo (minimal fields)."""
        surface_position = None
        surface_rotation_deg = None

        if obj.placement_info:
            surface_position = Position2D(
                x=float(obj.placement_info.position_2d[0]),
                y=float(obj.placement_info.position_2d[1]),
            )
            surface_rotation_deg = math.degrees(obj.placement_info.rotation_2d)

        dimensions = None
        if obj.bbox_min is not None and obj.bbox_max is not None:
            bbox_size = obj.bbox_max - obj.bbox_min
            dimensions = BoundingBox3D(
                width=float(bbox_size[0]),
                depth=float(bbox_size[1]),
                height=float(bbox_size[2]),
            )

        # Build composite metadata if this is a composite object.
        composite_metadata = None
        composite_type = obj.metadata.get("composite_type")
        if composite_type == "stack":
            member_assets = obj.metadata.get("member_assets", [])
            composite_metadata = {
                "type": "stack",
                "members": [m.get("asset_id", "unknown") for m in member_assets],
            }
        elif composite_type == "filled_container":
            container_asset = obj.metadata.get("container_asset", {})
            fill_assets = obj.metadata.get("fill_assets", [])
            placement_method = obj.metadata.get("placement_method", "random")

            if placement_method == "controlled":
                # For arrangements: show poses (x, y, rotation) and shape-aware bounds.
                composite_metadata = {
                    "type": "filled_container",
                    "container_id": container_asset.get("asset_id", "unknown"),
                    "fill_count": len(fill_assets),
                    "fill_items": [
                        {
                            "id": f.get("asset_id", "unknown"),
                            "x": f.get("user_pose", {}).get("x", 0),
                            "y": f.get("user_pose", {}).get("y", 0),
                            "rotation": f.get("user_pose", {}).get("rotation", 0),
                        }
                        for f in fill_assets
                    ],
                    "container_bounds": obj.metadata.get("container_bounds"),
                }
            else:
                # For random fills (fill_container): existing behavior.
                composite_metadata = {
                    "type": "filled_container",
                    "container_id": container_asset.get("asset_id", "unknown"),
                    "fill_object_ids": [
                        f.get("asset_id", "unknown") for f in fill_assets
                    ],
                    "fill_count": len(fill_assets),
                }
        elif composite_type == "pile":
            member_assets = obj.metadata.get("member_assets", [])
            composite_metadata = {
                "type": "pile",
                "members": [m.get("asset_id", "unknown") for m in member_assets],
                "pile_count": len(member_assets),
            }

        return SimplifiedManipulandInfo(
            object_id=str(obj.object_id),
            description=obj.description,
            surface_position=surface_position,
            surface_rotation_deg=surface_rotation_deg,
            dimensions=dimensions,
            composite_metadata=composite_metadata,
        )

    def _scene_object_to_simplified_furniture_info(
        self, obj: SceneObject
    ) -> SimplifiedFurnitureInfo:
        """Convert SceneObject (furniture) to SimplifiedFurnitureInfo."""
        dimensions = None
        if obj.bbox_min is not None and obj.bbox_max is not None:
            bbox_size = obj.bbox_max - obj.bbox_min
            dimensions = BoundingBox3D(
                width=float(bbox_size[0]),
                depth=float(bbox_size[1]),
                height=float(bbox_size[2]),
            )

        return SimplifiedFurnitureInfo(
            object_id=str(obj.object_id),
            description=obj.description,
            dimensions=dimensions,
        )

    def _support_surface_to_dto_with_manipulands(
        self, surface: SupportSurface, manipulands: list[SceneObject]
    ) -> SupportSurfaceWithManipulands:
        """Convert SupportSurface to DTO with its manipulands grouped together."""
        bounds_min_2d = Position2D(
            x=float(surface.bounding_box_min[0]),
            y=float(surface.bounding_box_min[1]),
        )
        bounds_max_2d = Position2D(
            x=float(surface.bounding_box_max[0]),
            y=float(surface.bounding_box_max[1]),
        )

        # Extract world-frame position from surface transform.
        world_position = surface.transform.translation()

        # Convert manipulands to simplified DTOs.
        manipuland_infos = [
            self._scene_object_to_simplified_manipuland_info(obj) for obj in manipulands
        ]

        return SupportSurfaceWithManipulands(
            surface_id=str(surface.surface_id),
            bounds_min=bounds_min_2d,
            bounds_max=bounds_max_2d,
            world_x=float(world_position[0]),
            world_y=float(world_position[1]),
            world_z=float(world_position[2]),
            clearance_height=float(surface.bounding_box_max[2]),
            manipulands=manipuland_infos,
        )
