import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.room import (
    ObjectType,
    SceneObject,
    UniqueID,
    _catalog_aabb_support_surfaces,
)
from scenesmith.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)


def make_object(name, source, bounds_max):
    return SceneObject(
        object_id=UniqueID(f"{name}_0"),
        object_type=ObjectType.FURNITURE,
        name=name,
        description=name,
        transform=RigidTransform(),
        metadata={"asset_source": source},
        bbox_min=np.array([-0.5, -1.0, 0.0]),
        bbox_max=np.array(bounds_max, dtype=float),
    )


def test_catalog_bed_uses_bounded_mattress_plane_without_loading_mesh():
    obj = make_object("medical treatment bed", "polyhaven", [0.5, 1.0, 1.2])
    config = SupportSurfaceExtractionConfig()

    surfaces = _catalog_aabb_support_surfaces(obj, config)

    assert surfaces is not None and len(surfaces) == 1
    surface = surfaces[0]
    assert surface.mesh is None
    assert surface.transform.translation()[2] == 1.2 * 0.48 + 0.01
    np.testing.assert_allclose(surface.bounding_box_min[:2], [-0.42, -0.84])
    np.testing.assert_allclose(surface.bounding_box_max[:2], [0.42, 0.84])


def test_catalog_cabinet_uses_inset_top_plane():
    obj = make_object("storage cabinet", "articulated", [0.5, 1.0, 1.4])

    surface = _catalog_aabb_support_surfaces(obj, SupportSurfaceExtractionConfig())[0]

    assert surface.transform.translation()[2] == 1.41
    assert surface.area > 1.0


def test_non_catalog_object_keeps_hsm_path():
    obj = make_object("custom table", "manual", [0.5, 1.0, 0.8])

    assert _catalog_aabb_support_surfaces(obj, SupportSurfaceExtractionConfig()) is None
