from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenesmith.agent_utils.geometry.support_surfaces.models import (
    SupportSurfaceExtractionConfig,
)
from scenesmith.agent_utils.scene.room_parts.room_models import (
    ObjectType,
    SceneObject,
    UniqueID,
)
from scenesmith.agent_utils.scene.room_parts.room_support import (
    _catalog_aabb_support_surfaces,
    extract_and_propagate_support_surfaces,
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


def test_intrinsic_hssd_bookcase_keeps_authored_tier_surface_path():
    obj = make_object("renaissance bookshelf", "hssd", [0.5, 0.18, 2.0])
    obj.metadata.update(
        {
            "catalog_id": "hssd__e3631a629a1ac3b71a75dff721192b90d26246e0",
            "ontology_path": "hssd/wordnet/bookcase.n.01",
            "catalog_semantics": "Revolve Lexington Tall Bookcase, Walnut",
        }
    )

    assert _catalog_aabb_support_surfaces(obj, SupportSurfaceExtractionConfig()) is None


def test_intrinsic_hssd_bookcase_loads_authored_tiers_from_catalog_id():
    obj = make_object("renaissance bookshelf", "hssd", [0.5, 0.18, 2.0])
    obj.geometry_path = Path("unused_when_prevalidated_surfaces_exist.gltf")
    obj.metadata.update(
        {
            "catalog_id": "hssd__e3631a629a1ac3b71a75dff721192b90d26246e0",
            "ontology_path": "hssd/wordnet/bookcase.n.01",
            "catalog_semantics": "Revolve Lexington Tall Bookcase, Walnut",
        }
    )

    class FakeScene:
        objects = {obj.object_id: obj}
        next_surface = 0

        def generate_surface_id(self):
            result = UniqueID(f"S_{self.next_surface}")
            self.next_surface += 1
            return result

    surfaces = extract_and_propagate_support_surfaces(
        FakeScene(), obj, SupportSurfaceExtractionConfig()
    )

    internal_heights = [
        float(surface.transform.translation()[2])
        for surface in surfaces
        if 0.15 < float(surface.transform.translation()[2]) < 1.85
    ]
    assert len(internal_heights) >= 5
