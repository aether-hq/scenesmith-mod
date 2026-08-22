"""Tests for deterministic parametric structural compilers."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scenesmith.agent_utils.structure.compiler.connector_primitives import (
    compile_straight_ramp,
    compile_straight_stairs,
)
from scenesmith.agent_utils.structure.compiler.mesh_assembly import (
    compile_structural_mesh,
)
from scenesmith.agent_utils.structure.compiler.models import (
    TriangleMesh,
    audit_triangle_mesh,
)
from scenesmith.agent_utils.structure.compiler.polygon_spaces import (
    compile_polygon_space,
)
from scenesmith.agent_utils.structure.compiler.surfaces import (
    compile_heightfield,
    compile_platform,
)
from scenesmith.agent_utils.structure.compiler.writing import write_compiled_structure
from scenesmith.agent_utils.structure.geometry_models.mesh_models import (
    MeshSurfaceAnnotation,
    StructuralMeshSpec,
)
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    Footprint2D,
    HeightfieldSpec,
    PlatformSpec,
    SurfaceRole,
    Transform3D,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
)
from scenesmith.agent_utils.structure.geometry_models.validation import (
    GeometryValidationError,
    UnsupportedGeometryError,
)


def _straight_stairs_connector() -> ConnectorSpec:
    return ConnectorSpec(
        connector_id="main_stairs",
        connector_type=ConnectorType.STAIRS_STRAIGHT,
        start=ConnectorEndpoint("lower", "ground", (0.0, 0.0, 0.0)),
        end=ConnectorEndpoint("upper", "upper", (4.0, 0.0, 3.0)),
        width=1.2,
        parameters={"riser_count": 18},
    )


class TestPlatforms(unittest.TestCase):
    def test_mezzanine_has_support_and_open_edge_semantics(self) -> None:
        compiled = compile_platform(
            PlatformSpec(
                platform_id="mezzanine",
                space_id="loft",
                footprint=Footprint2D.rectangle(4, 2),
                elevation=3.0,
                thickness=0.2,
                open_edge_indices=(2,),
            )
        )

        self.assertEqual(compiled.visual_mesh.bounds, ((0, 0, 2.8), (4, 2, 3.0)))
        self.assertIn(SurfaceRole.SUPPORT, compiled.surfaces[0].surface.roles)
        self.assertIn(SurfaceRole.TRAVERSABLE, compiled.surfaces[0].surface.roles)
        open_edges = [
            patch
            for patch in compiled.surfaces
            if SurfaceRole.OPEN_EDGE in patch.surface.roles
        ]
        self.assertEqual(len(open_edges), 1)
        self.assertEqual(open_edges[0].surface.metadata["edge_index"], 2)

    def test_nontraversable_display_platform_is_support_only(self) -> None:
        compiled = compile_platform(
            PlatformSpec(
                platform_id="plinth",
                space_id="gallery",
                footprint=Footprint2D.rectangle(1, 1),
                elevation=0.5,
                traversable=False,
            )
        )

        self.assertEqual(
            compiled.surfaces[0].surface.roles, frozenset({SurfaceRole.SUPPORT})
        )

    def test_guarded_atrium_hole_compiles_balustrade_and_mixed_collision(
        self,
    ) -> None:
        compiled = compile_platform(
            PlatformSpec(
                platform_id="gallery",
                space_id="library",
                footprint=Footprint2D(
                    outer=((0, 0), (8, 0), (8, 8), (0, 8)),
                    holes=(((2, 2), (2, 6), (6, 6), (6, 2)),),
                ),
                elevation=3.0,
                guarded_hole_indices=(0,),
            )
        )

        self.assertAlmostEqual(compiled.visual_mesh.bounds[1][2], 4.1)
        self.assertAlmostEqual(compiled.collision_mesh.bounds[1][2], 3.0)
        guard_surfaces = [
            patch
            for patch in compiled.surfaces
            if patch.surface.metadata.get("structure_type") == "platform_guard"
        ]
        self.assertEqual(len(guard_surfaces), 4)
        self.assertTrue(
            all(
                patch.surface.metadata["guard_style"] == "Renaissance posts and rails"
                for patch in guard_surfaces
            )
        )
        self.assertGreaterEqual(len(compiled.collision_primitives), 12)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, temporary_directory)
            sdf = ET.parse(paths.sdf_path)
        self.assertEqual(len(sdf.findall(".//collision/geometry/mesh")), 1)
        self.assertEqual(
            len(sdf.findall(".//collision/geometry/box")),
            len(compiled.collision_primitives),
        )


class TestHeightfields(unittest.TestCase):
    def test_grid_compiles_to_queryable_sloped_triangles(self) -> None:
        compiled = compile_heightfield(
            HeightfieldSpec(
                heightfield_id="terrain",
                space_id="cavern",
                heights=((0, 0.1, 0.2), (0.2, 0.3, 0.4), (0.4, 0.5, 0.6)),
                cell_size=(1, 1),
                origin=(-1, -1, 2),
            )
        )

        self.assertEqual(len(compiled.visual_mesh.vertices), 9)
        self.assertEqual(len(compiled.visual_mesh.triangles), 8)
        self.assertEqual(len(compiled.surfaces), 8)
        self.assertEqual(compiled.visual_mesh.bounds, ((-1, -1, 2), (1, 1, 2.6)))
        self.assertEqual(len(compiled.triangle_groups["traversable"]), 8)
        self.assertTrue(
            all(
                SurfaceRole.SUPPORT in patch.surface.roles
                for patch in compiled.surfaces
            )
        )

    def test_heightfield_budget_is_enforced_before_compilation(self) -> None:
        heightfield = HeightfieldSpec(
            heightfield_id="terrain",
            space_id="cavern",
            heights=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        )
        with self.assertRaisesRegex(GeometryValidationError, "budget"):
            compile_heightfield(heightfield, max_triangles=7)


class TestCompiledStructureExport(unittest.TestCase):
    def test_mesh_audit_reports_closed_oriented_positive_volume(self) -> None:
        mesh = TriangleMesh(
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            triangles=((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        )

        audit = audit_triangle_mesh(mesh)

        self.assertTrue(audit.is_closed)
        self.assertTrue(audit.is_winding_consistent)
        self.assertGreater(audit.signed_volume, 0.0)

    def test_default_export_is_an_atomic_content_addressed_bundle(self) -> None:
        compiled = compile_polygon_space(
            structure_id="default_bundle",
            footprint=Footprint2D.rectangle(4, 3),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, temporary_directory)

            self.assertEqual(paths.sdf_path.parent.parent, Path(temporary_directory))
            self.assertEqual(len(paths.sdf_path.parent.name), 64)
            self.assertEqual(
                {path.name for path in paths.sdf_path.parent.iterdir()},
                {
                    "default_bundle.obj",
                    "default_bundle.sdf",
                    "default_bundle.surfaces.json",
                },
            )

    def test_interrupted_default_publish_exposes_no_partial_products(self) -> None:
        compiled = compile_polygon_space(
            structure_id="interrupted_bundle",
            footprint=Footprint2D.rectangle(4, 3),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "scenesmith.agent_utils.structure.compiler.writing.os.replace",
                side_effect=OSError("simulated publish interruption"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_compiled_structure(compiled, temporary_directory)

            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_export_is_content_addressed_and_records_product_hashes(self) -> None:
        compiled = compile_polygon_space(
            structure_id="polygon_room",
            footprint=Footprint2D.rectangle(4, 3),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(
                compiled,
                temporary_directory,
                source_content_hash="source-hash",
                compiler_version="test-compiler-v1",
            )
            sidecar = json.loads(paths.surfaces_path.read_text(encoding="utf-8"))

            self.assertEqual(paths.sdf_path.parent.parent, Path(temporary_directory))
            self.assertEqual(len(paths.sdf_path.parent.name), 64)
            self.assertEqual(sidecar["source_content_hash"], "source-hash")
            self.assertEqual(sidecar["compiler_version"], "test-compiler-v1")
            self.assertEqual(
                set(sidecar["product_hashes"]),
                {"mesh_sha256", "sdf_sha256", "surface_semantics_sha256"},
            )
            self.assertEqual(sidecar["compilation"]["compile_options"], {})

    def test_compile_options_are_part_of_artifact_identity(self) -> None:
        compiled = compile_polygon_space(
            structure_id="option_identity",
            footprint=Footprint2D.rectangle(4, 3),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            coarse = write_compiled_structure(
                compiled,
                temporary_directory,
                source_content_hash="same-source",
                compile_options={"resolution": 1.0},
            )
            fine = write_compiled_structure(
                compiled,
                temporary_directory,
                source_content_hash="same-source",
                compile_options={"resolution": 0.5},
            )

            self.assertNotEqual(coarse.artifact_hash, fine.artifact_hash)
            coarse.artifact_ref.verify()
            fine.artifact_ref.verify()

    def test_distinct_collision_mesh_is_exported_and_authenticated(self) -> None:
        visual = TriangleMesh(
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            triangles=((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        )
        collision = TriangleMesh(
            vertices=((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
            triangles=((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        )
        compiled = compile_polygon_space(
            structure_id="collision_bundle", footprint=Footprint2D.rectangle(2, 2)
        )
        compiled = replace(
            compiled,
            visual_mesh=visual,
            collision_mesh=collision,
            collision_primitives=(),
            surfaces=(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, temporary_directory)
            sdf = ET.parse(paths.sdf_path)

            self.assertIsNotNone(paths.collision_mesh_path)
            self.assertTrue(paths.collision_mesh_path.is_file())
            self.assertEqual(
                sdf.findtext(".//collision/geometry/mesh/uri"),
                paths.collision_mesh_path.name,
            )
            paths.artifact_ref.verify()

    def test_failed_export_publishes_no_partial_artifact(self) -> None:
        compiled = compile_straight_stairs(_straight_stairs_connector())
        invalid_primitive = replace(
            compiled.collision_primitives[0], primitive_type="unsupported"
        )
        invalid = replace(compiled, collision_primitives=(invalid_primitive,))
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(UnsupportedGeometryError):
                write_compiled_structure(invalid, temporary_directory)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_concurrent_equal_exports_publish_one_complete_artifact(self) -> None:
        compiled = compile_polygon_space(
            structure_id="shared_room",
            footprint=Footprint2D.rectangle(4, 3),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = tuple(
                    executor.map(
                        lambda _: write_compiled_structure(
                            compiled,
                            temporary_directory,
                            source_content_hash="shared-source",
                        ),
                        range(32),
                    )
                )

            self.assertEqual(len({item.sdf_path for item in results}), 1)
            artifact_dirs = tuple(Path(temporary_directory).iterdir())
            self.assertEqual(len(artifact_dirs), 1)
            self.assertEqual(len(tuple(artifact_dirs[0].iterdir())), 3)

    def test_stairs_write_obj_sdf_and_surface_sidecar(self) -> None:
        connector = _straight_stairs_connector()
        compiled = compile_straight_stairs(connector)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, temporary_directory)

            self.assertTrue(paths.mesh_path.exists())
            self.assertTrue(paths.sdf_path.exists())
            self.assertTrue(paths.surfaces_path.exists())
            mesh_text = paths.mesh_path.read_text(encoding="utf-8")
            self.assertIn("vn ", mesh_text)
            self.assertIn("f 1//1 3//3 2//2", mesh_text)

            sdf = ET.parse(paths.sdf_path)
            self.assertEqual(sdf.findtext(".//model/static"), "false")
            collisions = sdf.findall(".//collision")
            self.assertEqual(len(collisions), 18)
            self.assertTrue(
                all(node.find("geometry/box") is not None for node in collisions)
            )
            self.assertEqual(
                sdf.findtext(".//visual/geometry/mesh/uri"), paths.mesh_path.name
            )

            sidecar = json.loads(paths.surfaces_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["structure_id"], "main_stairs")
            self.assertEqual(len(sidecar["surfaces"]), 18)
            self.assertEqual(sidecar["bounds"], [[0.0, -0.6, 0.0], [4.0, 0.6, 3.0]])

    def test_ramp_uses_closed_mesh_collision(self) -> None:
        connector = ConnectorSpec(
            connector_id="ramp",
            connector_type=ConnectorType.RAMP,
            start=ConnectorEndpoint("a", "ground", (0, 0, 0)),
            end=ConnectorEndpoint("b", "upper", (12, 0, 1)),
        )
        compiled = compile_straight_ramp(connector)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(compiled, Path(temporary_directory))
            sdf = ET.parse(paths.sdf_path)
            collision_uri = sdf.findtext(".//collision/geometry/mesh/uri")
            self.assertEqual(collision_uri, "ramp.obj")

    def test_writer_can_preserve_room_sdf_link_contract(self) -> None:
        compiled = compile_polygon_space(
            structure_id="polygon_room",
            footprint=Footprint2D.rectangle(4, 3),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_compiled_structure(
                compiled,
                temporary_directory,
                model_name="room_geometry",
                link_name="room_geometry_body_link",
            )
            root = ET.parse(paths.sdf_path).getroot()

            self.assertEqual(root.find(".//model").attrib["name"], "room_geometry")
            self.assertIsNotNone(root.find(".//link[@name='room_geometry_body_link']"))


class TestFreeformStructuralMesh(unittest.TestCase):
    @staticmethod
    def chamber_mesh(*, duplicate_face: bool = False) -> TriangleMesh:
        triangles = (
            (0, 1, 2),
            (0, 2, 3),  # floor, +Z
            (4, 6, 5),
            (4, 7, 6),  # ceiling, -Z
            (0, 5, 1),
            (0, 4, 5),
            (1, 6, 2),
            (1, 5, 6),
            (2, 7, 3),
            (2, 6, 7),
            (3, 4, 0),
            (3, 7, 4),
        )
        if duplicate_face:
            triangles += ((0, 1, 2),)
        return TriangleMesh(
            vertices=(
                (0, 0, 0),
                (200, 0, 0),
                (200, 200, 0),
                (0, 200, 0),
                (0, 0, 200),
                (200, 0, 200),
                (200, 200, 200),
                (0, 200, 200),
            ),
            triangles=triangles,
        )

    def test_cavern_import_applies_units_transform_and_auto_classifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory) / "chamber.obj"
            mesh_path.write_text(self.chamber_mesh().to_obj(), encoding="utf-8")
            compiled = compile_structural_mesh(
                StructuralMeshSpec(
                    mesh_id="cavern",
                    space_id="chamber",
                    mesh_path=str(mesh_path),
                    unit_scale=0.01,
                    transform=Transform3D(translation=(10, 20, -2)),
                    require_watertight=True,
                    normal_orientation="interior",
                )
            )

            self.assertEqual(compiled.visual_mesh.bounds, ((10, 20, -2), (12, 22, 0)))
            support = [
                surface
                for surface in compiled.surfaces
                if SurfaceRole.SUPPORT in surface.surface.roles
            ]
            overhead = [
                surface
                for surface in compiled.surfaces
                if SurfaceRole.OVERHEAD in surface.surface.roles
            ]
            self.assertEqual(len(support), 2)
            self.assertEqual(len(overhead), 2)
            self.assertTrue(
                all(
                    surface.surface.metadata["space_id"] == "chamber"
                    for surface in support
                )
            )

    def test_authored_annotation_overrides_normal_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory) / "chamber.obj"
            mesh_path.write_text(self.chamber_mesh().to_obj(), encoding="utf-8")
            compiled = compile_structural_mesh(
                StructuralMeshSpec(
                    mesh_id="cavern",
                    space_id="chamber",
                    mesh_path=str(mesh_path),
                    unit_scale=1.0,
                    annotations=(
                        MeshSurfaceAnnotation(
                            annotation_id="hazard",
                            triangle_indices=(0,),
                            roles=frozenset({SurfaceRole.NON_INTERACTIVE}),
                        ),
                    ),
                )
            )

            self.assertEqual(
                compiled.surfaces[0].surface.roles,
                frozenset({SurfaceRole.NON_INTERACTIVE}),
            )
            self.assertFalse(compiled.surfaces[0].surface.metadata["auto_classified"])

    def test_duplicate_faces_require_explicit_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory) / "duplicate.obj"
            mesh_path.write_text(
                self.chamber_mesh(duplicate_face=True).to_obj(), encoding="utf-8"
            )
            spec = StructuralMeshSpec(
                mesh_id="cavern",
                space_id="chamber",
                mesh_path=str(mesh_path),
                unit_scale=1.0,
            )

            with self.assertRaisesRegex(GeometryValidationError, "duplicate"):
                compile_structural_mesh(spec)
            repaired = compile_structural_mesh(spec, repair=True)
            self.assertEqual(len(repaired.visual_mesh.triangles), 12)


if __name__ == "__main__":
    unittest.main()
