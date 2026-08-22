#!/usr/bin/env python3
"""Generate a lit underground-prison escape tunnel showcase.

The scene is deterministic and uses SceneSmith's semantic environment model:

* a polygon room whose east wall contains a real collision/visual cutout;
* a 69 m irregular, descending variable-profile passage compiled from a graph;
* an embedded natural-passage connector sampled for support and headroom; and
* light fixtures mounted by querying the compiled overhead surface patches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Iterable, Sequence

from scenesmith.agent_utils.core.atomic_output import rebuild_directory_atomically
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import PlacedRoom
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.semantics.environment.models.chambers import (
    Bounds3D,
    EnvironmentRegionSpec,
)
from scenesmith.agent_utils.semantics.environment.models.common import EnvironmentKind
from scenesmith.agent_utils.semantics.environment.models.environment_spec import (
    SemanticEnvironmentSpec,
)
from scenesmith.agent_utils.semantics.environment.models.passages import (
    PassageCrossSectionSpec,
    PassageJunctionSpec,
    PassageNetworkSpec,
    PassageSegmentSpec,
)
from scenesmith.agent_utils.semantics.environment.semantic_environment_compiler import (
    SEMANTIC_ENVIRONMENT_COMPILER_VERSION,
)
from scenesmith.agent_utils.structure.compiler.models import TriangleMesh
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    Footprint2D,
    LevelSpec,
)
from scenesmith.agent_utils.structure.geometry_models.topology_models import (
    PortalSpec,
    PortalType,
)

Point3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


TUNNEL_PATH: tuple[Point3, ...] = (
    (7.0, 0.0, 0.0),
    (11.0, 0.2, -0.2),
    (16.0, 0.8, -0.7),
    (22.0, 1.8, -1.3),
    (29.0, 1.2, -2.0),
    (36.0, 2.5, -2.7),
    (43.0, 4.2, -3.3),
    (50.0, 3.8, -3.9),
    (57.0, 5.5, -4.4),
    (64.0, 7.5, -4.8),
    (70.0, 7.8, -5.0),
    (76.0, 8.0, -5.0),
)
TUNNEL_DIMENSIONS = (
    (3.4, 3.0),
    (3.8, 3.2),
    (4.3, 3.5),
    (4.8, 3.8),
    (4.2, 3.4),
    (5.0, 4.0),
    (4.5, 3.7),
    (5.4, 4.3),
    (5.1, 4.1),
    (6.0, 4.8),
    (7.2, 5.5),
    (8.5, 6.2),
)


def _cross_sections() -> tuple[PassageCrossSectionSpec, ...]:
    spans = tuple(
        math.dist(first, second) for first, second in zip(TUNNEL_PATH, TUNNEL_PATH[1:])
    )
    total = sum(spans)
    distance = 0.0
    stations = [0.0]
    for span in spans:
        distance += span
        stations.append(distance / total)
    return tuple(
        PassageCrossSectionSpec(station, width, height)
        for station, (width, height) in zip(stations, TUNNEL_DIMENSIONS)
    )


TUNNEL_CROSS_SECTIONS = _cross_sections()
TUNNEL_ENVIRONMENT = SemanticEnvironmentSpec(
    regions=(
        EnvironmentRegionSpec(
            "underground_escape",
            EnvironmentKind.SUBTERRANEAN,
            Bounds3D((-5, -15, -15), (90, 25, 20)),
            detail_seed=417,
        ),
    ),
    passage_networks=(
        PassageNetworkSpec(
            "escape_routes",
            "underground_escape",
            (
                PassageJunctionSpec(
                    "breach",
                    TUNNEL_PATH[0],
                    space_id="prison_block",
                    level_id="detention",
                    open_boundary=True,
                ),
                PassageJunctionSpec(
                    "outlet",
                    TUNNEL_PATH[-1],
                    space_id="escape_outlet",
                    level_id="lower_escape",
                    open_boundary=True,
                ),
            ),
            (
                PassageSegmentSpec(
                    "long_way_out",
                    "breach",
                    "outlet",
                    TUNNEL_PATH,
                    TUNNEL_CROSS_SECTIONS,
                ),
            ),
        ),
    ),
)


def _add(a: Point3, b: Point3) -> Point3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: Point3, amount: float) -> Point3:
    return tuple(value * amount for value in vector)  # type: ignore[return-value]


def _normalize(vector: Point3) -> Point3:
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _merge_meshes(meshes: Iterable[TriangleMesh]) -> TriangleMesh:
    vertices: list[Point3] = []
    triangles: list[Triangle] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        triangles.extend(
            tuple(offset + index for index in triangle)  # type: ignore[arg-type]
            for triangle in mesh.triangles
        )
    return TriangleMesh(tuple(vertices), tuple(triangles))


def _oriented_box(
    center: Point3,
    dimensions: Point3,
    axes: tuple[Point3, Point3, Point3] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
) -> TriangleMesh:
    half = tuple(value / 2.0 for value in dimensions)
    vertices = tuple(
        _add(
            center,
            _add(
                _scale(axes[0], sx * half[0]),
                _add(
                    _scale(axes[1], sy * half[1]),
                    _scale(axes[2], sz * half[2]),
                ),
            ),
        )
        for sz in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sx in (-1.0, 1.0)
    )
    triangles = (
        (0, 3, 1),
        (0, 2, 3),
        (4, 5, 7),
        (4, 7, 6),
        (0, 1, 5),
        (0, 5, 4),
        (1, 3, 7),
        (1, 7, 5),
        (3, 2, 6),
        (3, 6, 7),
        (2, 0, 4),
        (2, 4, 6),
    )
    return TriangleMesh(vertices, triangles)


def _section_across(path: Sequence[Point3], index: int) -> Point3:
    previous = path[max(0, index - 1)]
    following = path[min(len(path) - 1, index + 1)]
    dx, dy = following[0] - previous[0], following[1] - previous[1]
    length = math.hypot(dx, dy)
    return (-dy / length, dx / length, 0.0)


def _build_prison_details() -> TriangleMesh:
    meshes: list[TriangleMesh] = []
    # Three barred cell fronts and their bunks make the room read as a prison.
    for cell_center_x in (-4.8, -1.7, 1.4):
        for bar_index in range(6):
            meshes.append(
                _oriented_box(
                    (cell_center_x - 1.25 + bar_index * 0.5, 3.7, 1.4),
                    (0.055, 0.055, 2.8),
                )
            )
        meshes.append(_oriented_box((cell_center_x, 3.7, 0.55), (2.7, 0.08, 0.08)))
        meshes.append(_oriented_box((cell_center_x, 3.7, 2.25), (2.7, 0.08, 0.08)))
        meshes.append(_oriented_box((cell_center_x, 4.35, 0.45), (1.8, 0.65, 0.18)))
        meshes.append(_oriented_box((cell_center_x, 4.35, 1.65), (1.8, 0.65, 0.18)))

    # Broken masonry around the breach.
    rubble = (
        ((6.25, -1.55, 0.18), (0.55, 0.42, 0.35)),
        ((6.52, -1.05, 0.22), (0.42, 0.58, 0.42)),
        ((6.15, 1.45, 0.25), (0.65, 0.48, 0.48)),
        ((5.85, 1.85, 0.16), (0.50, 0.38, 0.30)),
        ((6.35, 1.10, 0.12), (0.35, 0.32, 0.24)),
    )
    meshes.extend(_oriented_box(center, size) for center, size in rubble)
    return _merge_meshes(meshes)


def _write_mesh_sdf(
    mesh: TriangleMesh,
    output_dir: Path,
    *,
    structure_id: str,
    link_name: str,
    color: tuple[float, float, float, float],
    point_lights: Sequence[Point3] = (),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / f"{structure_id}.obj"
    obj_path.write_text(mesh.to_obj(object_name=structure_id), encoding="utf-8")

    sdf = ET.Element("sdf", {"version": "1.12"})
    model = ET.SubElement(sdf, "model", {"name": structure_id})
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", {"name": link_name})
    visual = ET.SubElement(link, "visual", {"name": f"{structure_id}_visual"})
    geometry = ET.SubElement(visual, "geometry")
    mesh_element = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh_element, "uri").text = obj_path.name
    material = ET.SubElement(visual, "material")
    rgba = " ".join(f"{value:.4g}" for value in color)
    ET.SubElement(material, "ambient").text = rgba
    ET.SubElement(material, "diffuse").text = rgba
    if point_lights:
        ET.SubElement(material, "emissive").text = rgba
    collision = ET.SubElement(link, "collision", {"name": f"{structure_id}_collision"})
    collision_geometry = ET.SubElement(collision, "geometry")
    collision_mesh = ET.SubElement(collision_geometry, "mesh")
    ET.SubElement(collision_mesh, "uri").text = obj_path.name

    for index, position in enumerate(point_lights, start=1):
        light = ET.SubElement(
            model,
            "light",
            {"name": f"escape_light_{index:02d}", "type": "point"},
        )
        ET.SubElement(light, "pose").text = " ".join(
            f"{value:.6g}" for value in (*position, 0.0, 0.0, 0.0)
        )
        ET.SubElement(light, "cast_shadows").text = "true"
        ET.SubElement(light, "diffuse").text = "1 0.72 0.38 1"
        ET.SubElement(light, "specular").text = "0.45 0.32 0.18 1"
        attenuation = ET.SubElement(light, "attenuation")
        ET.SubElement(attenuation, "range").text = "11"
        ET.SubElement(attenuation, "constant").text = "0.7"
        ET.SubElement(attenuation, "linear").text = "0.08"
        ET.SubElement(attenuation, "quadratic").text = "0.015"

    ET.indent(sdf, space="  ")
    sdf_path = output_dir / f"{structure_id}.sdf"
    ET.ElementTree(sdf).write(sdf_path, encoding="utf-8", xml_declaration=True)
    return sdf_path


def _sample_light_mounts(layout: HouseLayout) -> tuple[list[dict], TriangleMesh]:
    index = layout.build_structural_surface_index()
    requested: list[tuple[float, float, float]] = [
        (-4.5, -1.8, 0.5),
        (0.0, -1.8, 0.5),
        (4.5, -1.8, 0.5),
    ]
    for first, second in zip(TUNNEL_PATH[1::2], TUNNEL_PATH[2::2]):
        requested.append(
            (
                (first[0] + second[0]) / 2.0,
                (first[1] + second[1]) / 2.0,
                (first[2] + second[2]) / 2.0 + 1.0,
            )
        )

    mounts: list[dict] = []
    fixture_meshes: list[TriangleMesh] = []
    for light_index, (x, y, reference_z) in enumerate(requested, start=1):
        pose = index.overhead_pose(x, y, reference_z=reference_z)
        if pose is None:
            raise RuntimeError(f"no overhead mount surface for light {light_index}")
        inward = _normalize(pose.normal)
        center = _add(pose.position, _scale(inward, 0.07))
        fixture_meshes.append(
            _oriented_box(
                center,
                (0.72, 0.28, 0.10),
                (pose.tangent_x, pose.tangent_y, inward),
            )
        )
        mounts.append(
            {
                "id": f"escape_light_{light_index:02d}",
                "surface_id": pose.surface_id,
                "position": list(center),
                "inward_normal": list(inward),
                "clearance_to_edge": pose.clearance_to_edge,
            }
        )
    return mounts, _merge_meshes(fixture_meshes)


def _write_preview(path: Path, light_mounts: Sequence[dict]) -> None:
    width, height = 1280, 800
    plan_left, plan_right = 72, 1210
    plan_top, plan_bottom = 88, 370
    section_top, section_bottom = 468, 738
    min_x, max_x = -8.0, 80.0
    min_y, max_y = -8.0, 13.0
    min_z, max_z = -7.0, 5.0

    def px(x: float) -> float:
        return plan_left + (x - min_x) / (max_x - min_x) * (plan_right - plan_left)

    def py(y: float) -> float:
        return plan_bottom - (y - min_y) / (max_y - min_y) * (plan_bottom - plan_top)

    def sy(z: float) -> float:
        return section_bottom - (z - min_z) / (max_z - min_z) * (
            section_bottom - section_top
        )

    left_edge: list[tuple[float, float]] = []
    right_edge: list[tuple[float, float]] = []
    for index, (point, dimensions) in enumerate(zip(TUNNEL_PATH, TUNNEL_DIMENSIONS)):
        across = _section_across(TUNNEL_PATH, index)
        width, _ = dimensions
        left_edge.append(
            (
                point[0] - across[0] * width / 2,
                point[1] - across[1] * width / 2,
            )
        )
        right_edge.append(
            (
                point[0] + across[0] * width / 2,
                point[1] + across[1] * width / 2,
            )
        )
    tunnel_plan = left_edge + list(reversed(right_edge))

    def points_xy(points: Sequence[tuple[float, float]]) -> str:
        return " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)

    floor_section = " ".join(
        f"{px(point[0]):.1f},{sy(point[2]):.1f}" for point in TUNNEL_PATH
    )
    ceiling_section = " ".join(
        f"{px(point[0]):.1f},{sy(point[2] + dimensions[1]):.1f}"
        for point, dimensions in zip(TUNNEL_PATH, TUNNEL_DIMENSIONS)
    )
    light_circles_plan = "\n".join(
        f'<circle cx="{px(item["position"][0]):.1f}" '
        f'cy="{py(item["position"][1]):.1f}" r="5" class="light"/>'
        for item in light_mounts
    )
    light_circles_section = "\n".join(
        f'<circle cx="{px(item["position"][0]):.1f}" '
        f'cy="{sy(item["position"][2]):.1f}" r="5" class="light"/>'
        for item in light_mounts
    )
    bars = "\n".join(
        f'<line x1="{px(x):.1f}" y1="{py(3.7):.1f}" '
        f'x2="{px(x):.1f}" y2="{py(5.0):.1f}" class="bar"/>'
        for x in (
            -6.0,
            -5.5,
            -5.0,
            -4.5,
            -4.0,
            -3.5,
            -2.9,
            -2.4,
            -1.9,
            -1.4,
            -0.9,
            -0.4,
            0.2,
            0.7,
            1.2,
            1.7,
            2.2,
            2.7,
        )
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ fill:#f3efe5; font:700 26px sans-serif }}
  .subtitle {{ fill:#b7b0a3; font:15px sans-serif }}
  .label {{ fill:#ded7ca; font:14px sans-serif }}
  .small {{ fill:#999184; font:12px monospace }}
  .room {{ fill:#373b40; stroke:#aeb5bd; stroke-width:3 }}
  .tunnel {{ fill:#4a3528; stroke:#bb8c65; stroke-width:3 }}
  .floor {{ fill:none; stroke:#c89a70; stroke-width:4 }}
  .ceiling {{ fill:none; stroke:#8a6248; stroke-width:4 }}
  .light {{ fill:#ffd369; stroke:#fff1b1; stroke-width:2 }}
  .bar {{ stroke:#78838d; stroke-width:2 }}
  .breach {{ stroke:#121619; stroke-width:8 }}
</style>
<rect width="100%" height="100%" fill="#121619"/>
<text x="52" y="43" class="title">The Long Way Out — underground prison escape</text>
<text x="52" y="67" class="subtitle">Real wall breach · 69 m irregular descending tunnel · surface-mounted ceiling lights</text>
<text x="52" y="104" class="label">PLAN</text>
<rect x="{px(-7):.1f}" y="{py(5):.1f}" width="{px(7)-px(-7):.1f}" height="{py(-5)-py(5):.1f}" class="room"/>
<polygon points="{points_xy(tunnel_plan)}" class="tunnel"/>
<line x1="{px(7):.1f}" y1="{py(-1.8):.1f}" x2="{px(7):.1f}" y2="{py(1.8):.1f}" class="breach"/>
{bars}
{light_circles_plan}
<text x="{px(-5.8):.1f}" y="{py(-3.8):.1f}" class="label">detention block</text>
<text x="{px(8.5):.1f}" y="{py(-2.5):.1f}" class="label">dug breach</text>
<text x="{px(43):.1f}" y="{py(8.2):.1f}" class="label">hand-cut escape tunnel</text>
<text x="{px(71):.1f}" y="{py(12):.1f}" class="label">terminal cavern</text>
<text x="52" y="447" class="label">LONGITUDINAL SECTION</text>
<rect x="{px(-7):.1f}" y="{sy(3.6):.1f}" width="{px(7)-px(-7):.1f}" height="{sy(0)-sy(3.6):.1f}" class="room"/>
<polyline points="{floor_section}" class="floor"/>
<polyline points="{ceiling_section}" class="ceiling"/>
{light_circles_section}
<line x1="{plan_left}" y1="{sy(0):.1f}" x2="{plan_right}" y2="{sy(0):.1f}" stroke="#5c6165" stroke-dasharray="6 7"/>
<text x="{plan_left}" y="{sy(0)-8:.1f}" class="small">detention datum z=0</text>
<text x="{px(61):.1f}" y="{sy(-5)-10:.1f}" class="small">z=-5 m · 6.2 m headroom</text>
<text x="52" y="780" class="small">Yellow markers are fixture meshes positioned from StructuralSurfaceIndex overhead queries; matching SDF point lights are exported.</text>
</svg>\n"""
    path.write_text(svg, encoding="utf-8")


def generate_scene(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    room_footprint = Footprint2D.rectangle(14.0, 10.0)
    breach = PortalSpec(
        portal_id="dug_wall_breach",
        portal_type=PortalType.CAVE_MOUTH,
        source_space_id="prison_block",
        width=3.6,
        height=2.8,
        boundary_loop_index=0,
        boundary_edge_index=1,
        position_along=5.0,
    )
    passage = TUNNEL_ENVIRONMENT.passage_networks[0].to_connector_spec(
        "long_way_out", connector_id="escape_tunnel_route"
    )
    layout = HouseLayout(
        wall_height=3.6,
        levels=(
            LevelSpec("detention", 0.0, 3.6),
            LevelSpec("lower_escape", -5.0, 6.2),
        ),
        room_specs=[
            RoomSpec(
                "prison_block",
                room_type="underground_prison",
                prompt="A decaying underground prison block with a dug escape breach",
                position=(-7.0, -5.0),
                length=14.0,
                width=10.0,
                level_id="detention",
                footprint=room_footprint,
            ),
            RoomSpec(
                "escape_outlet",
                room_type="terminal_cavern",
                prompt="The broad dark chamber reached after the escape tunnel",
                position=(71.0, 3.5),
                length=9.0,
                width=9.0,
                level_id="lower_escape",
            ),
        ],
        placed_rooms=[
            PlacedRoom(
                "prison_block",
                (-7.0, -5.0),
                14.0,
                10.0,
                level_id="detention",
                footprint=room_footprint,
            )
        ],
        connectors=[passage],
        semantic_environment=TUNNEL_ENVIRONMENT,
        portals=[breach],
    )
    layout.validate_structure()
    room_paths = layout.compile_polygon_rooms(output_dir / "structures" / "rooms")
    layout.compile_connectors(output_dir / "structures" / "connectors")
    environment_paths = layout.compile_semantic_environment(
        output_dir / "structures" / "meshes" / "escape_tunnel_shell",
        voxel_size=1.0,
        structure_id="escape_tunnel_shell",
    )
    surface_data = json.loads(environment_paths.surfaces_path.read_text())

    blocked = layout.geometrically_blocked_connectors(
        agent_height=1.9,
        agent_radius=0.45,
        sample_spacing=0.3,
    )
    if blocked:
        raise RuntimeError(
            f"generated tunnel route is geometrically blocked: {blocked}"
        )

    light_mounts, light_mesh = _sample_light_mounts(layout)
    lights_sdf = _write_mesh_sdf(
        light_mesh,
        output_dir / "details" / "lights",
        structure_id="ceiling_lights",
        link_name="lights_link",
        color=(1.0, 0.72, 0.25, 1.0),
        point_lights=[tuple(item["position"]) for item in light_mounts],
    )
    props_sdf = _write_mesh_sdf(
        _build_prison_details(),
        output_dir / "details" / "prison",
        structure_id="prison_details",
        link_name="props_link",
        color=(0.28, 0.31, 0.34, 1.0),
    )

    directive = layout.to_drake_directive(base_dir=output_dir)
    for model_name, sdf_path, link_name in (
        ("prison_details", props_sdf, "props_link"),
        ("ceiling_lights", lights_sdf, "lights_link"),
    ):
        relative = sdf_path.relative_to(output_dir)
        directive += f"""
- add_model:
    name: {model_name}
    file: package://scene/{relative}
- add_weld:
    parent: house_frame
    child: {model_name}::{link_name}"""
    (output_dir / "prison_escape.dmd.yaml").write_text(
        directive + "\n", encoding="utf-8"
    )
    (output_dir / "package.xml").write_text(
        """<?xml version="1.0"?>
<package format="2">
  <name>scene</name>
  <version>0.0.0</version>
  <description>SceneSmith prison escape geometry showcase</description>
  <maintainer email="noreply@example.com">SceneSmith</maintainer>
  <license>MIT</license>
</package>
""",
        encoding="utf-8",
    )

    layout_data = layout.to_dict(scene_dir=output_dir)
    (output_dir / "structural_layout.json").write_text(
        json.dumps(layout_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "name": "The Long Way Out",
        "description": "Underground prison room with a dug breach into a long lit escape tunnel",
        "build": {
            "status": "compiled",
            "rebuilt_from_recipe": True,
            "source_path": Path(__file__)
            .resolve()
            .relative_to(Path(__file__).resolve().parents[2])
            .as_posix(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provider": "semantic-compiler/cpu",
            "compiler_version": SEMANTIC_ENVIRONMENT_COMPILER_VERSION,
        },
        "architecture": {
            "mesh_path": str(
                room_paths["prison_block"].with_suffix(".obj").relative_to(output_dir)
            ),
            "sdf_path": str(room_paths["prison_block"].relative_to(output_dir)),
        },
        "wall_breach": {
            "width_m": breach.width,
            "height_m": breach.height,
            "wall_edge": "east",
        },
        "tunnel": {
            "centerline_length_m": sum(
                math.dist(first, second)
                for first, second in zip(TUNNEL_PATH, TUNNEL_PATH[1:])
            ),
            "sections": len(TUNNEL_CROSS_SECTIONS),
            "minimum_width_m": passage.width,
            "minimum_clearance_height_m": passage.clearance_height,
            "end_elevation_m": passage.end.position[2],
            "visual_triangles": surface_data["visual_triangles"],
            "semantic_source_id": "long_way_out",
            "environment_hash": TUNNEL_ENVIRONMENT.content_hash(),
            "mesh_path": str(environment_paths.mesh_path.relative_to(output_dir)),
            "sdf_path": str(environment_paths.sdf_path.relative_to(output_dir)),
        },
        "lighting": {
            "fixture_count": len(light_mounts),
            "mounts": light_mounts,
        },
        "verification": {
            "blocked_connectors": sorted(blocked),
            "walk_reachable": sorted(
                layout.build_topology().reachable("prison_block", capabilities={"walk"})
            ),
        },
        "entrypoint": "prison_escape.dmd.yaml",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_preview(output_dir / "preview.svg", light_mounts)
    return manifest


def rebuild_scene(output_dir: Path) -> dict:
    """Generate the demo in a fresh tree and publish it atomically."""

    return rebuild_directory_atomically(output_dir, generate_scene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("generated"),
        help="output directory (default: examples/prison_escape/generated)",
    )
    args = parser.parse_args()
    manifest = rebuild_scene(args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
