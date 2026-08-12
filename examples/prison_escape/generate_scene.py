#!/usr/bin/env python3
"""Generate a lit underground-prison escape tunnel showcase.

The scene is deterministic and uses SceneSmith's v2 structural model directly:

* a polygon room whose east wall contains a real collision/visual cutout;
* a 69 m irregular, descending freeform tunnel with annotated floor, walls,
  and overhead surfaces;
* an embedded natural-passage connector sampled for support and headroom; and
* light fixtures mounted by querying the compiled overhead surface patches.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scenesmith.agent_utils.house import HouseLayout, PlacedRoom, RoomSpec
from scenesmith.agent_utils.structural_compiler import (
    CompiledStructure,
    TriangleMesh,
    compile_structural_mesh,
    write_compiled_structure,
)
from scenesmith.agent_utils.structural_geometry import (
    ConnectorEndpoint,
    ConnectorSpec,
    ConnectorType,
    Footprint2D,
    LevelSpec,
    MeshSurfaceAnnotation,
    PortalSpec,
    PortalType,
    StructuralMeshSpec,
    SurfaceRole,
)

Point3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class TunnelSection:
    center: Point3
    width: float
    height: float


TUNNEL_SECTIONS = (
    TunnelSection((7.0, 0.0, 0.0), 3.4, 3.0),
    TunnelSection((11.0, 0.2, -0.2), 3.8, 3.2),
    TunnelSection((16.0, 0.8, -0.7), 4.3, 3.5),
    TunnelSection((22.0, 1.8, -1.3), 4.8, 3.8),
    TunnelSection((29.0, 1.2, -2.0), 4.2, 3.4),
    TunnelSection((36.0, 2.5, -2.7), 5.0, 4.0),
    TunnelSection((43.0, 4.2, -3.3), 4.5, 3.7),
    TunnelSection((50.0, 3.8, -3.9), 5.4, 4.3),
    TunnelSection((57.0, 5.5, -4.4), 5.1, 4.1),
    TunnelSection((64.0, 7.5, -4.8), 6.0, 4.8),
    TunnelSection((70.0, 7.8, -5.0), 7.2, 5.5),
    TunnelSection((76.0, 8.0, -5.0), 8.5, 6.2),
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


def _section_across(sections: Sequence[TunnelSection], index: int) -> Point3:
    previous = sections[max(0, index - 1)].center
    following = sections[min(len(sections) - 1, index + 1)].center
    dx, dy = following[0] - previous[0], following[1] - previous[1]
    length = math.hypot(dx, dy)
    return (-dy / length, dx / length, 0.0)


def build_tunnel_mesh(
    sections: Sequence[TunnelSection] = TUNNEL_SECTIONS,
) -> tuple[TriangleMesh, tuple[MeshSurfaceAnnotation, ...]]:
    """Build an open-ended octagonal tunnel shell with explicit semantics."""

    vertices: list[Point3] = []
    for index, section in enumerate(sections):
        across = _section_across(sections, index)
        x, y, floor_z = section.center
        offsets = (
            (-0.50, 0.00),
            (0.50, 0.00),
            (0.53, 0.20),
            (0.50, 0.68),
            (0.28, 1.00),
            (-0.28, 1.00),
            (-0.50, 0.68),
            (-0.53, 0.20),
        )
        for across_fraction, height_fraction in offsets:
            vertices.append(
                (
                    x + across[0] * across_fraction * section.width,
                    y + across[1] * across_fraction * section.width,
                    floor_z + height_fraction * section.height,
                )
            )

    triangles: list[Triangle] = []
    floor_indices: list[int] = []
    overhead_indices: list[int] = []
    wall_indices: list[int] = []
    ring_size = 8
    for section_index in range(len(sections) - 1):
        current = section_index * ring_size
        following = (section_index + 1) * ring_size
        for side in range(ring_size):
            next_side = (side + 1) % ring_size
            first_index = len(triangles)
            # Winding faces the tunnel interior: floor normals point up,
            # ceiling normals down, and side normals toward the centerline.
            triangles.extend(
                (
                    (current + side, following + next_side, current + next_side),
                    (current + side, following + side, following + next_side),
                )
            )
            target = (
                floor_indices
                if side == 0
                else overhead_indices if side == 4 else wall_indices
            )
            target.extend((first_index, first_index + 1))

    mesh = TriangleMesh(tuple(vertices), tuple(triangles))
    annotations = (
        MeshSurfaceAnnotation(
            "tunnel_floor",
            tuple(floor_indices),
            frozenset({SurfaceRole.SUPPORT, SurfaceRole.TRAVERSABLE}),
        ),
        MeshSurfaceAnnotation(
            "tunnel_overhead",
            tuple(overhead_indices),
            frozenset({SurfaceRole.OVERHEAD, SurfaceRole.ATTACHMENT}),
        ),
        MeshSurfaceAnnotation(
            "tunnel_walls",
            tuple(wall_indices),
            frozenset({SurfaceRole.BOUNDARY, SurfaceRole.ATTACHMENT}),
        ),
    )
    return mesh, annotations


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
    for first, second in zip(TUNNEL_SECTIONS[1::2], TUNNEL_SECTIONS[2::2]):
        requested.append(
            (
                (first.center[0] + second.center[0]) / 2.0,
                (first.center[1] + second.center[1]) / 2.0,
                (first.center[2] + second.center[2]) / 2.0 + 1.0,
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
    for index, section in enumerate(TUNNEL_SECTIONS):
        across = _section_across(TUNNEL_SECTIONS, index)
        left_edge.append(
            (
                section.center[0] - across[0] * section.width / 2,
                section.center[1] - across[1] * section.width / 2,
            )
        )
        right_edge.append(
            (
                section.center[0] + across[0] * section.width / 2,
                section.center[1] + across[1] * section.width / 2,
            )
        )
    tunnel_plan = left_edge + list(reversed(right_edge))

    def points_xy(points: Sequence[tuple[float, float]]) -> str:
        return " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)

    floor_section = " ".join(
        f"{px(section.center[0]):.1f},{sy(section.center[2]):.1f}"
        for section in TUNNEL_SECTIONS
    )
    ceiling_section = " ".join(
        f"{px(section.center[0]):.1f},{sy(section.center[2] + section.height):.1f}"
        for section in TUNNEL_SECTIONS
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
    source_dir = output_dir / "source_assets"
    source_dir.mkdir(parents=True, exist_ok=True)

    tunnel_mesh, annotations = build_tunnel_mesh()
    tunnel_source = source_dir / "escape_tunnel_source.obj"
    tunnel_source.write_text(
        tunnel_mesh.to_obj(object_name="escape_tunnel_source"), encoding="utf-8"
    )

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
    passage = ConnectorSpec(
        connector_id="escape_tunnel_route",
        connector_type=ConnectorType.NATURAL_PASSAGE,
        start=ConnectorEndpoint("prison_block", "detention", TUNNEL_SECTIONS[0].center),
        end=ConnectorEndpoint(
            "escape_outlet", "lower_escape", TUNNEL_SECTIONS[-1].center
        ),
        width=min(section.width for section in TUNNEL_SECTIONS),
        clearance_height=min(section.height for section in TUNNEL_SECTIONS),
        parameters={
            "geometry_embedded": True,
            "waypoints": [list(section.center) for section in TUNNEL_SECTIONS[1:-1]],
        },
    )
    tunnel_spec = StructuralMeshSpec(
        mesh_id="escape_tunnel_shell",
        space_id="prison_block",
        mesh_path=str(tunnel_source),
        unit_scale=1.0,
        annotations=annotations,
        require_watertight=False,
        normal_orientation="unspecified",
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
        structural_meshes=[tunnel_spec],
        portals=[breach],
    )
    layout.validate_structure()
    layout.compile_polygon_rooms(output_dir / "structures" / "rooms")
    layout.compile_connectors(output_dir / "structures" / "connectors")
    layout.compile_structural_meshes(output_dir / "structures" / "meshes")

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
    layout_data["structural_meshes"][0]["mesh_path"] = str(
        tunnel_source.relative_to(output_dir)
    )
    (output_dir / "structural_layout.json").write_text(
        json.dumps(layout_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "name": "The Long Way Out",
        "description": "Underground prison room with a dug breach into a long lit escape tunnel",
        "wall_breach": {
            "width_m": breach.width,
            "height_m": breach.height,
            "wall_edge": "east",
        },
        "tunnel": {
            "centerline_length_m": sum(
                math.dist(first.center, second.center)
                for first, second in zip(TUNNEL_SECTIONS, TUNNEL_SECTIONS[1:])
            ),
            "sections": len(TUNNEL_SECTIONS),
            "minimum_width_m": passage.width,
            "minimum_clearance_height_m": passage.clearance_height,
            "end_elevation_m": passage.end.position[2],
            "visual_triangles": len(tunnel_mesh.triangles),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("generated"),
        help="output directory (default: examples/prison_escape/generated)",
    )
    args = parser.parse_args()
    manifest = generate_scene(args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
