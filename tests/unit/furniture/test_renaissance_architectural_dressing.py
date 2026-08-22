"""Artifact regressions for deterministic Renaissance room dressing."""

import tempfile

from pathlib import Path

import trimesh

from scenesmith.agent_utils.design.renaissance_dressing import (
    write_renaissance_dressing_visuals,
)
from scenesmith.agent_utils.scene.house import HouseLayout
from scenesmith.agent_utils.scene.house_parts.openings import (
    Opening,
    OpeningType,
    PlacedRoom,
    Wall,
    WallDirection,
    WindowShape,
)
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.structure.geometry_models.surface_models import (
    Footprint2D,
    PlatformSpec,
)


def _library_layout(prompt: str) -> HouseLayout:
    south = Wall(
        wall_id="library_south",
        room_id="library",
        direction=WallDirection.SOUTH,
        start_point=(0.0, 0.0),
        end_point=(12.0, 0.0),
        length=12.0,
        openings=[
            Opening(
                opening_id="huge_arch",
                opening_type=OpeningType.WINDOW,
                position_along_wall=4.0,
                width=4.0,
                height=3.5,
                sill_height=0.35,
                shape=WindowShape.ARCHED,
            )
        ],
    )
    east = Wall(
        wall_id="library_east",
        room_id="library",
        direction=WallDirection.EAST,
        start_point=(12.0, 0.0),
        end_point=(12.0, 12.0),
        length=12.0,
    )
    north = Wall(
        wall_id="library_north",
        room_id="library",
        direction=WallDirection.NORTH,
        start_point=(12.0, 12.0),
        end_point=(0.0, 12.0),
        length=12.0,
    )
    west = Wall(
        wall_id="library_west",
        room_id="library",
        direction=WallDirection.WEST,
        start_point=(0.0, 12.0),
        end_point=(0.0, 0.0),
        length=12.0,
    )
    gallery = PlatformSpec(
        platform_id="upper_gallery",
        space_id="library",
        footprint=Footprint2D(
            outer=((-5.8, -5.8), (5.8, -5.8), (5.8, 5.8), (-5.8, 5.8)),
            holes=(((-2.5, -2.5), (-2.5, 2.5), (2.5, 2.5), (2.5, -2.5)),),
        ),
        elevation=4.0,
        guarded_hole_indices=(0,),
    )
    return HouseLayout(
        house_prompt=prompt,
        wall_height=12.0,
        room_specs=[RoomSpec("library")],
        placed_rooms=[
            PlacedRoom(
                "library", (0.0, 0.0), 12.0, 12.0, walls=[south, east, north, west]
            )
        ],
        platforms=[gallery],
    )


def test_renaissance_library_emits_colorized_architectural_dressing() -> None:
    layout = _library_layout(
        "grand Renaissance ornate library with burgundy and antique gold decor"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        visuals = write_renaissance_dressing_visuals(layout, Path(temporary_directory))

        assert len(visuals) == 1
        assert visuals[0]["role"] == "room_structure"
        assert visuals[0]["source_id"] == "renaissance_dressing_library"
        assert visuals[0]["arched_window_surrounds"] == 1
        assert visuals[0]["gallery_panels"] >= 20
        assert "gallery_finials" not in visuals[0]
        artifact = Path(str(visuals[0]["path"]))
        assert artifact.is_file()
        scene = trimesh.load(artifact, force="scene")
        names = set(scene.geometry)
        assert any("antique_gold" in name for name in names)
        assert any("burgundy" in name for name in names)
        assert all(
            geometry.metadata.get("shape") != "radius"
            for geometry in scene.geometry.values()
        )
        bounds = scene.bounds
        assert bounds[0][0] <= -5.9 and bounds[1][0] >= 5.9
        assert bounds[1][1] >= 11.6


def test_plain_modern_room_does_not_emit_renaissance_dressing() -> None:
    layout = _library_layout("plain modern reading room")
    with tempfile.TemporaryDirectory() as temporary_directory:
        assert (
            write_renaissance_dressing_visuals(layout, Path(temporary_directory)) == []
        )
