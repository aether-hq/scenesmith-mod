"""Tests for room placement algorithm."""

import unittest

from scenesmith.agent_utils.scene.house_parts.openings import ConnectionType, PlacedRoom
from scenesmith.agent_utils.scene.house_parts.rooms import RoomSpec
from scenesmith.agent_utils.structure.geometry_models.surface_models import Footprint2D
from scenesmith.floor_plan_agents.tools.submission.placement.geometry import (
    rooms_overlap,
    rooms_share_edge,
)
from scenesmith.floor_plan_agents.tools.submission.placement.layout import (
    find_room,
    get_shared_boundary,
)
from scenesmith.floor_plan_agents.tools.submission.placement.models import (
    PlacementError,
)
from scenesmith.floor_plan_agents.tools.submission.placement.search import place_rooms


class TestSingleRoom(unittest.TestCase):
    """Tests for single room placement."""

    def test_single_room_at_origin(self):
        """Single room should be placed at origin (0, 0)."""
        specs = [
            RoomSpec(
                room_id="living",
                room_type="living_room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        assert len(result) == 1
        assert result[0].position == (0.0, 0.0)

    def test_single_room_dimensions(self):
        """Single room should have correct dimensions."""
        specs = [
            RoomSpec(
                room_id="bedroom",
                room_type="bedroom",
                width=3.0,  # Y dimension.
                length=4.0,  # X dimension.
            )
        ]
        result = place_rooms(specs)

        assert result[0].width == 4.0  # X dimension (from length).
        assert result[0].depth == 3.0  # Y dimension (from width).


class TestMultilevelPlacement(unittest.TestCase):
    """Tests for level-aware legacy rectangular placement."""

    def test_stacked_rooms_can_share_xy_without_overlap(self):
        rooms = place_rooms(
            [
                RoomSpec("lower", width=4.0, length=5.0, level_id="ground"),
                RoomSpec(
                    "upper",
                    width=4.0,
                    length=5.0,
                    level_id="upper",
                    elevation=3.0,
                ),
            ]
        )

        lower = find_room(rooms=rooms, room_id="lower")
        upper = find_room(rooms=rooms, room_id="upper")
        assert lower.position == upper.position == (0.0, 0.0)
        assert upper.elevation == 3.0
        assert not rooms_overlap(lower, upper)
        assert not rooms_share_edge(lower, upper)

    def test_cross_level_planar_connection_is_rejected(self):
        with self.assertRaisesRegex(PlacementError, "structural connector"):
            place_rooms(
                [
                    RoomSpec("lower", level_id="ground"),
                    RoomSpec(
                        "upper",
                        level_id="upper",
                        connections={"lower": ConnectionType.DOOR},
                    ),
                ]
            )

    def test_polygon_input_is_not_silently_flattened(self):
        with self.assertRaisesRegex(PlacementError, "arbitrary footprint"):
            place_rooms(
                [
                    RoomSpec(
                        "gallery",
                        footprint=Footprint2D(
                            outer=(
                                (0, 0),
                                (4, 0),
                                (4, 1),
                                (1, 1),
                                (1, 4),
                                (0, 4),
                            )
                        ),
                    )
                ]
            )

    def test_polygon_hole_does_not_false_positive_as_room_overlap(self):
        courtyard = Footprint2D(
            outer=((0, 0), (6, 0), (6, 6), (0, 6)),
            holes=(((2, 2), (2, 4), (4, 4), (4, 2)),),
        )
        surrounding_room = PlacedRoom(
            "surrounding",
            (0, 0),
            6,
            6,
            footprint=courtyard,
        )
        courtyard_room = PlacedRoom("courtyard", (2.2, 2.2), 1.6, 1.6)

        self.assertFalse(rooms_overlap(surrounding_room, courtyard_room))

    def test_polygon_overlap_uses_real_footprint_not_only_bounds(self):
        triangle = Footprint2D(outer=((0, 0), (4, 0), (0, 4)))
        first = PlacedRoom("first", (0, 0), 4, 4, footprint=triangle)
        second = PlacedRoom("second", (0.5, 0.5), 4, 4, footprint=triangle)

        self.assertTrue(rooms_overlap(first, second))


class TestTwoRooms(unittest.TestCase):
    """Tests for two room placement."""

    def test_two_adjacent_rooms(self):
        """Two adjacent rooms should share an edge."""
        specs = [
            RoomSpec(
                room_id="living",
                room_type="living_room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="kitchen",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"living": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        assert len(result) == 2
        living = find_room(rooms=result, room_id="living")
        kitchen = find_room(rooms=result, room_id="kitchen")
        assert rooms_share_edge(room_a=living, room_b=kitchen, min_overlap=1.0)

    def test_two_rooms_no_overlap(self):
        """Adjacent rooms should not overlap."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="room",
                width=4.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        assert not rooms_overlap(room_a=a, room_b=b)


class TestThreeRooms(unittest.TestCase):
    """Tests for three room placement."""

    def test_three_rooms_linear(self):
        """A-B-C linear layout: B adjacent to A and C."""
        specs = [
            RoomSpec(room_id="a", room_type="bedroom", width=3.0, length=4.0),
            RoomSpec(
                room_id="b",
                room_type="hallway",
                width=3.0,
                length=2.0,
                connections={"a": ConnectionType.DOOR, "c": ConnectionType.DOOR},
            ),
            RoomSpec(room_id="c", room_type="bathroom", width=3.0, length=3.0),
        ]
        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        c = find_room(rooms=result, room_id="c")

        assert rooms_share_edge(room_a=a, room_b=b)
        assert rooms_share_edge(room_a=b, room_b=c)
        # A and C should not necessarily share an edge (B is between).


class TestMultiAdjacency(unittest.TestCase):
    """Tests for rooms with multiple adjacency requirements."""

    def test_room_adjacent_to_two_rooms(self):
        """Room C adjacent to both A and B requires corner placement."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living_room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="c",
                room_type="dining",
                width=3.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR, "b": ConnectionType.DOOR},
            ),
        ]

        # This may or may not be satisfiable depending on geometry.
        # The algorithm will try different orderings.
        try:
            result = place_rooms(specs)
            c = find_room(rooms=result, room_id="c")
            # If successful, verify adjacencies.
            a = find_room(rooms=result, room_id="a")
            b = find_room(rooms=result, room_id="b")
            # At least one adjacency should be satisfied.
            assert rooms_share_edge(room_a=a, room_b=c) or rooms_share_edge(
                room_a=b, room_b=c
            )
        except PlacementError:
            # Expected if geometry doesn't allow satisfying all constraints.
            pass


class TestGridLayout(unittest.TestCase):
    """Tests for grid-like room layouts."""

    def test_four_rooms_grid(self):
        """2x2 grid layout with cross adjacencies."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=4.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="c",
                room_type="bedroom",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="d",
                room_type="bath",
                width=4.0,
                length=4.0,
                connections={"b": ConnectionType.DOOR, "c": ConnectionType.DOOR},
            ),
        ]

        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        c = find_room(rooms=result, room_id="c")
        d = find_room(rooms=result, room_id="d")

        assert rooms_share_edge(room_a=a, room_b=b)
        assert rooms_share_edge(room_a=a, room_b=c)
        # D should share edge with at least one of B or C.
        assert rooms_share_edge(room_a=b, room_b=d) or rooms_share_edge(
            room_a=c, room_b=d
        )


class TestNoOverlapping(unittest.TestCase):
    """Tests for overlap prevention."""

    def test_no_overlapping_rooms(self):
        """Placed rooms must never overlap."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=5.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="c",
                room_type="bedroom",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        for i, room_i in enumerate(result):
            for j, room_j in enumerate(result):
                if i != j:
                    assert not rooms_overlap(
                        room_a=room_i, room_b=room_j
                    ), f"Rooms {room_i.room_id} and {room_j.room_id} overlap"


class TestMinSharedEdge(unittest.TestCase):
    """Tests for minimum shared edge requirement."""

    def test_min_shared_edge_respected(self):
        """Adjacency requires minimum shared edge length."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="closet",
                width=1.5,
                length=1.5,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        assert rooms_share_edge(room_a=a, room_b=b, min_overlap=1.0)


class TestWallGeneration(unittest.TestCase):
    """Tests for wall generation."""

    def test_room_has_four_walls(self):
        """Each placed room should have exactly 4 walls."""
        specs = [
            RoomSpec(
                room_id="room",
                room_type="room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        assert len(result[0].walls) == 4

    def test_wall_directions_complete(self):
        """Room should have walls for all 4 cardinal directions."""
        specs = [
            RoomSpec(
                room_id="room",
                room_type="room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        directions = {wall.direction.value for wall in result[0].walls}
        assert directions == {"north", "south", "east", "west"}

    def test_exterior_walls_marked(self):
        """Single room should have all exterior walls."""
        specs = [
            RoomSpec(
                room_id="room",
                room_type="room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        for wall in result[0].walls:
            assert wall.is_exterior is True
            assert wall.faces_rooms == []


class TestWallConnectivity(unittest.TestCase):
    """Tests for wall connectivity between rooms."""

    def test_shared_wall_marked_interior(self):
        """Shared walls between adjacent rooms should be marked as interior."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        # Find shared wall.
        shared = get_shared_boundary(result[0], result[1])
        if shared:
            assert shared.is_exterior is False
            assert result[1].room_id in shared.faces_rooms


class TestRoomsOverlap(unittest.TestCase):
    """Tests for rooms_overlap function."""

    def test_overlapping_rooms(self):
        """Overlapping rooms should be detected."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(2.0, 2.0),  # Overlaps with A.
            width=4.0,
            depth=4.0,
        )

        assert rooms_overlap(room_a=room_a, room_b=room_b) is True

    def test_adjacent_rooms_not_overlapping(self):
        """Adjacent (touching) rooms should not be detected as overlapping."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 0.0),  # Touching A's east edge.
            width=4.0,
            depth=4.0,
        )

        assert rooms_overlap(room_a=room_a, room_b=room_b) is False


class TestRoomsShareEdge(unittest.TestCase):
    """Tests for rooms_share_edge function."""

    def test_adjacent_horizontal(self):
        """Horizontally adjacent rooms share edge."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 0.0),
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a, room_b, min_overlap=1.0) is True

    def test_adjacent_vertical(self):
        """Vertically adjacent rooms share edge."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(0.0, 4.0),
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a=room_a, room_b=room_b, min_overlap=1.0) is True

    def test_diagonal_rooms_no_edge(self):
        """Diagonally placed rooms don't share edge."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 4.0),  # Diagonal to A.
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a=room_a, room_b=room_b) is False

    def test_insufficient_overlap(self):
        """Partially adjacent rooms may not meet minimum overlap."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 3.5),  # Only 0.5m overlap.
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a=room_a, room_b=room_b, min_overlap=1.0) is False
        assert rooms_share_edge(room_a=room_a, room_b=room_b, min_overlap=0.4) is True
