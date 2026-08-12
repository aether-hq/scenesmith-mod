# Structural Geometry Authoring Guide

SceneSmith keeps the original room-generation workflow, then adds structural
data with `set_structural_layout`. Call it immediately after
`generate_room_specs`, before doors, windows, materials, or furnishing.

## Coordinate conventions

- Room `position` is the room footprint's minimum `[x, y]` corner in the house
  frame. Rooms on different levels may intentionally use the same position.
- Room footprints are local to that minimum corner. Their bounds must match the
  room's `length × width`; polygon loops may be convex, concave, or holed.
- Connector endpoints and waypoints are `[x, y, z]` in the house structural
  frame.
- Platforms and heightfields are local to their `space_id` room frame.
- Imported structural-mesh transforms are applied in that mesh's room frame.
- Angles in the authoring tool use `yaw_degrees`; serialized `Transform3D`
  rotations use radians in `rotation_rpy`.

Invalid coordinates, references, topology, unsafe stair/ramp parameters, and
unsupported connector solids fail explicitly. They are never flattened into a
default room.

## Two stacked levels with a stair

The lower ceiling and upper floor have the stair opening independently. The
lower floor and upper ceiling remain solid.

```json
{
  "levels": [
    {"id": "ground", "elevation": 0.0, "nominal_height": 3.0},
    {"id": "upper", "elevation": 3.0, "nominal_height": 3.0}
  ],
  "rooms": [
    {
      "id": "lower_hall",
      "level_id": "ground",
      "position": [0.0, 0.0],
      "footprint": {
        "outer": [[0, 0], [6, 0], [6, 4], [0, 4]],
        "holes": []
      },
      "ceiling_footprint": {
        "outer": [[0, 0], [6, 0], [6, 4], [0, 4]],
        "holes": [[[0.5, 1.5], [0.5, 2.5], [5.5, 2.5], [5.5, 1.5]]]
      }
    },
    {
      "id": "upper_hall",
      "level_id": "upper",
      "position": [0.0, 0.0],
      "footprint": {
        "outer": [[0, 0], [6, 0], [6, 4], [0, 4]],
        "holes": []
      },
      "floor_footprint": {
        "outer": [[0, 0], [6, 0], [6, 4], [0, 4]],
        "holes": [[[0.5, 1.5], [0.5, 2.5], [5.5, 2.5], [5.5, 1.5]]]
      }
    }
  ],
  "connectors": [
    {
      "id": "main_stair",
      "type": "stairs_straight",
      "start": {
        "space_id": "lower_hall",
        "level_id": "ground",
        "position": [1.0, 2.0, 0.0]
      },
      "end": {
        "space_id": "upper_hall",
        "level_id": "upper",
        "position": [5.0, 2.0, 3.0]
      },
      "width": 1.0,
      "clearance_height": 2.1,
      "parameters": {"riser_count": 18}
    }
  ]
}
```

The runtime compiles both slabs and the stair, samples the stair route for
support/headroom, and rejects the scene if a slab was accidentally left across
the route.

## Concave, circular, and sloped rooms

- Supply `footprint.outer` and optional `footprint.holes` for polygonal rooms.
- A circle shorthand is available as
  `{"circle": {"radius": 3, "center": [3, 3], "chord_tolerance": 0.02}}`.
- A sloped plane uses, for example,
  `{"type": "sloped", "base_elevation": 0, "gradient": [0.1, 0]}` as
  `floor_profile` or `ceiling_profile`.
- Use `platforms` for discrete raised/sunken slabs, mezzanines, bridges,
  balconies, and catwalks. `open_edge_indices` identifies unguarded/open edges.

## Heightfield as the real floor

```json
{
  "heightfields": [
    {
      "id": "rough_cave_floor",
      "space_id": "cavern",
      "heights": [[0.0, 0.1, 0.0], [0.2, 0.4, 0.2], [0.0, 0.1, 0.0]],
      "cell_size": [1.0, 1.0],
      "origin": [0.0, 0.0, 0.0],
      "replaces_floor": true
    }
  ]
}
```

Set `replaces_floor` only when the grid is the structural floor. Otherwise it
is additive and the room's default slab intentionally remains.

## Cavern shells and natural passages

Use a mesh as the room itself, not as a large furniture asset:

```json
{
  "structural_meshes": [
    {
      "id": "lower_cavern_shell",
      "space_id": "lower_cavern",
      "mesh_path": "assets/lower_cavern.obj",
      "unit_scale": 1.0,
      "normal_orientation": "interior",
      "require_watertight": true,
      "replaces_room_shell": true
    },
    {
      "id": "upper_cavern_shell",
      "space_id": "upper_cavern",
      "mesh_path": "assets/upper_cavern.obj",
      "unit_scale": 1.0,
      "normal_orientation": "interior",
      "require_watertight": true,
      "replaces_room_shell": true
    }
  ],
  "connectors": [
    {
      "id": "rising_tunnel",
      "type": "natural_passage",
      "start": {
        "space_id": "lower_cavern",
        "level_id": "ground",
        "position": [1.0, 2.0, 0.0]
      },
      "end": {
        "space_id": "upper_cavern",
        "level_id": "upper",
        "position": [8.0, 3.0, 3.0]
      },
      "width": 1.5,
      "clearance_height": 2.0,
      "parameters": {
        "geometry_embedded": true,
        "waypoints": [[3.0, 2.2, 0.8], [6.0, 2.7, 2.1]]
      }
    }
  ]
}
```

`geometry_embedded` means the imported shell already physically contains the
entire clear tunnel. SceneSmith adds semantic topology and samples the supplied
centerline against structural support/headroom, but does not generate a second
overlapping model. A near-vertical embedded `shaft` defaults to the `climb`
capability; a natural passage defaults to `walk`.

## Current explicit boundaries

- Straight, L, U, and spiral stairs; straight/turning ramps; ladders; embedded
  natural passages/shafts are supported.
- General nonrectangular portal apertures, standalone tunnel booleans,
  parametric vault/dome generators, and elevator cars are not yet compiled.
- Freeform meshes are the supported escape hatch for caverns, overhangs,
  vaults, domes, arches, and other arbitrary shells.
- First-class branching passage networks, large generated caverns, exterior
  terrain, geological detail fields, layered substrate, and reproducible
  destruction operations are specified but not yet implemented. See
  [Semantic Environments Specification](SEMANTIC-ENVIRONMENTS-SPEC.md) rather
  than treating the prison example as their production authoring API.
- See [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md) for exact evidence
  and environment-blocked integration gates.
