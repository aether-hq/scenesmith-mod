# Structural Geometry Authoring Guide

SceneSmith uses one five-agent workflow for conventional rooms, multilevel
buildings, caves, and mixed scenes. During the floor-plan stage,
`set_structural_layout` adds the structural data required by the prompt. Call it
immediately after `generate_room_specs`, before doors, windows, materials, or
the furniture, wall-mounted, ceiling-mounted, and manipuland stages.

Semantic structure is not a separate cave-generation mode. Simple rectangular
rooms and complex natural environments must proceed through the same downstream
agents and final export path.

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

## Semantic caverns, passages, openings, and detail

Use the optional top-level `semantic_environment` object when the environment
is best described as navigable voids in rock rather than rooms in a house. The
LLM authors a compact graph and distributions: regions, ellipsoid or
superellipsoid chambers, variable-section passage networks, chamber openings,
seeded detail fields, and a few hero features. SceneSmith derives the unioned
shell, individual formations, collision, surface roles, and provenance.

The core record shape is:

```json
{
  "semantic_environment": {
    "schema_version": 1,
    "regions": [{
      "id": "underpeak",
      "kind": "subterranean",
      "bounds": {"minimum": [-120, -80, -40], "maximum": [120, 80, 100]}
    }],
    "chambers": [{
      "id": "colossal_chamber",
      "region_id": "underpeak",
      "center": [0, 0, 20],
      "size": [180, 120, 70],
      "shape": "superellipsoid"
    }],
    "passage_networks": [{
      "id": "approach_routes",
      "region_id": "underpeak",
      "junctions": [
        {"id": "entrance", "position": [-115, 0, 0]},
        {"id": "hall_entry", "position": [-75, 0, 0], "chamber_id": "colossal_chamber"}
      ],
      "segments": [{
        "id": "approach_tunnel",
        "start_junction_id": "entrance",
        "end_junction_id": "hall_entry",
        "path": [[-115, 0, 0], [-75, 0, 0]],
        "profile": "ellipse",
        "cross_sections": [
          {"station": 0, "width": 12, "height": 14},
          {"station": 1, "width": 15, "height": 16.8}
        ]
      }]
    }],
    "openings": [{
      "id": "sky_break",
      "region_id": "underpeak",
      "source_chamber_id": "colossal_chamber",
      "target": "sky",
      "center": [0, 0, 54],
      "normal": [0, 0, 1],
      "size": [24, 18],
      "depth": 30,
      "weather_exposed": true
    }],
    "detail_fields": [{
      "id": "ceiling_formations",
      "region_id": "underpeak",
      "target_chamber_id": "colossal_chamber",
      "formation_type": "stalactite",
      "surface_role": "overhead",
      "count": 60,
      "min_size": [0.8, 0.8, 2],
      "max_size": [4, 4, 18],
      "seed": 8675309,
      "protect_passage_network_ids": ["approach_routes"],
      "route_clearance": 6,
      "collision_policy": "coarse"
    }],
    "hero_features": [{
      "id": "central_rock_spire",
      "region_id": "underpeak",
      "target_chamber_id": "colossal_chamber",
      "feature_type": "rock_spire",
      "anchor": [30, 10, -5],
      "size": [18, 14, 32],
      "collision_policy": "full"
    }]
  }
}
```

Author the topology and dimensions; do not enumerate formation instances or
mesh triangles. Formation seeds are required for reproducibility. Protected
passage IDs reserve the complete 3D formation envelope around each route.
Openings must start inside their source chamber and extend outward; a `sky`
target removes real shell collision and adds exposure metadata.

Programmatic users call `HouseLayout.compile_semantic_environment(...)` for
the joined shell and `compile_semantic_environment_details(...)` for detail
and hero SDFs before Drake export. The structural authoring tool accepts the
same object atomically and clears stale compiled products when it changes.

Those direct calls describe the current compiler API, not the intended product
workflow. The normal stateful pipeline must compile the environment
automatically after the floor-plan agent and before furnishing. Until that
integration is complete, a successful gallery or direct compiler result proves
the structural subsystem only; it does not prove that all five agents produced
a complete scene.

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
- General nonrectangular room-portal apertures, standalone building booleans,
  parametric vault/dome generators, and elevator cars are not yet compiled.
- Freeform meshes are the supported escape hatch for caverns, overhangs,
  vaults, domes, arches, and other arbitrary shells.
- First-class branching passage networks, ellipsoid/superellipsoid generated
  caverns, sky/exterior chamber apertures, seeded geological detail, and hero
  primitives are implemented. Stable chunking/LOD, paired/clustering and
  sightline detail policies, exterior terrain seams, layered substrate, and
  reproducible destruction operations remain specified work. See
  [Semantic Environments Specification](SEMANTIC-ENVIRONMENTS-SPEC.md) rather
  than treating the prison example as their production authoring API.
- See [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md) for exact evidence
  and environment-blocked integration gates.
