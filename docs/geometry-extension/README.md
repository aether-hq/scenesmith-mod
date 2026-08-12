# SceneSmith General-Geometry Upgrade

Current implementation evidence and remaining seams are tracked in
[`IMPLEMENTATION-STATUS.md`](IMPLEMENTATION-STATUS.md).

The next capability layer—semantic branching caves, large caverns, exteriors,
layered substrate, and reproducible destruction—is specified in
[`SEMANTIC-ENVIRONMENTS-SPEC.md`](SEMANTIC-ENVIRONMENTS-SPEC.md), with its
test-first roadmap in
[`SEMANTIC-ENVIRONMENTS-DEVELOPMENT-PLAN.md`](SEMANTIC-ENVIRONMENTS-DEVELOPMENT-PLAN.md).
The semantic cave core plus the first large-cavern aperture/detail/hero slice
are now implemented; the linked status ledger distinguishes those proofs from
the remaining chunking, exterior, and destruction work.

Status: core upgrade implemented; advanced hardening/experiments remain

Date: 2026-08-12
Upstream baseline: `nepfaff/scenesmith` `main`

## Objective

Upgrade SceneSmith from a flat, axis-aligned indoor floor-plan generator into a
general structural scene generator that can represent, generate, furnish,
validate, simulate, and export:

- multiple stacked or offset levels;
- stairs, landings, ramps, ladders, lifts, shafts, and natural passages;
- raised and sunken areas, mezzanines, balconies, bridges, and atria;
- convex, concave, rotated, curved, and holed footprints;
- sloped, stepped, warped, and terrain-like floors and ceilings;
- tunnels, caverns, overhangs, arches, and imported or generated freeform shells;
- mixtures of the above in one connected, simulation-ready scene.

“All geometries” is treated as a finite set of behavioral equivalence classes,
not an attempt to enumerate every possible mesh. The test matrix in
[GEOMETRY-CAPABILITY-MATRIX.md](GEOMETRY-CAPABILITY-MATRIX.md) spans the
independent axes that change generation or downstream behavior: topology,
footprint, elevation profile, enclosure, connector, freeform surface,
degeneracy, and scale.

## Why this is an architectural change

The upstream representation encodes flat rectangular houses in its types and
algorithms:

- `RoomSpec.position` and `PlacedRoom.position` contain only `(x, y)`.
- `PlacedRoom` stores width/depth and documents exactly four walls.
- `WallDirection` is limited to north/south/east/west.
- `HouseLayout.to_drake_directive()` welds every room frame at `z = 0`.
- `RoomGeometry` exposes one rectangular floor, four walls, one wall height,
  and width/length bounds.
- room placement attaches rectangles along four axis-aligned slots.
- furniture bounds, wall surfaces, ceiling placement, clearance zones, and
  renders consume those rectangular assumptions.

Prompt changes alone therefore cannot solve the problem. The upgrade needs a
new structural intermediate representation and adapters for existing agents.

## Design principles

1. **Preserve existing scenes.** A v1 rectangular room must deserialize into
   the new representation and produce the same frame placement and geometry.
2. **Separate topology from mesh.** “Room A connects to Room B by stairs” is
   stable semantic data; a particular triangulation is derived output.
3. **Make surfaces first-class.** Furniture rests on support surfaces, wall
   objects attach to boundary surfaces, and lights attach to overhead surfaces.
   These operations must no longer infer surfaces from a room AABB.
4. **Use three representation tiers.** Parametric primitives cover common,
   editable structures; height fields cover terrain; triangle meshes are the
   escape hatch for caverns and arbitrary shells.
5. **Validate before expensive generation.** Reject self-intersecting
   footprints, invalid connector endpoints, unsafe stairs, non-finite meshes,
   and impossible topology before calling VLM or asset-generation stages.
6. **Keep physics explicit.** Visual mesh, collision mesh, support surfaces,
   traversable surfaces, and attachment surfaces may differ and are tracked
   separately.
7. **Prefer approximations with declared error.** Curves may initially be
   tessellated, but chord tolerance and resulting error must be recorded.
8. **Every new capability needs three proofs.** A deterministic geometry test,
   a downstream simulation/export test, and a prompt-to-structure experiment.

## Target architecture

```text
natural-language prompt
        |
        v
StructuralSceneSpec (semantic, editable, versioned)
  levels + spaces + boundaries + portals + connectors
        |
        v
Structural compiler and validators
  parametric | heightfield | freeform mesh
        |
        +--> RenderMesh / CollisionMesh
        +--> SupportSurface / AttachmentSurface / TraversableSurface
        +--> TopologyGraph / NavigationGraph
        |
        v
Existing SceneSmith stages through compatibility adapters
  furniture -> wall-mounted -> ceiling-mounted -> manipulands
        |
        v
Drake directives + Blender + MuJoCo/USD export
```

### Versioned semantic model

The v2 layout should contain these concepts:

| Concept | Required data | Purpose |
|---|---|---|
| `LevelSpec` | id, elevation, nominal height, optional transform | Stable vertical datum and grouping |
| `SpaceSpec` | id, level, footprint/volume, usage, local transform | Replaces the rectangle-only room definition |
| `Footprint2D` | outer loop, zero or more hole loops | Convex, concave, rotated, courtyard, and shaft plans |
| `ElevationProfile` | planar, stepped, slope, heightfield, mesh | Floor/ceiling Z as a function or surface |
| `BoundarySpec` | path/patch, vertical extent, inside normal, material | Arbitrary wall or cavern boundary semantics |
| `PortalSpec` | source/target, aperture, transform, type | Doors, open joins, windows, arches, cave mouths |
| `ConnectorSpec` | endpoints, type, path, clearance, parameters | Stairs, ramps, ladders, elevators, shafts, passages |
| `StructuralSurface` | geometry ref, role, frame, normal policy | Placement and navigation contract |
| `StructuralMesh` | visual/collision refs, units, transform, provenance | Freeform geometry escape hatch |

The first implementation may extend the current names (`RoomSpec`,
`PlacedRoom`, and `RoomGeometry`) to preserve API compatibility, while the new
types live beside them. A migration function converts v1 dictionaries into v2
objects; serializers always write an explicit `schema_version`.

### Geometry tiers

| Tier | Covers | Editing | Validation | Priority |
|---|---|---:|---:|---:|
| Parametric | levels, polygons, walls, slabs, steps, stairs, ramps, arches | strong | strong | P0 |
| Heightfield | rolling cave floors, outdoor/indoor terrain, warped slabs | moderate | strong | P1 |
| Freeform mesh | caverns, tunnels, overhangs, organic chambers | limited | moderate | P1 |

Freeform meshes are not allowed to erase semantics. They must ship with either
authored surface annotations or derived patches classified as support,
traversable, wall-like, overhead, or non-interactive.

## Compatibility contract

- Existing v1 JSON without `schema_version` is interpreted as v1.
- A v1 `RoomSpec(width, length, position=(x, y))` maps to a v2 rectangular
  footprint on level `ground` at elevation `0`.
- A v1 `PlacedRoom` maps to a space transform with `z=0` and yaw `0`.
- Existing `NORTH/SOUTH/EAST/WEST` identifiers remain valid aliases for the
  four generated edges of rectangular footprints.
- Existing rectangular furniture, wall, and ceiling code remains available
  through an adapter until each stage becomes surface-native.
- Old scene checkpoints round-trip without data loss.
- Export paths and default room/house modes retain their current behavior.

## Definition of done

The upgrade is complete only when:

1. every P0 and P1 test in the capability matrix passes its deterministic
   acceptance predicates;
2. all prior unit tests pass unchanged or through an intentional compatibility
   fixture update;
3. generated multilevel scenes preserve Z transforms in Drake, Blender, and
   at least one experimental MuJoCo/USD export;
4. objects can be placed on non-zero, stepped, sloped, and freeform annotated
   support surfaces without penetrating structure;
5. wall and ceiling attachments use explicit surfaces instead of cardinal/AABB
   inference;
6. connectivity validation distinguishes semantic reachability, geometric
   traversability, and agent-specific accessibility;
7. prompt-to-structure experiments meet the thresholds in
   [DEVELOPMENT-AND-EXPERIMENT-PLAN.md](DEVELOPMENT-AND-EXPERIMENT-PLAN.md);
8. unsupported or invalid inputs fail with a precise diagnostic rather than
   silently flattening, dropping, or corrupting geometry.

## Documents

- [Structural geometry authoring guide](AUTHORING-GUIDE.md)
- [Geometry capability and test matrix](GEOMETRY-CAPABILITY-MATRIX.md)
- [Development and experiment plan](DEVELOPMENT-AND-EXPERIMENT-PLAN.md)
- [Semantic environments specification](SEMANTIC-ENVIRONMENTS-SPEC.md)
- [Semantic environments development and test plan](SEMANTIC-ENVIRONMENTS-DEVELOPMENT-PLAN.md)
- [Implementation status and verified evidence](IMPLEMENTATION-STATUS.md)
