# Geometry Extension Implementation Status

Updated: 2026-08-12

Baseline: upstream SceneSmith `67cc408fd38334b4a926efef45e284302ed5055b`.

This is an evidence ledger, not a claim that the full capability matrix is
finished. `IMPLEMENTED` means the deterministic semantic/compiler path exists
and has focused tests. `INTEGRATED` additionally means a normal SceneSmith
runtime/export path consumes it. `PARTIAL` names the remaining seam explicitly.

## Current verified slice

| Capability | State | Evidence |
|---|---|---|
| V1 → V2 migration and round trip | INTEGRATED | `test_structural_geometry`, `test_house` |
| Independent levels/elevations and stacked XY rooms | INTEGRATED | room placement and Drake frame tests |
| Explicit house-frame room XY placement | INTEGRATED | structural authoring tool rebuilds placed rooms/walls after overrides |
| Level-aware polygon/rotated-room overlap | INTEGRATED | exact footprint intersection respects holes and stacked levels |
| Room yaw in Drake/combined-house transforms | INTEGRATED | directive tests |
| Convex/concave polygon rooms | INTEGRATED | constrained triangulation, runtime room adapter |
| Footprint holes/courtyards/atria | INTEGRATED | area/void collision and surface-query tests |
| Independent floor/ceiling slab openings | INTEGRATED | stair opening and solid-roof cross-feature tests |
| Planar sloped floors and ceilings | INTEGRATED | analytic normals, mesh and placement tests |
| Boundary-local rectangular portals | INTEGRATED | wall visual/collision cutout, legacy opening adapter, validation tests |
| Straight stairs | INTEGRATED | mesh, analytic step collision, treads, SDF, house export |
| L stairs with landing | INTEGRATED | multisegment compiler and dispatch tests |
| U/switchback stairs with landing | INTEGRATED | direction/landing validation and compiler tests |
| Spiral stairs | INTEGRATED | center/radius/turn closure, tread-depth and riser validation |
| Straight and switchback ramps | INTEGRATED | slope validation, mesh collision, surface normals |
| Vertical ladders | INTEGRATED | rung/rail compiler, rung-spacing validation, climb-gated topology |
| Raised/sunken platforms and plinths | INTEGRATED | platform compiler and room-frame export |
| Mezzanines/balconies/bridges/catwalks | INTEGRATED | platform footprints and explicit open-edge roles |
| Capability-aware topology | INTEGRATED | walk/climb/view reachability and blocked-edge veto |
| Heightfield terrain/floors | INTEGRATED | grid validation/compiler, slope classification, SDF export |
| Heightfield floor replacement | INTEGRATED | `replaces_floor` suppresses the default flat slab |
| Heightfield point queries | INTEGRATED | compiler-consistent triangular height and normal interpolation |
| Imported cavern/freeform meshes | INTEGRATED | units/transforms, budgets, winding/topology checks, SDF export |
| Cavern as the room shell | INTEGRATED | explicit `replaces_room_shell`; no surrounding flat box |
| Cavern surface semantics | INTEGRATED | authored annotations plus normal/slope auto-classification |
| Surface-native furniture position | INTEGRATED | polygon holes, slopes, stacked support selection |
| Surface-native furniture orientation | INTEGRATED | normal/tangent frame converted to roll/pitch/yaw |
| Surface-native wall mounting | INTEGRATED | arbitrary boundary panels, exact local polygon, true plane transform |
| Surface-native ceiling mounting | INTEGRATED | authored overhead height/tangent frame and void rejection |
| Local headroom/radius clearance | INTEGRATED | predicates sampled along walk connector centerlines; blocked topology edges |
| Embedded cavern passage/shaft | INTEGRATED | semantic centerline + clearance envelope over imported/room shell; no duplicate model |
| Text-agent structural authoring | INTEGRATED | atomic `set_structural_layout` tool and prompt routing |
| Typed invalid/unsupported diagnostics | INTEGRATED | footprint, reference, connector, mesh, portal tests |

The focused dependency-light regression command currently passes 170 tests:

```bash
.venv/bin/python -m unittest \
  tests.unit.test_structural_geometry \
  tests.unit.test_structural_compiler \
  tests.unit.test_structural_topology \
  tests.unit.test_structural_surfaces \
  tests.unit.test_structural_scenarios \
  tests.unit.test_prison_escape_example \
  tests.unit.test_house \
  tests.unit.test_room_placement \
  tests.unit.test_floor_plan_tools -q
```

## Matrix coverage

Implemented deterministic coverage includes the core/model/compiler layers of:

- G-001–G-007 (compatibility, elevations, stacking, yaw);
- G-010–G-017 (polygon spaces, concavity, and holes);
- G-030–G-034 (platforms and analytic slopes; split-level composition uses
  platforms plus connectors);
- G-040–G-048 (straight/L/U/spiral stairs, ramps, and ladders);
- G-061–G-065 (atria via holes; mezzanine/balcony/bridge/catwalk slabs);
- G-066 (sloped ceiling representation, query, and mounted-object transform);
- G-070 and G-074–G-075 foundation (freeform mesh import, annotated surfaces,
  mixed parametric/freeform assets, and embedded natural-passage centerlines);
- X-001–X-012 and X-016 for the implemented representations;
- G-018 (bounded-error circular tessellation), G-035–G-036 (sampled surface
  query/compiler agreement), and local G-076–G-077 clearance predicates;
- the structural predicate path needed by P-001–P-005 and P-007–P-011.

These rows are not marked completely done under the matrix completion rule
until the unavailable/full runtime layers (Drake load, Blender render, and
repeated prompt trials) also pass.

## Known partial seams

| Area | State | Remaining work |
|---|---|---|
| Full SE(3) room frames | PARTIAL | yaw + XYZ are integrated; imported meshes support roll/pitch, room frames do not yet |
| Curved walls/vaults/domes/arches | PARTIAL | circles have bounded tessellation and freeform meshes work; general curve source mapping and parametric vault/arch generators are not built |
| Portal shape breadth | PARTIAL | rectangular cutouts work; true arched/nonrectangular apertures remain mesh-tier |
| Headroom/agent-radius clearance | PARTIAL | local predicates and connector centerline sampling are integrated; simulator swept-volume confirmation remains |
| Cavern/tunnel generation | PARTIAL | imported shells can replace rooms and combine with built structures; automatic tunnel centerlines/clearance envelopes are not generated |
| Elevator/lift | UNSUPPORTED | semantic elevator type fails explicitly; static shaft/landing and dynamic-car compilers remain |
| Shaft connector | PARTIAL | an imported shell may embody a climb-gated shaft; no standalone shaft compiler exists |
| Natural passage connector | PARTIAL | embedded shell routes are integrated with topology and geometric clearance; no standalone passage solid/boolean compiler exists |
| MuJoCo/USD | PARTIAL | generated OBJ/SDF assets are portable; explicit experimental exporter tests remain |
| Prompt reliability | NOT RUN | requires configured model/API trials and the P-series experiment protocol |

## Environment evidence and blocker

The repository requires Drake and the current lock resolves `drake==1.49.0`,
whose published wheel is not available for this macOS ARM environment.
Consequently Drake parser/load and Blender/GPU
integration tests are classified `BLOCKED_ENV`, not silently counted as passes.
The dependency-light model, geometry, topology, agent-tool, placement, OBJ, SDF,
and sidecar tests run locally. A Linux x86_64 or supported Drake environment is
required for the Phase 7 simulation gate.

## Next implementation order

1. add deterministic general arc/spline source mapping and parametric
   vault/arch cases;
2. implement standalone shaft/elevator static geometry and an explicit
   dynamic-car policy;
3. add automatic cavern/tunnel centerline and clearance-envelope extraction;
4. run Drake/Blender integration on supported infrastructure;
5. execute P-001–P-012 repeated prompt trials and record pass rates.
