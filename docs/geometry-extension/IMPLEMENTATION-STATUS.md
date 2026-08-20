# Geometry Extension Implementation Status

Updated: 2026-08-14

Baseline: upstream SceneSmith `67cc408fd38334b4a926efef45e284302ed5055b`.

This is an evidence ledger, not a claim that the full capability matrix is
finished. `IMPLEMENTED` means the deterministic semantic/compiler path exists
and has focused tests. `INTEGRATED` additionally means a normal SceneSmith
runtime/export path consumes it. `PARTIAL` names the remaining seam explicitly.
In particular, direct compiler, example, or gallery use is `IMPLEMENTED`, not
five-agent runtime integration.

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
| Boundary-local rectangular portals | INTEGRATED | wall visual/collision cutout, rectangular compatibility adapter, validation tests |
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
| Semantic chamber/passage graph model | PARTIAL | canonical JSON/hash, atomic `set_structural_layout`, and HouseLayout serialization exist; normal runtime compilation is not wired |
| Variable-profile passage compiler | IMPLEMENTED | ellipse/keyhole/slot/arched profiles, variable cross-sections, open boundaries, surface roles |
| Branch and chamber void union | IMPLEMENTED | deterministic implicit union; watertight Y-junction, chamber join, provenance and metamorphic tests |
| Ellipsoid/superellipsoid cavern generation | IMPLEMENTED | rotated analytic chamber fields compile to visual/collision/surface products |
| Passage graph topology adapter | IMPLEMENTED | branches/cycles/dead ends plus bound passage edge → `ConnectorSpec` derivation |
| Generic/LLM-authorability guardrails | IMPLEMENTED | rename/order/translation/subdivision invariance, source sentinel, strict unknown-field rejection |
| Natural sky/exterior apertures | IMPLEMENTED | typed chamber openings compile as real non-watertight apertures with exposure/provenance metadata |
| Seeded geological detail fields | IMPLEMENTED | versioned deterministic sampler, inward-oriented formations, full-envelope route/opening/hero masks, typed exhaustion diagnostic, collision policy, and HouseLayout export |
| Semantic hero geological features | IMPLEMENTED | stable authored anchors compile independently with collision policy and field exclusion |
| Unified derived semantic scene contract | IMPLEMENTED | one `HouseLayout.derive_scene_contract` call derives authenticated products, physical-gated topology/openings, collision queries, and blocked-route evidence; normal runtime does not yet call it |
| Semantic hard invariants | IMPLEMENTED | safe globally unique authored/derived IDs, strict JSON scalar types, bounded work, declared-aperture manifold audits, outward closed formation meshes, and atomic content-addressed artifact bundles |
| Generated genericity suite | IMPLEMENTED | deterministic families cover 1–50 chamber/graph sizes, compiled graph degree 1–5, primitive cross-products, translation, rotation, scale, resolution, and cross-process hash-seed determinism |
| Dragon-scale single cavern | IMPLEMENTED | an actual held-out LLM recipe compiles a 160×100×60 m chamber, approach, real sky aperture, 24 protected coarse formations, and a full-collision hero spire |
| Linux Drake/render gates | CI GATED | generated content-addressed SDF is parsed with collision in Drake and the tunnel shell must produce a nonblank Blender render on Ubuntu CI |
| Held-out LLM authoring evidence | INITIAL EVIDENCE | two retained Claude Sonnet 5 trials pass deterministic requirement oracles; aggregate evidence passes after-repair but not before-repair threshold, and remains below the repeated-corpus sample requirement |
| Normal five-agent semantic scene generation | NOT INTEGRATED | floor-plan tool can store the recipe, but the stateful runner does not derive the semantic contract before furnishing/export; arbitrary-surface consumption remains incomplete across wall, ceiling, and manipuland stages |
| Stable large-scene chunking and advanced formation policies | NOT IMPLEMENTED | bounded chunk manifests/LOD, paired columns, clustering, spawn/sightline masks, cave-mouth terrain seams, and imported hero composition remain planned |
| Exterior terrain/environment seams | NOT IMPLEMENTED | existing heightfields are a foundation, not a complete exterior system |
| Layered substrate and destruction operations | NOT IMPLEMENTED | breach/collapse/fracture/burn/deform operation stack is specified only |

The focused dependency-light semantic/structural suite passes locally. Full
repository discovery still requires optional project runtimes and dependencies
that are not installed in this macOS environment. The reproducible local gate
is:

```bash
.venv/bin/python -m unittest -q \
  tests.unit.test_structural_geometry \
  tests.unit.test_structural_compiler \
  tests.unit.test_semantic_environments \
  tests.unit.test_semantic_environment_compiler \
  tests.unit.test_semantic_environment_details \
  tests.unit.test_semantic_environment_metamorphic \
  tests.unit.test_scene_contract \
  tests.unit.test_house \
  tests.unit.test_floor_plan_tools \
  tests.unit.test_structural_surfaces \
  tests.unit.test_structural_topology \
  tests.unit.test_structural_scenarios \
  tests.unit.test_prison_escape_example
.venv/bin/python scripts/validate_llm_trials.py
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
| Cavern/tunnel generation | PARTIAL | authored chamber/passages, real natural apertures, seeded details, and hero primitives compile directly; automatic five-agent runtime integration plus loft/vaulted/mesh chamber shapes, roughness, and stable chunking remain |
| Elevator/lift | UNSUPPORTED | semantic elevator type fails explicitly; static shaft/landing and dynamic-car compilers remain |
| Shaft connector | PARTIAL | an imported shell may embody a climb-gated shaft; no standalone shaft compiler exists |
| Natural passage connector | INTEGRATED | passage graphs generate the physical shell and derive embedded connector topology/clearance data |
| MuJoCo/USD | PARTIAL | generated OBJ/SDF assets are portable; explicit experimental exporter tests remain |
| Prompt reliability | INITIAL EVIDENCE | two actual held-out trials are retained and revalidated in CI; the statistically meaningful 10-run-per-case protocol remains open |

## Environment evidence and blocker

The local macOS ARM environment has neither the Drake nor Blender Python
runtime, so those two tests report explicit skips locally. Ubuntu CI now runs
both as required integration gates: Drake parses the generated SDF and sees
collision geometry, then Blender renders the generated semantic shell under
`xvfb` and image-content predicates reject blank output.

## Next implementation order

1. integrate the derived semantic scene contract into the normal stateful
   floor-plan runtime and all five agent stages, with full-fidelity room and
   mixed cave/constructed end-to-end regressions;
2. add stable large-scene chunk manifests/LOD plus loft/vaulted chamber forms;
3. extend detail fields with paired columns, clustering, spawn/sightline masks,
   cave-mouth terrain seams, and imported hero composition;
4. add exterior terrain, layered substrate, and reproducible damage operations;
5. expand the retained LLM authorability evidence from the initial two trials
   to the specification's repeated statistical corpus;
6. monitor the supported-host Drake/Blender gates and add more canonical
   semantic scenes as exterior and destruction primitives land.
