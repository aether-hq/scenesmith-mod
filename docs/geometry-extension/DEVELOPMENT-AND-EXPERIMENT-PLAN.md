# Development and Experiment Plan

## Strategy

The upgrade proceeds from semantic representation to deterministic geometry,
then downstream consumers, then agentic generation. This order isolates geometry
bugs from VLM variability and allows fast tests to guard every later stage.

Phases 0–7 below cover the delivered general-geometry foundation. The roadmap
for first-class branching cave graphs, large generated caverns, exteriors,
geological detail, layered substrate, and reproducible destruction continues in
[Semantic Environments Development and Experiment Plan](SEMANTIC-ENVIRONMENTS-DEVELOPMENT-PLAN.md).

No phase may silently flatten unsupported input. Until a capability is
implemented, validation must return a typed “unsupported geometry” diagnostic.

## Execution status (2026-08-12)

| Phase | State | Current boundary |
|---|---|---|
| 0 — Baseline | PARTIAL | upstream commit and dependency-light evidence recorded; heavy baseline prompts require API/runtime configuration |
| 1 — Semantic core | DELIVERED | v2 migration, levels, footprints, profiles, portals, connectors, transforms, typed diagnostics |
| 2 — Multilevel/connectors | DELIVERED_CORE | stairs, ramps, ladders, platforms, independent slab holes, route veto; Drake load gate is environment-blocked |
| 3 — Parametric spaces | DELIVERED_CORE | concave/holed/circular-tessellated spaces and arbitrary wall panels; general source-mapped splines/arches remain |
| 4 — Surface placement | DELIVERED_CORE | furniture, wall, and ceiling paths consume explicit support/attachment/overhead surfaces |
| 5 — Caverns/terrain | DELIVERED_CORE | replacing freeform shells, heightfields, annotations, embedded passage centerlines; automatic tunnel extraction remains |
| 6 — Agent tools | INTEGRATED | atomic structural authoring tool and prompt grammar exist; repeated P-series trials are not run |
| 7 — Export/hardening | PARTIAL | deterministic OBJ/SDF/sidecars pass; Drake/Blender/MuJoCo/USD matrix remains |
| E0–E2 — Semantic cave core | DELIVERED | canonical regions/chambers/passage graphs, implicit unions, topology adaptation, tool/export integration, and migrated prison example |
| E3 — Large cavern/detail slice | PARTIAL_DELIVERED | real sky apertures, 180×120×70 m compact fixture, seeded formations, full-envelope masks, hero primitives; stable chunks and advanced field policies remain |
| E4–E6 — Exteriors/destruction | PLANNED | terrain seams, layered substrate, breach/collapse/fracture/burn/deform operation compilers remain |
| E7 — Semantic LLM evaluation | PLANNED | public schema exists, but repeated canonical/held-out authoring trials and evidence manifests remain |

Exact evidence and remaining seams are kept in
[`IMPLEMENTATION-STATUS.md`](IMPLEMENTATION-STATUS.md).

## Phase 0 — Baseline and instrumentation

Goal: establish reproducible evidence for existing behavior and failure modes.

Deliverables:

- pin the upstream commit and record environment requirements;
- run/import the light unit-test subset and inventory heavy GPU/Blender tests;
- create golden v1 layout/checkpoint fixtures;
- add geometry test helpers for finite values, area, winding, bounds, normals,
  topology, transforms, and collision/support agreement;
- add a capability result format: `PASS`, `FAIL`, `UNSUPPORTED`, `BLOCKED_ENV`;
- run baseline prompts that demonstrate flattening/failure for P-001 to P-009.

Exit criteria:

- v1 golden fixtures exist;
- baseline results distinguish code failure from unavailable GPU/model assets;
- every matrix row has a machine-readable ID and expected predicate set.

## Phase 1 — Versioned 3D semantic core

Goal: represent levels, arbitrary footprints, surfaces, portals, and connectors
without yet requiring every mesh compiler.

Deliverables:

- `schema_version = 2` layout serialization;
- immutable value types for 2D/3D points, transforms, loops, and bounds;
- `LevelSpec`, `Footprint2D`, elevation profiles, `StructuralSurface`,
  `PortalSpec`, and `ConnectorSpec`;
- `RoomSpec`/`PlacedRoom` compatibility fields: `level_id`, `elevation`, yaw,
  optional footprint;
- v1→v2 migration and v2 round trip;
- validators with stable typed diagnostics;
- topology graph with semantic reachability and connector capability tags.

Tests: G-001–G-008, G-010–G-017 model layer, X-001–X-009.

Exit criteria:

- all old serialization tests pass;
- v1 scenes migrate without changing their physical transforms;
- stacked rooms are legal and distinct in 3D;
- invalid loops and references fail before mesh generation.

## Phase 2 — Multilevel transforms and connectors

Goal: make common multilevel scenes physically real in Drake and Blender.

Deliverables:

- full room-frame SE(3) in house directives and combined scene assembly;
- planar slabs at arbitrary elevations;
- platform, step, straight-stair, L/U-stair, landing, and ramp compilers;
- floor/ceiling cutouts for stairs, ramps, atria, and shafts;
- connector collision, support, traversable, open-edge, and clearance surfaces;
- stair/ramp ergonomic validation with configurable policy;
- level-aware room placement and overlap tests.

Tests: G-003–G-007, G-030–G-034, G-040–G-046, G-060–G-065,
X-007–X-008, P-001–P-005.

Exit criteria:

- P-001 succeeds in at least 8/10 structure-generation trials;
- deterministic stairs and ramps load in Drake with no structural
  penetrations above 2 mm;
- semantic and geometric connectivity agree for the canonical multilevel cases.

## Phase 3 — Arbitrary parametric spaces

Goal: remove cardinal four-wall and rectangular-bound assumptions.

Deliverables:

- polygon-with-holes slab triangulation;
- arbitrary boundary-segment wall extrusion and inward normals;
- portals/openings expressed in boundary-local coordinates;
- curve/arc tessellation with configured chord tolerance and source mapping;
- variable floor/ceiling plane profiles;
- polygon-aware clearance and room overlap;
- compatibility aliases for rectangular cardinal walls.

Tests: G-010–G-020, G-033–G-034, G-060–G-069, X-003–X-006,
X-013, P-003, P-004, P-006.

Exit criteria:

- concave and holed floors never fill their voids;
- wall/wall-mounted tooling addresses boundary IDs without cardinal names;
- curve tessellation error is below its declared tolerance;
- old rectangular rooms remain visually and physically equivalent.

## Phase 4 — Surface-native placement

Goal: make downstream stages independent of AABBs, one flat floor, and one
horizontal ceiling.

Deliverables:

- common surface query API: point containment, pose, normal, tangent frame,
  clearance, role, material, and source structure;
- furniture placement on arbitrary annotated support surfaces;
- slope-aware stable pose policy and max-slope constraints;
- wall attachments on arbitrary planar/tessellated boundary patches;
- ceiling attachments on planar or authored freeform anchors;
- surface-aware collision resolution and clearance zones;
- support graph for manipuland placement on generated furniture and structure.

Tests: placement layer for every P0 row; C-03–C-05.

Exit criteria:

- objects rest at correct Z on flat, raised, stepped, and sloped patches;
- surface normals drive orientation consistently;
- agents receive explicit diagnostics when an object cannot mount to a surface.

## Phase 5 — Heightfields and freeform caverns

Goal: support organic spaces without pretending they are polygonal houses.

Deliverables:

- heightfield schema, mesh compiler, interpolation, and normal queries;
- imported/generated `StructuralMesh` with explicit units and transforms;
- mesh validation/optional repair pipeline;
- semantic surface annotation format and automatic patch proposal by slope,
  normal, curvature, and clearance;
- cavern chamber/tunnel topology, portal apertures, centerlines, and clearance
  envelopes;
- parametric/freeform seam welding or tolerance-aware connector joints;
- collision simplification budgets separate from visual mesh budgets.

Tests: G-035–G-038, G-050–G-051, G-070–G-080, X-009–X-016,
P-007–P-010, C-06–C-08.

Exit criteria:

- canonical cavern cases load, collide, render, and expose useful support
  surfaces;
- no semantic connection is reported geometrically traversable when clearance
  analysis disagrees;
- freeform mesh failures are diagnosable and reproducible.

## Phase 6 — Agent tools and prompt grammar

Goal: allow SceneSmith’s designer/critic/orchestrator agents to create and
repair the new structures from natural language.

Deliverables:

- tools for levels, footprints, cutouts, elevation patches, portals,
  connectors, curves, heightfields, and freeform mesh registration;
- staged workflow: topology proposal → deterministic validation → geometry
  compile → visual critique → downstream furnishing;
- structured tool feedback with repair hints and offending IDs;
- 2D per-level ASCII views plus section/elevation and 3D overview renders;
- critic rubrics for topology, headroom, accessibility, geometric coherence,
  and prompt fidelity;
- prompt examples and few-shot repairs for the P-series corpus.

Tests: P-001–P-012 with repeated trials.

Exit criteria:

- P0 prompt predicates succeed in ≥80% of trials before critic repair and
  ≥95% after the allowed repair loop;
- P1 prompt predicates succeed in ≥65% before repair and ≥85% after repair;
- no trial silently substitutes a flat rectangle for an unsupported request.

## Phase 7 — Export, simulation, and regression hardening

Goal: prove the representation remains simulation-ready across supported
outputs and does not regress upstream behavior.

Deliverables:

- full-transform Drake directives and combined-house assembly;
- Blender structural render parity;
- experimental MuJoCo/USD coverage for each geometry tier;
- navigation/connectivity export or sidecar metadata;
- property/fuzz tests for loops, transforms, profiles, and connector parameters;
- complexity budgets and deterministic simplification;
- performance measurements and golden visual/collision artifacts.

Exit criteria:

- all P0/P1 deterministic rows pass;
- all legacy unit tests pass;
- representative scenes from all three geometry tiers run in Drake;
- export differences are quantified and no unsupported feature is silently lost.

## Experiment protocol

### Deterministic runs

Each geometry case records:

- schema and compiler version;
- random seed;
- visual/collision triangle counts;
- validation diagnostics;
- support/traversable/attachment patch counts;
- topology nodes/edges/components;
- compile time and peak memory;
- export/load result for each target;
- invariant predicate results.

Deterministic tests run without network, LLM, GPU asset generation, or Blender
where possible. Heavy renderer/simulator checks are marked separately rather
than weakening fast unit coverage.

### Prompt trials

For each P-series prompt:

1. run 10 seeds with the same model/configuration;
2. retain the entire tool trace and final semantic spec;
3. evaluate canonical structural predicates automatically;
4. compile only specs that pass semantic validation;
5. run mesh, topology, placement, and export checks;
6. record pre-critic and post-critic pass rates;
7. manually inspect a stratified sample of successes and failures.

Metrics:

- semantic predicate precision/recall;
- compile success rate;
- topology/geometric agreement rate;
- penetration and seam-gap distributions;
- support-surface placement success;
- critic repair success and regression rate;
- tokens, latency, GPU time, and generated-asset cost.

### Fuzz/property testing

Generators should cover:

- simple polygons with 3–30 vertices;
- concave polygons and 0–5 valid holes;
- transforms across practical coordinate ranges;
- 1–8 levels and mixed floor-to-floor heights;
- valid/invalid stair and ramp parameters around policy boundaries;
- heightfields with bounded slopes and deliberate discontinuities;
- mesh corruptions: duplicate faces, holes, inverted normals, nonmanifold edges,
  tiny components, NaNs, and extreme scales.

Properties include round-trip stability, deterministic hashing, no NaNs,
triangle validity, area/volume tolerances, no unintended support inside holes,
and topology invariance under serialization.

## Implementation order inside each phase

Every capability follows the same small loop:

1. add a failing deterministic case from the matrix;
2. implement semantic validation;
3. implement geometry/surface compilation;
4. add topology and placement checks;
5. add Drake/Blender/export smoke coverage;
6. expose the agent tool only after deterministic layers pass;
7. run prompt experiments and feed failures back into tools/prompts;
8. mark the matrix row complete with evidence paths.

## Key risks and decisions

| Risk | Mitigation/decision |
|---|---|
| Changing every consumer at once | Compatibility adapters; migrate one stage at a time |
| Treating caverns as giant generated props | Structural meshes have topology and surface annotations, not furniture semantics |
| Polygon booleans/triangulation instability | Central tolerance policy; deterministic library; aggressive invalid-input tests |
| Visual mesh too expensive for collision | Separate collision representation and explicit simplification budget |
| Stairs look correct but are not traversable | Analytic rise/run/headroom plus collision and route tests |
| Semantic connectivity differs from physics | Track semantic, geometric, and agent-capability reachability separately |
| Curves break stable IDs/openings | Keep source curve coordinates and map tessellated segments back to source IDs |
| LLM invents invalid geometry | Typed tools, deterministic validators, bounded repair loop |
| Existing stages assume local z=0 | Keep room-local frames; elevate/rotate frame first, then migrate surfaces |
| “All geometry” becomes unbounded | Equivalence-class matrix, tiered representation, explicit unsupported diagnostics |

## Immediate implementation slice

The first code slice is intentionally narrow but foundational:

1. introduce v2 semantic primitives and diagnostics;
2. extend rooms/placements with `level_id`, elevation, yaw, and optional
   polygon footprint while keeping v1 defaults;
3. add layout levels/connectors and versioned serialization;
4. update Drake room-frame translation to include Z;
5. add deterministic tests for G-001, G-003–G-006, G-010–G-017, and X-001–X-009;
6. then implement the straight stair/ramp compilers used by P-001 and P-005.
