# Semantic Environments Development and Experiment Plan

Status: implementation-ready roadmap

Date: 2026-08-12

Source of truth: [Semantic Environments Specification](SEMANTIC-ENVIRONMENTS-SPEC.md)

Execution status: E0–E2 delivered; the initial E3 aperture/detail/hero slice is
delivered with dependency-light tests. Stable chunking/LOD and the advanced E3
policies listed below remain open.

## 1. Outcome

Promote the prison-escape experiment's hand-authored passage into a general,
LLM-authorable environment system. The production result must create branching
caves, very large caverns, geological detail, exterior terrain, and layered
destroyed structures from semantic data—not from scenario-specific mesh code.

This is an extension of the delivered structural-geometry core. Existing
`LevelSpec`, polygonal rooms, `PortalSpec`, `ConnectorSpec`, heightfields,
`StructuralMeshSpec`, `StructuralSurface`, topology, surface queries, and SDF
export remain the downstream contracts.

## 2. Priorities

Importance ranks from `I5` (essential) to `I1` (specialized). Simplicity ranks
from `S5` (straightforward) to `S1` (highest implementation risk).

| Pri | Capability | I/S | Reason and dependency |
|---|---|---|---|
| P0 | Typed semantic recipes and normalized serialization | I5/S4 | Required before an LLM or compiler can use the feature safely |
| P0 | Swept single passage with variable sections | I5/S3 | Smallest vertical slice that replaces the demo generator |
| P0 | Chamber + passage graph topology | I5/S3 | Enables composition, branches, cycles, and held-out tests |
| P0 | Branch/junction physical blending | I5/S2 | Makes graph semantics physically real |
| P0 | Surface classification and swept clearance | I5/S3 | Prevents attractive but unusable caves |
| P0 | LLM authoring tools and typed repair diagnostics | I5/S3 | Core product requirement, not a later prompt tweak |
| P1 | Large cavern volumes and chunking | I5/S2 | Required for dragon-scale spaces |
| P1 | Sky openings and semantic cave mouths | I5/S3 | Connects subterranean, exterior, light, and exposure semantics |
| P1 | Seeded stalactite/stalagmite/boulder fields | I5/S3 | Provides detail without per-instance LLM authoring |
| P1 | Hero geological features | I4/S4 | Gives intentional landmarks and gameplay anchors |
| P1 | Layered constructed material assemblies | I5/S3 | Prerequisite for meaningful destruction/substrate exposure |
| P1 | Breach and debris operations | I5/S2 | High-value destroyed-interior slice |
| P1 | Terrain patches and cave/building seams | I4/S2 | Shared exterior foundation |
| P2 | Collapse and support invalidation | I5/S1 | Valuable but needs dependency/support reasoning |
| P2 | Fracture, burn, and deform operations | I4/S2 | Broader damage vocabulary after breach/collapse |
| P2 | Cliffs, roads, trenches, and exterior feature stacks | I4/S2 | Builds on terrain and operation composition |
| P2 | Visual/collision LOD and streaming-ready chunk metadata | I4/S1 | Needed for scale; full runtime streaming remains later |
| P3 | Runtime physics-driven destruction | I2/S1 | Explicitly deferred; precompiled damage comes first |
| P3 | Planet-scale terrain | I1/S1 | Outside initial scene-size target |

## 3. Delivery phases

Each phase follows test-first order: semantic tests, compiler tests, downstream
surface/topology tests, exports, then LLM trials. A phase is not complete from a
render alone.

### Phase E0 — Baseline, fixtures, and anti-demo guardrails

Goal: establish proof that future work is general and preserve the current
engine behavior.

Deliverables:

- copy the prison example's section sweep only into a test oracle; do not move
  its hard-coded sections into production;
- add semantic-environment result categories: `PASS`, `FAIL`, `UNSUPPORTED`,
  and `BLOCKED_ENV`;
- add reusable assertions for mesh validity, watertightness, seam tolerance,
  surface provenance, swept clearance, topology isomorphism, collision bounds,
  and content hashes;
- add a source-sentinel test that rejects canonical fixture IDs or scenario
  names in core compiler modules;
- preserve the existing 170-test dependency-light structural suite as the
  compatibility baseline;
- record compile time, memory, triangle/chunk counts, diagnostics, topology,
  surfaces, and hashes in a machine-readable result manifest.

Exit gate:

- GEN-001, GEN-002, GEN-008, and the legacy regression command pass;
- the prison example is provably an example consumer, not an imported core
  implementation dependency.

### Phase E1 — Typed semantic environment model

Goal: make cave, terrain, material, detail, and damage recipes serializable,
validatable, compact, and function-tool friendly.

Production types:

- `EnvironmentRegionSpec`, `Bounds3D`, and `ChunkPolicy`;
- `CavernChamberSpec` and explicit chamber shape/profile types;
- `PassageNetworkSpec`, `PassageJunctionSpec`, `PassageSegmentSpec`,
  `PassagePathSpec`, and `PassageCrossSectionSpec`;
- generalized `OpeningSpec` plus a compatibility conversion to/from
  rectangular `PortalSpec` where lossless;
- `DetailFieldSpec`, distribution records, exclusion masks, and
  `HeroFeatureSpec`;
- `MaterialAssemblySpec`, `MaterialLayerSpec`, and typed reinforcement/debris
  policies;
- `DamageOperationSpec` with typed operation payloads rather than a free-form
  parameter map;
- `TerrainRegionSpec`, `TerrainPatchSpec`, and typed terrain feature records;
- normalized `to_dict`/`from_dict`, version migration, stable hashing, and
  reference validation for every type.

Implementation locations:

- extend `scenesmith/agent_utils/structural_geometry.py` only for concepts that
  share existing structural contracts;
- place environment-specific types in
  `scenesmith/agent_utils/semantic_environments.py` to keep the existing module
  reviewable;
- add typed diagnostics in the existing geometry diagnostic hierarchy;
- export the public types from `scenesmith/agent_utils/__init__.py` only after
  their schema is stable.

Tests:

- MODEL-001–MODEL-008 and the schema/reference portions of ADV-001–ADV-004,
  ADV-009, and ADV-013;
- GEN-003 rename invariance and GEN-007 declaration-order invariance;
- compactness/token checks against hand-authored canonical recipes.

Exit gate:

- all canonical recipes round-trip without derived meshes or instances;
- invalid references and geometry values identify the semantic ID and field;
- the dragon cavern semantic JSON satisfies the 40-object/4,000-token budget.

### Phase E2 — Passages, chambers, and graph compiler

Goal: compile single, branching, cyclic, and multilevel cave networks into
physical void boundaries with shared structural semantics.

Work sequence:

1. implement an analytic variable-section passage sweep for straight and curved
   non-branching segments;
2. compile ellipsoid/superellipsoid/loft chamber fields;
3. implement a backend interface that represents void union independently of
   tessellation;
4. blend degree-1 through degree-5 passage junctions;
5. extract stable visual/collision chunks and semantic surface provenance;
6. bind endpoints to chambers, current portals, and natural openings;
7. generate `ConnectorSpec`/topology adapters and full swept-agent clearance;
8. add terrain/constructed seam hooks without implementing all terrain yet.

The graph compiler must not depend on a special `TunnelSection` sequence or
octagonal cross-section. The prison demo should be rewritten as an ordinary
`PassageNetworkSpec` consumer after these tests pass.

Tests:

- CAV-001–CAV-009;
- GEN-004 transform, GEN-005 scale, GEN-006 segment-subdivision, and GEN-009
  generated-family properties;
- ADV-001–ADV-006;
- HYB-001 current room-to-passage and HYB-002 imported chamber-to-passage.

Exit gate:

- all graph topologies agree across semantic, compiled, and navigation graphs;
- all required walk routes pass a swept capsule/clearance test;
- a renamed and transformed network compiles without production-code changes;
- the prison escape demo uses only public environment primitives.

### Phase E3 — Large caverns, openings, detail fields, and hero features

Goal: produce the dragon-cavern scale and visual vocabulary while keeping
routes and topology deterministic.

Current slice (2026-08-12): `PARTIAL_DELIVERED`. A data-only 180×120×70 m
cavern compiles as one chamber/passage shell with a real sky aperture, 60
deterministic formations, and an explicit hero spire. Formation masks protect
full 3D envelopes across 50 seeds, visual-only collision suppression exports
correctly, and exhausted fields fail with `no_legal_detail_samples`.

Deliverables:

- stable spatial chunking with bounded visual/collision budgets;
- ceiling/floor/ledge/overhead/boundary extraction for large chambers;
- natural sky openings and cave mouths with exposure metadata;
- seeded Poisson/stratified formation sampling on eligible surfaces;
- stalactite, stalagmite, paired column, flowstone, boulder, rubble, and scree
  primitive compilers;
- protected route, portal, spawn, sightline, and hero-feature masks;
- hero anchors and optional imported hero-mesh composition;
- deterministic lighting-anchor queries for emissive/fixture placement without
  baking demo lights into geometry code.

Tests:

- CAV-010–CAV-014;
- FORM-001–FORM-007 across at least 50 seeds for mask-sensitive fields;
- GEN-010 held-out cave compositions;
- ADV-007 contradictory sky opening and ADV-008 no-legal-sample field.

Implemented evidence currently covers CAV-007, the single-shell/compactness
portion of CAV-010, FORM-001/FORM-002, deterministic distribution and collision
parts of FORM-004, 50-seed route/opening/hero exclusions, FORM-007 primitive
hero composition, and ADV-008. The remaining cases are still required before
marking all of E3 complete.

Initial large-scene budget:

- semantic bounds up to 500 m on the longest axis;
- no visual chunk above 250,000 triangles and no collision chunk above 50,000
  triangles without an explicit override;
- dragon-cavern reference compile under 60 seconds and 2 GiB peak memory on
  the recorded CI reference host;
- chunking must not change topology, stable semantic IDs, or protected-route
  results.

Exit gate:

- the semantic dragon cavern renders, collides, and supports fly/walk queries;
- every formation seed preserves protected routes and portal clearance;
- sky exposure comes from a real aperture with no ceiling collision.

### Phase E4 — Exteriors and environment seams

Goal: extend the shared representation to outdoor terrain without forking
placement, navigation, or export semantics.

Deliverables:

- bounded heightfield, analytic, terraced, and mesh terrain patches;
- ordered cliff, ridge, berm, trench, plateau, road/path, riverbed, and
  building-pad features;
- exterior `StructuralSurface` roles and sky/weather exposure metadata;
- cave-mouth, sinkhole, cellar, constructed doorway, and foundation seams;
- terrain-aware support/navigation queries and agent-specific slope policies;
- surface-cover fields using the same seeded distribution/mask machinery as
  cavern formations.

Tests:

- EXT-001–EXT-008;
- HYB-003 building on terrain, HYB-004 cave mouth in terrain, and HYB-005
  imported cliff plus semantic path;
- CAV-012 sky opening remains consistent when terrain is present above part of
  a chamber.

Exit gate:

- one continuous route can travel exterior → cave mouth → branching cave →
  constructed interior with no false topology edge or collision seam;
- exterior scenes use the same surface query and export APIs as interiors.

### Phase E5 — Layered substrate and breach/debris operations

Goal: make a destroyed wall structurally meaningful and visually expose its
authored construction.

Deliverables:

- material assemblies bound to polygonal walls, slabs, roofs, columns, and
  freeform substrate regions;
- layer-aware extrusion and face provenance;
- deterministic breach volumes using box, cylinder, ellipsoid, polygonal
  prism, and bounded irregular profiles;
- passability policy that creates an `OpeningSpec` only after compiled
  clearance passes;
- debris generation from removed layer volumes and material-specific recipes;
- visual/collision/navigation/support invalidation from one operation result;
- ordered-operation normalization, hashing, and diagnostics.

Tests:

- DMG-001–DMG-008;
- GEN-011 operation reorder where disjoint and GEN-012 expected order
  sensitivity where overlapping;
- ADV-009–ADV-011;
- AUTH-007 and AUTH-008 prompt trials.

Exit gate:

- a layered prison wall breach exposes the configured material order and is a
  physical, passable opening;
- a small decorative hole remains non-passable;
- repeating the same operation stack produces equal semantic and mesh hashes.

### Phase E6 — Collapse, fracture, burn, and deformation

Goal: cover destroyed interiors beyond simple holes while preserving explicit
simulation limits.

Deliverables:

- static support-dependency graph for slabs, platforms, selected walls, beams,
  and columns;
- bounded collapse operations that remove/displace affected components and
  invalidate dependent support patches;
- structural versus visual fracture policies;
- material-state burn transitions;
- bounded deformation fields;
- debris fields derived from damaged source layers;
- typed diagnostics for operations requiring unsupported dynamic simulation.

Tests:

- DMG-009–DMG-016;
- ADV-012 cyclic support dependency and ADV-013 out-of-bounds deformation;
- cross-products between collapse, passages under buildings, and exterior
  terrain support.

Exit gate:

- a collapsed ceiling no longer reports support where geometry was removed;
- routes blocked or opened by collapse agree in collision and topology;
- requested dynamic destruction is explicitly reported unsupported rather
  than approximated silently.

### Phase E7 — LLM authoring, evaluation, and export hardening

Goal: prove that semantic primitives—not expert-authored Python—are the normal
authoring path.

Deliverables:

- add environment sections to `set_structural_layout` or introduce an atomic
  `set_semantic_environment` tool with staged validation;
- generate concise JSON schemas with enums, defaults, descriptions, and
  examples that do not leak held-out prompts;
- topology → shape → detail/damage staged tool flow;
- deterministic repair feedback with IDs, field paths, measurements, limits,
  and suggested changes;
- per-level plan, longitudinal section, topology graph, and 3D structural
  preview artifacts for critic review;
- experiment runner that records model/config/tool version, seeds, traces,
  normalized specs, predicate results, cost, and latency;
- Drake/Blender gates on supported infrastructure and experimental MuJoCo/USD
  coverage for each new representation path.

Tests:

- AUTH-001–AUTH-012 with 10 trials each;
- all held-out GEN cases;
- legacy floor-plan prompt trials to detect regressions;
- export cases EXP-001–EXP-006.

Exit gate:

- canonical prompts achieve at least 80% before repair and 95% after repair;
- held-out prompts achieve at least 65% before and 85% after repair;
- no semantic trial emits an opaque mesh, triangle list, or individually
  enumerated formation field;
- all invalid trials repair or fail explicitly within two passes.

## 4. Canonical deterministic test matrix

Every positive case checks normalized serialization, finite geometry,
visual/collision agreement, structural surface roles, topology, swept
clearance, deterministic hashes, and at least one surface query. Tests add the
case-specific predicates below.

### 4.1 Semantic model cases

| ID | Pri | Case | Required predicates |
|---|---|---|---|
| MODEL-001 | P0 | Minimal environment region | Defaults normalize; bounds/transform/seed round-trip |
| MODEL-002 | P0 | Branched passage graph | All junction/segment references and explicit fields round-trip |
| MODEL-003 | P0 | Dragon-cavern recipe | No derived instances/meshes; ≤40 objects and ≤4,000 serialized tokens |
| MODEL-004 | P0 | Seeded detail field | Distribution, mask, collision policy, and sampler version round-trip |
| MODEL-005 | P1 | Layered material assembly | Layer order, thickness, collision, fracture, debris, and reinforcement policies survive normalization |
| MODEL-006 | P1 | Ordered damage stack | List order is retained and participates in semantic content hash |
| MODEL-007 | P1 | Terrain feature stack | Patch references, feature order, and opening bindings validate |
| MODEL-008 | P0 | Existing v1/v2 structural scene | New empty environment collections do not alter legacy serialization/geometry behavior |

### 4.2 Cave and passage cases

| ID | Pri | I/S | Case | Required predicates |
|---|---|---|---|---|
| CAV-001 | P0 | I5/S4 | One straight variable-width passage | Section dimensions interpolate; endpoints and floor route align |
| CAV-002 | P0 | I5/S3 | Curved rising passage | Tangent frames do not flip; floor slope and headroom agree with profile |
| CAV-003 | P0 | I5/S3 | Chamber joined to passage | One physical void; no cap wall at the seam |
| CAV-004 | P0 | I5/S2 | Y branch | Degree-3 junction, exactly three routes, no false wall/cross-shell collision |
| CAV-005 | P0 | I5/S2 | Four-way junction plus dead end | Four navigable arms; dead end remains a leaf |
| CAV-006 | P0 | I5/S2 | Loop with spur | Cycle rank is one; spur remains distinct |
| CAV-007 | P1 | I5/S3 | Ceiling opening to sky | Real aperture, sky exposure, no overhead collision within opening |
| CAV-008 | P1 | I4/S2 | Multilevel network with ramp and chimney | Walk and climb graphs differ correctly |
| CAV-009 | P1 | I4/S2 | Narrow and low-clearance branches | Agent-specific accessibility changes without semantic topology loss |
| CAV-010 | P1 | I5/S2 | 180×120×70 m dragon cavern | Chunk budgets, stable surfaces, arena floor, perch anchor, walk and fly graph |
| CAV-011 | P1 | I4/S2 | Overlapping chambers | Intentional union retains provenance; accidental near-touch is diagnosed |
| CAV-012 | P1 | I5/S2 | Cavern, cave mouth, and partial terrain roof | Exterior transition and sky exposure agree with actual solids |
| CAV-013 | P2 | I4/S1 | Network of 40 chambers/70 segments | Compile/memory budgets and graph equality |
| CAV-014 | P2 | I3/S1 | Very small formations in very large cavern | Tolerance/LOD policy keeps route and hero geometry stable |

### 4.3 Formation/detail cases

| ID | Pri | I/S | Case | Required predicates |
|---|---|---|---|---|
| FORM-001 | P1 | I5/S4 | Ceiling stalactite field | Instances attach only to overhead surfaces and point inward |
| FORM-002 | P1 | I5/S4 | Floor stalagmite field | Instances attach to support floor; protected path remains empty |
| FORM-003 | P1 | I4/S3 | Paired column formation | Matching samples may join only when configured gap threshold passes |
| FORM-004 | P1 | I4/S3 | Clustered boulder field | Distribution is deterministic; collision policy respected |
| FORM-005 | P1 | I4/S3 | Portal and spawn exclusions | No instance intersects expanded exclusion envelopes over 50 seeds |
| FORM-006 | P1 | I4/S3 | Sightline exclusion to hero perch | Required visibility samples remain unobstructed over 50 seeds |
| FORM-007 | P1 | I4/S4 | Authored dragon perch plus surrounding field | Stable hero ID/anchor; field mask; mesh or primitive provenance retained |

### 4.4 Exterior cases

| ID | Pri | I/S | Case | Required predicates |
|---|---|---|---|---|
| EXT-001 | P1 | I4/S4 | Sloped heightfield meadow | Height/normal/support queries agree with compiled triangles |
| EXT-002 | P1 | I4/S3 | Terraces and traversable path | Slope/step policies segment surfaces correctly |
| EXT-003 | P1 | I5/S2 | Cave mouth cut into hillside | Terrain/cavern seam is open, connected, and lip-free |
| EXT-004 | P1 | I4/S2 | Building pad and foundation on slope | No floating/embedded foundation; interior entrance aligns |
| EXT-005 | P2 | I4/S2 | Cliff with path switchbacks | Cliff non-traversable; path remains continuous |
| EXT-006 | P2 | I3/S2 | Sinkhole into chamber | Opening and fall-hazard roles; no support over void |
| EXT-007 | P2 | I3/S2 | Road/trench/berm ordered stack | Declared ordering is deterministic and visible in provenance |
| EXT-008 | P2 | I3/S2 | Seeded exterior rocks/cover | Protected roads, entrances, and pads remain clear over 50 seeds |

### 4.5 Material and destruction cases

| ID | Pri | I/S | Case | Required predicates |
|---|---|---|---|---|
| DMG-001 | P1 | I5/S4 | Paint/plaster/brick wall assembly | Layer thickness/order and face provenance round-trip |
| DMG-002 | P1 | I5/S3 | Drywall/stud/insulation assembly | Cavity/repeating reinforcement policy deterministic |
| DMG-003 | P1 | I4/S3 | Reinforced concrete assembly | Concrete cut face and rebar instances remain distinct materials |
| DMG-004 | P1 | I4/S3 | Layered rock substrate | Natural opening exposes configured strata in correct order |
| DMG-005 | P1 | I5/S2 | Ellipsoid breach through brick wall | Visual/collision aperture and substrate exposure agree |
| DMG-006 | P1 | I5/S2 | Irregular breach large enough to walk | Measured clearance creates opening/topology edge |
| DMG-007 | P1 | I5/S2 | Small hole and surface crack | Visible damage exists; no walk topology edge |
| DMG-008 | P1 | I4/S2 | Breach-derived rubble field | Debris materials derive from removed layers and avoid route mask |
| DMG-009 | P2 | I5/S1 | Partial ceiling collapse | Removed support disappears; debris blocks only measured regions |
| DMG-010 | P2 | I5/S1 | Column removal under platform | Support dependency triggers authored collapse or explicit invalid state |
| DMG-011 | P2 | I4/S2 | Structural fracture across wall | Collision changes only under structural fracture policy |
| DMG-012 | P2 | I4/S3 | Burned timber/plaster wall | Material state changes; topology unchanged by default |
| DMG-013 | P2 | I3/S2 | Bent/deformed metal panel | Displacement is bounded; provenance survives retessellation |
| DMG-014 | P2 | I4/S1 | Two overlapping breaches | List order deterministic; combined opening and provenance correct |
| DMG-015 | P2 | I4/S1 | Breach followed by collapse | Operation sequence changes expected products and hashes |
| DMG-016 | P2 | I4/S1 | Destroyed multilevel room | Floors, walls, stairs, routes, and support graph remain consistent |

### 4.6 Hybrid seam cases

| ID | Pri | Case | Required predicates |
|---|---|---|---|
| HYB-001 | P0 | Polygon room wall opening → semantic passage | One portal and one clear seam; room collision actually cut |
| HYB-002 | P1 | Imported chamber shell → semantic branch network | Mesh anchors, units, annotations, and physical clearance agree |
| HYB-003 | P1 | Semantic building → terrain foundation | Shared transform/support and no seam penetration |
| HYB-004 | P1 | Terrain cave mouth → semantic cavern | Exterior and subterranean graph components join exactly once |
| HYB-005 | P2 | Imported cliff → semantic terrain path | Explicit augment/exclude policy; path support remains semantic |

### 4.7 Invalid/adversarial cases

| ID | Input | Required behavior |
|---|---|---|
| ADV-001 | Passage references unknown junction | Typed reference error with segment and field path |
| ADV-002 | Repeated/zero-length path points | Normalize exact duplicates or reject degenerate span explicitly |
| ADV-003 | Self-intersecting explicit cross-section | Reject before field/mesh compilation |
| ADV-004 | Negative or zero section dimensions | Reject with station and measured values |
| ADV-005 | Branches touch geometrically without graph junction | Do not connect; diagnose near-touch when within tolerance band |
| ADV-006 | Semantic route blocked by compiled substrate | Mark geometric inconsistency; never report fully reachable |
| ADV-007 | Sky opening still covered by roof/terrain | Reject exposure claim and identify blocking solids |
| ADV-008 | Detail exclusions consume eligible region | Emit zero samples and typed exhaustion diagnostic |
| ADV-009 | Damage target or material assembly missing | Reject operation before mutating derived state |
| ADV-010 | Breach leaves zero-thickness/tiny layer sliver | Deterministically simplify or reject according to tolerance policy |
| ADV-011 | Unsupported/nonmanifold boolean result | Fail with operation/target IDs; never substitute decal geometry |
| ADV-012 | Cyclic support dependencies | Reject or require explicit stable-group policy |
| ADV-013 | Deformation exceeds target bounds/budget | Reject with displacement and allowed limit |

### 4.8 Export and runtime cases

| ID | Pri | Case | Required predicates |
|---|---|---|---|
| EXP-001 | P0 | Small semantic cave OBJ/SDF | Visual/collision assets load; transforms, bounds, and surface sidecar agree |
| EXP-002 | P1 | Chunked dragon cavern | Every chunk loads; no duplicate model IDs or boundary cracks; topology sidecar unchanged |
| EXP-003 | P1 | Exterior/cave/building scene | Shared structural frames and transitions survive export |
| EXP-004 | P1 | Layered breached wall | Cut-face material groups and collision aperture survive export |
| EXP-005 | P1 | Blender structural render | Surface/material groups, sky opening, formations, and damage are present in deterministic overview renders |
| EXP-006 | P1 | Drake plus experimental MuJoCo/USD matrix | Supported representations load or report an explicit per-feature unsupported result; no silent loss |

## 5. Genericity test suite

These tests directly answer “is this a general engine capability or a demo?”

| ID | Proof | Assertion |
|---|---|---|
| GEN-001 | Public-data-only fixtures | Canonical cases use exported spec/tool APIs and no compiler callbacks |
| GEN-002 | Source sentinel | Core source contains no canonical IDs, prompt substrings, or scenario-coordinate tables |
| GEN-003 | Rename metamorphism | Renaming every semantic ID yields an isomorphic graph and equal geometry hashes after ID normalization |
| GEN-004 | Rigid-transform metamorphism | Applying one SE(3) transform maps vertices, normals, anchors, routes, and bounds accordingly |
| GEN-005 | Uniform-scale metamorphism | Scaling geometry and tolerances scales length/area/volume while preserving topology |
| GEN-006 | Passage subdivision metamorphism | Splitting one compatible segment into two at an interpolated station preserves physical volume/topology within tolerance |
| GEN-007 | Declaration-order metamorphism | Reordering independent chambers, segments, fields, or disjoint operations preserves normalized products |
| GEN-008 | Repeat/cross-process determinism | Equal normalized semantics and seeds produce equal instance, topology, and mesh hashes in separate processes |
| GEN-009 | Generated structural families | Property generator covers 1–50 chambers, tree/cycle graphs, degrees 1–5, profiles, and valid clearances |
| GEN-010 | Held-out cave compositions | Five unseen graph/formation combinations pass without editing core code |
| GEN-011 | Disjoint damage commutativity | Reordering operations on disjoint targets preserves products modulo provenance order |
| GEN-012 | Overlap order sensitivity | Reordering overlapping operations changes the result predictably and remains deterministic |
| GEN-013 | Parallel/interrupt safety | Parallel equal compiles yield equal hashes through isolated staging; interruption publishes no complete manifest or partial scene |

The source sentinel searches production paths only. Terms such as `dragon`,
`prison`, `escape_tunnel`, and all `CAV-*`/`DMG-*` fixture IDs are allowed in
examples, docs, and tests but cause failure in environment model/compiler/tool
implementation files.

## 6. LLM authorability trials

Prompt evaluators assert semantic predicates; they do not compare exact prose,
coordinates, or meshes. Ten trials are run per prompt with a pinned model,
temperature, tool schema, and authoring prompt.

| ID | Prompt theme | Required semantic predicates |
|---|---|---|
| AUTH-001 | Small cave with two chambers and a winding route | ≥2 chambers, connected passage graph, walk clearance, no mesh fallback |
| AUTH-002 | Branching rescue cave with one dead end | Degree-3 branch, exactly one requested dead end, protected main route |
| AUTH-003 | Lava-tube loop with narrow optional branch | Cycle retained; main route walkable; optional branch capability-gated |
| AUTH-004 | Giant dragon cavern | ≥100 m major extent, arena floor, high ceiling, hero perch, entrance route, formation fields; ≤40 objects/4,000 tokens |
| AUTH-005 | Dragon cavern open to sky | AUTH-004 plus real sky opening and sky-exposure metadata |
| AUTH-006 | Exterior hillside with cave entrance | Terrain, protected path, cave mouth, cavern connection, no seam |
| AUTH-007 | Prison wall dug through layers | Material assembly, structural breach, exposed substrate, rubble, passable opening |
| AUTH-008 | Bomb-damaged apartment corridor | Multiple damaged targets, at least one blocked and one open route, no unsupported dynamic claim |
| AUTH-009 | Ruined temple partly inside a cavern | Mixed semantic cave/building/platforms, seams, hero feature, route |
| AUTH-010 | Collapsed mine with alternate climb route | Support/collapse semantics, walk route blocked, climb route available |
| AUTH-011 | Held-out sea cave and cliff path | Exterior+cavern composition, water/terrain tags, sky/cave mouths, no few-shot phrase overlap |
| AUTH-012 | Adversarial request for impossible geometry | Agent repairs within bounds or returns typed unsupported result; never silently boxes/flattens |

### Trial scoring

A trial receives individual booleans for:

- schema validity;
- compact semantic authoring;
- requested graph predicates;
- requested scale/dimension predicates;
- required opening and exposure predicates;
- detail-field and hero-feature predicates;
- damage/material predicates where applicable;
- deterministic compiler success;
- physical/topological agreement;
- no prohibited opaque fallback;
- repair-pass count and prompt fidelity after repair.

The aggregate pass requires every mandatory boolean. “Mostly right” renders do
not pass. Report pre-repair and post-repair results independently with Wilson
confidence intervals; retain individual failures for tool/schema iteration.

### Held-out discipline

- AUTH-011 and at least four GEN-010 recipes are authored after compiler and
  tool descriptions freeze.
- Held-out wording and expected predicates are stored outside few-shot prompt
  fixtures.
- At least one held-out test combines cave, exterior, constructed structure,
  destruction, and a capability-gated route.
- A held-out failure may cause general fixes, but adding its nouns, IDs, exact
  coordinates, or a scenario branch to production code fails GEN-002.

## 7. Backend experiments

These are bounded spikes with a written decision record. They do not weaken the
semantic specification.

### Experiment A — Branching void representation

Compare:

- A1: analytic passage lofts plus explicit junction patch meshing;
- A2: implicit signed-distance fields with deterministic surface extraction;
- A3: hybrid analytic segments feeding an implicit junction/chamber union.

Corpus: CAV-001–CAV-006, CAV-010, CAV-013, ADV-005, and ADV-006.

Measure watertightness, seam defects, stable provenance, tessellation error,
clearance agreement, compile time, memory, triangle count, determinism, and
ease of chunking. Prefer the simplest backend that passes all P0 cases; the
semantic API must not expose the choice.

### Experiment B — Layer-aware destruction boolean

Compare:

- B1: exact per-layer mesh booleans;
- B2: implicit/voxel field subtraction with material-region labels;
- B3: analytic cuts for common shapes plus field fallback for irregular cuts.

Corpus: DMG-001–DMG-008, DMG-014–DMG-015, ADV-010, and ADV-011.

Measure robustness, layer-order correctness, cut-face labels, sliver rate,
determinism, collision simplification, and export compatibility. Any backend
that turns a failed structural cut into a decal is disqualified.

### Experiment C — Large-scene chunking

Compare fixed grid, semantic-region, and adaptive spatial chunks on CAV-010,
CAV-013, EXT-005, and a hybrid exterior/cave/building scene. Confirm that
chunking does not change topology, surface IDs, route clearance, detail seeds,
or visible seams. Record compile and incremental-recompile costs.

### Experiment D — Formation sampling

Compare seeded stratified, Poisson-disk, and clustered-process sampling on
FORM-001–FORM-006. Require stable cross-process results, exclusion-mask
correctness over 50 seeds, bounded retry behavior, and visually non-gridlike
distributions. Random library implementation details must not become the seed
contract; use a versioned project RNG/sampler.

## 8. Test implementation layout

Recommended files:

```text
tests/unit/test_semantic_environment_model.py
tests/unit/test_passage_network_compiler.py
tests/unit/test_cavern_compiler.py
tests/unit/test_environment_detail_fields.py
tests/unit/test_terrain_compiler.py
tests/unit/test_material_assemblies.py
tests/unit/test_damage_operations.py
tests/unit/test_environment_genericity.py
tests/unit/test_environment_authoring_tools.py
tests/integration/test_semantic_environment_exports.py
tests/experiments/test_semantic_environment_prompts.py
tests/fixtures/semantic_environments/*.json
```

Reusable helpers belong in `tests/geometry_assertions.py` or the project's
existing geometry-test helper location. Canonical fixture builders must return
public semantic types and may not return compiled mesh objects.

Fast CI runs all model/compiler/genericity tests without network, GPU, Blender,
or Drake. Supported infrastructure runs simulator/export gates. LLM trials are
versioned experiments, not flaky per-commit unit tests, but their latest result
manifest is a release gate for claiming LLM authorability.

## 9. Verification commands and evidence

Each phase adds its modules to a dependency-light command of the form:

```bash
.venv/bin/python -m unittest \
  tests.unit.test_semantic_environment_model \
  tests.unit.test_passage_network_compiler \
  tests.unit.test_environment_genericity -q
```

Before merging each completed phase, also run the existing structural
regression set documented in `IMPLEMENTATION-STATUS.md`.

Evidence manifests contain:

- semantic schema/compiler version and normalized spec hash;
- all seeds and sampler version;
- visual/collision triangle and chunk counts;
- surface counts by role and provenance;
- topology graph summaries and expected predicates;
- swept-clearance results by agent class;
- damage/material exposure summaries;
- compile time and peak memory;
- export/load status by target;
- prompt/tool/model configuration for LLM trials;
- explicit `PASS`, `FAIL`, `UNSUPPORTED`, or `BLOCKED_ENV` per layer.

## 10. Phase completion rules

A phase is complete only when:

1. its public semantic records have round-trip and invalid-input tests;
2. compiler output passes geometry and provenance invariants;
3. semantic topology agrees with physical clearance/navigation;
4. the corresponding genericity/metamorphic tests pass;
5. existing SceneSmith structural tests remain green;
6. supported export/simulator layers pass or are recorded `BLOCKED_ENV` with a
   reproducible reason;
7. the implementation status ledger links exact tests and remaining seams;
8. no capability is advertised as LLM-authorable until its repeated prompt
   threshold passes.

## 11. Next development/experiment slice

Close the remaining E3 risks before starting the exterior compiler:

1. split large implicit scenes into stable spatial chunks while preserving
   topology, surface provenance, and deterministic hashes;
2. measure the 180×120×70 m fixture against the 60-second/2-GiB and independent
   visual/collision budgets on the reference CI host;
3. add paired-column and clustered-field recipes plus spawn and sightline
   exclusion masks;
4. add imported hero-mesh composition and deterministic lighting-anchor
   queries;
5. add CAV-011–CAV-014, the remaining FORM cases, five held-out GEN-010
   compositions, and the contradictory-solid ADV-007 case;
6. run AUTH-004/AUTH-005 repeated tool trials and publish normalized recipes,
   diagnostics, compactness, cost, and repair-rate manifests;
7. only then begin E4 with one cave-mouth-to-heightfield seam reusing the same
   opening, surface, and detail contracts.
