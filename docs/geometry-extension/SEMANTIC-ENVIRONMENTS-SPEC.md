# Semantic Environments Specification

Status: active implementation contract (E0–E3 hardening complete; later phases partial)

Date: 2026-08-13

Related documents:

- [General-geometry architecture](README.md)
- [Existing capability matrix](GEOMETRY-CAPABILITY-MATRIX.md)
- [Semantic-environments development plan](SEMANTIC-ENVIRONMENTS-DEVELOPMENT-PLAN.md)
- [Current implementation evidence](IMPLEMENTATION-STATUS.md)

## 1. Goal

SceneSmith must let an LLM describe large natural environments, exterior
terrain, and reproducibly damaged buildings using compact semantic recipes.
Deterministic compilers turn those recipes into visual geometry, collision,
surface annotations, topology, and navigation data.

The system is hybrid:

- semantic primitives are the preferred and fully editable representation;
- seeded procedural recipes add bounded detail without making the LLM place
  every rock or fracture;
- imported/generated meshes remain an explicit escape hatch for shapes that
  the semantic vocabulary cannot express;
- all three paths produce the same downstream structural contracts.

The target is not a collection of cave, terrain, and destruction demos. A
single representation and compiler pipeline must support held-out combinations
without scenario-specific code.

## 2. Primary use cases

The first complete capability must express and compile:

1. a branching passage network with cycles, dead ends, elevation changes, and
   independently sized cross-sections;
2. a 100–300 m-scale cavern suitable for a dragon encounter, including a
   traversable floor, ledges, a hero perch, formation fields, and a ceiling
   opening to the sky;
3. a cave mouth that joins subterranean structure to exterior terrain without
   a false wall or collision seam;
4. an exterior terrain region containing slopes, terraces, cliffs, paths,
   building pads, and entrances into constructed or natural spaces;
5. a layered constructed wall with a passable breach that exposes paint,
   plaster, masonry or concrete, insulation, reinforcement, and other authored
   substrate layers;
6. a partially collapsed interior whose damage changes rendering, collision,
   support surfaces, portals, navigation, and debris from the same operation
   stack.

## 3. Non-goals for the first implementation

- Real-time fracture or physics-driven destruction during simulation. The
  first contract compiles a reproducible damaged state before simulation.
- Geological simulation, physically exact erosion, or engineering-grade
  structural-collapse prediction.
- Photoreal surface texturing or unique sculptural quality from primitives
  alone. Asset generation and imported hero meshes may improve appearance.
- Planet-scale streaming. The design must be chunkable, but the initial scale
  target is one connected scene up to roughly 500 m across.
- An LLM emitting triangle arrays, voxel grids, collision hulls, or explicit
  positions for thousands of repeated formations.
- Treating every procedural rock, debris fragment, or stalactite as a topology
  node. Only structurally or narratively meaningful objects require stable
  semantic identity.

### 3.1 Current baseline and gap

The fork provides the downstream foundation: versioned levels
and polygonal spaces, portals, stairs/ramps/ladders, heightfields, arbitrary
structural meshes, semantic surfaces, capability-aware topology, surface
queries, and structural OBJ/SDF export. Imported cavern shells can replace a
room, and an embedded natural-passage centerline can be checked for local
support and headroom.

The first E0–E3 implementation slice now provides canonical semantic regions,
ellipsoid/superellipsoid chambers, variable-section passage graphs, four typed
passage profiles, true implicit branch/chamber union, surface provenance,
HouseLayout/tool/export integration, graph-to-connector adaptation, real
sky/exterior chamber apertures, deterministic formation fields with protected
route/opening/hero masks, and explicit colliding hero primitives. The prison
example consumes the public passage primitives, and a held-out 180×120×70 m
cavern proves compact authoring at the intended encounter scale. Stable
large-scene chunking/LOD, advanced formation policies, exterior seams, layered
substrate, and structural damage operations remain unimplemented; this
specification continues to define that remaining delta.

## 4. Architectural model

Natural environments are easiest to author as **empty navigable volumes inside
solid substrate**. Buildings are easiest to author as **solid assemblies that
enclose empty spaces**. Destruction is easiest to author as **ordered operations
on those solids**. These representations meet at a shared compiled surface
model.

```text
LLM semantic recipe
  |
  +-- EnvironmentRegionSpec
  |     cavern | passage network | exterior | constructed
  +-- MaterialAssemblySpec
  |     ordered material layers and structural solids
  +-- DamageOperationSpec[]
  |     breach | collapse | fracture | burn | deform | debris field
  +-- DetailFieldSpec[] + HeroFeatureSpec[]
        seeded repeated detail     individually authored landmarks
  |
  v
Validated semantic graph
  spaces + solids + voids + openings + routes + operation stack
  |
  v
Geometry backends
  parametric sweep | implicit/CSG volume | heightfield | imported mesh
  |
  v
Shared compiled products
  visual mesh + collision mesh + structural surfaces + topology/navigation
  + material/substrate provenance + exposure/sky metadata + diagnostics
```

The semantic graph is authoritative. Tessellation density, collision
simplification, chunk boundaries, and procedural samples are derived products
and may be regenerated.

The implemented `DerivedSceneContract` is the publication boundary for semantic
scenes. It derives physical-gated topology and openings, collision/query
surfaces, detail route vetoes, runtime product references, and provenance from
one normalized semantic source. Each product is published as an atomic,
content-addressed directory and exposed through an authenticated `ArtifactRef`
that binds the semantic source hash, compiler version/options, and product
hashes. New runtime consumers must use this contract; the separate shell/detail
compile methods remain compatibility shims for existing exporters.

## 5. Normative semantic primitives

All coordinates use meters in the SceneSmith structural frame. All primitive
types have a stable `id`, explicit version, deterministic serialization, and a
seed where randomness is allowed. New public fields must be typed schema
fields, not undocumented entries in a generic `parameters` dictionary.

### 5.1 Regions and spaces

`EnvironmentRegionSpec` groups structure by environment and compilation
policy:

| Field | Contract |
|---|---|
| `id` | Stable unique identifier |
| `kind` | `constructed`, `subterranean`, `exterior`, or `hybrid` |
| `bounds` | Conservative 3D compilation bounds |
| `transform` | Full local-to-scene transform |
| `material_context` | Default substrate/ground material references |
| `detail_seed` | Integer seed for deterministic derived detail |
| `chunk_policy` | Maximum chunk extent/triangle budgets; not authored topology |

An environment may contain current polygonal rooms and structural meshes. It
also introduces the volume, terrain, material, damage, and detail primitives
below.

### 5.2 Cavern chambers

`CavernChamberSpec` declares one meaningful open volume:

| Field | Contract |
|---|---|
| `id`, `region_id` | Stable identity and owning region |
| `center`, `orientation` | 3D placement |
| `shape` | `ellipsoid`, `superellipsoid`, `loft`, `vaulted`, or `mesh` |
| `size` | Width, depth, and height or an explicit station profile |
| `floor_profile` | Flat, sloped, terraced, basin, heightfield, or mesh |
| `roughness` | Bounded amplitude/frequency recipe plus seed |
| `substrate_id` | Material assembly surrounding the void |
| `semantic_tags` | Usage such as `lair`, `lake_basin`, `ledge`, `arena` |

Chambers may overlap intentionally. The compiler unions their voids and must
retain each chamber's semantic provenance after surface extraction. Accidental
disjoint overlap or a connection below clearance policy produces a diagnostic.

### 5.3 Passage graphs

`PassageNetworkSpec` is a graph, not a list of unrelated tunnel props. It owns
`PassageJunctionSpec` nodes and `PassageSegmentSpec` edges.

A junction may reference a cavern chamber, a portal, or a free-standing 3D
position. A segment contains:

| Field | Contract |
|---|---|
| `id`, `start_junction_id`, `end_junction_id` | Graph identity |
| `path` | 3D polyline or spline control points |
| `cross_sections` | Width, height, and shape at normalized path stations |
| `profile` | `ellipse`, `keyhole`, `slot`, `arched`, or explicit 2D profile |
| `floor_mode` | Natural curved floor, graded path, steps, or non-traversable |
| `roughness` | Bounded wall/ceiling variation and seed |
| `capabilities` | `walk`, `crawl`, `climb`, `swim`, `fly`, or combinations |
| `clearance` | Minimum radius, headroom, slope, and optional agent classes |

The compiler blends incident segments at a junction into one physical void.
Branching, cycles, parallel routes, and dead ends must survive into both the
semantic and navigation graphs. Segment order must not change compiled
connectivity.

### 5.4 Natural and constructed openings

`EnvironmentOpeningSpec` is the implemented chamber-to-sky/exterior slice of
the broader `OpeningSpec` that generalizes the existing rectangular
`PortalSpec`:

- targets another space/region or the exterior/sky;
- supports rectangle, ellipse, arch, polygon, and imported aperture profiles;
- binds to a semantic boundary, chamber, passage junction, floor, ceiling, or
  terrain patch;
- carries passability, visibility, weather exposure, and sky-exposure roles;
- may be created explicitly or as the result of a damage operation.

An opening is a single source of truth for the visual aperture, collision
aperture, topology edge, and navigation transition. A decorative crack must not
become a passable portal unless its policy and measured clearance allow it.

### 5.5 Exterior terrain

`TerrainRegionSpec` reuses the surface and topology contracts rather than
creating a separate exterior engine:

| Primitive | Purpose |
|---|---|
| `TerrainPatchSpec` | Bounded heightfield, terraced surface, analytic slope, or mesh patch |
| `TerrainFeatureSpec` | Cliff, ridge, berm, trench, plateau, road/path, riverbed, or building pad |
| `TerrainOpeningSpec` | Cave mouth, sinkhole, shaft, cellar stair, or tunnel entrance |
| `SurfaceCoverFieldSpec` | Seeded rocks/vegetation/scree that respect clearance masks |

Terrain features compose in a deterministic ordered stack. Cave mouths and
building foundations bind to named terrain patches with seam tolerances. Sky is
represented as exposure metadata and boundary absence, not as a physical
ceiling mesh.

### 5.6 Geological detail and hero features

Repeated detail is region-authored rather than individually enumerated.

`DetailFieldSpec` contains:

- a target surface/volume region;
- a formation type: `stalactite`, `stalagmite`, `column`, `flowstone`,
  `boulder`, `rubble`, `scree`, or an asset population;
- density or count range;
- size/aspect/orientation distributions;
- a deterministic seed;
- exclusion masks for routes, portals, spawn points, sight lines, and hero
  features;
- collision policy: `visual_only`, `coarse`, or `full`;
- optional clustering and ceiling/floor pairing rules.

`HeroFeatureSpec` gives individual identity to the small number of features an
LLM should place deliberately, such as a dragon perch, a central stone column,
a collapsed bridge, or an altar. Hero features may use a semantic primitive or
an imported mesh, but they always declare anchors, collision policy, and
clearance intent.

Generated detail may never silently block a required route, close a portal, or
create a new topology connection. The compiler enforces masks after sampling
and reports any dropped samples.

The implemented E3 subset uses explicit `count`, `min_size`, `max_size`,
`surface_role`, seed, protected passage-network IDs, route clearance, and
collision policy. It provides stalactite/stalagmite/column-like, flowstone,
boulder, rubble, and scree meshes plus rock-spire/boulder hero primitives.
The sampler is versioned and independent of the platform RNG. Its conservative
3D envelope masks the complete formation, not merely the surface anchor.
Paired ceiling/floor columns, clustering, spawn/sightline masks, asset
populations, and imported hero composition remain future extensions.

### 5.7 Materials and substrate

`MaterialAssemblySpec` is an ordered physical layer stack:

| Field | Contract |
|---|---|
| `id` | Stable assembly identity |
| `layers` | Ordered outside-to-inside `MaterialLayerSpec` entries |
| `structural_layer_ids` | Layers that determine solid support/collision |
| `repeat_mode` | Planar, along-boundary, radial, or volume substrate |

Each `MaterialLayerSpec` contains material ID, thickness, visual treatment,
collision policy, fracture behavior, debris class, and optional reinforcement
pattern. Common assemblies include painted plaster over brick, drywall over
studs and insulation, reinforced concrete, timber framing, and layered rock.

Every compiled face retains source operation and material-layer provenance.
Cut faces can therefore expose correct substrate instead of stretching the
outer wall texture across the opening.

### 5.8 Reproducible destruction operations

`DamageOperationSpec` is an ordered, immutable operation applied to a named
structural target or target region:

| Operation | Required semantics |
|---|---|
| `breach` | Remove a shaped volume; optionally create a portal when clearance passes |
| `collapse` | Remove/displace a bounded portion, invalidate support, and emit debris |
| `fracture` | Add visible and/or structural cracks with depth and connectivity policy |
| `burn` | Change material state and appearance; collision changes only if explicitly requested |
| `deform` | Apply a bounded displacement field while preserving provenance |
| `debris_field` | Seed debris from source materials with route exclusion masks |

Every operation specifies `id`, ordered application index, target selector,
shape/region, severity, seed, affected products, and portal/support policy.
Applying the same base structure and operation stack twice must produce the
same semantic result and content hashes.

Overlapping operations use declared list order. Deleting or reordering an
operation invalidates downstream generated products but does not mutate the
base structure. Unsupported booleans fail with the operation and target IDs;
they do not degrade into decals or non-colliding decoration.

### 5.9 Mesh escape hatch

`StructuralMeshSpec` remains valid for whole shells, individual hero features,
terrain patches, and pre-fractured assets. Imported meshes must provide:

- explicit units and transform;
- intended solid/void orientation;
- visual and collision policies;
- authored annotations or deterministic surface classification;
- semantic anchors for every portal/route connection;
- provenance identifying which semantic primitive they replace or augment.

Using a mesh does not exempt a scene from topology, clearance, surface, and
export validation. An imported shell cannot be declared connected merely
because two semantic nodes say it is.

## 6. Compiler contract

Every supported semantic recipe compiles to a `CompiledEnvironment` containing:

1. visual meshes split into bounded, stable chunks;
2. collision meshes or analytic collision primitives with independent budgets;
3. `StructuralSurface` patches including support, traversable, boundary,
   overhead, attachment, open-edge, cut-face, exterior, and sky-exposed roles;
4. semantic topology and agent-specific navigation graphs;
5. portal/opening records that agree with physical apertures;
6. material-layer and damage-operation provenance for compiled faces;
7. deterministic detail instances and rejected-sample diagnostics;
8. metrics and content hashes sufficient to reproduce the result.

The compiler must preserve current SceneSmith room, connector, surface query,
Drake/SDF, and structural-mesh behavior. Existing polygon rooms are simply one
constructed-region backend.

### 6.1 Geometry invariants

- No NaN/Inf positions, normals, transforms, parameters, or bounds.
- No non-zero collision surface may be omitted without a declared collision
  policy.
- Required passages and openings meet their agent-specific swept-clearance
  envelopes, not only centerline samples.
- Solid/void orientation and material side are explicit.
- Junctions, cave mouths, damaged openings, and terrain/building seams have no
  collision gap or lip above the configured tolerance.
- Visual detail may deviate from collision detail only through a recorded
  simplification policy.
- Chunking, tessellation, and level of detail must not change topology.

## 7. LLM authoring contract

The authoring interface is judged on structural output, not prose similarity or
a golden mesh.

1. **Topology first.** The LLM first declares regions, chambers/spaces,
   junctions, passage/opening edges, and required routes.
2. **Shape second.** It adds bounded dimensions, paths, profiles, terrain, and
   material assemblies.
3. **Detail last.** It adds seeded fields and a small number of hero features.
4. **Validate between stages.** Every tool response identifies offending IDs,
   JSON paths, measured values, limits, and one or more repair hints.
5. **No mesh authoring for semantic trials.** A trial marked semantic fails if
   it contains triangle/voxel arrays or a `mesh_path`, except when the prompt
   explicitly requests an imported hero asset.
6. **Compactness.** The canonical dragon-cavern scene must be expressible in at
   most 40 semantic objects and 4,000 serialized JSON tokens, excluding
   generated detail instances and compiled meshes.
7. **Determinism.** All procedural primitives require explicit seeds. Repeating
   a validated tool call produces byte-equivalent normalized semantic JSON and
   equal compiled content hashes.
8. **Bounded repair.** The authoring agent gets at most two deterministic repair
   passes per stage. It may change the invalid semantic recipe but may not hide
   a failure by switching to an opaque mesh.

The tool schema must use bounded enums, explicit units, useful defaults, and
small composable records. Free-form dictionaries are reserved for metadata,
not geometry-defining behavior.

An illustrative recipe for the macro-structure of a dragon cavern is
deliberately small and uses the implemented E3 serializer fields:

```json
{
  "schema_version": 1,
  "regions": [
    {
      "id": "underpeak",
      "kind": "subterranean",
      "bounds": {"minimum": [0, 0, -30], "maximum": [240, 180, 90]},
      "detail_seed": 417
    }
  ],
  "chambers": [
    {
      "id": "dragon_lair",
      "region_id": "underpeak",
      "shape": "superellipsoid",
      "center": [150, 90, 20],
      "size": [180, 120, 70],
      "substrate_id": "granite_mass",
      "semantic_tags": ["lair", "arena"]
    }
  ],
  "passage_networks": [
    {
      "id": "approach",
      "region_id": "underpeak",
      "junctions": [
        {"id": "entrance", "position": [0, 35, 0]},
        {"id": "fork", "position": [65, 55, -4]},
        {"id": "lair_gate", "position": [135, 82, 8], "chamber_id": "dragon_lair"},
        {"id": "dead_end", "position": [90, 20, 8]}
      ],
      "segments": [
        {"id": "entry_run", "start_junction_id": "entrance", "end_junction_id": "fork", "path": [[0, 35, 0], [32, 42, -2], [65, 55, -4]], "profile": "keyhole", "cross_sections": [{"station": 0, "width": 6, "height": 7}, {"station": 1, "width": 7, "height": 8}]},
        {"id": "main_run", "start_junction_id": "fork", "end_junction_id": "lair_gate", "path": [[65, 55, -4], [110, 72, 2], [135, 82, 8]], "profile": "ellipse", "cross_sections": [{"station": 0, "width": 10, "height": 12}, {"station": 1, "width": 18, "height": 20}]},
        {"id": "side_run", "start_junction_id": "fork", "end_junction_id": "dead_end", "path": [[65, 55, -4], [78, 37, 3], [90, 20, 8]], "profile": "slot", "cross_sections": [{"station": 0, "width": 3, "height": 5}, {"station": 1, "width": 3, "height": 5}]}
      ]
    }
  ],
  "openings": [
    {"id": "sky_oculus", "region_id": "underpeak", "source_chamber_id": "dragon_lair", "target": "sky", "center": [150, 90, 54], "normal": [0, 0, 1], "shape": "ellipse", "size": [24, 16], "depth": 30, "weather_exposed": true}
  ],
  "detail_fields": [
    {"id": "ceiling_teeth", "region_id": "underpeak", "target_chamber_id": "dragon_lair", "formation_type": "stalactite", "surface_role": "overhead", "count": 80, "min_size": [0.8, 0.8, 2], "max_size": [4, 4, 18], "seed": 419, "protect_passage_network_ids": ["approach"], "route_clearance": 6, "collision_policy": "coarse"}
  ],
  "hero_features": [
    {"id": "dragon_perch", "region_id": "underpeak", "target_chamber_id": "dragon_lair", "feature_type": "rock_spire", "anchor": [185, 105, 12], "size": [20, 14, 16], "collision_policy": "full"}
  ]
}
```

The LLM chose the topology, scale, passage controls, opening, detail region,
and hero landmark. It did not choose tessellation, junction triangles,
individual stalactites, collision hulls, or navigation polygons.

### 7.1 Constraints

- Existing v1/v2 room, connector, heightfield, and structural-mesh scenes must
  preserve their normalized semantics and physical transforms.
- Fast model/compiler tests must run without network, an LLM, GPU, Blender, or
  Drake. Heavy integrations are separate, explicit gates.
- Visual and collision complexity budgets are independent and recorded.
- Semantic IDs and topology must remain stable across tessellation/chunking
  changes; derived triangle indices are not stable API.
- Randomness uses an explicit project sampler version and seed. Platform
  library RNG behavior must not be the long-term reproducibility contract.
- Unsupported or over-budget operations stop before export and return typed,
  actionable diagnostics.
- JSON authoring is non-coercive: booleans, integers, and numbers must have the
  corresponding JSON scalar type and all numbers must be finite.
- Authored identifiers and predictable derived instance identifiers share one
  safe, globally unique scene namespace.

## 8. Falsifiable requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| SE-001 | First-class chambers and passage graphs represent branches, cycles, dead ends, and vertical paths. | CAV-001–CAV-006 semantic, compiler, and topology tests pass. |
| SE-002 | Passage junctions compile into one connected physical void without duplicate walls or false connections. | Junction manifold/seam checks and swept-route tests pass for degrees 1–5. |
| SE-003 | A large cavern remains chunkable and preserves semantics at 100–300 m scale. | CAV-010 meets topology, chunk, determinism, memory, and collision budgets. |
| SE-004 | Sky openings and cave mouths are real boundary apertures with exposure and topology semantics. | CAV-007 and EXT-003 have no ceiling/wall collision across their apertures. |
| SE-005 | Seeded detail fields produce formations without blocking protected routes or portals. | FORM-001–FORM-006 pass over at least 50 seeds each. |
| SE-006 | Hero formations compose with procedural fields and imported meshes. | FORM-007 preserves anchor, collision policy, exclusion mask, and provenance. |
| SE-007 | Exterior terrain uses shared surface/topology contracts and joins caves/buildings without seams. | EXT-001–EXT-006 pass support, route, opening, and export checks. |
| SE-008 | Layered material assemblies expose the correct ordered substrate on cut faces. | DMG-001–DMG-004 verify face-layer provenance and rendered material groups. |
| SE-009 | Ordered damage operations deterministically update visual geometry, collision, support, navigation, openings, and debris. | DMG-005–DMG-016 and idempotency/order tests pass. |
| SE-010 | Passability is measured, never inferred from the operation label. | Decorative cracks stay closed; sufficiently clear breaches create topology edges. |
| SE-011 | Imported meshes remain a compatible escape hatch with equal semantic obligations. | HYB-001–HYB-005 pass annotations, anchors, clearance, and export checks. |
| SE-012 | Existing rooms/connectors/freeform meshes continue to compile unchanged. | Existing focused regression suite and v1/v2 goldens pass. |
| SE-013 | Semantic recipes compile without scenario-specific source code. | GEN-001–GEN-010, source-sentinel, metamorphic, and held-out composition tests pass. |
| SE-014 | The LLM can author canonical and held-out scenes through public tools. | AUTH-001–AUTH-012 meet the pass-rate thresholds in the development plan. |
| SE-015 | Unsupported inputs fail explicitly without flattening, decorative substitution, or opaque-mesh escape. | ADV-001–ADV-013 produce typed diagnostics and no compiled scene. |
| SE-016 | Compiled products are reproducible from normalized semantics and seeds. | Cross-process determinism tests match normalized JSON, topology, instance, and mesh hashes. |

## 9. Acceptance criteria

- [ ] No core compiler module contains special handling keyed to canonical
  fixture IDs, prompt text, `dragon`, `prison`, `escape_tunnel`, or test-case
  names.
- [ ] A single public semantic API authors all canonical cave networks,
  exteriors, and destroyed interiors.
- [ ] Translating, rotating, uniformly scaling, or consistently renaming a
  valid recipe produces the corresponding transformed or isomorphic result.
- [ ] Reordering independent graph/field declarations does not change
  normalized semantics or compiled hashes.
- [ ] The dragon-cavern fixture contains no mesh path, explicit triangle list,
  explicit formation instances, or hidden fixture callback.
- [ ] Every protected route remains open for all tested detail seeds; every
  deliberately blocked route is rejected or capability-gated.
- [ ] A wall breach changes both visible and collision geometry and exposes
  authored substrate layers.
- [ ] A non-passable crack or burn operation does not create a topology edge.
- [ ] Removing a load-bearing support invalidates or removes dependent support
  surfaces according to the authored collapse policy.
- [ ] All semantic cases serialize, round-trip, compile twice identically, and
  export with stable structural IDs.
- [ ] Existing SceneSmith structural tests remain green.
- [ ] Held-out LLM prompts meet the thresholds in Section 10 without producing
  opaque geometry for semantic cases.

## 10. Proof of genericity and LLM authorability

### 10.1 Deterministic genericity proof

The test suite must include all of the following; curated screenshots alone do
not count:

- **Data-only fixtures:** canonical scenes are JSON/Python data using public
  spec types. No fixture-specific compiler subclass or callback is allowed.
- **Source sentinel:** scan core modules for fixture IDs and scenario-only
  branches. Allow those terms only in docs, examples, and tests.
- **Metamorphic tests:** transform, scale, rename, declaration reorder,
  segmentation, and seed-change relations verify general behavior.
- **Generated families:** property tests generate valid chamber graphs,
  passage profiles, terrain stacks, material assemblies, and damage stacks.
- **Held-out composition:** at least five cases are written only after the
  implementation stabilizes and must pass without core changes.
- **Negative controls:** intentionally invalid, disconnected, or obstructed
  recipes must fail the exact predicates that positive cases pass.

### 10.2 LLM experiment protocol

For each prompt case, run 10 trials against a pinned model and tool schema.
Retain the prompt, tool calls, normalized semantic JSON, diagnostics, repair
calls, compiled metrics, and final predicate results.

A trial passes only if:

1. required semantic predicates pass;
2. the recipe uses no prohibited opaque geometry;
3. deterministic compilation succeeds;
4. topology agrees with physical swept-clearance checks;
5. the recipe stays within its semantic-object and token budgets;
6. any repair completes within two passes and does not remove a requested
   capability.

Thresholds:

- canonical/core prompts: at least 80% pass before repair and 95% after repair;
- held-out/cross-domain prompts: at least 65% before repair and 85% after;
- invalid/adversarial prompts: 100% must either repair to a valid structure or
  fail explicitly; silent flattening and opaque fallback are always zero-
  tolerance failures.

Prompt cases and exact predicates live in the development plan rather than in
few-shot examples shown to the model. Held-out prompt wording must not appear
in the authoring prompt or tool descriptions.

## 11. Edge and failure policies

| Edge | Required policy |
|---|---|
| Zero chambers/segments | Reject a subterranean region with no authored volume. |
| Junctions that only touch | Connect only when explicit junction IDs agree and geometric overlap meets tolerance. |
| Reversed segment endpoints | Preserve graph identity and reverse station/profile interpretation deterministically. |
| Overlapping chambers | Union intentionally when same network; otherwise require explicit merge policy. |
| Tiny passage at a large junction | Retain visually, but capability-gate or reject route based on swept clearance. |
| Sky opening under terrain/roof | Diagnose contradictory solids; never label it sky-exposed. |
| Detail field with no legal samples | Emit zero instances plus a diagnostic; never invade exclusion masks. |
| Empty damage stack | Compile byte-equivalent to the base structure. |
| Overlapping damage operations | Apply stable list order and record provenance from all contributing operations. |
| Breach smaller than agent | Visible cut may exist; no walk topology edge is created. |
| Destroyed layered wall at a corner | Preserve layer order and avoid exposing impossible interior material sides. |
| Mesh/semantic overlap | Require explicit `replace`, `augment`, or `exclude` composition policy. |

### 11.1 Edge-completeness coverage

The specification edge probe classified the requirements across collection,
numeric-range, stateful, and I/O behavior. The 67 applicable
requirement/category pairs reduce to the following shared policies; each is an
explicit acceptance rule or a held-out property-test backstop.

| QA category | Resolution |
|---|---|
| Empty/degenerate | An environment needs at least one region; a subterranean region needs at least one chamber/void. A chamber may have no passage network. Empty detail and damage collections are valid no-ops; empty eligible sampling regions emit a typed diagnostic. |
| Adjacency/touching | Geometry connects only through matching semantic junction/opening identity plus tolerance-valid physical overlap. Mere touching never creates topology. Explicit union/augment policies govern other overlaps. |
| Ordering/stability | Independent declarations normalize by stable ID and are order-invariant. Passage cross-section stations, terrain features, material layers, and damage operations retain declared order because that order is semantic. |
| Numeric boundaries | All dimensions are finite and positive. The 100 m and 300 m large-cavern boundaries are tested directly; values above the 500 m initial scene target require a budget override or typed unsupported result. Clearance thresholds are tested exactly and immediately on either side. |
| Precision/tolerance | Meters and full-precision serialized numbers are authoritative. One versioned tolerance policy owns welding, clearance, sliver removal, and hashing quantization; tests exercise values immediately around each threshold. |
| Idempotency/repetition | Revalidating or recompiling identical normalized semantics and seeds produces equal normalized JSON and content hashes. Empty operation stacks are byte-equivalent to the base result. |
| Concurrency/I/O | Compilation is functionally pure with respect to semantic input. Parallel compiles use isolated staging paths and publish only complete manifests; interruption leaves no result that can be mistaken for a completed scene. |
| Property backstop | Generated graph, profile, transform, scale, terrain, layer, and damage families exercise combinations not enumerated in canonical fixtures. |

## 12. Prohibitions

- The implementation must not teach the compiler named scenarios or magic
  coordinate sets.
- Procedural detail must not change required connectivity or agent safety in a
  seed-dependent way.
- The authoring agent must not claim success from a render when structural
  predicates fail.
- Imported meshes must not bypass unit, topology, collision, clearance, or
  provenance checks.
- Damage operations must not create passable routes unless the compiled
  aperture satisfies the configured agent envelope.
- A visual-only crack, scorch, decal, or rubble asset must not masquerade as
  structural destruction.
- Terrain or cavern compilation must not silently add flat floors, box walls,
  or ceilings around unsupported geometry.
- Generated detail must not require the LLM to enumerate individual instances.

## 13. Locked decisions

1. Use a hybrid representation, with semantic primitives preferred and meshes
   retained as an explicit escape hatch.
2. Let the LLM author macro-structure, regions, distributions, and seeds; let
   deterministic generators author repeated geological/environmental detail.
3. Permit individually authored hero formations and landmarks.
4. Model destruction as a reproducible ordered operation stack over semantic
   base structure and layered material assemblies.
5. Reuse the existing structural surface, topology, collision, and export
   contracts across caves, buildings, and exteriors.
6. Prove genericity with data-only, property, metamorphic, negative-control,
   source-sentinel, and held-out tests.

## 14. Research questions resolved by implementation spikes

The semantic contract above is independent of these backend choices. The
development plan must resolve them with measured spikes before committing the
production compiler:

1. implicit/SDF union and surface extraction versus explicit loft-and-junction
   meshing for branching cavern voids;
2. exact mesh booleans versus field/voxel-based booleans for damage and layered
   cut faces;
3. chunk boundary and level-of-detail policies that preserve stable surface IDs
   and topology in 100–500 m environments;
4. collision decomposition and swept-volume validation appropriate for large
   organic shells;
5. deterministic formation sampling that remains visually irregular without
   creating cross-platform hash drift;
6. terrain/building/cave seam representation across Drake, Blender, and future
   MuJoCo/USD exports.
