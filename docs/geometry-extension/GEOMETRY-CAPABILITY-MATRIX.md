# Geometry Capability and Test Matrix

## Ranking method

Each case has two independent ranks:

- **Importance:** `I5` is essential for the stated goal; `I1` is specialized.
- **Simplicity:** `S5` is easiest and lowest-risk; `S1` requires the most new
  geometry/topology machinery.

Delivery priority is based on value, dependency order, and risk:

- **P0:** foundation and common multilevel/irregular structures;
- **P1:** required broad geometry support, including caverns;
- **P2:** advanced variants and robustness breadth;
- **P3:** research-grade or deliberately bounded edge cases.

Within a priority, high-simplicity cases come first so they establish reusable
infrastructure and test harnesses.

## Universal acceptance predicates

Unless a row explicitly opts out, every deterministic case must verify:

1. schema construction, validation, serialization, and round-trip equality;
2. all coordinates, normals, transforms, and mesh values are finite;
3. triangles have non-zero area and consistent declared winding;
4. visual and collision bounds agree within configured tolerances;
5. structural objects have stable unique IDs and semantic roles;
6. portals and connectors refer to existing endpoints;
7. topology reachability matches the expected connected components;
8. generated SDF/Drake directives preserve full SE(3) transforms;
9. support surfaces return valid poses and normals for sample points;
10. unsupported/invalid geometry raises a typed diagnostic and never silently
    falls back to a flat rectangle.

## Ranked summary

| Priority | Capability family | Importance | Simplicity | First proof |
|---|---|---:|---:|---|
| P0 | Backward-compatible flat rectangle | I5 | S5 | Golden v1 round trip |
| P0 | Independent elevations and stacked levels | I5 | S5 | Room frames retain Z |
| P0 | Raised/sunken platforms and split levels | I5 | S4 | Stepped support queries |
| P0 | Straight stairs and landings | I5 | S4 | Rise/run + connectivity |
| P0 | Straight ramps | I5 | S4 | Slope + support normal |
| P0 | Rotated and convex polygonal spaces | I5 | S4 | Polygon extrusion |
| P0 | Concave polygonal spaces | I5 | S3 | Triangulation + placement |
| P0 | Holes, atria, courtyards, and shafts | I5 | S3 | Floor hole remains empty |
| P0 | L/U stairs and multisegment ramps | I4 | S3 | Landing continuity |
| P0 | Mezzanines, balconies, bridges, catwalks | I4 | S3 | Open-edge semantics |
| P0 | Variable/double-height spaces | I4 | S3 | Ceiling and portal extents |
| P1 | Sloped floors and ceilings | I5 | S3 | Plane-aligned object pose |
| P1 | Curved walls via controlled tessellation | I4 | S3 | Chord-error bound |
| P1 | Heightfield floors/ceilings | I4 | S2 | Sampled height/normal |
| P1 | Freeform cavern chamber | I5 | S2 | Mesh + annotated surfaces |
| P1 | Curved/branching tunnels | I5 | S2 | Portal graph + clearance |
| P1 | Semantic cavern chamber/passage graph | I5 | S2 | Data-only graph compiles without imported shell |
| P1 | Large cavern chunking and sky openings | I5 | S2 | Dragon-scale fixture preserves topology/exposure |
| P1 | Seeded geological detail fields | I5 | S3 | Routes remain clear across seeds |
| P1 | Layered substrate and authored breach | I5 | S2 | Cut faces expose source layers; collision opens |
| P1 | Exterior terrain and cave/building seams | I4 | S2 | Shared surfaces/topology across boundary |
| P1 | Natural vertical passage/shaft | I4 | S2 | 3D traversability class |
| P1 | Mixed parametric/freeform complex | I5 | S1 | Semantic seam integrity |
| P2 | Spiral stairs | I3 | S2 | Sweep + head clearance |
| P2 | Ladder, lift, elevator | I3 | S3 | Non-walk connector policy |
| P2 | Vault, dome, arch, overhang | I4 | S2 | Freeform overhead semantics |
| P2 | Rough terrain and stepping stones | I3 | S2 | Agent-specific traversability |
| P2 | Moving/articulated structure | I2 | S1 | State-dependent topology |
| P3 | Non-Euclidean/teleport adjacency | I1 | S3 | Semantic-only connector |
| P2 | Reproducible precompiled destruction | I5 | S1 | Ordered operations update every structural product |
| P3 | Runtime physics-driven destruction | I2 | S1 | Explicitly out of first semantic-environment scope |

## Detailed deterministic cases

### A. Compatibility and transforms

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-001 | P0 | I5/S5 | Existing 5×4 m flat rectangular room | Byte-stable v1 fields; v2 migration creates `ground`; same four wall poses |
| G-002 | P0 | I5/S5 | Existing multiroom flat house | All existing adjacency/door/window tests remain valid |
| G-003 | P0 | I5/S5 | One room at elevation +3 m | Drake room frame translation Z is 3; local floor remains z=0 |
| G-004 | P0 | I5/S5 | Basement at -2.8 m | Negative elevation round trips and exports |
| G-005 | P0 | I5/S4 | Two rooms with identical XY footprint at different levels | They do not count as colliding; topology stays disconnected without connector |
| G-006 | P0 | I5/S4 | Three levels with unequal floor-to-floor heights | Per-level elevations and nominal heights are preserved |
| G-007 | P0 | I4/S4 | Rotated local frame (yaw 30°) | Footprint, walls, support normals, and attachments transform consistently |
| G-008 | P2 | I3/S3 | Full SE(3) imported structural mesh transform | Roll/pitch/yaw + translation survive all exports |

### B. Footprints and planar topology

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-010 | P0 | I5/S5 | Axis-aligned rectangle represented as polygon | Matches existing dimensions and area |
| G-011 | P0 | I5/S4 | Rotated rectangle | Four arbitrary-direction boundary segments; no cardinal assumption |
| G-012 | P0 | I5/S4 | Convex triangle | Three walls, valid floor triangulation, usable support surface |
| G-013 | P0 | I5/S4 | Convex pentagon | Five walls/normals and correct area |
| G-014 | P0 | I5/S3 | L-shaped concave room | No floor triangles outside footprint; inside samples supported |
| G-015 | P0 | I5/S3 | U-shaped concave room | Correct inward normals and no bridging across void |
| G-016 | P0 | I5/S3 | Donut footprint with courtyard hole | Hole has no floor/collision; inner boundary classified separately |
| G-017 | P0 | I4/S3 | Multiple holes (columns/shafts) | All loops retain identity and winding |
| G-018 | P1 | I4/S3 | Circular room tessellated to tolerance | Max chord error below configured tolerance; stable edge IDs |
| G-019 | P1 | I4/S3 | Spline/arc boundary with straight segments | Tessellation is deterministic and portals map to source curve coordinates |
| G-020 | P2 | I3/S2 | Narrow neck/dumbbell footprint | Triangulation preserves passage; clearance detects agent-size constraints |
| G-021 | P2 | I2/S2 | Very large coordinates and 1 km complex | Tolerance policy avoids cracks or overflow |
| G-022 | P2 | I3/S2 | Millimeter-scale detail next to meter geometry | Minimum feature policy simplifies or rejects explicitly |

### C. Elevation profiles and platforms

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-030 | P0 | I5/S4 | Single raised platform in a room | Top is a support surface; vertical face is not; placement Z is correct |
| G-031 | P0 | I5/S4 | Sunken conversation pit | Bottom and surrounding floor are separate support patches |
| G-032 | P0 | I5/S4 | Two-step split-level room | Each tread/elevation patch has correct bounds and connectivity |
| G-033 | P1 | I5/S3 | Single planar slope at 5° | Height and normal queries match analytic plane |
| G-034 | P1 | I5/S3 | Sloped floor meeting flat landing | Seam is continuous within tolerance; normal discontinuity is intentional |
| G-035 | P1 | I4/S2 | Bilinear warped floor | Query interpolation and triangulation agree |
| G-036 | P1 | I4/S2 | Sampled heightfield | No foldovers; heights/normals agree at grid and interior samples |
| G-037 | P2 | I3/S2 | Terraced terrain/cave floor | Traversable patches segmented by max step and slope policies |
| G-038 | P2 | I3/S2 | Disconnected stepping stones | Global topology connected only for agents whose step/jump policy allows it |

### D. Vertical and non-planar connectors

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-040 | P0 | I5/S4 | Straight stair, 3 m rise | Integer risers; rise/run limits; exact endpoint elevations |
| G-041 | P0 | I5/S4 | Straight stair with top/bottom landings | Landing surfaces connect both spaces without collision seams |
| G-042 | P0 | I5/S3 | L stair with landing | Two flights and landing form a continuous ordered path |
| G-043 | P0 | I4/S3 | U stair with half landing | Opposing flights do not overlap incorrectly; headroom validated |
| G-044 | P0 | I5/S4 | Straight ramp | Slope limit and analytic normal; endpoints align |
| G-045 | P0 | I4/S3 | Switchback ramp | Segment/landing continuity and guard/open-edge annotations |
| G-046 | P1 | I4/S3 | Short stair between split levels in one space | Connector endpoints may share a space ID but differ in elevation patch |
| G-047 | P2 | I3/S2 | Spiral stair | Helical sweep, tread ordering, central clearance, headroom |
| G-048 | P2 | I3/S3 | Ladder | Semantic reachability true; walkability false; capability-gated access true |
| G-049 | P2 | I3/S3 | Elevator/lift shaft | Shaft volume and landings modeled; dynamic car is separable from static topology |
| G-050 | P1 | I4/S2 | Natural sloped cave passage | Freeform connector has centerline, clearance envelope, and endpoint portals |
| G-051 | P1 | I4/S2 | Near-vertical natural chimney | Classified climb-only or inaccessible, never silently treated as a ramp |

### E. Vertical volumes and composite structures

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-060 | P0 | I4/S3 | Double-height room beside normal room | Distinct ceiling elevations and shared-wall upper section |
| G-061 | P0 | I5/S3 | Atrium hole through upper floor | No upper slab collision over void; lower-to-upper visibility/topology metadata |
| G-062 | P0 | I4/S3 | Mezzanine inside tall space | Platform support/open edges; furniture placement on mezzanine |
| G-063 | P0 | I4/S3 | Balcony projecting into void | Support and guard-edge semantics; no enclosing wall inferred at open edge |
| G-064 | P0 | I4/S3 | Bridge between two upper spaces | Both endpoints align; bridge surface connects graph components |
| G-065 | P0 | I4/S3 | Catwalk in cavern | Parametric connector embedded in freeform volume; seam remains navigable |
| G-066 | P1 | I4/S2 | Sloped ceiling/roof plane | Ceiling placement aligns to plane or rejects fixtures requiring horizontal mount |
| G-067 | P2 | I4/S2 | Barrel vault | Overhead surface classification and wall/ceiling seam |
| G-068 | P2 | I3/S2 | Dome | Radial normals; ceiling fixture policy uses authored anchor rather than arbitrary tangent |
| G-069 | P2 | I4/S2 | Arch opening | Portal aperture is not reduced to a rectangular collision box |

### F. Caverns, tunnels, and freeform meshes

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-070 | P1 | I5/S2 | Single irregular cavern chamber mesh | Imports with units/transform; collision generated; floor/wall/overhead patches annotated |
| G-071 | P1 | I5/S2 | Two chambers joined by tunnel | Semantic portal graph and geometric clearance agree |
| G-072 | P1 | I5/S2 | Branching Y tunnel | Three branches; no accidental cross-wall connection |
| G-073 | P1 | I4/S2 | Loop tunnel | Cycle retained in topology and navigation graph |
| G-074 | P1 | I4/S2 | Multilevel cavern with natural ramp | Height-varying traversable patch joins elevation bands |
| G-075 | P1 | I4/S2 | Cavern plus built stair/platform | Parametric/freeform seam has no gap or collision lip above tolerance |
| G-076 | P2 | I4/S2 | Low overhang | Agent head-clearance marks route inaccessible for tall agent |
| G-077 | P2 | I4/S2 | Narrow squeeze passage | Radius/width clearance is agent-specific |
| G-078 | P2 | I4/S2 | Pit and ledge | Downward void has no support; open-edge and fall-hazard annotations exist |
| G-079 | P2 | I3/S2 | Natural column/island obstacle | Obstacle produces hole/blocked region without corrupting chamber boundary |
| G-080 | P2 | I3/S1 | Overlapping/near-touching cave shells | Boolean or collision policy produces deterministic diagnostic/result |

### G. Semantic caverns and geological detail

These cases prohibit imported chamber/tunnel meshes unless a row explicitly
names the mesh escape hatch. The production compiler receives only public
semantic records.

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-081 | P0 | I5/S3 | Variable-section curved passage primitive | Cross-section interpolation, tangent-frame continuity, floor route, and headroom agree |
| G-082 | P0 | I5/S2 | Semantic Y-junction | One blended physical void; exactly three graph routes; no cap wall or false connection |
| G-083 | P0 | I5/S2 | Four-way junction with dead end | Node degree and leaf topology preserved in compiled navigation |
| G-084 | P0 | I5/S2 | Semantic loop plus spur | Cycle rank retained after compilation and serialization |
| G-085 | P1 | I5/S2 | 180×120×70 m dragon cavern | Chunk budgets, arena support, hero perch, walk/fly routes, deterministic hashes |
| G-086 | P1 | I5/S3 | Cavern ceiling opening to sky | Physical aperture, sky-exposed surfaces, and no overhead collision in opening |
| G-087 | P1 | I4/S2 | Degree-5 multilevel passage junction | Stable blend and agent-specific clearance across incident paths |
| G-088 | P1 | I4/S2 | Overlapping semantic chambers | Explicit union retains provenance; accidental near-touch is diagnosed |
| G-089 | P2 | I4/S1 | 40-chamber/70-passage network | Compile/memory budgets; graph and chunking invariance |
| G-090 | P1 | I5/S3 | Seeded stalactite/stalagmite fields | Correct source surfaces; protected routes/portals clear over 50 seeds |
| G-091 | P1 | I4/S3 | Paired natural columns | Pair/join threshold deterministic; no unsupported floating formation |
| G-092 | P1 | I4/S3 | Boulder clusters around hero feature | Hero anchor stable; distribution respects route/sightline exclusion masks |
| G-093 | P1 | I4/S4 | Imported dragon perch in semantic cavern | Explicit augment policy, anchor, collision, and provenance; no topology bypass |

### H. Exteriors and environment seams

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-100 | P1 | I4/S4 | Bounded sloped exterior terrain | Height/normal/support queries agree with compiled surface |
| G-101 | P1 | I4/S3 | Terraced hillside with path | Traversable patches respect step/slope policy; path continuous |
| G-102 | P1 | I5/S2 | Semantic cave mouth cut into terrain | One exterior/cavern transition; no wall, gap, or collision lip |
| G-103 | P1 | I4/S2 | Constructed foundation on sloped terrain | Building pad and foundation support align without float/penetration |
| G-104 | P2 | I4/S2 | Cliff with switchback route | Cliff blocked, route continuous, open/fall edges annotated |
| G-105 | P2 | I3/S2 | Sinkhole into lower chamber | No support across void; visibility/fall/topology metadata agree |
| G-106 | P2 | I3/S2 | Ordered road, trench, and berm features | Feature-stack ordering deterministic and provenance retained |
| G-107 | P2 | I3/S2 | Seeded surface-cover rocks | Entrances, roads, paths, and building pads remain clear over 50 seeds |

### I. Layered substrate and reproducible destruction

| ID | Pri | I/S | Geometry | Expected checks |
|---|---|---|---|---|
| G-110 | P1 | I5/S4 | Paint/plaster/brick wall assembly | Layer order/thickness and face provenance round-trip |
| G-111 | P1 | I4/S3 | Drywall/stud/insulation assembly | Repeated structural/cavity layers compile deterministically |
| G-112 | P1 | I4/S3 | Reinforced concrete wall | Concrete and reinforcement remain distinct on cut faces/debris |
| G-113 | P1 | I5/S2 | Walkable breach through layered wall | Visual/collision cut, substrate exposure, measured portal, and navigation agree |
| G-114 | P1 | I5/S2 | Small non-passable hole and cracks | Visible damage creates no walk topology edge |
| G-115 | P1 | I4/S2 | Breach-derived rubble field | Debris derives from removed layers and respects route mask |
| G-116 | P2 | I5/S1 | Partial ceiling collapse | Removed support invalidated; collision/navigation/debris agree |
| G-117 | P2 | I5/S1 | Column removal below platform | Support dependency follows authored collapse/invalid-state policy |
| G-118 | P2 | I4/S2 | Structural and visual fractures | Collision changes only for structural policy |
| G-119 | P2 | I4/S3 | Burned layered wall | Material state changes; topology unchanged unless explicitly structural |
| G-120 | P2 | I3/S2 | Bounded deformation | Displacement budget and provenance survive retessellation |
| G-121 | P2 | I4/S1 | Overlapping ordered damage operations | Stable list order, combined geometry, and multi-operation provenance |

### J. Invalid and adversarial geometry

| ID | Pri | I/S | Input | Required behavior |
|---|---|---|---|---|
| X-001 | P0 | I5/S5 | Missing level reference | Reject with `UnknownLevelError` and source ID |
| X-002 | P0 | I5/S5 | Connector endpoint missing | Reject with `UnknownConnectorEndpointError` |
| X-003 | P0 | I5/S4 | Self-intersecting bow-tie footprint | Reject; never triangulate opportunistically |
| X-004 | P0 | I5/S4 | Hole outside outer loop | Reject with loop IDs |
| X-005 | P0 | I5/S4 | Wrong loop winding | Normalize deterministically or reject according to strictness setting |
| X-006 | P0 | I5/S4 | Zero-area/repeated edge | Reject with vertex indices |
| X-007 | P0 | I5/S4 | Unsafe stair rise/run | Reject unless explicit non-walkable decorative mode |
| X-008 | P0 | I5/S4 | Stair endpoint elevations do not match | Reject with numerical delta |
| X-009 | P1 | I5/S3 | NaN/Inf coordinates | Reject before hashing, mesh generation, or export |
| X-010 | P1 | I5/S2 | Nonmanifold cavern mesh | Repair only when configured and report changes; otherwise reject |
| X-011 | P1 | I5/S2 | Inverted freeform shell normals | Detect and repair/reject explicitly |
| X-012 | P1 | I5/S2 | Mesh with mixed units/no units | Require declared unit scale; no guessing in strict mode |
| X-013 | P1 | I4/S2 | Portal aperture outside boundary surface | Reject with both IDs |
| X-014 | P1 | I5/S2 | Semantic connection with no collision-free path | Mark topology/geometric inconsistency; do not report fully reachable |
| X-015 | P2 | I4/S2 | Duplicate/overlapping structural faces | De-duplicate or reject according to tolerance policy |
| X-016 | P2 | I3/S1 | Geometry above complexity budget | Stop with budget diagnostic and simplification suggestion |
| X-017 | P0 | I5/S3 | Passage references missing junction | Reject with segment ID and exact field path |
| X-018 | P0 | I5/S3 | Touching passage shells without shared junction | Keep disconnected and diagnose ambiguous near-touch |
| X-019 | P1 | I5/S2 | Sky opening covered by roof/terrain solid | Reject sky-exposure claim with blocking solid IDs |
| X-020 | P1 | I4/S3 | Detail exclusions consume eligible region | Emit no instances plus typed exhaustion diagnostic |
| X-021 | P1 | I5/S2 | Damage target or material assembly missing | Reject before producing mutated derived geometry |
| X-022 | P1 | I5/S1 | Boolean produces nonmanifold/sliver result | Deterministically repair within policy or fail with operation and target IDs |
| X-023 | P2 | I4/S1 | Cyclic support dependency | Reject or require explicit stable-group policy |
| X-024 | P2 | I4/S2 | Requested real-time destruction | Return typed unsupported result; never pretend precompiled damage is dynamic |

## Prompt-to-structure cases

These cases exercise the SceneSmith agent tools after deterministic geometry
primitives are reliable. Each prompt has a canonical structural predicate set,
not a single golden mesh.

| ID | Pri | Prompt theme | Required structural predicates |
|---|---|---|---|
| P-001 | P0 | Two-story townhouse with straight stair | 2 levels; connected stair; upper slab opening; rooms assigned correctly |
| P-002 | P0 | Sunken lounge and raised dining platform | 3 elevation patches; short connector(s); furniture zones at correct Z |
| P-003 | P0 | L-shaped gallery around a courtyard | Concave/holed footprint; courtyard has no slab; exterior/inner boundaries differ |
| P-004 | P0 | Loft with mezzanine and U stair | Double-height volume; mezzanine; stair; open edge |
| P-005 | P0 | Station platforms joined by ramp and bridge | Multiple elevations; accessible ramp path; bridge endpoints align |
| P-006 | P1 | Sloping wine cellar with vaulted ceiling | Sloped support; non-planar overhead; fixture constraints |
| P-007 | P1 | Natural cave with two chambers and tunnel | Freeform chamber meshes; annotated floors; one tunnel connector |
| P-008 | P1 | Multilevel mine with ramps, ladders, and shafts | Walk and climb connector classes; agent-specific reachability |
| P-009 | P1 | Cavern temple with built catwalks and stairs | Mixed freeform/parametric structures; seam continuity |
| P-010 | P1 | Branching lava tubes with a loop | Branch and cycle topology; clearance; no false connections |
| P-011 | P2 | Spiral tower interior | Rotated/curved boundary; spiral connector; repeated levels |
| P-012 | P2 | Cliff dwelling with irregular terraces | Heightfield/freeform exterior; stepped support patches; bridges |
| P-013 | P0 | Branching rescue cave with a dead end | Semantic chamber/passage graph; branch degree; protected main route; no mesh fallback |
| P-014 | P1 | Giant dragon cavern with formations and perch | ≥100 m extent; arena support; high ceiling; seeded fields; hero anchor; compact recipe |
| P-015 | P1 | Dragon cavern with opening to sky | P-014 plus real aperture and sky-exposure semantics |
| P-016 | P1 | Exterior hillside leading into cave | Terrain path; cave mouth seam; exterior-to-subterranean route |
| P-017 | P1 | Dug breach through layered prison wall | Material assembly; structural damage operation; substrate cut faces; rubble; passability |
| P-018 | P2 | Bomb-damaged apartment corridor | Ordered damage stack; one blocked/one open route; support/collision agreement |
| P-019 | P2 | Ruined temple inside cavern | Cave, constructed walls/platforms, material damage, and hero features compose |
| P-020 | P2 | Collapsed mine with climb-only alternate route | Collapse blocks walk route; climb route remains capability-gated |

## Cross-product suites

Individual primitives are insufficient; bugs often occur at representation
seams. The following pairwise suite covers high-risk combinations without an
unbounded Cartesian product:

| Suite | Combination |
|---|---|
| C-01 | concave footprint × second level × straight stairs |
| C-02 | holed footprint × atrium × bridge |
| C-03 | rotated polygon × sloped floor × wall attachment |
| C-04 | split level × short stairs × furniture/manipulands |
| C-05 | double-height room × mezzanine × ceiling fixtures |
| C-06 | cavern mesh × parametric platform × catwalk |
| C-07 | tunnel mesh × natural ramp × low-clearance segment |
| C-08 | heightfield × imported structure × USD/MuJoCo export |
| C-09 | existing rectangle × new multilevel neighbor × old checkpoint load |
| C-10 | invalid freeform mesh × repair mode × strict mode |
| C-11 | semantic cave graph × sky opening × terrain cover |
| C-12 | large cavern × formation fields × protected walk/fly routes |
| C-13 | terrain cave mouth × semantic passage × constructed room portal |
| C-14 | layered wall × breach × rubble field × navigation update |
| C-15 | ceiling collapse × support graph × multilevel connector |
| C-16 | imported hero mesh × semantic cavern × seeded detail exclusion |
| C-17 | exterior terrain × damaged building × cave below foundation |

## Capability completion rule

A matrix row is marked complete only after all applicable layers pass:

| Layer | Evidence |
|---|---|
| Model | deterministic schema and round-trip tests |
| Geometry | mesh/surface invariant tests |
| Topology | expected connectivity and accessibility tests |
| Placement | at least one object placement/attachment query |
| Simulation | Drake load and collision smoke test |
| Export | Blender and selected MuJoCo/USD round trip |
| Agent | prompt/tool call produces required predicates in repeated trials |
| Genericity | public data-only fixture plus rename/transform/scale/order metamorphic checks and source sentinel |
| Authorability | compact recipe budget, no opaque geometry, held-out prompt predicates, and bounded repair pass |
