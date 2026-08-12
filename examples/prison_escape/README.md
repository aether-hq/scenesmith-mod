# The Long Way Out

This deterministic showcase builds an underground prison block where someone
has broken a 3.6 m-wide opening through the wall into a long, irregular escape
tunnel. The tunnel descends about 5 m over roughly 69 m, widens into a terminal
chamber, and has ceiling-mounted light fixtures placed from actual overhead
surface queries.

![Plan and section preview](generated/preview.svg)

## Fly through it

Start the local walkthrough server:

```bash
.venv/bin/python -m examples.prison_escape.serve_demo
```

It opens `http://127.0.0.1:8765/viewer.html`. Click **Enter the prison**, then
use the mouse and WASD. Press `F` to switch between floor-following walk mode
and free flight; Space/Shift move vertically in flight. Keys `1`–`4` jump to
the prison, breach, mid-tunnel, and terminal cavern, and `M` shows the plan.

The viewer loads the exact OBJ assets used by the generated SDF/Drake scene,
not a separate artistic mockup. It needs a WebGL 2 browser and internet access
to load the pinned Three.js modules.

## Generate it

From the repository root:

```bash
.venv/bin/python -m examples.prison_escape.generate_scene
```

The committed `generated/` output contains:

- `prison_escape.dmd.yaml` — Drake model directives for the complete scene;
- `structural_layout.json` — the v2 semantic layout and embedded-passage spec;
- `structures/` — compiled visual/collision OBJ, SDF, and semantic-surface
  sidecars for the prison and tunnel;
- `details/` — barred cells, bunks, breach rubble, emissive ceiling fixtures,
  and matching SDF point lights;
- `manifest.json` — dimensions, surface-mount evidence, topology, and route
  validation results;
- `preview.svg` — dependency-free plan and longitudinal-section preview.

## What this proves

The opening is not decoration: it is a portal cut from both visual and collision
wall meshes. The tunnel is authored as a public `PassageNetworkSpec` with two
junctions, one 3D path, and twelve variable cross-sections. SceneSmith's shared
implicit-union compiler derives its collision shell and floor/wall/overhead
roles. The same passage edge derives the natural-passage `ConnectorSpec`, so
topology and physical geometry come from one semantic recipe.

Each light fixture is positioned and oriented using
`StructuralSurfaceIndex.overhead_pose`; the exported manifest names the exact
surface patch used for every mount. The route is generated only if a 1.9 m tall,
0.45 m radius walking agent can traverse it without a support or headroom veto.
