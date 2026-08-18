# Semantic Scene Gallery

This viewer discovers every retained `heldout_*.json` semantic-environment
trial plus the checked-in controls under `sources/`, compiles them, and exposes
the results through one browser gallery. The first bar control is SceneSmith's
full-fidelity scene 112: its real furniture geometry, materials, and embedded
textures are loaded from a pinned GLB. A separate Aether diagnostic recompiles
the 15.45 × 10.61 metre semantic shell and shows its 104 accepted placements as
closed-mesh proxies.

From the repository root:

```bash
.venv/bin/python examples/semantic_gallery/serve_gallery.py
```

The command regenerates the gallery and opens
`http://127.0.0.1:8766/viewer.html`. Use the scene list or number keys to switch
scenes, click the viewport for mouse look, and use `WASD`, `Space`, and `Shift`
to fly. Hold `Ctrl` for faster movement. `R` returns to the authored entry
view.

New retained trials appear automatically after they are added under
`docs/geometry-extension/llm-trials/results/`; checked-in controls are discovered
under `examples/semantic_gallery/sources/`. The viewer has no hardcoded scene
IDs and uses one manifest contract for both kinds of scene.

The two bar entries intentionally test different layers:

- `Original SceneSmith Bar — Full Render` is the visual baseline. The gallery
  verifies its pinned SHA-256 and refuses to display it unless at least 280 mesh
  instances, 187,086 triangles, 50 materials, and 50 textures survive loading.
- `Aether Bar — Semantic Proxy Diagnostic` tests current structural compilation,
  portal cuts, exact placement transforms, and semantic identity. It is clearly
  labeled `semantic_proxy_diagnostic`; it is not an appearance comparison.

The pinned browser GLB is a lossless Draco decode of SceneSmith's unaltered
provider artifact. Its provenance and checksum live in
`sources/original_scenesmith_bar.json`.
