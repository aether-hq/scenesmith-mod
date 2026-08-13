# Semantic Scene Gallery

This viewer discovers every retained `heldout_*.json` semantic-environment
trial plus the checked-in controls under `sources/`, compiles them, and exposes
the results through one browser gallery. The original Aether bar is a permanent
control: every gallery build recompiles its 15.45 × 10.61 metre shell and three
portal cuts, then renders all 104 accepted semantic placements as deterministic
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

The bar control preserves the accepted packet's exact transforms, roles,
opening identities, and review camera. Its proxy geometry is intentionally not
the archived 59,560-triangle beauty render: that render depended on Blender and
a frozen upstream asset cache. The gallery labels the control
`semantic_proxy_regression` and records the reference mesh/triangle counts, so
it proves architecture and semantic layout compatibility without pretending to
be an asset-fidelity comparison.
