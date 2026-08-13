# Semantic Scene Gallery

This viewer discovers every retained `heldout_*.json` semantic-environment
trial, compiles its shell and geological detail layers, and exposes the result
through one browser gallery.

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
`docs/geometry-extension/llm-trials/results/`; the viewer has no hardcoded
trial IDs.
