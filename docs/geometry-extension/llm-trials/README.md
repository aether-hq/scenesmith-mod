# Held-out semantic-environment authoring trials

This directory is the retained evidence contract for LLM authorability. Trials
are deliberately separate from compiler unit tests: the model receives only
the public semantic schema, authoring guide, and held-out prompt.

Each JSON result must record:

- `trial_id`, prompt text/hash, model identifier, and run timestamp;
- raw model response hash and normalized semantic recipe;
- initial validation diagnostics and every bounded repair attempt;
- semantic token/object counts and compiler options;
- mesh audit, topology, aperture, collision, route, and artifact predicates;
- final `PASS`, `FAIL`, `UNSUPPORTED`, or `BLOCKED_ENV` result.

The target suite contains branching cave, dragon-scale cavern, sky opening,
hybrid constructed/cave breach, and adversarial budget/type prompts. CI runs
the deterministic validator over every checked-in result. Live model trials
require credentials and must not be represented as passing when they were not
actually executed; the current two-trial evidence is an initial sample, not the
specification's statistically meaningful repeated corpus.

[`results/summary.json`](results/summary.json) records the aggregate evidence
without overstating it: this two-trial sample passes the after-repair threshold
but not the before-repair threshold, and it is below the required ten runs per
prompt. That is an evidence-backed preliminary failure, not a product-quality
pass-rate claim.

The repository currently retains two executed Claude Sonnet 5 trials: a
degree-five branching/cyclic network that passed on its first response, and a
dragon-scale cavern that passed after two typed diagnostic repair turns. Run
`python scripts/validation/validate_llm_trials.py` to authenticate each retained prompt,
revalidate prompt-specific requirements, compile the recipes, audit their
meshes, and authenticate their output bundles. `--update` is reserved for an
intentional oracle change; ordinary CI fails if retained predicates drift.
