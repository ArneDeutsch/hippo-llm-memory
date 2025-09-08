# Evaluation adapters

This package defines small wrappers around memory algorithms so the evaluation
harness can treat them uniformly. Each adapter implements:

- `present() -> str`: returns the algorithm name (`"episodic"`, `"relational"`,
  `"spatial"`).
- `build(cfg) -> dict`: construct store, model adapter, and optional gate from
  `cfg`.
- `retrieve(cfg, modules, item, *, context_key, hidden) -> RetrieveResult`:
  pull context for a task.
- `teach(cfg, modules, item, *, dry_run, gc, suite)`:
  ingest examples during teach or replay while updating `GateCounters`.
- `store_size(modules) -> tuple[int, dict]`: return item counts and diagnostics.

To add a new memory algorithm:

1. Create `<algo>.py` with a class implementing the `EvalAdapter` protocol.
2. Import it and register an instance in `REGISTRY` inside `__init__.py`.
3. Update tests and documentation if needed.

Adapters keep branching and gate wiring out of `harness.py` and preserve the
public API (`Task`, `evaluate`, `_run_replay`).
