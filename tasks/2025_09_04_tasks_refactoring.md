## T1 — Create `hippo_eval` package and migrate evaluation code

**Why**
Separate concerns so algorithms never depend on pipeline/eval. Reduce cognitive load and simplify testing.

**What to change**

1. Create `hippo_eval/` and **move** all Python modules from `hippo_mem/eval/` into it. Preserve module names (`datasets.py`, `harness.py`, `baselines.py`, etc.).
2. Fix imports inside moved modules: replace `from hippo_mem.eval.X import ...` with intra-package imports or with `from hippo_eval.X import ...`.
3. Add **shim** module at `hippo_mem/eval/__init__.py` that re-exports public API from `hippo_eval` and raises a deprecation warning on import. This prevents breaking existing tests/scripts.
4. Update `scripts/*.py` to import from `hippo_eval` directly (no `sys.path` hacks).

**Touch these files**

* Move: `hippo_mem/eval/*` → `hippo_eval/*`
* New: `hippo_mem/eval/__init__.py` (shim)
* Edit: `scripts/eval_model.py`, `scripts/eval_cli.py`, `scripts/run_baselines.py`, `scripts/eval_bench.py`, any script importing `hippo_mem.eval`

**Acceptance**

* `pytest -q` passes.
* Running `python scripts/eval_model.py --help` still works.
* Importing `hippo_mem.eval` prints a deprecation warning but still works.

---

## T2 — Colocate report templates with reporting code and lock down `reports/` as output only

**Why**
Avoid mixing source and generated assets. Make template discovery robust.

**What to change**

1. Move `reports/templates/` → `hippo_mem/reporting/templates/`.
2. In `hippo_mem/reporting/report.py`, set `_TEMPLATE_DIR = Path(__file__).parent / "templates"`.
3. Remove any remaining code in root `reports/` (keep the **directory** for generated outputs; add `.gitkeep` if necessary).
4. Update any references in docs/tasks mentioning `reports/templates/`.

**Touch these files**

* Move: `reports/templates/**` → `hippo_mem/reporting/templates/**`
* Edit: `hippo_mem/reporting/report.py`
* Edit docs: `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `README.md` (search & replace the old template path)

**Acceptance**

* `python -m hippo_mem.reporting.summarize --dry-run` (or your existing report entry) renders without path errors.
* `reports/<run_id>/` is only generated content.

---

## T3 — Slim all `scripts/` to true CLIs (no business logic, no path hacks)

**Why**
Ensure reuse from library, better testability, and no environment surprises.

**What to change**

1. Remove `sys.path.insert(...)` from `scripts/run_baselines.py` and others.
2. Hoist any non-trivial logic into library modules:

   * Add `hippo_eval/baselines.aggregate(metrics_root: Path) -> Dict` and use it from the script.
   * If scripts contain argument validation/IO logic, move to `hippo_eval/bench.py` or `hippo_eval/harness/io.py`.
3. Keep scripts to argument parsing + calling a library function + exit code handling.
4. Optionally add `console_scripts` in `pyproject.toml` (e.g., `hippo-eval = scripts.eval_cli:main`).

**Touch these files**

* Edit: All `scripts/*.py` that currently import from project internals via `sys.path` hacks.
* New/Expand: `hippo_eval/baselines.py`, `hippo_eval/harness/io.py`

**Acceptance**

* `grep -R "sys.path.insert" scripts` returns nothing.
* Running `python scripts/run_baselines.py --help` works and calls library code (verify via small dry run).

---

## T4 — Break up oversized functions in `hippo_eval/harness.py`

**Why**
Improve readability/coverage and enforce single responsibility.

**What to change**

1. Extract helpers so no function exceeds \~80 LOC:

   * `evaluate()` → delegates to `build_runner()`, `load_suite()`, `run_suite()`, `collect_metrics()`.
   * `_write_outputs()` → split into `write_metrics()`, `write_meta()`, `write_csv()`, `write_artifacts()`.
2. Pure logic gets pure functions (no file IO); IO goes through `harness/io.py`.

**Touch these files**

* Edit: `hippo_eval/harness.py` (now smaller, imports helpers)
* New: `hippo_eval/harness/runner.py`, `hippo_eval/harness/io.py`, `hippo_eval/harness/metrics.py`

**Acceptance**

* Unit tests for harness still pass; no diffs in produced `metrics.json/csv` for a known run (use a golden file in `tests/fixtures/expected/`).

---

## T5 — Unify adapter wrappers and reduce structural duplication

**Why**
Episodic and spatial memory adapters share the “fusion” protocol; unify the wrapper logic and keep algorithm-specific parts minimal.

**What to change**

1. Create `hippo_mem/adapters/memory_base.py` with a small abstract base: pooling, normalization, and `forward(hidden_states, memory_tokens, span=None)`.
2. Refactor `EpisodicMemoryAdapter` and `SpatialMemoryAdapter` to inherit the base and keep only algorithm-specific config/touchpoints.
3. Remove/inline thin duplicates if they only forward to the same base.
4. Keep the existing public names/import paths.

**Touch these files**

* New: `hippo_mem/adapters/memory_base.py`
* Edit: `hippo_mem/adapters/episodic_adapter.py`, `hippo_mem/adapters/spatial_adapter.py`
* (Optional) Add deprecation note in `hippo_mem/episodic/adapter.py` and `hippo_mem/spatial/adapter.py` if any API is duplicated there.

**Acceptance**

* Adapter unit tests still pass; parameter counts and shapes unchanged.
* A quick smoke run shows identical retrieval/fusion telemetry.

---

## T6 — Clean folder responsibilities (“generated only” rule)

**Why**
Make outputs predictable; avoid checked-in code under output dirs.

**What to change**

1. Ensure `reports/` and `runs/` are empty (except `.gitkeep`) in the repo and ignored in `.gitignore`.
2. Move `datasets/semantic/mini.jsonl` → `tests/fixtures/datasets/semantic/mini.jsonl`; update references.
3. Move `experiments/*` → `docs/experiments/*` and update docs to point there.

**Touch these files**

* Moves: `datasets/**`, `experiments/**`
* Edits: any code/docs referencing old paths (`EVAL_PLAN.md`, `README.md`, tests)

**Acceptance**

* `pytest -q` passes.
* `grep -R "datasets/"` in non-test code returns **no** matches (except under `tests/fixtures`).
* Example commands in `docs/experiments/*/RUN.md` still work.

---

## T7 — Split tests by concern and add fast smoke tests for the CLI

**Why**
Faster feedback loops and clearer ownership.

**What to change**

1. Create subdirs under `tests/`: `algo/`, `eval/`, `reporting/`, `cli/`, `fixtures/`.
2. Move existing tests accordingly (heuristic: module import path inside the test).
3. Add a couple of CLI smoke tests that execute scripts with `--help` and a tiny dry-run using the mini dataset fixture.

**Touch these files**

* Moves: many under `tests/`
* New: `tests/cli/test_eval_cli.py`, `tests/fixtures/datasets/...`

**Acceptance**

* `pytest -q -k cli` runs only CLI smoke tests and passes in <10s.
* `pytest -q` total runtime similar or improved.

---

## T8 — Document the new architecture and update references

**Why**
Prevent regressions and confusion after the move.

**What to change**

1. Update `README.md`, `DESIGN.md`, `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `PROJECT_PLAN.md` to reference `hippo_eval` and the new template location.
2. Add a short “Migration notes” section explaining the `hippo_mem.eval` shim and deprecation.

**Touch these files**

* Edits: the above docs.

**Acceptance**

* A new contributor can run:
  `python scripts/eval_model.py +suite=episodic sizes=[50] seeds=[1337]`
  and get outputs under `runs/<id>/…` and `reports/<id>/…` using the new layout.

---

## T9 — (Optional but recommended) Add style/lint gates for the new boundaries

**Why**
Keep the codebase healthy post-refactor.

**What to change**

1. Add `ruff` and `black` configs in `pyproject.toml`.
2. Add a tiny `pre-commit` config to enforce import boundaries (e.g., forbid `hippo_mem` importing `hippo_eval`).
3. Add a unit test that asserts `from hippo_mem import eval` is only available via the shim and warns.

**Acceptance**

* `make lint` passes locally and in CI.
* Import boundary test fails if someone reintroduces `hippo_eval` → `hippo_mem` cycle.

---

# Notes to Codex (global guardrails during refactor)

* **No behavior changes.** If you’re unsure, keep the old code path and add a shim that calls the new one.
* **Absolute imports.** Prefer `from hippo_eval...` over relative spaghetti.
* **Fail loud on path changes.** Add deprecation warnings instead of silent breaks.
* **Keep outputs identical.** For a known run in `data/…`, assert that `metrics.json/csv` before vs. after refactor are byte-identical.
* **CI first.** Commit in small steps; run `pytest -q` and the minimal CLI smoke after each step.

---

If you want, I can also spit this into two Markdown files (review + Codex task list) you can drop straight into your `review/` and `tasks/` folders.
