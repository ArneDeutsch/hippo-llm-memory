## T1 — Create `hippo_eval` and migrate **eval, metrics, reporting, tasks, and consolidation eval**

**Why**
Make `hippo_eval` the single home for pipelines, metrics, reporting, synthetic tasks, and consolidation evaluation orchestration.

**What to change**

1. **Move packages:**

   * `hippo_mem/eval/**` → `hippo_eval/**`
   * `hippo_mem/metrics/**` → `hippo_eval/metrics/**`
   * `hippo_mem/reporting/**` → `hippo_eval/reporting/**`
   * `hippo_mem/tasks/**` → `hippo_eval/tasks/**`
2. **Move root reporting code & templates:**

   * `reports/health.py`, `reports/render_baselines.py`, `reports/plots/**`, `reports/templates/**`
     → `hippo_eval/reporting/health.py`, `hippo_eval/reporting/render_baselines.py`, `hippo_eval/reporting/plots/**`, `hippo_eval/reporting/templates/**`
3. **Relocate consolidation eval runner:**

   * `hippo_mem/consolidation/test_eval.py` → `hippo_eval/consolidation/eval.py`
     (rename module to `eval.py` or `consolidation_eval.py`).
   * Add `scripts/test_consolidation.py` as a thin CLI calling `hippo_eval.consolidation.eval:main`.
4. **Add shims with deprecation warnings:**

   * `hippo_mem/eval/__init__.py` re-exports from `hippo_eval` (warns).
   * `hippo_mem/metrics/__init__.py` re-exports from `hippo_eval.metrics` (warns).
   * `hippo_mem/reporting/__init__.py` re-exports from `hippo_eval.reporting` (warns).
   * `hippo_mem/tasks/__init__.py` re-exports from `hippo_eval.tasks` (warns).
5. **Fix imports across the repo** to `hippo_eval.*` for callers (scripts/tests).
   Example: `from hippo_mem.metrics.scoring import ...` → `from hippo_eval.metrics.scoring import ...`.

**Acceptance**

* `pytest -q` fully green.
* All CLIs still work: `python scripts/eval_model.py --help`, `python scripts/test_consolidation.py --help`, etc.
* Importing old paths (`hippo_mem.eval`, `hippo_mem.metrics`, `hippo_mem.reporting`, `hippo_mem.tasks`) works but emits deprecation warnings.

> This expands your previous T1 to include `metrics`, `reporting`, `tasks`, and consolidation eval as discussed.&#x20;

---

## T2 — Lock root `reports/` to **outputs only** and update loaders

**Why**
Stop mixing source with generated artifacts; make template discovery robust.

**What to change**

1. Ensure **all** reporting code + templates now live under `hippo_eval/reporting/**`.
2. In `hippo_eval/reporting/report.py` and `hippo_eval/reporting/render_baselines.py`, set the template loader to:

   ```python
   _TEMPLATE_DIR = Path(__file__).parent / "templates"
   ```
3. Clean root `reports/` (leave `.gitkeep` and add `.gitignore` as needed).
4. Replace any imports of `reports.health` with `hippo_eval.reporting.health`.

**Acceptance**

* Report generation works on a sample run; paths resolve without errors.
* `grep -R "^from reports\."` returns nothing outside tests.

> This supersedes the narrower template-only move in your current T2.&#x20;

---

## T3 — Slim **all** scripts to true CLIs (and add a new consolidation CLI)

**Why**
Keep business logic in libraries; prevent path hacks.

**What to change**

1. Remove `sys.path.insert(...)` in `scripts/*.py`.
2. Hoist aggregation/IO logic into:

   * `hippo_eval/baselines.py` (e.g., `aggregate_metrics(root: Path) -> dict`)
   * `hippo_eval/harness/io.py`
3. Ensure the new `scripts/test_consolidation.py` only parses args and calls library code.

**Acceptance**

* `grep -R "sys.path.insert" scripts` → no results.
* `python scripts/run_baselines.py --help` and a tiny dry run still work.

---

Perfect—let’s slice T4 into bite-size Codex tasks. Each task is scoped to a single change set you can run independently and commit before moving to the next. I’ve grouped them by file, but each file has multiple tiny steps so Codex won’t balk.

---

# T4.harness — break up `hippo_eval/harness.py`

## T4.harness.1 — Extract runner (build + execute)

**Why** Keep orchestration separate from IO/metrics.
**What**

* Create `hippo_eval/harness/runner.py` with:

  * `build_runner(cfg) -> Runner`
  * `run_suite(runner, suite) -> RunResult`
* Move only code strictly needed for building/executing runs.
* In `hippo_eval/harness.py`, import and delegate to these.
  **Touch**
* New: `hippo_eval/harness/runner.py`
* Edit: `hippo_eval/harness.py`, imports in callers if any.
  **Acceptance**
* `pytest -q` passes.
* A mini dry-run produces identical `metrics.json/csv` (use your golden fixture).

## T4.harness.2 — Extract filesystem IO helpers

**Why** Centralize reads/writes and paths.
**What**

* Create `hippo_eval/harness/io.py` with:

  * `write_metrics(path, metrics)`, `write_meta(path, meta)`, `write_csv(path, rows)`
  * `ensure_run_dirs(root) -> Paths`
* Move FS/path code from `harness.py` into these functions.
* Replace direct FS calls in `harness.py` with calls to `io.py`.
  **Touch**
* New: `hippo_eval/harness/io.py`
* Edit: `hippo_eval/harness.py`
  **Acceptance**
* Golden mini run unchanged; unit tests green.

## T4.harness.3 — Extract metrics aggregation/schema

**Why** Separate metric math from orchestration.
**What**

* Create `hippo_eval/harness/metrics.py` with:

  * `collect_metrics(results) -> Dict`
  * `MetricSchema` (any dataclasses/validators you use)
* Move pure metric code from `harness.py` here.
* Update `harness.py` to import from this module.
  **Touch**
* New: `hippo_eval/harness/metrics.py`
* Edit: `hippo_eval/harness.py`
  **Acceptance**
* Golden outputs identical; tests green.

## T4.harness.4 — Thin the public API & remove dead code

**Why** Make `harness.py` a coordinator only.
**What**

* Keep only small public functions (e.g., `evaluate(cfg)`).
* Inline trivial pass-throughs into imports; delete dead/private helpers now in submodules.
* Add `__all__` exports in `harness/__init__.py` re-exporting `runner`, `io`, `metrics` surfaces.
  **Touch**
* New/Edit: `hippo_eval/harness/__init__.py`
* Edit: `hippo_eval/harness.py`
  **Acceptance**
* `from hippo_eval.harness import evaluate` still works.
* No output diffs on the golden mini run.

---

# T4.bench — break up `hippo_eval/bench.py`

## T4.bench.1 — Extract orchestration

**Why** Keep run loops separate from summarization/FS.
**What**

* Create `hippo_eval/bench/orchestrator.py` with:

  * `run_bench(cfg) -> BenchResult`
  * `run_matrix(matrix_cfg) -> List[BenchResult]`
* Move only control-flow logic.
  **Touch**
* New: `hippo_eval/bench/orchestrator.py`
* Edit: `hippo_eval/bench.py` (delegate)
  **Acceptance**
* CLI bench still works; tests pass.

## T4.bench.2 — Extract summarization

**Why** Isolate rollups independent of IO.
**What**

* Create `hippo_eval/bench/summarize.py` with:

  * `summarize(results) -> Summary`
  * `aggregate_across_runs(results) -> Dict`
* Move pure summarization logic here.
  **Touch**
* New: `hippo_eval/bench/summarize.py`
* Edit: `hippo_eval/bench.py`
  **Acceptance**
* Summaries identical for golden mini run.

## T4.bench.3 — Extract layout & paths

**Why** One place defines where artifacts live.
**What**

* Create `hippo_eval/bench/layout.py` with:

  * `bench_paths(root, run_id) -> Paths`
  * Any constants for folder/file names.
* Replace path building in `bench.py` with these helpers.
  **Touch**
* New: `hippo_eval/bench/layout.py`
* Edit: `hippo_eval/bench.py`
  **Acceptance**
* Files created in same locations as before (compare directory trees).

## T4.bench.4 — Final cleanup

**Why** Make `bench.py` thin.
**What**

* Keep only public entry points re-exporting the three submodules.
* Add `hippo_eval/bench/__init__.py` with `__all__` for `orchestrator`, `summarize`, `layout`.
  **Touch**
* New/Edit: `hippo_eval/bench/__init__.py`
* Edit: `hippo_eval/bench.py`
  **Acceptance**
* `from hippo_eval.bench import run_bench` works; no output diffs.

---

# T4.datasets — break up `hippo_eval/datasets.py`

## T4.datasets.1 — Extract dataset loaders

**Why** Separate IO from CLI and generation.
**What**

* Create `hippo_eval/datasets/loaders.py` with:

  * `load_dataset(name, cfg) -> Iterable[Example]`
  * `iter_split(ds, split) -> Iterator[Example]`
* Move pure loader/adaptor code here.
  **Touch**
* New: `hippo_eval/datasets/loaders.py`
* Edit: `hippo_eval/datasets.py`
  **Acceptance**
* All dataset consumers still load identical counts/samples.

## T4.datasets.2 — Move synthetic generators next to tasks

**Why** Make generation live with task definitions.
**What**

* Create/extend `hippo_eval/tasks/generators.py` with existing synthetic generators from `datasets.py`.
* Replace imports in `datasets.py` to call these generators.
  **Touch**
* New/Edit: `hippo_eval/tasks/generators.py`
* Edit: `hippo_eval/datasets.py`
  **Acceptance**
* Generated examples are byte-identical for a fixed seed.

## T4.datasets.3 — Extract CLI wrappers

**Why** Keep CLI thin.
**What**

* Create `hippo_eval/datasets/cli.py` with argparse-based entry points that call `loaders`/`tasks`.
* Make any script (`scripts/datasets_cli.py`) delegate to this module.
  **Touch**
* New: `hippo_eval/datasets/cli.py`
* Edit: `scripts/*` that expose dataset commands
  **Acceptance**
* `python scripts/datasets_cli.py --help` works; behavior unchanged.

## T4.datasets.4 — Final tidy & fixtures

**Why** Lock the surface and test it.
**What**

* Remove now-unused helpers from `datasets.py`; keep it as a thin coordinator.
* Ensure `tests/fixtures/datasets/semantic/mini.jsonl` path is used where relevant.
  **Touch**
* Edit: `hippo_eval/datasets.py`
* Edit: tests referencing datasets
  **Acceptance**
* Tests green; same dataset stats/hashes for golden samples.

---

# T4.report — break up `hippo_eval/reporting/report.py`

## T4.report.1 — Extract table assembly

**Why** Make Markdown/HTML table logic reusable and testable.
**What**

* Create `hippo_eval/reporting/tables.py` with:

  * `make_metrics_table(df) -> str`
  * `make_summary_table(df) -> str`
* Move pure formatting functions here.
  **Touch**
* New: `hippo_eval/reporting/tables.py`
* Edit: `hippo_eval/reporting/report.py`
  **Acceptance**
* Rendered tables (saved strings) match before/after exactly.

## T4.report.2 — Extract plotting helpers

**Why** Isolate matplotlib code.
**What**

* Create `hippo_eval/reporting/plots.py` with:

  * `plot_accuracy_over_k(df, out_path)`
  * `plot_confusion_matrix(cm, out_path)`
* Move plotting code; no style changes.
  **Touch**
* New: `hippo_eval/reporting/plots.py`
* Edit: `hippo_eval/reporting/report.py`
  **Acceptance**
* Produced images identical (checksum) or pixel-close for deterministic plots.

## T4.report.3 — Extract rollup/index generation

**Why** Separate multi-run aggregation from rendering.
**What**

* Create `hippo_eval/reporting/rollup.py` with:

  * `collect_runs(reports_root) -> DataFrame`
  * `write_index(df, out_dir)`
* Move rollup logic here.
  **Touch**
* New: `hippo_eval/reporting/rollup.py`
* Edit: `hippo_eval/reporting/report.py`
  **Acceptance**
* `reports/<run_id>/index.html` unchanged for the golden run set.

## T4.report.4 — Thin coordinator

**Why** Keep `report.py` small.
**What**

* Leave `render_run_report(run_dir)` and high-level glue only.
* Add `hippo_eval/reporting/__init__.py` re-exporting `report`, `tables`, `plots`, `rollup`.
  **Touch**
* Edit: `hippo_eval/reporting/report.py`
* New/Edit: `hippo_eval/reporting/__init__.py`
  **Acceptance**
* Report CLI works; all artifacts identical.

---

## Notes (apply to every micro-task)

* **No behavior changes.** Only move code and re-wire imports.
* Keep imports absolute (`from hippo_eval.reporting.plots import …`).
* Run your **golden mini run** before starting; after each task, re-run and compare key artifacts:

  * `metrics.json`, `metrics.csv`, selected plots, `index.html`.
* If a step touches many imports, commit after creating the new module with stubs, then migrate functions in small chunks.

If you want, I can also emit these as separate Markdown “Codex task” blocks you can paste one by one.

---

## T5 — Unify memory adapter wrappers (reduce structural duplication)

**What to change**

* Create `hippo_mem/adapters/memory_base.py` with pooling/norm and a stable `forward(...)` surface.
* Make episodic/spatial adapters inherit from it; keep public names/imports stable.

**Acceptance**

* Shapes/params unchanged; telemetry unchanged in a smoke run.

---

## T6 — Enforce folder responsibilities

**What to change**

* Root `reports/` and `runs/` are outputs only; ensure ignored in VCS (except `.gitkeep`).
* Move `datasets/semantic/mini.jsonl` → `tests/fixtures/datasets/...` and update references.
* Move `experiments/*` → `docs/experiments/*` and update links.

**Acceptance**

* `pytest -q` green; example `docs/experiments/*/RUN.md` commands still work.

---

## T7 — Split tests by concern (and fix the misplaced “test\_eval”)

**What to change**

1. Create `tests/{algo,eval,reporting,cli,fixtures}`.
2. Move tests accordingly; keep shared fixtures in `tests/fixtures`.
3. Add CLI smoke tests for `scripts/*` (help + tiny dry run).
4. Replace the previous in-package script `hippo_mem/consolidation/test_eval.py` with:

   * `hippo_eval/consolidation/eval.py` (library)
   * `scripts/test_consolidation.py` (CLI)
   * `tests/eval/test_consolidation_eval.py` (unit tests)

**Acceptance**

* `pytest -q -k cli` < 10s; full suite green.

---

## T8 — Documentation updates

**What to change**

* Update `README.md`, `DESIGN.md`, `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `PROJECT_PLAN.md` to reflect:

  * New package: `hippo_eval`
  * Reporting and templates now under `hippo_eval/reporting`
  * Shims and deprecation notes
* Add a short “Migration notes” section.

**Acceptance**

* A new contributor can follow docs to run an eval and produce reports successfully.

---

## T9 — Import-boundary guardrails

**What to change**

* Add `ruff` and `black` config.
* Add a simple import-cycle test: **forbid `hippo_mem` importing `hippo_eval`**.
* Add a test asserting `import hippo_mem.eval` emits a `DeprecationWarning`.

**Acceptance**

* `make lint` green; boundary test fails if a cycle is introduced.

---

# Quick answers to your specific questions

* **“metrics” — move?** Yes. All of `hippo_mem/metrics/**` is evaluation-only → **move to `hippo_eval/metrics/**`** with a deprecation shim in `hippo_mem.metrics`.
* **“reporting” — move?** Yes. Entire reporting package and all root `reports/*` code/templates → **`hippo_eval/reporting/**`**. Root `reports/` becomes outputs-only.
* **“tasks” — move?** Yes. Code under `hippo_mem/tasks/**` is synthetic data generation for eval → **`hippo_eval/tasks/**`**. Root `tasks/` (Markdown) stays as docs.
* **`consolidation/test_eval.py` — where?** It’s an **evaluation runner**, not a unit test. Move to **`hippo_eval/consolidation/eval.py`** and expose a thin CLI `scripts/test_consolidation.py`.
* **T2 needs update?** Yes — it must cover **moving all reporting code and templates** (not templates only) and locking down root `reports/` to outputs-only.&#x20;
* **T4 only for `harness.py`?** No — it should also decompose `bench.py`, `datasets.py`, and `report.py` (very large) as outlined above.&#x20;

