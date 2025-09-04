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

## T4 — Decompose **oversized modules**: harness, bench, datasets, report

**Why**
Improve readability/coverage and reduce risk.

**What to change**

Break these files into small, single-purpose modules:

* `hippo_eval/harness.py` (was \~1500 LOC) → extract:

  * `harness/runner.py` (`build_runner`, `run_suite`)
  * `harness/metrics.py` (`collect_metrics`, schema helpers)
  * `harness/io.py` (FS interactions)
* `hippo_eval/bench.py` (was \~700 LOC) → extract orchestration vs. summarization vs. FS layout.
* `hippo_eval/datasets.py` (was \~700 LOC) → split generators vs. CLI wrappers; move synthetic generators next to `hippo_eval/tasks/` where appropriate.
* `hippo_eval/reporting/report.py` (was \~1200 LOC) → extract:

  * `reporting/tables.py` (Markdown table assembly)
  * `reporting/plots.py` (matplotlib helpers)
  * `reporting/rollup.py` (index generation)
  * keep `report.py` as a thin coordinator.

**Acceptance**

* Byte-identical outputs for a known mini run (`metrics.json`/`metrics.csv`) pre vs. post (add a golden fixture).
* Unit tests cover the new helpers; CLI behavior unchanged.

> This extends your original T4 beyond `harness.py`, as you suspected.&#x20;

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
