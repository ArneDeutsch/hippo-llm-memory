# What’s in the repo (high-level)

* **Core package:** `hippo_mem/` — algorithms, stores, common utils, plus **evaluation** (`hippo_mem/eval`) and **reporting** (`hippo_mem/reporting`).
* **Scripts:** `scripts/` — \~23 Python entry points (40–157 LOC each).
* **Templates:** `reports/templates/` (Jinja2), but **reporting code** lives in `hippo_mem/reporting`.
* **Generated outputs:** `data/`, `runs/` (empty in the ZIP), `reports/` (currently also contains code templates).
* **Hydra configs:** `configs/` (widely referenced).
* **Small “edge” dirs:** `datasets/` (1 file), `experiments/` (3 subdirs with `RUN.md` + `run.yaml` each).
* **Tests:** `tests/` — large, and they mix algorithm, evaluation, reporting, and scripts concerns.

Notable hotspots:

* `hippo_mem/eval/harness.py` has several very long functions (e.g., `evaluate` \~360 LOC, `_write_outputs` \~240 LOC).
* Very similar adapters live in multiple places (e.g., `hippo_mem/episodic/adapter.py` vs `hippo_mem/spatial/adapter.py`) alongside wrapper adapters in `hippo_mem/adapters/…`.
* `scripts/run_baselines.py` reimplements aggregation logic and even uses `sys.path.insert(...)` instead of importing the library layer.

# Where I agree / disagree with your observations

* **“Some files/functions get too complex.”**
  ✅ Agree. `hippo_mem/eval/harness.py` (and a couple in `hippo_mem/reporting/report.py`) exceed reasonable function size and mix orchestration, IO, and pure evaluation in single bodies.

* **“Algorithms and evaluation live together; better to separate.”**
  ✅ Agree. Today `hippo_mem/eval` imports algorithms; the inverse doesn’t happen, but keeping eval inside the same top-level package muddies boundaries. A dedicated top-level package like `hippo_eval/` cleanly expresses the one-way dependency (`hippo_eval -> hippo_mem`).

* **“Pipeline design—add abstractions to simplify.”**
  ✅ Agree. The harness mixes: dataset loading, model wiring, run orchestration, metrics, and persistence. Extracting clear service layers (dataset→iterator, model→runner, metrics→writer) will shrink surface area and improve testability.

* **“Lots of near-duplication.”**
  ✅ Partly. There aren’t many literal clones, but there’s **structural duplication**: very similar adapter wrappers across episodic/spatial; duplicated “bench”/aggregation notions across `scripts/` and `hippo_mem/eval/`. These should converge on shared base classes/functions.

* **“Scripts folder should be slim.”**
  ✅ Agree. Several scripts contain logic and path hacks; that logic belongs in library modules with scripts acting as thin CLIs.

* **“Generated folders contain programming artifacts (e.g., templates in reports/).”**
  ✅ Agree. Templates should live with code (`hippo_mem/reporting/templates`); keep root-level `reports/` and `runs/` strictly outputs.

* **“Configs / datasets / experiments folders feel accidental.”**
  ⚠️ Mixed.

  * `configs/` is widely referenced by docs and eval code (Hydra). Keep it.
  * `datasets/semantic/mini.jsonl` is a **fixture**; it belongs under `tests/fixtures/datasets/`.
  * `experiments/*` (each with `RUN.md` + `run.yaml`) look like **examples/docs**; moving them under `docs/experiments/` (or `examples/`) will declutter the root.

* **“Tests for algorithms and pipelines are mixed.”**
  ✅ Agree. We can split into `tests/algo/`, `tests/eval/`, `tests/reporting/`, `tests/cli/` with a small compatibility layer for shared fixtures.

---

# Proposed target layout (after refactor)

```
hippo_mem/                  # algorithms, stores, adapters, utils (no evaluation here)
hippo_eval/                 # NEW: evaluation/pipeline code (moved out of hippo_mem)
  datasets.py
  harness/
    __init__.py
    runner.py               # extracted from harness.py
    io.py                   # write/read helpers
    metrics.py              # scoring + schema
  baselines.py
  bench.py
  models.py
  encode.py
  audit.py
scripts/                    # thin CLIs that call hippo_eval/hippo_mem APIs
configs/                    # keep (Hydra)
docs/experiments/           # moved from experiments/
tests/
  algo/
  eval/
  reporting/
  cli/
  fixtures/
    datasets/               # moved from datasets/
data/                       # generated only
runs/                       # generated only
reports/                    # generated only
```

# What should move (and what should stay)

Here’s a concrete decision table over the top-level packages under `hippo_mem/`:

| Path                                                                                                | Keep in `hippo_mem` (algos/core) | Move to `hippo_eval` (eval/pipeline) | Notes                                                                                                                                                                                                |
| --------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adapters/`                                                                                         | ✅                                |                                      | Algorithm integration code.                                                                                                                                                                          |
| `common/`                                                                                           | ✅                                |                                      | Telemetry, I/O, utilities used across both.                                                                                                                                                          |
| `consolidation/`                                                                                    | ✅                                | ➖ (partial)                          | Keep `trainer.py`, `worker.py`, `replay_dataset.py` in core. **Move** `consolidation/test_eval.py` to `hippo_eval/consolidation/eval.py` and add a thin CLI wrapper `scripts/test_consolidation.py`. |
| `episodic/`, `spatial/`, `relational/`, `memory/`, `planning/`, `training/`, `retrieval/`, `utils/` | ✅                                |                                      | Core algorithmic functionality.                                                                                                                                                                      |
| `eval/`                                                                                             |                                  | ✅                                    | Entire package moves (your T1).                                                                                                                                                                      |
| `metrics/`                                                                                          |                                  | ✅                                    | These are evaluation metrics (EM/F1/spatial path, etc.). Move to `hippo_eval/metrics/`. Keep a shim at `hippo_mem/metrics` for compatibility.                                                        |
| `reporting/`                                                                                        |                                  | ✅                                    | Reporting consumes eval outputs; move whole package to `hippo_eval/reporting/`. Keep a shim at `hippo_mem/reporting`.                                                                                |
| `tasks/`                                                                                            |                                  | ✅                                    | Synthetic task/data generation used by the harness; move to `hippo_eval/tasks/`. Update imports in `datasets.py`.                                                                                    |

Non-package folders:

* Root `reports/` currently contains **code** (`health.py`, `render_baselines.py`, `plots/`) and **templates** — these should move under `hippo_eval/reporting/…`; keep root `reports/` strictly **outputs**. (This supersedes your existing T2, which only moved templates.)&#x20;
* Root `tasks/` (Markdown Codex instructions) should **stay** as docs; it’s distinct from `hippo_eval/tasks/` (code).
* `datasets/semantic/mini.jsonl` is a **fixture** → move to `tests/fixtures/datasets/semantic/mini.jsonl`.
* `experiments/*` are examples/docs → move to `docs/experiments/*`.

# Why these moves

* Keeps a one-way boundary: **`hippo_eval → hippo_mem`** (never the other way).
* De-couples pipelines, metrics, reporting, and synthetic data from core memory algorithms.
* Reduces accidental import cycles and clarifies ownership for tests and CI.

This directly addresses the gaps in your current T1/T2 and expands T4 beyond `harness.py` to additional oversized modules you flagged implicitly (bench/datasets/report).&#x20;

---

# Updated Codex tasks (revised & expanded)

You can paste these as separate Codex tasks. They’re self-contained, safe-order, and include acceptance criteria.
