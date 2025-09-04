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

And move `reports/templates/` → `hippo_mem/reporting/templates/` (code and templates colocated). `hippo_mem/reporting/report.py` updated to load from its own package.
