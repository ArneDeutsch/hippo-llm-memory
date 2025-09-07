Awesome — I dug through the ZIP (at `/mnt/data/hippo-llm-memory-main.zip`) and mapped the current harness structure so the refactor plan is aligned with what’s actually there. I created a downloadable Markdown plan and pasted it below.

[Download the plan (Markdown)](sandbox:/mnt/data/refactor_eval_harness_stepwise_plan_2025-09-07.md)

---

# Stepwise Refactor Plan — `hippo_eval/eval/harness.py` and friends

**Project:** hippo-llm-memory
**Date:** 2025-09-07
**Goal:** Reduce branching and improve structure/readability/extensibility of the evaluation pipeline while **preserving functionality and CLI behavior**.

This plan is aligned with the current repository layout and code (ZIP inspected at `/mnt/data/hippo-llm-memory-main.zip`). It proposes **safe, incremental PRs** with explicit guardrails and test adaptations. The focus is to modularize per-algorithm logic (episodic / relational / spatial), centralize configuration handling, and keep public entry points stable.

---

## A. Current-State Snapshot (what we rely on not changing functionally)

* Primary entrypoints:

  * `hippo_eval/eval/harness.py` — contains:

    * `Task` dataclass (used by tests).
    * `evaluate(cfg, outdir, *, preflight=True)` — main single-run executor.
    * `main(cfg)` — CLI entrypoint wired by Hydra-based scripts (e.g., `scripts/eval_model.py`).
    * helpers for replay (`_run_replay`), store sizes (`_store_sizes`), and several algorithm-specific helpers (episodic/relational/spatial gating & ingestion).
    * merges shortcut keys `episodic`, `relational`, `spatial` into `cfg.memory.*` (see `configs/eval/default.yaml`).
* Runner utilities also import private harness helpers: `hippo_eval/harness/runner.py` calls `_h._run_replay`, expects `Task`, reset of `registry`/`gate_registry`, etc.
* Tests import private helpers directly: e.g., `tests/eval/test_harness_replay.py` imports `Task, _run_replay` from `hippo_eval.eval.harness`.
* Output shape and side-effects:

  * Writes `metrics.json`, `metrics.csv`, `meta.json` into `outdir` (via `hippo_eval/harness/io.py`).
  * Gate telemetry from `hippo_mem.common.telemetry.{registry, gate_registry}` is reset and populated.
  * Replay handling uses `_run_replay` and gate decisions per memory type.
* Config specifics:

  * Hydra config groups with shortcuts in `configs/eval/default.yaml`. Harness folds `episodic/relational/spatial` blocks into `cfg.memory.*` but keeps baselines possible.
  * CLI stability requirement: **no changes in options/semantics**.

**Observation:** `harness.py` has \~1800 lines with \~200 `if` branches; it interleaves algorithm selection, gating, retrieval and per-suite specifics. Good targets: extract per-algorithm logic behind a small interface; isolate config preprocessing; keep legacy imports as shims for tests.

---

## B. Design Principles for the Refactor

1. **Preserve behavior and CLI.** Outputs (`metrics.json/csv`, `meta.json`), telemetry, and default Hydra semantics remain identical.
2. **Small, verifiable steps.** Each PR compiles, passes tests, and includes golden-output checks.
3. **Adapter pattern per algorithm.** Introduce lightweight *Eval Adapters* to encapsulate episodic/relational/spatial differences for retrieval & teach/replay, plus telemetry wiring.
4. **Config preprocessing in one place.** A pure function that folds shortcut config blocks and applies ablations to the effective `cfg.memory.*` structure.
5. **Back-compat shims.** Keep `Task`, `_run_replay`, etc., in `harness.py` (or re-export from submodules) so tests and external imports keep working.
6. **No change in files under `hippo_eval/harness/*` I/O/metrics contracts.** Only import paths updated as needed.

---

## C. Target Structure After Refactor (new modules)

```
hippo_eval/
  eval/
    __init__.py
    harness.py                   # Thin orchestrator; public surface unchanged.
    adapters/                    # NEW: per-algorithm adapters for eval pipeline
      __init__.py
      base.py                    # EvalAdapter protocol/interface
      episodic.py                # EpisodicEvalAdapter
      relational.py              # RelationalEvalAdapter
      spatial.py                 # SpatialEvalAdapter
    config_utils.py              # NEW: config folding + ablation application
```

Notes:

* The repo already has `hippo_eval/harness/io.py`, `hippo_eval/harness/metrics.py`, `hippo_eval/harness/runner.py`. We won’t change their behavior; only wiring shifts in `harness.py`.
* We **do not** touch `hippo_mem` algorithm implementations; we only wrap them.

---

## D. Minimal Interface (to reduce `if`-sprawl)

In `hippo_eval/eval/adapters/base.py`:

* `class EvalAdapter(Protocol)` or ABC with:

  * `def present(self) -> str`: returns `"episodic" | "relational" | "spatial"`
  * `def build(self, cfg) -> dict`: materialize module objects (stores, gates, adapter objects) used later.
  * `def retrieve(self, cfg, modules, item, **ctx) -> tuple[list[str], list[tuple[int,int]], list[str], dict]`

    * returns injected texts, spans/positions, source-ids, and debug/telemetry context.
  * `def teach(self, cfg, modules, item, *, dry_run: bool, gc) -> None`

    * used by `_run_replay` and “teach” mode ingestion.
  * `def store_size(self, modules) -> tuple[int, dict]`

    * count and diagnostics for `metrics["store"]`.

Concrete adapters (`episodic.py`, `relational.py`, `spatial.py`) implement the above by extracting **existing code** from `harness.py` (e.g., `_ingest_episodic`, relational tuple extraction, spatial graph ingestion, gate usage, and retrieval packers).

A tiny factory in `adapters/__init__.py`:

```python
REGISTRY = {
  "episodic": EpisodicEvalAdapter(),
  "relational": RelationalEvalAdapter(),
  "spatial": SpatialEvalAdapter(),
}
def enabled_adapters(cfg) -> dict[str, EvalAdapter]:
    # Use presence and cfg.memory.* to select active algorithms
    return {k: a for k, a in REGISTRY.items() if k in (cfg.get("memory") or {})}
```

---

## E. Step-by-Step Refactor Plan (safe PRs)

### PR-0: Preflight: Characterize & freeze current behavior

* [ ] Add a small **golden snapshot** harness test that runs a tiny suite for each algorithm with seed `1337` and `n=5`, writing to a temp outdir, then asserts:

  * `metrics.json`, `meta.json` keys/values (excluding time and rss fields) are unchanged.
  * `metrics.csv` header and row ordering unchanged.
* [ ] Ensure `tests/eval/test_harness_replay.py` still imports `Task, _run_replay` from `hippo_eval.eval.harness` (no changes yet).
* [ ] CI: cache these outputs for comparison in subsequent PRs.

### PR-1: Extract **config folding** utilities

* Create `hippo_eval/eval/config_utils.py` with:

  * `merge_memory_shortcuts(cfg: DictConfig) -> None` (move the logic from `harness.py` lines \~186–215 in current file).
  * `apply_ablation_flags(cfg: DictConfig, flat_ablate: dict) -> None` (move logic that toggles `*.gate.enabled`, `episodic.use_completion`, etc.).
* In `harness.py`:

  * Replace inline code with calls to these new functions.
  * Keep function names/signatures identical for external callers.
* Tests: none should break. Golden snapshot must remain **identical**.

### PR-2: Introduce the **EvalAdapter base** and **episodic adapter**

* Add `adapters/base.py` and `adapters/episodic.py`.
* Move existing episodic helpers from `harness.py` into the episodic adapter with zero behavior change:

  * Key creation (`_episodic_key_from_text`), write-gate usage, `episodic_retrieve_and_pack` wiring.
  * Provide methods `build`, `retrieve`, `teach`, `store_size` that mirror current code paths.
* In `harness.py`, create a minimal adapter selection:

  ```python
  from hippo_eval.eval.adapters import enabled_adapters
  adapters = enabled_adapters(cfg)
  modules = {name: adapters[name].build(cfg) for name in adapters}
  ```
* Replace episodic-specific `if "episodic" in modules:` blocks with calls to the adapter API **only for episodic** code paths. Keep relational/spatial inline for now.
* Re-export `Task` and `_run_replay` from `harness.py` unchanged (temporarily call the adapter method inside `_run_replay`).
* Tests: update none. Verify golden snapshot.

### PR-3: Extract **relational adapter**

* Add `adapters/relational.py` with current logic:

  * Tuple extraction (`extract_tuples`), relational gate (`RelationalGate`), and retrieval (`relational_retrieve_and_pack`).
  * Migrate ingestion for replay/teach and telemetry increments.
* Update `harness.py`:

  * Replace relational branches with adapter calls.
* Tests: keep public imports stable. Run golden snapshot. Also ensure relational unit tests still pass (e.g., `tests/algo/test_relational_*`).

### PR-4: Extract **spatial adapter**

* Add `adapters/spatial.py` with:

  * Spatial gate (`SpatialGate`) usage and place graph ingestion (currently in `_ingest_spatial` and parts of `_run_replay`).
  * Retrieval via `spatial_retrieve_and_pack` and store diagnostics.
* Update `harness.py` to delegate spatial logic to adapter.
* Tests: run golden snapshot, spatial tests (`tests/adapters/test_spatial_adapter.py`) and `tests/algo/test_spatial_*` if present.

### PR-5: Thin the harness orchestrator

* With all adapters in place:

  * Keep `evaluate`, `main`, `_run_replay`, `Task` in `harness.py` as **stable public surface**.
  * Internally, `evaluate` becomes:

    1. Preflight + telemetry resets.
    2. Dataset load → `Task` list.
    3. `merge_memory_shortcuts(cfg)`; `apply_ablation_flags(cfg, flat_ablate)`.
    4. Build `adapters` and `modules` dicts via factory.
    5. Tokenizer/model loading unchanged.
    6. For each item: **loop over active adapters** to assemble injected context lists (concat in stable, existing order), and track per-adapter latencies/gating.
    7. Generation + metric calculation unchanged.
    8. Replay loop → `adapters[name].teach(...)`.
    9. Store sizes/diags via `adapters[name].store_size(...)`.
  * Ensure the concatenation order of contexts replicates the current order exactly (episodic, relational, spatial — confirm from code paths).

### PR-6: Back-compat shims & imports

* Keep `Task`, `_run_replay` same name and signature. Inside, dispatch to the right adapter methods.
* If any helper was renamed/moved, re-export names in `harness.py` to avoid breaking tests and external users.
* Add `__all__` to `harness.py` to explicitly expose the stable API.

### PR-7: Test suite updates & coverage hardening

* **Unit tests:** Add adapter-level tests mirroring the logic formerly tested indirectly via `harness.py`. These must use the same fixtures and Hypothesis strategies where applicable.
* **Property tests:** (lightweight) Ensure that enabling/disabling a memory module via ablation produces identical outputs to baseline for that module (or empty effects if disabled).
* **Golden outputs:** Re-run the snapshot tests.
* **Smoke tests:** Ensure `scripts/ci_smoke_eval.sh` still passes (preflight behavior unchanged).

### PR-8: Clean up dead code & docs

* Remove leftover algorithm-specific helpers from `harness.py` that are now internal to adapters (keep only re-exports if tests import them).
* Update `DESIGN.md` and `EVAL_PLAN.md` w/ a short section on the eval adapter architecture.
* Add developer docs in `hippo_eval/eval/adapters/README.md` describing the interface and how to add a new memory algorithm.

---

## F. Acceptance Criteria (per PR)

* **Zero functional drift:** Golden snapshot diff is empty (ignoring volatile fields: timestamps, RSS, token counts may vary slightly if we do not touch generation; if necessary, assert within tight tolerances and document).
* **CLI stable:** `scripts/eval_model.py` accepted flags and behaviors unchanged. `--preset`, `--suite`, `--n`, `--seed`, replay flags behave identically.
* **Test stability:** Existing tests either pass unchanged or are trivially updated to import from the same public names. No test rewrites of semantics.
* **Readability:** `harness.py` shrinks to a thin coordinator. Algorithm branching replaced by adapter dispatch.

---

## G. Risk Mitigation & Rollback

* Each PR is isolated and can be reverted independently.
* Keep large moves as pure “cut-and-paste with import updates”; avoid logic edits.
* Maintain strict type hints and keep the adapter interface minimal to avoid cross-module coupling.
* Use `gate_registry/registry` counters in tests to detect hidden side-effect changes.

---

## H. Concrete Mapping From Current Code → Adapters

* **Episodic (`adapters/episodic.py`)**

  * Move: `_episodic_key_from_text`, `_ingest_episodic`, calls to `WriteGate`, `episodic_retrieve_and_pack`, `TraceSpec` usage.
  * Modules dict keys preserved: `{ "store": EpisodicStore, "adapter": <model adapter>, "gate": WriteGate? }` (exact object names preserved to avoid confusion in metrics).

* **Relational (`adapters/relational.py`)**

  * Move: tuple extraction `extract_tuples`, `RelationalGate` usage, `relational_retrieve_and_pack`, relational teach ingestion.
  * Keep the same gating counters and provenance labeling.

* **Spatial (`adapters/spatial.py`)**

  * Move: `SpatialGate`, graph observe/aggregate logic for trajectories, `spatial_retrieve_and_pack`, and store diags.
  * Preserve the exact decision/action semantics for gate outcomes.

---

## I. Non-Goals

* No algorithmic changes.
* No changes to `hippo_mem` modules.
* No change in datasets or loaders, or prompt encoding/generation defaults.

---

## J. Follow-Ups (optional, after main refactor lands)

* Introduce a **per-adapter latency/telemetry block** into `metrics.json` for richer diagnostics (off by default).
* Adapter auto-discovery via entry points (kept out-of-scope for now to keep CLI 100% stable).

---

## K. Quick Task Checklist (for Codex or CI)

* [ ] PR-0 golden snapshot fixtures.
* [ ] PR-1 config\_utils extracted and wired.
* [ ] PR-2 episodic adapter extracted; harness delegates episodic logic.
* [ ] PR-3 relational adapter extracted; harness delegates relational logic.
* [ ] PR-4 spatial adapter extracted; harness delegates spatial logic.
* [ ] PR-5 orchestrator thinned; context concatenation order preserved.
* [ ] PR-6 back-compat shims validated against tests importing private names.
* [ ] PR-7 adapter tests + smoke tests green.
* [ ] PR-8 docs updated; dead code removed.

---

## L. How to Verify Locally (example)

```bash
# Baseline
RUN_ID=ref_before python scripts/eval_model.py \
  task=episodic n=5 seed=1337 preset=baselines/core compute.pre_metrics=true

# After PR-2 (episodic extraction)
RUN_ID=ref_after python scripts/eval_model.py \
  task=episodic n=5 seed=1337 preset=baselines/core compute.pre_metrics=true

# Compare (ignoring timestamps, rss)
diff <(jq -S 'del(.compute.rss_mb, .compute.time_ms_per_100)' runs/ref_before/**/metrics.json | jq -S) \
     <(jq -S 'del(.compute.rss_mb, .compute.time_ms_per_100)' runs/ref_after/**/metrics.json | jq -S)
```
