# Next Refactoring Plan — `hippo_eval/eval/harness.py` and the `hippo_eval` tree

**Project:** hippo-llm-memory  
**Date:** 2025-09-08  
**Goal:** Make the evaluation pipeline clearer, shorter, and easier to extend — *without changing behavior or the CLI*. Build directly on the prior plan in `plans/2025-09-07/refactor-harness.md`, keep the golden tests, and ship changes in small, reversible steps.

---

## 0) What I inspected (quick facts)

- File sizes/shape today:
  - `hippo_eval/eval/harness.py`: ~1520 lines, 57270 bytes.
  - Long functions that still dominate complexity:
- `evaluate` — ~386 LOC
- `_evaluate` — ~311 LOC
- `_write_outputs` — ~201 LOC
- `_dataset_path` — ~100 LOC
- `preflight_check` — ~81 LOC
- `main` — ~79 LOC
- Package layout tensions:
  - `hippo_eval/eval/harness.py` (orchestrator) **imports** helpers from `hippo_eval/harness/` (e.g., `io.py`, `metrics.py`). This split is confusing because both are called “harness”.
  - There are **two “metrics” concepts**:
    - **Scoring** in `hippo_eval/metrics/` (e.g., `scoring.py`, `spatial.py`) — *model answer scoring*.
    - **Aggregation/accumulation** in `hippo_eval/harness/metrics.py` — *harness-level rollups / diagnostics*.
  - Mode handling (`teach`/`test`/`replay`) appears in several places as `if/elif` checks rather than a first‑class concept.
  - Adapters in `hippo_eval/eval/adapters/` are nicely factored, but some mode‑ish concerns (e.g., “no retrieval during teach”) leak into the pipeline.

**Conclusion:** We can remove remaining branching and naming ambiguity by introducing an explicit **Mode** abstraction, consolidating “harness” utilities under one namespace, and splitting the 300–380 LOC functions into cohesive units (I/O, generation, teaching, scoring, write‑out).

---

## 1) Design principles for this round

1. **No behavior changes, no CLI changes.** All refactors must be covered by golden tests and existing unit tests.
2. **Backwards‑compatible imports.** Keep `hippo_eval.eval.harness` as a public entry point (re‑exports allowed).
3. **Make “mode” explicit.** Replace scattered `if cfg.mode == ...` with a small **strategy** that encapsulates what each mode does.
4. **Clarify terminology.** Reserve “metrics” for *scoring*; call harness rollups **aggregation**.
5. **Shorten functions.** Target ≤100 LOC for `evaluate()` and ≤120 LOC for internal workhorses.

---

## 2) Proposed structure (target end‑state, non‑breaking)

```
hippo_eval/
├─ eval/
│  ├─ harness.py              # thin façade; re-exports public API (Task, evaluate, evaluate_matrix)
│  ├─ pipeline.py             # NEW: orchestrator (formerly the guts of harness.py, but slimmer)
│  ├─ modes.py                # NEW: Mode enum + ModeStrategy implementations
│  ├─ generation.py           # NEW: tokenization/generation helpers (chat template, long ctx, etc.)
│  ├─ writers.py              # NEW: write_meta/write_metric/write_csv (moved from harness/io.py)
│  ├─ types.py                # NEW: Task dataclass & small result types
│  └─ adapters/               # as is
├─ harness/
│  ├─ __init__.py             # remains; re-exports now point to eval.writers and eval.aggregation
│  ├─ aggregation.py          # RENAMED from metrics.py (SuiteAccumulator etc.)
│  ├─ io.py                   # DEPRECATED: re-export layer to eval.writers (with warning)
│  └─ runner.py               # unchanged for now
├─ metrics/                   # scoring stays here (em_norm, f1, spatial_kpis, …)
│  └─ ...
```

- **Compatibility:** `hippo_eval.harness.metrics` continues to import from `hippo_eval.harness.aggregation`. `hippo_eval.harness.io` re-exports from `hippo_eval.eval.writers`.
- **Public API:** `from hippo_eval.eval.harness import Task, evaluate, evaluate_matrix` still works.

---

## 3) Mode as a first‑class concept (Strategy)

Create a narrow interface that hides branching:

```python
# hippo_eval/eval/modes.py
from enum import Enum
from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from .types import Task, RunInputs, RunOutputs

class Mode(Enum):
    TEACH = "teach"
    TEST = "test"
    REPLAY = "replay"

@dataclass
class ModeStrategy:
    """Decide retrieval rules and actions per mode."""

    def pre_run(self, inputs: RunInputs) -> None: ...
    def process_task(self, inputs: RunInputs, task: Task) -> RunOutputs: ...
    def post_run(self, inputs: RunInputs) -> None: ...

# Minimal behaviors:
# - TEACH: ingests, (optionally) disables retrieval; computes metrics iff configured.
# - TEST: retrieval on; no ingestion.
# - REPLAY: writes ground-truth answers back; updates gate telemetry (no metrics by default).
```

Wire `evaluate()` to select a strategy via a tiny factory (no conditionals scattered across the pipeline).

---

## 4) Extract cohesive helpers

**4.1 Generation & prompting (`eval/generation.py`)**

- `apply_chat_template(tokenizer, system_prompt, user_prompt) -> str`
- `generate(model, tokenizer, prompt, max_new_tokens, long_context=False) -> tuple[str, int, int]`
- `postprocess(raw_pred, task, mode, enforce_short, enforce_udlr) -> str`

**4.2 Aggregation & diagnostics (`harness/aggregation.py`)**

- Move `SuiteAccumulator`, `MetricSchema` from `hippo_eval/harness/metrics.py` into `aggregation.py`.  
  Keep `hippo_eval.harness.metrics` as a deprecation shim (`from .aggregation import *`).

**4.3 Writers (`eval/writers.py`)**

- Move `write_meta`, `write_metric`, `write_csv` from `hippo_eval/harness/io.py` into this module.
- Keep `hippo_eval.harness.io` as an import shim that emits a one‑line `DeprecationWarning`.

**4.4 Types (`eval/types.py`)**

- Move `Task` out of `eval/harness.py`.
- Add small dataclasses:
  - `RunInputs(cfg, modules, adapters, tokenizer, model, gating, suite, retrieval_enabled, long_context_enabled, use_chat_template, system_prompt)`
  - `RunOutputs(row, metrics, in_tokens, gen_tokens, elapsed_s)`

---

## 5) Shrink the heavy functions

**5.1 `_evaluate` → split into:**

- `build_run_inputs(cfg, modules, tokenizer, model, adapters, ...) -> RunInputs`
- `iterate_tasks(inputs, tasks, strategy) -> Iterable[RunOutputs]`
- `compute_task_metrics(...)` — (thin wrapper over `hippo_eval.metrics.scoring`)
- `update_gating_for_teach(...)` — (strategy‑controlled)
- `summarize_run(outputs) -> (rows, metrics, in_tok, gen_tok, elapsed)`

**5.2 `evaluate` → slim orchestrator (≤100 LOC)**

- Prepare config (`_apply_model_defaults`, `merge_memory_shortcuts`), load suite & modules.
- Build `RunInputs`.
- If `mode == TEACH and cfg.compute.pre_only`: run once; write outputs; return.
- Otherwise: run “pre”, optional `run_replay()`, then “post”.
- Delegate all writing to `writers.py`.

**5.3 `_write_outputs` → `writers.write_all(...)`**

- Single call site; contains CSV/JSON writes and success flag creation.

---

## 6) Naming cleanup to avoid “metrics vs metrics”

- Rename file `hippo_eval/harness/metrics.py` → `hippo_eval/harness/aggregation.py`.
- Add module shim:
  ```python
  # hippo_eval/harness/metrics.py
  from .aggregation import *  # noqa
  ```
- Update internal imports (non‑public) to `from hippo_eval.harness.aggregation import SuiteAccumulator`.

Rationale: keep **scoring** vs **aggregation** clearly separate.

---

## 7) Folder relationship: `eval/` vs `harness/`

- Keep both **for now** (least risk).
- Move functionality, not file paths that user code imports. Provide shims + warnings.
- Optional follow‑up (Plan+1 below) consolidates everything under `eval/` when CI is green.

---

## 8) Test & “golden” safety net

- Keep and run the existing **golden test** and all `tests/eval/test_harness_*` files.
- Add focused new tests:
  1. **Mode wiring:** `tests/eval/test_mode_strategy.py` — asserts retrieval/ingest toggles per mode and that outputs match pre‑refactor golden fixtures.
  2. **Writers roundtrip:** a temp dir write/read of `meta.json`, `metrics.json`, `metrics.csv`.
  3. **Adapters unchanged:** quick smoke that enabled adapters and order remain identical.
  4. **Import shims:** importing `hippo_eval.harness.metrics` and `hippo_eval.harness.io` still works.

**Acceptance gates for each step** are listed in the plan below.

---

## 9) Stepwise plan (safe, incremental commits)

### Step 1 — Introduce `Mode` enum + skeleton strategies
- Add `eval/modes.py` with `Mode` and `ModeStrategy` protocol.
- Create no‑op strategy implementations that only carry flags (`retrieval_enabled`, `ingest_enabled`).
- **Acceptance:** new unit test passes; no production code calls these yet.

### Step 2 — Extract writers
- Add `eval/writers.py`; move `write_meta`, `write_metric`, `write_csv` here.
- Convert `eval/harness.py` to import from `.writers`.
- Turn `hippo_eval/harness/io.py` into a re‑export shim with `warnings.warn`.
- **Acceptance:** `tests/eval/test_harness_io_fast.py` still green; grep shows `write_*` used only via the new module in repo code.

### Step 3 — Extract types
- Add `eval/types.py`; move `Task` and introduce `RunInputs`, `RunOutputs`.
- Adjust imports in `eval/harness.py` and tests that touched `Task`.
- Keep `Task` in `eval/harness.py` via `from .types import Task` and `__all__`.
- **Acceptance:** tests that import `Task` from the old path still pass.

### Step 4 — Extract generation helpers
- Add `eval/generation.py` with `apply_chat_template`, `generate`, `postprocess`.
- Replace the generation snippet in `_evaluate` with calls to these helpers.
- **Acceptance:** golden outputs of `metrics.json` and `meta.json` unchanged (byte‑for‑byte) for a fixed seed run.

### Step 5 — Introduce strategies and use them
- Implement real `TeachStrategy`, `TestStrategy`, `ReplayStrategy` controlling:
  - retrieval enablement (e.g., “no retrieval during teach”),
  - ingestion behavior (teach only),
  - replay writes (in `ReplayStrategy.post_run` or dedicated `run_replay()` wrapper).
- Replace remaining `if mode == ...` in `_evaluate`/`evaluate` with strategy calls.
- **Acceptance:** telemetry counters equal pre‑refactor values; all gate tests green.

### Step 6 — Split `_evaluate` and slim `evaluate`
- Factor `_evaluate` into `build_run_inputs`, `iterate_tasks`, `summarize_run`.
- Reduce `evaluate` to orchestration and writing (≤100 LOC).
- **Acceptance:** `pytest -k harness` green; LOC thresholds met.

### Step 7 — Rename `harness/metrics.py` → `harness/aggregation.py`
- Add shim at `harness/metrics.py` (re-export).
- Update internal imports in repo to `aggregation`.
- **Acceptance:** `ripgrep -n "from hippo_eval.harness.metrics"` in repo shows only tests for the shim; all tests pass.

---

## 10) CI & compatibility checks

- Run `scripts/ci_smoke_eval.sh` and `pytest -q`.
- Run `scripts/run_eval_example.sh` (or your local equivalent) and compare:
  - `meta.json`, `metrics.json`, `metrics.csv` via `cmp -s`.
  - Gate telemetry snapshots (counts only; timestamps excluded).

---

## 11) “Plan+1”: follow-ups **after** green CI

These are optional once the above lands cleanly.

1. **Consolidate “harness” under `eval/`:** Move `hippo_eval/harness/{aggregation.py,runner.py}` to `hippo_eval/eval/harness/` and keep top‑level shims for one release cycle.
2. **Telemetry context object:** Replace direct imports of global telemetry with an `EvalContext` that wraps `registry` and `gate_registry` access. Keep the same counters; just centralize access.
3. **Store isolation policy type:** Replace the `isolate: str = "none"` flag with an `Enum` (`NONE`, `TASK`, `SUITE`) and push policy to strategies.
4. **Public API docstring tests:** Add doctests to `eval/modes.py` showing expected behavior of each strategy.

---

## 12) Risks & mitigations

- **Hidden coupling in tests** (importing private helpers): mitigate with re‑exports and `__all__` in `eval/harness.py`.
- **Telemetry deltas**: assert equality on counters in unit tests; keep replay logic byte‑identical.
- **Naming churn**: keep shims for at least one release; log a one‑line deprecation warning.

---

## 13) Work breakdown (engineering tickets)

- **ENG‑201**: Add `Mode` and skeleton strategies; tests.
- **ENG‑202**: Move writers; add shim in `harness/io.py`.
- **ENG‑203**: Extract types (`Task`, `RunInputs`, `RunOutputs`); keep re‑exports.
- **ENG‑204**: Extract generation helpers; keep outputs identical.
- **ENG‑205**: Implement and adopt strategies; remove scattered `if mode ...`.
- **ENG‑206**: Split `_evaluate`; slim `evaluate`.
- **ENG‑207**: Rename `harness/metrics.py` → `aggregation.py`; add shim.
- **ENG‑208**: CI run + golden verification task; freeze fixtures.
- **ENG‑209 (optional)**: “Plan+1” consolidation of `harness` under `eval/`.

Each ENG task should include: before/after LOC snapshot, affected files list, and a golden run comparison.

---

## 14) Review checklist (to run after Step 6)

- `evaluate()` ≤ 100 LOC and `_evaluate` split as specified.
- `grep -R "if .*mode" hippo_eval/eval` shows only occurrences inside `modes.py` and tests.
- Imports of `hippo_eval.harness.metrics` and `hippo_eval.harness.io` still succeed.
- Golden outputs match for `SIZES=50`, `SEEDS=1337`.
- Gate counters match across modes in telemetry tests.

---

## 15) Next iteration seed plan (informed by the review)

If all green:
- Proceed with **Plan+1.1** (telemetry context) and **Plan+1.2** (store isolation enum).
If deltas appear:
- Add targeted tests where deltas arose (e.g., relational gating during teach).
- Keep strategies but add explicit adapter hooks (e.g., `adapter.on_teach_noop_decision`) to make intent obvious.

---

**Deliverables of this plan**
- PRs for Steps 1–7 with shims and zero behavioral drift (golden tested).
- A short migration note: “`harness.metrics` → `harness.aggregation`; `harness.io` → `eval.writers` (re-exports stay for now).”
