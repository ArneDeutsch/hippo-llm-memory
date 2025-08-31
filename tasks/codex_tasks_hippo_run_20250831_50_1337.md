# Codex Task Pack — Telemetry, Stores, Gating & Baseline Fixes
_Run: 20250831_50_1337 • Generated: 2025-08-31 09:27_

This task pack turns the findings from `hippo-pipeline-review_20250831_50_1337.md` into **actionable Codex issues**.  
Follow the order for quickest recovery: **telemetry reset → baseline isolation → retrieval gating → replay writes → datasets → reporting → docs**.

> Template for each task follows `.github/ISSUE_TEMPLATE/codex_task.md`.

---

## T1 — Reset telemetry counters per run (no cross‑contamination)
* **Goal:** Ensure retrieval/gate counters start at zero for every process/run and for each suite.
* **Files to touch:**
  - `hippo_mem/eval/harness.py` — at the very start of `evaluate()` and also at the start of `_evaluate()`, call `registry.reset()` and `gate_registry.reset()`.
  - `hippo_mem/eval/bench.py` — at the start of `evaluate()` and before `_eval_tasks(...)`, call the same resets.
* **Acceptance tests:**
  - Add `tests/test_telemetry_reset.py` asserting that two back‑to‑back calls to the harness for different presets do **not** carry counters over.
  - Update `tests/test_telemetry_sanity.py` to use `registry.reset()` in setup (if not already) and to verify zeros before any retrieval.
* **Run:** `pytest -q`
* **Context:** Cross‑run contamination produced non‑zero retrieval stats in baselines.

---

## T2 — Do **not** fold shortcut blocks into `memory.*` unless a memory preset is active
* **Goal:** Prevent baselines (e.g., `baselines/core`) from instantiating memory modules due to defaults in `configs/eval/default.yaml`.
* **Files to touch:**
  - `hippo_mem/eval/harness.py` — in the “Shortcut overrides folded into memory.*” block of `_apply_model_defaults()`:
    - Only merge `cfg.episodic/relational/spatial` into `cfg.memory` if **either** (a) `cfg.preset` path starts with `"memory/"` **or** (b) `cfg.memory` was **explicitly** set non‑null by the preset.
    - Pseudocode:
      ```python
      is_memory_preset = str(cfg.get("preset",""))).split("/",1)[0] == "memory"
      if is_memory_preset or cfg.get("memory") not in (None, {}):
          # fold episodic/relational/spatial blocks into cfg.memory
      ```
  - `configs/eval/default.yaml` — leave gate defaults but **ensure they don’t auto‑activate memory** (no `memory:` root key).
* **Acceptance tests:**
  - New `tests/test_baselines_have_no_memory.py`: run harness with `preset=baselines/core` and assert `modules == {}` and that the resulting `metrics.json` has `store.size==0` and `retrieval.*.requests==0` for all memories.
* **Run:** `pytest -q`
* **Context:** Baselines currently show retrieval requests and non‑zero stores because defaults were folded into `memory.*` unconditionally.

---

## T3 — Honor `retrieval.enabled=false` and `longctx.enabled=true` flags
* **Goal:** Retrieval should only run when explicitly enabled; long‑context concatenation should be used instead when requested.
* **Files to touch:**
  - `hippo_mem/eval/harness.py` — in `_evaluate()`:
    - Before building `mems`, read `cfg.get("retrieval", {}).get("enabled", False)`; **only** execute episodic/relational/spatial retrieval when `True`.
    - Respect `cfg.get("long_context", {}).get("enabled", False)`: if true and retrieval disabled, **do not** retrieve; pass long context to the prompt encoding path (simulate initially by appending a placeholder `"[CTX]"` span and count tokens in compute telemetry).
  - `hippo_mem/eval/bench.py` — mirror the same gating for the bench path to keep parity.
* **Acceptance tests:**
  - Extend `tests/test_adapter_wiring.py` or add `tests/test_retrieval_switch.py` to assert that with `retrieval.enabled=false` the `registry` remains zero after `_evaluate()`.
* **Run:** `pytest -q`
* **Context:** Baselines/longctx were recording retrieval requests.

---

## T4 — Wire **gate telemetry** into replay/teach writes
* **Goal:** Make gate stats meaningful: attempts/inserted/aggregated/etc should increment during replay/teach when memory is active.
* **Files to touch:**
  - `hippo_mem/eval/harness.py` — in `_run_replay(...)`:
    - For episodic: instantiate `hippo_mem.episodic.gating.WriteGate` from config (use defaults if missing) and call its batch apply function rather than unconditional `store.write(...)`. Episodic gating utilities already update `gate_registry`.
    - For relational: if `modules` contains relational + a `gate` is configured, route tuples via `KnowledgeGraph.ingest(...)` (which updates `gate_registry`).
    - For spatial: when simulating observations in replay, route via `SpatialGate.decide(...)` and then `PlaceGraph.observe(...)`/`connect(...)` accordingly.
  - Ensure `mode=teach` path uses the same gating write path when `gating_enabled=true`.
* **Acceptance tests:**
  - Add `tests/test_gate_increments_on_replay.py` that runs a short replay (`replay.cycles=1`) and asserts non‑zero `gate_registry` counters for the memories present.
* **Run:** `pytest -q`
* **Context:** All gate counters were zero because writes bypassed gate logic.

---

## T5 — Enforce **store isolation** and zero stores in baselines
* **Goal:** Baselines must not read/write stores; ablations like `longctx_no_retrieval` must also avoid retrieval/store updates.
* **Files to touch:**
  - `hippo_mem/eval/harness.py` — after `_init_modules(...)`, if not a memory preset, **empty** `modules = {}` to hard‑enforce no memory in baselines.
  - `scripts/validate_store.py` — add a check: when called with `--preset baselines/*` the derived store path must not exist; fail otherwise.
  - `scripts/store_paths.py` — expose helper `is_memory_preset(preset: str) -> bool` (used by harness/report/validators).
* **Acceptance tests:**
  - New `tests/test_store_isolation.py`: run a baseline and assert that no `runs/<id>/stores/*` directory was created and `metrics["store"]["size"] == 0`.
* **Run:** `pytest -q`
* **Context:** We observed non‑zero `store.size` in baseline runs.

---

## T6 — Compute telemetry parity (no placeholders)
* **Goal:** Every preset reports consistent compute metrics (`input_tokens`, `generated_tokens`, `total_tokens`, `time_ms_per_100`, `rss_mb`, `latency_ms_mean`). No hardcoded fallbacks.
* **Files to touch:**
  - `hippo_mem/eval/harness.py` — ensure compute block is always included in `metrics["metrics"]["compute"]` (teach/test/replay); remove any placeholder constants; derive from measurements only.
  - `hippo_mem/eval/bench.py` — mirror compute telemetry fields and remove fake timing; derive latency from the measured loop.
* **Acceptance tests:**
  - Extend `tests/test_report.py` smoke writer to assert presence of compute keys for all presets.
* **Run:** `pytest -q`
* **Context:** Reports showed placeholder‑like timings on some presets.

---

## T7 — Make datasets discriminative and curb saturation
* **Goal:** Avoid EM≈1.0 for memory runs and EM≈0.0 for baselines; target baselines at 15–40% EM with room for uplift.
* **Files to touch:**
  - `hippo_mem/eval/datasets.py` — add a `profile` parameter (`easy|default|hard`) for each generator:
    - Episodic: tune number/placement of distractors; include entity swaps; avoid repeating templates.
    - Semantic: increase schema confusion (near‑miss relations) for `hard`.
    - Spatial: introduce dead‑ends and alternative paths; measure `steps_to_goal` variance.
  - Thread `dataset_profile` through configs & CLI (`scripts/eval_model.py`).
* **Acceptance tests:**
  - New `tests/test_dataset_profiles.py` verifying output distributions and that `hard` reduces baseline EM on tiny mock models (deterministic heuristics).
* **Run:** `pytest -q`
* **Context:** Several suites saturated; uplift claims are not credible otherwise.

---

## T8 — Reporting guardrails & anomaly flags
* **Goal:** Make `scripts/report.py` surface issues immediately.
* **Files to touch:**
  - `scripts/report.py`:
    - When a suite/preset violates invariants (see below), add a ⚠️ row‑level note and an aggregate “Warnings” section per suite.
    - Invariants to check:
      1. Baselines: `retrieval.*.requests == 0` and `store.size == 0`.
      2. No‑retrieval ablation: `retrieval.*.requests == 0`.
      3. If `pre_em_norm ≥ 0.98` and baseline `pre_em_norm < 0.20` → mark as **SaturationSuspect**.
      4. Gate enabled but all gate counters zero → **GateNoOp**.
* **Acceptance tests:**
  - Extend `tests/test_report.py` to verify the new warnings are emitted when feeding crafted metrics.
* **Run:** `pytest -q`
* **Context:** We had to spot anomalies manually; make them explicit in reports.

---

## T9 — Documentation updates (Protocol & READMEs)
* **Goal:** Keep operators on the happy path.
* **Files to touch:**
  - `EVAL_PROTOCOL.md` — Call out the invariants from **T8** and the expected zero‑telemetry for baselines; clarify teach vs test; add a troubleshooting box for baseline retrieval > 0.
  - `EVAL_PLAN.md` — reflect dataset profiles and what each evaluates.
  - `README.md` — short “How to read reports” section linking to invariants.
* **Acceptance tests:** N/A (docs)
* **Run:** CI markdown link check.
* **Context:** Reduce operator confusion seen in this run.

---

## T10 — Store accounting consistency
* **Goal:** Ensure `metrics["store"]["per_memory"]` aligns with actual persisted items.
* **Files to touch:**
  - `hippo_mem/eval/harness.py` — `_store_sizes(...)`:
    - Episodic: count `SELECT COUNT(*) FROM traces` (already present).
    - Relational: count **edges** consistently; also include a `nodes_added` field in diagnostics.
    - Spatial: expose `PlaceGraph.log_status()["writes"]` and ensure it matches `spatial.jsonl`.
  - `scripts/validate_store.py` — compare JSONL line counts with `metrics["store"]` and raise on mismatch.
* **Acceptance tests:**
  - New `tests/test_store_accounting.py` simulating a tiny replay and asserting consistency between on‑disk JSONL sizes and metrics.
* **Run:** `pytest -q`
* **Context:** Counts looked synthetic and mismatched across memories.

---

## T11 — Strict telemetry mode in CI
* **Goal:** Fail fast when invariants are violated.
* **Files to touch:**
  - `hippo_mem/common/telemetry.py` — keep `_STRICT=False` by default.
  - In `scripts/smoke_eval.sh` and CI workflow, export `STRICT_TELEMETRY=1` or pass `--strict-telemetry`; harness should call `set_strict_telemetry(True)` when enabled.
  - `hippo_mem/eval/harness.py` — read `cfg.strict_telemetry`/env and call `set_strict_telemetry(...)` at entry.
* **Acceptance tests:** Extend `tests/test_telemetry_sanity.py` with a case that forces a violation and expects a `ValueError` when strict mode is on.
* **Run:** `pytest -q`
* **Context:** Hidden anomalies made it to reports.

---

## T12 — Baseline/harness parity: prefer a single path
* **Goal:** Avoid drift between `eval_bench` and `eval_model` outputs.
* **Files to touch:**
  - Make `scripts/run_baselines_bench.py` opt‑in only for CI smoke. Document in `EVAL_PROTOCOL.md` that the canonical path for real runs is `scripts/eval_model.py`.
  - Ensure `scripts/report.py` gracefully handles missing `retrieval`/`gates` keys (bench output) *and* full harness output.
* **Acceptance tests:** Expand `tests/test_report.py` matrix to include both bench‑style and harness‑style metrics files.
* **Run:** `pytest -q`
* **Context:** Mixed outputs can confuse operators and reporting.

---

## T13 — Quick regression seed sweep
* **Goal:** Establish non‑flaky baselines.
* **Files to touch:**
  - Add `scripts/smoke_n_seeds.sh` that runs `SIZES=(50)` across `SEEDS=(1337 2025 4242)` for `baselines/core` and one memory preset with strict telemetry on; prints a compact table.
* **Acceptance tests:** N/A (script)
* **Run:** `bash scripts/smoke_n_seeds.sh`
* **Context:** Early signal on stability before full grids.

---

### Notes for Codex
- Keep diffs tight and add unit tests alongside code changes.
- Follow `CODING_STANDARDS.md` and keep function docstrings up to date.
- Prefer **feature flags** and **config‑gated** behavior over hardcoded branches.
- Update any failing tests and extend coverage especially around harness entry points.

---

### Done definition (for this pack)
- Baselines show **zero** retrieval/store and no gating.
- Memory presets: retrieval counters reflect actual calls; gate counters increment during replay/teach.
- Reports surface anomalies explicitly and compute telemetry is present for all runs.
- Dataset profiles avoid trivial saturation while maintaining determinism.
