# Implementation Readiness Review — hippo_mem vs. review/hippo-pipeline-review_20250831_50_1337.md

_Generated 2025-08-31 15:44 UTC_

## Scope & Method
I inspected the repository in the ZIP with a focus on **implementation code and configs**. As requested, I **did not** inspect `data/`, `runs/`, or `reports/`. Primary targets: `hippo_mem/**`, `configs/eval/**`, `scripts/**`, `tests/**`, the review (`review/hippo-pipeline-review_20250831_50_1337.md`) and its task pack (`tasks/codex_tasks_hippo_run_20250831_50_1337.md`).

## Executive Verdict
**Go (with 1 small blocker):** The codebase reflects the review’s mitigation items in nearly all critical areas. Baselines are isolated, telemetry is reset per run and persisted to metrics, and gates are wired for **episodic** and **relational** memory. One remaining blocker before re‑running `EVAL_PROTOCOL.md`: **spatial gate counters are not wired into telemetry** (no `gate_registry` updates in the spatial path). Everything else looks ready.

---

## Findings — Mapped to the Review’s Themes

### 1) Telemetry plumbing
- **Reset on every run / suite:** Present in both harnesses.
  - `hippo_mem/eval/bench.py`: calls `registry.reset()` and `gate_registry.reset()` before evaluations.
  - `hippo_mem/eval/harness.py`: same resets in the evaluation flow.
- **Strict mode:** `EVAL_PROTOCOL.md` exposes `--strict-telemetry`; `scripts/eval_cli.py` passes it; `hippo_mem/common/telemetry.set_strict_telemetry` is called from `harness.py`. `validate_retrieval_snapshot` enforces invariants via `record_stats(...)`.
- **Persisted to metrics:** `harness.py` writes `retrieval` and `gates` snapshots plus compute (`time_ms_per_100`, `rss_mb`, `latency_ms_mean`). `reporting/report.py` aggregates and renders these.

**Assessment:** ✅ Satisfies the review’s telemetry requirements.

### 2) Store isolation & baseline hygiene
- **No stores for baselines:** `harness.py` empties `modules` when preset is not a memory preset; `utils/stores.is_memory_preset(...)` and `assert_store_exists(...)` enforce layout and prevent baseline pollution.  
- **Loading/saving only when requested:** `harness.py` loads stores only for `mode in ("test","replay")` with `store_dir` + `session_id`, and saves on replay with `persist=true`.
- **Tests:** `tests/test_store_isolation.py`, `tests/test_baselines_have_no_memory.py`, `tests/test_store_paths.py`, `scripts/validate_store.py` back this up.

**Assessment:** ✅ Matches the review’s isolation fixes.

### 3) Gate instrumentation
- **Episodic (HEI‑NW):** `episodic/gating.py` pulls `stats = gate_registry.get("episodic")` and increments `attempts` and `inserted` per decision; provenance logging is supported (`common/provenance.py`).
- **Relational (SGC‑RSS):** `relational/kg.py` increments `attempts`, `inserted`, `aggregated`, `routed_to_episodic` via `gate_registry.get("relational")` during `ingest(...)`.
- **Spatial (SMPD):** **Missing**: no `gate_registry` increments in `spatial/*` (no references to `gate_registry` in the spatial path).

**Assessment:** ⚠️ **Blocker** — add spatial gate counters before the next evaluation run.

### 4) Ablations & toggles
- Presets align with intent:
  - `configs/eval/baselines/*`: `retrieval.enabled` and `long_context.enabled` flags appropriately set for `core`, `rag`, and `longctx`.
  - `configs/eval/memory/*`: enable memory and gating; `replay.cycles=1` by default.
- `harness.py` threads `retrieval_enabled` and `long_context_enabled` into the scoring path.
- Gate ON/OFF ablations are recognized in reporting (`collect_gate_ablation`).

**Assessment:** ✅ Ready.

### 5) Dataset profiles (saturation guardrails)
- Synthetic generators implement profiles (`eval/datasets.py`), and tests assert that **hard** profiles add distractors/contradictions to avoid trivial saturation (`tests/test_dataset_profiles.py`). Harness accepts `dataset_profile` and threads it into `_dataset_path(...)`.
  
**Assessment:** ✅ Matches review intent; adequate for functional validation.

### 6) Reporting & diagnostics
- `reporting/report.py` compiles per‑suite summaries and aggregates telemetry (`retrieval` & `gates`) and compute columns. Gate ablation summaries are supported.
- CLI scripts (`scripts/smoke_n50.sh`, `scripts/eval_cli.py`, `scripts/eval_model.py`) align with the protocol, including RUN_ID/session store layout and `--strict-telemetry`.

**Assessment:** ✅ Consistent with the protocol and review.

---

## Minor Observations (non‑blocking)
- **Provenance files:** Gate provenance is implemented; if you want first‑class surfacing in reports, add a tiny parser to `reporting/` later (optional).
- **Docs:** `EVAL_PROTOCOL.md` already points to `scripts/eval_model.py` as canonical. Keep a short note that baselines must **never** set `persist=true`.

---

## Go / No‑Go Checklist
- [x] Baselines: no memory modules, no retrieval, no stores, no gates.
- [x] Memory presets: retrieval telemetry increments and is persisted.
- [x] Episodic/Relational: gate counters increment and appear in `metrics.json`.
- [ ] Spatial: **gate counters wired** (attempts/accepts) → _**do this before running**_.
- [x] `--strict-telemetry` plumbed and enabled in protocol.
- [x] Store layout validated via `validate_store.py` between teach/replay/test.
- [x] Dataset profiles selectable and non‑trivial (easy vs hard).

**Conclusion:** Proceed after wiring **spatial gate telemetry** (1 small code change + 1 test). Everything else is in shape to re‑run `EVAL_PROTOCOL.md` and get meaningful signals.

