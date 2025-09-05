# Readiness Review — Tasks A–G & EVAL_PROTOCOL.md (2025‑08‑28)
_Scope: verify Tasks A–G from `review/review-2025-08-28.md` are implemented with sufficient quality, and double‑check `EVAL_PROTOCOL.md` for completeness and compatibility with the current code._


## Executive summary


All seven tasks (A–G) are implemented and **functionally complete**. The evaluation harness, presets, and reporting pipeline line up with `EVAL_PROTOCOL.md`. I found **two documentation inconsistencies** and **one optional improvement** that would make gate‑threshold sweeps on episodic memory actually influence the harness. None of these are blockers for starting the human tasks.

**Green lights to proceed**, with the small fixes below (docs) recommended before long runs.


## Task-by-task verification

### A — Fix replay cycles key

`hippo_mem/eval/harness.py` normalizes `replay.cycles` → `replay_cycles` and uses it in both metadata and replay loops.

- Evidence: look for `nested = cfg.get("replay")` and `cfg.replay_cycles = int(nested.get("cycles", 0))`, and replay loops `for _ in range(int(cfg.replay_cycles)):`.
### B — Implement teach→test protocol

`mode=teach` writes audit rows and (when `persist=true` and `store_dir`/`session_id` are set) saves episodic/relational/spatial stores, which are then **loaded** in `mode=test` and `mode=replay`.

- Evidence: `if cfg.mode == "teach": ... if cfg.persist and cfg.store_dir and cfg.session_id: ... save(...)`; and prior to evaluation in `mode in ("test", "replay")` the stores are conditionally `load(...)`ed.
### C — Add ablation flags and gate sweeps

Ablation toggles from `+ablate=...` flatten correctly and drive gate enables per memory (`episodic.gate.enabled`, `relational.gate.enabled`, `spatial.gate.enabled`). Optional gate sweeps are wired via CLI overrides.

- Evidence: in `harness.py`, flattened ablates update `mem_cfg.episodic.gate.enabled`, `mem_cfg.relational.gate.enabled`, `mem_cfg.spatial.gate.enabled` under `open_dict`.
### D — Strengthen semantic generator

`generate_semantic(...)` supports 2–3 hop chains, optional contradictions, and a `require_memory` mode so tasks aren’t trivially answerable from the prompt alone.

- Evidence: see parameters `hop_depth`, `inject_contradictions`, `require_memory` in `hippo_mem/eval/datasets.py`.
### E — Spatial KPIs and parsing

Spatial scoring parses move sequences and reports `steps_pred`, `steps_opt`, `suboptimality`, and `success`; invalid moves are detected.

- Evidence: `spatial_kpis(...)` and helpers in `hippo_mem/eval/score.py`.
### F — Baseline runs for episodic variants

Baseline matrix covers core/span_short/rag/longctx across suites including `episodic_multi`, `episodic_cross`, `episodic_capacity`.

- Evidence: `hippo_mem/eval/baselines.py` (`SUITES` includes episodic variants); top‑level `Makefile` has `eval-baselines` → `scripts/run_baselines_bench.py --date $(DATE)`.
- Note: the review doc mentions `scripts/Makefile`, but the actual target lives in the **root `Makefile`** (this is correct; doc should be updated).
### G — Report deltas and CI guardrails

`scripts/report.py` derives `delta_*` metrics from matching `pre_*`/`post_*` pairs, renders tables/plots, and can emit a smoke sample. CI guardrails test refusal rates and retrieval counts.

- Evidence: `scripts/report.py` computes deltas and supports `--smoke`; see `tests/test_ci_guardrails.py` for automated checks.


## EVAL_PROTOCOL.md — completeness & compatibility


`EVAL_PROTOCOL.md` is **in sync** with the codebase:
- **Prelude** defines `DATE`, `RUNS`, `REPORTS`, `STORES`, `SESS`, and `MODEL` (all used later).
- **Datasets**: `make datasets` calls `scripts/build_datasets.py` across all suites/sizes/seeds and runs a dataset audit.
- **Baselines**: `python scripts/run_baselines_bench.py --date "$DATE"` matches `hippo_mem/eval/baselines.py` defaults and the `Makefile` target `eval-baselines`.
- **Memory grids (teach → replay → test)**: three calls per suite use `mode=teach`/`replay`/`test`, `persist=true`, `store_dir`, `session_id`, and `replay.cycles=3`. All these flags are recognised by `scripts/eval_model.py` → `hippo_mem.eval.harness` and are covered by `configs/eval/default.yaml` and memory presets in `configs/eval/memory/*.yaml`.
- **Ablations per memory**: overrides like `memory.episodic.gate.enabled=false` are applied in the harness before module init.
- **Reporting**: `scripts/report.py --date "$DATE" --plots --smoke` is valid; it finds the latest date if omitted, but explicit `--date` is fine for reproducibility.
- **Optional gate sweeps**: example loops set small ranges and are compatible with CLI overrides for relational (`relational.gate.threshold`) and spatial (`spatial.gate.block_threshold`).

**Sanity tip (optional but recommended):** run `make smoke` once before the full protocol to validate the path layout and metrics plumbing on tiny samples.


## Issues found (non‑blocking)

- **Review doc path mismatch.** Task F in `review/review-2025-08-28.md` refers to `scripts/Makefile`. The working target is in the root **`Makefile`**. This is only a documentation mismatch.
- **Config path hints in EVAL_PROTOCOL gate‑sweep comments.** The comments say `configs/memory/relational.yaml` and `configs/memory/spatial.yaml`, but the actual presets live under `configs/eval/memory/sgc_rss.yaml` and `configs/eval/memory/smpd.yaml`. This could confuse operators skimming for defaults.


## Optional improvement (makes episodic gate sweeps meaningful)


At present, episodic replay in the harness writes traces **unconditionally** (mock key, unconditional `store.write(...)`). As a result, sweeping `memory.episodic.gate.tau` will not affect outputs. If you want episodic gate‑threshold sweeps to change results in this harness, add a minimal WriteGate call around the replay write:

- Instantiate `WriteGate(tau=cfg.memory.episodic.gate.tau)` if present, else default.
- Compute a simple salience (e.g., reuse `prob`≈0.5; or use a deterministic function of the answer) and **skip** writes when below τ unless `pin=True` is requested.
- This keeps the harness cheap while making episodic gate sweeps observable in metrics.

This is **not required** to run the protocol or validate SGC‑RSS / SMPD, but makes the HEI‑NW sweep section more informative.


## Executable Codex tasks (only if you want the fixes)

**Fix doc path in review file**

Update the reference to the Makefile path in `review/review-2025-08-28.md`.

**Edit:** `review/review-2025-08-28.md`
**Change:** Replace `scripts/Makefile` with `Makefile` wherever mentioned.
**Accept:** The document no longer points to `scripts/Makefile`. 
**Correct config path hints in EVAL_PROTOCOL comments**

Fix the preset path hints in the gate-sweep comment block.

**Edit:** `EVAL_PROTOCOL.md`
**Change:** Replace `configs/memory/relational.yaml` → `configs/eval/memory/sgc_rss.yaml` and `configs/memory/spatial.yaml` → `configs/eval/memory/smpd.yaml`.
**Accept:** The comments reference the actual files that hold the defaults. 
**(Optional) Wire episodic WriteGate into replay writes**

Make episodic gate threshold sweeps affect the harness.

**Edit:** `hippo_mem/eval/harness.py`
**Where:** In `_run_replay(...)` before `store.write(...)`
**Change (sketch):**
```python
from hippo_mem.episodic.gating import WriteGate

gate_cfg = (cfg.get("memory") or {}).get("episodic", {}).get("gate", {})
tau = float(gate_cfg.get("tau", 0.5))  # keep default
wgate = WriteGate(tau=tau)

# derive a deterministic salience in [0,1] for the item (cheap mock)
sal = (hash(task.answer) % 100) / 100.0
if wgate.action(prob=sal, query=key, keys=store.index.keys()) == "insert":
    store.write(key, task.answer)
```
**Accept:** With `memory.episodic.gate.tau` raised above typical `sal`, fewer items are written during replay; `metrics.json["store"]["per_memory"]["episodic"]` decreases accordingly.


## Go/no‑go checklist


- Datasets build: **OK** (`make datasets`).
- Baselines matrix: **OK** (`make eval-baselines DATE=$DATE` or `python scripts/run_baselines_bench.py --date "$DATE"`).
- Memory grids (teach → replay → test): **OK** (`scripts/eval_model.py` flags match the harness).
- Reporting: **OK** (`python scripts/report.py --date "$DATE" --plots [--smoke]`).

**Verdict:** proceed with the human tasks as per `EVAL_PROTOCOL.md`.
