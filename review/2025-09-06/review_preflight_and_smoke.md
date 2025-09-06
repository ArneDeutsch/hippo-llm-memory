# Smoke Tests & Preflight Check — Review & Fix Plan

**Repo:** `hippo-llm-memory` • **Date:** 2025-09-06
**Scope:** Make CI smoke tests compatible with the *memory-first* evaluation pipeline and adapt the preflight checks accordingly.

## TL;DR (what to change)

1. **Use real preset names in CI & scripts.** Replace `memory/hei_nw_cross` → `memory/hei_nw` and `memory/sgc_rss_mem` → `memory/sgc_rss` (or add alias YAMLs).
2. **Preflight: resolve store files by kind, not exact algo name.** Normalize aliases (strip `_mem`, `_cross`) and map to `episodic|kg|spatial`; stop expecting `hei_nw_cross.jsonl`.
3. **Harden the dry-run gate check.** Count an attempt even in null-input paths (or relax the check when memories are configured but inputs are empty).
4. **(Optional)** Drop or scope the final `failed_preflight.json` grep in `ci_smoke_eval.sh`; preflight already throws on failure.

---

## What I inspected (repo paths)

* Smoke script: `scripts/ci_smoke_eval.sh`
* Harness & preflight: `hippo_eval/eval/harness.py`
* Eval wrapper: `scripts/eval_model.py`
* Store helpers: `hippo_mem/utils/stores.py`
* Configs: `configs/eval/default.yaml`, `configs/eval/memory/{hei_nw,sgc_rss,smpd}.yaml`
* Review/plan/tasks:

  * `review/2025-09-05/review_evaluation_strategy_run_20250904.md`
  * `review/2025-09-05/LEFTOVER_WORK_2025-09-05.md`
  * `plans/2025-09-05/pipeline_rework_stepwise_plan.md`
  * `tasks/2025-09-05/{TASKS_ALL_IN_ONE.md,CODEX_TASKS_to_Finish_Eval_Rework_2025-09-05.md}`

---

## Why the current smoke fails (root causes)

### 1) Preset name drift

CI passes `PRESET=memory/hei_nw_cross` (and `memory/sgc_rss_mem` elsewhere), but your in-tree presets are `configs/eval/memory/{hei_nw,sgc_rss,smpd}.yaml`. With a non-existent YAML, the preset merge never happens. That increases the chance the dry-run preflight sees partial/missing memory config and records `gate.attempts == 0`.

### 2) Brittle store filename mapping in preflight

Preflight maps algos → filenames like `hei_nw→episodic.jsonl`, `sgc_rss→kg.jsonl`, else `f"{algo}.jsonl"`. For `hei_nw_cross`, it expects `hei_nw_cross.jsonl`, but the episodic store persists to `episodic.jsonl`. Later test/replay checks will therefore flag an “empty/missing store” even when the store exists.

### 3) Over-strict gate-attempt check in dry-run

The dry-run does a tiny `teach` pass (`n=1`, `persist=false`) and then asserts the **sum of gate attempts** (episodic/relational/spatial) is non-zero. Gate implementations may increment `null_input` without incrementing `attempts` on early returns, producing false negatives in edge cases.

---

## Recommended changes (minimal and ordered)

### A) Align preset names (fastest win)

**Option A1 — Change CI & docs to match shipped presets**

* Episodic: `PRESET=memory/hei_nw`, `SUITE=episodic_cross_mem`
* Relational: `PRESET=memory/sgc_rss`, `SUITE=semantic_mem`
* Spatial: `PRESET=memory/smpd`, `SUITE=spatial_multi`

**Option A2 — Keep old names by adding alias YAMLs**
Create `configs/eval/memory/hei_nw_cross.yaml` and `configs/eval/memory/sgc_rss_mem.yaml` that simply replicate the canonical presets (point them to episodic/relational config; enable retrieval/gating).

### B) Fix preflight store resolution

In `hippo_eval/eval/harness.py` preflight:

* Normalize algo: `base = re.sub(r'(_mem|_cross)$', '', algo)`
* Map kind: `kind = {'sgc_rss':'kg','smpd':'spatial'}.get(base, 'episodic')`
* Expect `<kind>.jsonl` instead of `f'{algo}.jsonl'`.

This makes aliases like `hei_nw_cross` and `sgc_rss_mem` resolve to the correct `episodic.jsonl`/`kg.jsonl`.

### C) Relax/harden the gate-attempt rule

Pick one (both are fine):

1. **Gate implementations:** increment `attempts` before null-input early returns so any pass through the gate counts.
2. **Preflight:** if memories are configured and gate objects exist but `attempts==0`, warn instead of failing hard (only for the dry-run).

### D) (Optional) Remove/scoped grep in the smoke script

The script already fails on the preflight exception (with `set -e`). The final `find ... failed_preflight.json` check is redundant and can catch transient files. Either remove it or scope it to the final `test` outputs only.

---

## Concrete patch sketches

**CI env var (episodic example):**

```diff
# .github/workflows/ci.yaml
-  PRESET: memory/hei_nw_cross
+  PRESET: memory/hei_nw
```

**Preflight store logic:**

```diff
# hippo_eval/eval/harness.py
- store_file = meta_path.parent / store_files.get(algo, f"{algo}.jsonl")
+ base = re.sub(r"(_mem|_cross)$", "", algo)
+ kind = {"sgc_rss": "kg", "smpd": "spatial"}.get(base, "episodic")
+ store_file = meta_path.parent / f"{kind}.jsonl"
```

**Gate null-input accounting (example):**

```diff
# in each Gate.decide(...)
- stats.null_input += 1
+ stats.attempts += 1
+ stats.null_input += 1
```

---

## Validation checklist

* ✅ CI smoke runs use canonical (or aliased) presets.
* ✅ Dry-run preflight no longer fails spuriously; no `failed_preflight.json`.
* ✅ Test/replay preflight sees non-empty `episodic.jsonl`/`kg.jsonl`/`spatial.jsonl`.
* ✅ `STRICT_TELEMETRY=1` still passes.
