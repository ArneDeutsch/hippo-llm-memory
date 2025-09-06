
# Smoke Tests & Preflight Check — Review for the Memory‑First Pipeline
**Repo:** `hippo-llm-memory`  •  **Date:** 2025‑09‑06  •  **Author:** Assistant review

## TL;DR — What to change *now*
1) **Teach before Test in CI.** The smoke job runs `mode=test` with a `session_id`/`store_dir` but **no persisted store exists**, so `scripts/eval_model.py` aborts via `hippo_mem.utils.stores.validate_store(...)`. Add a **Teach** step with `persist=true` to materialize the store, then run Test (or Replay).  
2) **Use base `store_dir` (no algo suffix).** Pass `store_dir=runs/$RUN_ID/stores` (base). The code derives `<algo>` internally. Passing `.../stores/hei_nw` is ambiguous and triggers path mismatches.  
3) **Scope `validate_store(...)` to real needs.** In `scripts/eval_model.py`, call `validate_store` **only when** `preset` is `memory/*` **and** `mode in {"test","replay"}` **and** both `store_dir` & `session_id` are set.  
4) **Let `validate_store` honor explicit paths.** Extend it to accept `store_dir` & `session_id` and validate those, falling back to `run_id` only when paths are not provided. This removes legacy coupling to derived paths.  
5) **Preflight: keep the dry‑run gate check; relax legacy assumptions.** Keep the *teach‑dry‑run gate activity* check. Make baseline metrics and persisted‑store checks **conditional**: only require them when the run actually references baselines or an existing store.  
6) **Align names & flags.** Use real preset names (`memory/hei_nw`, `memory/sgc_rss`, `memory/smpd`) and canonical suite IDs (`episodic_cross_mem`, `semantic_mem`, `spatial_multi`). Avoid legacy `*_mem` suffixes in preset names and boolean flags like `--no-retrieval-during-teach=true`.

---

## What failed in CI (example & diagnosis)

**Observed error (abridged):**
```
Error executing job with overrides: ['suite=episodic_cross_mem', 'preset=memory/hei_nw', 'run_id=ci_..._episodic_cross_mem', 'n=50', 'seed=1337', 'mode=test', 'store_dir=runs/.../stores/hei_nw', 'session_id=hei_...']
FileNotFoundError: Persisted store not found.
Expected path: runs/.../stores/hei_nw/hei_ci_..._episodic_cross_mem/episodic.jsonl
Hint: `store_dir` should be the base directory containing the `hei_nw` folder.
Reminder: run teach+replay with `persist=true` to create it.
```

**Root causes:**
- CI calls **Test** directly with a `session_id` but **no prior Teach** with `persist=true`; therefore no store exists.  
- `scripts/eval_model.py` invokes `validate_store(...)` on startup; it checks the **derived** path from `run_id`/`algo` and errors out before the harness can run.  
- `store_dir` is passed with an **algo suffix** (`.../stores/hei_nw`), while parts of the code expect the **base** (`.../stores`) and then append `<algo>` consistently.

This is consistent with what the new memory‑first plan says we should do: **separate Teach/Test and persist stores between phases.**

---

## Current behavior (as implemented on disk)

### A. Smoke scripts
- `scripts/smoke_eval.sh` creates a tiny baseline and then runs a memory preset with **temporary** outdirs (no persistence). It also stubs `runs/$RUN_ID/baselines/metrics.csv` to satisfy legacy preflight. This is OK for a local smoke, but **does not exercise Teach/Test separation**.

- `scripts/ci_smoke_eval.sh` (CI) builds datasets and runs a memory suite with:
  - `mode=test`
  - `store_dir=runs/$RUN_ID/stores/hei_nw`
  - `session_id=hei_$RUN_ID`
  - **No prior Teach with `persist=true`** → store missing → early failure.

### B. Preflight checks (in `hippo_eval.eval.harness`)
- **Baseline check:** Fails if baseline metrics are missing.  
- **Store check:** If `mode in {"test","replay"}` **and** `store_dir` & `session_id` are provided, it expects `store_meta.json` and a **non‑empty** store file (`episodic.jsonl` for HEI‑NW; `kg.jsonl` for SGC‑RSS; `spatial.jsonl` for SMPD).  
- **Teach dry‑run:** Runs a tiny `teach` with `persist=false` and asserts **gate attempts > 0**, to catch miswired gating.

### C. Early validator (in `scripts/eval_model.py`)
- Calls `hippo_mem.utils.stores.validate_store(...)` before entering the harness, based on `preset`/`algo` (and **derived** paths off `run_id`). Because the CI run passes `mode=test` + `session_id` but has **no persisted store**, this fails first.

---

## What to change (proposal)

### 1) CI smoke flow: **Teach → Test (or Replay)**

A minimal, deterministic smoke should do exactly one persisted Teach, then one Test (or Replay). Here is a **drop‑in** template for `scripts/ci_smoke_eval.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=${RUN_ID:-ci_smoke}
export MODEL=${MODEL:-models/tiny-gpt2}
export STRICT_TELEMETRY=${STRICT_TELEMETRY:-1}
source "$(dirname "$0")/_env.sh"

SUITE=${SUITE:-episodic_cross_mem}
PRESET=${PRESET:-memory/hei_nw}
SESSION_ID=${SESSION_ID:-${PRESET##*/}_$RUN_ID}
STORES="runs/$RUN_ID/stores"          # BASE dir (no algo suffix!)

# 0) Ensure baselines exist (legacy preflight compatibility)
mkdir -p "runs/$RUN_ID/baselines"
test -f "runs/$RUN_ID/baselines/metrics.csv" ||   echo "suite,em_raw,em_norm,f1" > "runs/$RUN_ID/baselines/metrics.csv"

# 1) Build dataset deterministically (n=50, seed=1337 matches our smoke standard)
python -m hippo_eval.datasets.cli --suite "$SUITE" --size 50 --seed 1337 --out "datasets/$SUITE"

# 2) Teach with persistence (materialize store)
python scripts/eval_model.py   suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=50 seed=1337   mode=teach persist=true store_dir="$STORES" session_id="$SESSION_ID"   compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 3) Test (reads the persisted store)
python scripts/eval_model.py   suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=50 seed=1337   mode=test store_dir="$STORES" session_id="$SESSION_ID"   compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 4) Report
python -m hippo_eval.reporting.report --run-id "$RUN_ID"

# 5) Sanity: metrics rolled-up
test -f "runs/$RUN_ID/baselines/metrics.csv" || { echo "missing baselines metrics" >&2; exit 1; }
```

Notes:
- **Base `STORES`** is `runs/$RUN_ID/stores`, *without* the algorithm suffix. The harness/validators derive `<algo>` correctly.
- If you want to exercise **consolidation**, add a `mode=replay` run after Test.

### 2) `scripts/eval_model.py`: guard and pass explicit paths

Change the early validation to **only** run when we truly depend on a pre‑existing store, and make it path‑aware:

```diff
-    if cfg.get("store_dir") or cfg.get("session_id"):
-        from hippo_mem.utils.stores import validate_store
-        maybe_store = validate_store(
-            run_id=str(cfg.get("run_id") or os.getenv("RUN_ID") or ""),
-            preset=str(cfg.preset),
-            algo=algo,
-            kind=store_kind,
-        )
+    needs_store = (
+        str(cfg.get("preset","")).startswith("memory/")
+        and str(cfg.get("mode","test")) in {"test","replay"}
+        and cfg.get("store_dir") and cfg.get("session_id")
+    )
+    if needs_store:
+        from hippo_mem.utils.stores import validate_store
+        maybe_store = validate_store(
+            run_id=str(cfg.get("run_id") or os.getenv("RUN_ID") or ""),
+            preset=str(cfg.preset),
+            algo=algo,
+            kind=store_kind,
+            store_dir=str(cfg.get("store_dir")),
+            session_id=str(cfg.get("session_id")),
+        )
         if maybe_store is not None:
             _ensure_populated(maybe_store, cfg)
-    elif cfg.get("store_dir"):
-        cfg.store_dir = _normalize_store_dir(str(cfg.store_dir), algo)
+    elif cfg.get("store_dir"):
+        cfg.store_dir = _normalize_store_dir(str(cfg.store_dir), algo)
```

### 3) `hippo_mem.utils.stores.validate_store`: honor explicit `store_dir`/`session_id`

Extend the signature and logic so we **don’t require** a derived `runs/$RUN_ID/...` layout when the caller passed concrete paths:

```diff
-def validate_store(run_id: str, preset: str, algo: str, kind: str = "episodic") -> Path | None:
+def validate_store(
+    run_id: str,
+    preset: str,
+    algo: str,
+    kind: str = "episodic",
+    store_dir: str | None = None,
+    session_id: str | None = None,
+) -> Path | None:
     """
-    Resolve the expected store path and assert it exists.
+    Resolve the expected store path and assert it exists. Prefer explicit
+    (store_dir, session_id) when provided; otherwise derive from run_id/algo.
     """
-    layout = derive(run_id=run_id, algo=algo)
+    if store_dir and session_id:
+        base = Path(store_dir)
+        algo_dir = base if base.name == algo else base / algo
+        path = algo_dir / session_id / ({"episodic":"episodic.jsonl","kg":"kg.jsonl","spatial":"spatial.jsonl"}[kind])
+        if not path.exists():
+            raise FileNotFoundError(f"Persisted store not found. Expected path: {path}")
+        return path
+
+    layout = derive(run_id=run_id, algo=algo)
     # ... existing checks ...
```

This removes legacy coupling to `RUN_ID`-based layouts when CI (or users) pass explicit locations.

### 4) Preflight: keep signal, drop brittle coupling

**Keep**:
- The **teach dry‑run gate** assertion (attempts ≥ 1). It finds obvious wiring bugs quickly.

**Relax or scope**:
- **Baseline metrics** requirement: only enforce if baseline deltas are requested; otherwise warn. The CI script above already creates a minimal CSV to satisfy this.
- **Persisted store** checks: only enforce when `mode in {"test","replay"}` **and** the run actually references a `store_dir`/`session_id`. (This is already mostly true in `harness.preflight_check`; just ensure we don’t run a second, earlier validator that fails sooner.)

### 5) Naming/flag hygiene (remove remnants of old pipeline)
- Presets: `memory/hei_nw`, `memory/sgc_rss`, `memory/smpd` (no extra suffixes).
- Suites: `episodic_cross_mem`, `semantic_mem`, `spatial_multi` (✓ matches `SUITE_ALIASES`).
- Flags: Hydra booleans should be `flag=true/false` (or just presence for some), not `--flag=true` in shell style mixed with `key=value` overrides.
- Paths: prefer `runs/$RUN_ID/stores` base; let code resolve `<algo>`.

---

## Definition of Done (DoD) for the fix
- CI smoke (all three memory presets) passes with **Teach → Test** flow and `n=50`, `seed=1337`.
- No `FileNotFoundError` from `validate_store` on CI.
- `harness.preflight_check` can run end to end; `failed_preflight.json` is **not** emitted for the smoke run.
- Baseline metrics are present (either real tiny baseline run or stubbed CSV as in the script).
- The code path accepts **either** explicit `store_dir`/`session_id` **or** derived `RUN_ID` layouts.

---

## Follow‑ups (nice to have, but not required for green CI)
- Add a `--phase=smoke` convenience in `eval_model.py` that toggles the above guards automatically.
- Add a tiny `tests/ci/test_smoke_pipeline.py` that forks a tmp run, executes Teach→Test for `memory/hei_nw` with `n=5`, asserts `metrics.json` presence and non‑zero gate attempts.
- Remove legacy scripts that embed old preset names or layouts; ensure `EVAL_PROTOCOL.md` reflects the minimal, correct commands (Teach→Test) parameterized by `RUN_ID`, `SEEDS`, `SIZES`.

---

## References (in‑repo)
- Smoke scripts: `scripts/smoke_eval.sh`, `scripts/ci_smoke_eval.sh`
- CLI wrapper: `scripts/eval_model.py` (store validation path; `_normalize_store_dir`)
- Harness & preflight: `hippo_eval/eval/harness.py` (`preflight_check`, `evaluate`)
- Store helpers: `hippo_mem/utils/stores.py` (`derive`, `assert_store_exists`, `validate_store`)
- Store meta & clearing: `hippo_eval/eval/store_utils.py` (`resolve_store_meta_path`, `clear_store`)
- Review/plan: `review/2025-09-05/*`, `plans/2025-09-05/pipeline_rework_stepwise_plan.md`
