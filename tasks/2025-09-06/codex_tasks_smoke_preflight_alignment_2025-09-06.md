
# Codex Tasks — Align Smoke Tests & Preflight with the Memory‑First Pipeline
**Repo:** `hippo-llm-memory`  •  **Date:** 2025‑09‑06  •  **Owner:** Codex implementation  
**Goal:** Make CI smoke green by switching to a **Teach → Test** flow, using **base `store_dir`**, and scoping **preflight** to the new pipeline. Remove brittle legacy assumptions.

> Each task below is self‑contained: it includes context, concrete edits (with diff‑style hints), and verification steps. Execute in order.

---

## CT‑001 — Fix CI smoke script to **Teach → Test** with base `store_dir`
**Context**  
Current CI attempts `mode=test` with `store_dir=runs/$RUN_ID/stores/<algo>` but no prior persisted store exists; early validation fails. We must: (a) build dataset, (b) **Teach** with `persist=true`, (c) **Test** using the same `session_id`, (d) generate report, (e) assert tiny baseline CSV exists for compatibility.

**Files**  
- `scripts/ci_smoke_eval.sh`

**Edits (shell)**  
- Set `STORES="runs/$RUN_ID/stores"` (base — **no algo suffix**).  
- Derive `SESSION_ID` from the preset prefix: `SESSION_ID=${SESSION_ID:-${PRESET##*/}_$RUN_ID}` is OK (or `${PREFIX}_${RUN_ID}` if you keep `PREFIX`).  
- Add explicit **Teach** step with `persist=true`.  
- Pass `store_dir="$STORES"` (base) to Teach and Test.  
- Keep baseline CSV stub to satisfy legacy preflight.

**Suggested replacement block**
```bash
# --- BEGIN canonical CI smoke ---
SUITE=${SUITE:-episodic_cross_mem}
PRESET=${PRESET:-memory/hei_nw}
SESSION_ID=${SESSION_ID:-${PRESET##*/}_$RUN_ID}
STORES="runs/$RUN_ID/stores"    # BASE dir (no algo suffix)

# 0) Ensure minimal baselines to appease legacy preflight
mkdir -p "runs/$RUN_ID/baselines"
test -f "runs/$RUN_ID/baselines/metrics.csv" ||   echo "suite,em_raw,em_norm,f1" > "runs/$RUN_ID/baselines/metrics.csv"

# 1) Deterministic dataset
python -m hippo_eval.datasets.cli --suite "$SUITE" --size 50 --seed 1337 --out "datasets/$SUITE"

# 2) Teach with persistence
python scripts/eval_model.py   suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=50 seed=1337   mode=teach persist=true store_dir="$STORES" session_id="$SESSION_ID"   compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 3) Test (read persisted store)
python scripts/eval_model.py   suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=50 seed=1337   mode=test store_dir="$STORES" session_id="$SESSION_ID"   compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 4) Report
python -m hippo_eval.reporting.report --run-id "$RUN_ID"
# --- END canonical CI smoke ---
```

**Acceptance criteria**
- `scripts/ci_smoke_eval.sh` completes without `FileNotFoundError` and without emitting `failed_preflight.json` for the smoke run.
- Report exists under `runs/$RUN_ID` and baseline CSV exists (stub or real).

**Verify**
```bash
RUN_ID=ci_smoke MODEL=models/tiny-gpt2 bash scripts/ci_smoke_eval.sh && echo OK
```

---

## CT‑002 — Gate early store validation in `scripts/eval_model.py`
**Context**  
Early validation calls `validate_store(...)` even when we shouldn’t. It must only run if we truly depend on an existing store: memory preset, `mode in {"test","replay"}`, and both `store_dir` & `session_id` are present.

**Files**  
- `scripts/eval_model.py`

**Edits (Python; pseudo‑diff)**
```diff
- if cfg.get("store_dir") or cfg.get("session_id"):
+ needs_store = (
+   str(cfg.get("preset","")).startswith("memory/")
+   and str(cfg.get("mode","test")) in {"test","replay"}
+   and cfg.get("store_dir") and cfg.get("session_id")
+ )
+ if needs_store:
     from hippo_mem.utils.stores import validate_store
     maybe_store = validate_store(
         run_id=str(cfg.get("run_id") or os.getenv("RUN_ID") or ""),
         preset=str(cfg.preset),
         algo=algo,
         kind=store_kind,
+        store_dir=str(cfg.get("store_dir")),
+        session_id=str(cfg.get("session_id")),
     )
     if maybe_store is not None:
         _ensure_populated(maybe_store, cfg)
- elif cfg.get("store_dir"):
-     cfg.store_dir = _normalize_store_dir(str(cfg.store_dir), algo)
+ elif cfg.get("store_dir"):
+     cfg.store_dir = _normalize_store_dir(str(cfg.store_dir), algo)
```

**Acceptance criteria**
- Running Teach (no store yet) never trips early validation.
- Running Test/Replay with explicit `store_dir`+`session_id` validates the correct path and proceeds.

**Verify**
```bash
# Teach creates store
python scripts/eval_model.py suite=episodic_cross_mem preset=memory/hei_nw run_id=tmp n=5 seed=1 mode=teach persist=true store_dir=runs/tmp/stores session_id=hei_tmp model=models/tiny-gpt2
# Test reads it; no FileNotFoundError
python scripts/eval_model.py suite=episodic_cross_mem preset=memory/hei_nw run_id=tmp n=5 seed=1 mode=test store_dir=runs/tmp/stores session_id=hei_tmp model=models/tiny-gpt2
```

---

## CT‑003 — Make `validate_store(...)` accept explicit `store_dir` & `session_id`
**Context**  
The helper currently derives a path from `RUN_ID` and `algo`. We must prefer explicit inputs when provided, falling back to derived layout otherwise.

**Files**  
- `hippo_mem/utils/stores.py`

**Edits (Python; pseudo‑diff)**
```diff
-def validate_store(run_id: str, preset: str, algo: str, kind: str = "episodic") -> Path | None:
+def validate_store(run_id: str, preset: str, algo: str, kind: str = "episodic",
+                   store_dir: str | None = None, session_id: str | None = None) -> Path | None:
     """
-    Resolve the expected store path and assert it exists.
+    Resolve the expected store path and assert it exists. Prefer explicit
+    (store_dir, session_id) when provided; otherwise derive from run_id/algo.
     """
-    layout = derive(run_id=run_id, algo=algo)
+    if store_dir and session_id:
+        base = Path(store_dir)
+        algo_dir = base if base.name == algo else base / algo
+        filename = {"episodic":"episodic.jsonl", "kg":"kg.jsonl", "spatial":"spatial.jsonl"}[kind]
+        path = algo_dir / session_id / filename
+        if not path.exists():
+            raise FileNotFoundError(f"Persisted store not found. Expected path: {path}")
+        return path
+
+    layout = derive(run_id=run_id, algo=algo)
     # existing derived‑path checks...
```

**Acceptance criteria**
- Works with both explicit path and derived layout.
- Raises clear `FileNotFoundError` with the exact path it expected.

**Verify**
```bash
python - <<'PY'
from hippo_mem.utils.stores import validate_store
try:
    validate_store("tmp","memory/hei_nw","hei_nw","episodic","runs/tmp/stores","hei_tmp")
    print("explicit ok")
except FileNotFoundError as e:
    print("handled:", e)
PY
```

---

## CT‑004 — Relax preflight checks; keep the **gate activity** signal
**Context**  
Preflight should keep the **teach‑dry‑run gate attempts ≥ 1** check, but it must not require baselines or a persisted store unless the run actually uses them.

**Files**  
- `hippo_eval/eval/harness.py` (or wherever `preflight_check` lives)

**Changes**
- Gate store checks behind `mode in {"test","replay"}` **and** presence of `store_dir` & `session_id`.
- Baseline metrics: only require when computing baseline deltas; otherwise log a WARNING (don’t fail).
- Ensure there is no *other* earlier validator that can fail before `preflight_check` (that is handled by CT‑002).

**Acceptance criteria**
- Smoke run (Teach→Test) does not emit `failed_preflight.json` for missing baselines/store if the flow is correct.
- If `teach` wiring is broken (no gate attempts), preflight still fails fast.

**Verify**
```bash
RUN_ID=ci_smoke bash scripts/ci_smoke_eval.sh && echo "preflight green"
# Simulate broken teach: run with impossible gate config and confirm preflight fails.
```

---

## CT‑005 — Normalize `store_dir` handling and test it
**Context**  
We must avoid double‑suffixing algo directories (e.g., passing `.../stores/hei_nw` **and** appending `hei_nw` again).

**Files**  
- `scripts/eval_model.py` (`_normalize_store_dir`)  
- `hippo_mem/utils/stores.py` (if also normalizing)  
- Tests under `tests/`

**Changes**
- `_normalize_store_dir(path, algo)` should accept either the base `.../stores` **or** `.../stores/<algo>` and return a **normalized** base that the rest of the code expects. Prefer the convention “pass base; derive algo dir later.”
- Add unit tests to cover cases:
  - Input base → base
  - Input base/algo → base (no duplication)
  - Mixed path separators

**Verify (pytest)**
```bash
pytest -q tests/utils/test_store_dir_normalize.py -q
```

---

## CT‑006 — Align all scripts to new flow and naming
**Context**  
Remove remnants of old pipeline (flags, names, hardcoded paths).

**Files**  
- `scripts/smoke_eval.sh`
- `scripts/ci_smoke_eval.sh`
- Any other shell wrappers

**Changes**
- Presets: only `memory/hei_nw`, `memory/sgc_rss`, `memory/smpd`.
- Suites: `episodic_cross_mem`, `semantic_mem`, `spatial_multi`.
- Boolean flags in Hydra style (`flag=true/false`), no `--flag=...` mix.
- Always pass base `store_dir` in scripts; never `.../stores/<algo>`.

**Verify**
```bash
# Local smoke for three presets (n=5 for speed)
for P in memory/hei_nw memory/sgc_rss memory/smpd; do   RUN_ID=local_smoke MODEL=models/tiny-gpt2 PRESET=$P SUITE=episodic_cross_mem bash scripts/ci_smoke_eval.sh; done
```

---

## CT‑007 — Update `EVAL_PROTOCOL.md` to the minimal memory‑first steps
**Context**  
Protocol must reflect Teach→Test and parameterization by `RUN_ID`, `SEEDS`, `SIZES` (no deprecated calls).

**Files**  
- `EVAL_PROTOCOL.md`

**Changes**
- Provide a two‑step sequence (Teach with `persist=true`, then Test), parameterized by environment vars:
  - `RUN_ID=...`
  - `SEEDS=[...]`
  - `SIZES=[...]`
- Show base `store_dir` usage.
- Remove legacy sections that instruct direct Test without prior Teach.

**Verify**
- Manual read‑through; run a single end‑to‑end with the documented commands.

---

## CT‑008 — Tests: unit + integration for the new behavior
**Context**  
Codify guardrails.

**Files**  
- `tests/utils/test_validate_store.py`
- `tests/eval/test_smoke_memory_flow.py`

**Unit tests**
- `validate_store` returns explicit path when provided; raises with correct path string when missing.
- `_normalize_store_dir` behavior per CT‑005.

**Integration smoke (pytest)**
- Fixture runs Teach (`n=5`, `seed=1337`, `persist=true`) then Test with the same `session_id`, asserts:
  - No `failed_preflight.json` in run tree.
  - Store file exists at expected path.
  - `metrics.json` present and parseable.

**Verify**
```bash
pytest -q tests/utils/test_validate_store.py tests/eval/test_smoke_memory_flow.py -q
```

---

## CT‑009 — Clean up preflight triage script and workflow glue
**Context**  
Ensure triage scripts and CI workflow expect the new artifacts/paths.

**Files**  
- `scripts/tools/triage_preflight.sh`
- `.github/workflows/*` (where smoke runs)

**Changes**
- Triage should scan `runs/$RUN_ID/**/failed_preflight.json` but **not** fail on missing baselines when not used.
- Workflows call `scripts/ci_smoke_eval.sh` only; remove ad‑hoc command lists that bypass Teach.

**Verify**
- CI workflow passes using only the smoke script.

---

## CT‑010 — Repo hygiene: remove legacy pipeline remnants
**Context**  
Purge confusing, unused, or deprecated code/paths.

**Actions**
- Grep and remove or fix:
  - Hardcoded `stores/<algo>` usages passed into `store_dir=`
  - Old preset names / suite aliases
  - Mixed `--flag=value` styles in Hydra overrides
- Update `Makefile` targets to use the new flow.

**Verify**
```bash
git grep -nE 'stores/(hei_nw|sgc_rss|smpd)' || echo "no hardcoded algo paths"
git grep -nE 'mode=test[^\n]*session_id' | grep -v 'persist=true' || true
```

---

## Definition of Done (for the whole set)
- `scripts/ci_smoke_eval.sh` completes green on CI with **Teach → Test**, `n=50`, `seed=1337`.
- No `FileNotFoundError` from `validate_store` when running Test/Replay with explicit paths.
- `preflight_check` retains gate‑attempts signal and no longer fails on unused baseline/store conditions.
- Minimal `EVAL_PROTOCOL.md` documents the correct, parameterized flow.
- Unit + integration tests cover `validate_store`, normalization, and the smoke pipeline.

