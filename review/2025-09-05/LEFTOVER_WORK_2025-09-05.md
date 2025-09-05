# Gap Analysis & Fix Plan — Memory‑First Evaluation Rework
**Date:** 2025‑09‑05  
**Repo:** hippo‑llm‑memory  
**Scope:** Verify that the 2025‑09‑05 review → plan → tasks are implemented; identify gaps; supply concrete fixes (code + docs) and a clean, runnable protocol for the *memory‑first* suites.

---

## TL;DR (actionable)

1) **Fix broken commands & flags in `EVAL_PROTOCOL.md`.**  
   *Wrong:* `--no-retrieval-during-teach=true`, `preset=baseline`, `preset=memory/sgc_rss_mem`, `store_dir=stores/...`  
   *Right:* use boolean flags **without** `=true`, use `preset=baselines/core` (or omit), use `preset=memory/sgc_rss`, and prefer `store_dir=runs/$RUN_ID/stores`.

2) **Make `scripts/build_datasets.py` actually build datasets.**  
   It calls a **missing** script (`scripts/datasets_cli.py`). Call the module instead: `python -m hippo_eval.datasets.cli`.

3) **Baseline runs must not require a store.**  
   `scripts/eval_model.py` unconditionally asserts a store in `test` mode when `store_dir`/`session_id` are present. Route through `validate_store(...)` so baselines skip persistence checks.

4) **Align CI and docs with real preset names.**  
   `scripts/ci_smoke_eval.sh` and protocol still reference `memory/sgc_rss_mem`. The config on disk is `configs/eval/memory/sgc_rss.yaml`.

5) **Provide a minimal, copy‑pasteable “Memory‑First” recipe that runs.**  
   Included below; verified against the current tree layout.

---

## Evidence (where things break)

### A. Protocol commands (current repo)

**`EVAL_PROTOCOL.md`** – problematic lines:

- Lines 104–112 (and earlier duplicates) use:
```
python scripts/eval_cli.py suite=semantic_mem preset=baseline n=50 seed=1337 python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss_mem   mode=teach --no-retrieval-during-teach=true ...
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss_mem   mode=test ...
```
These cause (a) argparse error for the `--no-retrieval...` flag, (b) preset lookup mismatch, and (c) baseline forcing a non‑existent store. [Source: EVAL_PROTOCOL.md lines ~104–112]

### B. Dataset builder doesn’t exist

**`scripts/build_datasets.py`** tries to invoke a non‑existent helper:

```python
# scripts/build_datasets.py
cmd = [sys.executable, str(ROOT / "datasets_cli.py"), *args]
```

There is **no** `scripts/datasets_cli.py`; the actual entrypoint lives at **`hippo_eval/datasets/cli.py`**.

### C. Baseline store assertion in `test` mode

In **`scripts/eval_model.py`**:

```python
elif cfg.mode == "test":
    if cfg.get("store_dir") or cfg.get("session_id"):
        layout = _resolve_layout(cfg, algo)
        store_path = assert_store_exists( ... )
        _ensure_populated(store_path, cfg)
```

For `preset=baselines/...` this throws:
> FileNotFoundError: Persisted store not found … Expected path: runs/<RID>/stores/hei_nw/<SID>/episodic.jsonl

The helper **already** offers a `validate_store(run_id, preset, algo, kind)` that no‑ops for non‑memory presets; the wrapper simply isn’t using it.

### D. CI smoke script is out of sync

`scripts/ci_smoke_eval.sh` sets `PRESET=memory/sgc_rss_mem` and assumes the old CLI. Config files on disk are under `configs/eval/memory/sgc_rss.yaml` (no `_mem` suffix).

---

## What *is* implemented vs the plan

**Good news ✅**
- Memory‑first suites exist and map correctly:
  - `semantic_mem` → `hippo_eval.tasks.generators.generate_semantic(...)` via `hippo_eval/datasets/__init__.py`.
  - `episodic_cross_mem` and `spatial_multi` are present, with generators and dataset directories (`datasets/semantic_mem`, `datasets/episodic_cross_mem`, `datasets/spatial_multi`).
- Configs for memory presets are present: `configs/eval/memory/{hei_nw,sgc_rss,smpd}.yaml`.
- Store layout helpers (`hippo_mem/utils/stores.py`) formalize `{base_dir}/{algo}/{session_id}/{kind}.jsonl`.

**Gaps 🚧**
- Protocol & CI still reference **old names/flags**.
- Dataset builder entrypoint mismatch.
- Baseline `test` path wrongly asserts a store if `store_dir` is in scope.
- Several doc snippets encourage `store_dir=stores/...` (missing the `runs/` prefix used everywhere else).

---

## Concrete fixes (patches)

> Style: minimal, targeted diffs you can paste directly.

### 1) Fix dataset builder

**`scripts/build_datasets.py`**
```diff
@@
-def _run_cli(args: list[str]) -> None:
-    """Invoke ``datasets_cli.py`` with ``args``."""
-    cmd = [sys.executable, str(ROOT / "datasets_cli.py"), *args]
+def _run_cli(args: list[str]) -> None:
+    """Invoke the datasets CLI module with ``args``."""
+    # Use module path so we don't depend on a sibling script that doesn't exist.
+    cmd = [sys.executable, "-m", "hippo_eval.datasets.cli", *args]
     subprocess.run(cmd, check=True)
```

### 2) Baselines should not require a store in `test`

**`scripts/eval_model.py`**  
Route through `validate_store(...)`. Only call `_ensure_populated(...)` if a path is returned.

```diff
@@
-    elif cfg.mode == "test":
-        if cfg.get("store_dir") or cfg.get("session_id"):
-            layout = _resolve_layout(cfg, algo)
-            store_path = assert_store_exists(
-                str(layout.base_dir), str(cfg.session_id), algo, kind=store_kind
-            )
-            _ensure_populated(store_path, cfg)
-        elif cfg.get("store_dir"):
-            cfg.store_dir = _normalize_store_dir(str(cfg.store_dir), algo)
+    elif cfg.mode == "test":
+        if cfg.get("store_dir") or cfg.get("session_id"):
+            from hippo_mem.utils.stores import validate_store
+            layout = _resolve_layout(cfg, algo)
+            # Returns a Path for memory presets, or None for baselines.
+            maybe_store = validate_store(
+                run_id=str(cfg.run_id or os.getenv("RUN_ID") or ""),
+                preset=str(cfg.preset),
+                algo=algo,
+                kind=store_kind,
+            )
+            if maybe_store is not None:
+                _ensure_populated(maybe_store, cfg)
+        elif cfg.get("store_dir"):
+            cfg.store_dir = _normalize_store_dir(str(cfg.store_dir), algo)
```

*(If `validate_store` already derives the layout internally, you can simplify to `validate_store(run_id=..., preset=cfg.preset, algo=algo, kind=store_kind)` and drop `_resolve_layout`.)*

### 3) Make flags & presets correct in docs

**`EVAL_PROTOCOL.md`** — update the *Memory‑First Suite Recipes*:

```diff
- python scripts/build_datasets.py --suite semantic_mem --size 50 --seed 1337
- python scripts/eval_cli.py suite=semantic_mem preset=baseline n=50 seed=1337 + python -m hippo_eval.datasets.cli --suite semantic_mem --size 50 --seed 1337
+ # Baseline (defaults to baselines/core)
+ python scripts/eval_cli.py suite=semantic_mem n=50 seed=1337     outdir=runs/$RUN_ID/semantic_mem_baseline
- python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss_mem -  mode=teach --no-retrieval-during-teach=true n=50 seed=1337 + # Teach (disable retrieval via boolean flag, no '=true')
+ python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss +  mode=teach --no-retrieval-during-teach n=50 seed=1337 -  outdir=runs/$RUN_ID/semantic_mem_teach -  store_dir=stores/$RUN_ID/semantic_mem session_id=$RUN_ID
+  outdir=runs/$RUN_ID/semantic_mem_teach +  store_dir=runs/$RUN_ID/stores session_id=$RUN_ID
- python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss_mem -  mode=test n=50 seed=1337 + # Test
+ python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss +  mode=test n=50 seed=1337 -  outdir=runs/$RUN_ID/semantic_mem_test -  store_dir=stores/$RUN_ID/semantic_mem session_id=$RUN_ID
+  outdir=runs/$RUN_ID/semantic_mem_test +  store_dir=runs/$RUN_ID/stores session_id=$RUN_ID
```

Apply the same substitutions in the `episodic_cross_mem` and `spatial_multi` sections:
- `preset=baseline` → *(omit)* or `preset=baselines/core`
- `memory/sgc_rss_mem` → `memory/sgc_rss`
- `--no-retrieval-during-teach=true` → `--no-retrieval-during-teach`
- `store_dir=stores/$RUN_ID/...` → `store_dir=runs/$RUN_ID/stores`

### 4) Synchronize CI smoke script

**`scripts/ci_smoke_eval.sh`**
```diff
-SUITE=${SUITE:-semantic_mem}
-PRESET=${PRESET:-memory/sgc_rss_mem}
-SESSION_ID=${SESSION_ID:-sgc_${RUN_ID}}
+SUITE=${SUITE:-semantic_mem}
+PRESET=${PRESET:-memory/sgc_rss}
+SESSION_ID=${SESSION_ID:-${SUITE}_${RUN_ID}}
@@
-python scripts/build_datasets.py --suite "$SUITE" --size 50 --seed 1337
+python -m hippo_eval.datasets.cli --suite "$SUITE" --size 50 --seed 1337
@@
-  store_dir="$STORES/$SUITE" session_id="$SESSION_ID"
+  store_dir="$STORES" session_id="$SESSION_ID"
```

---

## Clean, runnable recipe (copy‑paste)

> Assumes you have a GPU environment ready, models configured, and this repo checked out.  
> Replace `Qwen/Qwen2.5-1.5B-Instruct` if needed.

```bash
# 0) Setup
export RUN_ID=run_20250905
source scripts/_env.sh  # defines RUNS, STORES=runs/$RUN_ID/stores, HEI_SESSION_ID=hei_$RUN_ID

# 1) Build datasets (n=50, seed=1337)
python -m hippo_eval.datasets.cli --suite semantic_mem --size 50 --seed 1337
python -m hippo_eval.datasets.cli --suite episodic_cross_mem --size 50 --seed 1337
python -m hippo_eval.datasets.cli --suite spatial_multi --size 50 --seed 1337

# 2) semantic_mem
# 2a) Baseline (no store needed; do NOT pass store_dir/session_id)
python scripts/eval_cli.py suite=semantic_mem n=50 seed=1337   outdir=runs/$RUN_ID/semantic_mem_baseline

# 2b) Teach (retrieval OFF)
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss   mode=teach --no-retrieval-during-teach n=50 seed=1337   outdir=runs/$RUN_ID/semantic_mem_teach   store_dir="$STORES" session_id="$RUN_ID"

# 2c) Test (retrieval ON, reads persisted store)
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss   mode=test n=50 seed=1337   outdir=runs/$RUN_ID/semantic_mem_test   store_dir="$STORES" session_id="$RUN_ID"

# 3) Repeat 2a–2c for episodic_cross_mem and spatial_multi (use the same $STORES/$RUN_ID)
```

---

## Follow‑ups to fully fulfil the 2025‑09‑05 plan

- [ ] **Update `EVAL_PROTOCOL.md`** with the corrected recipe and a short callout:
  - Boolean flags are specified without `=true` (argparse `BooleanOptionalAction`).
  - The recommended `store_dir` is `runs/$RUN_ID/stores` (algorithm subfolder is added automatically).
  - For baselines, do not pass persistence flags unless you really want to co‑locate outputs.

- [ ] **Fix `scripts/build_datasets.py`** (patch above) and add a unit smoke:
  - `python -m hippo_eval.datasets.cli --suite semantic_mem --size 5 --seed 0` writes 2 JSONL files
  - `datasets/semantic_mem/*_teach.jsonl` and `*_test.jsonl` exist and are non‑empty.

- [ ] **Patch `scripts/eval_model.py`** (patch above) and add a regression test:
  - `preset=baselines/core`, `mode=test`, with `store_dir` present should **not** raise.
  - For memory presets, missing store should still raise with the helpful message.

- [ ] **Synchronize CI (`scripts/ci_smoke_eval.sh`)**:
  - Use `memory/sgc_rss`, call the module builder, and pass `store_dir="$STORES"`.

- [ ] **Doc polish in `EVAL_PLAN.md`** (minor but useful):
  - Replace any remaining mentions of `preset=baseline` with `preset=baselines/core` (or omit).
  - Ensure the *Paired Datasets* section explicitly says: “facts only in teach, queries only in test.”
  - Add a **Guardrail** bullet: fail preflight if a *memory‑required* suite gets ≥ X% solved without store access.

- [ ] **Remove deprecated crumbs once green twice** (keep a tag/branch before deletion):
  - Any reference to `sgc_rss_mem` (search in repo) → rename to `sgc_rss` or delete dead snippets.
  - Replace `store_dir=stores/...` examples with `store_dir=runs/$RUN_ID/stores`.
  - If you still have old suite aliases (`semantic`, `episodic_cross`, `spatial` without the new suffixes) in docs/scripts, remove or redirect to the `*_mem`/`*_multi` names.

---

## Why the earlier error happened (root cause recap)

1) **Arg flag misuse.** `--no-retrieval-during-teach=true` is parsed as a positional `true` by argparse (`BooleanOptionalAction`), so it errors. Use just `--no-retrieval-during-teach`.

2) **Wrong preset name.** `memory/sgc_rss_mem` doesn’t exist; the in‑tree config is `memory/sgc_rss`.

3) **Store assertion on baseline.** When `preset=baseline` (or default baseline) is combined with `store_dir/session_id` in the wrapper, `eval_model.py` currently asserts that a store file exists. Baselines don’t create or need stores. The wrapper needs to call `validate_store(...)`, which already skips the check for non‑memory presets.

4) **Path confusion.** Using `store_dir=stores/$RUN_ID/...` mixes the intended `runs/$RUN_ID/stores` with a sibling path. Helpers assume `base_dir` is the path that **contains** `<algo>/<session_id>/<kind>.jsonl`. Prefer `runs/$RUN_ID/stores`—or, if you truly want suite‑scoped stores, use `runs/$RUN_ID/stores/<suite>` consistently (the helper will still append `/algo/<session_id>`).

---

## Acceptance checks (define “done”)

- A fresh developer can:
  1. Build datasets for all three suites with a single command per suite.  
  2. Run baselines **without** passing any store flags.  
  3. Run teach/test for `memory/sgc_rss` using a shared `runs/$RUN_ID/stores` base and a human‑readable `session_id`.  
  4. See `*_teach.jsonl`, `*_test.jsonl`, `metrics.json`, and persisted stores materialize in the documented locations.  
  5. Re‑run `test` with `--isolate=per_episode` and observe identical pre‑metrics; then run `replay` (optional) and observe `post_*` deltas written.

When all of the above are green twice in a row, proceed to delete stale snippets and archive the old protocol.

---

*Prepared for the 2025‑09‑05 review loop. Ping me if you want this split into bite‑size Codex tasks; the patches above are safe to land first.*
