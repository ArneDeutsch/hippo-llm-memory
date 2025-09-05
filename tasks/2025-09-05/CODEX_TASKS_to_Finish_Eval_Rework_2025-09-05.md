# Codex Tasks — Finish the Evaluation Rework (2025‑09‑05)
**Repo:** hippo-llm-memory  
**Goal:** Land the minimal code & docs changes so the *memory‑first* evaluation runs cleanly and the protocol is short, unambiguous, and parameterized.

---

## Conventions for all tasks
- **Editor mode:** Apply the exact diffs provided. Keep surrounding code unchanged.
- **Shell:** Assume repo root as CWD. Use bash.
- **Variables:** `RUN_ID`, `SIZES`, `SEEDS` (arrays), `RUNS`, `STORES` as defined by `scripts/_env.sh`.
- **Preset names:** `memory/sgc_rss`, `memory/hei_nw`, `memory/smpd` (no `_mem` suffix).
- **Boolean flags:** use `--no-retrieval-during-teach` (no `=true`).

---

## Task 0 — Create branch and preflight
**Context:** Keep changes isolated.
- **Run**
```bash
git checkout -b chore/eval-rework-min-protocol-2025-09-05
python -V
```

**DoD**
- New branch exists.

---

## Task 1 — Fix dataset builder shim
**Context:** `scripts/build_datasets.py` calls a missing helper. Use the package module entrypoint instead.
**File:** `scripts/build_datasets.py`

**Patch**
```diff
@@
-def _run_cli(args: list[str]) -> None:
-    """Invoke ``datasets_cli.py`` with ``args``."""
-    cmd = [sys.executable, str(ROOT / "datasets_cli.py"), *args]
+def _run_cli(args: list[str]) -> None:
+    """Invoke the datasets CLI module with ``args``."""
+    cmd = [sys.executable, "-m", "hippo_eval.datasets.cli", *args]
     subprocess.run(cmd, check=True)
```

**Quick check**
```bash
python scripts/build_datasets.py --suite semantic_mem --size 5 --seed 0
test -s datasets/semantic_mem/teach_00005_s000.jsonl
test -s datasets/semantic_mem/test_00005_s000.jsonl
```

**DoD**
- Command produces both `teach_*.jsonl` and `test_*.jsonl` for the suite.

---

## Task 2 — Do not require a store for baselines in `test` mode
**Context:** Baseline runs were failing because `scripts/eval_model.py` asserted a persisted store whenever `store_dir/session_id` were present.
**File:** `scripts/eval_model.py`

**Patch (route via `validate_store`)**
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
+        # Only memory presets need a persisted store.
+        if cfg.get("store_dir") or cfg.get("session_id"):
+            from hippo_mem.utils.stores import validate_store
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

**Quick checks**
```bash
# Baseline should run without a store (no store flags passed).
python scripts/eval_cli.py suite=semantic_mem n=5 seed=0 outdir=runs/dev_smoke/baseline

# Memory preset should still error if store is missing in test mode.
! python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss mode=test n=5 seed=0     outdir=runs/dev_smoke/mem_test_missing_store 2>&1 | grep -q "Persisted store not found" && echo "ok"
```

**DoD**
- Baseline `test` does not require a store.
- Memory `test` still guards against missing persistence.

---

## Task 3 — Synchronize CI smoke script
**Context:** CI referenced a non-existent preset and old dataset builder.
**File:** `scripts/ci_smoke_eval.sh`

**Patch**
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

**DoD**
- CI uses valid preset and the module builder.

---

## Task 4 — Replace `EVAL_PROTOCOL.md` with the minimal version
**Context:** The protocol has grown bloated. Replace it with a short, parameterized, copy‑pasteable recipe.
**File:** `EVAL_PROTOCOL.md`

**Action:** Replace the entire file content with the contents of `EVAL_PROTOCOL_minimal.md` (attached below in Task 4 Deliverable).

**DoD**
- File contains only the minimal protocol with variables `RUN_ID`, `SIZES`, `SEEDS` and the three suites.

**Deliverable**  
Paste the content from `EVAL_PROTOCOL_minimal.md` (see attachment/file in this task bundle).

---

## Task 5 — Remove/rename deprecated references
**Context:** Old names like `sgc_rss_mem` and path examples `store_dir=stores/...` linger in the repo.
**Action:** Search and fix/remove.

**Run**
```bash
# 1) Replace preset name
rg -n "sgc_rss_mem" | cut -d: -f1 | sort -u | xargs -I{} sed -i 's/sgc_rss_mem/sgc_rss/g' {}

# 2) Fix store_dir examples in docs/scripts (prefer runs/$RUN_ID/stores)
rg -n "store_dir=stores/" | cut -d: -f1 | sort -u | xargs -I{}   sed -i 's#store_dir=stores/#store_dir=runs/$RUN_ID/stores#g' {}

# 3) Remove any dead config aliases (manual): configs/eval/** if present
rg -n "sgc_rss_mem" configs || true
```

**DoD**
- No repository hits for `sgc_rss_mem`.
- No examples using `store_dir=stores/` remain (use `runs/$RUN_ID/stores`).

---

## Task 6 — Smoke the full minimal protocol (tiny sizes)
**Context:** Validate the end‑to‑end path with small inputs before committing.
**Run**
```bash
export RUN_ID=run_local_smoke
export SIZES=(5)
export SEEDS=(0)
source scripts/_env.sh  # exports RUNS, STORES=runs/$RUN_ID/stores

# Build datasets
python -m hippo_eval.datasets.cli --suite semantic_mem --size 5 --seed 0
python -m hippo_eval.datasets.cli --suite episodic_cross_mem --size 5 --seed 0
python -m hippo_eval.datasets.cli --suite spatial_multi --size 5 --seed 0

# semantic_mem
python scripts/eval_cli.py suite=semantic_mem n=5 seed=0 outdir=$RUNS/semantic_mem_baseline/5_0
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss mode=teach --no-retrieval-during-teach   n=5 seed=0 outdir=$RUNS/semantic_mem_teach/5_0 store_dir="$STORES" session_id="$RUN_ID"
python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss mode=test   n=5 seed=0 outdir=$RUNS/semantic_mem_test/5_0 store_dir="$STORES" session_id="$RUN_ID"

# episodic_cross_mem (hei_nw)
python scripts/eval_cli.py suite=episodic_cross_mem n=5 seed=0 outdir=$RUNS/episodic_cross_mem_baseline/5_0
python scripts/eval_cli.py suite=episodic_cross_mem preset=memory/hei_nw mode=teach --no-retrieval-during-teach   n=5 seed=0 outdir=$RUNS/episodic_cross_mem_teach/5_0 store_dir="$STORES" session_id="$RUN_ID"
python scripts/eval_cli.py suite=episodic_cross_mem preset=memory/hei_nw mode=test   n=5 seed=0 outdir=$RUNS/episodic_cross_mem_test/5_0 store_dir="$STORES" session_id="$RUN_ID"

# spatial_multi (smpd)
python scripts/eval_cli.py suite=spatial_multi n=5 seed=0 outdir=$RUNS/spatial_multi_baseline/5_0
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=teach --no-retrieval-during-teach   n=5 seed=0 outdir=$RUNS/spatial_multi_teach/5_0 store_dir="$STORES" session_id="$RUN_ID"
python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd mode=test   n=5 seed=0 outdir=$RUNS/spatial_multi_test/5_0 store_dir="$STORES" session_id="$RUN_ID"
```

**DoD**
- Each phase writes `metrics.json` under its `outdir`.
- Store files exist under `runs/$RUN_ID/stores/<algo>/$RUN_ID/`

---

## Task 7 — Commit and PR
**Run**
```bash
git add -A
git commit -m "chore(eval): minimal EVAL_PROTOCOL + dataset CLI fix + baseline store guard + CI sync"
git push -u origin chore/eval-rework-min-protocol-2025-09-05
```

**DoD**
- PR opened with passing smoke.

---

## Task 4 Deliverable — `EVAL_PROTOCOL_minimal.md`
> Use the separate file attached in this task bundle, or copy it verbatim into `EVAL_PROTOCOL.md`.
