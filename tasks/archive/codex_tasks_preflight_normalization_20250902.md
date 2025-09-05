# Codex Task List — Preflight Normalization & Robustness
**Scope:** Fix preflight RUN_ID normalization, unify store path resolution, normalize replay cycles config, improve diagnostics, add tests, and update docs.  
**Target date:** ASAP  
**Owner:** Codex

---

## Task 1 — Preflight accepts both RUN_ID forms for baselines
**Files:** `hippo_mem/eval/harness.py`  
**Context:** Aggregator writes to underscored RUN_ID; preflight sometimes expects digits‑only.  
**Change:** In `preflight_check`, when `mode != "teach"`, look for baseline metrics at both candidate paths.

### Patch (illustrative)
```diff
@@ def preflight_check(cfg: DictConfig, outdir: Path) -> None:
-    baseline_metrics = Path("runs") / str(cfg.get("run_id")) / "baselines" / "metrics.csv"
-    if not baseline_metrics.exists():
-        failures.append(
-            "missing baseline metrics: "
-            f"{baseline_metrics} — generate via: python scripts/run_baselines.py --run-id {cfg.get('run_id')}"
-        )
+    rid = str(cfg.get("run_id"))
+    digits = rid.replace("_", "")
+    candidates = [Path("runs") / rid / "baselines" / "metrics.csv"]
+    if digits != rid:
+        candidates.append(Path("runs") / digits / "baselines" / "metrics.csv")
+    if not any(p.exists() for p in candidates):
+        shown = " or ".join(str(p) for p in candidates)
+        failures.append(
+            f"missing baseline metrics: {shown} — generate via: "
+            f"python scripts/run_baselines.py --run-id {rid} (or {digits})"
+        )
```

**Acceptance criteria:**
- Preflight passes when **either** path exists.
- `failed_preflight.json` lists **both** candidate paths when missing.

---

## Task 2 — Normalize replay cycles configuration key
**Files:** `hippo_mem/eval/harness.py` (and any module reading cycles)  
**Change:** Introduce helper `get_replay_cycles(cfg)` that returns `cfg.get("replay_cycles", cfg.get("replay", {}).get("cycles", 0))`, and use it everywhere cycles are read (loop count, meta writing, logs).

**Acceptance criteria:**
- Passing `replay.cycles=3` or `replay_cycles=3` yields identical behavior.
- Unit test covers both spellings.

---

## Task 3 — Unify `store_dir` resolution (base vs algo dir)
**Files:** `hippo_mem/eval/harness.py`, `hippo_mem/eval/store_utils.py` (new)  
**Change:** Add `resolve_store_meta_path(preset: str, store_dir: Path, session_id: str) -> Path`:
- Infer `<algo>` from `preset` (`memory/hei_nw` → `hei_nw`).
- If `store_dir` ends with `<algo>`, use `store_dir/<session_id>/store_meta.json`.
- Else, assume base and use `store_dir/<algo>/<session_id>/store_meta.json`.
Use this function in **both** teach and preflight.

**Acceptance criteria:**
- Works for both `store_dir=runs/<RID>/stores` and `store_dir=runs/<RID>/stores/<algo>`.
- Validator (`scripts/validate_store.py`) uses the same resolver.

---

## Task 4 — Improve preflight diagnostics
**Files:** `hippo_mem/eval/harness.py`  
**Change:** When store meta is missing, show **both** attempted paths (base+algo). When baselines are missing, show **both** RUN_ID candidates and the exact commands to generate them.

**Acceptance criteria:**
- `failed_preflight.json` includes actionable next steps and all candidate paths.

---

## Task 5 — Documentation updates
**Files:** `EVAL_PROTOCOL.md`, `scripts/eval_cli.py` (help text), `README.md`  
**Changes:**
- Add explicit “Run baselines” step **before** memory runs:
  ```bash
  python scripts/run_baselines.py --run-id "$RUN_ID"
  ```
- Explain RUN_ID normalization and that preflight accepts *both* forms.
- Clarify `store_dir` semantics with two canonical patterns and recommend one.
- Show correct replay cycles flag(s).

**Acceptance criteria:**
- Following the protocol verbatim yields a green preflight for fresh clones.

---

## Task 6 — Unit tests
**Files:** `tests/test_preflight.py`, `tests/test_store_resolution.py` (new)  
**Add tests:**
1. `test_preflight_accepts_both_runid_forms`  
   - Create fake `runs/<rid_u>/baselines/metrics.csv`, run preflight with `rid_d`, expect pass.
2. `test_store_meta_path_resolution_base_and_algo_dir`  
   - Ensure resolver finds meta for both `store_dir` styles.
3. `test_preflight_missing_baselines_lists_both_paths`  
   - Expect both candidates in the error message/diagnostics.
4. `test_replay_cycles_config_fallback`  
   - Passing cycles via nested and flat keys behaves the same.

**Acceptance criteria:**
- All new tests pass in CI. Failing tests produce clear diffs.

---

## Task 7 — Make teach meta robust (optional safety)
**Files:** `hippo_mem/memory/store.py` or wherever meta is finalized  
**Change:** If `replay_samples > 0`, always set `source="replay"`. If not, but gating attempted writes during teach, set `source="teach"` rather than `"stub"`.

**Acceptance criteria:**
- No false `"stub"` metas after a successful teach with replay activity.

---

## Task 8 — Add developer triage helper
**Files:** `scripts/tools/triage_preflight.sh` (new)  
**Content:** Include the shell from the review doc to print meta, both baseline paths, and diagnostics.

**Acceptance criteria:**
- Running the script surfaces the exact misconfiguration in under 5 seconds.

---

## Task 9 — (Optional) Aggregator dual‑write for compatibility
**Files:** `scripts/run_baselines.py`  
**Change:** If `RUN_ID` contains `_`, optionally write/symlink metrics to both underscored and digits‑only paths (behind a `--write-compat-symlink` flag).

**Acceptance criteria:**
- With the flag, either RUN_ID form will find baselines without re‑running aggregation.

---

## Definition of Done
- Code merged with Tasks 1–6 mandatory; Task 7–9 optional but recommended.
- CI green with new tests.
- `EVAL_PROTOCOL.md` updated and verified on a clean workspace.
- Manual smoke test:
  1. `python scripts/run_baselines.py --run-id 2025_09_02_50_1337_2025`  
  2. Teach with `store_dir=runs/<RID>/stores/hei_nw`, persist true.  
  3. Replay/Test with either `run_id` form and either cycles key → preflight passes.
