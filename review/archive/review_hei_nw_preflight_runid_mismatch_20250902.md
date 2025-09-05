# HEI‑NW Evaluation Preflight Failures — Root Cause Analysis & Recommendations
**Date:** 2025‑09‑02  
**Run context:** `RUN_ID=20250902_50_1337_2025`, model `Qwen/Qwen2.5-1.5B-Instruct`, SIZES=50, SEEDS=(1337, 2025)

## TL;DR
Preflight kept failing in `mode=replay`/`mode=test` **despite** successful teach because the harness looked for baseline metrics under a **normalized (digits‑only) RUN_ID** (`202509025013372025`), while the baseline aggregator wrote to the **underscored RUN_ID** (`20250902_50_1337_2025`). Everything else was healthy (store persisted, meta shows `source="replay"`). Re‑running baselines for the digits‑only RUN_ID made preflight pass; replay and test then ran cleanly.

---

## What we observed
- Teach phase processes all 50 tasks and logs healthy telemetry:
  - `write_accept_rate=1.00`
  - Fast retrieval latencies
- Store metadata exists and is valid:
  ```json
  {"schema": "episodic.store_meta.v1", "replay_samples": 50, "source": "replay", "created_at": "..."}
  ```
  at `runs/20250902_50_1337_2025/stores/hei_nw/hei_20250902_50_1337_2025/store_meta.json`
- `run_baselines.py` executed and wrote:  
  `runs/20250902_50_1337_2025/baselines/metrics.csv`
- **Preflight still fails** for replay/test and writes diagnostics under each outdir, e.g.:
  ```json
  {
    "errors": [
      "missing baseline metrics: runs/202509025013372025/baselines/metrics.csv — generate via: python scripts/run_baselines.py --run-id 202509025013372025"
    ]
  }
  ```

## Root cause
A **RUN_ID normalization mismatch** inside the preflight baseline path resolution:
- Aggregator writes to `runs/<RUN_ID_with_underscores>/baselines/metrics.csv`.
- Preflight looks at `runs/<RUN_ID_without_underscores>/baselines/metrics.csv`.
- Result: baseline file “missing” from preflight’s perspective even though it exists for the underscored RUN_ID.

### Why we initially missed it
- Earlier issues (store path semantics and baseline aggregation missing) masked the final blocker.
- Telemetry was *too good* (100% accepts), so we assumed store meta/teach path was the problem. It wasn’t.
- Diagnostics were clear once we checked `failed_preflight.json` systematically.

## Evidence
**Meta present and non‑stub:**
```
Expect meta here: runs/20250902_50_1337_2025/stores/hei_nw/hei_20250902_50_1337_2025/store_meta.json
Found:
{"schema": "episodic.store_meta.v1", "replay_samples": 50, "source": "replay", "created_at": "..."}
```
**Baselines present (underscored):**
```
-rw-rw-r-- 1 arne arne ... runs/20250902_50_1337_2025/baselines/metrics.csv
```
**Preflight diagnostics expecting digits‑only path:**
```
missing baseline metrics: runs/202509025013372025/baselines/metrics.csv
```

## Immediate remediation applied
- **Option B**: Re‑aggregate baselines for the digits‑only RUN_ID and run replay/test with `run_id=202509025013372025` **while** pointing `store_dir` and `session_id` at the already‑persisted underscored store. Result: replay/test executed without error.

## Recommended permanent fixes
1. **Accept both RUN_ID forms in preflight baseline lookup.**  
   Check `runs/<rid>/baselines/metrics.csv` and, if `rid` contains underscores, also check `runs/<rid_without_underscores>/baselines/metrics.csv`. Pass if either exists.
2. **Unify `store_dir` contract** (already aligned in practice): preflight should resolve stores identically to teach/replay.
3. **Config key normalization for replay cycles.**  
   Use `cfg.get("replay_cycles") or cfg.get("replay", {}).get("cycles")` everywhere.
4. **Improve diagnostics**: when baselines are missing, print *both* candidate paths and suggest commands for both RUN_ID forms.
5. **Documentation updates** (`EVAL_PROTOCOL.md`, `scripts/eval_cli.py --help`):  
   - Explicit “Run baselines” step.  
   - Clarify RUN_ID normalization and accepted forms.  
   - Clarify `store_dir` semantics with examples for memory presets.
6. **Tests** to prevent regression:  
   - Preflight accepts either RUN_ID form for baselines.  
   - Store meta path resolved correctly for base vs algo directory forms.  
   - Replay cycles key fallback works.  
   - Diagnostic messages include both candidate baseline paths.

## Risks & mitigations
- **Mixed RUN_IDs in a single run**: Allowed, but confusing. Diagnostics now make it explicit which path is used.  
- **Legacy runs without digits‑only baselines**: Covered by the fallback in (1).  
- **Replay cycles key mismatch** could break runs post‑preflight. We normalize the key to avoid this.

## What “meaningful validation” now looks like
With preflight unblocked:
- Baseline vs memory deltas computed from the correct `baselines/metrics.csv`.
- Stores verified (`source="replay"`, `replay_samples>0`) before replay/test.  
- Reproducible teach → replay (multi‑cycle) → test measurements across suites and seeds.

## Handy triage (keep in repo under `scripts/tools/triage_preflight.sh`)
```bash
RID=20250902_50_1337_2025
ALGO=hei_nw
SID="hei_${RID}"
META="runs/$RID/stores/$ALGO/$SID/store_meta.json"
echo "Expect meta: $META"
[ -f "$META" ] && cat "$META" || { echo "MISSING: $META"; find "runs/$RID/stores" -maxdepth 3 -name store_meta.json -print; }
ls -l "runs/$RID/baselines/metrics.csv" || echo "Baseline CSV (underscored) missing"
DIGITS="${RID//_/}"
ls -l "runs/$DIGITS/baselines/metrics.csv" || echo "Baseline CSV (digits) missing"
for p in runs/$RID/memory/$ALGO/*/50_1337; do
  [ -f "$p/failed_preflight.json" ] && echo "---- $p" && cat "$p/failed_preflight.json"
done
```

---

## Final status (2025‑09‑02)
- ✅ HEI‑NW **replay/test** run completes after baselines were generated for the digits‑only RUN_ID.
- ⬜ Code hardening and docs/tests to be merged (see Codex task list).

