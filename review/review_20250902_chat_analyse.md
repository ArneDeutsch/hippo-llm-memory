# Review: Baseline Aggregation Failures & RUN\_ID/Date Inconsistencies — Root Cause & Fix Plan

**Date:** 2025-09-02
**Scope:** EVAL pipeline (baselines → roll-up → memory runs), harness & scripts, protocol

---

## 1) Executive Summary

We diagnosed and fixed two primary, systemic issues that caused repeated baseline aggregation failures and preflight aborts:

1. **Hydra list quoting bug** — Passing Hydra lists as *quoted strings* (e.g., `presets="[a,b]"`) led the harness to iterate characters as items, writing results into nonsense directories.
   **Fix:** Use *unquoted* Hydra lists (e.g., `presets=[a,b]`) **and** add type guards in code to fail fast if strings are provided.

2. **`date` normalization mismatch (RUN\_ID drift)** — The harness **normalizes** `date` to `YYYYMMDD(_HHMM)?` and writes to `runs/<normalized>`, while `scripts/run_baselines.py` used the CLI `--date` **verbatim** to search `runs/<verbatim>/baselines`. If the user supplied a decorated ID like `20250902_50_1337_2025`, the aggregator looked in a path the harness never used.
   **Fix now:** Re-run with `RUN_ID=20250902` and aggregate at the normalized date; this succeeded.
   **Permanent fix:** Replace the fragile “date as directory key” with a **single, explicit `run_id`** argument used verbatim for all IO paths (with a strict slug regex and consistent validation across harness, aggregator, and report). Keep `date` only as metadata.

These two defects fully explain the repeated errors (“no baseline metrics found”, missing `baselines/metrics.csv`, preflight failing on baselines).

---

## 2) Evidence and Code Pointers

### 2.1 Hydra list quoting bug

* **Symptom:** Matrix logs showed “run 1/12 … run 12/12”, but aggregator found no files.
* **Mechanism:** With `presets="[baselines/core,...]"`, `cfg.presets` is a string; Python iterates characters (`'[', 'b', 'a', ...`).
* **Affected code:** `hippo_mem/eval/harness.py::main` loops `for preset in presets:`; if `presets` is a string, incorrect per-char directories are produced.
* **Fix:** Use unquoted lists in the protocol; add type guards that raise if `presets`, `tasks`, `n_values`, or `seeds` are strings rather than lists.

### 2.2 `date` normalization mismatch

* **Symptom:** `scripts/run_baselines.py --date 20250902_50_1337_2025` raised `FileNotFoundError: runs/<id>/baselines not found`.
* **Mechanism:** `harness._date_str()` normalizes to `YYYYMMDD(_HHMM)?` and **changes** the folder name (e.g., to `20250902`). The aggregator used the unnormalized CLI `--date` and searched the wrong directory.
* **Affected code:**

  * Normalize in `hippo_mem/eval/harness.py::main` via `_date_str`.
  * No normalization in `scripts/run_baselines.py::main` (uses `args.date` verbatim).
* **Fix now (operational):** Use `RUN_ID=20250902` (normalized) for both steps; aggregation succeeded and `runs/20250902/baselines/metrics.csv` exists.
* **Permanent fix:** Introduce `--run_id` used **verbatim** for all pathing; deprecate `date` for IO pathing. Add a single slug validator and consistent usage across harness, run\_baselines, report, and tests.

---

## 3) Changes Applied During This Session

* **Protocol corrections:**

  * Switched to **unquoted** Hydra lists in all commands.
  * Used a **normalized** ID (`RUN_ID=20250902`) consistently for baselines and aggregation.
  * Successfully produced `runs/20250902/baselines/metrics.csv` (roll-up present).

* **Validated behavior:**

  * Matrix runs executed to completion (12/12).
  * Aggregator succeeded at normalized path.

---

## 4) Proposed Simplified & Robust Run-ID Schema

### 4.1 Goals

* Zero ambiguity between producer (harness) and consumer (aggregator/report).
* Human-readable, taggable identifiers that don’t break path logic.
* Fail-fast validation on malformed inputs.

### 4.2 The schema

* **`run_id` (new, required for IO):** a strict slug used verbatim as `runs/<run_id>/...` and `reports/<run_id>/...`.

  * **Regex:** `^[A-Za-z0-9._-]{3,64}$`
  * Examples: `20250902`, `20250902_50_1337_2025`, `2025-09-02.hei-matrix`.
* **`date` (optional metadata):** ISO date string recorded in metadata files but **never** used to construct paths.

### 4.3 Backwards compatibility

* If only `date` is provided: `run_id := _date_str(date)` (current behavior) and **emit a deprecation warning**.
* If both provided: use `run_id` for IO; record `date` in metadata.
* Remove normalization from IO paths entirely.

---

## 5) Protocol (to be written into `EVAL_PROTOCOL.md`)

**Prelude**

```bash
set -euo pipefail
export RUN_ID=20250902_50_1337_2025     # any slug matching ^[A-Za-z0-9._-]{3,64}$
source scripts/_env.sh                  # defines RUNS, REPORTS, STORES, MODEL, HEI_SESSION_ID
export REPLAY_CYCLES=3
mkdir -p "$RUNS" "$REPORTS" "$STORES"
```

**1) Baselines (matrix) → roll-up**

```bash
python scripts/eval_model.py +run_matrix=true \
  run_id="$RUN_ID" \
  presets=[baselines/core,baselines/span_short,baselines/rag,baselines/longctx] \
  tasks=[episodic,episodic_multi,episodic_cross,episodic_capacity,semantic,spatial] \
  n_values=[50] \
  seeds=[1337,2025] \
  dataset_profile=hard \
  compute.pre_metrics=true \
  model="$MODEL"

python scripts/run_baselines.py --run-id "$RUN_ID"
test -f "runs/$RUN_ID/baselines/metrics.csv"
```

**2) Memory runs (teach → replay → test, consistent store/session)**

```bash
for SEED in 1337 2025; do
  for SUITE in episodic episodic_multi episodic_cross episodic_capacity semantic spatial; do
    python scripts/eval_model.py suite="$SUITE" preset=memory/hei_nw \
      run_id="$RUN_ID" n=50 seed="$SEED" mode=teach persist=true \
      store_dir="$STORES" session_id="$HEI_SESSION_ID" \
      compute.pre_metrics=true strict_telemetry=true model="$MODEL"

    python scripts/eval_model.py suite="$SUITE" preset=memory/hei_nw \
      run_id="$RUN_ID" n=50 seed="$SEED" mode=replay \
      store_dir="$STORES" session_id="$HEI_SESSION_ID" \
      replay.cycles="$REPLAY_CYCLES" strict_telemetry=true model="$MODEL"

    python scripts/eval_model.py suite="$SUITE" preset=memory/hei_nw \
      run_id="$RUN_ID" n=50 seed="$SEED" mode=test \
      store_dir="$STORES" session_id="$HEI_SESSION_ID" \
      strict_telemetry=true model="$MODEL"
  done
done
```

**3) Reports**

```bash
python scripts/report.py --run-id "$RUN_ID"
test -f "reports/$RUN_ID/index.md"
```

> Note: **All lists are unquoted.** All tools use **`run_id`**. `date` is not used for IO.

---

## 6) Additional Hardening

1. **Type guards** for Hydra list/sequence inputs (`presets`, `tasks`, `n_values`, `seeds`) → raise with actionable messages.
2. **Aggregator**: better error strings (show what *was* found); accept only the `run_id` parameter.
3. **Preflight message**: when baseline roll-up missing, suggest the exact aggregator command.
4. **Tests**:

   * Round-trip `run_id` across harness → aggregator → report.
   * Reject invalid `run_id` strings (regex test).
   * Detect quoted-string lists and assert failure with the new message.
5. **Docs**: EVAL\_PROTOCOL.md updated; README quick-start updated with valid `run_id` examples.

---

## 7) Conclusion

The repeating failures stemmed from (1) Hydra list quoting and (2) ambiguous/normalized date usage. We’ve now (a) corrected the protocol, (b) successfully built the baseline roll-up, and (c) designed a simple, robust **`run_id`** scheme that eliminates the mismatch class entirely. The attached Codex tasks implement this end-to-end with tests to prevent regressions.

