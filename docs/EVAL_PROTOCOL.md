# Evaluation protocol quick guide

The full, copy-pasteable runbook lives in the repository root at
[`EVAL_PROTOCOL.md`](../EVAL_PROTOCOL.md). This shorter note highlights the
updates from the pipeline review.

## 0) Stable run identifier

Set a single `RUN_ID` once per experiment and source the shared prelude:

```bash
export RUN_ID=my_run
source scripts/env_prelude.sh
```

The prelude mirrors `RUN_ID` into `DATE` for back‑compat and defines helper
paths such as `$RUNS`, `$REPORTS`, and `$STORES`.

## 1) Dataset difficulty profiles

Datasets ship with `base` and `hard` profiles. Use `dataset_profile=hard` for
suites like `episodic_cross` or `episodic_capacity` to avoid saturation.

```bash
# Recommended flags for sensitive suites
suite=episodic_cross dataset_profile=hard --strict-telemetry
suite=episodic_capacity dataset_profile=hard --strict-telemetry
```

```bash
make datasets DATE="$RUN_ID"
```

## 2) Teach → replay → test

Memory evaluations run in three phases, all under the same `RUN_ID` and a
stable `session_id` (e.g., `hei_$RUN_ID`):

1. `mode=teach persist=true` — write traces to `$STORES`.
2. `mode=replay persist=true` — optional rehearsal; writes `post_*` and
   `delta_*` metrics.
3. `mode=test` — load stores and answer queries without writing.

## 3) New CLI flags

`test_consolidation.py` now exposes uplift gating controls:

```bash
--uplift-mode [fixed|ci]  # default: fixed
--min-uplift FLOAT        # default: 0.05 when fixed
--alpha FLOAT             # CI significance when using 'ci'
```

## 4) Gate ablations

Toggle gates per memory with `episodic.gate.enabled=false`,
`relational.gate.enabled=false`, or `spatial.gate.enabled=false` and compare
the resulting `Gate ON vs OFF` table in reports.

## 5) Metric choices per suite

- **semantic**: `EM(raw)` is the primary metric. Reports rank presets by
  `EM(raw)` and show `EM(norm)` alongside.
- **other suites**: `EM(norm)` remains primary; `EM(raw)` is secondary.

## 6) Smoke script

`bash scripts/smoke_n50.sh` runs a fast end‑to‑end test at `n=50`,
`seed=1337` and confirms that `post_*` and `delta_*` metrics appear.

## 7) Troubleshooting Step 9

Step 9 of the full protocol runs `test_consolidation.py`. If it aborts with
`EM uplift < ...`:

- ensure `mode=replay persist=true` was executed beforehand;
- lower the threshold via `--min-uplift 0.05`; or
- collect multiple seeds and switch to `--uplift-mode ci`.

This produces `post_*` metrics needed for uplift tables.
