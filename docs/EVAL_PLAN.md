# Evaluation plan notes

See the full plan in [`../EVAL_PLAN.md`](../EVAL_PLAN.md). These highlights
capture the latest pipeline changes.

- **Stable `RUN_ID`**: source `scripts/env_prelude.sh` once; all stages reuse the
  same `RUN_ID` and derived session ids.
- **Difficulty profiles**: dataset generators accept `base` and `hard` profiles.
  Use `dataset_profile=hard` for `episodic_cross` and `episodic_capacity`.
- **Teach → replay → test**: memory runs require an explicit replay step
  (`mode=replay persist=true`) to emit `post_*` and `delta_*` metrics.
- **Gate ablations**: toggle with `episodic.gate.enabled=false`,
  `relational.gate.enabled=false`, or `spatial.gate.enabled=false` and compare
  results in reports.
- **Metric choices**: `semantic` ranks by `EM(raw)`; other suites keep
  `EM(norm)` primary.
- **Consolidation gate flags**: `test_consolidation.py` exposes
  `--uplift-mode`, `--min-uplift`, and `--alpha`. When runs cover two or more
  seeds, the script automatically switches to CI mode and reports the 95%
  confidence interval.
- **Smoke script**: `bash scripts/smoke_n50.sh` runs a full n=50, seed=1337
  cycle and verifies that `post_*` metrics exist.
