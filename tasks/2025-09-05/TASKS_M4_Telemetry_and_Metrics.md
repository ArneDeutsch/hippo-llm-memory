# TASKS — M4: Telemetry & Metrics

## Context Recap
We need to attribute gains to memory: hit‑rate, pack size, latency, uplift.


> **Path hints (adjust to your repo):**
> - Datasets: `data/{semantic,semantic_hard,episodic*,spatial*}`
> - Generators: `hippo_eval/tasks/generators.py`, `hippo_eval/tasks/spatial/generator.py`
> - Harness & eval: `hippo_eval/eval/harness.py`, `hippo_eval/harness/*`, `scripts/eval_cli.py`
> - Stores: `hippo_mem/{episodic,relational,spatial}/*`, `hippo_eval/stores/*`
> - Reporting: `hippo_eval/reporting/*`, `reports/*`
> - Configs: `configs/datasets/*`, `configs/presets/*`
> - Docs: `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, `DESIGN.md`, `MILESTONE_9_PLAN.md`
> - Artifacts (your run): `runs/run_20250904/`, `reports/run_20250904/`


## Goal
Add per‑item logs and report aggregates; update report tables.

## Tasks

### T4.1 — Per‑item telemetry
**Edit/Add**
- `hippo_eval/metrics.py`, `hippo_eval/harness/scenario_runner.py`:
  - Log `retrieval.hit`, `retrieval.k`, `retrieval.tokens`, `retrieval.latency_ms`, `gen.latency_ms`, `scores.em/f1`.

**Acceptance**
- JSON lines per item include fields; schema tested in `tests/test_metrics_schema.py`.

### T4.2 — Aggregates & reports
**Edit/Add**
- `hippo_eval/reporting/*` to compute and render:
  - Closed‑book baseline EM/F1
  - Memory EM/F1
  - Uplift
  - Hit‑rate
  - Avg pack tokens
  - Latencies
  - SMPD success/optimality

**Acceptance**
- `reports/run_*/index.md` shows new tables.
