# TASKS — M7: Scale & Reporting

## Context Recap
After smoke, scale to n=50/100 and build richer dashboards.


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
Scale runs, add ablations, and enhance reports.

## Tasks

### T7.1 — Scale
- Increase `n` to 50/100; maintain `seed=1337` for reproducibility.
- Ensure runtime budgets are acceptable.

### T7.2 — Ablations
- Long‑context baseline (token‑capped).
- Retrieval k sweep (k=1/3/5) and pack size sweep (256/384/512).
- Namespace stress: withhold `session_id` filter on 10% items to quantify interference.

### T7.3 — Reporting
- Add per‑suite and overall dashboards plotting EM vs pack tokens, and latency histograms.

**Acceptance**
- Reports include uplift curves and ablation summaries; conclusions clearly attributable to memory.

