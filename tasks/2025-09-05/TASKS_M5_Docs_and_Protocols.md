# TASKS — M5: Docs & Protocols

## Context Recap
Docs must reflect the new pipeline; legacy references should be removed.


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
Update `EVAL_PLAN.md`, `EVAL_PROTOCOL.md`, and add `DEPRECATIONS.md`.

## Tasks

### T5.1 — EVAL_PLAN.md
- Describe closed‑book schema, sessionization, retrieval packing limits, telemetry fields, and acceptance gates.

### T5.2 — EVAL_PROTOCOL.md
- Provide step‑by‑step commands (n=10 smoke then scale), including teach→test runs.

### T5.3 — DEPRECATIONS.md
- List all removed datasets, generators, configs; point to `legacy-datasets-v1` tag.

**Acceptance**
- Docs lint clean; a new contributor can reproduce a closed‑book run in <15 minutes.

