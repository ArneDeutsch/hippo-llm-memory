# TASKS — M6: QA & Smoke

## Context Recap
We want tight feedback loops before scaling.


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
Add tiny smoke runs in CI and local scripts.

## Tasks

### T6.1 — CI smoke
**Add**
- `.github/workflows/ci.yml` (or equivalent) to run:
  ```bash
  python scripts/gen_closed_book.py --suite semantic_closed_book --n 10 --seed 1337
  python scripts/eval_cli.py suite=semantic_closed_book preset=memory/sgc_rss n=10 seed=1337 run_id=ci mode=teach persist=true
  python scripts/eval_cli.py suite=semantic_closed_book preset=memory/sgc_rss n=10 seed=1337 run_id=ci mode=test  persist=true
  ```
**Accept**
- CI green; artifacts uploaded; uplift ≥ 0 on at least one suite.

### T6.2 — Local smoke script
**Add**
- `scripts/smoke_closed_book.sh` automating the above with clear logs.

