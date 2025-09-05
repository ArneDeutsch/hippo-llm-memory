# TASKS — M2: Harness Sessionization (ScenarioRunner)

## Context Recap
We must run **teach→reset→test** with `session_id` scoping and ensure **writes disabled in test**.


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
Introduce `ScenarioRunner`, propagate `session_id`, and wire to CLI.

## Tasks

### T2.1 — ScenarioRunner
**Add**
- `hippo_eval/harness/scenario_runner.py` with:
  - `run_teach(scenario, algo)` → iterate `teach[]`, writes allowed
  - `reset_conversation()`
  - `run_test(scenario, algo)` → reads only, retrieval enabled, long‑context disabled

**Acceptance**
- Unit tests in `tests/test_scenario_runner.py` simulate a toy store and verify call order.

### T2.2 — CLI wiring
**Edit**
- `scripts/eval_cli.py`
  - New suites: `semantic_closed_book`, `episodic_closed_book`, `spatial_explore`
  - Modes: `teach` (writes), `test` (reads only)

**Acceptance**
```bash
python scripts/eval_cli.py suite=semantic_closed_book preset=memory/sgc_rss n=10 seed=1337 run_id=dev mode=teach persist=true
python scripts/eval_cli.py suite=semantic_closed_book preset=memory/sgc_rss n=10 seed=1337 run_id=dev mode=test  persist=true
```

### T2.3 — Session scoping
**Edit**
- All store write/read APIs accept `session_id` and **filter by default**.

**Acceptance**
- `tests/test_session_scoping.py` writes A and B; retrieval with A never returns B.

