# TASKS — M1: Data Model & Generators (Closed‑Book)

## Context Recap
We introduce a **closed‑book** schema and generators for semantic/episodic and spatial explore→exploit.


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
Implement `ClosedBookScenario` schema, IO loaders, and generators:

- `semantic_closed_book`, `episodic_closed_book`
- `spatial_explore`

## Tasks

### T1.1 — Schema & loader
**Add**
- `hippo_eval/data/schema.py` with `ClosedBookScenario` (Pydantic/dataclass).
- `hippo_eval/data/io.py` with `load_closed_book_jsonl(path)`.

**Acceptance**
```bash
pytest -q tests/test_schema_closed_book.py::test_load_scenarios
```

### T1.2 — Semantic/Episodic closed‑book generator
**Add**
- `scripts/gen_closed_book.py`
  - `--suite semantic_closed_book|episodic_closed_book`
  - `--n`, `--seed`
  - emits `data/<suite>/<n>_<seed>.jsonl`

**Acceptance**
- File exists and validates via loader.
- Items do not repeat entities across `session_id` unless deliberate.

### T1.3 — Spatial explore generator
**Add**
- `scripts/gen_spatial_explore.py`
  - Emits teach observations like `OBS: (x,y)->(u,v)`
  - Ensures test start/goal solvable and shortest path unique.

**Acceptance**
```bash
python scripts/gen_spatial_explore.py --n 10 --seed 1337
pytest -q tests/test_gen_spatial_explore.py
```

### T1.4 — Tests
**Add**
- `tests/test_schema_closed_book.py`
- `tests/test_gen_closed_book.py`
- `tests/test_gen_spatial_explore.py`

**Notes**
- Keep grids tiny (e.g., 5x5) for fast CI.

