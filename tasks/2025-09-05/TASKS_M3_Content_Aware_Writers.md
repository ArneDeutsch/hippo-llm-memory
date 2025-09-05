# TASKS — M3: Content‑Aware Writers & Retrievers

## Context Recap
Replace placeholder writes with content‑aware implementations aligned with the algorithms.


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
Implement HEI‑NW, SGC‑RSS, and SMPD writers/retrievers.

## Tasks

### T3.1 — HEI‑NW (Episodic)
**Implement**
- `hippo_mem/episodic/store.py`: write `(session_id, key, text, ts)`; key via k‑WTA over embeddings.
- `hippo_mem/episodic/retrieve.py`: cosine/overlap top‑k; pack ≤ 256 tokens.

**Acceptance**
- `tests/test_episodic_writer.py` validates keys/recall on fixtures.
- Memory hit‑rate > 0 on `semantic_closed_book` (n=10).

### T3.2 — SGC‑RSS (Relational)
**Implement**
- `hippo_mem/relational/ie_rules.py`: simple patterns for synthetic templates.
- `hippo_mem/relational/kg.py`: per‑session KG with evidence counters.
- `hippo_mem/relational/retrieve.py`: k‑hop subgraph → compact text.

**Acceptance**
- `tests/test_relational_writer.py` passes.
- Multi‑hop queries in `semantic_closed_book` show EM uplift vs closed‑book baseline.

### T3.3 — SMPD (Spatial)
**Implement**
- `hippo_mem/spatial/graph.py`: adjacency with costs.
- `hippo_mem/spatial/planner.py`: Dijkstra/A*; stringify to `UDLR`.
- Teach: parse `OBS:` lines; Test: plan and return action string.

**Acceptance**
- `tests/test_spatial_planner.py` passes (shortest paths on toy mazes).
- Success rate > 0 on `spatial_explore` (n=10).

