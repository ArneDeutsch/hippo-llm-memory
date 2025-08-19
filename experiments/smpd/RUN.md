# SMPD — Spatial Map + Replay‑to‑Policy Distillation

**Goal:** Maintain a place/route graph (nodes=contexts/places; edges=transitions). Provide `PLAN(goal)` returning a route scaffold; store successful trajectories and distill **macro policies** (action scripts) via behavior cloning.

## Components to implement

- `hippo_mem/spatial/map.py` — `PlaceGraph.observe()` inserts places and
  `plan()` computes paths via A*/Dijkstra.
- `hippo_mem/spatial/macros.py` — `MacroLib.store()` records trajectories
  and `suggest()` returns top‑k macros (shortest first).
- Integration glue: expose planning/macro hints to the LLM as structured inputs.

## Acceptance tests (pytest)

- `tests/test_spatial.py`
  - Graph grows deterministically from sequences.
  - Planner finds shortest path on toy grids/graphs.
  - Macro replay reduces steps vs. baseline in a scripted multi‑step task.

## Sample Codex task prompt

> Implement spatial map and macro library. Create `hippo_mem/spatial/{map.py,macros.py}` and tests in `tests/test_spatial.py`. Provide A\* planner and a small macro replay interface. Ensure `make lint` and `make test` pass. Update RUN.md.

## CLI notes

- Build a graph by sequentially calling `PlaceGraph.observe(context)` or
  explicitly `connect(a, b, cost)`.
- Plan routes with `PlaceGraph.plan(start, goal, method="astar"|"dijkstra")`.
- Record successful trajectories via `MacroLib.store(name, traj)` and
  retrieve suggestions with `MacroLib.suggest(start, goal, k)`.

## Local training

- Train a small scoring head (LoRA) to prefer macro‑augmented plans; evaluate path optimality and task latency.

### Recommended LoRA defaults

| parameter | value |
|-----------|-------|
| learning_rate | 2e-4 |
| gradient_accumulation_steps | 16 |
| max_steps | 500 |
| lora_r | 8 |
| lora_alpha | 16 |
| lora_dropout | 0.05 |

Derived from the Alpaca‑LoRA setup (rank 8, alpha 16, dropout 0.05) with the QLoRA learning rate of 2e‑4; gradient accumulation emulates a larger batch on a 12 GB GPU.

## Training & evaluation commands

```bash
# fine-tune with spatial memory
python scripts/train_lora.py run_name=smpd \
  spatial.enabled=true episodic.enabled=false relational=false

# sweep spatial evaluation across sizes and seeds
python scripts/eval_bench.py +run_matrix=true preset=memory/smpd

# ablate macro suggestions
python scripts/eval_bench.py preset=memory/smpd \
  +ablate=spatial.macros=false

# combined model with all memories
python scripts/train_lora.py run_name=all \
  episodic.enabled=true relational=true spatial.enabled=true
python scripts/eval_bench.py +run_matrix=true preset=memory/all
```
