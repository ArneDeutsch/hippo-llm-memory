# SMPD — Spatial Map + Replay‑to‑Policy Distillation

**Goal:** Maintain a place/route graph (nodes=contexts/places; edges=transitions). Provide `PLAN(goal)` returning a route scaffold; store successful trajectories and distill **macro policies** (action scripts) via behavior cloning.

## Components to implement

- `hippo_mem/spatial/map.py` — place encoder stub, graph updates, A\*/Dijkstra planner.
- `hippo_mem/spatial/macros.py` — macro library from successful trajectories; scoring head stub.
- Integration glue: expose planning/macro hints to the LLM as structured inputs.

## Acceptance tests (pytest)

- `tests/test_spatial.py`
  - Graph grows deterministically from sequences.
  - Planner finds shortest path on toy grids/graphs.
  - Macro replay reduces steps vs. baseline in a scripted multi‑step task.

## Sample Codex task prompt

> Implement spatial map and macro library. Create `hippo_mem/spatial/{map.py,macros.py}` and tests in `tests/test_spatial.py`. Provide A\* planner and a small macro replay interface. Ensure `make lint` and `make test` pass. Update RUN.md.

## Local training

- Train a small scoring head (LoRA) to prefer macro‑augmented plans; evaluate path optimality and task latency.
