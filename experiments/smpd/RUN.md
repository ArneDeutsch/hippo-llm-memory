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
| learning_rate | 5e-5 |
| gradient_accumulation_steps | 4 |
| max_steps | 500 |
| lora_r | 16 |
| lora_alpha | 16 |
| lora_dropout | 0.1 |

Derived from recommendations in `research/lora-fine-tuning-overview.md` (rank 16 with α =r, dropout 0.1 and a modest 5e‑5 learning rate); gradient accumulation of 4 keeps the effective batch small while fitting a 12 GB GPU.

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

## Run plan (SMPD)

**Training (short)**

python scripts/train_lora.py
model_name=Qwen/Qwen2-1.5B-Instruct
data_format=jsonl train_files='["data/spatial_200_1337.jsonl"]' val_files='["data/spatial_50_2025.jsonl"]'
lora_r=16 lora_alpha=32
target_modules='["q_proj","k_proj","v_proj","o_proj"]'
max_steps=300 learning_rate=5e-5 gradient_accumulation_steps=8
fusion_insert_block_index=-4

**Evaluation**

python scripts/eval_model.py suite=spatial preset=memory/smpd n=50 seed=1337

## Acceptance criteria

- Logs contain "Adapter fusion attached at block" and trainable params > 0.
- Metrics include path success and suboptimality.
- Artifacts under `runs/YYYYMMDD/smpd/spatial/`.


