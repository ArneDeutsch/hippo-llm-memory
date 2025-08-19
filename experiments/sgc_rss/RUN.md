# SGC‑RSS — Schema‑Guided Consolidation with Relational Semantic Store

**Goal:** Build a lightweight tuple extractor and a semantic graph store. At inference, retrieve top‑k subgraphs and top‑k episodes, and fuse via dual cross‑attention adapters. Implement schema‑fit routing for consolidation priority.

## Components to implement

- `hippo_mem/relational/tuples.py` — starter rule‑/prompt‑based extraction producing (entity, relation, context, time) + confidence.
- `hippo_mem/relational/kg.py` — NetworkX + SQLite for a small KG; embeddings for nodes/edges; subgraph retrieval by query.
- `hippo_mem/relational/adapter.py` — adapter that cross‑attends to encoded subgraphs.

## Acceptance tests (pytest)

- `tests/test_relational.py`
  - Extractor yields tuples with ≥ 0.9 precision on small fixtures.
  - KG retrieval returns relevant subgraphs for simple multi‑hop queries.
  - Dual‑path fuse returns deterministic merge of episodic and semantic hits for a toy question.

## Sample Codex task prompt

> Implement a minimal relational store and tuple extractor. Create `hippo_mem/relational/{tuples.py,kg.py,adapter.py}` and tests in `tests/test_relational.py`. Use NetworkX and simple embedding averages. Provide subgraph retrieval API. Ensure `make lint` and `make test` pass. Update RUN.md with notes.

## Local training

- Enable **RelationalAdapter** in configs and fine‑tune with interleaved replay batches.

### Recommended LoRA defaults

Use the same baseline hyperparameters as HEI‑NW:

| parameter | value |
|-----------|-------|
| learning_rate | 2e-4 |
| gradient_accumulation_steps | 16 |
| max_steps | 500 |
| lora_r | 8 |
| lora_alpha | 16 |
| lora_dropout | 0.05 |

These settings follow the Alpaca‑LoRA recipe (rank 8, alpha 16, dropout 0.05) and QLoRA’s default 2e‑4 learning rate.

## Training & evaluation commands

```bash
# fine-tune with relational memory
python scripts/train_lora.py run_name=sgc_rss \
  relational=true episodic.enabled=false spatial.enabled=false

# sweep semantic evaluation across sizes and seeds
python scripts/eval_bench.py +run_matrix=true preset=memory/sgc_rss

# ablate schema fast-track routing
python scripts/eval_bench.py preset=memory/sgc_rss \
  +ablate=relational.schema_fasttrack=false

# combined model with all memories
python scripts/train_lora.py run_name=all \
  episodic.enabled=true relational=true spatial.enabled=true
python scripts/eval_bench.py +run_matrix=true preset=memory/all
```
