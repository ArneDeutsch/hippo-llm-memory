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
| learning_rate | 5e-5 |
| gradient_accumulation_steps | 4 |
| max_steps | 500 |
| lora_r | 16 |
| lora_alpha | 16 |
| lora_dropout | 0.1 |

These settings follow the guidance from `research/lora-fine-tuning-overview.md` for 3–4B models (rank 16, α =r, dropout 0.1 and a conservative 5e‑5 learning rate).

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

## Run plan (SGC-RSS)

**Training (short)**

python scripts/train_lora.py
model_name=Qwen/Qwen2-1.5B-Instruct
data.format=jsonl
data.train=data/semantic_200_1337.jsonl
data.val=data/semantic_50_2025.jsonl
lora_r=16 lora_alpha=32
target_modules='["q_proj","k_proj","v_proj","o_proj"]'
max_steps=300 learning_rate=5e-5 gradient_accumulation_steps=8

**Evaluation**

python scripts/eval_model.py suite=semantic preset=memory/sgc_rss n=50 seed=1337

**Acceptance criteria:** metrics show multi-hop accuracy, contradiction rate;
artifacts under `runs/YYYYMMDD/sgc_rss/semantic/`.

