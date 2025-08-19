# HEI‑NW — Hippocampal Episodic Index with Neuromodulated Writes

**Goal:** Add an episodic store with k‑WTA‑like sparse keys, content‑addressable recall (kNN / Hopfield‑style completion stub), neuromodulatory write‑gate (surprise + novelty + reward + pin/τ threshold), and prioritized replay hooks.

## Components to implement

- `hippo_mem/episodic/store.py` — FAISS index + SQLite metadata (write/recall/update/delete, key tracking, MLP completion).
- `hippo_mem/episodic/gating.py` — compute **surprise**/**novelty**, combine with reward and a pin override to gate writes.
- `hippo_mem/episodic/replay.py` — prioritized queue mixing gating score, recency and diversity.
- **EpisodicAdapter** — cross‑attention over recalled traces (LoRA‑targeted modules); training script already loads adapters.

## Acceptance tests (pytest)

- `tests/test_episodic.py`
  - One‑shot write→recall EM@1 ≥ 0.95 on toy data.
  - Partial‑cue recall outperforms random under distractors.
  - Gating blocks writes when score ≤ τ and pinned writes always succeed.
  - Deleting a trace removes it from recall.

## Sample Codex task prompt (paste in ChatGPT → Codex → Code)

> Implement the episodic memory store and gating functions. Create `hippo_mem/episodic/{store.py,gating.py,replay.py}` and unit tests in `tests/test_episodic.py`. The store uses FAISS‑CPU (cosine/IP) and SQLite for metadata. Provide `write(key,value)`, recall/query, and deletion. Add a simple completion step (kNN + small MLP). Ensure `make lint` and `make test` pass. Update this RUN.md with any CLI notes.

## Local training

- After merging PRs, run LoRA training with `scripts/train_lora.py` and enable the **EpisodicAdapter** via config.

### Recommended LoRA defaults

| parameter | value |
|-----------|-------|
| learning_rate | 5e-5 |
| gradient_accumulation_steps | 4 |
| max_steps | 500 |
| lora_r | 16 |
| lora_alpha | 16 |
| lora_dropout | 0.1 |

These values follow the recommendations in `research/lora-fine-tuning-overview.md` (rank 16 with α =r, dropout 0.1 and a 5e‑5 learning rate) for small 3–4B models.

## Training & evaluation commands

```bash
# fine-tune with episodic memory
python scripts/train_lora.py run_name=hei_nw \
  episodic.enabled=true relational=false spatial.enabled=false

# sweep episodic evaluation across sizes and seeds
python scripts/eval_bench.py +run_matrix=true preset=memory/hei_nw

# disable replay for an ablation run
python scripts/eval_bench.py preset=memory/hei_nw \
  +ablate=replay.enabled=false

# combined model with all memories
python scripts/train_lora.py run_name=all \
  episodic.enabled=true relational=true spatial.enabled=true
python scripts/eval_bench.py +run_matrix=true preset=memory/all
```

## Notes

- Store now supports update/delete, exposes keys for recall and includes a tiny MLP completion stub.
- WriteGate combines surprise, novelty, reward and a pin flag with thresholding.
- ReplayQueue orders events by gating score, recency and diversity.
- Run `pytest tests/test_episodic.py` to exercise gating and deletion behaviour.
