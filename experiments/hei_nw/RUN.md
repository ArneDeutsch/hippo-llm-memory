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

## Run plan (HEI-NW)

**Training (short)**

python scripts/train_lora.py
model_name=Qwen/Qwen2-1.5B-Instruct
data_format=jsonl train_files='["data/episodic_200_1337.jsonl"]' val_files='["data/episodic_50_2025.jsonl"]'
lora_r=16 lora_alpha=32
target_modules='["q_proj","k_proj","v_proj","o_proj"]'
max_steps=300 learning_rate=5e-5 gradient_accumulation_steps=8
fusion_insert_block_index=-4 replay.enabled=true

**Evaluation (real harness)**

python scripts/eval_model.py suite=episodic preset=memory/hei_nw n=50 seed=1337 replay.cycles=1
python scripts/report.py --date YYYYMMDD

## Acceptance criteria
- Logs contain "Adapter fusion attached at block" and trainable params > 0.
- `runs/YYYYMMDD/hei_nw/episodic/metrics.json` exists with EM/F1 fields.
- `reports/YYYYMMDD/episodic/summary.md` includes HEI-NW rows.

## Notes

- Store now supports update/delete, exposes keys for recall and includes a tiny MLP completion stub.
- WriteGate combines surprise, novelty, reward and a pin flag with thresholding.
- ReplayQueue orders events by gating score, recency and diversity.
- Run `pytest tests/test_episodic.py` to exercise gating and deletion behaviour.
