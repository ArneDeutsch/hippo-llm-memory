# HEI‑NW — Hippocampal Episodic Index with Neuromodulated Writes

**Goal:** Add an episodic store with k‑WTA‑like sparse keys, content‑addressable recall (kNN / Hopfield‑style completion stub), neuromodulatory write‑gate (surprise + novelty + reward), and prioritized replay hooks.

## Components to implement

- `hippo_mem/episodic/store.py` — FAISS index + SQLite metadata (write/recall/update/delete).
- `hippo_mem/episodic/gating.py` — compute **surprise** from logprobs/entropy and **novelty** from embedding distance; threshold to write.
- `hippo_mem/episodic/replay.py` — prioritized queue (salience, recency, diversity).
- **EpisodicAdapter** — cross‑attention over recalled traces (LoRA‑targeted modules); training script already loads adapters.

## Acceptance tests (pytest)

- `tests/test_episodic.py`
  - One‑shot write→recall EM\@1 ≥ 0.95 on toy data.
  - Partial‑cue recall outperforms random under distractors.
  - Gating writes only when `S > τ` and can be forced by a “pin” flag.

## Sample Codex task prompt (paste in ChatGPT → Codex → Code)

> Implement the episodic memory store and gating functions. Create `hippo_mem/episodic/{store.py,gating.py,replay.py}` and unit tests in `tests/test_episodic.py`. The store uses FAISS‑CPU (cosine/IP) and SQLite for metadata. Provide `write(key,value)`, `recall(query,k)`, and deletion. Add a simple completion step (kNN + small MLP). Ensure `make lint` and `make test` pass. Update this RUN.md with any CLI notes.

## Local training

- After merging PRs, run LoRA training with `scripts/train_lora.py` and enable the **EpisodicAdapter** via config.
