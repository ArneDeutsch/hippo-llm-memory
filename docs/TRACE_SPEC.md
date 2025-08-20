# TRACE_SPEC — Adapter I/O Contract

**Purpose:** Standardize the tensors passed from runtime memory retrieval into adapters.

## 1) Common shapes
- `memory_tokens`: float32, shape `[B, M, d_model]` (packed and projected).
- `memory_mask`: bool, shape `[B, M]` (True for valid positions).
- Optional `meta`: per-example dict with counts, salience stats, timestamps (not used by adapter math, only logging).

## 2) HEI-NW (episodic)
- Retrieval: top-K episodes by FAISS; optional Hopfield completion.
- Pack: concat pooled value embeddings (or short token spans) up to `M_epi`.
- Metadata: salience S, time, reward, pin.

## 3) SGC-RSS (semantic)
- Retrieval: k-hop neighborhood / path embeddings around query entities.
- Pack: node/edge embeddings, relation type encodings up to `M_sem`.

## 4) SMPD (spatial/procedural)
- Retrieval: local neighborhood embeddings + macro/plan steps.
- Pack: sequence of macro embeddings up to `M_spa`.

## 5) Projection
- Each subsystem exposes a `project_to_model(x) -> [*, d_model]`.
- Global cap: `M = min(M_epi+M_sem+M_spa, M_max)`; excess is dropped with logging.

## 6) Adapter contract
- Adapters accept: `hidden_states`, `memory_tokens`, `memory_mask`.
- When `M=0`, adapters must be a no-op (return `hidden_states`).
- All adapters keep residual form: `return hidden_states + f(...)`.

## 7) Config knobs
- `K`, `M_max`, per-subsystem caps, gating threshold `τ`, and insert block index.

End.
