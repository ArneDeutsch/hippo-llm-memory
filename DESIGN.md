# 1) Purpose & scope

This document specifies the end‑to‑end design for hippocampus‑inspired memory extensions for small open LLMs trained/adapted with LoRA/QLoRA on a single 12 GB GPU. It translates the research workflow (Pass 1–4) into concrete software modules, data schemas, training/eval procedures, and Codex tasking.

**Experiments:**

- **HEI‑NW** — Episodic index + neuromodulated writes + prioritized replay.
- **SGC‑RSS** — Schema‑guided consolidation + relational/graph semantic store.
- **SMPD** — Spatial map + replay‑to‑policy macro distillation.

# 2) System assumptions & constraints

- **Hardware:** 1× NVIDIA GPU (12 GB), Ubuntu Linux.
- **Models:** small, open‑weights causal LMs (e.g., Llama‑3.2‑3B, Phi‑3‑Mini, Qwen2‑1.5B) with **QLoRA (4‑bit NF4)**, gradient checkpointing, seq‑len ≤1024.
- **Frameworks:** PyTorch, Hugging Face Transformers, TRL, PEFT, bitsandbytes, FAISS‑CPU, Hydra, NetworkX, pytest.
- **Codex (web):** runs in an ephemeral CPU container, installs `codex-env/requirements.txt`, runs `make lint`/`make test`, opens PRs.
- **Non‑goals:** Full pretraining; multi‑GPU distributed training; online reinforcement learning.

# 3) High‑level architecture

```
                  ┌──────────────────────────────────────────────────────┐
                  │                      LLM core                        │
                  │  (HF Transformers; LoRA/QLoRA adapters inserted)    │
                  └───────────┬───────────────────────────────┬──────────┘
                              │                               │
                         [Episodic Adapter]              [Relational Adapter]
                              │                               │
             ┌────────────────┴──────────────┐        ┌───────┴────────────────┐
             │   HEI‑NW Episodic Store      │        │   SGC‑RSS Semantic KG  │
             │  (FAISS + SQLite + Replay)   │        │  (NetworkX + Embeds)   │
             └────────────────┬──────────────┘        └───────────┬────────────┘
                              │                                   │
                         [Spatial Adapter / Tool]                 │
                              │                                   │
                         SMPD Map + Macros  <──────────────────────┘
```

Shared library `hippo_mem/` provides retrieval, adapters, and utils. Each experiment adds its own configs and eval tasks.

# 4) Data structures

## 4.1 Episodic (HEI‑NW)

- `DGKey`: sparse key (indices + values; CSR‑like), `dim=d_k`.
- `TraceValue`: `{tokens_span_ref, entity_slots(who,what,where,when), state_sketch, salience_tags}`.
- `AssocStore`: FAISS index (IP/cosine), optional small MLP for completion.
- `ReplayQueue` item: `{key_id, S (salience), recency, diversity_sig, timestamp}`.

## 4.2 Relational (SGC‑RSS)

- Tuple: `(head, relation, tail, context, time, conf, provenance)`.
- KG: `NetworkX` multigraph with node/edge embeddings; SQLite for persistence.
- `SchemaIndex`: prototypes/frames with slot types; similarity scores for schema‑fit routing.

## 4.3 Spatial (SMPD)

- `PlaceGraph`: nodes = place/context embeddings; edges = transitions with weights `{cost, success_prob}`.
- `MacroLib`: list of `{signature, steps, success_stats}`; small scoring head during inference.

# 5) Core algorithms

## 5.1 Write‑gate (neuromodulatory analog)

`S = α · surprise + β · novelty + γ · reward + δ · pin`.

- **surprise:** `−log p(next_token)` or local entropy from LM logits.
- **novelty:** `1 − max cos(query, catalog_keys)`.
- **reward/pin:** explicit task flags. Write iff `S > τ`.

## 5.2 Recall (content‑addressable)

1. Build query from current hidden/residual state.
2. Obtain sparse key via k‑WTA projection (or plain dense embedding v0).
3. FAISS KNN → top‑k traces; optional MLP completion → memory tokens.
4. Cross‑attend via **EpisodicAdapter**.

## 5.3 Prioritized replay (CA2‑like)

- Priority = mixture of `S`, recency, and diversity (reduce gradient overlap).
- Offline worker yields interleaved batches to the trainer for consolidation.

## 5.4 Relational retrieval & schema routing

- Tuple extractor → KG insert with provenance.
- Schema score determines **fast‑track** consolidation vs. extended replay.
- Dual‑path inference: top‑k subgraphs ⊕ top‑k episodes → fused cross‑attention.

## 5.5 Spatial planning & macro distillation

- Place encoder groups contexts; planner = A\*/Dijkstra on `PlaceGraph`.
- Successful trajectories → behavior cloning into `MacroLib`.

# 6) Adapter design (LoRA/QLoRA)

- **Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, and optionally FFN `{up,down}` in adapter blocks only.
- **Defaults:** r=16, α=32, dropout=0.05; load base in 4‑bit NF4; gradient checkpointing on.
- **Placement:**
  - **EpisodicAdapter:** after transformer block N (configurable), adds a cross‑attn over retrieved traces.
  - **RelationalAdapter:** parallel cross‑attn over KG subgraph encodings; gating head merges outputs.
  - **Spatial:** either a lightweight cross‑attn to plan/macro embeddings or tool‑call interface producing structured hints.

# 7) Configuration (Hydra)

Example keys:

```yaml
model:
  name: llama32-3b
  dtype: bfloat16
train:
  load_in_4bit: true
  gradient_checkpointing: true
  micro_batch: 1
  grad_accum: 8
  seq_len: 1024
lora:
  r: 16
  alpha: 32
  dropout: 0.05
memory:
  episodic: {k: 8, metric: cosine, write_threshold: 0.7}
  relational: {topk_subgraphs: 4, tuple_conf_min: 0.8}
  spatial: {planner: astar, heuristic: manhattan}
```

# 8) Module APIs (summary)

## 8.1 Episodic

```python
class EpisodicStore:
    def write(self, key: np.ndarray, value: dict) -> int: ...
    def recall(self, query: np.ndarray, k: int) -> list[dict]: ...
    def delete(self, key_id: int) -> None: ...

class WriteGate:
    def score(self, surprise: float, novelty: float, reward: bool, pin: bool) -> float: ...
```

## 8.2 Relational

```python
class TupleExtractor:
    def extract(self, text: str) -> list[Tuple]: ...

class KG:
    def upsert(self, tuples: list[Tuple]) -> None: ...
    def retrieve(self, query: str, k: int) -> Subgraph: ...
```

## 8.3 Spatial

```python
class PlaceGraph:
    def observe(self, context: dict) -> int: ...
    def plan(self, start: int, goal: int) -> list[int]: ...

class MacroLib:
    def add(self, trajectory: list[dict]) -> str: ...
    def suggest(self, context: dict, topk: int = 3) -> list[dict]: ...
```

# 9) Training & consolidation

- **Phase A (adapters only):** freeze base LM; train Episodic/Relational adapters with synthetic tasks (Pass‑4) and small curated corpora.
- **Phase B (replay distillation):** run prioritized replay → interleaved batches; optionally allow low‑LR updates to select base layers.
- **Checkpoints:** save LoRA weights per adapter; `export_adapter.py` can merge/unmerge.

# 10) Evaluation (harness)

- **Episodic:** EM/F1 from partial cues; robustness vs. distractors; latency.
- **Semantic:** multi‑hop QA against KG; contradiction rate; schema‑fit acceleration.
- **Spatial:** path success, suboptimality, steps.
- **Procedural:** macro reuse rate; steps‑to‑solve; tool‑call latency.
- **Ablations:** remove sparsity, completion, gate, replay scheduler, schema routing, or macros.
- **Compute:** FLOPs saved vs. long‑context; KV‑cache memory vs. baseline.

# 11) Logging, provenance, and rollback

- Every write stores `{text_span, doc_id, time, conf, source}`.
- Provide `delete_by_provenance()` and snapshot/restore of stores.
- Structured logs: JSON lines in `runs/<date>/<exp>/events.jsonl`.

# 12) Risks & mitigations

- **Noisy writes (low precision gating):** raise τ, add user “pin”, post‑hoc pruning.
- **Interference during replay:** diversity‑aware scheduler; cap batch similarity.
- **Extractor drift (KG):** confidence thresholds, rollback, tests with golden tuples.
- **Map explosion:** node merging by similarity; TTL for stale places.

# 13) Milestones & PRs (Codex‑friendly)

1. **Scaffold & CI** (PR‑1): repo tree, lint/test green.
2. **Episodic v0** (PR‑2): store + gating + tests.
3. **Relational v0** (PR‑3): tuples + KG + tests.
4. **Spatial v0** (PR‑4): map + planner + tests.
5. **Training script** (PR‑5): QLoRA trainer + dry‑run test.
6. **Eval harness** (PR‑6): metrics & fixtures; ablation toggles.
7. **Adapters wired** (PR‑7..9): Episodic/Relational/Spatial adapters + configs.

# 14) Example Codex task snippets

- *“Implement EpisodicStore with FAISS‑CPU, add tests, ensure make lint/test pass, open PR ‘feat(episodic): v0 store+gating’.”*
- *“Create KG with NetworkX & embeddings; retrieval API; tests for simple multi‑hop; PR ‘feat(relational): kg v0’.”*
- *“Add A* planner to PlaceGraph; tests on toy grid; PR ‘feat(spatial): planner v0’.”\*

# 15) Reproducibility

- Fix random seeds in tests; persist FAISS/SQLite snapshots under `tests/data/tmp` during CI.
- Record exact model IDs and config hashes in run metadata.

# 16) ASCII appendix

## 16.1 HEI‑NW dataflow

```
Tokens → LLM → Query
              │
      k‑WTA/Embed → EpisodicStore.recall → Traces → EpisodicAdapter → LLM
              │
      WriteGate(S) → [write] → ReplayQueue → (offline) trainer
```

## 16.2 SGC‑RSS dataflow

```
Text → TupleExtractor → KG.upsert;
Query → KG.retrieve + EpisodicStore.recall → RelationalAdapter/EpisodicAdapter → LLM
```

## 16.3 SMPD dataflow

```
Context → PlaceGraph.observe → plan(goal) → route scaffold
Successful trajectories → MacroLib.add → suggest() → LLM hints
```

