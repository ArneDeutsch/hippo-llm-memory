# 1) Purpose & scope

This document specifies a production-ready design for hippocampus-inspired memory extensions to small open LLMs (single 12 GB GPU). It consolidates the research outcome (Pass 1–4) into software modules, data schemas, training/eval protocols, and CI-friendly tasks. It implements **three experiments**:

* **HEI-NW** — Episodic index with **k-WTA sparse keys**, **modern-Hopfield completion**, **neuromodulated one-shot writes**, **CA2-like prioritized replay**.
* **SGC-RSS** — Schema-guided consolidation with a **relational semantic store** and **schema fast-track** routing.
* **SMPD** — Spatial map with **place-like codes**, optional **path-integration**, and **replay-to-policy** macro distillation.

# 2) Assumptions & constraints

* **Hardware:** 1× NVIDIA GPU (12 GB), Ubuntu Linux.
* **Base models:** small causal LMs (e.g., Llama-3.2-3B / Phi-3-Mini / Qwen2-1.5B).
* **Efficiency:** **QLoRA (NF4)** + gradient checkpointing; **FlashAttention** kernels when supported; **MQA/GQA** for KV memory reduction in both core and adapters; context length ≤1024 by default.
* **Frameworks:** PyTorch, HF Transformers, TRL, PEFT, bitsandbytes, FAISS-CPU (+PQ), Hydra, NetworkX (or DGL), pytest.
* **Non-goals:** Full pretraining; multi-GPU; online RL.

# 3) High-level architecture

```
                    ┌───────────── LLM core (HF, FlashAttn, MQA/GQA) ─────────────┐
                    │                   LoRA/QLoRA adapters                        │
                    └───────────────┬───────────────────────┬──────────────────────┘
                                    │                       │
                           [Episodic Adapter]        [Relational Adapter]
                                    │                       │
         ┌──────────────────────────┴────────────┐    ┌─────┴───────────────┐
         │ HEI-NW Episodic Store (FAISS+PQ,     │    │ SGC-RSS Semantic KG │
         │ SQLite, modern-Hopfield, ReplayQ)    │    │ (graph + GNN enc.)  │
         └───────────────────┬───────────────────┘    └─────────┬───────────┘
                             │                                  │
                        [Spatial Adapter / Tool API]            │
                             │                                  │
                     SMPD Map (PlaceGraph, path-int)  <─────────┘
                             + MacroLib
```

**Retrieval fabric:** one embedding/ANN layer shared across stores; store-type tags; latency SLAs and budgets.

# 4) Data structures

## 4.1 Episodic (HEI-NW)

* `DGKey`: **sparse k-WTA** key (indices+values, CSR-like), dim `d_k`.
* `TraceValue`: `{tokens_span_ref, entity_slots(who,what,where,when), state_sketch, salience_tags, provenance}`.
* `AssocStore`: FAISS index with **product quantization (PQ)** for RAM efficiency; a **modern-Hopfield** readout layer for CA3-style completion.
* `ReplayQueue` item: `{key_id, S (salience), recency, diversity_sig, grad_overlap_proxy, timestamp}`.

## 4.2 Relational (SGC-RSS)

* Tuple: `(head, relation, tail, context, time, conf, provenance)`.
* `SemanticGraph`: multigraph with **GNN-maintained node/edge embeddings**; persisted in SQLite/Parquet.
* `SchemaIndex`: prototypes/frames (slot types, constraints) with similarity metrics.

## 4.3 Spatial (SMPD)

* `PlaceGraph`: nodes = **place-like codes** for contexts; edges carry `{cost, success_prob, last_seen}`; optional **path-integration accumulator** for sequential contexts.
* `MacroLib`: `{signature, steps (tool calls), success_stats, last_update}`; small scoring head for suggestion ranking.

# 5) Algorithms

## 5.1 Write-gate (neuromodulatory analog)

`S = α·surprise + β·novelty + γ·reward + δ·pin`.

* `surprise`: `−log p(next_token)` or local entropy from logits.
* `novelty`: `1 − max cos(query, catalog_keys)` (pre-retrieval).
* `reward/pin`: flags from environment/user.
* **Write iff `S > τ`**, attach provenance.

## 5.2 Recall (content-addressable, CA3 completion)

1. Query from current residual stream (hidden state at the insertion block).
2. Sparse key via **k-WTA projection**.
3. FAISS-PQ KNN → candidate traces.
4. **Modern-Hopfield** completion to densify/“fill in” the episodic pattern.
5. Pack to “memory tokens”; **EpisodicAdapter** cross-attends (MQA/GQA to cap KV).

## 5.3 CA2-like prioritized replay (interference-aware)

* Priority mix: `λ1·S + λ2·recency + λ3·diversity` with **successive-batch gradient-overlap minimization** (proxy via representation cosine).
* **Consolidation mix:** default **50% episodic**, **30% semantic graph**, **20% fresh tasks** per batch window.
* Interleave hard negatives to stabilize adapters before any weight-unfreezing.

## 5.4 Relational retrieval & schema routing

* Tuple extractor → KG upsert (with confidence & provenance).
* **Schema score** gates: high → **fast-track** to KG + light replay; low → remain episodic + heavy replay before generalization.
* Dual-path inference: top-k subgraphs ⊕ top-k episodes; **gating head** merges adapter outputs.

## 5.5 Spatial planning & macro distillation

* Place encoder builds/merges nodes; optional **path-integration** over recent segments.
* Planner: A\*/Dijkstra (text worlds) or learned planner (ablation).
* Successful trajectories → **behavior cloning** into `MacroLib`; suggest at inference (top-k).

# 6) Adapters & Memory I/O

* **Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, optionally FFN `{up, down}` in adapter blocks only.
* **Defaults:** r=16, α=32, dropout=0.05, NF4 4-bit, grad checkpointing.
* **Placement:**

  * **EpisodicAdapter**: after block `N` (configurable).
  * **RelationalAdapter**: parallel cross-attention over KG encodings; gated merge.
  * **SpatialAdapter**: lightweight cross-attention to plan/macro embeddings or via tool-API.
* **Efficiency:** enable **FlashAttention** kernels and **MQA/GQA** in adapter attention.
* **MemoryTokens & flow:** retrieval hooks gather top‑K features from each store,
  project to `d_model` and pack to `memory_tokens` `[B, M, d_model]` (+ mask).
  Adapters no‑op when the tensor is empty.
* `_hippo_retrieval_cb(hidden)` attaches to the target block, issues store queries,
  and returns `MemoryTokens` plus latency stats.
* **Write path:** `surprise` from model logits and `novelty` from pre-retrieval
  cosine drive the gate; accepted items are enqueued to an async writer thread.
  After forward, persist when `S > τ`.

# 7) Configuration (Hydra)

```yaml
model:
  name: llama32-3b
  dtype: bfloat16
efficiency:
  flash_attention: true
  mqa_gqa: "gqa"
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
  episodic: {k: 8, metric: cosine, write_threshold: 0.7, hopfield: true, pq: true}
  relational: {topk_subgraphs: 4, tuple_conf_min: 0.8, schema_fasttrack: true}
  spatial: {planner: astar, path_integration: true}
consolidation:
  batch_mix: {episodic: 0.5, semantic: 0.3, fresh: 0.2}
  schedule: {minimize_grad_overlap: true}
```

# 8) Public APIs

## 8.1 Episodic

```python
class EpisodicStore:
    def write(self, key: np.ndarray, value: dict) -> int: ...
    def recall(self, query: np.ndarray, k: int) -> list[dict]: ...
    def delete(self, key_id: int) -> None: ...
    def decay(self, now: float) -> None: ...

class WriteGate:
    def score(self, surprise: float, novelty: float, reward: bool, pin: bool) -> float: ...
```

## 8.2 Relational

```python
class TupleExtractor: ...
class KG:
    def upsert(self, tuples: list[Tuple]) -> None: ...
    def retrieve(self, query: str, k: int) -> "Subgraph": ...
    def prune(self, ttl_days: int) -> None: ...
```

## 8.3 Spatial

```python
class PlaceGraph:
    def observe(self, context: dict) -> int: ...
    def plan(self, start: int, goal: int) -> list[int]: ...
    def merge_similar(self, thr: float) -> None: ...

class MacroLib:
    def add(self, trajectory: list[dict]) -> str: ...
    def suggest(self, context: dict, topk: int = 3) -> list[dict]: ...
```

# 9) Training & consolidation

* **Phase A (adapters only):** freeze base; train Episodic/Relational/Spatial adapters on synthetic tasks + small curated corpora.
* **Phase B (replay distillation):** enable **CA2-like scheduler** with the 50/30/20 mix; optionally unfreeze a small subset of base layers at low LR.
* **KV-cache interplay:** prefer **retrieval + adapters** over brute long contexts; cap adapter tokens via learned budgeters; rely on **GQA** to bound KV.
* **Checkpointing:** save per-adapter LoRA weights; export/merge tools provided.

# 10) Evaluation & ablations

* **Episodic:** partial-cue EM/F1; robustness vs. distractors; latency.
* **Semantic:** multi-hop QA vs. KG; contradiction rate; **schema acceleration** delta.
* **Spatial:** path success, suboptimality, plan length.
* **Procedural:** macro reuse, steps-to-solve, tool-latency.
* **Stress tests:** rapid novel-episode bursts; schema-flip events; large-map growth.
* **User-like setting:** interactive sessions with explicit **pin** signals to validate gating.
* **Baselines:** core LM (FlashAttn), +RAG, +long-context (ALiBi/RoPE), +Compressive/Longformer variants.
* **Ablations:** −DG sparsity, −Hopfield, −gate, −replay scheduler, −schema fast-track, −path-integration, −macros.
* **Compute:** FLOPs saved vs. long-context; KV memory with/without MQA/GQA.

# 11) Ops: logging, provenance, rollback, and maintenance

* Every write records `{text_span, doc_id, time, conf, source}`; **delete\_by\_provenance()** and snapshot/restore for all stores.
* **Nightly jobs:**

  * Episodic decay of low-S traces; dedupe near-duplicates.
  * KG pruning of stale/low-confidence edges.
  * PlaceGraph node merging and TTL on unused nodes; MacroLib success aging.
* Structured logs as JSONL under `runs/<date>/<exp>/events.jsonl`.

# 12) Risks & mitigations

* **Noisy writes:** raise τ; allow user **pin**; post-hoc pruning.
* **Replay interference:** enforce diversity and **grad-overlap minimization**; cap similarity per batch.
* **Extractor drift (KG):** confidence thresholds, provenance rollback, golden tests.
* **Map explosion:** similarity-based node merge; TTLs; cap degree.
* **Latency spikes:** ANN budgets per store; adapter token budgets; FlashAttention and GQA.

# 13) Milestones & CI-friendly tasks

1. **Scaffold/CI**: repo tree, lint/test green.
2. **Episodic v0**: store + gating + FAISS-PQ; tests.
3. **Hopfield completion**: CA3 readout; ablations.
4. **Relational v0**: tuple extractor + KG + GNN enc.; tests.
5. **Spatial v0**: PlaceGraph + A\* + optional path-integration; tests.
6. **Adapters wired**: Episodic/Relational/Spatial with GQA & FlashAttn.
7. **Trainer**: QLoRA trainer + consolidation mix + scheduler.
8. **Eval harness**: metrics, baselines, stress tests, user-pin flows.

# 14) Reproducibility

* Fixed seeds; deterministic dataloaders where feasible; snapshot FAISS/KG/PlaceGraph for CI.
* Record model IDs and config hashes in run metadata.

# 15) ASCII appendix

## 15.1 HEI-NW

```
Tokens → LLM → Query
              │
      k-WTA → FAISS-PQ KNN → Hopfield completion → Traces → EpisodicAdapter (GQA)
              │
      WriteGate(S) → [write] → ReplayQueue → (CA2 scheduler) → Trainer (50/30/20)
```

## 15.2 SGC-RSS

```
Text → TupleExtractor → KG.upsert (GNN embeds, provenance)
Query → {KG.subgraphs ⊕ Episodic traces} → Relational/Episodic Adapters → LLM
Schema score ↑ → fast-track vs. heavy replay
```

## 15.3 SMPD

```
Context → Place encoder (+path-integration) → PlaceGraph
Goal → plan() → route scaffold → SpatialAdapter/tool hints → LLM
Trajectories → MacroLib (behavior cloning) → suggest()
```
