# 1) Purpose & scope

This document specifies a production-ready design for hippocampus-inspired memory extensions to small open LLMs (single 12 GB GPU). It consolidates the research outcome (Pass 1–4) into software modules, data schemas, training/eval protocols, and CI-friendly tasks. It implements **three experiments**:

* **HEI-NW** — Episodic index with **k-WTA sparse keys**, **modern-Hopfield completion**, **neuromodulated one-shot writes**, **CA2-like prioritized replay**.
* **SGC-RSS** — Schema-guided consolidation with a **relational semantic store** and **schema fast-track** routing.
* **SMPD** — Spatial map with **place-like codes**, optional **path-integration**, and **replay-to-policy** macro distillation.

All evaluation pipelines, metrics, reporting utilities, and synthetic tasks now
live in a separate top-level package, `hippo_eval`. The `hippo_mem` package is
limited to core memory algorithms; legacy import paths have been removed.

# 2) Assumptions & constraints

* **Hardware:** 1× NVIDIA GPU (12 GB), Ubuntu Linux.
* **Base models:** small causal LMs (e.g., Llama-3.2-3B / Phi-3.5-Mini / Qwen2.5-1.5B).
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

## 5.6 Optional salience gates (relational & spatial)

*Scope.* Engineering option to reduce churn and capacity waste while preserving provenance. Complements the episodic neuromodulatory gate and schema fast‑track; **ablatable**.

**Principles**

1. **Route/aggregate, don’t drop.** Low‑score items are routed to the episodic store or **aggregated as evidence**; redundant graph/map insertions are blocked, but the observation is *not discarded*.
2. **Provenance preserved.** Every decision records a `gate_reason` and destination.
3. **Ablation‑friendly.** A single flag disables the gates for clean comparisons.

**Relational (SGC‑RSS).** Compute a score `S = w_conf·conf + w_schema·schema_fit + w_rec·recency − w_hub·degree_penalty`. Decision:

* **insert\_graph** if `S ≥ τ` and edge does not exist;
* **aggregate\_duplicate** if edge exists → increment edge evidence/weight and update recency; no new edge is created;
* **route\_to\_episodic** otherwise (defer consolidation; replay later).

**Spatial (SMPD).** Penalize immediate repeats and short‑window A↔B flapping; cap node degree. Decision:

* **add\_node/edge** when novel and within degree limits;
* **aggregate\_duplicate** for repeated transitions (increase edge weight/recency);
* **block\_new\_edge** only when creating a *redundant* new edge; the observation still contributes to weights/recency.

**Telemetry.** Per‑memory counters: `attempts, inserted, aggregated, routed_to_episodic (relational), blocked_new_edges (spatial)`.

**Guarantee.** Gates never delete inputs; provenance and rollback are maintained.

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
  name: gpt2
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
  relational:
    topk_subgraphs: 4
    tuple_conf_min: 0.8
    schema_fasttrack: true
    gate:
      enabled: true
      threshold: 0.6
      w_conf: 0.6
      w_schema: 0.5
      w_hub: 0.4
      w_rec: 0.2
      max_degree: 64
  spatial:
    planner: astar
    path_integration: true
    gate:
      enabled: true
      block_threshold: 1.0
      repeat_N: 3
      recent_window: 20
      max_degree: 64
consolidation:
  batch_mix: {episodic: 0.5, semantic: 0.3, fresh: 0.2}
  schedule: {minimize_grad_overlap: true}
```

**Note.** Gates are optional and ablatable; see §10 (Ablations) for on/off switches and §11 for runtime counters and provenance logs.

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
    def aggregate_duplicate(self, tup: Tuple) -> None:
        """Increase evidence/weight and refresh recency for an existing edge."""
    def route_to_episodic(self, tup: Tuple) -> None:
        """Append tuple to episodic writer queue for later replay/consolidation."""

from hippo_mem.common import GateDecision

class RelationalGate:
    def decide(self, tup: Tuple, kg: "KG") -> GateDecision:
        """Return a :class:`GateDecision` where action ∈ {"insert","aggregate","route_to_episodic"}."""
```

## 8.3 Spatial

```python
class PlaceGraph:
    def observe(self, context: dict) -> int: ...
    def plan(self, start: int, goal: int) -> list[int]: ...
    def merge_similar(self, thr: float) -> None: ...
    def aggregate_duplicate(self, prev_ctx: dict, ctx: dict) -> None:
        """Bump edge weight/evidence and recency without adding a new edge."""

class MacroLib:
    def add(self, trajectory: list[dict]) -> str: ...
    def suggest(self, context: dict, topk: int = 3) -> list[dict]: ...

class SpatialGate:
    def decide(self, prev_ctx: dict, ctx: dict, graph: "PlaceGraph") -> GateDecision:
        """Return a :class:`GateDecision` where action ∈ {"insert","aggregate","block_new_edge"}."""
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
* **Baseline presets (Milestone 8):** all memory adapters, retrieval calls, and gates are disabled;
  memory telemetry is a no-op so latency and metrics reflect the core model only.
* **Ablations:** −DG sparsity, −Hopfield, −gate, −replay scheduler, −schema fast-track, −path-integration, −macros.
* **Compute:** FLOPs saved vs. long-context; KV memory with/without MQA/GQA.

## 10.1 Teach→test protocol and success bars

```bash
export RUN_ID=my_experiment
export STORES=runs/$RUN_ID/stores
export SID=hei_$RUN_ID
# Teach then test episodic memory
python scripts/eval_model.py suite=episodic preset=memory/hei_nw \
  run_id=$RUN_ID mode=teach persist=true store_dir=$STORES session_id=$SID
python scripts/eval_model.py suite=episodic preset=memory/hei_nw \
  run_id=$RUN_ID mode=test store_dir=$STORES session_id=$SID
```

Store layout:

```
runs/$RUN_ID/stores/
  hei_nw/$SID/episodic.jsonl
  sgc_rss/sgc_$RUN_ID/kg.jsonl
  smpd/smpd_$RUN_ID/spatial.jsonl
```

Success bars:

- episodic: `ΔEM(core→memory) ≥ 0.10` and `EM(memory) ≥ EM(longctx)` with
  `memory_hit_rate ≥ 0.3`.
- semantic: EM uplift over `baselines/longctx` on the `semantic(hard)` split.
- spatial: `EM ≥ 0.10` or `steps_to_goal` reduced by ≥20%.

`semantic(default)` and `episodic_cross(default)` act as **smoke tests** only.

## 10.2 Context-Keyed Memory Access

Every example derives a stable **`context_key`** (e.g., `episode_id` or
timestamp). The evaluation harness passes this key to store APIs:

```python
store.write(trace, context_key=episode_id)
store.retrieve(query, k=K, context_key=episode_id)
```

Adapters propagate the key so retrieved traces can be attributed to the
correct teaching context. Isolation modes (`per_item`, `per_episode`) use the
`context_key` to fork or filter stores, and telemetry records the key for every
write/read event.

## 10.3 Justification & Leakage Telemetry

When a prediction uses memory, telemetry logs **trace IDs** and a
`context_match_rate` measuring how many retrieved traces match the supplied
`context_key`.

```json
{
  "qid": "semantic_mem/00042",
  "retrieval": {"requests": 3, "hits": 3, "context_match_rate": 1.0,
                "trace_ids": ["t101", "t099", "t055"]}
}
```

Leakage probes inject contradictory facts across items; mismatched traces are
counted under `leakage.mismatched` and should remain at zero under strict
isolation.

## 10.4 Fail-fast telemetry guards

- Abort when stores fall below expected sizes per suite.
- Require `embedding.nonzero_ratio ≥ 0.9` for episodic and semantic stores.
- Log `injected_context` on every retrieval attempt.

# 11) Ops: logging, provenance, rollback, and maintenance

* Every write records `{text_span, doc_id, time, conf, source}`; **delete\_by\_provenance()** and snapshot/restore for all stores.
* **Nightly jobs:**

  * Episodic decay of low-S traces; dedupe near-duplicates.
  * KG pruning of stale/low-confidence edges.
  * PlaceGraph node merging and TTL on unused nodes; MacroLib success aging.
* Structured logs as JSONL under `runs/<date>/<exp>/events.jsonl`.
* **Gate telemetry (JSON):** Counters emitted in `metrics.json` under `gates.{relational|spatial}` with fields `attempts, inserted, aggregated, routed_to_episodic/blocked_new_edges`.
* **Provenance NDJSON:** Per‑decision records in `runs/<date>/<exp>/provenance.ndjson` with `{ts, memory, action, reason, payload}` where payload minimally includes tuple ids or `(prev_ctx, ctx)` and degree/conf/score.
* **Config echo:** Effective gate config serialized into `meta.json` for audit and reproducibility.

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
9. **Gating semantics + telemetry:** implement decide‑actions, aggregation/routing, counters, provenance; tests & docs updated.
10. **Ablations & reports:** ON/OFF gate runs; duplicate‑rate and map‑growth tables in reports; CI smoke with gates enabled.

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

## 15.4 Gating flows

```
Relational ingest:
  tuple → gate.decide() → {insert_graph | aggregate_duplicate | route_to_episodic}
  counters++ ; provenance.log(...)

Spatial ingest:
  (prev_ctx, ctx) → gate.decide() → {add_edge | aggregate_duplicate | block_new_edge}
  counters++ ; provenance.log(...)
```

## Migration notes

- Evaluation, metrics, reporting, and synthetic tasks were extracted to the
  `hippo_eval` package. `hippo_mem` now contains only core memory algorithms.
- Reporting templates moved under `hippo_eval/reporting/templates`; the root
  `reports/` directory stores generated artifacts only. Legacy import paths like
  `hippo_mem.eval` have been removed.

