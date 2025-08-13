# Pass 1 – Independent Deep Reading

## Paper A — *How Large Language Models Work: Architecture and Mechanisms* (large-language-models.md)

### Summary (technical)

The paper presents a decoder-only Transformer view of LLMs trained with next-token prediction, detailing self-attention (causal masking), multi-head attention, positional encoding variants (RoPE, relative, ALiBi), feed-forward blocks with gated activations (SwiGLU/GEGLU), residual connections and (pre-)LayerNorm. It explains inference-time autoregressive decoding with KV caching, and surveys methods for long-context efficiency and scalability: optimized kernels (FlashAttention), architectural KV-memory reductions (MQA/GQA), positional schemes for length extrapolation (ALiBi/xPos), segment recurrence (Transformer-XL/Compressive Transformer), sparse/approximate attention (Longformer/BigBird/Linformer/Performer/Reformer), and application heuristics (sliding windows/summarization). Trends include deep dense models (GPT-3/PaLM/LLaMA), MoE layers for sparse activation, and exploratory non-Transformer directions (RNN-style RWKV, state-space “Mamba”), while noting Transformers remain dominant as of 2025.         &#x20;

### Key Terms & Definitions

* **Causal self-attention:** masked attention permitting only past-token dependencies during generation.&#x20;
* **Multi-Head Attention (MHA):** parallel attention heads with distinct Q/K/V projections.&#x20;
* **KV cache:** layerwise storage of past Keys/Values to reduce recomputation in decoding.&#x20;
* **RoPE / Relative / ALiBi:** positional schemes for extrapolation/generalization to longer sequences.&#x20;
* **FlashAttention:** exact attention kernel improving memory locality and throughput.&#x20;
* **MQA / GQA:** share K/V across heads (or groups) to shrink KV memory during inference.&#x20;
* **Sparse/approx. attention (Longformer, BigBird, Linformer, Performer, Reformer):** reduce O(n²) complexity with local/global patterns or low-rank/LSH approximations. &#x20;
* **MoE:** sparsely activated expert FFNs with learned routing.&#x20;
* **State-space/RNN alternatives (RWKV, Mamba):** recurrent or SSM formulations aiming for long-range efficiency.&#x20;

### Workflow Diagram (training & inference)

```
[Tokens] → [Embed + Positional (RoPE/ALiBi/etc.)]
   → repeat L times: [Self-Attn (causal, MHA) → Residual/Pre-LN → FFN (SwiGLU) → Residual/Pre-LN]
   → [LM head → Softmax]  --loss: cross-entropy; optimize via backprop
                                                          :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

Inference (autoregressive with KV cache):
prompt → forward pass (cache K,V per layer)
loop:
  compute next token’s Q per layer → attend to cached K,V → sample/argmax → append → update cache
(stop when EOS/length limit)                                   :contentReference[oaicite:22]{index=22}
```

### Constraints / Strengths

**Strengths:** parallel sequence processing; direct long-range dependency modeling; scalable depth/width; efficient kernels (FlashAttention) and KV-memory reductions (MQA/GQA) enable longer contexts and faster inference. &#x20;
**Constraints:** quadratic attention cost and KV-cache memory pressure; degraded use of ultra-long contexts without specific training; sparse/approximate patterns trade some accuracy; recurrence/segment memory adds complexity; MoE introduces routing/balancing overhead.   &#x20;

---

## Paper B — *Hippocampal Memory Storage: Mechanisms and Distinctive Features* (hippocampal-memory-storage.md)

### Summary (technical)

The paper characterizes the hippocampus as a rapid, sparse, content-addressable episodic memory system embedded in the trisynaptic circuit (EC→DG→CA3→CA1). DG performs pattern separation via sparse codes; CA3’s recurrent collaterals implement autoassociation for one-shot binding and pattern completion; CA1 integrates CA3/EC inputs to broadcast outputs to cortex. Encoding relies on fast Hebbian plasticity (LTP) modulated by neuromodulators (DA/NE) with novelty/salience gating. Consolidation proceeds from synaptic stabilization to systems-level replay during slow-wave sleep (sharp-wave ripples), coordinating with cortex (and possibly orchestrated by CA2) to gradually establish long-term cortical traces. Engrams are sparse neuronal assemblies serving as indices into distributed cortical content; retrieval is via content-based partial cues. The paper reviews selection pressures on storage (novelty, attention, reward), outlines open questions (indexing implementation, pattern separation/completion specifics, forgetting, schema-accelerated cortical learning), and maps hippocampal principles to ML (CLS, Hopfield attractors, SDM, generative replay, prosthetic MIMO models).       &#x20;

### Key Terms & Definitions

* **Trisynaptic circuit (DG→CA3→CA1):** canonical loop for pattern separation (DG) and autoassociation/completion (CA3) with CA1 as integrative output to cortex.&#x20;
* **Pattern separation / completion:** orthogonalization of inputs vs. recall from partial cues via attractor dynamics.&#x20;
* **LTP (Hebbian):** rapid NMDA-dependent synaptic potentiation enabling one-trial encoding.&#x20;
* **Engram:** sparse assembly whose strengthened synapses index cortical representations; content-addressable.&#x20;
* **Systems consolidation:** hippocampal-to-cortical transfer via SWR replay during SWS; CA2 may coordinate ripple timing. &#x20;
* **CLS (Complementary Learning Systems):** fast hippocampal episodic vs. slow cortical semantic learning.&#x20;
* **Neuromodulatory gating:** DA/NE boost encoding of novelty/salience (e.g., CA3 D1-dependent one-trial learning).&#x20;

### Workflow Diagram (encoding → consolidation → recall)

```
ENCODING (wake, high ACh/DA/NE):
EC input → [DG sparse code] → [CA3 recurrent binding] → [CA1 integration] → cortical reinstatement
   ↑ novelty/attention/reward modulate LTP thresholds                           :contentReference[oaicite:45]{index=45} :contentReference[oaicite:46]{index=46}

SYNAPTIC CONSOLIDATION (hours):
late-phase LTP / gene expression stabilizes hippocampal engram                   :contentReference[oaicite:47]{index=47}

SYSTEMS CONSOLIDATION (sleep/quiet wake):
HPC SWR replay ↔ cortical spindles/slow oscillations → gradual cortical wiring
CA2 coordinates ripple timing to reduce interference                             :contentReference[oaicite:48]{index=48} :contentReference[oaicite:49]{index=49}

RECALL (later, cue-driven):
partial cortical cue → HPC index (CA3 attractor) completes pattern → CA1 → cortex
(content-addressable retrieval; hippocampus may remain necessary for rich detail) :contentReference[oaicite:50]{index=50} :contentReference[oaicite:51]{index=51}
```

### Constraints / Strengths

**Strengths:** one-trial episodic encoding; sparse/orthogonalized codes limit interference; autoassociative completion; selective consolidation prioritized by salience/novelty; replay-driven cortical teaching. &#x20;
**Constraints/open issues:** early traces are labile; capacity/interference if selection fails; consolidation depends on offline dynamics; precise indexing/replay selection algorithms unresolved; degree/duration of hippocampal dependence for remote memories debated; schema-accelerated cortical learning complicates classic CLS assumptions. &#x20;

# Pass 2 – Cross-Domain Mapping & Gap Analysis

## 5) Mapping Table: Hippocampal Mechanisms ↔ LLM Mechanisms

| Hippocampal mechanism (role)                                                  | Concrete neuro features                                                                                                | Closest LLM analogue                                             | Concrete LLM features                                                                                                                                                                            |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Pattern separation (encode orthogonalized episodes)**                       | DG sparse codes; k-WTA‐like sparsity; adult neurogenesis hypothesis; reduces interference                              | Sparse activation & routing; embedding space separation          | MoE routing (top-k experts), token-level sparsity; attention masks/sparse patterns (Longformer/BigBird); representation regularization to increase inter-episode distance                        |
| **Autoassociative binding & completion (content-addressable episodic index)** | CA3 recurrent collaterals; Hopfield-like attractors; one-trial binding; content-addressable recall from partial cues   | Attention as content-addressable lookup; associative memories    | Softmax(Q·Kᵀ) attention; KV cache as growing episodic buffer at inference; modern Hopfield layers (in ML) conceptual tie (not standard in LLM cores)                                             |
| **Indexing theory (hippocampal pointer to cortical content)**                 | Sparse engrams that “point” to distributed cortical representations; associative, relational code                      | Retrieval-augmented generation (RAG) / external key-value stores | Query→nearest-neighbors over vector DB; cross-attention over retrieved passages; sliding-window summarization as a crude index substitute                                                        |
| **Encoding vs retrieval modes (ACh/DA/NE gating)**                            | High ACh/DA/NE during novelty/attention → fast LTP; lowered plasticity otherwise; novelty/salience filters             | Write/read control & salience-aware memory ops                   | Heuristic “write” policies to external memory; reward- or uncertainty-triggered caching; RLHF-style signals as coarse neuromodulatory analogues (no native neurochemical gate in standard LLMs)  |
| **Systems consolidation (offline replay → cortex)**                           | SWR replay; CA2 orchestration; prioritized consolidation of salient episodes; CLS framework                            | Offline distillation/replay to weights                           | Experience replay; generative replay; periodic fine-tuning/distillation from episodic buffer into base model weights (parametric “cortex”)                                                       |
| **Temporal segmentation & interference control**                              | Time-separated encoding; sleep/rest reduce overlap; forgetting as filtering                                            | Segment recurrence & memory compression                          | Transformer-XL segment memory; Compressive Transformer (lossy long-term state)                                                                                                                   |
| **Capacity/efficiency constraints**                                           | Sparse codes raise capacity; local recurrent circuitry; rapid but labile traces                                        | KV-cache/memory engineering                                      | MQA/GQA to shrink KV memory; FlashAttention for exact but memory-efficient attention                                                                                                             |

---

## 6) Functional Gaps (where LLMs fall short of hippocampal function)

1. **True one-shot episodic writes with durable recall.**
   Hippocampus can bind a novel episode in a single trial via rapid LTP and CA3 autoassociation; standard LLMs lack an intrinsic rapid-plasticity path at inference—context/KV cache is transient and not weight-consolidated. &#x20;

2. **Content-addressable completion from partial, multimodal cues.**
   CA3 retrieves full patterns from sparse cues; attention performs content-based lookup but only over tokens present in context/RAG results, not a persistent associative store with guaranteed attractor dynamics. &#x20;

3. **Neuromodulatory salience gating.**
   DA/NE/ACh set encoding thresholds and consolidation priority; LLMs have no endogenous channel to gate “write now, consolidate later” based on novelty/surprise/valence—only engineered heuristics. &#x20;

4. **Systems-level replay with interference-aware scheduling.**
   Hippocampal SWR and CA2 orchestrate replay order to avoid memory collisions; LLM fine-tuning/replay lacks principled, biologically inspired schedulers for interference mitigation under limited budgets.&#x20;

5. **Schema-accelerated consolidation.**
   Cortex can learn quickly when inputs fit existing schemas; LLMs don’t explicitly model schema alignment to decide when to “fast-track” consolidation versus defer.&#x20;

6. **Explicit relational/episodic indexing.**
   Hippocampus forms relational engrams linking who-what-where-when; LLM embeddings are dense but do not expose a structured tuple-index with pointer-like retrieval guarantees.&#x20;

7. **Forgetting and write-protection.**
   Biological systems down-scale or prune low-value engrams; LLMs lack principled decay of outdated episodic memory in external stores and lack safe write-protection of high-value entries.

8. **Spatial and navigational codes.**
   Place/grid-like representations support spatial episodic context; most LLMs have no native spatial memory module; long-context tricks don’t substitute for metric/topological maps. (Implied gap; spatial role noted as hippocampal strength.)&#x20;

---

## 7) Overlaps & Unique Strengths

### Overlaps

* **Content-addressable operations:** Softmax attention implements differentiable content lookup; with KV cache or RAG, this approximates hippocampal cue-based retrieval at short timescales. &#x20;
* **Sparsity for capacity/efficiency:** DG-like sparsity ↔ MoE/top-k routing; both reduce interference and compute. &#x20;
* **Segment memory & compression:** Transformer-XL / Compressive Transformer echo the idea of carrying forward a compact “state” akin to temporally spaced encoding.&#x20;

### Unique strengths (Hippocampus)

* **Rapid, neuromodulated one-trial learning** with **pattern separation + completion** yielding robust episodic indices; **offline SWR replay** with **CA2-style orchestration** for low-interference consolidation. &#x20;
* **Relational engrams** that bind entities across who/what/where/when with cue-driven completion.&#x20;

### Unique strengths (LLMs)

* **Massive semantic compression in weights** (parametric “cortex”) with global regularities distilled across corpora; **engineered memory scaling** (FlashAttention, MQA/GQA) enabling long contexts; **modular sparsity (MoE)** for scalable capacity.  &#x20;
* **Deterministic, controllable compute** and **pluggable external memory** (RAG, summarization windows) even if biologically crude.&#x20;


# Pass 3 – Algorithm & Architecture Design

Below are three hippocampus-inspired LLM memory designs. Each covers **episodic, semantic, spatial, and procedural** aspects (explicitly noted per design), with operations, integration points, neuroscience links, benefits, and an ASCII schematic.

---

## 8.1) Hippocampal Episodic Index with Neuromodulated Writes (HEI-NW)

**High-level description**
Augment a decoder-only LLM with a *persistent episodic store* that (i) **separates** new experiences into sparse keys (DG-like), (ii) supports **content-addressable completion** from partial cues via a modern-Hopfield/associative layer (CA3-like), and (iii) applies **salience-gated one-shot writes** driven by novelty/surprise/reward signals (neuromodulators) plus **prioritized offline replay** for consolidation into the model’s semantic weights (CLS). DG → sparse code; CA3 → autoassociation; CA1-like readout replays to cortex (the base model) during consolidation.   &#x20;

**Step-by-step operation**

1. **Cue & read phase (inference):** From the running KV-cache state, extract a query embedding; DG-encoder produces a *k-WTA* sparse key; CA3-associative layer (modern Hopfield / nearest-neighbor with learned associative readout) **completes** the episode; the completed trace is cross-attended by the LLM. &#x20;
2. **Write gate:** Compute salience $S$ = f(novelty via predictive surprise, reward tag, user “pin” signal). If $S>\tau$, **write once**: store (sparse key, value bundle), where the value holds (i) compressed token subsequences, (ii) entity/slot tuples (who/what/where/when), and (iii) decoder state sketches. Neuromodulatory inspiration: novelty/DA/NE “switch to high-plasticity” mode.&#x20;
3. **Prioritized replay (offline):** A scheduler replays high-S episodes interleaved with others (CA2-style orchestration) to **distill** into base weights (semantic cortex) and to update compact summaries; prevents interference (CLS).&#x20;
4. **Eviction/decay:** Low-S, stale episodes decay (use age-/use-based TTL with diversity regularizers) to keep capacity and reduce collisions.

**Integration points (LLM)**

* Hooks at **KV-cache** step to form queries/values; cross-attention layer to episodic traces; background distillation pipeline that fine-tunes from replay (RAG-like but from internal store). Efficient attention (FlashAttention) and MQA/GQA help long contexts while this store handles **true one-shot** persistence. &#x20;

**Neuroscience justification**

* DG **pattern separation** (sparse codes) → k-WTA encoder; CA3 **autoassociation/completion** → Hopfield-style recall; **hippocampal indexing** of cortical content; **neuromodulatory gating** (DA/NE) for novelty-tagged writes; **SWR replay** for systems consolidation (CLS).  &#x20;

**Benefits**

* Durable **one-trial episodic** storage with **partial-cue** retrieval; reduced interference; principled consolidation vs. ad-hoc long-context heuristics.&#x20;

**Memory-type coverage**

* **Episodic:** primary (episodes are first-class entries).
* **Semantic:** via replay/distillation into base weights.
* **Spatial:** store place-slots in the episode tuple; links to Design 8.3 for richer spatial codes.
* **Procedural:** tag episodes with action traces; replay can distill frequent traces into macro-policies.

**ASCII schematic**

```
Text stream → LLM(core) → [Query]
                    │
               [DG k-WTA encoder]  --sparse key-->  [CA3 assoc./Hopfield]
                    │                                   │
                    └────── write gate (DA/NE-like) ────┘
                              │
                         Episodic Store  ↔  Replay Scheduler(CA2-like)
                              │                   │
                      Cross-attend (recall)   Distill → LLM weights
```

---

## 8.2) Schema-Guided Consolidation with Relational Semantic Store (SGC-RSS)

**High-level description**
Introduce an explicit **relational semantic memory** (graph/tuple store) that captures schemas. The hippocampal episodic buffer (8.1) attempts **schema alignment** on write; **aligned** episodes are *fast-tracked* to the semantic store and prioritized for consolidation; **unaligned/novel** ones linger episodically and are replayed more before gradual integration. This mirrors **CLS** plus evidence for **schema-accelerated** learning and **orchestrated replay**. &#x20;

**Step-by-step operation**

1. **Relational parse:** From tokens, extract candidate (entity, relation, context, time) tuples and soft schemas (typed patterns).
2. **Schema score:** Compute similarity to existing schemas; if high, **consolidate immediately** into the graph store and schedule light replay; if low, keep in episodic store and schedule **rich replay**.
3. **RAG-style read:** At inference, queries hit both stores: fast schema nodes (semantic) and high-S episodes (episodic), combined through cross-attention.&#x20;
4. **Anti-interference scheduling:** A CA2-inspired replay planner orders reactivations to **avoid overlap** and interleave new with old (minimize catastrophic forgetting).&#x20;

**Integration points (LLM)**

* Adds a **Relational Memory Adapter** (cross-attention over graph embeddings); background **consolidation worker** that fine-tunes the base model from interleaved batches (episodes + schema exemplars).
* Works with long-context and **MQA/GQA** to keep runtime tractable.&#x20;

**Neuroscience justification**

* **Indexing theory**: hippocampus binds who/what/where/when and replays to cortex; **schemas** enable faster cortical learning; **prioritized replay** and **orderly ripple timing** (CA2). &#x20;

**Benefits**

* Improves factual consistency and multi-hop reasoning by elevating **semantic structure** to a first-class store, while preserving **episodic specificity** for edge cases.

**Memory-type coverage**

* **Semantic:** primary (relational graph).
* **Episodic:** feeder buffer; exceptions remain episodic.
* **Spatial:** schema nodes can include spatial relations (where/route), delegating geometry to 8.3.
* **Procedural:** schemas include event scripts; frequent scripts consolidated faster.

**ASCII schematic**

```
LLM → Tuple/Schema Extractor → Schema Score ─┬─> Semantic Graph Store (KG)
                                             │
                              Episodic Store ─┘ (if low score)
                Inference: Cross-attend {Semantic nodes} ∪ {Top-S episodes}
                  Offline: CA2-like planner → Interleaved replay → Weight update
```

---

## 8.3) Spatial Map + Replay-to-Policy Distillation (SMPD)

**High-level description**
A **spatial memory module** builds a topological/metric map (nodes = places/contexts; edges = transitions/relations) with **place-like codes** for contexts and **path-integration** over sequences; paired with a **procedural chunker** that distills frequently replayed action sequences (“scripts”) into reusable **macro-policies**. Hippocampus supports **spatial memory for navigation** and uses replay for planning; here, replay also trains procedural controllers. &#x20;

**Step-by-step operation**

1. **Spatial/context keying:** Map prompts, documents, or environments to “places” via encoder; create/merge nodes; maintain adjacency and edge attributes (cost, reward).
2. **Navigation/read:** Given a goal, run graph search (or learned planner) over the map; return a **route plan** as a scaffold that the LLM fills with language/tool calls.
3. **Replay-to-Policy:** Episodes that solved tasks well are **replayed**; a policy head is trained to imitate/abstract them into **procedural macros** (habit library).
4. **Interplay with episodic/semantic:** Spatial nodes are linked to episodic traces and schema facts, enabling “recall by place” and schema-aware route selection.

**Integration points (LLM)**

* The module is a **tool-augmented memory**: the LLM issues a `SPATIAL.READ/WRITE/PLAN` call; returned plans and macros are fed back as structured hints.
* Long-context limits are mitigated by map retrieval + episodic recall instead of raw tokens.&#x20;

**Neuroscience justification**

* Hippocampus’ role in **spatial navigation** and **episodic replay** informing action selection; replay ordering helps avoid interference and supports planning. &#x20;

**Benefits**

* Strong gains on tasks requiring **navigation, layout reasoning, multi-step tool use**, and **habit formation** from repeated successes.

**Memory-type coverage**

* **Spatial:** primary (maps/graphs).
* **Procedural:** macro-policy library via replay.
* **Episodic:** stores trajectory episodes; used as training targets.
* **Semantic:** attaches facts to places/edges for richer planning.

**ASCII schematic**

```
Env/Text ↔ Spatial Encoder → [Place Nodes + Edges]
                                  │          │
                             Route Planner   │
                                  │          │
        Episodic traces ←─────────┘      Schema facts (relations)
               │
        Replay → Policy Head → Macro Library  → Hints/tool-calls → LLM
```

---

### Cross-design notes (implementation-adjacent)

* **Data structures:** sparse keys (product quantization or locality-sensitive hashing), associative matrices (Hopfield-like), tuple/graph stores, map graphs, and macro libraries.
* **Algorithms:** novelty/surprise estimation for write-gating; CA2-like **priority replay scheduler**; interleaved fine-tuning for CLS-style consolidation; efficient attention + **MQA/GQA** to control KV-cache growth during memory-augmented inference.  &#x20;


# Pass 4 – Implementation & Validation

## 9) Implementation Plan

Below are concrete build plans for the three designs from Pass 3, plus shared infrastructure. I keep LLM hooks standard: KV-cache taps, cross-attention adapters, and offline replay/distillation—all consistent with decoder-only Transformers, FlashAttention kernels, and MQA/GQA to control memory/latency.  &#x20;

---

### 9.1) HEI-NW — Hippocampal Episodic Index with Neuromodulated Writes

**Data structures**

* `DGKey`: sparse key vector (dimension d), generated by k-WTA encoder; store as indices + values (CSR-like) for compactness. (DG pattern separation rationale.) &#x20;
* `TraceValue`: tuple `{tokens_span_ref, entity_slots: (who, what, where, when), lm_state_sketch, salience_tags}`; optional compressed token span (e.g., sentencepiece IDs) and small sketch of hidden state for recall bootstrap. (Indexing / relational engram.) &#x20;
* `AssocStore`: CA3-like associative memory. Start with (i) K-NN over `DGKey` with product quantization; add (ii) modern-Hopfield readout (energy-based layer) for pattern completion. (Autoassociation / completion.) &#x20;
* `ReplayQueue`: prioritized deque with keys `{salience S, recency, diversity, gradient-overlap score}`; CA2-like ordering to reduce collisions during consolidation.&#x20;

**Algorithms**

*Write-gate (neuromodulatory analog)*

* Inputs: predictive surprise $u=-\log p_\theta(x_t|x_{<t})$, novelty $1-\max\_j \text{sim}(q, k_j)$, explicit reward flag, user “pin”.
* Gate: $S = \alpha u + \beta \cdot \text{novelty} + \gamma \cdot \text{reward} + \delta \cdot \text{pin}$. If $S>\tau$, create `{DGKey, TraceValue}` and push to `ReplayQueue`. (DA/NE/ACh inspiration for encoding priority.) &#x20;

*Recall*

1. Build query from current hidden state (post-attention residual stream).
2. `DGKey = kWTA(Proj(query))`.
3. `hits = AssocStore.lookup(DGKey, top_k)`; optional Hopfield completion step.
4. Pack `TraceValue`→memory tokens and **cross-attend** from the LLM (standard masked self-attn + an extra cross-attn block). (Content-addressable recall with partial cue.)&#x20;

*Consolidation (CLS-style)*

* Offline worker samples from `ReplayQueue` in an interference-aware order (minimize gradient cosine similarity between successive items), mixes with regular pretraining/fine-tuning batches, and updates base weights (semantic cortex) via LoRA or full finetune. (SWR replay → cortical teaching.) &#x20;

*Eviction/decay*

* Age-based TTL; usage-based reinforcement; stochastic down-scaling for low-S traces (forgetting as filtering).&#x20;

**Integration details**

* Insert a slim “Episodic Adapter” (cross-attn over retrieved traces) after block N in the stack.
* Surprise/novelty computed from existing LM logits and retrieval sims (no extra model).
* Keep runtime cost bounded via **MQA/GQA** in the adapter and **FlashAttention** in core attention. &#x20;

**Trade-offs**

* * True one-shot persistence; partial-cue completion; controllable write policy.
* – Extra memory I/O; risk of noisy writes if gate poorly tuned; requires careful consolidation scheduling to avoid drift. (Hippocampal dependence early; consolidation later.)&#x20;

---

### 9.2) SGC-RSS — Schema-Guided Consolidation + Relational Semantic Store

**Data structures**

* `SemanticGraph`: nodes = entities/types/schemas; edges = relations with confidences and provenance; embeddings maintained with GNN-style encoder.
* `SchemaIndex`: prototypes for event scripts (frames) against which episodes are scored (schema-fit). (Schema-accelerated learning.) &#x20;

**Algorithms**

*Relational extraction*

* From tokens/hidden states, a light tuple extractor emits (e,r,c,t); attach to `TraceValue`.

*Schema scoring & routing*

* Compute `score = max_sim(episode_slots, schema_prototypes)`.
* If high: **fast-track** consolidation—promote facts/edges to `SemanticGraph` immediately; short replay.
* If low: keep episodic; schedule more replay before adding generalized edges. (Mirrors “schema fit → faster cortical uptake”.)&#x20;

*Inference fuse*

* Dual-path retrieval: top-k schema subgraphs ∪ top-k episodic traces → two cross-attn adapters; merge via gating head.

**Integration details**

* New “Relational Adapter” (cross-attn over graph embeddings).
* Background trainer interleaves `SemanticGraph` facts with episodic replays in mini-batches to reduce forgetting (CLS).&#x20;

**Trade-offs**

* * Better multi-hop consistency; faster uptake when knowledge fits schema.
* – Extractor errors can pollute graph; need provenance and rollback.

---

### 9.3) SMPD — Spatial Map + Replay-to-Policy Distillation

**Data structures**

* `PlaceGraph`: nodes = places/contexts (textual or embodied); edges = transitions with weights (success prob, cost).
* `MacroLib`: library of distilled procedural macros learned from high-quality episodic trajectories.

**Algorithms**

*Map building*

* `place = SpatialEnc(h_state)`; merge with nearest node or create new; update edges using observed transitions/rewards. (Hippocampal spatial/episodic role.) &#x20;

*Planning/read*

* Given goal spec, run graph search (A\* / value iteration) to produce a route plan; return as scaffold to LLM.

*Replay→policy*

* From `ReplayQueue`, sample successful trajectories; behavior-clone into `MacroLib` (small policy heads) and expose as tool-call suggestions.

**Integration details**

* Expose `SPATIAL.{WRITE,READ,PLAN}` tool endpoints; surface route plans/macros as structured hints that the LLM can attend to.
* Tie places to episodic traces and schema facts for “recall by location” and context-conditioned plans.&#x20;

**Trade-offs**

* * Strong on navigation/layout/multi-step tool use; reusable habits.
* – Extra complexity if task domain is non-spatial; must avoid map explosion (use node merging, TTL).

---

### 9.4) Shared infrastructure & engineering

* **Attention/Context budget:** keep core LM efficient via FlashAttention; set adapter layers to GQA; cap adapter tokens (episodic/graph/map) with learned budgeters. &#x20;
* **Retrieval fabric:** standardized embedding and ANN layer for all stores; store-type tags; per-type latency SLAs.
* **Safety & provenance:** every write stores source span, time, and confidence; support removal/rollback.
* **Forgetting/maintenance:** nightly jobs decay low-S episodes, prune stale graph edges, and merge near-duplicate places (biological forgetting/filtering).&#x20;
* **Consolidation worker:** batches: 50% replay (episodic), 30% semantic edges, 20% fresh tasks; schedule uses CA2-inspired ordering to reduce interference.&#x20;
* **KV-cache interplay:** at inference, prefer retrieval to brute-force long context; lean on MQA/GQA for KV-memory control.&#x20;

---

## 10) Validation & Evaluation Plan

### 10.1) Benchmarks by memory type

**Episodic (one-shot, partial-cue)**

* *Synthetic Who/What/Where/When*: single-exposure stories; evaluate recall from partial cues (e.g., “who at where?”) after N distractors; measure exact match and time-lag robustness. (Pattern separation & completion.) &#x20;
* *Delayed recall & consolidation*: test immediately, then after offline consolidation, then after “sleep” (replay) cycles. Expect improved corticalized recall post-replay.&#x20;

**Semantic (schema-fit vs schema-mismatch)**

* *Schema Acceleration Test*: present facts/events that either fit existing schemas or violate them; measure time-to-stabilize in `SemanticGraph` and factual consistency. Expect faster consolidation on schema-fit. &#x20;

**Spatial**

* *Textual Navigation*: grid/graph worlds described in text; evaluate shortest-path reasoning and “recall by place” queries; route-planning success. (Spatial role.)&#x20;

**Procedural**

* *Macro acquisition*: repeated multi-tool tasks; measure emergence and reuse rate of macros; latency reduction over episodes.

**Long-context control**

* Compare to baselines relying solely on long context with dense attention vs. our memory adapters; report quality vs. compute (FlashAttention/MQA/GQA enablement noted). &#x20;

### 10.2) Ablation studies

* **Remove DG sparsity** → measure interference (error vs. number of similar episodes). (DG’s role.)&#x20;
* **Disable CA3/Hopfield completion** → measure drop in partial-cue recall.&#x20;
* **No neuromodulatory gate** (random writes) → clutter and retrieval precision degrade; quantify P\@k vs. store size. &#x20;
* **No consolidation/replay** → episodic recall ok, semantic generalization poor; track drift over time.&#x20;
* **No CA2-like scheduling** → induce interference by replay order; compare retention after sleep cycles.&#x20;
* **Turn off Schema fast-track** → slower factual stabilization on schema-fit items.&#x20;

### 10.3) Metrics

* **Episodic:** partial-cue EM/F1; recall\@k; robustness vs. distractors; latency.
* **Semantic:** factual consistency (open-book QA accuracy against `SemanticGraph`), contradiction rate across sessions.
* **Spatial:** success rate, path suboptimality, plan length.
* **Procedural:** success rate with/without macros; steps-to-solve; tool-call latency.
* **Consolidation:** pre- vs post-replay deltas; retention curves over days.&#x20;
* **Compute:** % attention FLOPs reduced via retrieval vs. long context; KV-cache memory with/without MQA/GQA.&#x20;
* **Safety/provenance:** rollback rate, incorrect-write detection (precision).

### 10.4) Experimental protocol

1. **Baselines:** core LLM (FlashAttention), +RAG, +long-context only (ALiBi/RoPE), +Compressive/Longformer-style variants for fairness. &#x20;
2. **Training phasing:** (a) freeze base LM; train adapters & memory only; (b) enable consolidation worker with limited LR; (c) full end-to-end fine-tuning on target tasks.
3. **Replay cycles:** emulate “sleep” windows; run prioritized replay; re-test after each cycle (expect improvements analogous to SWR effects).&#x20;
4. **Stress tests:** rapid new episode bursts to test interference; schema-flip events to test gating; map-scale growth tests for SMPD.
5. **User-like setting:** interactive sessions with explicit “pin” (reward) signals to probe neuromodulatory gating behavior.&#x20;

