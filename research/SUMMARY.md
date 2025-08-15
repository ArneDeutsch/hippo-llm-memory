# HEI-NW — Hippocampal Episodic Index with Neuromodulated Writes

**What it is.** A persistent **episodic store** for an LLM with DG-like **sparse keys**, CA3-style **associative completion** (e.g., modern Hopfield), and a **salience gate** that writes one-shot episodes when surprise/reward is high; high-value traces are **replayed offline** to consolidate into base weights (CLS).&#x20;
**Why it helps.** Brings true one-trial storage and **partial-cue recall** while limiting interference via **prioritized replay/decay**.&#x20;
**How it plugs in.** A lightweight **Episodic Adapter** (cross-attend to retrieved traces) after block *N*; write-gate from model logits/similarities; consolidation via LoRA/full FT.&#x20;
**Neuroscience lens.** DG **pattern separation**, CA3 **completion**, neuromodulator-gated plasticity, and **SWR replay** guiding systems consolidation.&#x20;

# SGC-RSS — Schema-Guided Consolidation with a Relational Semantic Store

**What it is.** A **graph/tuple semantic memory** that receives episodes parsed into (entity, relation, context, time). If a new episode **fits an existing schema**, it is **fast-tracked** into the graph and lightly replayed; otherwise it stays episodic and receives heavier replay before gradual integration. Inference fuses **graph reads** with **episodic recalls**.&#x20;
**Why it helps.** Improves **factual consistency** and multi-hop reasoning by elevating structure, while preserving episodic specificity for outliers; reduces forgetting via **anti-interference replay scheduling**.&#x20;
**How it plugs in.** Adds a **Relational Adapter** (cross-attend over graph embeddings) plus a background **consolidation worker** interleaving schema facts with episodic replays.&#x20;
**Neuroscience lens.** **Schema-accelerated** cortical learning coordinated by hippocampal replay/planning.&#x20;

# SMPD — Spatial Map + Replay-to-Policy Distillation

**What it is.** A **spatial memory** (topological/metric map) with place-like codes and path integration for planning; repeated successful trajectories are **replayed** to distill **macro-policies** (procedural skills) usable by the LLM.&#x20;
**Why it helps.** Enables navigation/layout reasoning, multi-step tool use, and accumulation of **habit libraries** for efficiency.&#x20;
**How it plugs in.** Tool-style interface returning routes/skills that the LLM expands; spatial nodes link to episodic traces and schema facts (“recall by place”).&#x20;
**Neuroscience lens.** Hippocampal **cognitive maps** and **replay** guiding planning and skill formation.&#x20;
