# Module inventory

## hippo_mem package

### HEI-NW (episodic)
- `hippo_mem/episodic/gating.py` — DG key encoding and neuromodulated write gate.
- `hippo_mem/episodic/store.py` — FAISS-backed trace store with optional Hopfield completion.
- `hippo_mem/episodic/replay.py` — prioritized replay queue and batch scheduler.
-   Config: `configs/memory/episodic.yaml::{replay_weight,decay_rate,prune.*}`
- `hippo_mem/episodic/adapter.py` — LoRA cross-attention over recalled traces.
- `hippo_mem/episodic/index.py` — FAISS vector index wrapper.
- `hippo_mem/episodic/db.py` — SQLite helper for trace metadata.

### SGC-RSS (relational)
- `hippo_mem/relational/tuples.py` — heuristic tuple extractor from text.
- `hippo_mem/relational/schema.py` — schema prototypes and fast-track router.
- `hippo_mem/relational/kg.py` — SQLite-backed knowledge graph with pruning.
- `hippo_mem/relational/adapter.py` — dual-path cross-attention fuse of KG and episodes.

### SMPD (spatial)
- `hippo_mem/spatial/map.py` — place graph with optional path integration and planning.
- `hippo_mem/spatial/macros.py` — macro library for distilled action sequences.
- `hippo_mem/spatial/adapter.py` — cross-attention over plans and macro embeddings.

### Shared utilities
- `hippo_mem/retrieval/embed.py` — deterministic placeholder text embedding.
- `hippo_mem/retrieval/faiss_index.py` — FAISS index with NumPy fallback.
  - Config: `configs/memory/episodic.yaml::index_str`
- `hippo_mem/adapters/lora.py` — helper wrappers around PEFT LoRA adapters.
- `hippo_mem/consolidation/worker.py` — background replay worker for adapter finetuning.
  - Config: `configs/memory/{episodic,relational,spatial}.yaml::prune.*`

## scripts
- `scripts/train_lora.py` — QLoRA trainer with optional memory modules.
- `scripts/eval_bench.py` — synthetic evaluation harness and ablation plumbing.

## experiments configs
- `configs/memory/*.yaml` — default parameters for episodic, relational, and spatial stores.
- `configs/eval/memory/*.yaml` — evaluation presets for HEI-NW, SGC-RSS, SMPD, and combined.

## Ablation and configuration flags

| Flag | Algorithm | Default | Description | Source |
| --- | --- | --- | --- | --- |
| `memory.episodic.hopfield` | HEI-NW | True | toggle Hopfield completion in episodic store | CLI `+ablate.memory.episodic.hopfield=false` |
| `memory.episodic.pq` | HEI-NW | True | disable product quantization for episodic index | CLI `+ablate.memory.episodic.pq=false` |
| `episodic.use_sparsity` | HEI-NW | True | drop k-WTA sparse encoding | Hydra `+ablate=episodic.use_sparsity=false` |
| `episodic.use_completion` | HEI-NW | True | disable Hopfield completion | Hydra `+ablate=episodic.use_completion=false` |
| `episodic.use_gate` | HEI-NW | True | bypass write gate | Hydra `+ablate=episodic.use_gate=false` |
| `replay.enabled` | HEI-NW | True | turn off prioritized replay scheduling | Hydra `+ablate=replay.enabled=false` |
| `relational.schema_fasttrack` | SGC-RSS | True | bypass schema-based KG routing | Hydra `+ablate=relational.schema_fasttrack=false` |
| `spatial.macros` | SMPD | True | ignore MacroLib suggestions | Hydra `+ablate=spatial.macros=false` |
| `spatial.path_integration` | SMPD | True | disable path integration in PlaceGraph | Hydra `+ablate=spatial.path_integration=false` |

## Planning & review artifacts
- `review/review-2025-08-19.md` — audit of implementation & testing.
- `review/action-plan-2025-08-19.md` — concrete follow-ups from the review.
- `MILESTONE_9_PLAN.md` — expanded with C0/C11/C12 tasks.
- `EVAL_PLAN.md` — now includes harness requirements & command matrix.
