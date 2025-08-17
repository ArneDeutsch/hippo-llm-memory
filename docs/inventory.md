# Module inventory

## hippo_mem package

### HEI-NW (episodic)
- `hippo_mem/episodic/gating.py` — DG key encoding and neuromodulated write gate.
- `hippo_mem/episodic/store.py` — FAISS-backed trace store with optional Hopfield completion.
- `hippo_mem/episodic/replay.py` — prioritized replay queue and batch scheduler.
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
- `hippo_mem/adapters/lora.py` — helper wrappers around PEFT LoRA adapters.
- `hippo_mem/consolidation/worker.py` — background replay worker for adapter finetuning.

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

## Key configuration fields

| Key | Default | Description | Module |
| --- | --- | --- | --- |
| `train.replay_weight` | 0.5 | fraction of batches drawn from replay (50/30/20 mix) | `configs/train.yaml` |
| `train.gating_weights.write` | 1.0 | global multiplier for write decisions | `configs/train.yaml` |
| `memory.episodic.replay_weight` | 0.8 | bias toward episodic items in scheduler | `hippo_mem/episodic/replay.py` |
| `memory.episodic.decay_rate` | 0.01 | salience decay per maintenance pass | `hippo_mem/consolidation/worker.py` |
| `memory.relational.schema_threshold` | 0.7 | confidence needed for schema fast-track | `hippo_mem/relational/schema.py` |
| `memory.relational.prune.max_age` | 86400 | seconds before tuples expire | `hippo_mem/consolidation/worker.py` |
| `memory.spatial.prune.max_age` | 1000 | steps before places/edges expire | `hippo_mem/consolidation/worker.py` |
