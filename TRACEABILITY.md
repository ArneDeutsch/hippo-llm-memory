# Normative Requirements

| Module | Requirement | Code | Tests | Verdict |
|---|---|---|---|---|
| HEI-NW | k-WTA keys & `DGKey` | `hippo_mem/episodic/gating.py` L13-40【F:hippo_mem/episodic/gating.py†L13-L40】 | `test_sparse_encode_k_wta_idempotent`【F:tests/test_episodic.py†L190-L200】 | ✅ |
| HEI-NW | FAISS+PQ episodic store | `hippo_mem/episodic/store.py` L45-75【F:hippo_mem/episodic/store.py†L45-L75】 | `test_faiss_index_trains_and_queries`【F:tests/test_episodic.py†L264-L272】 | ✅ |
| HEI-NW | Neuromodulated write gate `S=α·surprise+β·novelty+γ·reward+δ·pin` | `hippo_mem/episodic/gating.py` L90-135【F:hippo_mem/episodic/gating.py†L90-L135】 | `test_gating_threshold_and_pin_weight`【F:tests/test_episodic.py†L56-L71】 | ✅ |
| HEI-NW | Modern-Hopfield completion | `hippo_mem/episodic/store.py` L214-229【F:hippo_mem/episodic/store.py†L214-L229】 | `test_hopfield_completion_restores_sparse_cue`【F:tests/test_episodic.py†L35-L48】 | ✅ |
| HEI-NW | CA2-like prioritized replay (salience/recency/diversity, grad-overlap) | `hippo_mem/episodic/replay.py` L28-151【F:hippo_mem/episodic/replay.py†L28-L151】 | `test_replay_queue_avoids_consecutive_gradients`【F:tests/test_episodic.py†L152-L163】 | ✅ |
| HEI-NW | EpisodicAdapter with GQA/FlashAttn hooks | `hippo_mem/episodic/adapter.py` L74-181【F:hippo_mem/episodic/adapter.py†L74-L181】 | `test_flash_attention_flag`【F:tests/test_episodic.py†L88-L109】 | ✅ |
| SGC-RSS | Tuple extractor | `hippo_mem/relational/tuples.py` L23-80【F:hippo_mem/relational/tuples.py†L23-L80】 | `test_tuple_precision`【F:tests/test_relational.py†L1-L15】 | ✅ |
| SGC-RSS | KG store with embeddings | `hippo_mem/relational/kg.py` L18-86 & L185-205【F:hippo_mem/relational/kg.py†L18-L86】【F:hippo_mem/relational/kg.py†L185-L205】 | `test_multi_hop_retrieval`【F:tests/test_relational.py†L18-L31】 | ✅ |
| SGC-RSS | Schema index fast-track routing | `hippo_mem/relational/schema.py` L43-57【F:hippo_mem/relational/schema.py†L43-L57】 | `test_schema_fast_track_routing_threshold`【F:tests/test_relational.py†L52-L63】 | ✅ |
| SGC-RSS | Dual-path retrieval with gated fusion | `hippo_mem/relational/adapter.py` L11-41【F:hippo_mem/relational/adapter.py†L11-L41】 | `test_dual_path_fusion_deterministic`【F:tests/test_relational.py†L21-L33】 | ✅ |
| SMPD | PlaceGraph with planner (A*/Dijkstra) | `hippo_mem/spatial/map.py` L191-256【F:hippo_mem/spatial/map.py†L191-L256】 | `test_planner_astar_matches_dijkstra`【F:tests/test_spatial.py†L115-L136】 | ✅ |
| SMPD | MacroLib for replay distillation | `hippo_mem/spatial/macros.py` L16-71【F:hippo_mem/spatial/macros.py†L16-L71】 | `test_macro_replay_improves_success`【F:tests/test_spatial.py†L26-L40】 | ✅ |
| SMPD | SpatialAdapter tool interface (MQA/GQA capable) | `hippo_mem/spatial/adapter.py` L29-119【F:hippo_mem/spatial/adapter.py†L29-L119】 | `test_spatial_adapter_integration`【F:tests/test_spatial.py†L42-L64】 | ✅ |
| Shared | Hydra configs & ablation toggles | `configs/train.yaml` L9-L18【F:configs/train.yaml†L9-L18】 | `test_flash_attention_toggle`【F:tests/test_training.py†L251-L277】 | ✅ |
| Shared | Consolidation worker with 50/30/20 mix & maintenance | `scripts/train_lora.py` L97-L104【F:scripts/train_lora.py†L97-L104】, `hippo_mem/consolidation/worker.py` L23-L120【F:hippo_mem/consolidation/worker.py†L23-L120】 | `test_worker_updates_adapter`【F:tests/test_consolidation_worker.py†L18-L33】 | ✅ |
| Shared | Maintenance/rollback logging | `hippo_mem/episodic/store.py` L232-L239【F:hippo_mem/episodic/store.py†L232-L239】, `hippo_mem/relational/kg.py` L257-L285【F:hippo_mem/relational/kg.py†L257-L285】, `hippo_mem/spatial/map.py` L258-L275【F:hippo_mem/spatial/map.py†L258-L275】 | `test_store_decay_prune_and_rollback`【F:tests/test_episodic.py†L164-L189】, `test_gnn_update_and_rollback_restores_embeddings`【F:tests/test_relational.py†L90-L115】, `test_placegraph_maintenance_and_rollback`【F:tests/test_spatial.py†L66-L94】 | ✅ |

# Symbols

| Symbol | File | Description |
|---|---|---|
| `DGKey` | `hippo_mem/episodic/gating.py` | k-WTA sparse key structure |
| `WriteGate` | `hippo_mem/episodic/gating.py` | computes S from surprise/novelty/reward/pin |
| `EpisodicStore` | `hippo_mem/episodic/store.py` | FAISS+SQLite store with Hopfield completion |
| `ReplayQueue` | `hippo_mem/episodic/replay.py` | salience/recency/diversity prioritised queue |
| `ReplayScheduler` | `hippo_mem/episodic/replay.py` | mixes episodic/semantic/fresh batches |
| `EpisodicAdapter` | `hippo_mem/episodic/adapter.py` | cross-attention adapter with MQA/GQA & FlashAttn |
| `KnowledgeGraph` | `hippo_mem/relational/kg.py` | NetworkX + SQLite semantic store |
| `SchemaIndex` | `hippo_mem/relational/schema.py` | schema-based fast-track routing |
| `RelationalAdapter` | `hippo_mem/relational/adapter.py` | dual-path fusion of KG & episodic features |
| `PlaceGraph` | `hippo_mem/spatial/map.py` | path-integrated graph with A*/Dijkstra planner |
| `MacroLib` | `hippo_mem/spatial/macros.py` | store and rank procedural macros |
| `SpatialAdapter` | `hippo_mem/spatial/adapter.py` | plan/macro cross-attention adapter |
| `ConsolidationWorker` | `hippo_mem/consolidation/worker.py` | background replay & maintenance worker |
| `build_datasets.py` | `scripts/build_datasets.py` | deterministic synthetic dataset generator |
| `eval_bench.py` | `scripts/eval_bench.py` | evaluation harness with preset configs |
| `train_lora.py` | `scripts/train_lora.py` | LoRA/QLoRA training wrapper with efficiency flags |
