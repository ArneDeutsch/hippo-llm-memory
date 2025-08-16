# Normative Requirements

| Requirement | Code | Tests | Verdict |
| --- | --- | --- | --- |
| k-WTA sparse keys via `DGKey` | `hippo_mem/episodic/gating.py` | `tests/test_episodic.py::test_kwta_produces_sparse_indices` | ✅ |
| FAISS+PQ episodic store with Hopfield completion | `hippo_mem/episodic/store.py` | `tests/test_episodic.py::test_hopfield_completion_restores_sparse_cue` | ✅ |
| Neuromodulated write gate `S=α·surprise+β·novelty+γ·reward+δ·pin` | `hippo_mem/episodic/gating.py` | `tests/test_episodic.py::test_gating_threshold_and_pin` | ✅ |
| CA2-like replay prioritising salience/recency/diversity & grad-overlap | `hippo_mem/episodic/replay.py` | `tests/test_replay_queue.py::test_replay_queue_similarity_constraint`; `tests/test_replay_scheduler.py` | ✅ |
| EpisodicAdapter with GQA & FlashAttention hooks | `hippo_mem/episodic/adapter.py` | `tests/test_episodic.py::test_flash_attention_flag` | ⚠️ (FlashAttention optional; no GQA toggle test) |
| Tuple extractor with ≥0.9 precision | `hippo_mem/relational/tuples.py` | `tests/test_relational.py::test_tuple_precision` | ✅ |
| KnowledgeGraph store with embeddings & retrieval | `hippo_mem/relational/kg.py` | `tests/test_relational.py::test_multi_hop_retrieval` | ⚠️ (no GNN updates) |
| SchemaIndex fast-track routing | `hippo_mem/relational/schema.py` | `tests/test_relational.py::test_schema_fast_track_threshold` | ✅ |
| Dual-path retrieval & gated fusion | `hippo_mem/relational/adapter.py` | `tests/test_relational.py::test_dual_path_fusion_deterministic` | ✅ |
| PlaceGraph with path integration & planner | `hippo_mem/spatial/map.py` | `tests/test_spatial.py::test_path_integration_planning`; `tests/test_spatial.py::test_planner_astar_matches_dijkstra` | ✅ |
| MacroLib storing & ranking macros | `hippo_mem/spatial/macros.py` | `tests/test_spatial.py::test_macro_replay_improves_success` | ✅ |
| SpatialAdapter cross‑attends plan embeddings | `hippo_mem/spatial/adapter.py` | `tests/test_spatial.py::test_spatial_adapter_integration` | ✅ |
| Hydra configs & ablation flags | `scripts/train_lora.py`, `configs/` | `tests/test_training.py::test_adapter_ablation_flags` | ✅ |
| Consolidation worker with 50/30/20 mix & maintenance jobs | `hippo_mem/consolidation/worker.py`, `hippo_mem/episodic/replay.py` | `tests/test_consolidation_worker.py::test_worker_records_maintenance_logs` | ✅ |
| Provenance & rollback for stores | `hippo_mem/episodic/store.py`, `hippo_mem/relational/kg.py`, `hippo_mem/spatial/map.py` | `tests/test_episodic.py::test_store_decay_prune_and_rollback`; `tests/test_relational.py::test_knowledgegraph_maintenance_log_records_events`; `tests/test_spatial.py::test_placegraph_maintenance_and_rollback` | ✅ |
| Reproducibility via fixed seeds & config hashes | `scripts/train_lora.py`, `scripts/eval_bench.py` | `tests/test_training.py::test_train_sets_seeds`; `tests/test_eval_plumbing.py` | ✅ |
| FlashAttention/MQA/GQA efficiency options | `scripts/train_lora.py`, adapters | `tests/test_training.py::test_flash_attention_toggle` | ⚠️ (MQA/GQA toggles missing) |
| CI lint & test workflow | `.github/workflows/ci.yml` | GitHub Actions | ✅ |

# Symbols

| Symbol | File | Description |
| --- | --- | --- |
| `DGKey` | `hippo_mem/episodic/gating.py` | Sparse key structure for k‑WTA encoding. |
| `k_wta` | `hippo_mem/episodic/gating.py` | Selects top‑k indices to form `DGKey`. |
| `WriteGate` | `hippo_mem/episodic/gating.py` | Computes neuromodulated write scores. |
| `TraceValue` | `hippo_mem/episodic/store.py` | Metadata for episodic traces. |
| `EpisodicStore` | `hippo_mem/episodic/store.py` | FAISS‑backed store with Hopfield completion, decay, prune, rollback. |
| `ReplayQueue` | `hippo_mem/episodic/replay.py` | Prioritised replay queue with gradient overlap avoidance. |
| `ReplayScheduler` | `hippo_mem/episodic/replay.py` | Mixes episodic/semantic/fresh batches. |
| `EpisodicAdapter` | `hippo_mem/episodic/adapter.py` | Cross‑attention adapter supporting LoRA, GQA and FlashAttention. |
| `extract_tuples` | `hippo_mem/relational/tuples.py` | Heuristic tuple extractor. |
| `KnowledgeGraph` | `hippo_mem/relational/kg.py` | NetworkX+SQLite graph with embeddings and pruning/rollback. |
| `SchemaIndex` | `hippo_mem/relational/schema.py` | Routes tuples to KG via fast‑track threshold. |
| `RelationalAdapter` | `hippo_mem/relational/adapter.py` | Dual‑path attention and gated fusion. |
| `PlaceGraph` | `hippo_mem/spatial/map.py` | Context graph with optional path integration and A*/Dijkstra planner. |
| `MacroLib` | `hippo_mem/spatial/macros.py` | Stores trajectories and ranks macros by success. |
| `SpatialAdapter` | `hippo_mem/spatial/adapter.py` | Cross‑attention over plan/macro embeddings. |
| `ConsolidationWorker` | `hippo_mem/consolidation/worker.py` | Background trainer mixing replay batches and maintenance jobs. |
| `train` | `scripts/train_lora.py` | LoRA/QLoRA training loop with seed control and adapter toggles. |
| `evaluate` | `scripts/eval_bench.py` | Evaluation harness producing metrics and metadata. |
| `generate_episodic/semantic/spatial` | `scripts/build_datasets.py` | Deterministic synthetic task generators. |
