# Normative Requirements
| Requirement | Code Evidence | Test Evidence | Verdict |
|---|---|---|---|
| HEI‑NW: k‑WTA sparse keys | `EpisodicStore.sparse_encode`【F:hippo_mem/episodic/store.py†L149-L161】 | Covered by `test_one_shot_write_recall` | ✅ |
| HEI‑NW: Neuromodulated write gate `S=α·surprise+β·novelty+γ·reward+δ·pin` with threshold | `WriteGate.score` and `__call__`【F:hippo_mem/episodic/gating.py†L52-L106】 | `test_gating_threshold_and_pin` | ✅ |
| HEI‑NW: FAISS+PQ store with Hopfield completion | `EpisodicStore.complete`【F:hippo_mem/episodic/store.py†L263-L278】 | `test_hopfield_completion_restores_sparse_cue` | ⚠️ partial (maintenance/rollback untested) |
| HEI‑NW: CA2‑like prioritized replay (salience/recency/diversity, grad overlap) | `ReplayQueue` & `ReplayScheduler`【F:hippo_mem/episodic/replay.py†L29-L120】【F:hippo_mem/episodic/replay.py†L131-L200】 | no direct tests | ❌ missing tests |
| HEI‑NW: EpisodicAdapter cross‑attention with MQA/GQA/FlashAttn | `EpisodicAdapter` (MQA/GQA via `num_kv_heads`)【F:hippo_mem/episodic/adapter.py†L103-L148】 | not exercised for FlashAttn | ⚠️ FlashAttn missing |
| SGC‑RSS: Tuple extractor | `extract_tuples`【F:hippo_mem/relational/tuples.py†L1-L57】 | `test_tuple_precision` | ✅ |
| SGC‑RSS: KG store with embeddings & schema fast‑track | `KnowledgeGraph.upsert` & `SchemaIndex.fast_track`【F:hippo_mem/relational/kg.py†L19-L47】【F:hippo_mem/relational/schema.py†L22-L57】 | `test_schema_threshold_routes_confident_tuples` | ⚠️ partial (GNN updates untested) |
| SGC‑RSS: Dual‑path retrieval + gated fusion | `RelationalAdapter.__call__`【F:hippo_mem/relational/adapter.py†L7-L29】 | `test_dual_path_fusion_deterministic` | ✅ |
| SMPD: PlaceGraph with path integration & planner | `PlaceGraph.__init__` and `plan`【F:hippo_mem/spatial/map.py†L64-L140】【F:hippo_mem/spatial/map.py†L200-L238】 | `test_path_integration_planning` | ✅ |
| SMPD: MacroLib with behavior‑cloned macros | `MacroLib.store` & `update_stats`【F:hippo_mem/spatial/macros.py†L13-L52】 | `test_macro_replay_improves_success` | ✅ |
| SMPD: SpatialAdapter/tool interface | `SpatialAdapter.forward`【F:hippo_mem/spatial/adapter.py†L27-L83】 | `test_spatial_adapter_integration` | ✅ |
| Shared: Hydra configs & ablations | `TrainConfig` fields【F:scripts/train_lora.py†L45-L88】【F:scripts/train_lora.py†L113-L142】 | `test_adapter_ablation_flags` | ⚠️ efficiency toggles missing |
| Shared: Consolidation worker 50/30/20 mix | `ReplayScheduler.BatchMix` default 0.5/0.3/0.2【F:scripts/train_lora.py†L113-L142】 | `test_replay_flag_controls_scheduler` (partial) | ⚠️ worker error at runtime |
| Shared: Nightly decay/pruning jobs with provenance/rollback | `EpisodicStore.decay/prune/rollback`【F:hippo_mem/episodic/store.py†L280-L320】 | no tests | ❌ missing tests |
| Shared: Logging/reproducibility (git SHA etc.) | `eval_bench._git_sha`【F:scripts/eval_bench.py†L32-L45】 | metrics files generation | ⚠️ config hash missing |

# Symbols
| Symbol | File | Summary |
|---|---|---|
| `WriteGate` | `hippo_mem/episodic/gating.py` | Neuromodulated write decision.
| `EpisodicStore` | `hippo_mem/episodic/store.py` | FAISS+SQLite episodic store with Hopfield completion.
| `ReplayScheduler` | `hippo_mem/episodic/replay.py` | Mixes episodic/semantic/fresh items.
| `KnowledgeGraph` | `hippo_mem/relational/kg.py` | NetworkX/SQLite graph with embeddings.
| `SchemaIndex` | `hippo_mem/relational/schema.py` | Routes tuples to KG or episodic buffer.
| `RelationalAdapter` | `hippo_mem/relational/adapter.py` | Dual-path cross-attention fusion.
| `PlaceGraph` | `hippo_mem/spatial/map.py` | Place-like graph with optional path integration.
| `MacroLib` | `hippo_mem/spatial/macros.py` | Stores and ranks procedural macros.
| `SpatialAdapter` | `hippo_mem/spatial/adapter.py` | Cross-attention between hidden states and plans.
| `ConsolidationWorker` | `hippo_mem/consolidation/worker.py` | Background replay and maintenance worker.
