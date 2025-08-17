# Normative Requirements
| Requirement | Code | Tests | Verdict |
|-------------|------|-------|---------|
| HEI‑NW: k‑WTA DGKey, FAISS+PQ store | `episodic/gating.py`, `episodic/store.py` | `tests/test_episodic.py::test_sparse_encode_k_wta_idempotent` | ✅ |
| HEI‑NW: modern‑Hopfield completion | `episodic/store.py::complete` | `tests/test_episodic.py::test_hopfield_completion_restores_sparse_cue` | ✅ |
| HEI‑NW: write gate S=α·surprise+β·novelty+γ·reward+δ·pin | `episodic/gating.py::WriteGate` | `tests/test_episodic.py::test_gating_threshold_and_pin` | ⚠️ pin bypasses δ term |
| HEI‑NW: CA2‑like replay (salience/recency/diversity, grad‑overlap) | `episodic/replay.py::ReplayScheduler` | `tests/test_replay_scheduler.py` | ✅ |
| SGC‑RSS: tuple extractor & SchemaIndex fast‑track | `relational/tuples.py`, `relational/schema.py` | `tests/test_relational.py::test_schema_fast_track_routing_threshold` | ✅ |
| SGC‑RSS: KnowledgeGraph with embeddings and pruning | `relational/kg.py` | `tests/test_relational.py::test_prune_max_age_removes_stale_edges_and_orphans` | ✅ |
| SGC‑RSS: dual‑path retrieval + gated fusion | `relational/adapter.py` | `tests/test_relational.py::test_dual_path_fusion_deterministic` | ✅ |
| SMPD: PlaceGraph with path integration & planner | `spatial/map.py` | `tests/test_spatial.py::test_path_integration_planning`, `test_planner_astar_matches_dijkstra` | ✅ |
| SMPD: MacroLib replay‑to‑policy | `spatial/macros.py` | `tests/test_spatial.py::test_macro_replay_improves_success` | ✅ |
| SpatialAdapter with MQA/GQA | `spatial/adapter.py` | `tests/test_spatial.py::test_spatial_adapter_integration` | ✅ |
| Consolidation worker 50/30/20 mix | `consolidation/worker.py` & `episodic/replay.py` | `tests/test_consolidation_worker.py` | ✅ |
| Hydra configs & ablations | `configs/*` | `tests/test_eval_plumbing.py` | ✅ |
| Nightly maintenance & rollback hooks | `episodic/store.py`, `relational/kg.py`, `spatial/map.py` | `tests/test_episodic.py::test_store_decay_prune_and_rollback`, etc. | ✅ |

# Symbols
| Symbol | File |
|--------|------|
| `DGKey`, `k_wta`, `WriteGate` | `hippo_mem/episodic/gating.py` |
| `EpisodicStore` | `hippo_mem/episodic/store.py` |
| `ReplayQueue`, `ReplayScheduler` | `hippo_mem/episodic/replay.py` |
| `RelationalAdapter`, `KnowledgeGraph`, `SchemaIndex` | `hippo_mem/relational/{adapter.py,kg.py,schema.py}` |
| `PlaceGraph`, `MacroLib`, `SpatialAdapter` | `hippo_mem/spatial/{map.py,macros.py,adapter.py}` |
| `ConsolidationWorker` | `hippo_mem/consolidation/worker.py` |
