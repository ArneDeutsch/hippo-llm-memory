# Normative Requirements

| Requirement | Design Ref | Code | Test | Verdict |
|-------------|------------|------|------|---------|
| k-WTA sparse keys (`DGKey`) used with FAISS+PQ store | DESIGN.md §5.2, §4.1 | `hippo_mem/episodic/store.py` lacks k-WTA; `DGKey` dataclass defined but unused | `tests/test_episodic.py::test_one_shot_write_recall` uses dense vectors | ❌ Missing | 
| Write gate S=α·surprise+β·novelty+γ·reward+δ·pin, τ threshold | DESIGN.md §5.1 | `hippo_mem/episodic/gating.py::WriteGate` | `tests/test_episodic.py::test_gating_threshold_and_pin` | ✅ | 
| Modern-Hopfield completion on recall | DESIGN.md §5.2 | `hippo_mem/episodic/store.py::complete` | `tests/test_episodic.py::test_hopfield_completion_restores_sparse_cue` | ✅ |
| CA2-like replay with salience/recency/diversity & grad-overlap check | DESIGN.md §5.3 | `hippo_mem/episodic/replay.py::ReplayQueue.sample` & `ReplayScheduler` | `tests/test_episodic.py::test_replay_queue_avoids_consecutive_gradients`, `test_replay_scheduler_mix_and_unique_ids` | ✅ |
| Tuple extractor & schema fast-track routing | DESIGN.md §5.4 | `hippo_mem/relational/tuples.py::extract_tuples`, `schema.py::SchemaIndex.fast_track` | `tests/test_relational.py::test_schema_fast_track` (missing) | ⚠️ Partial |
| Dual-path retrieval + gated fusion | DESIGN.md §5.4 | `hippo_mem/relational/adapter.py` | `tests/test_relational.py::test_adapter_fusion_deterministic` | ✅ |
| PlaceGraph with optional path integration & A*/Dijkstra planner | DESIGN.md §5.5 | `hippo_mem/spatial/map.py::PlaceGraph` | `tests/test_spatial.py::test_planner_equivalence` | ✅ |
| Macro library for replay-to-policy | DESIGN.md §5.5 | `hippo_mem/spatial/macros.py::MacroLib` | `tests/test_spatial.py::test_macro_suggestion_improves` | ✅ |
| Hydra configs and ablation toggles | DESIGN.md §7, §10 | `configs/` YAMLs & dataclass configs | `tests/test_eval_plumbing.py` | ✅ |
| Consolidation worker with 50/30/20 mix & maintenance jobs | DESIGN.md §9, §11 | `hippo_mem/consolidation/worker.py` | `tests/test_consolidation_worker.py` | ✅ |

# Symbols

| Symbol | Path | Summary |
|--------|------|---------|
| `WriteGate` | `hippo_mem/episodic/gating.py` | Neuromodulated write decision |
| `EpisodicStore` | `hippo_mem/episodic/store.py` | FAISS-backed store with Hopfield completion |
| `ReplayQueue` | `hippo_mem/episodic/replay.py` | Priority queue mixing salience, recency, diversity |
| `ReplayScheduler` | `hippo_mem/episodic/replay.py` | Batch mixer for episodic/semantic/fresh items |
| `EpisodicAdapter` | `hippo_mem/episodic/adapter.py` | Cross-attention with optional FlashAttn & GQA |
| `extract_tuples` | `hippo_mem/relational/tuples.py` | Heuristic tuple extractor |
| `KnowledgeGraph` | `hippo_mem/relational/kg.py` | NetworkX+SQLite store with embeddings |
| `SchemaIndex` | `hippo_mem/relational/schema.py` | Fast-track routing for schema-fit tuples |
| `RelationalAdapter` | `hippo_mem/relational/adapter.py` | Gated fusion of KG and episodic features |
| `PlaceGraph` | `hippo_mem/spatial/map.py` | Deterministic map with optional path integration |
| `MacroLib` | `hippo_mem/spatial/macros.py` | Stores and ranks macros |
| `SpatialAdapter` | `hippo_mem/spatial/adapter.py` | Cross-attention over plan/macro embeddings |
| `ConsolidationWorker` | `hippo_mem/consolidation/worker.py` | Background replay & maintenance |
