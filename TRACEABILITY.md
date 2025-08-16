# Normative Requirements

| Requirement | Code | Tests | Verdict |
| --- | --- | --- | --- |
| Episodic: k-WTA sparse keys via `DGKey` | `hippo_mem/episodic/store.py` | `tests/test_episodic.py::test_one_shot_write_recall` | ✅ |
| Episodic: FAISS+PQ store | `hippo_mem/episodic/store.py` | `tests/test_episodic.py` | ✅ |
| Episodic: Hopfield completion | `hippo_mem/episodic/store.py::complete` | *(no direct test)* | ⚠️ |
| Episodic: Neuromodulated write gate | `hippo_mem/episodic/gating.py` | `tests/test_episodic.py::test_gating_threshold_and_pin` | ✅ |
| Episodic: Prioritized replay (salience/recency/diversity) | `hippo_mem/episodic/replay.py` | `tests/test_replay_scheduler.py` | ✅ |
| Episodic: Adapter cross-attention with MQA/GQA hooks | `hippo_mem/episodic/adapter.py` | `tests/test_training.py` (adapter wiring) | ⚠️ (no FlashAttn)
| Relational: Tuple extractor | `hippo_mem/relational/tuples.py` | `tests/test_relational.py::test_tuple_precision` | ✅ |
| Relational: KG store with embeddings | `hippo_mem/relational/kg.py` | `tests/test_relational.py::test_multi_hop_retrieval` | ✅ |
| Relational: Schema index + fast-track routing | `hippo_mem/relational/schema.py` (unused) | *(no test)* | ❌ |
| Relational: Dual-path retrieval + gated fusion | `hippo_mem/relational/adapter.py` | `tests/test_relational.py::test_dual_path_fusion_deterministic` | ✅ |
| Spatial: PlaceGraph with optional path integration | `hippo_mem/spatial/map.py` | `tests/test_spatial.py::test_path_integration_planning` | ✅ |
| Spatial: Planner (A*/Dijkstra) | `hippo_mem/spatial/map.py` | `tests/test_spatial.py::test_path_integration_planning` | ✅ |
| Spatial: MacroLib for behaviour-cloned macros | `hippo_mem/spatial/macros.py` | `tests/test_spatial.py::test_macro_replay_improves_success` | ✅ |
| Spatial: SpatialAdapter/tool interface | `hippo_mem/spatial/adapter.py` | `tests/test_training.py` | ⚠️ (limited use)
| Shared: Hydra configs & ablations | `configs/` | `tests/test_training.py` | ✅ |
| Shared: Consolidation worker 50/30/20 batch mix | `hippo_mem/consolidation/worker.py` & `hippo_mem/episodic/replay.py` | `tests/test_consolidation_worker.py` | ✅ |
| Shared: Nightly decay/pruning jobs | hooks in stores but no scheduler | *(no test)* | ⚠️ |
| Shared: Provenance/rollback | `TraceValue.provenance` | *(no rollback)* | ⚠️ |
| Shared: Logging | `_log` in stores | `tests/test_eval_plumbing.py` | ✅ |
| Milestones M1–M7 files present | repository | `tests/` | ✅ |

# Symbols

<!-- generated from symbol_inventory.json -->

<!-- include symbol table -->
| Symbol | File | Docstring |
|---|---|---|
| hippo_mem.adapters.lora.export_adapter | hippo_mem/adapters/lora.py | Save the adapter weights to ``output_dir``. |
| hippo_mem.adapters.lora.load_adapter | hippo_mem/adapters/lora.py | Load a LoRA adapter from ``adapter_path`` and attach it to ``base_model``. |
| hippo_mem.adapters.lora.merge_adapter | hippo_mem/adapters/lora.py | Merge the LoRA weights into the base model and return it. |
| hippo_mem.consolidation.worker.ConsolidationWorker | hippo_mem/consolidation/worker.py | Background thread that fine‑tunes memory adapters using replay. |
| hippo_mem.episodic.adapter.AdapterConfig | hippo_mem/episodic/adapter.py | Configuration options for :class:`EpisodicAdapter`. |
| hippo_mem.episodic.adapter.EpisodicAdapter | hippo_mem/episodic/adapter.py | Cross-attention over recalled episodic traces. |
| hippo_mem.episodic.adapter.LoraLinear | hippo_mem/episodic/adapter.py | ``nn.Linear`` with an optional LoRA adaptation. |
| hippo_mem.episodic.gating.GateDecision | hippo_mem/episodic/gating.py | Result of a write-gating decision. |
| hippo_mem.episodic.gating.WriteGate | hippo_mem/episodic/gating.py | Combine surprise, novelty and reward/pin signals into a write decision. |
| hippo_mem.episodic.gating.novelty | hippo_mem/episodic/gating.py | Compute novelty as ``1 - max_cos`` between ``query`` and stored ``keys``. |
| hippo_mem.episodic.gating.surprise | hippo_mem/episodic/gating.py | Return the information content ``-log(p)`` of an event. |
| hippo_mem.episodic.replay.BatchMixLike | hippo_mem/episodic/replay.py | Protocol describing the batch mix structure. |
| hippo_mem.episodic.replay.ReplayItem | hippo_mem/episodic/replay.py | Metadata stored for replay scheduling. |
| hippo_mem.episodic.replay.ReplayQueue | hippo_mem/episodic/replay.py | Replay queue mixing gating score, recency and diversity. |
| hippo_mem.episodic.replay.ReplayScheduler | hippo_mem/episodic/replay.py | Scheduler that interleaves episodic, semantic and fresh items. |
| hippo_mem.episodic.store.DGKey | hippo_mem/episodic/store.py | Sparse k-WTA encoded key. |
| hippo_mem.episodic.store.EpisodicStore | hippo_mem/episodic/store.py | Simple vector store for episodic memories. |
| hippo_mem.episodic.store.Trace | hippo_mem/episodic/store.py | A retrieved memory trace. |
| hippo_mem.episodic.store.TraceValue | hippo_mem/episodic/store.py | Metadata associated with a stored trace. |
| hippo_mem.relational.adapter.RelationalAdapter | hippo_mem/relational/adapter.py | Stub adapter bridging model, KG and episodic memory. |
| hippo_mem.relational.kg.KnowledgeGraph | hippo_mem/relational/kg.py | Knowledge graph backed by NetworkX and SQLite. |
| hippo_mem.relational.schema.Schema | hippo_mem/relational/schema.py | Schema(name: 'str', relation: 'str', head_type: 'Optional[str]' = None, tail_type: 'Optional[str]' = None) |
| hippo_mem.relational.schema.SchemaIndex | hippo_mem/relational/schema.py | Store schema prototypes and route tuples based on confidence. |
| hippo_mem.relational.tuples._parse_triplet | hippo_mem/relational/tuples.py | Very small heuristic parser returning ``(head, relation, tail)``. |
| hippo_mem.relational.tuples.extract_tuples | hippo_mem/relational/tuples.py | Extract ``(head, relation, tail, context, time, conf, provenance)`` tuples. |
| hippo_mem.retrieval.embed.embed_text | hippo_mem/retrieval/embed.py | Return a deterministic placeholder embedding for ``text``. |
| hippo_mem.retrieval.faiss_index.FaissIndex | hippo_mem/retrieval/faiss_index.py | Very small wrapper around :mod:`faiss` or a Python fallback. |
| hippo_mem.spatial.adapter.AdapterConfig | hippo_mem/spatial/adapter.py | Configuration for :class:`SpatialAdapter`. |
| hippo_mem.spatial.adapter.SpatialAdapter | hippo_mem/spatial/adapter.py | Cross-attention between LLM states and plan/macro embeddings. |
| hippo_mem.spatial.macros.Macro | hippo_mem/spatial/macros.py | A reusable action sequence. |
| hippo_mem.spatial.macros.MacroLib | hippo_mem/spatial/macros.py | In‑memory store for macros. |
| hippo_mem.spatial.map.ContextEncoder | hippo_mem/spatial/map.py | Deterministic context→coordinate encoder. |
| hippo_mem.spatial.map.Edge | hippo_mem/spatial/map.py | Connection information between two places. |
| hippo_mem.spatial.map.Place | hippo_mem/spatial/map.py | A place in the environment with pseudo coordinates. |
| hippo_mem.spatial.map.PlaceGraph | hippo_mem/spatial/map.py | Graph of observed places with light‑weight planning. |