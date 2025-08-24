# Public API surface

## HEI-NW (episodic)
- `hippo_mem.episodic.gating.DGKey`
- `hippo_mem.episodic.gating.k_wta()`
- `hippo_mem.episodic.gating.densify()`
- `hippo_mem.episodic.gating.surprise()`
- `hippo_mem.episodic.gating.novelty()`
- `hippo_mem.episodic.gating.WriteGate`
- `hippo_mem.episodic.types.TraceValue`
- `hippo_mem.episodic.store.Trace`
- `hippo_mem.episodic.store.EpisodicStore`
- `hippo_mem.episodic.replay.ReplayQueue`
- `hippo_mem.episodic.replay.ReplayScheduler`
- `hippo_mem.episodic.adapter.AdapterConfig`
- `hippo_mem.episodic.adapter.EpisodicAdapter`
- `hippo_mem.episodic.adapter.LoraLinear`
- `hippo_mem.episodic.index.FaissIndex`
- `hippo_mem.episodic.index.NumpyIndex`
- `hippo_mem.episodic.db.TraceDB`

## SGC-RSS (relational)
- `hippo_mem.relational.tuples.TupleType`
- `hippo_mem.relational.tuples.extract_tuples()`
- `hippo_mem.relational.tuples.split_sentences()`
- `hippo_mem.relational.tuples.strip_time()`
- `hippo_mem.relational.tuples.score_confidence()`
- `hippo_mem.relational.schema.Schema`
- `hippo_mem.relational.schema.SchemaIndex`
- `hippo_mem.relational.kg.KnowledgeGraph`
- `hippo_mem.relational.adapter.RelationalAdapter`

## SMPD (spatial)
- `hippo_mem.spatial.map.Edge`
- `hippo_mem.spatial.map.Place`
- `hippo_mem.spatial.map.ContextEncoder`
- `hippo_mem.spatial.map.PlaceGraph`
- `hippo_mem.spatial.macros.Macro`
- `hippo_mem.spatial.macros.MacroLib`
- `hippo_mem.spatial.adapter.AdapterConfig`
- `hippo_mem.spatial.adapter.SpatialAdapter`

## Shared utilities
- `hippo_mem.retrieval.embed.embed_text()`
- `hippo_mem.retrieval.faiss_index.FaissIndex`
- `hippo_mem.adapters.lora.load_adapter()`
- `hippo_mem.adapters.lora.merge_adapter()`
- `hippo_mem.adapters.lora.export_adapter()`
- `hippo_mem.consolidation.worker.ConsolidationWorker`
- `_hippo_retrieval_cb(hidden: Tensor) -> MemoryTokens`
- `hippo_mem.common.gates.GateDecision`
