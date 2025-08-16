# Test Gaps

| Priority | Area | Gap | Proposed Test |
| --- | --- | --- | --- |
| P0 | Relational KG | `KnowledgeGraph.prune` lacks edge/node rollback and embedding assertions | `test_kg_prune_rollback_restores_graph`: create edges with embeddings, prune by confidence, rollback, assert embeddings and topology restored. |
| P0 | Relational Schema | `SchemaIndex.flush` untested; fast-track negative path only partially covered | `test_schema_flush_promotes_buffered_tuples`: buffer low-confidence tuples then increase threshold and flush; ensure promotion and buffer empty. |
| P1 | Retrieval | `faiss_index.py` functions not exercised | `test_faiss_index_add_search_delete`: build index, add vectors, search, remove id, assert results; include PQ training branch. |
| P1 | Adapters | `hippo_mem/adapters/lora.py` utilities untested | `test_lora_adapter_load_merge_export`: create tiny model, load adapter weights, merge, export; assert weight changes. |
| P1 | Episodic Adapter | GQA head expansion logic untested | `test_episodic_adapter_gqa_expansion`: set `num_kv_heads=1` and `num_heads>1`, assert `_expand_kv` duplicates keys correctly. |
| P2 | Efficiency flags | No test for MQA/GQA config toggles in `train_lora.py` | `test_train_respects_mqa_gqa_flag`: override config, verify `num_kv_heads` updated and adapters receive grouped heads. |
| P2 | CLI tools | `scripts/export_adapter.py` lacks coverage | `test_export_adapter_cli`: run script on dummy adapter, verify output file exists. |
