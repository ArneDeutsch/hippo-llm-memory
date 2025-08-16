# Missing or Weak Tests

1. **test_hopfield_completion_restores_sparse_cue**
   - *Given* an EpisodicStore with multiple traces and hopfield enabled
   - *When* recalling with a noisy cue
   - *Then* `complete` should reconstruct the original sparse key within a tolerance.

2. **test_schema_fast_track_routing_threshold**
   - *Given* tuples around the schema confidence threshold
   - *When* `SchemaIndex.fast_track` is invoked
   - *Then* tuples with confidence ≥ threshold are inserted into the KG; others remain in episodic buffer.

3. **test_spatial_adapter_integration**
   - *Given* a `SpatialAdapter` with a simple map and macros
   - *When* adapter is applied to hidden states and place embeddings
   - *Then* output dimensions match and gradients flow.

4. **test_ablation_toggles_effect**
   - *Given* training/eval configs with ablations (e.g., `memory.episodic.hopfield=false`)
   - *When* running `train_lora.py` or `eval_bench.py`
   - *Then* modules respect toggles (e.g., Hopfield completion skipped).

5. **test_replay_scheduler_similarity_guard**
   - *Given* replay queue items with high cosine similarity
   - *When* sampling successive batches
   - *Then* no two consecutive items exceed the similarity threshold when alternatives exist (property-based).
