# Missing or Weak Tests

1. **test_hopfield_completion_restores_sparse_cue** — FIXED
   - *Given* an EpisodicStore with multiple traces and hopfield enabled
   - *When* recalling with a noisy cue
   - *Then* `complete` should reconstruct the original sparse key within a tolerance.

2. **test_schema_fast_track_routing_threshold** — FIXED
   - *Given* tuples around the schema confidence threshold
   - *When* `SchemaIndex.fast_track` is invoked
   - *Then* tuples with confidence ≥ threshold are inserted into the KG; others remain in episodic buffer.

3. **test_spatial_adapter_integration** — FIXED
   - *Given* a `SpatialAdapter` with a simple map and macros
   - *When* adapter is applied to hidden states and place embeddings
   - *Then* output dimensions match and gradients flow.

4. **test_ablation_toggles_effect** — NOT RELEVANT
   - Existing tests already cover ablation flags for training and evaluation.

5. **test_replay_scheduler_similarity_guard** — NOT RELEVANT
   - Replay scheduler uses diversity weighting but has no explicit similarity threshold.
