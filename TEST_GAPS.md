# Test Gaps

## Missing or Weak Tests

1. **Schema fast-track routing**
   - *Spec*: `test_schema_fast_track_threshold`
   - *Given*: tuples around the SchemaIndex threshold.
   - *When*: ingesting tuples with confidence just below and above the threshold.
   - *Then*: only above-threshold tuples are written to KG; others stay buffered.
   - *Files*: `tests/test_relational.py`

2. **k-WTA sparse encoding**
   - *Spec*: `test_kwta_produces_sparse_indices`
   - *Given*: dense vectors and k value.
   - *When*: applying k-WTA projection.
   - *Then*: exactly `k` non-zero entries; repeated projection yields same indices.
   - *Files*: `tests/test_episodic.py`

3. **Retrieval layer utilities**
   - *Spec*: `test_faiss_index_add_search`
   - *Given*: embeddings and queries.
   - *When*: using `FaissIndex.add` and `search`.
   - *Then*: nearest neighbour ids match expectation.
   - *Files*: `tests/test_retrieval.py`

4. **Train script ablation toggles**
   - *Spec*: `test_train_respects_hopfield_flag`
   - *Given*: `episodic.hopfield=false`.
   - *When*: running `train_lora.py` dry-run.
   - *Then*: `EpisodicStore.complete` is bypassed.
   - *Files*: `tests/test_training.py`

## Property-Based Tests (Hypothesis)

- k-WTA produces exactly `k` active units and is idempotent under masking.
- Replay scheduler never schedules two items with cosine similarity above τ consecutively when possible.
- Schema fast-track never promotes tuples below confidence threshold.
- Planner returns optimal path on random weighted DAG fixtures.

## Mutation Testing

`mutmut` failed due to missing `paths_to_mutate` configuration; add minimal config and target `hippo_mem/episodic` and `hippo_mem/relational` for future runs.
