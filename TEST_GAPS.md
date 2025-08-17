# Test Gaps

1. **test_faiss_index_edge_cases** *(P1)*
   - **Given** an EpisodicStore with PQ index and pending keys.
   - **When** recalls are issued before training and after index training.
   - **Then** search results should be consistent and handle zero hits.

2. **test_schema_mismatch_replay_weighting** *(P1)*
   - **Given** tuples that fall below SchemaIndex threshold.
   - **When** ReplayScheduler mixes episodic vs semantic items.
   - **Then** low-confidence tuples remain episodic with heavier replay share.

3. **test_planner_optimality_property** *(P1)*
   - **Given** randomly generated DAGs with positive edge costs (Hypothesis).
   - **When** planning with A* and Dijkstra.
   - **Then** returned paths and costs are identical and minimal.

4. **test_consolidation_worker_fault_logging** *(P2)*
   - **Given** a failing adapter step.
   - **When** ConsolidationWorker encounters an exception.
   - **Then** worker should log the failure and continue with remaining batches.

5. **test_eval_provenance_hash** *(P2)*
   - **Given** evaluation runs with different presets and seeds.
   - **When** metrics are written.
   - **Then** `meta.json` must include `config_hash` and `seed` for reproducibility.
