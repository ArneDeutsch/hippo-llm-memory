# Refactor Opportunities

1. **ConsolidationWorker.run complexity** *(P1)*
   - **Risk**: Cyclomatic complexity B (7) and long method with mixed concerns.
   - **Rationale**: Split into smaller methods for queue polling, adapter stepping and error handling to ease testing.
   - **Files**: `hippo_mem/consolidation/worker.py`.

2. **EpisodicStore monolith** *(P2)*
   - **Risk**: Store handles FAISS, SQLite, Hopfield and maintenance in one class (~250 lines).
   - **Rationale**: Extract FAISS/SQLite helpers into separate modules to reduce coupling and allow targeted tests.
   - **Files**: `hippo_mem/episodic/store.py`.

3. **Evaluation harness configuration** *(P2)*
   - **Risk**: `eval_bench.py` mixes CLI parsing, dataset generation and metric writing; low coverage (74%).
   - **Rationale**: Separate runner logic from CLI; add fixtures for provenance logging.
   - **Files**: `scripts/eval_bench.py`.

4. **FAISS index wrapper** *(P3)*
   - **Risk**: Low coverage (68%) and multiple untested branches for training/remove paths.
   - **Rationale**: Simplify API and add explicit error handling.
   - **Files**: `hippo_mem/retrieval/faiss_index.py`.
