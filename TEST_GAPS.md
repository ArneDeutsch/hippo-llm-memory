# Test Gaps

All previously identified gaps now have corresponding tests:

1. **test_faiss_index_edge_cases** – covered by `tests/test_retrieval.py::test_faiss_index_edge_cases`.
2. **test_schema_mismatch_replay_weighting** – covered by `tests/test_replay_scheduler.py::test_schema_mismatch_replay_weighting`.
3. **test_planner_optimality_property** – covered by `tests/test_spatial.py::test_planner_optimality_property`.
4. **test_consolidation_worker_fault_logging** – worker stops on failure and is covered by `tests/test_consolidation_worker.py::test_handle_errors_logs_and_stops`.
5. **test_eval_provenance_hash** – covered by `tests/test_eval_plumbing.py::test_eval_bench`.
