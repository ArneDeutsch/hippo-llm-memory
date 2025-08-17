# Test Gaps

1. **`faiss_index` edge cases**
   - **Spec**: `test_faiss_index_add_delete_update`
   - **Given**: a `FaissIndex` with PQ backend.
   - **When**: adding, updating and removing vectors, including failing remove.
   - **Then**: index reflects operations and raises/logs on invalid ids.
   - **Files**: `tests/test_retrieval.py`

2. **Evaluation harness presets**
   - **Spec**: `test_eval_bench_presets`
   - **Given**: run `scripts/eval_bench.py` with `preset=baselines/core|rag|longctx` and `suite`.
   - **When**: CLI executes in dry-run mode.
   - **Then**: creates expected `runs/` subdirs and metrics files.
   - **Files**: `tests/test_eval.py`

3. **Synthetic dataset determinism across suites**
   - **Spec**: `test_build_datasets_deterministic`
   - **Given**: call `scripts/build_datasets.py` for episodic/semantic/spatial with fixed seeds.
   - **When**: generating twice.
   - **Then**: JSONL outputs are identical; each suite respects `n`.
   - **Files**: `tests/test_datasets.py`
