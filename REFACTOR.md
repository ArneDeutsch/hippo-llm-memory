# Refactor Opportunities

1. **EpisodicStore SQL and Error Handling** (`hippo_mem/episodic/store.py`)
   - Risk: f-string SQL and silent exception handlers (`bandit` B608/B110).
   - Proposal: switch to parameterised queries, log exceptions, add rollback logic.

2. **SchemaIndex Integration** (`hippo_mem/relational/schema.py`)
   - Risk: currently unused; routing logic scattered.
   - Proposal: refactor KG ingestion to call `SchemaIndex.fast_track` and consolidate episodic buffer management.

3. **SpatialAdapter Coverage & API** (`hippo_mem/spatial/adapter.py`)
   - Risk: 42% coverage and minimal use; forward passes lack validation.
   - Proposal: break out helper functions, add type hints and docstrings, and expand tests.

4. **Long Lines and Lint Failures** (multiple files)
   - Risk: flake8 reports >100 E501 violations, reducing readability.
   - Proposal: wrap lines at 79 chars, run formatter (e.g., `black`) and enable in CI.

5. **ConsolidationWorker Optimiser Step** (`hippo_mem/consolidation/worker.py`)
   - Risk: optimizer step executed even when no params; may raise at runtime.
   - Proposal: guard `optimizer.step()` and expose learning-rate schedule.
