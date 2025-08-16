# Refactor Opportunities

1. **Centralise k-WTA encoding**
   - `DGKey` defined but unused; implement a utility to generate sparse keys and update `EpisodicStore.write` to accept `DGKey`.

2. **Relational schema scoring**
   - `SchemaIndex.score` only checks relation equality. Factor into pluggable similarity metrics and unit-test.

3. **Retrieval utilities**
   - `hippo_mem/retrieval/embed.py` and `faiss_index.py` are unused and untested. Either integrate into stores or remove.

4. **Training script resilience**
   - `ConsolidationWorker` raises `ValueError` when adapters have no LoRA params, causing dry-run failures. Add graceful skip.

5. **Line length & import order**
   - Widespread `E501` and `E402` flake8 violations (see `flake8` report). Adopt formatter or adjust max line length.
