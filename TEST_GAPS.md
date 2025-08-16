# Missing or Weak Tests
| Priority | Name | Given | When | Then | Files |
|---|---|---|---|---|---|
| P0 | `test_replay_scheduler_mix_ratios` | scheduler with default BatchMix | calling `next_batch` on empty KG | batch contains `0.5/0.3/0.2` split and no duplicates | `tests/test_episodic.py` |
| P0 | `test_episodic_maintenance_rollback` | store with several traces and maintenance config | decay and prune run; then `rollback(1)` | salience restored and pruned traces returned | `tests/test_episodic.py` |
| P1 | `test_kg_gnn_updates_and_rollback` | KG with embeddings | upsert edges then `rollback(1)` | node embeddings and edges revert | `tests/test_relational.py` |
| P1 | `test_placegraph_rollback_and_decay` | PlaceGraph with path integration | run `decay` and `prune`, then `rollback` | coordinates and edges restored | `tests/test_spatial.py` |
| P1 | `test_faiss_index_train_and_query` | EpisodicStore with many writes | training threshold exceeded | PQ index becomes trained and recalls succeed | `tests/test_episodic.py` |
| P2 | `test_relational_adapter_gating_ablation` | RelationalAdapter with KG & episodic feats | toggle ablation flag | output respects gated fusion weights | `tests/test_relational.py` |
| P2 | `test_eval_bench_records_config_hash` | eval_bench run with memory modules | metrics file inspected | `meta.json` contains `config_hash` and seed | `tests/test_eval_plumbing.py` |

# Property-Based Tests (Hypothesis)
- **k‑WTA sparsity**: generating random vectors, `sparse_encode` should yield exactly `k` non‑zeros and decoding should be idempotent.
- **Replay similarity constraint**: for distinct keys with cosine similarity >τ, scheduler should not return them consecutively when queue size allows.
- **Schema fast‑track threshold**: for tuples near threshold, `fast_track` should promote only when `conf ≥ threshold`.
- **Planner optimality**: generated weighted DAG fixtures; `plan(method="astar")` must equal Dijkstra result and match known shortest path length.
