# Refactor Opportunities
| Priority | Component | Issue | Rationale | Suggested Refactor |
|---|---|---|---|---|
| P0 | `ConsolidationWorker.run` | Cyclomatic complexity C=16, thread error on missing gradients | Hard to reason about background replay; risk of silent failures | Split into smaller helpers (`_process_batch`, `_apply_loss`) and add explicit gradient checks |
| P1 | `EpisodicStore.prune` | Manual SQL string concatenation flagged by Bandit; complexity B | Risk of SQL injection and maintenance errors | Use parametrized queries via `WHERE salience < ? AND ts < ?` built incrementally |
| P1 | `KnowledgeGraph.upsert` | Long function mixing DB writes and graph updates | Difficult to unit test; coverage gaps | Extract DB interaction into helper; add transactional context |
| P2 | `scripts/eval_bench` imports at mid-file | `E402` lint errors and readability | Ensure top-level imports at top of file | Reorder imports and enforce `flake8` compliance |
| P2 | Line-length violations across modules | Numerous E501 warnings | Hinders readability and CI passes | Apply `black` formatting and wrap long strings |
