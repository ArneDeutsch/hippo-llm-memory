# Refactor Opportunities

| Area | Issue | Rationale | Suggested Refactor |
|---|---|---|---|
| `hippo_mem/relational/kg.py` | `prune` and `rollback` (CC 8)【F:static/radon_cc.txt†L17-L19】 | Nested loops and SQL handling increase risk of subtle bugs. | Extract helper functions for query building and node/edge restoration; add unit tests for error paths. |
| `hippo_mem/spatial/map.py` | `prune` (CC 9)【F:static/radon_cc.txt†L33-L35】 | Complex graph and cache manipulation hard to follow. | Split into node pruning and edge pruning helpers; consider using `networkx` for graph operations. |
| `hippo_mem/relational/tuples.py` | `extract_tuples` (CC 9)【F:static/radon_cc.txt†L14-L16】 | Heuristic parser mixes tokenization, confidence and time extraction. | Break into smaller functions for sentence splitting, time stripping and tuple scoring. |
