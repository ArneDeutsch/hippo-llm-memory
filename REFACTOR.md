# Refactor Opportunities

| Priority | Location | Issue | Rationale & Plan |
| --- | --- | --- | --- |
| P1 | `hippo_mem/consolidation/worker.py::ConsolidationWorker.__init__` | Cyclomatic complexity 20 | Split configuration parsing and thread setup into helpers to improve readability and testability. |
| P1 | `hippo_mem/episodic/replay.py::ReplayQueue.sample` | Cyclomatic complexity 17 | Separate priority computation and gradient-overlap filtering; add small functions for clarity and easier testing. |
| P2 | `hippo_mem/relational/kg.py::prune` | Cyclomatic complexity 12 | Extract SQL building and node removal loops; reduces risk of edge-case bugs and aids coverage. |
| P3 | `scripts/eval_bench.py` main function | Large monolithic function handling config, I/O and evaluation | Decompose into loader, runner and writer modules to support future metrics and datasets. |
