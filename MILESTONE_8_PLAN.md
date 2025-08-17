# Milestone 8 – Baseline datasets & evaluation runs

**Objective**

Generate baseline datasets, run baseline evaluations and establish reference metrics.

**Gate**

Baseline datasets exist with checksums; baseline metrics files are generated and aggregated; report summarises baseline performance.

## Tasks

| ID | Headline | Description | Owner |
|----|----------|-------------|-------|
| DG1 | Audit and extend dataset generator | Ensure `scripts/build_datasets.py` supports specifying suite, size, seed and output path. Add SHA256 checksum generation and write results to `data/checksums.txt`. Document dataset naming in `data/README.md`. | Codex |
| DG2 | Generate datasets | Run `scripts/build_datasets.py` for episodic, semantic and spatial suites with sizes {50, 200, 1000} and seeds {1337, 2025, 4242}. Save JSONL files under `data/` and update `data/checksums.txt`. | Codex |
| BR1 | Validate baseline configs | Confirm `configs/eval/baselines/{core,rag,longctx}.yaml` load in `scripts/eval_bench.py`. Add a dry-run test to verify evaluation harness uses each preset without errors. | Codex |
| BR2 | Document baseline commands | Provide a concise run guide (`docs/baselines.md` or similar) listing commands to execute `scripts/eval_bench.py` for each preset/seed/size. Specify output layout `runs/<date>/baselines/<preset>/<suite>/<size>_<seed>/`. | Codex |
| BR3 | Execute baseline evaluations | On a GPU machine, run `scripts/eval_bench.py` for each suite (episodic, semantic, spatial) with presets {core, rag, longctx} across all sizes and seeds. Store metrics.json/csv and meta.json under `runs/<date>/baselines/`. | Human |
| RA1 | Implement aggregation script | Create `scripts/report.py` that collects baseline metrics from `runs/**/baselines/`, computes aggregates and renders Markdown tables and optional plots to `reports/<date>/baseline_summary.md`. | Codex |
| RA2 | Aggregate baseline results | After evaluations, run `scripts/report.py` to produce the summary report in `reports/<date>/baseline_summary.md`. | Human |
| RA3 | Verify gate conditions | Confirm datasets and checksums exist, baseline runs produce metrics, and report summarises results. Mark milestone completion in issue tracker or `PROJECT_PLAN.md`. | Human |

