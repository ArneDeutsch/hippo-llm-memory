# Milestone 9 – Memory-augmented training, evaluation & ablations

**Objective**: train models with each memory module, evaluate them, perform ablations and compare against baselines.

**Gate**: experiments are logged under `runs/YYYYMMDD/`; metrics show improvements over baselines; ablation results highlight contribution of each component; aggregated report summarises findings.

## Tasks

| ID | Headline | Description | Owner |
|----|----------|-------------|-------|
| C1 | Audit datasets and configs | Verify that episodic, semantic and spatial datasets (sizes 50, 200, 1000; seeds 1337, 2025, 4242) exist with checksums. Ensure Hydra presets for memory modules (`configs/eval/memory/*.yaml`) and baseline references are present. | Codex |
| C2 | Automate memory evaluation run matrix | Add a script or extend `scripts/eval_bench.py` to iterate over suites, dataset sizes and seeds, saving metrics under `runs/<date>/<module>/<suite>/`. Provide CLI so a single command runs the full matrix for a given preset. | Codex |
| C3 | Prepare training/eval commands | Document example commands in `experiments/*/RUN.md` for training with `scripts/train_lora.py` and evaluating with `scripts/eval_bench.py`. Commands must cover HEI-NW, SGC-RSS, SMPD, combined `memory/all`, and support `+ablate=` flags. | Codex |
| C4 | Extend reporting script | Modify `scripts/report.py` to aggregate memory-augmented runs and ablation results into tables/plots under `reports/<date>/<suite>/summary.md`. Include compute/memory overhead columns. | Codex |
| C5 | HEI-NW training & evaluation | Using a 12 GB GPU, run the prepared commands for the episodic module across all dataset sizes and seeds. Log metrics before/after replay and store outputs under `runs/<date>/hei_nw/episodic/`. | Human |
| C6 | SGC-RSS training & evaluation | Run relational module training/eval as above, recording multi-hop accuracy, contradiction rate and time-to-stabilize. Save logs under `runs/<date>/sgc_rss/semantic/`. | Human |
| C7 | SMPD training & evaluation | Execute spatial module training/eval, capturing success rate, path suboptimality and macro step reduction. Store logs under `runs/<date>/smpd/spatial/`. | Human |
| C8 | Combined model run | Train and evaluate with all memory modules enabled (`preset=memory/all`) on all suites, logging metrics under `runs/<date>/all/`. | Human |
| C9 | Ablation sweeps | For each module, run the automated script with key flags toggled (e.g., `episodic.use_sparsity=false`, `replay.enabled=false`, `relational.schema_fasttrack=false`, `spatial.macros=false`). Store results under `runs/<date>/ablations/`. | Human |
| C10 | Aggregate and verify results | Execute the updated `scripts/report.py` to generate summaries comparing baselines, memory variants and ablations. Confirm that improvements over baselines meet success criteria and that all outputs are present for gate review. | Human |

