# Action Plan — Follow-ups to Review (2025-08-19)

**Scope:** Close the gap between design and reality for adapter wiring, dataset usage, and evaluation.

## A. Integration (end-to-end wiring)
- Wire Episodic/Relational/Spatial adapters into the base LM forward pass after block N (configurable).
- Add assertion that LoRA attaches to >0 params; log trainable param count and target_modules.
- Replace IMDB default with JSONL loaders for data/{episodic,semantic,spatial}_*.jsonl; include train/val splits.
- Mix replay batches per ratios in config.

**Gate:** unit tests enforce non-zero trainables; dry run passes; real run consumes JSONL and prints adapter hooks active.

## B. Evaluation harness (real model)
- Add scripts/eval_model.py (new) to run models with memory presets (configs/eval/memory/*.yaml).
- Keep scripts/eval_bench.py as CI plumbing.
- Emit metrics/meta under runs/<date>/<preset>/<suite>/ as in EVAL_PLAN.md.

**Gate:** one small real evaluation (n=50) completes for HEI-NW.

## C. Planning docs to update
- PROJECT_PLAN.md: insert Milestone 7b and adjust Milestone 9 gates.
- EVAL_PLAN.md: add “Harness requirements” + “Command matrix”.
- MILESTONE_9_PLAN.md: add tasks C0 (adapter wiring), C11 (eval_model.py), C12 (smoke eval).

**References:** review/review-2025-08-19.md
