#!/usr/bin/env bash
set -euo pipefail
export RUN_ID=${1:-dev}
source "$(dirname "$0")/_env.sh"
export MODEL=models/tiny-gpt2
export HF_MODEL_PATH="$MODEL"
export ALLOW_BENCH=1
IFS=$'\n\t'

for suite in episodic semantic spatial; do
  python -m hippo_eval.eval.datasets --suite "$suite" --size 50 --seed 1337 --out "data/$suite/50_1337.jsonl"
done

python -m hippo_eval.eval.baselines --run-id "$RUN_ID" --presets baselines/core --sizes 50 --seeds 1337
python -m hippo_eval.reporting.report --run-id "$RUN_ID"

for suite in episodic semantic spatial; do
  run_dir="runs/$RUN_ID/baselines/core/$suite/50_1337"
  for f in metrics.json metrics.csv meta.json; do
    if [ ! -s "$run_dir/$f" ]; then
      echo "Missing $run_dir/$f" >&2
      exit 1
    fi
  done
  if [ ! -s "reports/$RUN_ID/$suite/summary.md" ]; then
    echo "Missing reports/$RUN_ID/$suite/summary.md" >&2
    exit 1
  fi
done

echo "Milestone 8 smoke test passed"

