#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
IFS=$'\n\t'

RUN_ID=${1:-$RUN_ID}
DATE="$RUN_ID"

for suite in episodic semantic spatial; do
  python scripts/build_datasets.py --suite "$suite" --size 50 --seed 1337 --out "data/$suite/50_1337.jsonl"
done

python scripts/run_baselines_bench.py --date "$RUN_ID" --presets baselines/core --sizes 50 --seeds 1337
python scripts/report.py --date "$RUN_ID"

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

