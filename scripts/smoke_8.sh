#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

DATE=${1:-20250822}

for suite in episodic semantic spatial; do
  python scripts/build_datasets.py --suite "$suite" --size 50 --seed 1337 --out "data/$suite/50_1337.jsonl"
done

python scripts/run_baselines.py --date "$DATE" --presets baselines/core --sizes 50 --seeds 1337
python scripts/report.py --date "$DATE"

for suite in episodic semantic spatial; do
  run_dir="runs/$DATE/baselines/core/$suite/50_1337"
  for f in metrics.json metrics.csv meta.json; do
    if [ ! -s "$run_dir/$f" ]; then
      echo "Missing $run_dir/$f" >&2
      exit 1
    fi
  done
  if [ ! -s "reports/$DATE/$suite/summary.md" ]; then
    echo "Missing reports/$DATE/$suite/summary.md" >&2
    exit 1
  fi
done

echo "Milestone 8 smoke test passed"

