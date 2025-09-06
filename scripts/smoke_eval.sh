#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=${RUN_ID:-smoke}
export MODEL=${MODEL:-models/tiny-gpt2}
source "$(dirname "$0")/_env.sh"

SUITE=${SUITE:-episodic_cross_mem}
PRESET=${PRESET:-memory/hei_nw}
SESSION_ID=${SESSION_ID:-${PRESET##*/}_$RUN_ID}
STORES="runs/$RUN_ID/stores"

mkdir -p "runs/$RUN_ID/baselines"
if [ ! -f "runs/$RUN_ID/baselines/metrics.csv" ]; then
  echo "suite,em_raw,em_norm,f1" > "runs/$RUN_ID/baselines/metrics.csv"
fi

python -m hippo_eval.datasets.cli --suite "$SUITE" --size 5 --seed 1337 --out "datasets/$SUITE"

python scripts/eval_model.py \
  suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=5 seed=1337 \
  mode=teach persist=true store_dir="$STORES" session_id="$SESSION_ID" \
  compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

python scripts/eval_model.py \
  suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=5 seed=1337 \
  mode=test store_dir="$STORES" session_id="$SESSION_ID" \
  compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

python -m hippo_eval.reporting.report --run-id "$RUN_ID"
