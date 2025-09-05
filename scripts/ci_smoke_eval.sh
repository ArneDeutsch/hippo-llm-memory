#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=${RUN_ID:-ci_smoke}
export MODEL=${MODEL:-models/tiny-gpt2}
export STRICT_TELEMETRY=${STRICT_TELEMETRY:-1}
source "$(dirname "$0")/_env.sh"

SUITE=${SUITE:-semantic_mem}
PRESET=${PRESET:-memory/sgc_rss_mem}
SESSION_ID=${SESSION_ID:-sgc_${RUN_ID}}

# 1. Matrix baselines
python scripts/eval_model.py +run_matrix=true \
  run_id="$RUN_ID" \
  presets=[baselines/core] \
  tasks=[${SUITE}] \
  n_values=[50] \
  seeds=[1337] \
  compute.pre_metrics=true \
  model="$MODEL" \
  > /dev/null

# 2. Aggregation
python -m hippo_eval.baselines --run-id "$RUN_ID"

# 3. Teach and test with strict telemetry
python scripts/eval_model.py suite=$SUITE preset=$PRESET \
  run_id="$RUN_ID" n=50 seed=1337 mode=teach persist=true \
  store_dir="$STORES" session_id="$SESSION_ID" \
  compute.pre_metrics=true strict_telemetry=true \
  model="$MODEL" > /dev/null

python scripts/eval_model.py suite=$SUITE preset=$PRESET \
  run_id="$RUN_ID" n=50 seed=1337 mode=test \
  store_dir="$STORES" session_id="$SESSION_ID" \
  strict_telemetry=true \
  model="$MODEL" > /dev/null

# 4. Report generation
python -m hippo_eval.reporting.report --run-id "$RUN_ID"

# Fail if baseline metrics missing
if [ ! -f "runs/$RUN_ID/baselines/metrics.csv" ]; then
  echo "missing baselines metrics" >&2
  exit 1
fi

# Fail if any preflight failures exist
if find "runs/$RUN_ID" -name failed_preflight.json -print -quit | grep -q .; then
  echo "found failed preflight" >&2
  exit 1
fi
