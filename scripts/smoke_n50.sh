#!/usr/bin/env bash
set -euo pipefail
export RUN_ID=${RUN_ID:-dev}
export MODEL=${MODEL:-models/tiny-gpt2}
source "$(dirname "$0")/_env.sh"
export HF_MODEL_PATH="$MODEL"
export ALLOW_BENCH=1
IFS=$'\n\t'

# Baseline runs for core preset on key suites
python -m hippo_eval.eval.baselines \
  --run-id "$RUN_ID" \
  --presets baselines/core \
  --suites episodic semantic spatial \
  --sizes 50 \
  --seeds 1337
mkdir -p runs/"$RUN_ID"/baselines
echo "suite,em_raw,em_norm,f1" > runs/"$RUN_ID"/baselines/metrics.csv

# Parameters for memory algorithm
SUITE=${SUITE:-episodic}
PRESET=${PRESET:-memory/hei_nw}
SESSION_ID=${SESSION_ID:-$HEI_SESSION_ID}
ALGO=${ALGO:-hei_nw}
KIND=${KIND:-episodic}
outdir="runs/$RUN_ID/${PRESET}/$SUITE/50_1337"

# Teach phase with persistence
python scripts/eval_cli.py \
  suite=$SUITE preset=$PRESET n=50 seed=1337 model=$MODEL \
  outdir=$outdir mode=teach persist=true \
  store_dir=$STORES session_id=$SESSION_ID gating_enabled=true --strict-telemetry >/dev/null

# Test phase to obtain pre metrics
python scripts/eval_cli.py \
  suite=$SUITE preset=$PRESET n=50 seed=1337 model=$MODEL \
  outdir=$outdir store_dir=$STORES session_id=$SESSION_ID gating_enabled=true --strict-telemetry >/dev/null

# Validate store layout before replay
python scripts/validate_store.py --run_id "$RUN_ID" --algo=$ALGO --kind=$KIND >/dev/null

# Replay phase with cycles=3
python scripts/eval_cli.py \
  suite=$SUITE preset=$PRESET n=50 seed=1337 model=$MODEL \
  outdir=$outdir mode=replay persist=true replay.cycles=3 \
  store_dir=$STORES session_id=$SESSION_ID gating_enabled=true --strict-telemetry >/dev/null

# Verify post_* and delta_* metrics
jq -e ".metrics[\"$SUITE\"] | has(\"post_em\") and has(\"delta_em\")" "$outdir/metrics.json" >/dev/null

python -m hippo_eval.reporting.report --run-id "$RUN_ID"

echo "Smoke n50 pipeline finished for RUN_ID=$RUN_ID"
