#!/usr/bin/env bash
set -euo pipefail
export RUN_ID=${RUN_ID:-dev}
source "$(dirname "$0")/_env.sh"
export MODEL=models/tiny-gpt2
export HF_MODEL_PATH="$MODEL"
export ALLOW_BENCH=1
IFS=$'\n\t'

# Baseline runs for core preset on key suites
python scripts/run_baselines_bench.py \
  --run-id "$RUN_ID" \
  --presets baselines/core \
  --suites episodic semantic spatial \
  --sizes 50 \
  --seeds 1337
mkdir -p runs/"$RUN_ID"/baselines
echo "suite,em_raw,em_norm,f1" > runs/"$RUN_ID"/baselines/metrics.csv

# Session identifiers for memory algorithms
suites=(episodic)
presets=(memory/hei_nw)
sessions=("$HEI_SESSION_ID")
algos=(hei_nw)
kinds=(episodic)

for i in "${!suites[@]}"; do
  suite=${suites[$i]}
  preset=${presets[$i]}
  session_id=${sessions[$i]}
  outdir="runs/$RUN_ID/${preset}/$suite/50_1337"
  # Teach phase with persistence
  python scripts/eval_cli.py \
    suite=$suite preset=$preset n=50 seed=1337 \
    outdir=$outdir mode=teach persist=true \
    store_dir=$STORES session_id=$session_id gating_enabled=true --strict-telemetry >/dev/null
  # Test phase to obtain pre metrics
  python scripts/eval_cli.py \
    suite=$suite preset=$preset n=50 seed=1337 \
    outdir=$outdir store_dir=$STORES session_id=$session_id gating_enabled=true --strict-telemetry >/dev/null
  # Validate store layout before replay
  python scripts/validate_store.py --run_id "$RUN_ID" --algo=${algos[$i]} --kind=${kinds[$i]} >/dev/null
  # Replay phase with cycles=3
  python scripts/eval_cli.py \
    suite=$suite preset=$preset n=50 seed=1337 \
    outdir=$outdir mode=replay persist=true replay.cycles=3 \
    store_dir=$STORES session_id=$session_id gating_enabled=true --strict-telemetry >/dev/null
  # Verify post_* and delta_* metrics
  jq -e ".metrics[\"$suite\"] | has(\"post_em\") and has(\"delta_em\")" "$outdir/metrics.json" >/dev/null
done

python scripts/report.py --run-id "$RUN_ID"

echo "Smoke n50 pipeline finished for RUN_ID=$RUN_ID"
