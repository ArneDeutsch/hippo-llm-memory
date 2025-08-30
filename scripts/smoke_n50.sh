#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
export MODEL=models/tiny-gpt2
IFS=$'\n\t'

RUN_ID=${RUN_ID}
DATE="$RUN_ID"

# Baseline runs for core preset on key suites
python scripts/run_baselines_bench.py \
  --date "$RUN_ID" \
  --presets baselines/core \
  --suites episodic semantic spatial \
  --sizes 50 \
  --seeds 1337

# Session identifiers for memory algorithms
SGC_SESSION_ID="sgc_${RUN_ID}"
SMPD_SESSION_ID="smpd_${RUN_ID}"

suites=(episodic semantic spatial)
presets=(memory/hei_nw memory/sgc_rss memory/smpd)
sessions=("$HEI_SESSION_ID" "$SGC_SESSION_ID" "$SMPD_SESSION_ID")
algos=(hei_nw sgc_rss smpd)
kinds=(episodic kg map)

for i in "${!suites[@]}"; do
  suite=${suites[$i]}
  preset=${presets[$i]}
  session_id=${sessions[$i]}
  outdir="runs/$RUN_ID/${preset}/$suite/50_1337"
  # Teach phase with persistence
  python scripts/eval_cli.py \
    suite=$suite preset=$preset n=50 seed=1337 \
    outdir=$outdir mode=teach persist=true \
    store_dir=$STORES session_id=$session_id --strict-telemetry >/dev/null
  # Test phase to obtain pre metrics
  python scripts/eval_cli.py \
    suite=$suite preset=$preset n=50 seed=1337 \
    outdir=$outdir store_dir=$STORES session_id=$session_id --strict-telemetry >/dev/null
  # Validate store layout before replay
  python scripts/validate_store.py --algo=${algos[$i]} --kind=${kinds[$i]} >/dev/null
  # Replay phase with cycles=3
  python scripts/eval_cli.py \
    suite=$suite preset=$preset n=50 seed=1337 \
    outdir=$outdir mode=replay persist=true replay.cycles=3 \
    store_dir=$STORES session_id=$session_id --strict-telemetry >/dev/null
  # Verify post_* and delta_* metrics
  jq -e ".metrics[\"$suite\"] | has(\"post_em\") and has(\"delta_em\")" "$outdir/metrics.json" >/dev/null
done

python scripts/report.py --date "$RUN_ID"

echo "Smoke n50 pipeline finished for RUN_ID=$RUN_ID"
