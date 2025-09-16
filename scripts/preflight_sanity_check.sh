#!/usr/bin/env bash
set -euo pipefail
RUN_ID="sanity_${RANDOM}"
STORES="runs/${RUN_ID}/stores"
MODEL="hippo/fake-tiny-gpt2"
SIZES=(8)
SEEDS=(1337)

suites=("episodic_cross_mem" "semantic_mem" "spatial_multi")
presets=("memory/hei_nw" "memory/sgc_rss" "memory/smpd")

for i in "${!suites[@]}"; do
  suite="${suites[$i]}"
  preset="${presets[$i]}"
  out="runs/${RUN_ID}/${preset##*/}/${suite}"
  python scripts/eval_model.py suite="$suite" preset="$preset" run_id="$RUN_ID" n="${SIZES[0]}" seed="${SEEDS[0]}" \
    mode=teach persist=true store_dir="$STORES" session_id="san_${RUN_ID}" compute.pre_metrics=true strict_telemetry=true model="$MODEL"
  test -s "${out}/failed_preflight.json" && { echo "Unexpected preflight fail"; exit 1; } || true
done
echo "Preflight sanity OK"
