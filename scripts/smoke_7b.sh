#!/usr/bin/env bash
set -euo pipefail
export RUN_ID=${RUN_ID:-dev}
source "$(dirname "$0")/_env.sh"
export MODEL=hippo/fake-tiny-gpt2
export HF_MODEL_PATH="$MODEL"
IFS=$'\n\t'

LOG=$(mktemp)

python -m hippo_mem.training.lora \
  dry_run=true \
  data_format=jsonl \
  train_files='["data/episodic/50_2025.jsonl"]' \
  fusion_insert_block_index=-1 \
  replay.enabled=true \
  > "$LOG" 2>&1

if ! grep -q "Adapter fusion attached" "$LOG"; then
  echo "Adapter fusion not attached" >&2
  exit 1
fi

if ! grep -iq "Trainable parameters" "$LOG"; then
  echo "Missing Trainable parameters" >&2
  exit 1
fi

if ! grep -iq "Train dataset size" "$LOG"; then
  echo "Missing Train dataset size" >&2
  exit 1
fi

echo "Milestone 7b smoke test passed"
