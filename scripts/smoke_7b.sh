#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

LOG=$(mktemp)

python scripts/train_lora.py \
  data_format=jsonl \
  train_files='["data/episodic_50_2025.jsonl"]' \
  fusion_insert_block_index=-4 \
  replay.enabled=true \
  max_steps=50 \
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
