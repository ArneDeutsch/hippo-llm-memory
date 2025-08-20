#!/usr/bin/env bash
set -euo pipefail

LOG=$(mktemp)

python scripts/train_lora.py \
  data_format=jsonl \
  train_files='["data/episodic_50_2025.jsonl"]' \
  fusion_insert_block_index=-4 \
  replay.enabled=true \
  max_steps=50 \
  > "$LOG" 2>&1

grep -q "Adapter fusion attached" "$LOG"
grep -iq "trainable params" "$LOG"
grep -q "Loaded JSONL" "$LOG"

echo "Milestone 7b smoke test passed"
