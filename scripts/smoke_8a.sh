#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

LOG=$(mktemp)

python scripts/train_lora.py dry_run=true episodic.enabled=true > "$LOG" 2>&1

if ! grep -q "episodic_retrieval_k" "$LOG"; then
  echo "Missing episodic_retrieval_k" >&2
  exit 1
fi

if ! grep -q "episodic_latency_ms" "$LOG"; then
  echo "Missing episodic_latency_ms" >&2
  exit 1
fi

if ! grep -q "write_accept_rate" "$LOG"; then
  echo "Missing write_accept_rate" >&2
  exit 1
fi

echo "Milestone 8a smoke test passed"
