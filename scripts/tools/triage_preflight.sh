#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 RUN_ID" >&2
  exit 1
fi

RID="$1"
BASE="runs/$RID"

echo "Failed preflights under $BASE:" >&2
find "$BASE" -name failed_preflight.json -print || true

CSV="$BASE/baselines/metrics.csv"
if [ -f "$CSV" ]; then
  ls -l "$CSV"
else
  echo "Baseline CSV missing (ok if baselines not used)" >&2
fi
