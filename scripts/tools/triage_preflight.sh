#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 RUN_ID [ALGO]" >&2
  exit 1
fi

RID="$1"
ALGO="${2:-hei_nw}"
# Use algo prefix (e.g., hei from hei_nw) for session id
SID="${ALGO%%_*}_${RID}"
META="runs/$RID/stores/$ALGO/$SID/store_meta.json"

echo "Expect meta: $META"
if [ -f "$META" ]; then
  cat "$META"
else
  echo "MISSING: $META"
  find "runs/$RID/stores" -maxdepth 3 -name store_meta.json -print
fi

# Check baseline metrics
if ! ls -l "runs/$RID/baselines/metrics.csv" 2>/dev/null; then
  echo "Baseline CSV missing"
fi

# Dump any failed preflight diagnostics
for p in runs/$RID/memory/$ALGO/*/*; do
  if [ -f "$p/failed_preflight.json" ]; then
    echo "---- $p"
    cat "$p/failed_preflight.json"
  fi
done
