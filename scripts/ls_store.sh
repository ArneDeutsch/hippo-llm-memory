#!/usr/bin/env bash
source scripts/env_prelude.sh
SID="hei_${RUN_ID}"
P="$STORES/hei_nw/$SID/episodic.jsonl"
if [[ -f "$P" ]]; then
  echo "OK: $P"
else
  echo "MISSING: $P" >&2
  exit 2
fi
