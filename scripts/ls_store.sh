#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
SID="$HEI_SESSION_ID"
P="$STORES/hei_nw/$SID/episodic.jsonl"
if [[ -f "$P" ]]; then
  echo "OK: $P"
else
  echo "MISSING: expected $P" >&2
  echo "Run §4.1 (teach+replay with persist=true) to create it." >&2
  echo "Hint: STORES should point to the base directory containing the hei_nw folder." >&2
  exit 2
fi
