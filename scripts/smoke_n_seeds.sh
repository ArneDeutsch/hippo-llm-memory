#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
MODEL=models/tiny-gpt2
SUITE=episodic
printf "%-6s %-9s %-9s\n" "seed" "baseline" "hei_nw"
for seed in 1337 2025 4242; do
  bdir=$(mktemp -d)
  python scripts/eval_cli.py \
    suite=$SUITE preset=baselines/core n=50 seed=$seed \
    model=$MODEL outdir="$bdir" --strict-telemetry >/dev/null
  bem=$(jq -r '.metrics.episodic.pre_em // .metrics.episodic.em // 0' "$bdir/metrics.json")
  mdir=$(mktemp -d)
  python scripts/eval_cli.py \
    suite=$SUITE preset=memory/hei_nw n=50 seed=$seed \
    model=$MODEL outdir="$mdir" --strict-telemetry >/dev/null
  mem=$(jq -r '.metrics.episodic.pre_em // .metrics.episodic.em // 0' "$mdir/metrics.json")
  printf "%-6s %-9.3f %-9.3f\n" "$seed" "$bem" "$mem"
done
