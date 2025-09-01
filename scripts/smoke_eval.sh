#!/usr/bin/env bash
set -euo pipefail
export STRICT_TELEMETRY=1
source "$(dirname "$0")/_env.sh"

outdir=$(mktemp -d)
python scripts/eval_model.py suite=episodic preset=baselines/core n=50 seed=1337 \
  model=models/tiny-gpt2 outdir="$outdir" > /dev/null

python scripts/eval_model.py suite=episodic preset=baselines/span_short n=50 seed=1337 \
  model=models/tiny-gpt2 use_chat_template=true max_new_tokens=8 \
  outdir="$(mktemp -d)" > /dev/null

python scripts/eval_model.py suite=episodic preset=memory/hei_nw n=50 seed=1337 \
  model=models/tiny-gpt2 use_chat_template=true max_new_tokens=8 \
  outdir="$(mktemp -d)" > /dev/null

python - <<PY
import json, pathlib, sys
p = pathlib.Path("$outdir") / "metrics.json"
with p.open() as f:
    data = json.load(f)
metrics = data.get("metrics", {}).get("episodic", {})
sys.exit(0 if metrics else 1)
PY
