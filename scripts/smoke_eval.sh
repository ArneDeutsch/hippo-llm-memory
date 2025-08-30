#!/usr/bin/env bash
set -euo pipefail
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
metrics = data["metrics"]["episodic"]
em = metrics.get("em", metrics.get("pre_em", 0.0))
f1 = metrics.get("f1", metrics.get("pre_f1", 0.0))
# Fail if both EM==0 and F1<0.20
sys.exit(0 if (em > 0 or f1 >= 0.20) else 1)
PY
