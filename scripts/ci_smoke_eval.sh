#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=${RUN_ID:-ci_smoke}
export MODEL=${MODEL:-models/tiny-gpt2}
export STRICT_TELEMETRY=${STRICT_TELEMETRY:-1}
source "$(dirname "$0")/_env.sh"

# --- BEGIN canonical CI smoke ---
SUITE=${SUITE:-episodic_cross_mem}
PRESET=${PRESET:-memory/hei_nw}
ALGO=${PRESET##*/}
PREFIX=${ALGO%%_*}
SESSION_ID=${SESSION_ID:-${PREFIX}_$RUN_ID}
STORES="runs/$RUN_ID/stores"    # BASE dir (no algo suffix)

# 0) Ensure minimal baselines to appease legacy preflight
mkdir -p "runs/$RUN_ID/baselines"
test -f "runs/$RUN_ID/baselines/metrics.csv" || \
  echo "suite,em_raw,em_norm,f1" > "runs/$RUN_ID/baselines/metrics.csv"

# 1) Deterministic dataset
python -m hippo_eval.datasets.cli --suite "$SUITE" --size 8 --seed 1337 --out "datasets/$SUITE"

# 2) Teach with persistence
python scripts/eval_model.py \
  suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=8 seed=1337 \
  mode=teach persist=true store_dir="$STORES" session_id="$SESSION_ID" \
  compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 3) Test (read persisted store)
python scripts/eval_model.py \
  suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n=8 seed=1337 \
  mode=test store_dir="$STORES" session_id="$SESSION_ID" \
  compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 4) Validate store and gating
python scripts/validate_store.py --run_id "$RUN_ID" --algo "$ALGO" --kind episodic --strict-telemetry > /dev/null
python - <<'PY'
import json, os, sys, pathlib
run_id=os.environ["RUN_ID"]
preset=os.environ["PRESET"].split("/")[-1]
suite=os.environ["SUITE"]
metrics_path=pathlib.Path("runs")/run_id/preset/suite/"metrics.json"
with metrics_path.open() as f:
    metrics=json.load(f)
attempts=sum(m.get("attempts",0) for m in metrics.get("gating", {}).values())
print(f"gate_attempts={attempts}")
if attempts<=0:
    sys.exit(1)
PY

# 5) Report
python -m hippo_eval.reporting.report --run-id "$RUN_ID"
# --- END canonical CI smoke ---

# Sanity: baseline CSV exists
if [ ! -f "runs/$RUN_ID/baselines/metrics.csv" ]; then
  echo "missing baselines metrics" >&2
  exit 1
fi
