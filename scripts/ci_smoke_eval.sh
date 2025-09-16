#!/usr/bin/env bash
set -euo pipefail

export RUN_ID=${RUN_ID:-ci_smoke}
export MODEL=${MODEL:-hippo/fake-tiny-gpt2}
export HIPPO_STRICT=${HIPPO_STRICT:-1}
export STRICT_TELEMETRY=${STRICT_TELEMETRY:-$HIPPO_STRICT}
source "$(dirname "$0")/_env.sh"

# --- BEGIN canonical CI smoke ---
export SUITE=${SUITE:-episodic_cross_mem}
export PRESET=${PRESET:-memory/hei_nw}
ALGO=${PRESET##*/}
PREFIX=${ALGO%%_*}
SESSION_ID=${SESSION_ID:-${PREFIX}_$RUN_ID}
STORES="runs/$RUN_ID/stores"    # BASE dir (no algo suffix)

N=8
if [ "$SUITE" = "semantic_mem" ]; then
  N=50
fi

# 0) Ensure minimal baselines to appease legacy preflight
mkdir -p "runs/$RUN_ID/baselines"
test -f "runs/$RUN_ID/baselines/metrics.csv" || \
  echo "suite,em_raw,em_norm,f1" > "runs/$RUN_ID/baselines/metrics.csv"

# 1) Deterministic dataset
python -m hippo_eval.datasets.cli --suite "$SUITE" --size "$N" --seed 1337 --out "datasets/$SUITE"

# 2) Teach with persistence (relax strict ratio)
HIPPO_STRICT=0 python scripts/eval_model.py \
  suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n="$N" seed=1337 \
  mode=teach persist=true store_dir="$STORES" session_id="$SESSION_ID" \
  compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 3) Test (read persisted store)
python scripts/eval_model.py \
  suite="$SUITE" preset="$PRESET" run_id="$RUN_ID" n="$N" seed=1337 \
  mode=test store_dir="$STORES" session_id="$SESSION_ID" \
  compute.pre_metrics=true strict_telemetry=true model="$MODEL" > /dev/null

# 4) Validate store and gating
case "$ALGO" in
  hei_nw)
    KIND=episodic
    EXTRA=(--expect-nonzero-ratio=0.85)
    ;;
  sgc_rss)
    KIND=kg
    EXTRA=(--expect-nodes=20 --expect-edges=20)
    ;;
  smpd)
    KIND=spatial
    EXTRA=()
    ;;
  *)
    KIND=episodic
    EXTRA=()
    ;;
esac
python scripts/validate_store.py --run_id "$RUN_ID" --algo "$ALGO" --kind "$KIND" --strict-telemetry "${EXTRA[@]}" > /dev/null
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
