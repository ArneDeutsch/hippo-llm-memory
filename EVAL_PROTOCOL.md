# EVAL_PROTOCOL (slim)
## Params
export RUN_ID=run$(date +%Y%m%d)
export SIZES=(50)     # or: 50 100
export SEEDS=(1337)   # or: 1 2 3
export MODEL=gpt-4o-mini
## Run
scripts/eval_cli.py suite=episodic preset=memory/hei_nw n="$SIZES" seed="$SEEDS" run_id="$RUN_ID" mode=teach persist=true --strict-telemetry
scripts/validate_store.py --strict-telemetry
scripts/eval_cli.py suite=semantic preset=memory/sgc_rss n="$SIZES" seed="$SEEDS" run_id="$RUN_ID" mode=teach persist=true --strict-telemetry
scripts/validate_store.py --strict-telemetry
scripts/eval_cli.py suite=spatial preset=memory/smpd n="$SIZES" seed="$SEEDS" run_id="$RUN_ID" mode=teach persist=true --strict-telemetry
scripts/validate_store.py --strict-telemetry
