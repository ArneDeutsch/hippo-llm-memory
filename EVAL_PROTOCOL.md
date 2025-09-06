# EVAL_PROTOCOL — Memory-First Pipeline

Use a two-phase flow: **Teach** with persistence, then **Test** reading the
store. All paths are parameterized by environment variables so runs are
reproducible and isolated.

## Variables
```bash
export RUN_ID=run_YYYYMMDD
export SUITES=(episodic_cross_mem semantic_mem spatial_multi)
export PRESETS=(memory/hei_nw memory/sgc_rss memory/smpd)  # align with SUITES
export SIZES=(50)      # e.g., 50 100 200
export SEEDS=(1337)    # e.g., 1337 2025
source scripts/_env.sh  # sets RUNS, STORES=runs/$RUN_ID/stores
```

## Build datasets
```bash
for SUITE in "${SUITES[@]}"; do
  python -m hippo_eval.datasets.cli --suite "$SUITE" --size ${SIZES[0]} \
    --seed ${SEEDS[0]} --out data/$SUITE
done
```

## Baseline runs (required for preflight)
```bash
for SUITE in "${SUITES[@]}"; do
  python scripts/eval_model.py \
    suite=$SUITE preset=baselines/core run_id=$RUN_ID \
    n=${SIZES[0]} seed=${SEEDS[0]} model=Qwen/Qwen2.5-1.5B-Instruct
done
python -m hippo_eval.baselines --run-id $RUN_ID
```

## Teach with persistence
```bash
for i in "${!SUITES[@]}"; do
  SUITE=${SUITES[$i]}
  PRESET=${PRESETS[$i]}
  python scripts/eval_model.py \
    suite=$SUITE preset=$PRESET run_id=$RUN_ID \
    n=${SIZES[0]} seed=${SEEDS[0]} mode=teach persist=true \
    store_dir=$STORES session_id=${PRESET##*/}_$RUN_ID \
    model=Qwen/Qwen2.5-1.5B-Instruct
done
```

## Test using the persisted store
```bash
for i in "${!SUITES[@]}"; do
  SUITE=${SUITES[$i]}
  PRESET=${PRESETS[$i]}
  python scripts/eval_model.py \
    suite=$SUITE preset=$PRESET run_id=$RUN_ID \
    n=${SIZES[0]} seed=${SEEDS[0]} mode=test \
    store_dir=$STORES session_id=${PRESET##*/}_$RUN_ID \
    model=Qwen/Qwen2.5-1.5B-Instruct
done
```

`SUITES` and `PRESETS` must stay aligned:
`episodic_cross_mem`→`memory/hei_nw`, `semantic_mem`→`memory/sgc_rss`,
and `spatial_multi`→`memory/smpd`. All flags use Hydra's `key=value`
style; avoid legacy `--flag=value` forms.
