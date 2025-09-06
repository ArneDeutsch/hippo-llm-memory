# EVAL_PROTOCOL — Memory-First Pipeline

Use a two-phase flow: **Teach** with persistence, then **Test** reading the
store. All paths are parameterized by environment variables so runs are
reproducible and isolated.

## Variables
```bash
export RUN_ID=run_YYYYMMDD
export SIZES=(50)      # e.g., 50 100 200
export SEEDS=(1337)    # e.g., 1337 2025
source scripts/_env.sh  # sets RUNS, STORES=runs/$RUN_ID/stores
```

## Build datasets
```bash
for suite in episodic_cross_mem semantic_mem spatial_multi; do
  python -m hippo_eval.datasets.cli --suite "$suite" --size ${SIZES[0]} \
    --seed ${SEEDS[0]} --out datasets/$suite
done
```

## Teach with persistence
```bash
for suite in episodic_cross_mem semantic_mem spatial_multi; do
  python scripts/eval_model.py \
    suite=$SUITE preset=$PRESET run_id=$RUN_ID \
    n=${SIZES[0]} seed=${SEEDS[0]} mode=teach persist=true \
    store_dir=$STORES session_id=${PRESET##*/}_$RUN_ID \
    model=Qwen/Qwen2.5-1.5B-Instruct
done
```

## Test using the persisted store
```bash
for suite in episodic_cross_mem semantic_mem spatial_multi; do
  python scripts/eval_model.py \
    suite=$SUITE preset=$PRESET run_id=$RUN_ID \
    n=${SIZES[0]} seed=${SEEDS[0]} mode=test \
    store_dir=$STORES session_id=${PRESET##*/}_$RUN_ID \
    model=Qwen/Qwen2.5-1.5B-Instruct
done
```

`SUITE` takes values `episodic_cross_mem`, `semantic_mem`, or `spatial_multi`.
`PRESET` takes `memory/hei_nw`, `memory/sgc_rss`, or `memory/smpd`.
All flags use Hydra's `key=value` style; avoid legacy `--flag=value` forms.
