# EVAL_PROTOCOL — Minimal, Parameterized (GPU)
**Purpose:** Run the evaluation pipeline with minimal, consistent commands.

## Variables
Set once per run:
```bash
export RUN_ID=run_YYYYMMDD
export SIZES=(50)      # e.g., 50 100 200
export SEEDS=(1337)    # e.g., 1337 2025
source scripts/_env.sh   # exports RUNS, STORES=runs/$RUN_ID/stores
```

## Build Datasets
Run per suite (adjust size/seed loops as needed):
```bash
python scripts/datasets_cli.py --suite semantic_mem --size ${SIZES[0]} \
  --seed ${SEEDS[0]} --out data/semantic_mem
python scripts/datasets_cli.py --suite episodic_cross_mem --size ${SIZES[0]} \
  --seed ${SEEDS[0]} --out data/episodic_cross_mem
python scripts/datasets_cli.py --suite spatial_multi --size ${SIZES[0]} \
  --seed ${SEEDS[0]} --out data/spatial_multi
```

## Semantic Memory (sgc_rss)
```bash
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python scripts/eval_cli.py suite=semantic_mem n="$n" seed="$seed" \
      outdir="$RUNS/semantic_mem_baseline/${n}_${seed}"

    python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss \
      mode=teach --no-retrieval-during-teach --persist \
      n="$n" seed="$seed" \
      outdir="$RUNS/semantic_mem_teach/${n}_${seed}"

    python scripts/eval_cli.py suite=semantic_mem preset=memory/sgc_rss \
      mode=test n="$n" seed="$seed" \
      outdir="$RUNS/semantic_mem_test/${n}_${seed}"
  done
done
```

## Episodic Cross Memory (hei_nw)
```bash
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python scripts/eval_cli.py suite=episodic_cross_mem n="$n" seed="$seed" \
      outdir="$RUNS/episodic_cross_mem_baseline/${n}_${seed}"

    python scripts/eval_cli.py suite=episodic_cross_mem preset=memory/hei_nw \
      mode=teach --no-retrieval-during-teach --persist \
      n="$n" seed="$seed" \
      outdir="$RUNS/episodic_cross_mem_teach/${n}_${seed}"

    python scripts/eval_cli.py suite=episodic_cross_mem preset=memory/hei_nw \
      mode=test n="$n" seed="$seed" \
      outdir="$RUNS/episodic_cross_mem_test/${n}_${seed}"
  done
done
```

## Spatial Multi (smpd)
```bash
for n in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python scripts/eval_cli.py suite=spatial_multi n="$n" seed="$seed" \
      outdir="$RUNS/spatial_multi_baseline/${n}_${seed}"

    python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd \
      mode=teach --no-retrieval-during-teach --persist \
      n="$n" seed="$seed" \
      outdir="$RUNS/spatial_multi_teach/${n}_${seed}"

    python scripts/eval_cli.py suite=spatial_multi preset=memory/smpd \
      mode=test n="$n" seed="$seed" \
      outdir="$RUNS/spatial_multi_test/${n}_${seed}"
  done
done
```

**Notes**
- `store_dir` and `session_id` are derived from `RUN_ID`; no manual overrides needed.
- Boolean flags are specified without `=true`.
