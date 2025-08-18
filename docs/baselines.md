# Baseline evaluation commands

This guide lists the shell commands to evaluate the three suites with each baseline preset. Each
run writes `metrics.json`, `metrics.csv` and `meta.json` to `runs/<date>/baselines/<preset>/<suite>/<size>_<seed>/` where `<date>` is `$(date +%Y%m%d)`.

```bash
DATE=$(date +%Y%m%d)
for PRESET in core rag longctx; do
  for SUITE in episodic semantic spatial; do
    for SIZE in 50 200 1000; do
      for SEED in 1337 2025 4242; do
        python scripts/eval_bench.py \
          suite=$SUITE preset=baselines/$PRESET \
          n=$SIZE seed=$SEED \
          outdir=runs/$DATE/baselines/$PRESET/$SUITE/${SIZE}_${SEED}
      done
    done
  done
done
```

Each invocation uses the dataset generator internally, so `n` matches the dataset size and `seed`
ensures reproducibility.
