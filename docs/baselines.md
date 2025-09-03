# Baseline evaluation commands

This guide lists the shell commands to evaluate the three suites with each baseline preset. The
current `scripts/eval_bench.py` uses a **mock model** that simply returns the ground-truth
answers, so metrics are a sanity check of the evaluation pipeline. Each run writes
`metrics.json`, `metrics.csv` and `meta.json` to
`runs/<RUN_ID>/baselines/<preset>/<suite>/<size>_<seed>/` where `<RUN_ID>` is a
slug of 3–64 characters.

```bash
RUN_ID=my_baselines
for PRESET in core rag longctx; do
  for SUITE in episodic semantic spatial; do
    for SIZE in 50 200 1000; do
      for SEED in 1337 2025 4242; do
        python scripts/eval_bench.py \
          suite=$SUITE preset=baselines/$PRESET \
          n=$SIZE seed=$SEED \
          outdir=runs/$RUN_ID/baselines/$PRESET/$SUITE/${SIZE}_${SEED}
      done
    done
  done
done
```

Each invocation uses the dataset generator internally, so `n` matches the dataset size and `seed`
ensures reproducibility. To obtain **true baseline metrics**, integrate actual model inference in
a later milestone.
