# EVAL_PROTOCOL

Minimal runbook for evaluating memory suites. Commands assume GNU bash and a
single 12 GB GPU.

## Params

```bash
export RUN_ID=run$(date +%Y%m%d)
export SIZES=(50)        # e.g. 50 100
export SEEDS=(1337)      # e.g. 1 2 3
export MODEL=models/tiny-gpt2
```

## Episodic (HEI‑NW)

```bash
for N in "${SIZES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python scripts/eval_model.py \
      suite=episodic_cross_mem preset=memory/hei_nw \
      n=$N seed=$SEED model=$MODEL run_id=$RUN_ID mode=teach \
      store_dir=runs/$RUN_ID/stores session_id=hei_run$RUN_ID \
      --strict-telemetry
    python scripts/eval_model.py \
      suite=episodic_cross_mem preset=memory/hei_nw \
      n=$N seed=$SEED model=$MODEL run_id=$RUN_ID mode=test \
      store_dir=runs/$RUN_ID/stores session_id=hei_run$RUN_ID \
      --strict-telemetry
    python scripts/validate_store.py \
      --run_id $RUN_ID --algo hei_nw --kind episodic \
      --expect-nonzero-ratio 0.9 --strict-telemetry
  done
done
```

## Semantic (SGC‑RSS)

```bash
for N in "${SIZES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python scripts/eval_model.py \
      suite=semantic_mem preset=memory/sgc_rss \
      n=$N seed=$SEED model=$MODEL run_id=$RUN_ID mode=teach \
      store_dir=runs/$RUN_ID/stores session_id=sgc_run$RUN_ID \
      --strict-telemetry
    python scripts/eval_model.py \
      suite=semantic_mem preset=memory/sgc_rss \
      n=$N seed=$SEED model=$MODEL run_id=$RUN_ID mode=test \
      store_dir=runs/$RUN_ID/stores session_id=sgc_run$RUN_ID \
      --strict-telemetry
    python scripts/validate_store.py \
      --run_id $RUN_ID --algo sgc_rss --kind kg \
      --expect-nodes 100 --expect-edges 100 \
      --expect-embedding-coverage 0.9 --strict-telemetry
  done
done
```

## Spatial (SMPD)

```bash
for N in "${SIZES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    python scripts/eval_model.py \
      suite=spatial_multi preset=memory/smpd \
      n=$N seed=$SEED model=$MODEL run_id=$RUN_ID mode=teach \
      store_dir=runs/$RUN_ID/stores session_id=smpd_run$RUN_ID \
      --strict-telemetry
    python scripts/eval_model.py \
      suite=spatial_multi preset=memory/smpd \
      n=$N seed=$SEED model=$MODEL run_id=$RUN_ID mode=replay \
      store_dir=runs/$RUN_ID/stores session_id=smpd_run$RUN_ID \
      --strict-telemetry
    python scripts/eval_model.py \
      suite=spatial_multi preset=memory/smpd \
      n=$N seed=$SEED model=$MODEL run_id=$RUN_ID mode=test \
      store_dir=runs/$RUN_ID/stores session_id=smpd_run$RUN_ID \
      --strict-telemetry
    python scripts/validate_store.py \
      --run_id $RUN_ID --algo smpd --kind spatial --strict-telemetry
  done
done
```

Add `--oracle` to test commands to log upper‑bound metrics (`oracle_em`,
`oracle_f1`).

